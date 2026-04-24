# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Union, cast

import torch
import torch.fx as fx
from torch.distributed.pipelining.schedules import (
    FULL_BACKWARD,
    _Action,
    _PipelineContext,
    _PipelineScheduleRuntime,
    _wait_batch_p2p,
)
from torch.distributed.pipelining.stage import (
    PipelineStage,
    _normalize_model_output_as_tuple,
)
from torch.distributed.tensor import DTensor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _execute_graph(callable_or_gm: Any, args: list[Any]) -> Any:
    """Execute a precompiled callable or raw GraphModule."""
    if isinstance(callable_or_gm, fx.GraphModule):
        return fx.Interpreter(callable_or_gm).boxed_run(args)
    return callable_or_gm(args)


@dataclass
class GraphCallables:
    fw: fx.GraphModule
    full_bw: fx.GraphModule
    bw_dI: Optional[fx.GraphModule] = None
    bw_dW: Optional[fx.GraphModule] = None
    unshard: Optional[fx.GraphModule] = None
    reduce_grad: Optional[fx.GraphModule] = None


@dataclass
class GraphMeta:
    num_mutate_inputs: int
    num_user_outputs: int
    num_symints_saved_for_bw: int
    num_params: int
    num_buffers: int
    num_input_grads: int


class MultiplexFwBwGraphPass(Protocol):
    """Protocol defining the contract for forward-backward graph multiplexing passes.

    Implementations must accept two GraphModules (forward and backward) and return a fused
    GraphModule that multiplexes their execution.

    Contract Requirements:
        1. Input placeholders ordering: The returned GraphModule's placeholders must be ordered
           as ``bw_placeholders + fw_placeholders`` (backward placeholders concatenated with
           forward placeholders, each maintaining their original order from the input graphs).

        2. Output node args ordering: The returned GraphModule's output node args must contain
           outputs ordered as ``bw_outputs + fw_outputs`` (backward outputs concatenated with
           forward outputs, each maintaining their original order from the input graphs).

    Example::

        def my_multiplex_pass(
            fw_graph: fx.GraphModule,
            bw_graph: fx.GraphModule
        ) -> fx.GraphModule:
            # Implementation that satisfies the contract
            ...
            return multiplexed_graph
    """

    def __call__(
        self,
        fw_graph: fx.GraphModule,
        bw_graph: fx.GraphModule,
    ) -> fx.GraphModule:
        """Multiplex forward and backward graphs into a single fused graph.

        Args:
            fw_graph (fx.GraphModule): Forward graph module.
            bw_graph (fx.GraphModule): Backward graph module.

        Returns:
            fx.GraphModule: Fused graph module satisfying the contract requirements.
        """
        ...


def get_multiplexed_graph_callables(
    stage_graphs: dict[int, GraphCallables],
    multiplex_fw_bw_graph_pass: MultiplexFwBwGraphPass,
) -> dict[tuple[int, int], fx.GraphModule]:
    """Generate multiplexed graph modules that fuse forward and backward passes from different stages.

    Creates fused modules for all stage pairs where fw_stage_idx != bw_stage_idx. This enables
    pipeline schedules (e.g., DualPipe) to overlap communication with computation.

    Args:
        stage_graphs (dict[int, GraphCallables]): Mapping from stage index to GraphCallables
            containing forward/backward modules.
        multiplex_fw_bw_graph_pass (MultiplexFwBwGraphPass): A callable that takes two
            GraphModules (forward and backward) and returns a fused GraphModule that multiplexes
            their execution. Must satisfy the contract defined in
            :class:`MultiplexFwBwGraphPass`.

    Returns:
        dict[tuple[int, int], fx.GraphModule]: Mapping from (fw_stage_idx, bw_stage_idx) to fused
            GraphModule that executes forward from fw_stage_idx and backward from bw_stage_idx.
    """
    multiplexed_graph_callables: dict[tuple[int, int], torch.fx.GraphModule] = {}
    for bw_stage_idx, bw_stage_graph_callables in stage_graphs.items():
        for fw_stage_idx, fw_stage_graph_callables in stage_graphs.items():
            if bw_stage_idx != fw_stage_idx:
                fw_bw_module = multiplex_fw_bw_graph_pass(
                    fw_stage_graph_callables.fw,
                    bw_stage_graph_callables.full_bw,
                )
                multiplexed_graph_callables[(fw_stage_idx, bw_stage_idx)] = fw_bw_module
    return multiplexed_graph_callables


class GraphPipelineStage(PipelineStage):
    def __init__(
        self,
        submodule: torch.nn.Module,
        graph_callables: GraphCallables,
        graph_meta: GraphMeta,
        stage_index: int,
        num_stages: int,
        device: torch.device,
        input_args: Optional[Union[torch.Tensor, tuple[torch.Tensor, ...]]] = None,
        output_args: Optional[Union[torch.Tensor, tuple[torch.Tensor, ...]]] = None,
        group: Optional[torch.distributed.ProcessGroup] = None,
        dw_builder: Optional[Callable[[], Callable[..., None]]] = None,
    ):
        super().__init__(
            submodule=submodule,
            stage_index=stage_index,
            num_stages=num_stages,
            device=device,
            input_args=input_args,
            output_args=output_args,
            group=group,
            dw_builder=dw_builder,
        )
        self.graph_callables = graph_callables
        self.graph_meta = graph_meta
        self.state: dict[str, list[Any]] = {
            "sharded_params": [],
            "unsharded_params": [],
            "buffers": [],
            "sharded_grads": [],
            "unsharded_grads": [],
        }
        self.bwd_activation_cache: dict[int, tuple[Any]] = {}

    def scale_grads(self, grad_scale_factor: int) -> None:
        """Scale stage's gradients by `grad_scale_factor`, which should be specified in coordination with the
        loss function used with pipelining.  For loss functions which perform 'mean' loss reduction, `grad_scale_factor`
        should be set to num_microbatches.  For loss functions that use `sum` reduction, `grad_scale_factor` should
        be set to 1.

        Should only be called once per pipeline schedule step, after all backwards passes have completed.
        """

        # PP scales only for its own contribution (microbatches), but relies on DP to scale further
        # for DP degree.
        if grad_scale_factor != 1:
            for grad in self.state["unsharded_grads"]:
                if grad is not None:
                    grad.div_(grad_scale_factor)

    def _accumulate_stage_unsharded_grads(
        self,
        param_buffer_grads: list[Union[torch.Tensor, None]],
    ) -> None:
        unsharded_grads = self.state["unsharded_grads"]
        grads_to_accumulate = param_buffer_grads[: self.graph_meta.num_params]
        assert len(unsharded_grads) == len(grads_to_accumulate)
        assert not all(
            grad is None for grad in grads_to_accumulate
        ), "All grads are None"
        for i in range(len(unsharded_grads)):
            if grads_to_accumulate[i] is not None:
                if unsharded_grads[i] is None:
                    unsharded_grads[i] = grads_to_accumulate[i]
                else:
                    unsharded_grads[i] += grads_to_accumulate[i]


def _run_fw_module(
    fw_module: Any,
    graph_meta: GraphMeta,
    fw_args: list[Any],
) -> tuple[Any, tuple[tuple[Any], tuple[Any]]]:
    fw_outputs = _execute_graph(fw_module, fw_args)

    num_inner_fwd_outputs = graph_meta.num_mutate_inputs + graph_meta.num_user_outputs
    saved_intermediates = fw_outputs[num_inner_fwd_outputs:]
    num_tensors_for_backward = (
        len(saved_intermediates) - graph_meta.num_symints_saved_for_bw
    )
    tensors_for_backward = saved_intermediates[:num_tensors_for_backward]
    non_tensors_for_backward = saved_intermediates[num_tensors_for_backward:]
    save_for_backward = (tensors_for_backward, non_tensors_for_backward)
    user_outputs = fw_outputs[graph_meta.num_mutate_inputs : num_inner_fwd_outputs]
    if len(user_outputs) == 1:
        user_outputs = user_outputs[0]
    return user_outputs, save_for_backward


def _run_full_bw_module(
    bw_module: Any, graph_meta: GraphMeta, bw_args
) -> tuple[list[Any], list[Any]]:
    bw_outputs = _execute_graph(bw_module, bw_args)
    num_params_buffers = graph_meta.num_params + graph_meta.num_buffers
    param_buffer_grads = bw_outputs[:num_params_buffers]
    input_grads = bw_outputs[num_params_buffers:]
    return input_grads, param_buffer_grads


def _run_dI_bw_module(
    bw_dI_module: Any,
    graph_meta: GraphMeta,
    bw_dI_args,
) -> tuple[list[Any], list[Any]]:
    inp_grads_and_activations = _execute_graph(bw_dI_module, bw_dI_args)
    inp_grads, activations = inp_grads_and_activations[
        : graph_meta.num_input_grads
    ], list(inp_grads_and_activations[graph_meta.num_input_grads :])
    return inp_grads, activations


def _run_dW_bw_module(
    bw_dW_module: Any,
    graph_meta: GraphMeta,
    bw_dW_args,
) -> list[Any]:
    param_buffer_grads = _execute_graph(bw_dW_module, bw_dW_args)
    return param_buffer_grads


def _run_unshard_module(
    unshard_module: Any,
    graph_meta: GraphMeta,
    unshard_args,
) -> list[Any]:
    unsharded_params = _execute_graph(unshard_module, unshard_args)
    return unsharded_params


def _run_reduce_grad_module(
    reduce_grad_module: Any,
    graph_meta: GraphMeta,
    reduce_grad_args,
) -> list[Any]:
    sharded_grads = _execute_graph(reduce_grad_module, reduce_grad_args)
    return sharded_grads


def _run_multiplexed_fw_bw_module(
    multiplexed_fw_bw_module: Any,
    fw_graph_meta: GraphMeta,
    bw_graph_meta: GraphMeta,
    bw_fw_args,
) -> tuple[list[Any], list[Any], Any, tuple[tuple[Any], tuple[Any]]]:
    multiplexed_outs = _execute_graph(multiplexed_fw_bw_module, bw_fw_args)

    num_params_buffers = bw_graph_meta.num_params + bw_graph_meta.num_buffers
    num_bw_outs = bw_graph_meta.num_input_grads + num_params_buffers
    bw_outputs = multiplexed_outs[:num_bw_outs]
    param_buffer_grads = bw_outputs[:num_params_buffers]
    input_grads = bw_outputs[num_params_buffers:]

    fw_outputs = multiplexed_outs[num_bw_outs:]
    num_inner_fwd_outputs = (
        fw_graph_meta.num_mutate_inputs + fw_graph_meta.num_user_outputs
    )
    saved_intermediates = fw_outputs[num_inner_fwd_outputs:]
    num_tensors_for_backward = (
        len(saved_intermediates) - fw_graph_meta.num_symints_saved_for_bw
    )
    tensors_for_backward = saved_intermediates[:num_tensors_for_backward]
    non_tensors_for_backward = saved_intermediates[num_tensors_for_backward:]
    save_for_backward = (tensors_for_backward, non_tensors_for_backward)
    user_outputs = fw_outputs[fw_graph_meta.num_mutate_inputs : num_inner_fwd_outputs]
    if len(user_outputs) == 1:
        user_outputs = user_outputs[0]

    return input_grads, param_buffer_grads, user_outputs, save_for_backward


def _get_stage_from_action(
    action: _Action,
    ctx: _PipelineContext,
) -> tuple[_PipelineScheduleRuntime, dict[int, GraphPipelineStage], GraphPipelineStage]:
    """Helper to extract schedule, stage mapping, and specific stage from action and context.

    Args:
        action: The action containing the stage index.
        ctx: The pipeline context containing the schedule.

    Returns:
        A tuple containing:
            - schedule: The pipeline schedule runtime object.
            - stage_index_to_stage: Dictionary mapping stage indices to GraphPipelineStage objects.
            - stage: The specific GraphPipelineStage for the action's stage index.
    """
    schedule = ctx.schedule_ref
    assert isinstance(schedule, _PipelineScheduleRuntime)
    stage_index_to_stage: dict[int, GraphPipelineStage] = {
        stage.stage_index: cast(GraphPipelineStage, stage) for stage in schedule._stages
    }
    stage = stage_index_to_stage[action.stage_index]
    return schedule, stage_index_to_stage, stage


def _prepare_fwd_common(
    action: _Action,
    ctx: _PipelineContext,
) -> tuple[
    _PipelineScheduleRuntime,
    dict[int, GraphPipelineStage],
    GraphPipelineStage,
    int,
    bool,
    bool,
]:
    """Common setup for forward stage: retrieve stage info and handle recv ops.

    This function performs the shared initialization logic for forward operations,
    including waiting for activation receives from the previous pipeline stage.

    Args:
        action: The forward action to execute, containing the stage index and microbatch index.
        ctx: The pipeline context containing the schedule and pipeline state.

    Returns:
        A tuple containing:
            - schedule: The pipeline schedule runtime object managing the execution.
            - stage_index_to_stage: Dictionary mapping stage indices to GraphPipelineStage objects.
            - stage: The GraphPipelineStage for which forward is being computed.
            - mb_index: The microbatch index being processed.
            - is_next_stage_on_this_rank: True if stage_index + 1 exists on this rank (V-schedule).
            - is_prev_stage_on_this_rank: True if stage_index - 1 exists on this rank (V-schedule).
    """
    schedule, stage_index_to_stage, stage = _get_stage_from_action(action, ctx)
    stage_index = stage.stage_index

    mb_index = action.microbatch_index
    assert mb_index is not None

    is_next_stage_on_this_rank = stage_index + 1 in stage_index_to_stage
    is_prev_stage_on_this_rank = stage_index - 1 in stage_index_to_stage

    if (
        not stage.is_first
        # no recv op expected for V-schedule special case (see [Note: V-schedule special case])
        and not is_prev_stage_on_this_rank
    ):
        fwd_recv_ops = schedule.fwd_recv_ops
        assert (
            stage_index,
            mb_index,
        ) in fwd_recv_ops, f"Computing {action=} before receiving input"
        _wait_batch_p2p(fwd_recv_ops.pop((stage_index, mb_index)))

    return (
        schedule,
        stage_index_to_stage,
        stage,
        mb_index,
        is_next_stage_on_this_rank,
        is_prev_stage_on_this_rank,
    )


def _prepare_fwd_args(
    stage: GraphPipelineStage,
    mb_index: int,
    ctx: _PipelineContext,
) -> list[Any]:
    """Prepare forward args from user inputs or received activations.

    Args:
        stage: The GraphPipelineStage for which to prepare forward arguments.
        mb_index: The microbatch index being processed.
        ctx: The pipeline context containing arg_mbs, kwarg_mbs, and target_mbs.

    Returns:
        List of forward arguments including unsharded_params, buffers, and composite_args.
    """
    arg_mbs = ctx.arg_mbs
    kwarg_mbs = ctx.kwarg_mbs

    args = arg_mbs[mb_index]  # type: ignore[index]
    kwargs = kwarg_mbs[mb_index]  # type: ignore[index]
    assert not kwargs  # TODO: if kwargs can always be ignored, maybe remove?

    if stage.is_first:
        # First stage doesn't need to receive anything
        composite_args = args
    else:
        # Receive activations for this chunk
        # Activations only come in args form
        composite_args = stage._retrieve_recv_activations(mb_index)
        if stage.is_last and ctx.target_mbs is not None:
            assert isinstance(
                composite_args, tuple
            ), f"Expected composite args to be a tuple but got {type(composite_args)}"
            composite_args = composite_args + (ctx.target_mbs[mb_index],)  # type: ignore[index]

    # stage._validate_fwd_input(args, kwargs) Maybe need to validate composite args?
    fw_args = [
        *stage.state["unsharded_params"],
        *stage.state["buffers"],
        *composite_args,
    ]
    del composite_args
    return fw_args


def _post_fwd_common(
    action: _Action,
    stage: GraphPipelineStage,
    mb_index: int,
    output: Any,
    saved_intermediates: tuple[tuple[Any], tuple[Any]],
    schedule: _PipelineScheduleRuntime,
    stage_index_to_stage: dict[int, GraphPipelineStage],
    ctx: _PipelineContext,
    is_next_stage_on_this_rank: bool,
) -> None:
    """Common post-processing after forward pass: cache outputs and propagate.

    This function handles the shared finalization logic for forward operations,
    including normalizing outputs, caching for backward, validating outputs,
    and propagating activations to the next pipeline stage.

    Args:
        stage: The stage that just completed forward computation.
        mb_index: The microbatch index that was processed.
        output: The output from the forward pass.
        saved_intermediates: The intermediates saved for backward pass.
        schedule: The pipeline schedule runtime object.
        stage_index_to_stage: Dictionary mapping stage indices to GraphPipelineStage objects.
        ctx: The pipeline context.
        is_next_stage_on_this_rank: True if the next stage exists on this rank.
    """
    # See [Note: pipeline model output type]
    output_tuple = _normalize_model_output_as_tuple(output)

    # Prepare for final output merge or reduction
    # Output chunks is only used for the last stage since we only merge the output of the last stage
    if stage.is_last:
        stage.output_chunks.append(output)
        if ctx.target_mbs is not None:
            ctx.schedule_ref._internal_losses.append(output)

    stage.fwd_cache[mb_index] = (output_tuple, saved_intermediates)  # type: ignore[assignment]

    if hasattr(stage, "_validate_fwd_outputs"):
        stage._validate_fwd_outputs(output_tuple)

    schedule._maybe_compute_loss(stage, output, ctx.target_mbs, mb_index)

    # SEND/RECV op are avoided for special case with 2 adjacent stages on same rank
    # see [Note: V-schedule special case]
    if is_next_stage_on_this_rank:
        stage_index_to_stage[stage.stage_index + 1].set_local_fwd_input(
            output, mb_index
        )


def stage_forward(
    action: _Action,
    ctx: _PipelineContext,
) -> None:
    (
        schedule,
        stage_index_to_stage,
        stage,
        mb_index,
        is_next_stage_on_this_rank,
        is_prev_stage_on_this_rank,
    ) = _prepare_fwd_common(action, ctx)

    fw_args = _prepare_fwd_args(stage, mb_index, ctx)

    logger.debug(
        "GraphPPRunner running action %s",
        action,
    )
    output, saved_intermediates = _run_fw_module(
        stage.graph_callables.fw,
        stage.graph_meta,
        fw_args,
    )

    _post_fwd_common(
        action,
        stage,
        mb_index,
        output,
        saved_intermediates,
        schedule,
        stage_index_to_stage,
        ctx,
        is_next_stage_on_this_rank,
    )


def _prepare_backward_common(
    action: _Action,
    ctx: _PipelineContext,
) -> tuple[
    _PipelineScheduleRuntime,
    dict[int, GraphPipelineStage],
    GraphPipelineStage,
    int,
    bool,
    bool,
]:
    """Common setup for backward stages: retrieve stage info and handle recv ops.

    This function performs the shared initialization logic for all backward operations,
    including waiting for gradient receives from the next pipeline stage and incrementing
    the backward counter.

    Args:
        action: The backward action to execute, containing the stage index and microbatch index.
        ctx: The pipeline context containing the schedule and pipeline state.

    Returns:
        A tuple containing:
            - schedule: The pipeline schedule runtime object managing the execution.
            - stage_index_to_stage: Dictionary mapping stage indices to GraphPipelineStage objects.
            - bw_stage: The GraphPipelineStage for which backward is being computed.
            - bw_mb_index: The microbatch index being processed.
            - is_next_stage_on_this_rank: True if stage_index + 1 exists on this rank (V-schedule).
            - is_prev_stage_on_this_rank: True if stage_index - 1 exists on this rank (V-schedule).
    """
    schedule, stage_index_to_stage, bw_stage = _get_stage_from_action(action, ctx)

    bw_mb_index = action.microbatch_index
    assert bw_mb_index is not None
    is_next_stage_on_this_rank = bw_stage.stage_index + 1 in stage_index_to_stage
    is_prev_stage_on_this_rank = bw_stage.stage_index - 1 in stage_index_to_stage

    if not bw_stage.is_last and not is_next_stage_on_this_rank:
        bwd_recv_ops = schedule.bwd_recv_ops
        assert (
            bw_stage.stage_index,
            bw_mb_index,
        ) in bwd_recv_ops, f"Attempted to run compute {action=} before receiving input"
        _wait_batch_p2p(bwd_recv_ops.pop((bw_stage.stage_index, bw_mb_index)))

    schedule.backward_counter[bw_stage.stage_index] += 1

    return (
        schedule,
        stage_index_to_stage,
        bw_stage,
        bw_mb_index,
        is_next_stage_on_this_rank,
        is_prev_stage_on_this_rank,
    )


def _prepare_backward_args(
    bw_stage: GraphPipelineStage,
    bw_mb_index: int,
) -> list[Any]:
    """Prepare backward kwargs from cached forward outputs."""
    (
        stage_output,
        saved_intermediates,
    ) = bw_stage.fwd_cache.pop(bw_mb_index)

    if bw_stage.is_last:
        assert len(stage_output) == 1
        loss = stage_output[0]
        tangents = (torch.ones_like(loss),)
    else:
        tangents = bw_stage._retrieve_recv_grads(bw_mb_index)

    tensors_for_backward, non_tensors_for_backward = saved_intermediates

    bw_args = [
        *non_tensors_for_backward,
        *tensors_for_backward,
        *tangents,
    ]
    del tensors_for_backward, non_tensors_for_backward, tangents, saved_intermediates
    return bw_args


def _post_backward_common(
    bw_stage: GraphPipelineStage,
    bw_mb_index: int,
    input_grads: list[Any],
    stage_index_to_stage: dict[int, GraphPipelineStage],
    is_prev_stage_on_this_rank: bool,
) -> None:
    """Common post-processing after backward pass: cache input grads and propagate.

    This function handles the shared finalization logic for backward operations,
    including caching input gradients and propagating gradients to the previous
    pipeline stage.

    Note: Gradient accumulation and scaling are NOT included here as they occur
    at different points for full_backward vs split dI/dW:
    - full_backward: accumulation and scaling happen immediately after backward
    - split dI/dW: accumulation and scaling happen in backward_weight (dW), not backward_input (dI)

    Args:
        bw_stage: The stage that just completed backward computation.
        bw_mb_index: The microbatch index that was processed.
        input_grads: The computed input gradients to cache.
        stage_index_to_stage: Dictionary mapping stage indices to GraphPipelineStage objects.
        is_prev_stage_on_this_rank: True if the previous stage exists on this rank.
    """
    bw_stage.bwd_cache[bw_mb_index] = (
        tuple(input_grads) if not isinstance(input_grads, tuple) else input_grads
    )

    if is_prev_stage_on_this_rank:
        stage_index_to_stage[bw_stage.stage_index - 1].set_local_bwd_input(
            bw_stage.get_local_bwd_output(bw_mb_index),
            bw_mb_index,
        )


def stage_full_backward(
    action: _Action,
    ctx: _PipelineContext,
) -> None:
    (
        schedule,
        stage_index_to_stage,
        bw_stage,
        bw_mb_index,
        is_next_stage_on_this_rank,
        is_prev_stage_on_this_rank,
    ) = _prepare_backward_common(action, ctx)

    last_backward = (
        schedule.backward_counter[bw_stage.stage_index] == schedule._n_microbatches
    )
    grad_scale_factor = schedule._n_microbatches if schedule.scale_grads else 1

    if not bw_stage.has_backward:
        logger.debug("Returning early for backward stage")
        return

    bw_args = _prepare_backward_args(bw_stage, bw_mb_index)

    logger.debug(
        "GraphPPRunner running action %s",
        action,
    )
    input_grads, param_buffer_grads = _run_full_bw_module(
        bw_stage.graph_callables.full_bw,
        bw_stage.graph_meta,
        bw_args,

    )
    bw_stage._accumulate_stage_unsharded_grads(param_buffer_grads)

    _post_backward_common(
        bw_stage,
        bw_mb_index,
        input_grads,
        stage_index_to_stage,
        is_prev_stage_on_this_rank,
    )

    if last_backward:
        bw_stage.scale_grads(grad_scale_factor)


def stage_backward_input(
    action: _Action,
    ctx: _PipelineContext,
) -> None:
    schedule, stage_index_to_stage, bw_stage = _get_stage_from_action(action, ctx)

    if bw_stage.is_first and bw_stage.graph_callables.bw_dI is None:
        # First stage does not have bw_dI graph since usually the inputs of the first stage do not require gradients
        # Hence, we do not do a split_dI_dW pass, and call full backward instead during dI action
        logger.debug(
            "GraphPPRunner skipping action %s",
            action,
        )
        new_action = _Action(
            action.stage_index,
            FULL_BACKWARD,
            action.microbatch_index,
            action.sub_actions,
        )
        stage_full_backward(new_action, ctx)
        return

    (
        schedule,
        stage_index_to_stage,
        bw_stage,
        bw_mb_index,
        is_next_stage_on_this_rank,
        is_prev_stage_on_this_rank,
    ) = _prepare_backward_common(action, ctx)

    if not bw_stage.has_backward:
        logger.debug("Returning early for backward stage")
        return

    bw_args = _prepare_backward_args(bw_stage, bw_mb_index)

    logger.debug(
        "GraphPPRunner running action %s",
        action,
    )
    assert bw_stage.graph_callables.bw_dI is not None
    input_grads, activations_for_backward = _run_dI_bw_module(
        bw_stage.graph_callables.bw_dI,
        bw_stage.graph_meta,
        bw_args,

    )

    bw_stage.bwd_activation_cache[bw_mb_index] = (
        tuple(activations_for_backward)
        if not isinstance(activations_for_backward, tuple)
        else activations_for_backward
    )

    _post_backward_common(
        bw_stage,
        bw_mb_index,
        input_grads,
        stage_index_to_stage,
        is_prev_stage_on_this_rank,
    )


def stage_backward_weight(
    action: _Action,
    ctx: _PipelineContext,
) -> None:
    schedule, stage_index_to_stage, bw_stage = _get_stage_from_action(action, ctx)
    bw_mb_index = action.microbatch_index
    assert bw_mb_index is not None
    if bw_stage.is_first and bw_stage.graph_callables.bw_dW is None:
        # First stage does not have bw_dW graph since usually the inputs of the first stage do not require gradients
        # Hence, we do not do a split_dI_dW pass, and call full backward instead during dI action
        # which also performs dW implicitly, hence we skip this step.
        logger.debug(
            "GraphPPRunner skipping action %s",
            action,
        )
        return

    last_backward = (
        schedule.backward_counter[bw_stage.stage_index] == schedule._n_microbatches
    )
    grad_scale_factor = schedule._n_microbatches if schedule.scale_grads else 1

    if not bw_stage.has_backward:
        logger.debug("Returning early for backward stage")
        return

    activations_for_backward = bw_stage.bwd_activation_cache.pop(bw_mb_index)
    logger.debug(
        "GraphPPRunner running action %s",
        action,
    )
    bw_args = list(activations_for_backward)
    del activations_for_backward
    assert bw_stage.graph_callables.bw_dW is not None
    param_buffer_grads = _run_dW_bw_module(
        bw_stage.graph_callables.bw_dW,
        bw_stage.graph_meta,
        bw_args,

    )
    bw_stage._accumulate_stage_unsharded_grads(param_buffer_grads)

    if last_backward:
        bw_stage.scale_grads(grad_scale_factor)


def overlap_fw_bw(
    multiplexed_graph_callables: dict[tuple[int, int], fx.GraphModule],
    action: _Action,
    ctx: _PipelineContext,
) -> None:
    assert action.sub_actions is not None, "Expected sub actions for overlap callback"
    fw_action = action.sub_actions[0]
    bw_action = action.sub_actions[1]

    (
        schedule,
        stage_index_to_stage,
        fw_stage,
        fw_mb_index,
        fw_is_next_stage_on_this_rank,
        fw_is_prev_stage_on_this_rank,
    ) = _prepare_fwd_common(fw_action, ctx)

    (
        _,
        _,
        bw_stage,
        bw_mb_index,
        bw_is_next_stage_on_this_rank,
        bw_is_prev_stage_on_this_rank,
    ) = _prepare_backward_common(bw_action, ctx)

    last_backward = (
        schedule.backward_counter[bw_stage.stage_index] == schedule._n_microbatches
    )
    grad_scale_factor = schedule._n_microbatches if schedule.scale_grads else 1

    if not bw_stage.has_backward:
        logger.debug("Returning early for backward stage")
        return

    fw_args = _prepare_fwd_args(fw_stage, fw_mb_index, ctx)
    bw_args = _prepare_backward_args(bw_stage, bw_mb_index)
    bw_fw_args = bw_args + fw_args
    del bw_args, fw_args
    multiplexed_fw_bw_module = multiplexed_graph_callables.get(
        (fw_action.stage_index, bw_action.stage_index)
    )
    assert (
        multiplexed_fw_bw_module is not None
    ), "Expected multiplexed graph callables for overlap callback"
    logger.debug(
        "GraphPPRunner running action %s",
        action,
    )
    (
        input_grads,
        param_buffer_grads,
        output,
        saved_intermediates,
    ) = _run_multiplexed_fw_bw_module(
        multiplexed_fw_bw_module,
        fw_stage.graph_meta,
        bw_stage.graph_meta,
        bw_fw_args,

    )

    bw_stage._accumulate_stage_unsharded_grads(param_buffer_grads)

    _post_fwd_common(
        action,
        fw_stage,
        fw_mb_index,
        output,
        saved_intermediates,
        schedule,
        stage_index_to_stage,
        ctx,
        fw_is_next_stage_on_this_rank,
    )
    _post_backward_common(
        bw_stage,
        bw_mb_index,
        input_grads,
        stage_index_to_stage,
        bw_is_prev_stage_on_this_rank,
    )

    if last_backward:
        bw_stage.scale_grads(grad_scale_factor)


def stage_unshard(
    action: _Action,
    ctx: _PipelineContext,
) -> None:
    schedule, stage_index_to_stage, stage = _get_stage_from_action(action, ctx)
    logger.debug(
        "GraphPPRunner running action %s",
        action,
    )
    if stage.graph_callables.unshard is None:
        stage.state["unsharded_params"] = stage.state["sharded_params"]
    else:
        sharded_params = list(stage.state["sharded_params"])
        unsharded_params = _run_unshard_module(
            stage.graph_callables.unshard,
            stage.graph_meta,
            sharded_params,

        )
        stage.state["unsharded_params"] = unsharded_params


def stage_reshard(
    action: _Action,
    ctx: _PipelineContext,
):
    schedule, stage_index_to_stage, stage = _get_stage_from_action(action, ctx)
    logger.debug(
        "GraphPPRunner running action %s",
        action,
    )
    stage.state["unsharded_params"] = []


def stage_reduce_grad(
    action: _Action,
    ctx: _PipelineContext,
) -> None:
    schedule, stage_index_to_stage, stage = _get_stage_from_action(action, ctx)
    logger.debug(
        "GraphPPRunner running action %s",
        action,
    )
    if stage.graph_callables.reduce_grad is None:
        stage.state["sharded_grads"] = stage.state["unsharded_grads"]
    else:
        sharded_grads = _run_reduce_grad_module(
            stage.graph_callables.reduce_grad,
            stage.graph_meta,
            stage.state["unsharded_grads"],

        )
        stage.state["sharded_grads"] = sharded_grads


class GraphPPRunner:
    def __init__(
        self,
        schedule: _PipelineScheduleRuntime,
    ):
        self.schedule = schedule
        if not schedule._backward_requires_autograd:
            assert all(
                isinstance(stage, GraphPipelineStage)
                and (
                    stage.graph_callables.full_bw is not None
                    or (
                        stage.graph_callables.bw_dI is not None
                        and stage.graph_callables.bw_dW is not None
                    )
                )
                for stage in schedule._stages
            )
            self.schedule._has_backward = True
        for stage in schedule._stages:
            assert isinstance(stage, GraphPipelineStage)

    def _populate_stage_states(self, stage: GraphPipelineStage) -> None:
        sharded_params = [
            v.to_local() if isinstance(v, DTensor) else v
            for k, v in dict(
                stage.submod.named_parameters(remove_duplicate=False)
            ).items()
        ]
        buffers = [
            v.to_local() if isinstance(v, DTensor) else v
            for k, v in dict(stage.submod.named_buffers(remove_duplicate=False)).items()
        ]
        stage.state["sharded_params"] = sharded_params
        stage.state["buffers"] = buffers
        stage.state["unsharded_grads"] = [None] * len(sharded_params)

    def _accumulate_stage_sharded_grads(self, stage: GraphPipelineStage) -> None:
        grads = stage.state["sharded_grads"]
        params = list(stage.submod.parameters())
        for param, grad in zip(params, grads):
            if param.requires_grad and grad is not None:
                assert isinstance(grad, torch.Tensor)
                if isinstance(param, DTensor):
                    param_spec = param._spec
                    _grad = DTensor.from_local(
                        grad,
                        device_mesh=param_spec.device_mesh,
                        placements=param_spec.placements,
                        shape=param_spec.shape,
                        stride=param_spec.stride,
                    )
                else:
                    _grad = grad  # type: ignore[assignment]
                if param.grad is None:
                    param.grad = _grad
                else:
                    param.grad += _grad

    def step(self, *args, **kwargs) -> None:
        has_targets_and_loss = (
            "losses" in kwargs and "targets" in kwargs if kwargs else False
        )
        for stage in self.schedule._stages:
            assert isinstance(stage, GraphPipelineStage)
            self._populate_stage_states(stage)

        self.schedule.step(*args, **kwargs)

        for stage in self.schedule._stages:
            assert isinstance(stage, GraphPipelineStage)
            self._accumulate_stage_sharded_grads(stage)
            stage.state.clear()

        if has_targets_and_loss:
            losses = kwargs["losses"]
            assert len(self.schedule._internal_losses) == self.schedule._n_microbatches
            losses.extend(self.schedule._internal_losses)
            self.schedule._internal_losses.clear()
