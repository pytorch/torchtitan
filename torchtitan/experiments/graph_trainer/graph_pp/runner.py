# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""GraphPP runtime action handlers.

GraphPP uses upstream runtime PP schedules for ordering, communication,
microbatch splitting, and stage metadata initialization. This module only maps
schedule actions onto bound stage graph executors.
"""

from typing import Any, cast

import torch
from torch.distributed.pipelining.schedules import (
    _Action,
    _PipelineContext,
    _PipelineScheduleRuntime,
    _wait_batch_p2p,
    BACKWARD_INPUT,
    BACKWARD_WEIGHT,
    FORWARD,
    FULL_BACKWARD,
    OVERLAP_F_B,
    REDUCE_GRAD,
    RESHARD,
    UNSHARD,
)
from torch.distributed.pipelining.stage import _normalize_model_output_as_tuple

from torchtitan.experiments.graph_trainer.common_utils import accumulate_param_grads_
from torchtitan.experiments.graph_trainer.graph_pp.stage import (
    GraphPipelineStage,
    GraphPPOverlapGraphs,
    GraphPPStageGraphs,
    GraphPPStageGraphsProvider,
)
from torchtitan.experiments.graph_trainer.graph_pp.utils import (
    flatten_graph_values,
    overlap_fw_bw_sub_actions,
)
from torchtitan.tools.logging import logger


__all__ = [
    "GraphPPRunner",
    "register_graph_pp_schedule",
]


def _scale_grad_values_(grads: list[Any], grad_scale_factor: int) -> None:
    if grad_scale_factor == 1:
        return
    for grad in grads:
        if isinstance(grad, torch.Tensor):
            grad.div_(grad_scale_factor)


def _scale_graph_pp_sharded_grads(
    stage: GraphPipelineStage,
    schedule: _PipelineScheduleRuntime,
) -> None:
    if stage._graph_pp_grads_scaled:
        return
    grad_scale_factor = schedule._n_microbatches if schedule.scale_grads else 1
    _scale_grad_values_(stage.state.sharded_param_grads, grad_scale_factor)
    stage._graph_pp_grads_scaled = True


def _accumulate_flat_grad_values_(
    accumulated: list[Any],
    grads: list[Any],
    *,
    label: str,
) -> None:
    """Accumulate raw flat graph-gradient values before boundary rewrapping."""
    if len(grads) != len(accumulated):
        raise ValueError(
            f"GraphPP {label} grad count mismatch: "
            f"expected {len(accumulated)}, got {len(grads)}"
        )
    for index, grad in enumerate(grads):
        if grad is None:
            continue
        if not isinstance(grad, torch.Tensor):
            if accumulated[index] is None:
                accumulated[index] = grad
            elif accumulated[index] is not grad and accumulated[index] != grad:
                raise ValueError(
                    "GraphPP flat gradient metadata changed across "
                    f"microbatches at index {index}: "
                    f"{accumulated[index]!r} != {grad!r}"
                )
            continue
        if accumulated[index] is None:
            accumulated[index] = grad
        else:
            accumulated[index] += grad


def _require_stage_graphs(
    stage: GraphPipelineStage,
    action_name: str,
) -> GraphPPStageGraphs:
    if stage.graphs is None:
        raise ValueError(
            "GraphPP stage graphs must be built before runtime execution. "
            f"Missing graphs for stage {stage.stage_index} action {action_name}."
        )
    return stage.graphs


def _ensure_unsharded_param_values(stage: GraphPipelineStage) -> None:
    graphs = _require_stage_graphs(stage, "UNSHARD")
    if stage.state.unsharded_param_values:
        return
    stage.state.unsharded_param_values = graphs.unshard_params(
        stage.state.flat_param_values
    )


def _accumulate_stage_unsharded_grads(
    stage: GraphPipelineStage,
    grads: list[Any],
) -> None:
    _accumulate_flat_grad_values_(
        stage.state.unsharded_param_grads,
        grads,
        label="unsharded",
    )


def _prepare_fwd_user_args(
    stage: GraphPipelineStage,
    mb_index: int,
    ctx: _PipelineContext,
) -> tuple[tuple[Any, ...], dict[str, Any], Any]:
    """Package runtime forward inputs for a stage graph call.

    Upstream PP owns microbatch splitting and stage metadata initialization.
    This helper only translates those upstream runtime containers into the
    GraphPP forward calling convention.
    """
    arg_mbs = ctx.arg_mbs
    kwarg_mbs = ctx.kwarg_mbs
    if arg_mbs is None or kwarg_mbs is None:
        raise ValueError("GraphPP forward requires upstream PP microbatch inputs")
    kwargs = kwarg_mbs[mb_index]
    if stage.is_first:
        args = arg_mbs[mb_index]
    else:
        args = _normalize_model_output_as_tuple(
            stage._retrieve_recv_activations(mb_index)
        )
    target = ctx.target_mbs[mb_index] if stage.is_last and ctx.target_mbs else None
    return tuple(args), kwargs, target


def _stage_map_and_stage_from_action(
    schedule: _PipelineScheduleRuntime,
    action: _Action,
) -> tuple[dict[int, GraphPipelineStage], GraphPipelineStage]:
    """Return local GraphPP stages for an upstream schedule action.

    This is PP schedule mechanics, not runner state: upstream schedules address
    stages by global stage index, while GraphPP handlers need both the selected
    local stage and the local-stage map for same-rank neighbor propagation.
    """
    stage_index_to_stage = {
        stage.stage_index: cast(GraphPipelineStage, stage) for stage in schedule._stages
    }
    return (
        stage_index_to_stage,
        stage_index_to_stage[action.stage_index],
    )


def _prepare_fwd_common(
    schedule: _PipelineScheduleRuntime,
    action: _Action,
) -> tuple[dict[int, GraphPipelineStage], GraphPipelineStage, int, bool]:
    """Prepare PP schedule state before one GraphPP forward graph call."""
    # 1. Resolve the local stage and rank-local topology for this action.
    stage_index_to_stage, stage = _stage_map_and_stage_from_action(schedule, action)
    mb_index = action.microbatch_index
    if mb_index is None:
        raise ValueError(f"GraphPP FORWARD action must have microbatch index: {action}")
    is_next_stage_on_this_rank = stage.stage_index + 1 in stage_index_to_stage
    is_prev_stage_on_this_rank = stage.stage_index - 1 in stage_index_to_stage

    # 2. Remote previous-stage activations must arrive before the graph reads
    #    the upstream PP receive buffer. Same-rank V-schedule neighbors bypass
    #    P2P and use PipelineStage local input caches.
    if not stage.is_first and not is_prev_stage_on_this_rank:
        fwd_recv_ops = schedule.fwd_recv_ops
        if (stage.stage_index, mb_index) not in fwd_recv_ops:
            raise ValueError(
                "GraphPP missing forward recv op for "
                f"stage {stage.stage_index}, microbatch {mb_index}."
            )
        _wait_batch_p2p(fwd_recv_ops.pop((stage.stage_index, mb_index)))
    return (
        stage_index_to_stage,
        stage,
        mb_index,
        is_next_stage_on_this_rank,
    )


def _post_fwd_common(
    stage: GraphPipelineStage,
    mb_index: int,
    output: Any,
    saved_values_for_backward: tuple[Any, ...],
    schedule: _PipelineScheduleRuntime,
    stage_index_to_stage: dict[int, GraphPipelineStage],
    is_next_stage_on_this_rank: bool,
) -> None:
    """Store forward graph outputs and propagate same-rank activations."""
    # 1. Cache user outputs plus graph-saved values for the later backward
    #    action. Last-stage losses are kept in upstream's internal loss list so
    #    schedule._update_losses() remains the only public loss updater.
    output_tuple = _normalize_model_output_as_tuple(output)
    if stage.is_last:
        stage.output_chunks.append(output)
        schedule._internal_losses.append(output)
    stage.fwd_cache[mb_index] = (output_tuple, saved_values_for_backward)

    # 2. Adjacent same-rank stages avoid SEND/RECV actions, so hand the output
    #    directly to the next stage's upstream local forward-input cache.
    if is_next_stage_on_this_rank:
        stage_index_to_stage[stage.stage_index + 1].set_local_fwd_input(
            output, mb_index
        )


def _prepare_backward_values(
    stage: GraphPipelineStage,
    mb_index: int,
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
    """Package cached forward values and received output grads for backward."""
    stage_output, saved_values_for_backward = stage.fwd_cache.pop(mb_index)
    if stage.is_last:
        output_grads_from_next = ()
    else:
        output_grads_from_next = _normalize_model_output_as_tuple(
            stage._retrieve_recv_grads(mb_index)
        )
    return stage_output, saved_values_for_backward, output_grads_from_next


def _prepare_backward_common(
    schedule: _PipelineScheduleRuntime,
    action: _Action,
) -> tuple[dict[int, GraphPipelineStage], GraphPipelineStage, int, bool]:
    """Prepare PP schedule state before one GraphPP backward graph call."""
    # 1. Resolve the local stage and rank-local topology for this action.
    stage_index_to_stage, stage = _stage_map_and_stage_from_action(schedule, action)
    mb_index = action.microbatch_index
    if mb_index is None:
        raise ValueError(
            f"GraphPP backward action must have microbatch index: {action}"
        )
    is_next_stage_on_this_rank = stage.stage_index + 1 in stage_index_to_stage
    is_prev_stage_on_this_rank = stage.stage_index - 1 in stage_index_to_stage

    # 2. Remote next-stage output grads must arrive before backward reads the
    #    upstream PP grad receive buffer. Same-rank neighbors use local caches.
    if not stage.is_last and not is_next_stage_on_this_rank:
        bwd_recv_ops = schedule.bwd_recv_ops
        if (stage.stage_index, mb_index) not in bwd_recv_ops:
            raise ValueError(
                "GraphPP missing backward recv op for "
                f"stage {stage.stage_index}, microbatch {mb_index}."
            )
        _wait_batch_p2p(bwd_recv_ops.pop((stage.stage_index, mb_index)))

    # 3. Match upstream runtime semantics: full-backward and backward-input
    #    count as the microbatch backward action for reduce-grad placement.
    schedule.backward_counter[stage.stage_index] += 1
    return (
        stage_index_to_stage,
        stage,
        mb_index,
        is_prev_stage_on_this_rank,
    )


def _post_backward_common(
    stage: GraphPipelineStage,
    mb_index: int,
    input_grads: list[Any],
    stage_index_to_stage: dict[int, GraphPipelineStage],
    is_prev_stage_on_this_rank: bool,
) -> None:
    """Store input gradients and propagate same-rank backward inputs."""
    # 1. Cache input grads in upstream's stage cache so SEND_B actions, or
    #    same-rank previous stages, can retrieve them by microbatch.
    stage.bwd_cache[mb_index] = tuple(input_grads)

    # 2. Adjacent same-rank stages avoid SEND/RECV actions, so hand the grads
    #    directly to the previous stage's upstream local backward-input cache.
    if is_prev_stage_on_this_rank:
        stage_index_to_stage[stage.stage_index - 1].set_local_bwd_input(
            stage.get_local_bwd_output(mb_index),
            mb_index,
        )


class GraphPPRunner:
    """Execute a runtime PP schedule with GraphPP stage graph handlers.

    ``GraphPPRunner`` registers bound action handlers on an upstream
    ``_PipelineScheduleRuntime``. Ownership is split along the runtime
    boundary:

    1. Upstream PP owns microbatch splitting, schedule action ordering, P2P op
       creation, and stage metadata initialization.
    2. ``GraphPPRunner`` owns per-step GraphPP runtime state: graph readiness,
       action dispatch, ``loss_kwargs``, overlap graphs, grad scaling, and
       final parameter-gradient accumulation.
    3. ``GraphPPStageGraphs`` owns graph calling conventions, metadata packing,
       FX execution, and output unwrapping for one stage.
    4. Graph construction and tracing stay outside the runner in a
       ``GraphPPStageGraphsProvider`` implementation.

    The runner does not inspect GraphTrainer graph metadata. Calling
    conventions live behind ``GraphPPStageGraphs`` implementations attached to
    each ``GraphPipelineStage``. Module-private helpers in this file own the
    upstream PP schedule mechanics that surround each graph call: recv waits,
    forward/backward caches, and same-rank local propagation.

    Args:
        schedule (_PipelineScheduleRuntime): Runtime pipeline schedule to
            execute.
        graph_provider (GraphPPStageGraphsProvider | None): Optional provider
            that attaches bound stage graphs before the first runtime action.
            If omitted, every local stage must already have ``stage.graphs``
            populated.

    Raises:
        TypeError: If any local schedule stage is not a ``GraphPipelineStage``.
    """

    def __init__(
        self,
        schedule: _PipelineScheduleRuntime,
        *,
        graph_provider: GraphPPStageGraphsProvider | None = None,
    ) -> None:
        self.schedule = schedule
        self.graph_provider = graph_provider
        self.overlap_graphs: dict[tuple[int, int], GraphPPOverlapGraphs] = {}
        self.loss_kwargs: dict[str, Any] = {}
        self._graph_pp_ready = False
        self.schedule._has_backward = True
        for stage in schedule._stages:
            if not isinstance(stage, GraphPipelineStage):
                raise TypeError(
                    "GraphPPRunner requires GraphPipelineStage instances, got "
                    f"{type(stage).__name__}"
                )

    def ensure_ready(self, ctx: _PipelineContext) -> None:
        """Ensure local stage graphs and runtime state are ready for execution.

        Args:
            ctx (_PipelineContext): Pipeline schedule context for the current
                step. The graph provider uses it to derive trace inputs and
                overlap graph pairs.

        Raises:
            ValueError: If a local stage has no bound graph executor after the
                optional graph provider runs.
        """
        if self._graph_pp_ready:
            return
        if self.graph_provider is not None:
            self.overlap_graphs = self.graph_provider.ensure_stage_graphs(
                self.schedule,
                ctx,
                loss_kwargs=self.loss_kwargs,
            )
        for stage in self.schedule._stages:
            graph_stage = cast(GraphPipelineStage, stage)
            _require_stage_graphs(graph_stage, "step")
            self._populate_stage_states(graph_stage)
        self._graph_pp_ready = True

    def _populate_stage_states(self, stage: GraphPipelineStage) -> None:
        graphs = _require_stage_graphs(stage, "state initialization")
        flat_param_values = []
        flat_buffer_values = []
        trainable_params = []
        for _, value in stage.submod.named_parameters(remove_duplicate=False):
            flat_param_values.extend(flatten_graph_values([value]))
            if value.requires_grad:
                trainable_params.append(value)
        for _, value in stage.submod.named_buffers(remove_duplicate=False):
            flat_buffer_values.extend(flatten_graph_values([value]))
        stage.state.flat_param_values = flat_param_values
        stage.state.flat_buffer_values = flat_buffer_values
        stage.state.trainable_params = trainable_params
        stage.state.unsharded_param_values = []
        stage.state.unsharded_param_grads = [
            None
        ] * graphs.num_unsharded_param_grad_values
        stage.state.sharded_param_grads = []
        stage._graph_pp_grads_scaled = False

    def _ensure_reduced_grads(self, stage: GraphPipelineStage) -> None:
        if stage.state.sharded_param_grads:
            return
        if not any(grad is not None for grad in stage.state.unsharded_param_grads):
            return
        graphs = _require_stage_graphs(stage, "final gradient accumulation")
        stage.state.sharded_param_grads = graphs.reduce_grads(
            stage.state.unsharded_param_grads
        )
        _scale_graph_pp_sharded_grads(stage, self.schedule)

    def _accumulate_stage_sharded_grads(self, stage: GraphPipelineStage) -> None:
        self._ensure_reduced_grads(stage)
        if not stage.state.sharded_param_grads:
            return
        graphs = _require_stage_graphs(stage, "final gradient accumulation")
        param_grads = graphs.param_grads_for_accumulation(
            stage.state.sharded_param_grads
        )
        accumulate_param_grads_(stage.state.trainable_params, param_grads)

    def _handle_forward(self, action: _Action, ctx: _PipelineContext) -> None:
        self.ensure_ready(ctx)
        (
            stage_index_to_stage,
            stage,
            mb_index,
            is_next_stage_on_this_rank,
        ) = _prepare_fwd_common(self.schedule, action)
        args, kwargs, target = _prepare_fwd_user_args(stage, mb_index, ctx)
        graphs = _require_stage_graphs(stage, "FORWARD")
        _ensure_unsharded_param_values(stage)
        output, saved_values_for_backward = graphs.forward(
            args,
            kwargs,
            target,
            self.loss_kwargs,
            unsharded_param_values=stage.state.unsharded_param_values,
            flat_buffer_values=stage.state.flat_buffer_values,
        )
        _post_fwd_common(
            stage,
            mb_index,
            output,
            saved_values_for_backward,
            self.schedule,
            stage_index_to_stage,
            is_next_stage_on_this_rank,
        )

    def _handle_full_backward(self, action: _Action, ctx: _PipelineContext) -> None:
        self.ensure_ready(ctx)
        (
            stage_index_to_stage,
            stage,
            mb_index,
            is_prev_stage_on_this_rank,
        ) = _prepare_backward_common(self.schedule, action)
        if not stage.has_backward:
            return
        graphs = _require_stage_graphs(stage, "FULL_BACKWARD")
        (
            stage_output,
            saved_values_for_backward,
            output_grads_from_next,
        ) = _prepare_backward_values(stage, mb_index)
        input_grads, param_grads = graphs.full_backward(
            stage_output,
            saved_values_for_backward,
            output_grads_from_next,
        )
        _accumulate_stage_unsharded_grads(stage, param_grads)
        _post_backward_common(
            stage,
            mb_index,
            input_grads,
            stage_index_to_stage,
            is_prev_stage_on_this_rank,
        )

    def _handle_backward_input(self, action: _Action, ctx: _PipelineContext) -> None:
        self.ensure_ready(ctx)
        _, stage = _stage_map_and_stage_from_action(self.schedule, action)
        graphs = _require_stage_graphs(stage, "BACKWARD_INPUT")
        if not graphs.supports_backward_input_weight_split:
            logger.debug(
                "GraphPP skipping BACKWARD_INPUT for stage %s", stage.stage_index
            )
            return
        (
            stage_index_to_stage,
            stage,
            mb_index,
            is_prev_stage_on_this_rank,
        ) = _prepare_backward_common(self.schedule, action)
        if not stage.has_backward:
            return
        graphs = _require_stage_graphs(stage, "BACKWARD_INPUT")
        (
            stage_output,
            saved_values_for_backward,
            output_grads_from_next,
        ) = _prepare_backward_values(stage, mb_index)
        input_grads, saved_values_for_backward_weight = graphs.backward_input(
            stage_output,
            saved_values_for_backward,
            output_grads_from_next,
        )
        stage.saved_values_for_backward_weight_cache[
            mb_index
        ] = saved_values_for_backward_weight
        _post_backward_common(
            stage,
            mb_index,
            input_grads,
            stage_index_to_stage,
            is_prev_stage_on_this_rank,
        )

    def _handle_backward_weight(self, action: _Action, ctx: _PipelineContext) -> None:
        self.ensure_ready(ctx)
        _, stage = _stage_map_and_stage_from_action(self.schedule, action)
        mb_index = action.microbatch_index
        if mb_index is None:
            raise ValueError(
                f"GraphPP BACKWARD_WEIGHT action must have microbatch index: {action}"
            )
        graphs = _require_stage_graphs(stage, "BACKWARD_WEIGHT")
        if not graphs.supports_backward_input_weight_split:
            new_action = _Action(
                action.stage_index,
                FULL_BACKWARD,
                action.microbatch_index,
                action.sub_actions,
            )
            self._handle_full_backward(new_action, ctx)
            return
        if not stage.has_backward:
            return
        saved_values_for_backward_weight = (
            stage.saved_values_for_backward_weight_cache.pop(mb_index)
        )
        param_grads = graphs.backward_weight(saved_values_for_backward_weight)
        _accumulate_stage_unsharded_grads(stage, param_grads)

    def _handle_unshard(self, action: _Action, ctx: _PipelineContext) -> None:
        self.ensure_ready(ctx)
        _, stage = _stage_map_and_stage_from_action(self.schedule, action)
        _ensure_unsharded_param_values(stage)

    def _handle_reshard(self, action: _Action, ctx: _PipelineContext) -> None:
        self.ensure_ready(ctx)
        _, stage = _stage_map_and_stage_from_action(self.schedule, action)
        _require_stage_graphs(stage, "RESHARD")
        stage.state.unsharded_param_values = []

    def _handle_reduce_grad(self, action: _Action, ctx: _PipelineContext) -> None:
        self.ensure_ready(ctx)
        _, stage = _stage_map_and_stage_from_action(self.schedule, action)
        graphs = _require_stage_graphs(stage, "REDUCE_GRAD")
        stage.state.sharded_param_grads = graphs.reduce_grads(
            stage.state.unsharded_param_grads
        )
        _scale_graph_pp_sharded_grads(stage, self.schedule)

    def _handle_overlap_fw_bw(self, action: _Action, ctx: _PipelineContext) -> None:
        fw_action, bw_action = overlap_fw_bw_sub_actions(action)

        self.ensure_ready(ctx)
        (
            stage_index_to_stage,
            fw_stage,
            fw_mb_index,
            fw_is_next_stage_on_this_rank,
        ) = _prepare_fwd_common(self.schedule, fw_action)
        (
            _,
            bw_stage,
            bw_mb_index,
            bw_is_prev_stage_on_this_rank,
        ) = _prepare_backward_common(self.schedule, bw_action)
        if not bw_stage.has_backward:
            return

        args, kwargs, target = _prepare_fwd_user_args(fw_stage, fw_mb_index, ctx)
        _require_stage_graphs(fw_stage, "OVERLAP_F_B")
        _require_stage_graphs(bw_stage, "OVERLAP_F_B")
        _ensure_unsharded_param_values(fw_stage)
        pair = (fw_action.stage_index, bw_action.stage_index)
        # The multiplexed graph is runner-owned state because it is built once
        # from the graph provider and reused across OVERLAP_F_B actions.
        overlap_graph = self.overlap_graphs.get(pair)
        if overlap_graph is None:
            raise ValueError(
                "GraphPP overlap graph must be built before OVERLAP_F_B runtime "
                f"execution for pair {pair}."
            )
        (
            bw_stage_output,
            bw_saved_values_for_backward,
            output_grads_from_next,
        ) = _prepare_backward_values(bw_stage, bw_mb_index)
        (
            input_grads,
            param_grads,
            output,
            saved_values_for_backward,
        ) = overlap_graph.forward_backward(
            backward_stage_output=bw_stage_output,
            backward_saved_values_for_backward=bw_saved_values_for_backward,
            output_grads_from_next=output_grads_from_next,
            forward_args=args,
            forward_kwargs=kwargs,
            forward_target=target,
            forward_loss_kwargs=self.loss_kwargs,
            forward_unsharded_param_values=fw_stage.state.unsharded_param_values,
            forward_flat_buffer_values=fw_stage.state.flat_buffer_values,
        )

        _accumulate_stage_unsharded_grads(bw_stage, param_grads)
        _post_fwd_common(
            fw_stage,
            fw_mb_index,
            output,
            saved_values_for_backward,
            self.schedule,
            stage_index_to_stage,
            fw_is_next_stage_on_this_rank,
        )
        _post_backward_common(
            bw_stage,
            bw_mb_index,
            input_grads,
            stage_index_to_stage,
            bw_is_prev_stage_on_this_rank,
        )

    def step(self, *args: Any, **kwargs: Any) -> None:
        """Run one training step through the wrapped pipeline schedule.

        Args:
            *args (Any): Positional arguments forwarded to ``schedule.step``.
            **kwargs (Any): Keyword arguments forwarded to ``schedule.step``.
                GraphPP reads ``loss_kwargs`` from this mapping and forwards
                all kwargs to the upstream schedule unchanged.
        """
        self._graph_pp_ready = False
        self.loss_kwargs = kwargs.get("loss_kwargs") or {}
        step_succeeded = False
        try:
            self.schedule.step(*args, **kwargs)
            step_succeeded = True
        finally:
            for stage in self.schedule._stages:
                graph_stage = cast(GraphPipelineStage, stage)
                if step_succeeded:
                    self._accumulate_stage_sharded_grads(graph_stage)
                graph_stage.state.clear()
            self.loss_kwargs = {}
            self._graph_pp_ready = False

    def eval(self, *args: Any, **kwargs: Any) -> Any:
        """Run evaluation through the wrapped pipeline schedule.

        Evaluation reuses upstream PP eval semantics, which disable backward
        and delegate to ``schedule.step``. GraphPP only mirrors the per-step
        runtime-state setup and cleanup from training, without accumulating
        parameter gradients.

        Args:
            *args (Any): Positional arguments forwarded to ``schedule.eval``.
            **kwargs (Any): Keyword arguments forwarded to ``schedule.eval``.

        Returns:
            Any: The value returned by ``schedule.eval``.
        """
        self._graph_pp_ready = False
        self.loss_kwargs = kwargs.get("loss_kwargs") or {}
        try:
            return self.schedule.eval(*args, **kwargs)
        finally:
            for stage in self.schedule._stages:
                graph_stage = cast(GraphPipelineStage, stage)
                graph_stage.state.clear()
            self.loss_kwargs = {}
            self._graph_pp_ready = False


def register_graph_pp_schedule(
    schedule: _PipelineScheduleRuntime,
    *,
    graph_provider: GraphPPStageGraphsProvider | None = None,
) -> GraphPPRunner:
    """Register GraphPP action handlers on a runtime PP schedule.

    Args:
        schedule (_PipelineScheduleRuntime): Runtime pipeline schedule whose
            compute actions should be handled by GraphPP.
        graph_provider (GraphPPStageGraphsProvider | None): Optional provider
            that builds or attaches stage graphs before the first runtime
            action in each step.

    Returns:
        GraphPPRunner: Runner that owns the registered bound action handlers.

    Raises:
        TypeError: If any local schedule stage is not a ``GraphPipelineStage``.
    """
    runner = GraphPPRunner(
        schedule,
        graph_provider=graph_provider,
    )
    for computation_type, handler in (
        (FORWARD, runner._handle_forward),
        (FULL_BACKWARD, runner._handle_full_backward),
        (UNSHARD, runner._handle_unshard),
        (RESHARD, runner._handle_reshard),
        (REDUCE_GRAD, runner._handle_reduce_grad),
        (BACKWARD_INPUT, runner._handle_backward_input),
        (BACKWARD_WEIGHT, runner._handle_backward_weight),
        (OVERLAP_F_B, runner._handle_overlap_fw_bw),
    ):
        schedule.register_custom_function(computation_type, handler)
    return runner
