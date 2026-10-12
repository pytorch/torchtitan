# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""SPMD with gradient accumulation: graph construction and executor.

Builds the per-microbatch variants (first / middle / last) of the joint graph
traced by ``spmd_graph_builder`` and executes them with explicit gradient
accumulators.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from copy import deepcopy
from typing import Any, cast, TYPE_CHECKING

import torch
import torch.fx as fx
import torch.nn as nn
import torch.utils._pytree as pytree
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.distributed import ParallelismContext
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.grad_accumulation import (
    insert_graph_gradient_accumulation_before_reduction,
    insert_graph_gradient_accumulation_from_outputs,
)
from torchtitan.experiments.graph_trainer.graph_builder_utils import (
    _apply_graph_pp_pre_partition_or_extraction_passes,
    _compile_graph_pp_module,
    _configure_fsdp_bucketing_pass,
    _execute_graph_module,
    _pack_graph_args,
)
from torchtitan.experiments.graph_trainer.graph_pp.runner import (
    _GraphComputationType,
    FORWARD_BACKWARD_FIRST_WITH_UNSHARD,
    FORWARD_BACKWARD_LAST_WITH_REDUCE_GRAD,
    FORWARD_BACKWARD_NOGRADACCUM,
)
from torchtitan.experiments.graph_trainer.graph_pp.split_fsdp_collectives import (
    extract_fsdp_reduce_grad_graph,
    extract_fsdp_unshard_graph,
    FSDPReduceGradExtraction,
    FSDPUnshardExtraction,
)
from torchtitan.experiments.graph_trainer.graph_pp.stage import (
    GraphPipelineStage,
    JointStageGraphs,
)
from torchtitan.experiments.graph_trainer.graph_pp.utils import (
    flatten_graph_values,
    graph_outputs,
    graph_pp_value_spec,
    GraphPPValueSpec,
    output_names,
    placeholder_names,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import TracedResult
from torchtitan.experiments.graph_trainer.passes import apply_graph_passes
from torchtitan.experiments.graph_trainer.spmd_graph_builder import (
    _trace_spmd_stage_graph,
)
from torchtitan.experiments.graph_trainer.wgrad_accumulation import (
    fuse_wgrad_accumulation_pass,
)


if TYPE_CHECKING:
    from torchtitan.experiments.graph_trainer.graph_builder import GraphExecutionPlan
    from torchtitan.experiments.graph_trainer.trainer import GraphTrainer


def _append_nodes_to_outputs(gm: fx.GraphModule, node_names: tuple[str, ...]) -> None:
    """Append existing graph nodes to its flat output tuple by name.

    Example::

        Input::
            graph outputs: (loss, grad)
            node_names: ("unsharded_param",)

        Output::
            graph outputs: (loss, grad, unsharded_param)
    """
    if not node_names:
        return
    nodes_by_name = {node.name: node for node in gm.graph.nodes}
    added_outputs = [nodes_by_name[name] for name in node_names]
    output_node = gm.graph.find_nodes(op="output")[0]
    output_node.args = ((*graph_outputs(gm.graph), *added_outputs),)
    gm.graph.lint()
    gm.recompile()


@dataclasses.dataclass(frozen=True, slots=True)
class _FwdBwdCallSpec:
    """FX module and runtime input metadata for one forward-backward action."""

    module: fx.GraphModule
    input_names: tuple[str, ...]
    flat_input_indices: tuple[int, ...]
    output_names: tuple[str, ...]
    num_param_inputs: int
    grad_accumulator_input_indices: tuple[int, ...] = ()


@dataclasses.dataclass(slots=True)
class _ScheduledFwdBwdGraphs:
    """FX modules backing SPMD with gradient accumulation actions."""

    # Repeated graph. With gradient accumulation:
    #     (..., grad_accumulators) -> loss, updated_grad_accumulators
    # Mutually exclusive MB0 graphs for gradient accumulation:
    #     fwd_bwd_nogradaccum(...) -> loss, initial_grad_accumulators
    #     fwd_bwd_with_unshard(...) -> loss, initial_grad_accumulators,
    #         unsharded_params
    # Last graph with deferred reduction:
    #     (..., grad_accumulators) -> loss, sharded_grads
    #
    # FORWARD_BACKWARD_FIRST_WITH_UNSHARD action
    # Calling convention:
    #     sharded parameters, buffers, inputs
    #     -> loss, unsharded gradients, unsharded parameters
    # Input names and indices pack flat runtime values in placeholder order.
    #
    # Repeated forward-backward action
    # FORWARD_BACKWARD_NOGRADACCUM omits accumulator inputs and returns its
    # gradients as initial accumulators. Repeated actions receive those tensors
    # at the trailing input indices below.
    # Used for every microbatch without a first- or last-microbatch override.
    # Its schedule action identifies which FSDP communication remains inside.
    # Calling convention:
    #     sharded or unsharded parameters, buffers, inputs, gradient accumulators
    #     -> loss, unsharded or reduced gradients
    # FULL_FORWARD_BACKWARD receives sharded parameters and returns reduced
    # gradients. FORWARD_BACKWARD receives the parameters unsharded by the
    # first microbatch and returns unsharded gradients.
    # Input names and indices pack non-parameter runtime values after the
    # leading parameter inputs.
    # These indices select accumulator inputs from the stage gradient state.
    #
    # FORWARD_BACKWARD_LAST_WITH_REDUCE_GRAD action
    # Calling convention:
    #     unsharded parameters, buffers, inputs, unsharded gradient accumulators
    #     -> loss, reduced sharded gradients
    # Input names and indices pack non-parameter runtime values after the
    # leading unsharded parameter inputs.
    # These indices select accumulator inputs from the stage gradient state.
    call_specs: dict[_GraphComputationType, _FwdBwdCallSpec]
    repeated_computation_type: _GraphComputationType


@dataclasses.dataclass(frozen=True, slots=True)
class _FwdBwdGraphsMeta:
    """Calling-convention metadata for SPMD with gradient accumulation graphs.

    ``flat`` means an ordered list of pytree leaves. It does not describe
    whether a parameter or gradient is FSDP-sharded.
    """

    # Shared metadata
    # num_sharded_param_values counts model parameter leaves before unsharding.
    # num_param_grad_values counts gradient leaves returned by a joint graph.
    # param_grad_values restores final sharded gradients to parameter structure.
    num_param_grad_values: int
    num_sharded_param_values: int
    param_grad_values: GraphPPValueSpec

    # num_unsharded_param_values identifies the trailing parameter outputs of
    # FORWARD_BACKWARD_FIRST_WITH_UNSHARD.
    num_unsharded_param_values: int

    # Names of the gradient-reduction inputs, in reduction order. They locate
    # where FORWARD_BACKWARD_LAST_WITH_REDUCE_GRAD accumulates before reducing.
    reduce_grad_input_names: tuple[str, ...]


@dataclasses.dataclass(slots=True)
class GraphTrainerScheduledFwdBwdStageGraphs(JointStageGraphs):
    """Execute SPMD with gradient accumulation graphs."""

    graphs: _ScheduledFwdBwdGraphs
    meta: _FwdBwdGraphsMeta
    runtime_meshes: tuple[DeviceMesh, ...] = ()

    def _grad_accumulator_args(
        self,
        grad_accumulators: list[Any] | None,
        input_indices: tuple[int, ...],
    ) -> list[torch.Tensor]:
        if not input_indices:
            return []
        if grad_accumulators is None:
            raise ValueError("Gradient accumulator inputs are missing")
        return [grad_accumulators[index] for index in input_indices]

    def unshard_params(
        self,
        sharded_param_values: list[Any],
        *,
        runtime_validate: bool = False,
    ) -> list[Any]:
        if (
            runtime_validate
            and len(sharded_param_values) != self.meta.num_sharded_param_values
        ):
            raise ValueError(
                "Scheduled forward-backward graph expected one runtime value "
                "per flat param: "
                f"{len(sharded_param_values)} != "
                f"{self.meta.num_sharded_param_values}"
            )
        return list(sharded_param_values)

    @staticmethod
    def _model_input(args: tuple[Any, ...]) -> Any:
        if len(args) != 1:
            raise ValueError(
                "SPMD joint forward/backward expects one model input, got "
                f"{len(args)}"
            )
        return args[0]

    def _full_args(
        self,
        call_spec: _FwdBwdCallSpec,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        target: Any,
        loss_kwargs: dict[str, Any],
        *,
        param_values: list[Any],
        buffer_values: list[Any],
        grad_accumulators: list[Any] | None,
        runtime_validate: bool,
    ) -> list[Any]:
        runtime_args = (
            self._model_input(args),
            target,
            loss_kwargs["global_loss_token_counts"],
            kwargs,
        )
        user_inputs, _ = pytree.tree_flatten((runtime_args, {}))
        full_args = _pack_graph_args(
            graph_name="Scheduled forward-backward graph",
            input_names=call_spec.input_names,
            flat_input_indices=call_spec.flat_input_indices,
            num_param_inputs=call_spec.num_param_inputs,
            num_sharded_param_values=self.meta.num_sharded_param_values,
            unshard_extracted=call_spec.num_param_inputs > 0,
            unsharded_param_values=param_values,
            flat_non_param_inputs=[
                *buffer_values,
                *self.runtime_meshes,
                *flatten_graph_values(list(user_inputs)),
            ],
            runtime_validate=runtime_validate,
        )
        full_args.extend(
            self._grad_accumulator_args(
                grad_accumulators,
                call_spec.grad_accumulator_input_indices,
            )
        )
        return full_args

    def _run_forward_backward(
        self,
        call_spec: _FwdBwdCallSpec,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        target: Any,
        loss_kwargs: dict[str, Any],
        *,
        unsharded_param_values: list[Any],
        buffer_values: list[Any],
        grad_accumulators: list[Any] | None,
        runtime_validate: bool,
    ) -> tuple[Any, list[Any]]:
        full_args = self._full_args(
            call_spec,
            args,
            kwargs,
            target,
            loss_kwargs,
            param_values=unsharded_param_values,
            buffer_values=buffer_values,
            grad_accumulators=grad_accumulators,
            runtime_validate=runtime_validate,
        )
        placeholders = call_spec.module.graph.find_nodes(op="placeholder")
        if runtime_validate and len(full_args) != len(placeholders):
            raise ValueError(
                "Scheduled forward-backward graph input mismatch: "
                f"expected {len(placeholders)} args, got {len(full_args)}"
            )
        outputs = _execute_graph_module(call_spec.module, full_args)
        expected_num_outputs = self.meta.num_param_grad_values + 1
        if runtime_validate and len(outputs) != expected_num_outputs:
            raise ValueError(
                "Scheduled forward-backward graph output count mismatch: "
                f"expected {expected_num_outputs}, got {len(outputs)}"
            )
        return outputs[0], list(outputs[1:])

    def forward_backward_nogradaccum(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        target: Any,
        loss_kwargs: dict[str, Any],
        *,
        unsharded_param_values: list[Any],
        buffer_values: list[Any],
        runtime_validate: bool = False,
    ) -> tuple[Any, list[Any]]:
        """Run the first microbatch and return its gradients as accumulators."""
        call_spec = self.graphs.call_specs[FORWARD_BACKWARD_NOGRADACCUM]
        return self._run_forward_backward(
            call_spec,
            args,
            kwargs,
            target,
            loss_kwargs,
            unsharded_param_values=unsharded_param_values,
            buffer_values=buffer_values,
            grad_accumulators=None,
            runtime_validate=runtime_validate,
        )

    def forward_backward(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        target: Any,
        loss_kwargs: dict[str, Any],
        *,
        unsharded_param_values: list[Any],
        buffer_values: list[Any],
        grad_accumulators: list[Any] | None = None,
        runtime_validate: bool = False,
    ) -> tuple[Any, list[Any]]:
        call_spec = self.graphs.call_specs[self.graphs.repeated_computation_type]
        return self._run_forward_backward(
            call_spec,
            args,
            kwargs,
            target,
            loss_kwargs,
            unsharded_param_values=unsharded_param_values,
            buffer_values=buffer_values,
            grad_accumulators=grad_accumulators,
            runtime_validate=runtime_validate,
        )

    def forward_backward_with_unshard(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        target: Any,
        loss_kwargs: dict[str, Any],
        *,
        sharded_param_values: list[Any],
        buffer_values: list[Any],
        runtime_validate: bool = False,
    ) -> tuple[Any, list[Any], list[Any]]:
        call_spec = self.graphs.call_specs[FORWARD_BACKWARD_FIRST_WITH_UNSHARD]
        full_args = self._full_args(
            call_spec,
            args,
            kwargs,
            target,
            loss_kwargs,
            param_values=sharded_param_values,
            buffer_values=buffer_values,
            grad_accumulators=None,
            runtime_validate=runtime_validate,
        )
        outputs = _execute_graph_module(call_spec.module, full_args)
        grad_end = 1 + self.meta.num_param_grad_values
        expected_num_outputs = grad_end + self.meta.num_unsharded_param_values
        if runtime_validate and len(outputs) != expected_num_outputs:
            raise ValueError(
                "First-microbatch joint graph output count mismatch: "
                f"expected {expected_num_outputs}, got {len(outputs)}"
            )
        return outputs[0], list(outputs[1:grad_end]), list(outputs[grad_end:])

    def forward_backward_with_reduce_grad(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        target: Any,
        loss_kwargs: dict[str, Any],
        *,
        unsharded_param_values: list[Any],
        buffer_values: list[Any],
        grad_accumulators: list[Any] | None = None,
        runtime_validate: bool = False,
    ) -> tuple[Any, list[Any]]:
        call_spec = self.graphs.call_specs[FORWARD_BACKWARD_LAST_WITH_REDUCE_GRAD]
        full_args = self._full_args(
            call_spec,
            args,
            kwargs,
            target,
            loss_kwargs,
            param_values=unsharded_param_values,
            buffer_values=buffer_values,
            grad_accumulators=grad_accumulators,
            runtime_validate=runtime_validate,
        )
        outputs = _execute_graph_module(call_spec.module, full_args)
        expected_num_outputs = self.meta.num_param_grad_values + 1
        if runtime_validate and len(outputs) != expected_num_outputs:
            raise ValueError(
                "Last-microbatch joint graph output count mismatch: "
                f"expected {expected_num_outputs}, got {len(outputs)}"
            )
        return outputs[0], list(outputs[1:])

    def reduce_grads(
        self,
        unsharded_param_grads: list[Any],
        *,
        runtime_validate: bool = False,
    ) -> list[Any]:
        if runtime_validate and len(unsharded_param_grads) != (
            self.meta.num_param_grad_values
        ):
            raise ValueError(
                "Scheduled forward-backward graph parameter grad count mismatch: "
                f"{len(unsharded_param_grads)} != "
                f"{self.meta.num_param_grad_values}"
            )
        return list(unsharded_param_grads)

    def param_grads_for_accumulation(
        self,
        param_grads: list[Any],
    ) -> list[Any]:
        return self.meta.param_grad_values.wrap_flat_values(param_grads)


def _apply_fsdp_action_overlap_scheduling(
    gm: fx.GraphModule,
    fsdp_bucketing_pass: Callable | None,
    *,
    compile_config: GraphTrainerCompileConfig,
    bucket_all_gathers: bool,
    bucket_reduce_scatters: bool,
    bucket_all_reduces: bool,
) -> fx.GraphModule:
    """Apply the deferred FSDP overlap pass to selected collectives."""
    if fsdp_bucketing_pass is None:
        return gm
    configured_bucketing_pass = _configure_fsdp_bucketing_pass(
        fsdp_bucketing_pass,
        bucket_all_gathers=bucket_all_gathers,
        bucket_reduce_scatters=bucket_reduce_scatters,
        bucket_all_reduces=bucket_all_reduces,
    )
    if configured_bucketing_pass is None:
        return gm
    return apply_graph_passes(
        gm,
        (),
        [configured_bucketing_pass],
        compile_config=compile_config,
    )


def _extract_fwd_bwd_action_graphs(
    stage: GraphPipelineStage,
    traced: TracedResult,
    *,
    trainer_config: "GraphTrainer.Config",
    parallelism_context: ParallelismContext,
    plan: "GraphExecutionPlan",
    num_param_grads: int,
) -> tuple[_ScheduledFwdBwdGraphs, _FwdBwdGraphsMeta, Callable | None,]:
    """Prepare SPMD with gradient accumulation actions from a joint graph
    with all FSDP communication.

    Gradient-accumulation calling conventions:

    - ``FORWARD_BACKWARD_NOGRADACCUM`` has no accumulator inputs and returns
      ``(loss, initial_grads)``. It has the same FSDP communication as the
      repeated graph.
    - The repeated graph receives ``initial_grads`` and returns
      ``(loss, updated_grads)`` after updating them in place.
    - ``FORWARD_BACKWARD_FIRST_WITH_UNSHARD`` returns
      ``(loss, initial_grads, unsharded_params)``.
    - ``FORWARD_BACKWARD_LAST_WITH_REDUCE_GRAD`` receives
      unsharded accumulators and returns ``(loss, sharded_grads)``.

    FSDP communication in the repeated graph:

    - ``FULL_FORWARD_BACKWARD``: unshard and reduce-grad
    - ``FORWARD_BACKWARD``: no FSDP communication

    ``fwd_bwd_repeat`` holds the action selected by this plan.

    Metadata maps flat inputs and outputs. The returned bucketing pass runs
    after extraction.
    """
    # Extraction of fsdp comms happens before bucketing.
    # fsdp bucketing reordering pass will be applied after extraction.
    fsdp_bucketing_pass: Callable | None = (
        _apply_graph_pp_pre_partition_or_extraction_passes(
            stage,
            traced,
            config=trainer_config,
            parallelism_context=parallelism_context,
            split_fsdp_param_unshard=plan.split_fsdp_param_unshard,
            split_fsdp_grad_reduction=plan.split_fsdp_grad_reduction,
        )
    )
    params: list[nn.Parameter] = [
        parameter
        for _, parameter in stage.submod.named_parameters(remove_duplicate=False)
    ]
    num_sharded_param_values: int = len(flatten_graph_values(params))
    param_grad_values: GraphPPValueSpec = graph_pp_value_spec(
        traced.output_subclass_layouts,
        start=1,
        count=num_param_grads,
    )
    num_param_grad_values: int = param_grad_values.num_flat_values

    reduce_grad_extraction: FSDPReduceGradExtraction = extract_fsdp_reduce_grad_graph(
        traced.gm,
        num_param_grads=num_param_grad_values,
        param_grad_output_start=1,
        mode="cut" if plan.split_fsdp_grad_reduction else "keep",
    )
    joint_input_names: tuple[str, ...] = placeholder_names(
        reduce_grad_extraction.compute_module
    )
    unshard_extraction: FSDPUnshardExtraction = extract_fsdp_unshard_graph(
        reduce_grad_extraction.compute_module,
        num_params=num_sharded_param_values,
        input_names=joint_input_names,
        flat_input_indices=tuple(range(len(joint_input_names))),
        mode="cut" if plan.split_fsdp_param_unshard else "keep",
    )

    repeated_computation_type = plan.repeated_computation_type
    fwd_bwd_repeat = unshard_extraction.compute_module
    call_specs: dict[_GraphComputationType, _FwdBwdCallSpec] = {
        repeated_computation_type: _FwdBwdCallSpec(
            module=fwd_bwd_repeat,
            input_names=unshard_extraction.compute_input_names,
            flat_input_indices=unshard_extraction.compute_flat_input_indices,
            output_names=output_names(fwd_bwd_repeat),
            num_param_inputs=unshard_extraction.num_compute_param_inputs,
        )
    }
    if plan.unshard_in_first_microbatch:
        fwd_bwd_with_unshard = reduce_grad_extraction.compute_module
        fwd_bwd_with_unshard_input_names = placeholder_names(fwd_bwd_with_unshard)
        _append_nodes_to_outputs(
            fwd_bwd_with_unshard,
            unshard_extraction.unshard_output_names,
        )
        call_specs[FORWARD_BACKWARD_FIRST_WITH_UNSHARD] = _FwdBwdCallSpec(
            module=fwd_bwd_with_unshard,
            input_names=fwd_bwd_with_unshard_input_names,
            flat_input_indices=tuple(range(len(fwd_bwd_with_unshard_input_names))),
            output_names=output_names(fwd_bwd_with_unshard),
            num_param_inputs=0,
        )

    if plan.reduce_grad_in_last_microbatch:
        last_input_names: tuple[str, ...] = placeholder_names(traced.gm)
        last_unshard_extraction: FSDPUnshardExtraction = extract_fsdp_unshard_graph(
            traced.gm,
            num_params=num_sharded_param_values,
            input_names=last_input_names,
            flat_input_indices=tuple(range(len(last_input_names))),
            mode="cut" if plan.split_fsdp_param_unshard else "keep",
        )
        fwd_bwd_with_reduce_grad = last_unshard_extraction.compute_module
        call_specs[FORWARD_BACKWARD_LAST_WITH_REDUCE_GRAD] = _FwdBwdCallSpec(
            module=fwd_bwd_with_reduce_grad,
            input_names=last_unshard_extraction.compute_input_names,
            flat_input_indices=last_unshard_extraction.compute_flat_input_indices,
            output_names=output_names(fwd_bwd_with_reduce_grad),
            num_param_inputs=last_unshard_extraction.num_compute_param_inputs,
        )

    graphs = _ScheduledFwdBwdGraphs(
        call_specs=call_specs,
        repeated_computation_type=repeated_computation_type,
    )
    meta: _FwdBwdGraphsMeta = _FwdBwdGraphsMeta(
        num_param_grad_values=num_param_grad_values,
        num_sharded_param_values=num_sharded_param_values,
        param_grad_values=param_grad_values,
        num_unsharded_param_values=len(unshard_extraction.unshard_output_names),
        reduce_grad_input_names=reduce_grad_extraction.reduce_grad_input_names,
    )
    return graphs, meta, fsdp_bucketing_pass


def _configure_scheduled_fwd_bwd_gradient_accumulation(
    stage: GraphPipelineStage,
    graphs: _ScheduledFwdBwdGraphs,
    meta: _FwdBwdGraphsMeta,
    *,
    plan: "GraphExecutionPlan",
) -> None:
    """Make first-microbatch outputs the accumulators for later microbatches."""
    if not plan.has_gradient_accumulation:
        return

    repeat_type = graphs.repeated_computation_type
    repeat_call = graphs.call_specs[repeat_type]
    if FORWARD_BACKWARD_FIRST_WITH_UNSHARD not in graphs.call_specs:
        graphs.call_specs[FORWARD_BACKWARD_NOGRADACCUM] = dataclasses.replace(
            repeat_call,
            module=deepcopy(repeat_call.module),
        )
    accumulator_values: tuple[
        Any, ...
    ] = insert_graph_gradient_accumulation_from_outputs(
        repeat_call.module,
        num_param_grads=meta.num_param_grad_values,
        param_grad_output_start=1,
        device=stage.device,
    )
    grad_accumulator_input_indices: tuple[int, ...] = tuple(
        dict.fromkeys(index for index in accumulator_values if index is not None)
    )
    graphs.call_specs[repeat_type] = dataclasses.replace(
        repeat_call,
        grad_accumulator_input_indices=grad_accumulator_input_indices,
    )

    reduce_grad_call = graphs.call_specs.get(FORWARD_BACKWARD_LAST_WITH_REDUCE_GRAD)
    if reduce_grad_call is not None:
        reduce_grad_accumulator_values: tuple[
            Any, ...
        ] = insert_graph_gradient_accumulation_before_reduction(
            reduce_grad_call.module,
            param_grad_output_names=repeat_call.output_names[
                1 : 1 + meta.num_param_grad_values
            ],
            reduce_grad_input_names=meta.reduce_grad_input_names,
            accumulators=accumulator_values,
            device=stage.device,
        )
        graphs.call_specs[FORWARD_BACKWARD_LAST_WITH_REDUCE_GRAD] = dataclasses.replace(
            reduce_grad_call,
            grad_accumulator_input_indices=tuple(
                cast(int, value) for value in reduce_grad_accumulator_values
            ),
        )

    if plan.fuse_wgrad_accumulation:
        fuse_wgrad_accumulation_pass(repeat_call.module)
        if reduce_grad_call is not None:
            fuse_wgrad_accumulation_pass(reduce_grad_call.module)


def _schedule_fwd_bwd_edge_fsdp_collectives(
    graphs: _ScheduledFwdBwdGraphs,
    fsdp_bucketing_pass: Callable | None,
    *,
    compile_config: GraphTrainerCompileConfig,
) -> _ScheduledFwdBwdGraphs:
    """Schedule the collective direction retained by each edge graph."""
    # FSDP extraction runs on unbucketed graphs. Each edge graph then schedules
    # only the collective direction that it retains.
    call_specs = dict(graphs.call_specs)
    first_call = call_specs.get(FORWARD_BACKWARD_FIRST_WITH_UNSHARD)
    if first_call is not None:
        call_specs[FORWARD_BACKWARD_FIRST_WITH_UNSHARD] = dataclasses.replace(
            first_call,
            module=_apply_fsdp_action_overlap_scheduling(
                first_call.module,
                fsdp_bucketing_pass,
                compile_config=compile_config,
                bucket_all_gathers=True,
                bucket_reduce_scatters=False,
                bucket_all_reduces=False,
            ),
        )

    last_call = call_specs.get(FORWARD_BACKWARD_LAST_WITH_REDUCE_GRAD)
    if last_call is not None:
        call_specs[FORWARD_BACKWARD_LAST_WITH_REDUCE_GRAD] = dataclasses.replace(
            last_call,
            module=_apply_fsdp_action_overlap_scheduling(
                last_call.module,
                fsdp_bucketing_pass,
                compile_config=compile_config,
                bucket_all_gathers=False,
                bucket_reduce_scatters=True,
                bucket_all_reduces=True,
            ),
        )

    return dataclasses.replace(graphs, call_specs=call_specs)


def _compile_scheduled_fwd_bwd_graphs(
    stage: GraphPipelineStage,
    graphs: _ScheduledFwdBwdGraphs,
    *,
    compile_config: GraphTrainerCompileConfig,
) -> _ScheduledFwdBwdGraphs:
    """Compile the FX module of each schedule action."""
    callable_names = {
        FORWARD_BACKWARD_NOGRADACCUM: "forward_backward_nogradaccum",
        FORWARD_BACKWARD_FIRST_WITH_UNSHARD: "forward_backward_with_unshard",
        FORWARD_BACKWARD_LAST_WITH_REDUCE_GRAD: "forward_backward_with_reduce_grad",
    }
    compiled_calls: dict[_GraphComputationType, _FwdBwdCallSpec] = {}
    for computation_type, call_spec in graphs.call_specs.items():
        callable_name = callable_names.get(
            computation_type,
            computation_type.value.lower(),
        )
        compiled_calls[computation_type] = dataclasses.replace(
            call_spec,
            module=_compile_graph_pp_module(
                call_spec.module,
                compile_config=compile_config,
                graph_name=f"stage_{stage.stage_index}_{callable_name}",
            ),
        )

    return _ScheduledFwdBwdGraphs(
        call_specs=compiled_calls,
        repeated_computation_type=graphs.repeated_computation_type,
    )


def _build_scheduled_fwd_bwd_graphs(
    stage: GraphPipelineStage,
    traced: TracedResult,
    *,
    trainer_config: "GraphTrainer.Config",
    parallelism_context: ParallelismContext,
    plan: "GraphExecutionPlan",
    num_param_grads: int,
) -> GraphTrainerScheduledFwdBwdStageGraphs:
    """Build the SPMD with gradient accumulation graph actions."""
    graphs: _ScheduledFwdBwdGraphs
    meta: _FwdBwdGraphsMeta
    fsdp_bucketing_pass: Callable | None
    graphs, meta, fsdp_bucketing_pass = _extract_fwd_bwd_action_graphs(
        stage,
        traced,
        trainer_config=trainer_config,
        parallelism_context=parallelism_context,
        plan=plan,
        num_param_grads=num_param_grads,
    )
    _configure_scheduled_fwd_bwd_gradient_accumulation(
        stage,
        graphs,
        meta,
        plan=plan,
    )
    graphs = _schedule_fwd_bwd_edge_fsdp_collectives(
        graphs,
        fsdp_bucketing_pass,
        compile_config=trainer_config.compile,
    )
    graphs = _compile_scheduled_fwd_bwd_graphs(
        stage,
        graphs,
        compile_config=trainer_config.compile,
    )
    return GraphTrainerScheduledFwdBwdStageGraphs(
        graphs=graphs,
        meta=meta,
    )


def _build_gradient_accumulation_fwd_bwd_graphs(
    stage: GraphPipelineStage,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    target: Any,
    loss_kwargs: dict[str, Any],
    *,
    loss_fn: Callable,
    trainer_config: "GraphTrainer.Config",
    parallelism_context: ParallelismContext,
    plan: "GraphExecutionPlan",
) -> None:
    """Build the SPMD with gradient accumulation graph executor."""
    traced, _, num_param_grads = _trace_spmd_stage_graph(
        stage,
        args,
        kwargs,
        target,
        loss_kwargs,
        loss_fn=loss_fn,
        compile_config=trainer_config.compile,
        parallelism_context=parallelism_context,
    )
    stage.graphs = _build_scheduled_fwd_bwd_graphs(
        stage,
        traced,
        trainer_config=trainer_config,
        parallelism_context=parallelism_context,
        plan=plan,
        num_param_grads=num_param_grads,
    )
