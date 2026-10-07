# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""PP stage graph construction, partitioning, and executors."""

from __future__ import annotations

import dataclasses
import functools
import logging
from copy import deepcopy
from collections.abc import Callable
from typing import Any, cast, TYPE_CHECKING

import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch.distributed.pipelining.schedules import _PipelineScheduleRuntime

from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_parameter_gradient,
    compute_annotated_loss,
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.fsdp_passes import (
    merge_all_all_gathers,
    merge_all_all_reduces,
    merge_all_reduce_scatters,
)
from torchtitan.experiments.graph_trainer.grad_accumulation import (
    graph_gradient_accumulator_input_indices,
    insert_graph_gradient_accumulation_from_outputs,
)
from torchtitan.experiments.graph_trainer.graph_builder_utils import (
    _apply_graph_pp_pre_partition_or_extraction_passes,
    _compile_graph_pp_module,
    _execute_graph_module,
    _pack_graph_args,
    GraphTrainerConfigView,
)
from torchtitan.experiments.graph_trainer.graph_pp import stage_builder
from torchtitan.experiments.graph_trainer.graph_pp.partition import (
    GraphMeta as PartitionGraphMeta,
    partition_joint_graph,
)
from torchtitan.experiments.graph_trainer.graph_pp.split_fsdp_collectives import (
    extract_fsdp_reduce_grad_graph,
    extract_fsdp_unshard_graph,
    remove_fsdp_reduction_tail,
)
from torchtitan.experiments.graph_trainer.graph_pp.stage import (
    GraphPipelineStage,
    OverlapStageGraphs,
    SplitStageGraphs,
)
from torchtitan.experiments.graph_trainer.graph_pp.utils import (
    flatten_graph_values,
    graph_pp_value_spec,
    GraphPPValueSpec,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    extract_module_state,
    minimal_fx_tracer,
    TracedResult,
)
from torchtitan.experiments.graph_trainer.passes import apply_graph_passes
if TYPE_CHECKING:
    from torchtitan.experiments.graph_trainer.trainer import GraphTrainer


logger = logging.getLogger(__name__)


def _operator_argument_index(op: Any, argument_name: str) -> int:
    """Return the unique named argument's position in an operator schema."""
    matches = [
        index
        for index, argument in enumerate(op._schema.arguments)
        if argument.name == argument_name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected {op} to have one {argument_name!r} argument, "
            f"found {len(matches)}"
        )
    return matches[0]


@functools.cache
def _dist_moe_forward_slot_arguments() -> tuple[tuple[Any, int], ...]:
    """Load exact Dist-MoE operator schemas only for Dist-MoE graph tracing."""
    from dist_moe._blockscaled import _block_scaled_forward_op
    from dist_moe.api import _bf16_forward_op, _bf16_forward_with_clip_stats_op

    return tuple(
        (op, _operator_argument_index(op, "activation_slot_id_1"))
        for op in (
            _bf16_forward_op._opoverload,
            _bf16_forward_with_clip_stats_op._opoverload,
            _block_scaled_forward_op._opoverload,
        )
    )


@dataclasses.dataclass(slots=True)
class _StageGraphModules:
    """FX graph modules produced by GraphTrainer stage graph construction."""

    fw: fx.GraphModule
    full_bw_repeat: fx.GraphModule
    full_bw_first: fx.GraphModule | None = None
    bw_di: fx.GraphModule | None = None
    bw_dw_repeat: fx.GraphModule | None = None
    bw_dw_first: fx.GraphModule | None = None
    unshard: fx.GraphModule | None = None
    reduce_grad: fx.GraphModule | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class _StageGraphMeta:
    """GraphTrainer metadata for one bound GraphPP stage graph executor.

    The fields describe how the flat FX graph signatures map to the PP calling
    convention:

    1. Forward: ``fwd_input_names`` and ``fwd_flat_input_indices`` select the
       parameter, buffer, user-input, target, and loss-kwarg leaves consumed by
       the forward graph.
    2. Saved values: ``num_saved_for_backward`` counts forward outputs that are
       hidden from PP users but fed to the backward graph.
    3. Backward: ``partition`` names saved values and output gradients from the
       next stage so runtime can pack backward placeholders deterministically.
    4. Grad outputs: ``param_grad_values`` and ``input_grad_values`` describe
       how flat graph outputs rewrap into parameter grads and input grads sent
       to the previous stage.
    5. FSDP edges: ``unshard_flat_param_indices`` and
       ``reduce_grad_input_names`` bind optional ``UNSHARD`` and ``REDUCE_GRAD``
       graphs to the same flat calling convention.

    Counts ending in ``_values`` refer to flat graph values, not privacy. This
    metadata is private to ``GraphTrainerStageGraphs``; the PP runtime must not
    inspect it.
    """

    num_user_outputs: int
    num_saved_for_backward: int
    num_param_grad_values: int
    num_input_grad_values: int
    num_sharded_param_values: int
    fwd_output_values: GraphPPValueSpec
    param_grad_values: GraphPPValueSpec
    input_grad_values: GraphPPValueSpec
    partition: PartitionGraphMeta
    fwd_input_names: tuple[str, ...]
    fwd_flat_input_indices: tuple[int, ...]
    uses_dist_moe_activation_slot: bool = False
    bw_no_fsdp_output_names: tuple[str, ...] = ()
    reduce_grad_input_names: tuple[str, ...] = ()
    unshard_flat_param_indices: tuple[int, ...] = ()
    num_fw_param_inputs: int = 0
    is_last_stage: bool = False


@dataclasses.dataclass(slots=True)
class GraphTrainerStageGraphs(SplitStageGraphs):
    """GraphTrainer-backed bound graph executor for one GraphPP stage.

    The object owns the GraphTrainer-specific FX modules and metadata needed to
    pack flat graph inputs and unwrap graph outputs. ``GraphRuntime`` calls
    this object through the generic ``SplitStageGraphs`` protocol and
    never inspects the private metadata directly.

    Args:
        modules: Stage-local FX graph modules after GraphPP graph passes.
        meta: GraphTrainer calling-convention metadata for those modules.
        compiled: Whether the FX modules have already been compiled.
    """

    modules: _StageGraphModules
    meta: _StageGraphMeta
    compiled: bool = False
    full_bw_grad_accumulator_indices: tuple[int, ...] = ()
    bw_dw_grad_accumulator_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        num_param_inputs = self.meta.num_fw_param_inputs
        if len(self.meta.fwd_input_names) != (
            num_param_inputs + len(self.meta.fwd_flat_input_indices)
        ):
            raise ValueError(
                "GraphPP forward input metadata must be a parameter-value "
                "prefix followed by traced flat input indices: "
                f"names={self.meta.fwd_input_names}, "
                f"num_params={num_param_inputs}, "
                f"flat_indices={self.meta.fwd_flat_input_indices}"
            )

    def _grad_accumulator_args(
        self,
        grad_accumulators: list[Any] | None,
        indices: tuple[int, ...],
    ) -> list[Any]:
        if not indices:
            return []
        if grad_accumulators is None:
            raise ValueError("Gradient accumulator inputs are missing")
        return [grad_accumulators[index] for index in indices]

    @property
    def supports_backward_input_weight_split(self) -> bool:
        return (
            self.modules.bw_di is not None
            and self.modules.bw_dw_repeat is not None
        )

    def unshard_params(
        self,
        sharded_param_values: list[Any],
        *,
        runtime_validate: bool = False,
    ) -> list[Any]:
        """Run the optional FSDP unshard graph.

        Calling convention:
            ``unshard(*selected_flat_params) -> (*forward_param_inputs)``

        ``sharded_param_values`` is the live stage parameter list flattened with
        the tracer's subclass rules. The unshard graph consumes only the flat
        parameters that own an all-gather chain and returns the parameter-derived
        values consumed by the forward graph. Replicated parameters and raw shards
        needed by backward rematerialization pass through unchanged.
        """

        if (
            runtime_validate
            and len(sharded_param_values) != self.meta.num_sharded_param_values
        ):
            raise ValueError(
                "GraphPP unshard expected one runtime value per flat param: "
                f"{len(sharded_param_values)} != "
                f"{self.meta.num_sharded_param_values}"
            )
        if self.modules.unshard is None:
            return list(sharded_param_values)
        unshard_args = []
        for param_index in self.meta.unshard_flat_param_indices:
            if runtime_validate and (
                param_index < 0 or param_index >= len(sharded_param_values)
            ):
                raise ValueError(
                    "GraphPP unshard parameter index is out of range: "
                    f"index {param_index}, but runtime has "
                    f"{len(sharded_param_values)} sharded params"
                )
            unshard_args.append(sharded_param_values[param_index])
        unsharded_param_values = list(
            _execute_graph_module(self.modules.unshard, unshard_args)
        )
        expected_num_outputs = self.meta.num_fw_param_inputs
        if runtime_validate and len(unsharded_param_values) != expected_num_outputs:
            raise ValueError(
                "GraphPP unshard graph output count must match its forward "
                "parameter input count: "
                f"{len(unsharded_param_values)} != {expected_num_outputs}"
            )
        return unsharded_param_values

    def _flat_user_forward_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        target: Any,
        loss_kwargs: dict[str, Any],
    ) -> list[Any]:
        """Flatten non-state forward inputs in trace order.

        Calling convention:
            First/last stage: ``(args, kwargs, target, loss_kwargs)``
            Other stages: ``(args, kwargs)``
        """

        if self.meta.is_last_stage:
            flat_user_inputs, _ = pytree.tree_flatten(
                ((args, kwargs, target, loss_kwargs), {})
            )
        else:
            flat_user_inputs, _ = pytree.tree_flatten(((args, kwargs), {}))
        return flatten_graph_values(list(flat_user_inputs))

    def _forward_args(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        target: Any,
        loss_kwargs: dict[str, Any],
        *,
        unsharded_param_values: list[Any],
        buffer_values: list[Any],
        activation_slot_id_1: torch.Tensor | None = None,
        runtime_validate: bool = False,
    ) -> list[Any]:
        """Pack forward parameter, state, and user inputs in placeholder order."""

        flat_user_inputs = self._flat_user_forward_inputs(
            args,
            kwargs,
            target,
            loss_kwargs,
        )
        if self.meta.uses_dist_moe_activation_slot:
            if activation_slot_id_1 is None:
                raise ValueError("GraphPP Dist-MoE forward requires an activation slot")
            runtime_values = [activation_slot_id_1]
        else:
            runtime_values = []
        # Calling convention:
        # (*forward_param_inputs, *selected_state_and_user_inputs,
        #  [activation_slot_id_1])
        return _pack_graph_args(
            graph_name="GraphPP forward",
            input_names=self.meta.fwd_input_names,
            flat_input_indices=self.meta.fwd_flat_input_indices,
            num_param_inputs=self.meta.num_fw_param_inputs,
            num_sharded_param_values=self.meta.num_sharded_param_values,
            unshard_extracted=self.modules.unshard is not None,
            unsharded_param_values=unsharded_param_values,
            flat_non_param_inputs=[
                *buffer_values,
                *flat_user_inputs,
                *runtime_values,
            ],
            runtime_validate=runtime_validate,
        )

    def _split_forward_outputs(
        self,
        fw_outputs: tuple[Any, ...],
    ) -> tuple[Any, tuple[Any, ...]]:
        """Extract runtime groups from forward graph outputs.

        Calling convention:
            ``(*user_outputs, *saved_for_backward, *side_effect_outputs)``
        Side-effect outputs only keep mutations live.
        """

        user_outputs = fw_outputs[: self.meta.num_user_outputs]
        saved_start = self.meta.num_user_outputs
        saved_end = saved_start + self.meta.num_saved_for_backward
        saved_values_for_backward = tuple(fw_outputs[saved_start:saved_end])
        output = self.meta.fwd_output_values.unflatten(user_outputs)
        return output, saved_values_for_backward

    def forward(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        target: Any,
        loss_kwargs: dict[str, Any],
        *,
        unsharded_param_values: list[Any],
        buffer_values: list[Any],
        activation_slot_id_1: torch.Tensor | None = None,
        runtime_validate: bool = False,
    ) -> tuple[Any, tuple[Any, ...]]:
        """Return ``(stage_output, saved_values_for_backward)``."""

        fw_args = self._forward_args(
            args,
            kwargs,
            target,
            loss_kwargs,
            unsharded_param_values=unsharded_param_values,
            buffer_values=buffer_values,
            activation_slot_id_1=activation_slot_id_1,
            runtime_validate=runtime_validate,
        )
        placeholders = self.modules.fw.graph.find_nodes(op="placeholder")
        if runtime_validate and len(fw_args) != len(placeholders):
            raise ValueError(
                "GraphPP forward graph input mismatch: "
                f"expected {len(placeholders)} args, got {len(fw_args)}. "
                f"Placeholders: {[node.name for node in placeholders]}. "
                f"Graph inputs: {list(self.meta.fwd_input_names)}."
            )
        return self._split_forward_outputs(
            _execute_graph_module(self.modules.fw, fw_args)
        )

    def _backward_args(
        self,
        saved_values_for_backward: tuple[Any, ...],
        output_grads_from_next: tuple[Any, ...],
        *,
        runtime_validate: bool = False,
    ) -> list[Any]:
        """Pack backward graph inputs in placeholder order.

        Calling convention:
            Each name in ``partition.bwd_input_names`` selects a saved value or
            a next-stage output gradient.
        """

        if runtime_validate and self.meta.is_last_stage and output_grads_from_next:
            raise ValueError(
                "GraphPP last stage backward must not receive "
                "output_grads_from_next."
            )
        raw_output_grads_from_next = flatten_graph_values(list(output_grads_from_next))
        # The partitioner names every backward placeholder. At runtime those
        # placeholders are supplied either by forward-saved values or by the
        # output gradients received from the next PP stage.
        saved_by_name = dict(
            zip(
                self.meta.partition.saved_for_backward_names,
                saved_values_for_backward,
                strict=True,
            )
        )
        backward_grad_by_name = dict(
            zip(
                self.meta.partition.backward_grad_input_names,
                [
                    raw_output_grads_from_next[index]
                    for index in self.meta.partition.backward_grad_input_indices
                ],
                strict=True,
            )
        )
        bwd_args = []
        for name in self.meta.partition.bwd_input_names:
            if name in saved_by_name:
                bwd_args.append(saved_by_name[name])
            elif name in backward_grad_by_name:
                bwd_args.append(backward_grad_by_name[name])
            else:
                raise ValueError(f"Missing GraphPP backward input {name}")
        return bwd_args

    def _split_full_backward_outputs(
        self,
        bw_outputs: tuple[Any, ...],
    ) -> tuple[list[Any], list[Any]]:
        """Convert flat backward outputs to runtime gradient groups.

        Calling convention:
            ``(*param_grads, *input_grads)``
            ``-> (input_grads, param_grads)``
        """

        num_param_grads = self.meta.num_param_grad_values
        param_grads = list(bw_outputs[:num_param_grads])
        input_grads = self.meta.input_grad_values.wrap_flat_values(
            bw_outputs[num_param_grads:]
        )
        return input_grads, param_grads

    def full_backward(
        self,
        saved_values_for_backward: tuple[Any, ...],
        output_grads_from_next: tuple[Any, ...],
        *,
        grad_accumulators: list[Any] | None = None,
        runtime_validate: bool = False,
    ) -> tuple[list[Any], list[Any]]:
        """Run the full backward graph.

        Calling convention:
            ``full_bw(*backward_inputs)``
            ``-> (*param_grads, *input_grads)``
        """

        first_graph = self.modules.full_bw_first
        accumulating = first_graph is not None and bool(grad_accumulators)
        graph = (
            self.modules.full_bw_repeat
            if accumulating or first_graph is None
            else first_graph
        )
        backward_args = self._backward_args(
            saved_values_for_backward,
            output_grads_from_next,
            runtime_validate=runtime_validate,
        )
        if accumulating:
            backward_args.extend(
                self._grad_accumulator_args(
                    grad_accumulators,
                    self.full_bw_grad_accumulator_indices,
                )
            )
        return self._split_full_backward_outputs(
            _execute_graph_module(graph, backward_args)
        )

    def backward_input(
        self,
        saved_values_for_backward: tuple[Any, ...],
        output_grads_from_next: tuple[Any, ...],
        *,
        runtime_validate: bool = False,
    ) -> tuple[list[Any], tuple[Any, ...]]:
        """Run the input-gradient graph.

        Calling convention:
            ``bw_di(*backward_inputs)``
            ``-> (*input_grads, *saved_for_weight_backward)``
        """

        if self.modules.bw_di is None:
            raise ValueError("GraphPP stage does not have a backward-input graph")
        outputs = _execute_graph_module(
            self.modules.bw_di,
            self._backward_args(
                saved_values_for_backward,
                output_grads_from_next,
                runtime_validate=runtime_validate,
            ),
        )
        input_grad_outputs = outputs[: self.meta.num_input_grad_values]
        saved_values_for_backward_weight = outputs[self.meta.num_input_grad_values :]
        input_grads = self.meta.input_grad_values.wrap_flat_values(input_grad_outputs)
        return input_grads, tuple(saved_values_for_backward_weight)

    def backward_weight(
        self,
        saved_values_for_backward_weight: tuple[Any, ...],
        *,
        grad_accumulators: list[Any] | None = None,
    ) -> list[Any]:
        """Run the weight-gradient graph.

        Calling convention:
            ``bw_dw(*saved_for_weight_backward)``
            ``-> (*param_grads)``
        """

        if self.modules.bw_dw_repeat is None:
            raise ValueError("GraphPP stage does not have a backward-weight graph")
        first_graph = self.modules.bw_dw_first
        accumulating = first_graph is not None and bool(grad_accumulators)
        graph = (
            self.modules.bw_dw_repeat
            if accumulating or first_graph is None
            else first_graph
        )
        backward_args = list(saved_values_for_backward_weight)
        if accumulating:
            backward_args.extend(
                self._grad_accumulator_args(
                    grad_accumulators,
                    self.bw_dw_grad_accumulator_indices,
                )
            )
        return list(
            _execute_graph_module(
                graph,
                backward_args,
            )
        )

    def _validate_unsharded_param_grads(
        self,
        unsharded_param_grads: list[Any],
        *,
        runtime_validate: bool = False,
    ) -> list[Any]:
        raw_grads = list(unsharded_param_grads)
        if runtime_validate and len(raw_grads) != self.meta.num_param_grad_values:
            raise ValueError(
                "GraphPP raw unsharded grad count mismatch: "
                f"expected {self.meta.num_param_grad_values}, got "
                f"{len(raw_grads)}"
            )
        return raw_grads

    def reduce_grads(
        self,
        unsharded_param_grads: list[Any],
        *,
        runtime_validate: bool = False,
    ) -> list[Any]:
        """Run the optional FSDP reduce-grad graph.

        Calling convention:
            ``reduce_grad(*selected_raw_param_grads)``
            ``-> (*reduced_param_grads)``
        """

        raw_grads = self._validate_unsharded_param_grads(
            unsharded_param_grads,
            runtime_validate=runtime_validate,
        )
        if self.modules.reduce_grad is None:
            return raw_grads
        # ``compute_module`` returns one raw grad slot per trainable parameter.
        # ``reduce_grad`` consumes the subset/name order selected by the FSDP
        # split pass and returns the original sharded/reduced grad slots.
        grad_values_by_name = dict(
            zip(
                self.meta.bw_no_fsdp_output_names[: self.meta.num_param_grad_values],
                raw_grads,
                strict=True,
            )
        )
        reduce_grad_args = [
            grad_values_by_name[name] for name in self.meta.reduce_grad_input_names
        ]
        return list(_execute_graph_module(self.modules.reduce_grad, reduce_grad_args))

    def param_grads_for_accumulation(
        self,
        param_grads: list[Any],
    ) -> list[Any]:
        """Return parameter gradients in optimizer accumulation structure."""

        return self.meta.param_grad_values.wrap_flat_values(param_grads)


def _requires_grad_like(value: Any) -> Any:
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        value = value.detach().requires_grad_(True)
    return value


def _grad_input_leaves(
    stage_args: tuple[Any, ...],
    stage_kwargs: dict[str, Any],
) -> list[torch.Tensor]:
    flat_inputs, _ = pytree.tree_flatten((stage_args, stage_kwargs))
    return [
        value
        for value in flat_inputs
        if isinstance(value, torch.Tensor) and value.requires_grad
    ]


def _compile_stage_graphs(
    stage: GraphPipelineStage,
    *,
    compile_config: GraphTrainerCompileConfig,
) -> None:
    """Compile the GraphTrainer graphs attached to ``stage`` once."""

    if stage.graphs is None:
        raise ValueError(
            "GraphPP cannot compile missing stage graphs for "
            f"stage {stage.stage_index}."
        )
    graphs = cast(GraphTrainerStageGraphs, stage.graphs)
    if graphs.compiled:
        return
    compiled_modules: dict[str, fx.GraphModule | None] = {}
    for name, gm in (
        ("fw", graphs.modules.fw),
        ("full_bw_repeat", graphs.modules.full_bw_repeat),
        ("full_bw_first", graphs.modules.full_bw_first),
        ("bw_di", graphs.modules.bw_di),
        ("bw_dw_repeat", graphs.modules.bw_dw_repeat),
        ("bw_dw_first", graphs.modules.bw_dw_first),
        ("unshard", graphs.modules.unshard),
        ("reduce_grad", graphs.modules.reduce_grad),
    ):
        compiled_modules[name] = (
            None
            if gm is None
            else _compile_graph_pp_module(
                gm,
                compile_config=compile_config,
                graph_name=f"stage_{stage.stage_index}_{name}",
            )
        )
    graphs.modules = _StageGraphModules(
        fw=cast(fx.GraphModule, compiled_modules["fw"]),
        full_bw_repeat=cast(fx.GraphModule, compiled_modules["full_bw_repeat"]),
        full_bw_first=compiled_modules["full_bw_first"],
        bw_di=compiled_modules["bw_di"],
        bw_dw_repeat=compiled_modules["bw_dw_repeat"],
        bw_dw_first=compiled_modules["bw_dw_first"],
        unshard=compiled_modules["unshard"],
        reduce_grad=compiled_modules["reduce_grad"],
    )
    graphs.compiled = True


def _split_stage_step_output_spec(
    traced: TracedResult,
    *,
    stage_index: int,
) -> tuple[pytree.TreeSpec, pytree.TreeSpec, pytree.TreeSpec]:
    """Split the traced ``(forward_output, param_grads, input_grads)`` spec."""

    output_spec = traced.output_spec
    if output_spec.num_children != 3:
        raise ValueError(
            "GraphPP stage traces must return exactly "
            "(forward_output, param_grads, input_grads). "
            f"Stage {stage_index} returned {output_spec.num_children} groups."
        )
    forward_output_spec = output_spec.child(0)
    param_grad_spec = output_spec.child(1)
    input_grad_spec = output_spec.child(2)
    return forward_output_spec, param_grad_spec, input_grad_spec


def _validate_stage_step_output_spec(
    *,
    stage_index: int,
    fwd_output_spec: pytree.TreeSpec,
    param_grad_spec: pytree.TreeSpec,
    input_grad_spec: pytree.TreeSpec,
    num_grad_params: int,
    num_input_grad_leaves: int,
) -> None:
    """Validate traced grouped outputs against the GraphPP calling convention."""

    if fwd_output_spec.num_leaves < 1:
        raise ValueError(
            "GraphPP stage trace must return at least one forward output leaf "
            f"for stage {stage_index}."
        )
    if param_grad_spec.num_leaves != num_grad_params:
        raise ValueError(
            "GraphPP traced param grad count does not match trainable params: "
            f"expected {num_grad_params}, got {param_grad_spec.num_leaves} "
            f"for stage {stage_index}."
        )
    if input_grad_spec.num_leaves != num_input_grad_leaves:
        raise ValueError(
            "GraphPP traced input grad count does not match differentiable "
            f"stage inputs: expected {num_input_grad_leaves}, got "
            f"{input_grad_spec.num_leaves} for stage {stage_index}."
        )


def _rewrite_dist_moe_activation_slot_input(
    traced: TracedResult,
    *,
    input_index: int,
) -> None:
    """Connect the explicit stage slot input to every Dist-MoE forward op."""
    placeholders = list(traced.gm.graph.find_nodes(op="placeholder"))
    if input_index >= len(placeholders):
        raise ValueError(
            "GraphPP Dist-MoE slot input index is out of range: "
            f"{input_index} >= {len(placeholders)}"
        )
    activation_slot_id_1 = placeholders[input_index]
    captured_slot_nodes: set[fx.Node] = set()
    for op, argument_index in _dist_moe_forward_slot_arguments():
        for node in traced.gm.graph.find_nodes(op="call_function", target=op):
            captured_slot = node.args[argument_index]
            if not isinstance(captured_slot, fx.Node) or captured_slot.op != "get_attr":
                raise ValueError(
                    f"{op} did not capture its activation slot as a graph attribute"
                )
            captured_slot_nodes.add(captured_slot)
    if not captured_slot_nodes:
        raise ValueError(
            "GraphPP received a Dist-MoE activation slot but traced no Dist-MoE "
            "forward operation"
        )
    for captured_slot in captured_slot_nodes:
        captured_slot.replace_all_uses_with(activation_slot_id_1)
        traced.gm.graph.erase_node(captured_slot)
    traced.gm.graph.lint()
    traced.gm.recompile()


def _build_stage_graphs(
    stage: GraphPipelineStage,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    target: Any,
    loss_kwargs: dict[str, Any],
    *,
    loss_fn: Callable | None = None,
    config: "GraphTrainer.Config | GraphTrainerConfigView",
    compile_graphs: bool = True,
    extract_fsdp_param_unshard: bool = True,
    extract_fsdp_grad_reduction: bool = True,
    gradient_accumulation: bool = False,
    activation_slot_id_1: torch.Tensor | None = None,
) -> None:
    """Trace one stage-local train step and attach bound GraphPP graphs."""
    compile_config: GraphTrainerCompileConfig = config.compile
    maybe_register_blockmask_pytree_node()

    # 1. Prepare representative trace inputs. ``minimal_fx_tracer`` fakeifies
    # these tensors before running the stage function, so this must not execute
    # the real eager stage forward outside tracing.
    stage_args = pytree.tree_map_only(torch.Tensor, _requires_grad_like, args)
    stage_kwargs = pytree.tree_map_only(torch.Tensor, _requires_grad_like, kwargs)

    state_params = [p for _, p in stage.submod.named_parameters(remove_duplicate=False)]
    state_buffers = [b for _, b in stage.submod.named_buffers(remove_duplicate=False)]
    grad_params = [p for p in state_params if p.requires_grad]
    num_state_param_values = len(flatten_graph_values(state_params))
    num_state_buffer_values = len(flatten_graph_values(state_buffers))
    num_grad_params = len(grad_params)
    num_input_grad_leaves = len(_grad_input_leaves(stage_args, stage_kwargs))

    # 2. Trace the stage functions.
    # Calling convention:
    #    Last stage:
    #      (stage_args, stage_kwargs, target, loss_kwargs)
    #      -> (loss, parameter_gradients, input_gradients)
    #    Other stages:
    #      (stage_args, stage_kwargs, output_grads_from_next)
    #      -> (forward_output, parameter_gradients, input_gradients)
    if stage.is_last:
        if loss_fn is None:
            raise ValueError(
                "GraphPP last-stage graph construction requires a loss function."
            )

        def stage_step(
            stage_args,
            stage_kwargs,
            target,
            loss_kwargs,
            activation_slot_id_1,
        ):
            pred = stage.submod(*stage_args, **stage_kwargs)
            loss = compute_annotated_loss(
                loss_fn,
                pred,
                target,
                loss_kwargs,
            )
            named_grad_params = [
                (name, parameter)
                for name, parameter in stage.submod.named_parameters(
                    remove_duplicate=False
                )
                if parameter.requires_grad
            ]
            grad_params = [parameter for _, parameter in named_grad_params]
            grad_inputs = [
                *grad_params,
                *_grad_input_leaves(stage_args, stage_kwargs),
            ]
            grads = torch.autograd.grad(
                loss,
                grad_inputs,
                allow_unused=True,
            )
            param_grads = tuple(
                None
                if grad is None
                else annotate_parameter_gradient(grad, parameter_fqn)
                for (parameter_fqn, _), grad in zip(
                    named_grad_params,
                    grads[: len(grad_params)],
                    strict=True,
                )
            )
            return (
                loss,
                param_grads,
                tuple(grads[len(grad_params) :]),
            )

        traced = minimal_fx_tracer(stage_step, module=stage.submod)(
            stage_args,
            stage_kwargs,
            target,
            loss_kwargs,
            activation_slot_id_1,
        )
        backward_only_indices = ()
    else:
        output_grads = stage_builder._flat_output_grads_from_stage_metadata(stage)

        def stage_step(
            stage_args,
            stage_kwargs,
            activation_slot_id_1,
            output_grads_from_next,
        ):
            output = stage.submod(*stage_args, **stage_kwargs)
            flat_outputs, _ = pytree.tree_flatten(output)
            flat_output_grads, _ = pytree.tree_flatten(output_grads_from_next)
            named_grad_params = [
                (name, parameter)
                for name, parameter in stage.submod.named_parameters(
                    remove_duplicate=False
                )
                if parameter.requires_grad
            ]
            grad_params = [parameter for _, parameter in named_grad_params]
            grad_inputs = [
                *grad_params,
                *_grad_input_leaves(stage_args, stage_kwargs),
            ]
            grads = torch.autograd.grad(
                flat_outputs,
                grad_inputs,
                grad_outputs=flat_output_grads,
                allow_unused=True,
            )
            param_grads = tuple(
                None
                if grad is None
                else annotate_parameter_gradient(grad, parameter_fqn)
                for (parameter_fqn, _), grad in zip(
                    named_grad_params,
                    grads[: len(grad_params)],
                    strict=True,
                )
            )
            return (
                output,
                param_grads,
                tuple(grads[len(grad_params) :]),
            )

        traced = minimal_fx_tracer(stage_step, module=stage.submod)(
            stage_args,
            stage_kwargs,
            activation_slot_id_1,
            output_grads,
        )
        state_flat, _ = pytree.tree_flatten(extract_module_state(stage.submod))
        prefix_user_flat, _ = pytree.tree_flatten(
            ((stage_args, stage_kwargs, activation_slot_id_1), {})
        )
        backward_only_start = len(
            flatten_graph_values([*state_flat, *prefix_user_flat])
        )
        backward_only_count = len(flatten_graph_values(list(output_grads)))
        backward_only_indices = tuple(
            range(backward_only_start, backward_only_start + backward_only_count)
        )

    if activation_slot_id_1 is not None:
        slot_input_index = (
            len(traced.example_inputs) - 1
            if stage.is_last
            else min(backward_only_indices) - 1
        )
        _rewrite_dist_moe_activation_slot_input(
            traced,
            input_index=slot_input_index,
        )

    # 3. Validate the grouped trace output before any graph extraction. The
    # partitioner depends on this exact grouping.
    fwd_output_spec, param_grad_spec, input_grad_spec = _split_stage_step_output_spec(
        traced,
        stage_index=stage.stage_index,
    )
    _validate_stage_step_output_spec(
        stage_index=stage.stage_index,
        fwd_output_spec=fwd_output_spec,
        param_grad_spec=param_grad_spec,
        input_grad_spec=input_grad_spec,
        num_grad_params=num_grad_params,
        num_input_grad_leaves=num_input_grad_leaves,
    )
    num_fwd_output_leaves = fwd_output_spec.num_leaves
    if not stage.is_last and len(output_grads) != num_fwd_output_leaves:
        raise ValueError(
            "GraphPP output grad metadata does not match traced stage output "
            f"structure: {len(output_grads)} metadata entries for "
            f"{num_fwd_output_leaves} output leaves"
        )
    # 4. Apply metadata-preserving GraphTrainer passes before partitioning.
    _apply_graph_pp_pre_partition_or_extraction_passes(
        stage,
        traced,
        config=config,
        split_fsdp_param_unshard=extract_fsdp_param_unshard,
        split_fsdp_grad_reduction=extract_fsdp_grad_reduction,
    )
    fwd_output_values = graph_pp_value_spec(
        traced.output_subclass_layouts,
        start=0,
        count=num_fwd_output_leaves,
        tree_spec=fwd_output_spec,
    )
    param_grad_values = graph_pp_value_spec(
        traced.output_subclass_layouts,
        start=num_fwd_output_leaves,
        count=num_grad_params,
    )
    input_grad_values = graph_pp_value_spec(
        traced.output_subclass_layouts,
        start=num_fwd_output_leaves + num_grad_params,
        count=num_input_grad_leaves,
    )
    num_fwd_output_values = fwd_output_values.num_flat_values
    num_param_grad_values = param_grad_values.num_flat_values
    num_input_grad_values = input_grad_values.num_flat_values
    # 5. Extract the runtime graph pieces in schedule order: stage
    # forward/backward, optional FSDP unshard/reduce-grad, optional dI/dW split.
    fw_module, bw_module, partition_meta = partition_joint_graph(
        traced,
        num_fwd_outputs=num_fwd_output_values,
        backward_only_input_indices=backward_only_indices,
    )
    fsdp_fw = extract_fsdp_unshard_graph(
        fw_module,
        num_params=num_state_param_values,
        input_names=partition_meta.fwd_input_names,
        flat_input_indices=partition_meta.fwd_flat_input_indices,
        side_effect_output_names=partition_meta.fwd_side_effect_output_names,
        mode="split" if extract_fsdp_param_unshard else "keep",
    )
    partition_meta = dataclasses.replace(
        partition_meta,
        fwd_side_effect_output_names=tuple(
            name
            for name in partition_meta.fwd_side_effect_output_names
            if name in fsdp_fw.compute_output_names
        ),
    )
    fsdp_bw = extract_fsdp_reduce_grad_graph(
        bw_module,
        num_param_grads=num_param_grad_values,
        mode="split" if extract_fsdp_grad_reduction else "keep",
    )
    remove_fsdp_reduction_tail(
        fsdp_fw.compute_module,
        reduction_node_names=fsdp_bw.reduction_node_names,
    )
    if fsdp_fw.unshard_module is not None:
        fsdp_fw = dataclasses.replace(
            fsdp_fw,
            unshard_module=apply_graph_passes(
                fsdp_fw.unshard_module,
                (),
                [merge_all_all_gathers],
                compile_config=compile_config,
            ),
        )
    if fsdp_bw.reduce_grad_module is not None:
        fsdp_bw = dataclasses.replace(
            fsdp_bw,
            reduce_grad_module=apply_graph_passes(
                fsdp_bw.reduce_grad_module,
                (),
                [merge_all_reduce_scatters, merge_all_all_reduces],
                compile_config=compile_config,
            ),
        )
    full_bw_repeat = fsdp_bw.compute_module
    didw_split = stage_builder._split_stage_backward_graph(
        full_bw_repeat,
        num_param_grads=num_param_grad_values,
        num_input_grads=num_input_grad_values,
    )
    full_bw_first = None
    full_bw_grad_accumulator_indices: tuple[int, ...] = ()
    if gradient_accumulation:
        full_bw_first = full_bw_repeat
        full_bw_repeat = deepcopy(full_bw_repeat)
        insert_graph_gradient_accumulation_from_outputs(
            full_bw_repeat,
            num_param_grads=num_param_grad_values,
            device=stage.device,
        )
        full_bw_grad_accumulator_indices = graph_gradient_accumulator_input_indices(
            full_bw_repeat
        )
    bw_dw_repeat = None if didw_split is None else didw_split.bw_dw_module
    bw_dw_first = None
    bw_dw_grad_accumulator_indices: tuple[int, ...] = ()
    if gradient_accumulation and bw_dw_repeat is not None:
        bw_dw_first = bw_dw_repeat
        bw_dw_repeat = deepcopy(bw_dw_repeat)
        insert_graph_gradient_accumulation_from_outputs(
            bw_dw_repeat,
            num_param_grads=num_param_grad_values,
            device=stage.device,
        )
        bw_dw_grad_accumulator_indices = graph_gradient_accumulator_input_indices(
            bw_dw_repeat
        )
    # 6. Attach the callable container and the GraphTrainer-only metadata used
    # to pack/unpack its flat graph inputs and outputs.
    graph_modules = _StageGraphModules(
        fw=fsdp_fw.compute_module,
        full_bw_repeat=full_bw_repeat,
        full_bw_first=full_bw_first,
        bw_di=None if didw_split is None else didw_split.bw_di_module,
        bw_dw_repeat=bw_dw_repeat,
        bw_dw_first=bw_dw_first,
        unshard=fsdp_fw.unshard_module,
        reduce_grad=fsdp_bw.reduce_grad_module,
    )
    graph_meta = _StageGraphMeta(
        num_user_outputs=partition_meta.num_fwd_user_outputs,
        num_saved_for_backward=partition_meta.num_saved_for_backward,
        num_param_grad_values=num_param_grad_values,
        num_input_grad_values=num_input_grad_values,
        num_sharded_param_values=num_state_param_values,
        fwd_output_values=fwd_output_values,
        param_grad_values=param_grad_values,
        input_grad_values=input_grad_values,
        partition=partition_meta,
        fwd_input_names=fsdp_fw.compute_input_names,
        fwd_flat_input_indices=fsdp_fw.compute_flat_input_indices,
        uses_dist_moe_activation_slot=activation_slot_id_1 is not None,
        bw_no_fsdp_output_names=fsdp_bw.compute_output_names,
        reduce_grad_input_names=fsdp_bw.reduce_grad_input_names,
        unshard_flat_param_indices=fsdp_fw.unshard_flat_param_indices,
        num_fw_param_inputs=fsdp_fw.num_compute_param_inputs,
        is_last_stage=stage.is_last,
    )
    stage.graphs = GraphTrainerStageGraphs(
        modules=graph_modules,
        meta=graph_meta,
        full_bw_grad_accumulator_indices=full_bw_grad_accumulator_indices,
        bw_dw_grad_accumulator_indices=bw_dw_grad_accumulator_indices,
    )
    logger.info(
        "GraphPP traced stage %s: fwd_outputs=%s saved=%s "
        "backward_grad_inputs=%s params=%s buffers=%s",
        stage.stage_index,
        graph_meta.num_user_outputs,
        graph_meta.num_saved_for_backward,
        partition_meta.num_backward_grad_inputs,
        num_state_param_values,
        num_state_buffer_values,
    )
    if compile_graphs:
        _compile_stage_graphs(stage, compile_config=compile_config)


def _build_graph_pp_overlap_graphs(
    schedule: _PipelineScheduleRuntime,
    *,
    compile_config: GraphTrainerCompileConfig,
) -> dict[tuple[int, int], OverlapStageGraphs]:
    return stage_builder._build_graph_pp_overlap_graphs(
        schedule,
        compile_config=compile_config,
        compile_graph_module=_compile_graph_pp_module,
        execute_graph_module=_execute_graph_module,
    )
