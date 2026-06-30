# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""GraphTrainer-backed graph construction for GraphPP stages.

Flat calling convention and wrapping contract:
1. Extracted graphs execute on the flat values produced by
   ``minimal_fx_tracer``. Tensor subclasses are unwrapped into plain leaves by
   the tracer before FX execution.
2. Only values that cross the PP/runtime boundary are rewrapped: stage forward
   outputs, input gradients sent to the previous stage, and parameter gradients
   before assigning to live ``param.grad``.
3. Internal graph values stay flat because they never escape GraphPP graph
   execution: saved-for-backward values, unsharded FSDP params, raw grad
   leaves, reduce-grad inputs, and multiplexed intermediate outputs.
4. DTensor and other traceable tensor subclasses use the existing tracer layout
   metadata. GraphPP must not add a separate DTensor-specific wrapping path.
"""

import dataclasses
import types
import warnings
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch.distributed.pipelining.schedules import (
    _PipelineContext,
    _PipelineScheduleRuntime,
    OVERLAP_F_B,
)

from torchtitan.experiments.graph_trainer.common_utils import (
    compute_annotated_loss,
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.graph_pp.split_fsdp_collectives import (
    split_backward_fsdp_collectives,
    split_forward_fsdp_collectives,
)
from torchtitan.experiments.graph_trainer.graph_pp.graph_multiplex import (
    multiplex_fw_bw_graph,
)
from torchtitan.experiments.graph_trainer.graph_pp.partition import (
    GraphMeta as PartitionGraphMeta,
    partition_joint_graph,
)
from torchtitan.experiments.graph_trainer.graph_pp.split_di_dw import (
    GraphPPDiDwSplit,
    split_di_dw_graph,
)
from torchtitan.experiments.graph_trainer.graph_pp.stage import (
    GraphPPOverlapGraphs,
    GraphPPStageGraphs,
    GraphPipelineStage,
)
from torchtitan.experiments.graph_trainer.graph_pp.utils import (
    example_inputs_from_placeholders,
    flatten_graph_values,
    GraphPPValueSpec,
    graph_pp_value_spec,
    normalize_graph_pp_microbatch_inputs,
    overlap_fw_bw_sub_actions,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    extract_module_state,
    minimal_fx_tracer,
    TracedResult,
)
from torchtitan.experiments.graph_trainer.passes import (
    apply_graph_passes,
    compile_time_passes,
    final_inductor_compile_passes,
)
from torchtitan.tools.logging import logger


@dataclasses.dataclass(frozen=True, slots=True)
class _GraphPPPassConfig:
    compile: GraphTrainerCompileConfig
    parallelism: Any
    model_spec: Any


@dataclasses.dataclass(slots=True)
class _StageGraphModules:
    """FX graph modules produced by GraphTrainer stage graph construction."""

    fw: fx.GraphModule
    full_bw: fx.GraphModule
    bw_di: fx.GraphModule | None = None
    bw_dw: fx.GraphModule | None = None
    unshard: fx.GraphModule | None = None
    reduce_grad: fx.GraphModule | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class _StageGraphMeta:
    """GraphTrainer metadata for one bound GraphPP stage graph executor.

    Counts ending in ``_values`` refer to flat tensor values in extracted FX
    graphs. This metadata is private to ``GraphTrainerStageGraphs``; the PP
    runner must not inspect it.
    """

    num_user_outputs: int
    num_saved_for_backward: int
    num_param_grad_values: int
    num_input_grad_values: int
    num_flat_param_values: int
    fwd_output_values: GraphPPValueSpec
    param_grad_values: GraphPPValueSpec
    input_grad_values: GraphPPValueSpec
    partition: PartitionGraphMeta
    fwd_input_names: tuple[str, ...]
    fwd_flat_input_indices: tuple[int, ...]
    bw_no_fsdp_output_names: tuple[str, ...] = ()
    reduce_grad_input_names: tuple[str, ...] = ()
    unshard_flat_param_indices: tuple[int, ...] = ()
    num_fw_unsharded_param_inputs: int = 0
    is_last_stage: bool = False


def _execute_graph_module(
    gm: fx.GraphModule,
    args: list[Any],
) -> tuple[Any, ...]:
    """Execute one FX graph module and normalize its result to a tuple."""

    with torch.no_grad():
        outputs = gm(*args)
    if isinstance(outputs, tuple):
        return outputs
    if isinstance(outputs, list):
        return tuple(outputs)
    return (outputs,)


@dataclasses.dataclass(slots=True)
class GraphTrainerStageGraphs(GraphPPStageGraphs):
    """GraphTrainer-backed bound graph executor for one GraphPP stage.

    The object owns the GraphTrainer-specific FX modules and metadata needed to
    pack flat graph inputs and unwrap graph outputs. ``GraphPPRunner`` calls
    this object through the generic ``GraphPPStageGraphs`` protocol and never
    inspects the private metadata directly.

    Args:
        modules: Stage-local FX graph modules after GraphPP graph passes.
        meta: GraphTrainer calling-convention metadata for those modules.
        compiled: Whether the FX modules have already been compiled.
    """

    modules: _StageGraphModules
    meta: _StageGraphMeta
    compiled: bool = False

    @property
    def supports_backward_input_weight_split(self) -> bool:
        return self.modules.bw_di is not None and self.modules.bw_dw is not None

    @property
    def num_unsharded_param_grad_values(self) -> int:
        return self.meta.num_param_grad_values

    def unshard_params(self, flat_param_values: list[Any]) -> list[Any]:
        """Run the optional FSDP unshard graph.

        ``flat_param_values`` is the live stage parameter list flattened with
        the tracer's subclass rules. The unshard graph consumes only the flat
        parameters that own an all-gather chain and returns one flat value for
        every original parameter input. Replicated parameters pass through
        unchanged so later forward calls can use a uniform parameter prefix.
        """

        if self.modules.unshard is None:
            return list(flat_param_values)
        unshard_args = []
        for param_index in self.meta.unshard_flat_param_indices:
            if param_index < 0 or param_index >= len(flat_param_values):
                raise ValueError(
                    "GraphPP unshard parameter index is out of range: "
                    f"index {param_index}, but runtime has "
                    f"{len(flat_param_values)} flat params"
                )
            unshard_args.append(flat_param_values[param_index])
        unsharded_param_values = list(
            _execute_graph_module(self.modules.unshard, unshard_args)
        )
        if len(unsharded_param_values) != self.meta.num_flat_param_values:
            raise ValueError(
                "GraphPP unshard graph output count must match flat parameter "
                "count: "
                f"{len(unsharded_param_values)} != {self.meta.num_flat_param_values}"
            )
        return unsharded_param_values

    def _flat_user_forward_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        target: Any,
        loss_kwargs: dict[str, Any],
    ) -> list[Any]:
        """Flatten the non-state runtime inputs supplied by PP for forward."""

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
        flat_buffer_values: list[Any],
    ) -> list[Any]:
        """Pack the extracted forward graph inputs in placeholder order."""

        num_unsharded_inputs = self.meta.num_fw_unsharded_param_inputs
        if len(unsharded_param_values) != self.meta.num_flat_param_values:
            raise ValueError(
                "GraphPP forward expected one unsharded value per flat param: "
                f"{len(unsharded_param_values)} != {self.meta.num_flat_param_values}"
            )
        if num_unsharded_inputs > len(unsharded_param_values):
            raise ValueError(
                "GraphPP forward graph needs more unsharded params than "
                f"runtime has: {num_unsharded_inputs} > "
                f"{len(unsharded_param_values)}"
            )
        if len(self.meta.fwd_input_names) != (
            num_unsharded_inputs + len(self.meta.fwd_flat_input_indices)
        ):
            raise ValueError(
                "GraphPP forward input metadata must be an unsharded-param "
                "prefix followed by traced flat input indices: "
                f"names={self.meta.fwd_input_names}, "
                f"num_unsharded={num_unsharded_inputs}, "
                f"flat_indices={self.meta.fwd_flat_input_indices}"
            )

        flat_user_inputs = self._flat_user_forward_inputs(
            args,
            kwargs,
            target,
            loss_kwargs,
        )
        flat_inputs = [
            *unsharded_param_values,
            *flat_buffer_values,
            *flat_user_inputs,
        ]
        # Forward placeholders are a prefix of unsharded parameter values
        # followed by explicit indices into params, buffers, and user inputs.
        fw_args = list(unsharded_param_values[:num_unsharded_inputs])
        flat_input_names = self.meta.fwd_input_names[num_unsharded_inputs:]
        for name, flat_index in zip(
            flat_input_names,
            self.meta.fwd_flat_input_indices,
            strict=True,
        ):
            if flat_index < 0 or flat_index >= len(flat_inputs):
                raise ValueError(
                    "GraphPP forward placeholder index is out of range: "
                    f"{name} indexes {flat_index}, but runtime has "
                    f"{len(flat_inputs)} flattened inputs"
                )
            fw_args.append(flat_inputs[flat_index])
        return fw_args

    def _split_forward_outputs(
        self,
        fw_outputs: tuple[Any, ...],
    ) -> tuple[Any, tuple[Any, ...]]:
        """Split user-visible outputs from saved values for backward."""

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
        flat_buffer_values: list[Any],
    ) -> tuple[Any, tuple[Any, ...]]:
        """Return ``(stage_output, saved_values_for_backward)``."""

        fw_args = self._forward_args(
            args,
            kwargs,
            target,
            loss_kwargs,
            unsharded_param_values=unsharded_param_values,
            flat_buffer_values=flat_buffer_values,
        )
        placeholders = self.modules.fw.graph.find_nodes(op="placeholder")
        if len(fw_args) != len(placeholders):
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
        stage_output: tuple[Any, ...],
        saved_values_for_backward: tuple[Any, ...],
        output_grads_from_next: tuple[Any, ...],
    ) -> list[Any]:
        """Pack backward graph inputs from saved tensors and next-stage grads."""

        if self.meta.is_last_stage:
            if len(stage_output) != 1:
                raise ValueError(
                    "GraphPP last stage backward expects the traced forward "
                    f"graph to return one loss tensor, got {len(stage_output)} "
                    "outputs."
                )
            if output_grads_from_next:
                raise ValueError(
                    "GraphPP last stage backward must not receive "
                    "output_grads_from_next."
                )
        raw_output_grads_from_next = flatten_graph_values(
            list(output_grads_from_next)
        )
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
        """Split ``full_bw`` outputs into input grads and param grads."""

        num_param_grads = self.meta.num_param_grad_values
        param_grads = list(bw_outputs[:num_param_grads])
        input_grads = self.meta.input_grad_values.wrap_flat_values(
            bw_outputs[num_param_grads:]
        )
        return input_grads, param_grads

    def full_backward(
        self,
        stage_output: tuple[Any, ...],
        saved_values_for_backward: tuple[Any, ...],
        output_grads_from_next: tuple[Any, ...],
    ) -> tuple[list[Any], list[Any]]:
        """Return ``(input_grads_to_prev, unsharded_param_grads)``."""

        return self._split_full_backward_outputs(
            _execute_graph_module(
                self.modules.full_bw,
                self._backward_args(
                    stage_output,
                    saved_values_for_backward,
                    output_grads_from_next,
                ),
            )
        )

    def backward_input(
        self,
        stage_output: tuple[Any, ...],
        saved_values_for_backward: tuple[Any, ...],
        output_grads_from_next: tuple[Any, ...],
    ) -> tuple[list[Any], tuple[Any, ...]]:
        """Return input grads and saved values for later weight backward."""

        if self.modules.bw_di is None:
            raise ValueError("GraphPP stage does not have a backward-input graph")
        outputs = _execute_graph_module(
            self.modules.bw_di,
            self._backward_args(
                stage_output,
                saved_values_for_backward,
                output_grads_from_next,
            ),
        )
        input_grad_outputs = outputs[: self.meta.num_input_grad_values]
        saved_values_for_backward_weight = outputs[self.meta.num_input_grad_values :]
        input_grads = self.meta.input_grad_values.wrap_flat_values(input_grad_outputs)
        return input_grads, tuple(saved_values_for_backward_weight)

    def backward_weight(
        self,
        saved_values_for_backward_weight: tuple[Any, ...],
    ) -> list[Any]:
        """Return flat unsharded parameter gradients."""

        if self.modules.bw_dw is None:
            raise ValueError("GraphPP stage does not have a backward-weight graph")
        return list(
            _execute_graph_module(
                self.modules.bw_dw,
                list(saved_values_for_backward_weight),
            )
        )

    def _validate_unsharded_param_grads(
        self,
        unsharded_param_grads: list[Any],
    ) -> list[Any]:
        raw_grads = list(unsharded_param_grads)
        if len(raw_grads) != self.meta.num_param_grad_values:
            raise ValueError(
                "GraphPP raw unsharded grad count mismatch: "
                f"expected {self.meta.num_param_grad_values}, got "
                f"{len(raw_grads)}"
            )
        return raw_grads

    def reduce_grads(
        self,
        unsharded_param_grads: list[Any],
    ) -> list[Any]:
        """Run the optional FSDP reduce-grad graph."""

        raw_grads = self._validate_unsharded_param_grads(unsharded_param_grads)
        if self.modules.reduce_grad is None:
            return raw_grads
        # ``bw_no_fsdp`` returns one raw grad slot per trainable parameter.
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
        sharded_param_grads: list[Any],
    ) -> list[Any]:
        """Return parameter gradients in optimizer accumulation structure."""

        return self.meta.param_grad_values.wrap_flat_values(sharded_param_grads)


@dataclasses.dataclass(slots=True)
class GraphTrainerOverlapGraphs(GraphPPOverlapGraphs):
    """GraphTrainer-backed executor for one multiplexed ``OVERLAP_F_B`` pair."""

    fw_graphs: GraphTrainerStageGraphs
    bw_graphs: GraphTrainerStageGraphs
    multiplexed_graph: fx.GraphModule

    def forward_backward(
        self,
        *,
        backward_stage_output: tuple[Any, ...],
        backward_saved_values_for_backward: tuple[Any, ...],
        output_grads_from_next: tuple[Any, ...],
        forward_args: tuple[Any, ...],
        forward_kwargs: dict[str, Any],
        forward_target: Any,
        forward_loss_kwargs: dict[str, Any],
        forward_unsharded_param_values: list[Any],
        forward_flat_buffer_values: list[Any],
    ) -> tuple[list[Any], list[Any], Any, tuple[Any, ...]]:
        """Run the multiplexed graph and split backward/forward outputs."""

        bw_args = self.bw_graphs._backward_args(
            backward_stage_output,
            backward_saved_values_for_backward,
            output_grads_from_next,
        )
        fw_args = self.fw_graphs._forward_args(
            forward_args,
            forward_kwargs,
            forward_target,
            forward_loss_kwargs,
            unsharded_param_values=forward_unsharded_param_values,
            flat_buffer_values=forward_flat_buffer_values,
        )
        multiplexed_outputs = _execute_graph_module(
            self.multiplexed_graph,
            [*bw_args, *fw_args],
        )
        num_bw_outputs = (
            self.bw_graphs.meta.num_param_grad_values
            + self.bw_graphs.meta.num_input_grad_values
        )
        input_grads, param_grads = self.bw_graphs._split_full_backward_outputs(
            multiplexed_outputs[:num_bw_outputs]
        )
        output, saved_values_for_backward = self.fw_graphs._split_forward_outputs(
            multiplexed_outputs[num_bw_outputs:]
        )
        return input_grads, param_grads, output, saved_values_for_backward


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


def _compile_graph_pp_module(
    gm: fx.GraphModule,
    *,
    compile_config: GraphTrainerCompileConfig,
    graph_name: str,
) -> fx.GraphModule:
    """Compile one extracted GraphPP callable with GraphTrainer Inductor passes."""
    if not compile_config.enable or not compile_config.enable_passes:
        return gm

    example_inputs = example_inputs_from_placeholders(gm)
    pass_config = _GraphPPPassConfig(
        compile=compile_config,
        parallelism=None,
        model_spec=types.SimpleNamespace(model=None),
    )
    gm = apply_graph_passes(
        gm,
        example_inputs,
        final_inductor_compile_passes(pass_config, use_cudagraph=False),
        compile_config=compile_config,
    )
    logger.info(
        "GraphPP compiled %s with %s inductor",
        graph_name,
        compile_config.inductor_compilation,
    )
    return gm


def _compile_graph_pp_modules(
    modules: _StageGraphModules,
    *,
    compile_config: GraphTrainerCompileConfig,
    stage_index: int,
) -> _StageGraphModules:
    """Compile the independent FX graph modules for one stage."""

    def compile_optional(
        gm: fx.GraphModule | None,
        name: str,
    ) -> fx.GraphModule | None:
        if gm is None:
            return None
        return _compile_graph_pp_module(
            gm,
            compile_config=compile_config,
            graph_name=f"stage_{stage_index}_{name}",
        )

    return _StageGraphModules(
        fw=_compile_graph_pp_module(
            modules.fw,
            compile_config=compile_config,
            graph_name=f"stage_{stage_index}_fw",
        ),
        full_bw=_compile_graph_pp_module(
            modules.full_bw,
            compile_config=compile_config,
            graph_name=f"stage_{stage_index}_full_bw",
        ),
        bw_di=compile_optional(modules.bw_di, "bw_di"),
        bw_dw=compile_optional(modules.bw_dw, "bw_dw"),
        unshard=compile_optional(modules.unshard, "unshard"),
        reduce_grad=compile_optional(modules.reduce_grad, "reduce_grad"),
    )


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
    if not compile_config.enable or not compile_config.enable_passes:
        return
    graphs.modules = _compile_graph_pp_modules(
        graphs.modules,
        compile_config=compile_config,
        stage_index=stage.stage_index,
    )
    graphs.compiled = True


def _annotate_graph_pp_graph(
    gm: fx.GraphModule,
    *,
    stage_index: int,
    callable_name: str,
    action_name: str,
) -> None:
    for node in gm.graph.nodes:
        node.meta = dict(node.meta)
        node.meta["graph_pp_stage_index"] = stage_index
        node.meta["graph_pp_callable"] = callable_name
        node.meta["graph_pp_action"] = action_name
        if node.op == "placeholder":
            node.meta["graph_pp_slot"] = f"input:{node.name}"
        elif node.op == "output":
            node.meta["graph_pp_slot"] = "output"


def _annotate_graph_pp_modules(
    modules: _StageGraphModules,
    *,
    stage_index: int,
) -> None:
    graph_specs = (
        (modules.fw, "fw", "FORWARD"),
        (modules.full_bw, "full_bw", "FULL_BACKWARD"),
        (modules.bw_di, "bw_di", "BACKWARD_INPUT"),
        (modules.bw_dw, "bw_dw", "BACKWARD_WEIGHT"),
        (modules.unshard, "unshard", "UNSHARD"),
        (modules.reduce_grad, "reduce_grad", "REDUCE_GRAD"),
    )
    for gm, callable_name, action_name in graph_specs:
        if gm is not None:
            _annotate_graph_pp_graph(
                gm,
                stage_index=stage_index,
                callable_name=callable_name,
                action_name=action_name,
            )


def _apply_graph_pp_pre_partition_passes(
    stage: GraphPipelineStage,
    traced: TracedResult,
    *,
    compile_config: GraphTrainerCompileConfig,
    model_config: Any,
    parallelism: Any,
) -> None:
    """Apply metadata-preserving GraphTrainer passes before GraphPP partition."""
    if not compile_config.enable_passes:
        return
    if model_config is None or parallelism is None:
        raise ValueError(
            "GraphPP requires model_config and parallelism when compile passes "
            "are enabled before stage graph partitioning."
        )

    passes = compile_time_passes(
        traced,
        _GraphPPPassConfig(
            compile=compile_config,
            parallelism=parallelism,
            model_spec=types.SimpleNamespace(model=model_config),
        ),
        use_cudagraph=False,
        include_inductor=False,
    )
    traced.gm = apply_graph_passes(
        traced.gm,
        traced.example_inputs,
        passes,
        compile_config=compile_config,
    )


def _flat_output_grads_from_stage_metadata(
    stage: GraphPipelineStage,
) -> tuple[Any, ...]:
    """Return flat non-last output grads supplied by upstream PP metadata."""

    stage_meta = stage._stage_meta
    output_grad_metas = stage_meta.output_grads
    if output_grad_metas is None:
        raise ValueError(
            "GraphPP requires upstream PP backward metadata before tracing a "
            f"non-last stage. Missing output_grads for stage {stage.stage_index}."
        )
    output_metas = stage_meta.outputs
    if output_metas is not None and len(output_grad_metas) != len(output_metas):
        raise ValueError(
            "GraphPP output grad metadata does not match stage output structure: "
            f"{len(output_grad_metas)} metadata entries for "
            f"{len(output_metas)} output leaves"
        )
    return tuple(
        None if meta is None else stage._to_tensor(meta) for meta in output_grad_metas
    )


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


def _build_stage_graphs(
    stage: GraphPipelineStage,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    target: Any,
    loss_kwargs: dict[str, Any],
    *,
    loss_fn: Callable | None = None,
    compile_config: GraphTrainerCompileConfig,
    model_config: Any = None,
    parallelism: Any = None,
    compile_graphs: bool = True,
) -> None:
    """Trace one stage-local train step and attach bound GraphPP graphs."""
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

    # 2. Trace the stage calling convention. Last stages differentiate a scalar
    # loss. Non-last stages differentiate stage outputs against runtime
    # ``output_grads_from_next`` metadata supplied by upstream PP.
    if stage.is_last:
        if loss_fn is None:
            raise ValueError(
                "GraphPP last-stage graph construction requires a loss function."
            )

        def stage_step(stage_args, stage_kwargs, target, loss_kwargs):
            pred = stage.submod(*stage_args, **stage_kwargs)
            loss = compute_annotated_loss(
                loss_fn,
                pred,
                target,
                loss_kwargs,
            )
            grad_params = [
                p
                for _, p in stage.submod.named_parameters(remove_duplicate=False)
                if p.requires_grad
            ]
            grad_inputs = [
                *grad_params,
                *_grad_input_leaves(stage_args, stage_kwargs),
            ]
            grads = torch.autograd.grad(
                loss,
                grad_inputs,
                allow_unused=True,
            )
            return (
                loss,
                tuple(grads[: len(grad_params)]),
                tuple(grads[len(grad_params) :]),
            )

        traced = minimal_fx_tracer(stage_step, module=stage.submod)(
            stage_args,
            stage_kwargs,
            target,
            loss_kwargs,
        )
        backward_only_indices = ()
    else:
        output_grads = _flat_output_grads_from_stage_metadata(stage)

        def stage_step(stage_args, stage_kwargs, output_grads_from_next):
            output = stage.submod(*stage_args, **stage_kwargs)
            flat_outputs, _ = pytree.tree_flatten(output)
            flat_output_grads, _ = pytree.tree_flatten(output_grads_from_next)
            grad_params = [
                p
                for _, p in stage.submod.named_parameters(remove_duplicate=False)
                if p.requires_grad
            ]
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
            return (
                output,
                tuple(grads[: len(grad_params)]),
                tuple(grads[len(grad_params) :]),
            )

        traced = minimal_fx_tracer(stage_step, module=stage.submod)(
            stage_args,
            stage_kwargs,
            output_grads,
        )
        state_flat, _ = pytree.tree_flatten(extract_module_state(stage.submod))
        prefix_user_flat, _ = pytree.tree_flatten(((stage_args, stage_kwargs), {}))
        backward_only_start = len(
            flatten_graph_values([*state_flat, *prefix_user_flat])
        )
        backward_only_count = len(flatten_graph_values(list(output_grads)))
        backward_only_indices = tuple(
            range(backward_only_start, backward_only_start + backward_only_count)
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
    _apply_graph_pp_pre_partition_passes(
        stage,
        traced,
        compile_config=compile_config,
        model_config=model_config,
        parallelism=parallelism,
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
    fsdp_fw = split_forward_fsdp_collectives(
        fw_module,
        num_params=num_state_param_values,
        fwd_input_names=partition_meta.fwd_input_names,
        fwd_flat_input_indices=partition_meta.fwd_flat_input_indices,
    )
    fsdp_bw = split_backward_fsdp_collectives(
        bw_module,
        num_param_grads=num_param_grad_values,
    )
    didw_split: GraphPPDiDwSplit | None = split_di_dw_graph(
        fsdp_bw.bw_no_fsdp_module,
        num_param_grads=num_param_grad_values,
    )
    if didw_split is not None and didw_split.num_input_grads != num_input_grad_values:
        raise ValueError(
            "GraphPP dI/dW split changed the raw input-gradient count: "
            f"expected {num_input_grad_values}, got {didw_split.num_input_grads}"
        )
    # 6. Attach the callable container and the GraphTrainer-only metadata used
    # to pack/unpack its flat graph inputs and outputs.
    graph_modules = _StageGraphModules(
        fw=fsdp_fw.fw_no_fsdp_module,
        full_bw=fsdp_bw.bw_no_fsdp_module,
        bw_di=None if didw_split is None else didw_split.bw_di_module,
        bw_dw=None if didw_split is None else didw_split.bw_dw_module,
        unshard=fsdp_fw.unshard_module,
        reduce_grad=fsdp_bw.reduce_grad_module,
    )
    _annotate_graph_pp_modules(
        graph_modules,
        stage_index=stage.stage_index,
    )
    graph_meta = _StageGraphMeta(
        num_user_outputs=partition_meta.num_fwd_user_outputs,
        num_saved_for_backward=partition_meta.num_saved_for_backward,
        num_param_grad_values=num_param_grad_values,
        num_input_grad_values=num_input_grad_values,
        num_flat_param_values=num_state_param_values,
        fwd_output_values=fwd_output_values,
        param_grad_values=param_grad_values,
        input_grad_values=input_grad_values,
        partition=partition_meta,
        fwd_input_names=fsdp_fw.fw_no_fsdp_input_names,
        fwd_flat_input_indices=fsdp_fw.fw_no_fsdp_flat_input_indices,
        bw_no_fsdp_output_names=fsdp_bw.bw_no_fsdp_output_names,
        reduce_grad_input_names=fsdp_bw.reduce_grad_input_names,
        unshard_flat_param_indices=fsdp_fw.unshard_flat_param_indices,
        num_fw_unsharded_param_inputs=fsdp_fw.num_fw_unsharded_param_inputs,
        is_last_stage=stage.is_last,
    )
    stage.graphs = GraphTrainerStageGraphs(
        modules=graph_modules,
        meta=graph_meta,
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


def _required_multiplex_pairs(
    schedule: _PipelineScheduleRuntime,
) -> set[tuple[int, int]]:
    try:
        pipeline_order = schedule.pipeline_order_with_comms
    except AttributeError as exc:
        raise ValueError(
            "GraphPP overlap graph construction requires a runtime PP schedule "
            "with pipeline_order_with_comms."
        ) from exc
    required_pairs: set[tuple[int, int]] = set()
    for action in pipeline_order.get(schedule.rank, []):
        if action.computation_type != OVERLAP_F_B:
            continue
        fw_action, bw_action = overlap_fw_bw_sub_actions(action)
        required_pairs.add((fw_action.stage_index, bw_action.stage_index))
    return required_pairs


def _require_graph_trainer_stage_graphs(
    stage: GraphPipelineStage,
    action_name: str,
) -> GraphTrainerStageGraphs:
    if stage.graphs is None:
        raise ValueError(
            "GraphPP stage graphs must be built before "
            f"{action_name}: stage {stage.stage_index}."
        )
    return cast(GraphTrainerStageGraphs, stage.graphs)


def _build_graph_pp_overlap_graphs(
    schedule: _PipelineScheduleRuntime,
    *,
    compile_config: GraphTrainerCompileConfig,
) -> dict[tuple[int, int], GraphPPOverlapGraphs]:
    """Build multiplexed graphs required by ``OVERLAP_F_B`` schedule actions."""

    stage_index_to_stage = {
        stage.stage_index: cast(GraphPipelineStage, stage) for stage in schedule._stages
    }
    overlap_graphs: dict[tuple[int, int], GraphPPOverlapGraphs] = {}
    for fw_stage_idx, bw_stage_idx in _required_multiplex_pairs(schedule):
        pair = (fw_stage_idx, bw_stage_idx)
        fw_stage = stage_index_to_stage[fw_stage_idx]
        bw_stage = stage_index_to_stage[bw_stage_idx]
        fw_graphs = _require_graph_trainer_stage_graphs(fw_stage, "OVERLAP_F_B")
        bw_graphs = _require_graph_trainer_stage_graphs(bw_stage, "OVERLAP_F_B")
        if fw_graphs.compiled or bw_graphs.compiled:
            raise ValueError(
                "GraphPP overlap graphs must be built before stage graphs are compiled."
            )
        multiplexed_graph = multiplex_fw_bw_graph(
            fw_graphs.modules.fw,
            bw_graphs.modules.full_bw,
        )
        _annotate_graph_pp_graph(
            multiplexed_graph,
            stage_index=fw_stage_idx,
            callable_name="multiplex",
            action_name="OVERLAP_F_B",
        )
        compiled_graph = _compile_graph_pp_module(
            multiplexed_graph,
            compile_config=compile_config,
            graph_name=f"stage_{fw_stage_idx}_fw_stage_{bw_stage_idx}_bw_multiplex",
        )
        overlap_graphs[pair] = GraphTrainerOverlapGraphs(
            fw_graphs=fw_graphs,
            bw_graphs=bw_graphs,
            multiplexed_graph=compiled_graph,
        )
    return overlap_graphs


def _example_args_from_stage_metadata(stage: GraphPipelineStage) -> tuple[Any, ...]:
    if stage._stage_meta.inputs is None:
        raise ValueError(
            "GraphPP stage metadata was not initialized before graph construction: "
            f"stage {stage.stage_index}"
        )
    return tuple(stage._to_tensor(meta) for meta in stage._stage_meta.inputs)


def _trace_args_for_stage(
    stage: GraphPipelineStage,
    ctx: _PipelineContext,
) -> tuple[Any, ...]:
    """Return representative positional inputs for a stage trace.

    Match upstream PP metadata/runtime semantics: first-stage positional inputs
    are true user microbatch args, while non-first positional inputs are
    received activations represented by PP stage metadata. ``minimal_fx_tracer``
    fakeifies tensor inputs before executing the traced function.
    """
    if stage.is_first and ctx.arg_mbs is not None:
        return tuple(ctx.arg_mbs[0])
    return _example_args_from_stage_metadata(stage)


def _trace_kwargs_from_context(ctx: _PipelineContext) -> dict[str, Any]:
    if ctx.kwarg_mbs is None:
        return {}
    return ctx.kwarg_mbs[0]


def _trace_target_from_context(
    stage: GraphPipelineStage,
    ctx: _PipelineContext,
) -> Any:
    if not stage.is_last or ctx.target_mbs is None:
        return None
    return ctx.target_mbs[0]


@dataclasses.dataclass(slots=True)
class GraphTrainerStageGraphProvider:
    """Build bound GraphPP stage graphs with GraphTrainer tracing and passes.

    Args:
        loss_fn: Loss function used to trace last-stage loss and backward.
        compile_config: GraphTrainer compile configuration.
        model_config: Model config consumed by GraphTrainer compile passes.
        parallelism: Parallelism config consumed by GraphTrainer compile passes.
    """

    loss_fn: Callable
    compile_config: GraphTrainerCompileConfig
    model_config: Any
    parallelism: Any
    _warned_cudagraph: bool = False
    _overlap_graphs: dict[tuple[int, int], GraphPPOverlapGraphs] | None = None

    def _maybe_warn_cudagraph_excluded(self) -> None:
        if self._warned_cudagraph:
            return
        if not self.compile_config.enable or not self.compile_config.enable_passes:
            return
        if "cudagraph_pass" in self.compile_config.disable_passes:
            return
        warnings.warn(
            "GraphPP currently skips cudagraph_pass for extracted stage "
            "modules; CUDA graph capture needs a separate GraphPP runtime "
            "integration.",
            stacklevel=3,
        )
        self._warned_cudagraph = True

    def _ensure_overlap_graphs(
        self,
        schedule: _PipelineScheduleRuntime,
    ) -> dict[tuple[int, int], GraphPPOverlapGraphs]:
        """Return cached multiplexed graphs required by overlap actions."""

        required_pairs = _required_multiplex_pairs(schedule)
        if not required_pairs:
            self._overlap_graphs = {}
            return {}
        if self._overlap_graphs is not None:
            missing_pairs = required_pairs - set(self._overlap_graphs)
            if missing_pairs:
                raise ValueError(
                    "GraphPP cached overlap graphs do not cover current "
                    f"schedule pairs: missing {sorted(missing_pairs)}."
                )
            return {pair: self._overlap_graphs[pair] for pair in required_pairs}
        self._overlap_graphs = _build_graph_pp_overlap_graphs(
            schedule,
            compile_config=self.compile_config,
        )
        return dict(self._overlap_graphs)

    def ensure_stage_graphs(
        self,
        schedule: _PipelineScheduleRuntime,
        ctx: _PipelineContext,
        *,
        loss_kwargs: dict[str, Any],
    ) -> dict[tuple[int, int], GraphPPOverlapGraphs]:
        self._maybe_warn_cudagraph_excluded()
        graph_stages = [
            cast(GraphPipelineStage, stage) for stage in schedule._stages
        ]
        maybe_register_blockmask_pytree_node()
        trace_ctx = ctx
        if ctx.arg_mbs is not None and ctx.kwarg_mbs is not None:
            # Upstream PP creates a fresh _PipelineContext for each action, but
            # all contexts share the same arg_mbs/kwarg_mbs lists. Mutate those
            # lists in place so later actions see graphable BlockMask objects.
            # Use distinct objects for tracing so make_fx does not consume the
            # same closure tensor objects that runtime replay will receive.
            trace_arg_mbs, trace_kwarg_mbs = normalize_graph_pp_microbatch_inputs(
                ctx.arg_mbs,
                ctx.kwarg_mbs,
            )
            runtime_arg_mbs, runtime_kwarg_mbs = normalize_graph_pp_microbatch_inputs(
                ctx.arg_mbs,
                ctx.kwarg_mbs,
            )
            ctx.arg_mbs[:] = runtime_arg_mbs
            ctx.kwarg_mbs[:] = runtime_kwarg_mbs
            trace_ctx = _PipelineContext(
                schedule,
                trace_arg_mbs,
                trace_kwarg_mbs,
                ctx.target_mbs,
                ctx.losses,
            )
        if all(stage.graphs is not None for stage in graph_stages):
            overlap_graphs = self._ensure_overlap_graphs(schedule)
            for stage in graph_stages:
                _compile_stage_graphs(
                    stage,
                    compile_config=self.compile_config,
                )
            return overlap_graphs

        for stage in graph_stages:
            if stage.graphs is not None:
                continue
            _build_stage_graphs(
                stage,
                _trace_args_for_stage(stage, trace_ctx),
                _trace_kwargs_from_context(trace_ctx),
                _trace_target_from_context(stage, trace_ctx),
                loss_kwargs,
                loss_fn=self.loss_fn,
                compile_config=self.compile_config,
                model_config=self.model_config,
                parallelism=self.parallelism,
                compile_graphs=False,
            )
        overlap_graphs = self._ensure_overlap_graphs(schedule)
        for stage in graph_stages:
            _compile_stage_graphs(stage, compile_config=self.compile_config)
        return overlap_graphs
