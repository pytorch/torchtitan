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

from __future__ import annotations

import dataclasses
import functools
import logging
import warnings
from collections.abc import Callable
from copy import deepcopy
from typing import Any, cast, Literal, TYPE_CHECKING

import torch
import torch.fx as fx
import torch.nn as nn
import torch.utils._pytree as pytree
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import PipelineStageInfo
from torch.distributed.pipelining.schedules import (
    _PipelineContext,
    _PipelineScheduleRuntime,
)

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelismContext
from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_parameter_gradient,
    BOXED_CODEGEN_META,
    compute_annotated_loss,
    compute_parameter_gradients,
    ensure_boxed_graph_module,
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.fsdp_passes import (
    joint_transformer_block_bucketing_reordering_pass,
    merge_all_all_gathers,
    merge_all_all_reduces,
    merge_all_reduce_scatters,
)
from torchtitan.experiments.graph_trainer.grad_accumulation import (
    insert_graph_gradient_accumulation_before_reduction,
    insert_graph_gradient_accumulation_from_outputs,
)
from torchtitan.experiments.graph_trainer.graph_pp import stage_builder
from torchtitan.experiments.graph_trainer.graph_pp.partition import (
    GraphMeta as PartitionGraphMeta,
    partition_joint_graph,
)
from torchtitan.experiments.graph_trainer.graph_pp.runner import (
    _GraphComputationType,
    FORWARD_BACKWARD,
    FORWARD_BACKWARD_FIRST_WITH_UNSHARD,
    FORWARD_BACKWARD_LAST_WITH_REDUCE_GRAD,
    FORWARD_BACKWARD_NOGRADACCUM,
    FULL_FORWARD_BACKWARD,
)
from torchtitan.experiments.graph_trainer.graph_pp.split_fsdp_collectives import (
    extract_fsdp_reduce_grad_graph,
    extract_fsdp_unshard_graph,
    GraphPPFSDPReduceGradExtraction,
    GraphPPFSDPUnshardExtraction,
    remove_fsdp_reduction_tail,
)
from torchtitan.experiments.graph_trainer.graph_pp.stage import (
    GraphPipelineStage,
    JointStageGraphs,
    OverlapStageGraphs,
    SplitStageGraphs,
)
from torchtitan.experiments.graph_trainer.graph_pp.utils import (
    example_inputs_from_placeholders,
    flatten_graph_values,
    graph_outputs,
    graph_pp_value_spec,
    GraphPPValueSpec,
    normalize_graph_pp_microbatch_inputs,
    output_names,
    placeholder_names,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    extract_module_state,
    minimal_fx_tracer,
    run_traced,
    TracedResult,
)
from torchtitan.experiments.graph_trainer.passes import (
    apply_graph_passes,
    canonicalize_graph_pass,
    compile_time_passes,
    construct_default_graph_passes,
    construct_mandatory_graph_passes,
    deduplicate_fsdp_unshard_chains_pass,
    eliminate_dead_code_pass,
    final_inductor_compile_passes,
)
from torchtitan.experiments.graph_trainer.precompile import (
    _FX_TRACE_ARTIFACT_KEY,
    compute_config_fingerprint,
    flatten_runtime_inputs,
    get_spmd_precompile_meshes,
    precompile_fx_trace_load,
)
from torchtitan.experiments.graph_trainer.registry import PASS_PIPELINE_REGISTRY
from torchtitan.experiments.graph_trainer.storage import DiskStorageAdapter
from torchtitan.experiments.graph_trainer.wgrad_accumulation import (
    fuse_wgrad_accumulation_pass,
)
from torchtitan.protocols.model import BaseModel


if TYPE_CHECKING:
    from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
    from torchtitan.models.common.dist_moe.runtime import _DistMoeForwardContext


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


def _append_nodes_to_outputs(
    gm: fx.GraphModule, node_names: tuple[str, ...]
) -> None:
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


def make_fwd_bwd_step(model, loss_fn):
    """Return a function that computes loss and explicit parameter gradients.

    Calling convention:
        ``(inputs, labels, global_valid_tokens, extra_kwargs)``
        ``-> (loss, *parameter_gradients)``

    ``model`` and ``loss_fn`` are captured in the closure so neither shows up
    as a graph input. Pass ``model`` through ``minimal_fx_tracer(fn, module=model)``
    to thread its parameters/buffers as static graph inputs.
    """

    def fwd_bwd_step(inputs, labels, global_valid_tokens, extra_kwargs):
        pred = model(inputs, **extra_kwargs)
        # The loss function is not a submodule of the model, so
        # annotate_module_fqns won't tag it. Annotate it here so that
        # downstream passes (bucketing, SAC, kernel annotations) can
        # attribute loss nodes in the traced graph.
        loss = compute_annotated_loss(
            loss_fn,
            pred,
            labels,
            {"global_valid_tokens": global_valid_tokens},
        )
        named_params = [
            (name, parameter)
            for name, parameter in model.named_parameters(remove_duplicate=False)
            if parameter.requires_grad
        ]
        grads = compute_parameter_gradients(loss, named_params)
        return [loss, *grads]

    return fwd_bwd_step


@dataclasses.dataclass(frozen=True, slots=True)
class GraphTrainerConfigView:
    """Subset of ``GraphTrainer.Config`` read by PP graph construction.

    GraphPP is entered through TorchTitan's generic pipelining function API,
    which passes decomposed config fields instead of the full
    ``GraphTrainer.Config``. PP graph construction reads only ``compile``,
    ``parallelism``, and ``model``, so GraphPP exposes exactly those fields
    instead of synthesizing a fake full trainer config. Both SPMD paths pass
    the full ``GraphTrainer.Config``, which has the same fields.
    """

    compile: GraphTrainerCompileConfig
    parallelism: ParallelismConfig
    model: BaseModel.Config


def _find_fsdp_bucketing_pass(
    passes: list[Callable],
) -> Callable | None:
    for pass_fn in passes:
        if (
            isinstance(pass_fn, functools.partial)
            and pass_fn.func is joint_transformer_block_bucketing_reordering_pass
        ):
            return pass_fn
    return None


def _configure_fsdp_bucketing_pass(
    fsdp_bucketing_pass: Callable | None,
    *,
    bucket_all_gathers: bool,
    bucket_reduce_scatters: bool,
    bucket_all_reduces: bool,
) -> Callable | None:
    """Restrict one FSDP bucketing pass to selected collective types."""
    if fsdp_bucketing_pass is None or not (
        bucket_all_gathers or bucket_reduce_scatters or bucket_all_reduces
    ):
        return None
    assert isinstance(fsdp_bucketing_pass, functools.partial)
    return functools.partial(
        fsdp_bucketing_pass.func,
        *fsdp_bucketing_pass.args,
        **dict(
            fsdp_bucketing_pass.keywords or {},
            bucket_all_gathers=bucket_all_gathers,
            bucket_reduce_scatters=bucket_reduce_scatters,
            bucket_all_reduces=bucket_all_reduces,
        ),
    )


def _apply_passes_with_extracted_fsdp_bucketing(
    traced: TracedResult,
    passes: list[Callable],
    fsdp_bucketing_pass: Callable | None,
    *,
    compile_config: GraphTrainerCompileConfig,
    bucket_all_gathers: bool,
    bucket_reduce_scatters: bool,
    bucket_all_reduces: bool,
) -> None:
    """Apply passes with bucketing limited to collectives kept in this graph.

    At the original bucketing-pass position, a configured copy processes only
    collective types that will remain in the joint graph. Collectives selected
    for extraction stay unbucketed until their action graphs are created.
    """
    configured_bucketing_pass: Callable | None = _configure_fsdp_bucketing_pass(
        fsdp_bucketing_pass,
        bucket_all_gathers=bucket_all_gathers,
        bucket_reduce_scatters=bucket_reduce_scatters,
        bucket_all_reduces=bucket_all_reduces,
    )
    configured_passes: list[Callable] = []
    for pass_fn in passes:
        if pass_fn is fsdp_bucketing_pass:
            if configured_bucketing_pass is not None:
                configured_passes.append(configured_bucketing_pass)
        else:
            configured_passes.append(pass_fn)
    traced.gm = apply_graph_passes(
        traced.gm,
        traced.example_inputs,
        configured_passes,
        compile_config=compile_config,
    )


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


def _execute_graph_module(
    gm: fx.GraphModule,
    args: list[Any],
) -> tuple[Any, ...]:
    """Execute one boxed FX graph module and normalize its result to a tuple."""

    with torch.no_grad():
        outputs = gm(args)
    if args:
        raise ValueError(
            "GraphPP graph call expected boxed FX codegen to clear its mutable "
            f"argument list, but {len(args)} entries remain."
        )
    if isinstance(outputs, tuple):
        return outputs
    if isinstance(outputs, list):
        return tuple(outputs)
    return (outputs,)


def _pack_graph_args(
    *,
    graph_name: str,
    input_names: tuple[str, ...],
    flat_input_indices: tuple[int, ...],
    num_param_inputs: int,
    num_sharded_param_values: int,
    unshard_extracted: bool,
    unsharded_param_values: list[Any],
    flat_non_param_inputs: list[Any],
    runtime_validate: bool,
) -> list[Any]:
    """Pack flat runtime values in graph placeholder order."""

    expected_num_param_inputs = (
        num_param_inputs if unshard_extracted else num_sharded_param_values
    )
    if runtime_validate and len(unsharded_param_values) != expected_num_param_inputs:
        raise ValueError(
            f"{graph_name} parameter input count mismatch: "
            f"{len(unsharded_param_values)} != {expected_num_param_inputs}"
        )

    flat_inputs = [*unsharded_param_values, *flat_non_param_inputs]
    graph_args = list(unsharded_param_values[:num_param_inputs])
    for name, flat_index in zip(
        input_names[num_param_inputs:],
        flat_input_indices,
        strict=True,
    ):
        runtime_flat_index = flat_index
        if unshard_extracted:
            # The traced parameter prefix may contain unused aliases from
            # parametrized modules. The unshard graph omits those leaves,
            # shifting every following buffer and user input to the left.
            runtime_flat_index -= num_sharded_param_values - num_param_inputs
        if runtime_validate and (
            runtime_flat_index < 0 or runtime_flat_index >= len(flat_inputs)
        ):
            raise ValueError(
                f"{graph_name} placeholder index is out of range: "
                f"{name} indexes {runtime_flat_index}, but runtime has "
                f"{len(flat_inputs)} flattened inputs"
            )
        graph_args.append(flat_inputs[runtime_flat_index])
    return graph_args


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

    @property
    def supports_backward_input_weight_split(self) -> bool:
        return self.modules.bw_di is not None and self.modules.bw_dw is not None

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
        runtime_validate: bool = False,
    ) -> tuple[list[Any], list[Any]]:
        """Run the full backward graph.

        Calling convention:
            ``full_bw(*backward_inputs)``
            ``-> (*param_grads, *input_grads)``
        """

        return self._split_full_backward_outputs(
            _execute_graph_module(
                self.modules.full_bw,
                [
                    *self._backward_args(
                        saved_values_for_backward,
                        output_grads_from_next,
                        runtime_validate=runtime_validate,
                    ),
                ],
            )
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
    ) -> list[Any]:
        """Run the weight-gradient graph.

        Calling convention:
            ``bw_dw(*saved_for_weight_backward)``
            ``-> (*param_grads)``
        """

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


@dataclasses.dataclass(slots=True)
class GraphTrainerJointStageGraphs(JointStageGraphs):
    """Execute SPMD without gradient accumulation as one joint graph."""

    traced: TracedResult
    module: nn.Module
    num_param_grads: int
    runtime_meshes: list[DeviceMesh] | None = None
    _run: Callable[..., Any] = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._run = run_traced(
            self.traced,
            module=self.module,
            precompile_meshes=self.runtime_meshes,
        )

    def unshard_params(
        self,
        sharded_param_values: list[Any],
        *,
        runtime_validate: bool = False,
    ) -> list[Any]:
        return list(sharded_param_values)

    def reduce_grads(
        self,
        unsharded_param_grads: list[Any],
        *,
        runtime_validate: bool = False,
    ) -> list[Any]:
        return list(unsharded_param_grads)

    def _model_input(self, args: tuple[Any, ...]) -> Any:
        if len(args) != 1:
            raise ValueError(
                "SPMD joint forward/backward expects one model input, got "
                f"{len(args)}"
            )
        return args[0]

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
        global_valid_tokens = loss_kwargs["global_valid_tokens"]
        outputs = self._run(
            self._model_input(args),
            target,
            global_valid_tokens,
            kwargs,
        )
        if len(outputs) != self.num_param_grads + 1:
            raise ValueError(
                "SPMD joint forward/backward output count mismatch: "
                f"expected {self.num_param_grads + 1}, got {len(outputs)}"
            )
        # Calling convention:
        # (loss, *parameter_gradients) -> (loss, list(parameter_gradients))
        return outputs[0], list(outputs[1:])

    def param_grads_for_accumulation(
        self,
        param_grads: list[Any],
    ) -> list[Any]:
        return param_grads


UnshardPlacement = Literal["first_microbatch", "schedule", "every_microbatch"]
ReduceGradPlacement = Literal["last_microbatch", "schedule", "every_microbatch"]


@dataclasses.dataclass(frozen=True, kw_only=True)
class GraphExecutionPlan:
    """Resolved FSDP placement, gradient accumulation, and action variants.

    ``unshard`` and ``reduce_grad`` are ``None`` without FSDP. With FSDP:

    - SPMD without gradient accumulation keeps both boundaries in its single
      joint graph.
    - SPMD with gradient accumulation either keeps both in every microbatch
      or moves them into the first and last microbatches.
    - PP places both boundaries in the schedule.

    ``reuse_unsharded_parameters`` and the computation-type helpers describe
    the SPMD schedules only.
    """

    pp_enabled: bool
    num_microbatches: int
    unshard: UnshardPlacement | None
    reduce_grad: ReduceGradPlacement | None
    fuse_wgrad_accumulation: bool

    @property
    def has_gradient_accumulation(self) -> bool:
        return self.num_microbatches > 1

    @property
    def extract_fsdp_param_unshard(self) -> bool:
        return self.unshard == "schedule"

    @property
    def extract_fsdp_grad_reduction(self) -> bool:
        return self.reduce_grad == "schedule"

    @property
    def unshard_in_first_microbatch(self) -> bool:
        return self.unshard == "first_microbatch"

    @property
    def reduce_grad_in_last_microbatch(self) -> bool:
        return self.reduce_grad == "last_microbatch"

    @property
    def split_fsdp_param_unshard(self) -> bool:
        return self.extract_fsdp_param_unshard or self.unshard_in_first_microbatch

    @property
    def split_fsdp_grad_reduction(self) -> bool:
        return self.extract_fsdp_grad_reduction or self.reduce_grad_in_last_microbatch

    @property
    def reuse_unsharded_parameters(self) -> bool:
        """Whether SPMD with gradient accumulation keeps parameters unsharded
        across all microbatches."""
        return not self.pp_enabled and self.split_fsdp_param_unshard

    @property
    def repeated_computation_type(self) -> _GraphComputationType:
        return (
            FORWARD_BACKWARD
            if self.unshard_in_first_microbatch
            else FULL_FORWARD_BACKWARD
        )

    def computation_type_for_microbatch(self, index: int) -> _GraphComputationType:
        if index == 0 and self.unshard_in_first_microbatch:
            return FORWARD_BACKWARD_FIRST_WITH_UNSHARD
        if index == 0 and self.has_gradient_accumulation:
            return FORWARD_BACKWARD_NOGRADACCUM
        if index == self.num_microbatches - 1 and self.reduce_grad_in_last_microbatch:
            return FORWARD_BACKWARD_LAST_WITH_REDUCE_GRAD
        return self.repeated_computation_type


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
            loss_kwargs["global_valid_tokens"],
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
    if compile_config is None or not compile_config.enable_passes:
        return ensure_boxed_graph_module(gm)

    example_inputs = example_inputs_from_placeholders(gm)
    gm = apply_graph_passes(
        gm,
        example_inputs,
        final_inductor_compile_passes(
            compile_config,
            use_cuda_graph=False,
            boxed_codegen=True,
        ),
        compile_config=compile_config,
    )
    if gm.meta.get(BOXED_CODEGEN_META) is not True:
        raise ValueError(
            "GraphPP compiled graph did not use boxed codegen. Check that the "
            "terminal Inductor pass was not disabled."
        )
    logger.info(
        "GraphPP compiled %s with %s inductor",
        graph_name,
        compile_config.inductor_compilation,
    )
    return gm


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
        ("full_bw", graphs.modules.full_bw),
        ("bw_di", graphs.modules.bw_di),
        ("bw_dw", graphs.modules.bw_dw),
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
        full_bw=cast(fx.GraphModule, compiled_modules["full_bw"]),
        bw_di=compiled_modules["bw_di"],
        bw_dw=compiled_modules["bw_dw"],
        unshard=compiled_modules["unshard"],
        reduce_grad=compiled_modules["reduce_grad"],
    )
    graphs.compiled = True


def _apply_graph_pp_pre_partition_or_extraction_passes(
    stage: GraphPipelineStage,
    traced: TracedResult,
    *,
    config: "GraphTrainer.Config | GraphTrainerConfigView",
    split_fsdp_param_unshard: bool,
    split_fsdp_grad_reduction: bool,
) -> Callable | None:
    """Apply graph invariants before GraphPP partitioning or extraction.

    Required normalization is not controlled by ``enable_passes`` or
    ``disable_passes`` because partitioning and extraction assume canonical FX
    structure: dead code is gone, no-op patterns are collapsed, and every flat
    FSDP parameter has at most one unshard chain. ``enable_passes`` only gates
    the optional GraphTrainer optimization passes that run after normalization.
    When FSDP collectives are extracted, bucketing is split into two steps:

    1. Select the bucketing pass for reuse on extracted action graphs.
    2. Apply the pass pipeline, bucketing only collectives kept in this graph.

    The returned pass is later configured for each extracted action graph.
    """
    compile_config: GraphTrainerCompileConfig = config.compile
    traced.gm = apply_graph_passes(
        traced.gm,
        traced.example_inputs,
        [
            eliminate_dead_code_pass,
            canonicalize_graph_pass,
            deduplicate_fsdp_unshard_chains_pass,
        ],
        compile_config=compile_config,
        respect_disable_passes=False,
    )

    if not compile_config.enable_passes:
        traced.gm = apply_graph_passes(
            traced.gm,
            traced.example_inputs,
            construct_mandatory_graph_passes(),
            compile_config=compile_config,
            respect_disable_passes=False,
        )
        return None

    passes = compile_time_passes(
        traced,
        config,
        use_cuda_graph=False,
        include_inductor=False,
        include_mandatory_normalization=False,
    )

    fsdp_bucketing_pass: Callable | None = None
    if split_fsdp_param_unshard or split_fsdp_grad_reduction:
        # Step 1: retain the original pass for extracted action graphs.
        fsdp_bucketing_pass = _find_fsdp_bucketing_pass(passes)

    # Step 2: apply the pipeline without bucketing extracted collectives.
    _apply_passes_with_extracted_fsdp_bucketing(
        traced,
        passes,
        compile_config=compile_config,
        fsdp_bucketing_pass=fsdp_bucketing_pass,
        bucket_all_gathers=not split_fsdp_param_unshard,
        bucket_reduce_scatters=not split_fsdp_grad_reduction,
        bucket_all_reduces=not split_fsdp_grad_reduction,
    )
    return fsdp_bucketing_pass


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


def construct_joint_train_step_passes(
    traced: TracedResult,
    trainer_config: "GraphTrainer.Config",
    *,
    parallelism_context: ParallelismContext,
    use_graph_trainer_cuda_graph: bool,
) -> list[Callable]:
    """Construct SPMD without gradient accumulation passes from the full
    config."""
    if trainer_config.compile.precompile_artifact_dir:
        if trainer_config.compile.enable_passes and use_graph_trainer_cuda_graph:
            return construct_default_graph_passes(
                traced,
                trainer_config,
                parallelism_context=parallelism_context,
            )
        return []
    if not trainer_config.compile.enable_passes:
        return construct_mandatory_graph_passes()

    pipeline_fn = PASS_PIPELINE_REGISTRY.get(trainer_config.compile.pass_pipeline)
    if pipeline_fn is not None:
        return pipeline_fn(traced, trainer_config, parallelism_context=parallelism_context)

    if use_graph_trainer_cuda_graph:
        return construct_default_graph_passes(
            traced,
            trainer_config,
            parallelism_context=parallelism_context,
        )

    return compile_time_passes(
        traced,
        trainer_config,
        parallelism_context=parallelism_context,
    )


def _trace_joint_stage_graph(
    stage: GraphPipelineStage,
    runtime_args: tuple[Any, Any, Any, dict[str, Any]],
    *,
    loss_fn: Callable,
    compile_config: GraphTrainerCompileConfig,
    parallelism_context: ParallelismContext,
) -> tuple[TracedResult, list[DeviceMesh] | None]:
    """Trace or load the joint forward/backward graph for either SPMD path."""
    runtime_meshes: list[DeviceMesh] | None = None
    if compile_config.precompile_artifact_dir:
        storage: DiskStorageAdapter = DiskStorageAdapter(
            compile_config.precompile_artifact_dir
        )
        if not storage.exists(_FX_TRACE_ARTIFACT_KEY):
            raise ValueError(
                "Precompiled fx_trace artifact not found at "
                f"'{compile_config.precompile_artifact_dir}/"
                f"{_FX_TRACE_ARTIFACT_KEY}.bin'. Run precompile_main first."
            )
        runtime_meshes = get_spmd_precompile_meshes(parallelism_context)
        traced: TracedResult = precompile_fx_trace_load(
            storage,
            expected_fingerprint=compute_config_fingerprint(
                stage.submod,
                compile_config,
                parallelism_context,
            ),
            example_inputs=flatten_runtime_inputs(
                stage.submod,
                runtime_args,
                {},
                precompile_meshes=runtime_meshes,
            ),
        )
        return traced, runtime_meshes

    full_forward_backward_step: Callable[..., Any] = make_fwd_bwd_step(
        stage.submod,
        loss_fn,
    )
    traced = minimal_fx_tracer(
        full_forward_backward_step,
        module=stage.submod,
    )(*runtime_args)
    return traced, runtime_meshes


def _bind_direct_joint_stage_graph(
    stage: GraphPipelineStage,
    traced: TracedResult,
    *,
    trainer_config: "GraphTrainer.Config",
    parallelism_context: ParallelismContext,
    num_param_grads: int,
    runtime_meshes: list[DeviceMesh] | None,
) -> None:
    """Apply passes and bind the SPMD without gradient accumulation executor."""
    passes: list[Callable] = construct_joint_train_step_passes(
        traced,
        trainer_config,
        parallelism_context=parallelism_context,
        use_graph_trainer_cuda_graph=trainer_config.training.disable_cuda_graphs,
    )
    traced.gm = apply_graph_passes(
        traced.gm,
        traced.example_inputs,
        passes,
        compile_config=trainer_config.compile,
        respect_disable_passes=trainer_config.compile.enable_passes,
    )
    stage.graphs = GraphTrainerJointStageGraphs(
        traced=traced,
        module=stage.submod,
        num_param_grads=num_param_grads,
        runtime_meshes=runtime_meshes,
    )


def _extract_fwd_bwd_action_graphs(
    stage: GraphPipelineStage,
    traced: TracedResult,
    *,
    trainer_config: "GraphTrainer.Config",
    plan: GraphExecutionPlan,
    num_param_grads: int,
) -> tuple[
    _ScheduledFwdBwdGraphs,
    _FwdBwdGraphsMeta,
    Callable | None,
]:
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

    reduce_grad_extraction: GraphPPFSDPReduceGradExtraction = (
        extract_fsdp_reduce_grad_graph(
            traced.gm,
            num_param_grads=num_param_grad_values,
            param_grad_output_start=1,
            extract_grad_reduction=plan.split_fsdp_grad_reduction,
        )
    )
    joint_input_names: tuple[str, ...] = placeholder_names(
        reduce_grad_extraction.compute_module
    )
    unshard_extraction: GraphPPFSDPUnshardExtraction = extract_fsdp_unshard_graph(
        reduce_grad_extraction.compute_module,
        num_params=num_sharded_param_values,
        input_names=joint_input_names,
        flat_input_indices=tuple(range(len(joint_input_names))),
        extract_fsdp_param_unshard=plan.split_fsdp_param_unshard,
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
        fwd_bwd_with_unshard_input_names = placeholder_names(
            fwd_bwd_with_unshard
        )
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
        last_unshard_extraction: GraphPPFSDPUnshardExtraction = (
            extract_fsdp_unshard_graph(
                traced.gm,
                num_params=num_sharded_param_values,
                input_names=last_input_names,
                flat_input_indices=tuple(range(len(last_input_names))),
                extract_fsdp_param_unshard=plan.split_fsdp_param_unshard,
            )
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
    plan: GraphExecutionPlan,
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
    accumulator_values: tuple[Any, ...] = (
        insert_graph_gradient_accumulation_from_outputs(
            repeat_call.module,
            num_param_grads=meta.num_param_grad_values,
            param_grad_output_start=1,
            device=stage.device,
        )
    )
    grad_accumulator_input_indices: tuple[int, ...] = tuple(
        dict.fromkeys(index for index in accumulator_values if index is not None)
    )
    graphs.call_specs[repeat_type] = dataclasses.replace(
        repeat_call,
        grad_accumulator_input_indices=grad_accumulator_input_indices,
    )

    reduce_grad_call = graphs.call_specs.get(
        FORWARD_BACKWARD_LAST_WITH_REDUCE_GRAD
    )
    if reduce_grad_call is not None:
        reduce_grad_accumulator_values: tuple[Any, ...] = (
            insert_graph_gradient_accumulation_before_reduction(
                reduce_grad_call.module,
                param_grad_output_names=repeat_call.output_names[
                    1 : 1 + meta.num_param_grad_values
                ],
                reduce_grad_input_names=meta.reduce_grad_input_names,
                accumulators=accumulator_values,
                device=stage.device,
            )
        )
        graphs.call_specs[
            FORWARD_BACKWARD_LAST_WITH_REDUCE_GRAD
        ] = dataclasses.replace(
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
    plan: GraphExecutionPlan,
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


def _build_fwd_bwd_graphs(
    stage: GraphPipelineStage,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    target: Any,
    loss_kwargs: dict[str, Any],
    *,
    loss_fn: Callable,
    trainer_config: "GraphTrainer.Config",
    parallelism_context: ParallelismContext,
    plan: GraphExecutionPlan,
) -> None:
    """Build the SPMD graph executor, with or without gradient accumulation."""
    if not stage.is_first or not stage.is_last or len(args) != 1:
        raise ValueError(
            "Joint forward/backward requires one SPMD stage and one model input"
        )

    # Calling convention:
    # (model_input, target, global_valid_tokens, model_kwargs)
    runtime_args: tuple[Any, Any, Any, dict[str, Any]] = (
        args[0],
        target,
        loss_kwargs["global_valid_tokens"],
        kwargs,
    )
    traced: TracedResult
    runtime_meshes: list[DeviceMesh] | None
    traced, runtime_meshes = _trace_joint_stage_graph(
        stage,
        runtime_args,
        loss_fn=loss_fn,
        compile_config=trainer_config.compile,
        parallelism_context=parallelism_context,
    )
    num_param_grads: int = sum(
        parameter.requires_grad
        for _, parameter in stage.submod.named_parameters(remove_duplicate=False)
    )
    if not plan.has_gradient_accumulation:
        _bind_direct_joint_stage_graph(
            stage,
            traced,
            trainer_config=trainer_config,
            parallelism_context=parallelism_context,
            num_param_grads=num_param_grads,
            runtime_meshes=runtime_meshes,
        )
        return

    stage.graphs = _build_scheduled_fwd_bwd_graphs(
        stage,
        traced,
        trainer_config=trainer_config,
        plan=plan,
        num_param_grads=num_param_grads,
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
    num_rewritten = 0
    for op, argument_index in _dist_moe_forward_slot_arguments():
        for node in traced.gm.graph.find_nodes(op="call_function", target=op):
            if argument_index >= len(node.args):
                raise ValueError(
                    f"{op} has no activation-slot operand at index {argument_index}"
                )
            node_args = list(node.args)
            captured_slot = node_args[argument_index]
            if not isinstance(captured_slot, fx.Node):
                raise ValueError(
                    f"{op} captured a non-node activation slot: {captured_slot!r}"
                )
            captured_slot_nodes.add(captured_slot)
            node_args[argument_index] = activation_slot_id_1
            node.args = tuple(node_args)
            num_rewritten += 1
    if num_rewritten == 0:
        raise ValueError(
            "GraphPP received a Dist-MoE activation slot but traced no Dist-MoE "
            "forward operation"
        )
    for captured_slot in captured_slot_nodes:
        if not captured_slot.users and captured_slot.op == "get_attr":
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
        extract_fsdp_param_unshard=extract_fsdp_param_unshard,
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
        extract_grad_reduction=extract_fsdp_grad_reduction,
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
    didw_split = stage_builder._split_stage_backward_graph(
        fsdp_bw.compute_module,
        num_param_grads=num_param_grad_values,
        num_input_grads=num_input_grad_values,
    )
    # 6. Attach the callable container and the GraphTrainer-only metadata used
    # to pack/unpack its flat graph inputs and outputs.
    graph_modules = _StageGraphModules(
        fw=fsdp_fw.compute_module,
        full_bw=fsdp_bw.compute_module,
        bw_di=None if didw_split is None else didw_split.bw_di_module,
        bw_dw=None if didw_split is None else didw_split.bw_dw_module,
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


def _trace_kwargs_from_context(
    ctx: _PipelineContext,
    microbatch_index: int = 0,
) -> dict[str, Any]:
    if ctx.kwarg_mbs is None:
        return {}
    return ctx.kwarg_mbs[microbatch_index]


def _resolve_dist_moe_activation_slot(
    forward_context: _DistMoeForwardContext | None,
    *,
    stage_index: int,
    microbatch_index: int = 0,
) -> torch.Tensor | None:
    """Resolve the representative Dist-MoE slot used to trace one stage."""
    if forward_context is None:
        return None
    return forward_context.resolve_activation_slot(
        PipelineStageInfo(
            stage_index=stage_index,
            microbatch_index=microbatch_index,
        )
    )


@dataclasses.dataclass(slots=True)
class GraphTrainerStageGraphProvider:
    """Build bound GraphPP stage graphs with GraphTrainer tracing and passes.

    Args:
        loss_fn: Loss function used to trace last-stage loss and backward.
        config: Full Trainer configuration for SPMD, or its compile,
            parallelism, and model fields for PP.
        plan: Resolved FSDP placement and gradient accumulation choices.
    """

    loss_fn: Callable
    config: "GraphTrainer.Config | GraphTrainerConfigView"
    plan: GraphExecutionPlan
    parallelism_context: ParallelismContext | None = None
    _warned_cuda_graph: bool = False
    # Calling convention:
    # key = (forward_stage_index, backward_stage_index); the graph is reused
    # across microbatches for that stage pair.
    _overlap_graphs: dict[tuple[int, int], OverlapStageGraphs] | None = None

    def _warn_if_cuda_graph_pass_requested(self) -> None:
        if self._warned_cuda_graph:
            return
        if not self.config.compile.enable_passes:
            return
        if "cuda_graph_pass" in self.config.compile.disable_passes:
            return
        warnings.warn(
            "GraphPP compiles extracted stage graphs with use_cuda_graph=False "
            "even though cuda_graph_pass is enabled. CUDA graph capture needs "
            "a separate GraphPP runtime integration. Pass "
            "Add 'cuda_graph_pass' to compile.disable_passes to silence this warning.",
            stacklevel=3,
        )
        self._warned_cuda_graph = True

    def prepare_graphs(
        self,
        schedule: _PipelineScheduleRuntime,
        ctx: _PipelineContext,
        *,
        loss_kwargs: dict[str, Any],
        dist_moe_forward_context: _DistMoeForwardContext | None = None,
    ) -> dict[tuple[int, int], OverlapStageGraphs]:
        """Build one graph per stage with its optional Dist-MoE slot input."""
        graph_stages = [cast(GraphPipelineStage, stage) for stage in schedule._stages]
        maybe_register_blockmask_pytree_node()
        trace_ctx = ctx
        if ctx.arg_mbs is not None or ctx.kwarg_mbs is not None:
            # Upstream PP creates a fresh _PipelineContext for each action, but
            # all contexts share the same arg_mbs/kwarg_mbs lists. Mutate lists
            # that upstream provided so later actions see graphable BlockMask objects.
            # Use distinct objects for tracing so make_fx does not consume the
            # same closure tensor objects that runtime replay will receive.
            num_microbatches = (
                len(ctx.arg_mbs) if ctx.arg_mbs is not None else len(ctx.kwarg_mbs)
            )
            arg_mbs = (
                ctx.arg_mbs
                if ctx.arg_mbs is not None
                else [() for _ in range(num_microbatches)]
            )
            kwarg_mbs = (
                ctx.kwarg_mbs
                if ctx.kwarg_mbs is not None
                else [{} for _ in range(num_microbatches)]
            )
            trace_arg_mbs, trace_kwarg_mbs = normalize_graph_pp_microbatch_inputs(
                arg_mbs,
                kwarg_mbs,
            )
            runtime_arg_mbs, runtime_kwarg_mbs = normalize_graph_pp_microbatch_inputs(
                arg_mbs,
                kwarg_mbs,
            )
            if ctx.arg_mbs is not None:
                ctx.arg_mbs[:] = runtime_arg_mbs
            if ctx.kwarg_mbs is not None:
                ctx.kwarg_mbs[:] = runtime_kwarg_mbs
            trace_ctx = _PipelineContext(
                schedule,
                trace_arg_mbs,
                trace_kwarg_mbs,
                ctx.target_mbs,
                ctx.losses,
            )
        if not self.plan.pp_enabled:
            if len(graph_stages) != 1 or self.parallelism_context is None:
                raise ValueError(
                    "Joint forward/backward requires one stage and parallel dims"
                )
            stage = graph_stages[0]
            if stage.graphs is None:
                _build_fwd_bwd_graphs(
                    stage,
                    stage_builder._trace_args_for_stage(stage, trace_ctx),
                    _trace_kwargs_from_context(trace_ctx),
                    stage_builder._trace_target_from_context(stage, trace_ctx),
                    loss_kwargs,
                    loss_fn=self.loss_fn,
                    trainer_config=cast("GraphTrainer.Config", self.config),
                    parallelism_context=self.parallelism_context,
                    plan=self.plan,
                )
            return {}

        for stage in graph_stages:
            if stage.graphs is not None:
                continue
            _build_stage_graphs(
                stage,
                stage_builder._trace_args_for_stage(stage, trace_ctx),
                _trace_kwargs_from_context(trace_ctx),
                stage_builder._trace_target_from_context(stage, trace_ctx),
                loss_kwargs,
                loss_fn=self.loss_fn,
                config=self.config,
                compile_graphs=False,
                extract_fsdp_param_unshard=self.plan.extract_fsdp_param_unshard,
                extract_fsdp_grad_reduction=self.plan.extract_fsdp_grad_reduction,
                activation_slot_id_1=_resolve_dist_moe_activation_slot(
                    dist_moe_forward_context,
                    stage_index=stage.stage_index,
                ),
            )

        required_overlap_pairs = stage_builder._required_multiplex_pairs(schedule)
        if not required_overlap_pairs:
            self._overlap_graphs = {}
            overlap_graphs: dict[tuple[int, int], OverlapStageGraphs] = {}
        elif self._overlap_graphs is None:
            self._overlap_graphs = _build_graph_pp_overlap_graphs(
                schedule,
                compile_config=self.config.compile,
            )
            overlap_graphs = dict(self._overlap_graphs)
        else:
            missing_pairs = required_overlap_pairs - set(self._overlap_graphs)
            if missing_pairs:
                raise ValueError(
                    "GraphPP cached overlap graphs do not cover current "
                    f"schedule pairs: missing {sorted(missing_pairs)}."
                )
            overlap_graphs = {
                pair: self._overlap_graphs[pair] for pair in required_overlap_pairs
            }

        for stage in graph_stages:
            _compile_stage_graphs(stage, compile_config=self.config.compile)
        return overlap_graphs
