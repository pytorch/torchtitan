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
import functools
import logging
import warnings
from collections.abc import Callable
from typing import Any, cast, TYPE_CHECKING

import torch
import torch.fx as fx
import torch.nn as nn
import torch.utils._pytree as pytree
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining.schedules import (
    _PipelineContext,
    _PipelineScheduleRuntime,
    FULL_BACKWARD,
)

from torchtitan.config import ParallelismConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_parameter_gradient,
    BOXED_CODEGEN_META,
    compute_annotated_loss,
    compute_parameter_gradients,
    ensure_boxed_graph_module,
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    trace_input_preparer_keys,
)
from torchtitan.experiments.graph_trainer.fsdp_passes import (
    joint_transformer_block_bucketing_reordering_pass,
    merge_all_all_gathers,
    merge_all_all_reduces,
    merge_all_reduce_scatters,
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
    JointStageGraphs,
    OverlapStageGraphs,
    SplitStageGraphs,
)
from torchtitan.experiments.graph_trainer.graph_pp.utils import (
    example_inputs_from_placeholders,
    flatten_graph_values,
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
from torchtitan.experiments.graph_trainer.registry import (
    PASS_PIPELINE_REGISTRY,
    TRACE_CALL_INPUT_PREPARERS,
    TRACE_INPUT_PREPARERS,
)
from torchtitan.experiments.graph_trainer.storage import DiskStorageAdapter
from torchtitan.experiments.graph_trainer.wgrad_accumulation import (
    fuse_wgrad_accumulation_pass,
    insert_graph_gradient_accumulation,
)
from torchtitan.protocols.model import BaseModel


if TYPE_CHECKING:
    from torchtitan.experiments.graph_trainer.trainer import GraphTrainer


logger = logging.getLogger(__name__)


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
class _GraphTrainerPassConfigView:
    """Minimal config-shaped view needed by reused GraphTrainer pass builders.

    GraphPP is entered through TorchTitan's generic pipelining function API,
    which passes decomposed config fields instead of the full
    ``GraphTrainer.Config``. The pre-partition GraphTrainer passes read only
    ``compile``, ``parallelism``, and ``model``, so GraphPP exposes
    exactly those fields instead of synthesizing a fake full trainer config.
    """

    compile: GraphTrainerCompileConfig
    parallelism: ParallelismConfig
    model: BaseModel.Config


def _defer_fsdp_action_bucketing(
    passes: list[Callable],
    *,
    bucket_all_gathers: bool,
    bucket_reduce_scatters: bool,
    bucket_all_reduces: bool,
) -> list[Callable]:
    """Restrict joint bucketing to collectives retained in the joint graph."""
    configured_passes: list[Callable] = []
    for pass_fn in passes:
        if not (
            isinstance(pass_fn, functools.partial)
            and pass_fn.func is joint_transformer_block_bucketing_reordering_pass
        ):
            configured_passes.append(pass_fn)
            continue
        if not (bucket_all_gathers or bucket_reduce_scatters or bucket_all_reduces):
            continue
        configured_passes.append(
            functools.partial(
                pass_fn.func,
                *pass_fn.args,
                **dict(
                    pass_fn.keywords or {},
                    bucket_all_gathers=bucket_all_gathers,
                    bucket_reduce_scatters=bucket_reduce_scatters,
                    bucket_all_reduces=bucket_all_reduces,
                ),
            )
        )
    return configured_passes


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
    num_flat_param_values: int,
    unshard_extracted: bool,
    unsharded_param_values: list[Any],
    flat_non_param_inputs: list[Any],
    runtime_validate: bool,
) -> list[Any]:
    """Pack flat runtime values in graph placeholder order."""

    expected_num_param_inputs = (
        num_param_inputs if unshard_extracted else num_flat_param_values
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
            runtime_flat_index -= num_flat_param_values - num_param_inputs
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
    grad_accumulators: tuple[Any, ...] = ()
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

    @property
    def accumulates_gradients_in_graph(self) -> bool:
        return bool(self.grad_accumulators)

    def zero_grad_(self) -> list[Any]:
        grads = self._grad_accumulator_args()
        if grads:
            torch._foreach_zero_(grads)
        return list(self.grad_accumulators)

    def _grad_accumulator_args(self) -> list[torch.Tensor]:
        return list(
            {
                id(grad): grad
                for grad in self.grad_accumulators
                if isinstance(grad, torch.Tensor)
            }.values()
        )

    def unshard_params(
        self,
        flat_param_values: list[Any],
        *,
        runtime_validate: bool = False,
    ) -> list[Any]:
        """Run the optional FSDP unshard graph.

        Calling convention:
            ``unshard(*selected_flat_params) -> (*forward_param_inputs)``

        ``flat_param_values`` is the live stage parameter list flattened with
        the tracer's subclass rules. The unshard graph consumes only the flat
        parameters that own an all-gather chain and returns the parameter-derived
        values consumed by the forward graph. Replicated parameters and raw shards
        needed by backward rematerialization pass through unchanged.
        """

        if (
            runtime_validate
            and len(flat_param_values) != self.meta.num_flat_param_values
        ):
            raise ValueError(
                "GraphPP unshard expected one runtime value per flat param: "
                f"{len(flat_param_values)} != {self.meta.num_flat_param_values}"
            )
        if self.modules.unshard is None:
            return list(flat_param_values)
        unshard_args = []
        for param_index in self.meta.unshard_flat_param_indices:
            if runtime_validate and (
                param_index < 0 or param_index >= len(flat_param_values)
            ):
                raise ValueError(
                    "GraphPP unshard parameter index is out of range: "
                    f"index {param_index}, but runtime has "
                    f"{len(flat_param_values)} flat params"
                )
            unshard_args.append(flat_param_values[param_index])
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
        flat_buffer_values: list[Any],
        runtime_validate: bool = False,
    ) -> list[Any]:
        """Pack forward parameter, state, and user inputs in placeholder order."""

        flat_user_inputs = self._flat_user_forward_inputs(
            args,
            kwargs,
            target,
            loss_kwargs,
        )
        # Calling convention:
        # (*forward_param_inputs, *selected_state_and_user_inputs)
        return _pack_graph_args(
            graph_name="GraphPP forward",
            input_names=self.meta.fwd_input_names,
            flat_input_indices=self.meta.fwd_flat_input_indices,
            num_param_inputs=self.meta.num_fw_param_inputs,
            num_flat_param_values=self.meta.num_flat_param_values,
            unshard_extracted=self.modules.unshard is not None,
            unsharded_param_values=unsharded_param_values,
            flat_non_param_inputs=[*flat_buffer_values, *flat_user_inputs],
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
        flat_buffer_values: list[Any],
        runtime_validate: bool = False,
    ) -> tuple[Any, tuple[Any, ...]]:
        """Return ``(stage_output, saved_values_for_backward)``."""

        fw_args = self._forward_args(
            args,
            kwargs,
            target,
            loss_kwargs,
            unsharded_param_values=unsharded_param_values,
            flat_buffer_values=flat_buffer_values,
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
        stage_output: tuple[Any, ...],
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

        if runtime_validate and self.meta.is_last_stage:
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
        stage_output: tuple[Any, ...],
        saved_values_for_backward: tuple[Any, ...],
        output_grads_from_next: tuple[Any, ...],
        *,
        runtime_validate: bool = False,
    ) -> tuple[list[Any], list[Any]]:
        """Run the full backward graph.

        Calling convention:
            ``full_bw(*backward_inputs, *grad_accumulators)``
            ``-> (*param_grads, *input_grads)``
        """

        return self._split_full_backward_outputs(
            _execute_graph_module(
                self.modules.full_bw,
                [
                    *self._backward_args(
                        stage_output,
                        saved_values_for_backward,
                        output_grads_from_next,
                        runtime_validate=runtime_validate,
                    ),
                    *self._grad_accumulator_args(),
                ],
            )
        )

    def backward_input(
        self,
        stage_output: tuple[Any, ...],
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
                stage_output,
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
            ``bw_dw(*saved_for_weight_backward, *grad_accumulators)``
            ``-> (*param_grads)``
        """

        if self.modules.bw_dw is None:
            raise ValueError("GraphPP stage does not have a backward-weight graph")
        return list(
            _execute_graph_module(
                self.modules.bw_dw,
                [
                    *saved_values_for_backward_weight,
                    *self._grad_accumulator_args(),
                ],
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
    """Execute one monolithic forward/loss/backward graph for PP=1."""

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

    @property
    def accumulates_gradients_in_graph(self) -> bool:
        return False

    def zero_grad_(self) -> list[Any]:
        return []

    def unshard_params(
        self,
        flat_param_values: list[Any],
        *,
        runtime_validate: bool = False,
    ) -> list[Any]:
        return list(flat_param_values)

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
                "PP=1 joint forward/backward expects one model input, got "
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
        flat_buffer_values: list[Any],
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
                "PP=1 joint forward/backward output count mismatch: "
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


@dataclasses.dataclass(slots=True)
class _ScheduledJointGraphModules:
    """FX modules backing one scheduled PP=1 joint graph."""

    full_fwd_bwd: fx.GraphModule
    unshard: fx.GraphModule | None = None
    reduce_grad: fx.GraphModule | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class _ScheduledJointGraphMeta:
    """Flat calling-convention metadata for scheduled PP=1 joint graphs."""

    num_param_grad_values: int
    num_flat_param_values: int
    param_grad_values: GraphPPValueSpec
    full_input_names: tuple[str, ...]
    full_flat_input_indices: tuple[int, ...]
    full_output_names: tuple[str, ...]
    reduce_grad_input_names: tuple[str, ...]
    unshard_flat_param_indices: tuple[int, ...]
    num_full_param_inputs: int


@dataclasses.dataclass(slots=True)
class GraphTrainerScheduledJointStageGraphs(JointStageGraphs):
    """Execute a joint PP=1 graph with optional FSDP boundary actions."""

    modules: _ScheduledJointGraphModules
    meta: _ScheduledJointGraphMeta
    grad_accumulators: tuple[Any, ...] = ()
    runtime_meshes: tuple[DeviceMesh, ...] = ()

    @property
    def accumulates_gradients_in_graph(self) -> bool:
        return bool(self.grad_accumulators)

    def _grad_accumulator_args(self) -> list[torch.Tensor]:
        return list(
            {
                id(grad): grad
                for grad in self.grad_accumulators
                if isinstance(grad, torch.Tensor)
            }.values()
        )

    def zero_grad_(self) -> list[Any]:
        grads = self._grad_accumulator_args()
        if grads:
            torch._foreach_zero_(grads)
        return list(self.grad_accumulators)

    def unshard_params(
        self,
        flat_param_values: list[Any],
        *,
        runtime_validate: bool = False,
    ) -> list[Any]:
        if (
            runtime_validate
            and len(flat_param_values) != self.meta.num_flat_param_values
        ):
            raise ValueError(
                "Scheduled joint graph expected one runtime value per flat param: "
                f"{len(flat_param_values)} != {self.meta.num_flat_param_values}"
            )
        if self.modules.unshard is None:
            return list(flat_param_values)
        unshard_args = [
            flat_param_values[index]
            for index in self.meta.unshard_flat_param_indices
        ]
        return list(_execute_graph_module(self.modules.unshard, unshard_args))

    @staticmethod
    def _model_input(args: tuple[Any, ...]) -> Any:
        if len(args) != 1:
            raise ValueError(
                "PP=1 joint forward/backward expects one model input, got "
                f"{len(args)}"
            )
        return args[0]

    def _full_args(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        target: Any,
        loss_kwargs: dict[str, Any],
        *,
        unsharded_param_values: list[Any],
        flat_buffer_values: list[Any],
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
            graph_name="Scheduled joint graph",
            input_names=self.meta.full_input_names,
            flat_input_indices=self.meta.full_flat_input_indices,
            num_param_inputs=self.meta.num_full_param_inputs,
            num_flat_param_values=self.meta.num_flat_param_values,
            unshard_extracted=self.modules.unshard is not None,
            unsharded_param_values=unsharded_param_values,
            flat_non_param_inputs=[
                *flat_buffer_values,
                *self.runtime_meshes,
                *flatten_graph_values(list(user_inputs)),
            ],
            runtime_validate=runtime_validate,
        )
        full_args.extend(self._grad_accumulator_args())
        return full_args

    def forward_backward(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        target: Any,
        loss_kwargs: dict[str, Any],
        *,
        unsharded_param_values: list[Any],
        flat_buffer_values: list[Any],
        runtime_validate: bool = False,
    ) -> tuple[Any, list[Any]]:
        full_args = self._full_args(
            args,
            kwargs,
            target,
            loss_kwargs,
            unsharded_param_values=unsharded_param_values,
            flat_buffer_values=flat_buffer_values,
            runtime_validate=runtime_validate,
        )
        placeholders = self.modules.full_fwd_bwd.graph.find_nodes(op="placeholder")
        if runtime_validate and len(full_args) != len(placeholders):
            raise ValueError(
                "Scheduled joint graph input mismatch: "
                f"expected {len(placeholders)} args, got {len(full_args)}"
            )
        outputs = _execute_graph_module(self.modules.full_fwd_bwd, full_args)
        expected_num_outputs = self.meta.num_param_grad_values + 1
        if runtime_validate and len(outputs) != expected_num_outputs:
            raise ValueError(
                "Scheduled joint graph output count mismatch: "
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
                "Scheduled joint graph parameter grad count mismatch: "
                f"{len(unsharded_param_grads)} != "
                f"{self.meta.num_param_grad_values}"
            )
        if self.modules.reduce_grad is None:
            return list(unsharded_param_grads)
        grad_output_names = self.meta.full_output_names[
            1 : 1 + self.meta.num_param_grad_values
        ]
        grad_values_by_name = dict(
            zip(grad_output_names, unsharded_param_grads, strict=True)
        )
        reduce_grad_args = [
            grad_values_by_name[name] for name in self.meta.reduce_grad_input_names
        ]
        return list(_execute_graph_module(self.modules.reduce_grad, reduce_grad_args))

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


def _apply_graph_pp_pre_partition_or_extraction_passes(
    stage: GraphPipelineStage,
    traced: TracedResult,
    *,
    compile_config: GraphTrainerCompileConfig,
    model_config: BaseModel.Config | None,
    parallelism: ParallelismConfig | None,
    extract_fsdp_param_unshard: bool,
    extract_fsdp_grad_reduction: bool,
) -> None:
    """Apply graph invariants before GraphPP partitioning or extraction.

    Required normalization is not controlled by ``enable_passes`` or
    ``disable_passes`` because partitioning and extraction assume canonical FX
    structure: dead code is gone, no-op patterns are collapsed, and every flat
    FSDP parameter has at most one unshard chain. ``enable_passes`` only gates
    the optional GraphTrainer optimization passes that run after normalization.
    """

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
        return
    if model_config is None or parallelism is None:
        raise ValueError(
            "GraphPP requires model_config and parallelism when compile passes "
            "are enabled before stage graph partition or extraction."
        )

    passes = compile_time_passes(
        traced,
        _GraphTrainerPassConfigView(
            compile=compile_config,
            parallelism=parallelism,
            model=model_config,
        ),
        use_cuda_graph=False,
        include_inductor=False,
        include_mandatory_normalization=False,
    )

    # If fsdp unshard or reduce_grad is extracted in a separate schedule action,
    # bucketing is applied only on extracted action graph.
    if extract_fsdp_param_unshard or extract_fsdp_grad_reduction:
        passes = _defer_fsdp_action_bucketing(
            passes,
            bucket_all_gathers=not extract_fsdp_param_unshard,
            bucket_reduce_scatters=not extract_fsdp_grad_reduction,
            bucket_all_reduces=not extract_fsdp_grad_reduction,
        )

    traced.gm = apply_graph_passes(
        traced.gm,
        traced.example_inputs,
        passes,
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
    parallel_dims: ParallelDims,
    use_graph_trainer_cuda_graph: bool,
) -> list[Callable]:
    """Construct passes using the full config available to the PP=1 caller."""
    if trainer_config.compile.precompile_artifact_dir:
        if trainer_config.compile.enable_passes and use_graph_trainer_cuda_graph:
            return construct_default_graph_passes(
                traced,
                trainer_config,
                parallel_dims=parallel_dims,
            )
        return []
    if not trainer_config.compile.enable_passes:
        return construct_mandatory_graph_passes()

    pipeline_fn = PASS_PIPELINE_REGISTRY.get(trainer_config.compile.pass_pipeline)
    if pipeline_fn is not None:
        return pipeline_fn(traced, trainer_config, parallel_dims=parallel_dims)

    if use_graph_trainer_cuda_graph:
        return construct_default_graph_passes(
            traced,
            trainer_config,
            parallel_dims=parallel_dims,
        )

    return compile_time_passes(
        traced,
        trainer_config,
        parallel_dims=parallel_dims,
    )


def _build_joint_stage_graph(
    stage: GraphPipelineStage,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    target: Any,
    loss_kwargs: dict[str, Any],
    *,
    loss_fn: Callable,
    compile_config: GraphTrainerCompileConfig,
    trainer_config: "GraphTrainer.Config",
    parallel_dims: ParallelDims,
    extract_fsdp_param_unshard: bool = False,
    extract_fsdp_grad_reduction: bool = False,
    accumulate_gradients_in_graph: bool = False,
    fuse_wgrad_accumulation: bool = False,
) -> None:
    """Build the monolithic PP=1 graph with optional FSDP boundary extraction."""
    if not stage.is_first or not stage.is_last or len(args) != 1:
        raise ValueError(
            "Joint forward/backward requires one PP=1 stage and one model input"
        )

    # Calling convention:
    # (model_input, target, global_valid_tokens, model_kwargs)
    runtime_args = (
        args[0],
        target,
        loss_kwargs["global_valid_tokens"],
        kwargs,
    )
    requires_graph_extraction = (
        extract_fsdp_param_unshard
        or extract_fsdp_grad_reduction
        or accumulate_gradients_in_graph
    )
    runtime_meshes = None
    if compile_config.precompile_artifact_dir:
        if requires_graph_extraction:
            raise ValueError(
                "PP=1 precompiled artifacts do not support extracted FSDP "
                "boundaries or in-graph gradient accumulation"
            )
        storage = DiskStorageAdapter(compile_config.precompile_artifact_dir)
        if not storage.exists(_FX_TRACE_ARTIFACT_KEY):
            raise ValueError(
                "Precompiled fx_trace artifact not found at "
                f"'{compile_config.precompile_artifact_dir}/"
                f"{_FX_TRACE_ARTIFACT_KEY}.bin'. Run precompile_main with "
                "--compile.mode aot_fx_trace first."
            )
        runtime_meshes = get_spmd_precompile_meshes(parallel_dims)
        traced = precompile_fx_trace_load(
            storage,
            expected_fingerprint=compute_config_fingerprint(
                stage.submod,
                compile_config,
                parallel_dims,
            ),
            example_inputs=flatten_runtime_inputs(
                stage.submod,
                runtime_args,
                {},
                precompile_meshes=runtime_meshes,
            ),
        )
    else:
        full_forward_backward_step = make_fwd_bwd_step(stage.submod, loss_fn)

        def prepare_trace_inputs(
            trace_args: tuple[Any, ...], trace_kwargs: dict[str, Any]
        ) -> None:
            for pass_name in trace_input_preparer_keys(compile_config):
                prepare = TRACE_INPUT_PREPARERS.get(pass_name)
                if prepare is not None:
                    prepare(compile_config, trace_args, trace_kwargs)

        def prepare_trace_call_inputs(
            trace_args: tuple[Any, ...], trace_kwargs: dict[str, Any]
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            for pass_name in trace_input_preparer_keys(compile_config):
                prepare = TRACE_CALL_INPUT_PREPARERS.get(pass_name)
                if prepare is not None:
                    prepared = prepare(compile_config, trace_args, trace_kwargs)
                    if prepared is not None:
                        trace_args, trace_kwargs = prepared
            return trace_args, trace_kwargs

        traced = minimal_fx_tracer(
            full_forward_backward_step,
            module=stage.submod,
            prepare_inputs=prepare_trace_inputs,
            prepare_call_inputs=prepare_trace_call_inputs,
        )(*runtime_args)
    num_param_grads = sum(
        parameter.requires_grad
        for _, parameter in stage.submod.named_parameters(remove_duplicate=False)
    )
    if not requires_graph_extraction:
        passes = construct_joint_train_step_passes(
            traced,
            trainer_config,
            parallel_dims=parallel_dims,
            use_graph_trainer_cuda_graph=trainer_config.training.disable_cuda_graphs,
        )
        traced.gm = apply_graph_passes(
            traced.gm,
            traced.example_inputs,
            passes,
            compile_config=compile_config,
            respect_disable_passes=compile_config.enable_passes,
        )
        stage.graphs = GraphTrainerJointStageGraphs(
            traced=traced,
            module=stage.submod,
            num_param_grads=num_param_grads,
            runtime_meshes=runtime_meshes,
        )
        return

    if compile_config.enable_fsdp_dense_region_overlap and (
        extract_fsdp_param_unshard or extract_fsdp_grad_reduction
    ):
        raise ValueError(
            "FSDP dense-region overlap requires parameter all-gathers and "
            "gradient reductions to remain inside the joint graph"
        )

    _apply_graph_pp_pre_partition_or_extraction_passes(
        stage,
        traced,
        compile_config=compile_config,
        model_config=trainer_config.model,
        parallelism=trainer_config.parallelism,
        extract_fsdp_param_unshard=extract_fsdp_param_unshard,
        extract_fsdp_grad_reduction=extract_fsdp_grad_reduction,
    )
    state_params = [
        parameter
        for _, parameter in stage.submod.named_parameters(remove_duplicate=False)
    ]
    num_flat_param_values = len(flatten_graph_values(state_params))
    param_grad_values = graph_pp_value_spec(
        traced.output_subclass_layouts,
        start=1,
        count=num_param_grads,
    )
    num_param_grad_values = param_grad_values.num_flat_values
    fsdp_grads = extract_fsdp_reduce_grad_graph(
        traced.gm,
        num_param_grads=num_param_grad_values,
        param_grad_output_start=1,
        extract_grad_reduction=extract_fsdp_grad_reduction,
    )
    joint_input_names = placeholder_names(fsdp_grads.compute_module)
    fsdp_full = extract_fsdp_unshard_graph(
        fsdp_grads.compute_module,
        num_params=num_flat_param_values,
        input_names=joint_input_names,
        flat_input_indices=tuple(range(len(joint_input_names))),
        extract_fsdp_param_unshard=extract_fsdp_param_unshard,
    )
    if fsdp_full.unshard_module is not None:
        fsdp_full = dataclasses.replace(
            fsdp_full,
            unshard_module=apply_graph_passes(
                fsdp_full.unshard_module,
                (),
                [merge_all_all_gathers],
                compile_config=compile_config,
            ),
        )
    if fsdp_grads.reduce_grad_module is not None:
        fsdp_grads = dataclasses.replace(
            fsdp_grads,
            reduce_grad_module=apply_graph_passes(
                fsdp_grads.reduce_grad_module,
                (),
                [merge_all_reduce_scatters, merge_all_all_reduces],
                compile_config=compile_config,
            ),
        )

    full_output_names = output_names(fsdp_full.compute_module)
    grad_accumulators: tuple[Any, ...] = ()
    if accumulate_gradients_in_graph:
        grad_accumulators = insert_graph_gradient_accumulation(
            fsdp_full.compute_module,
            num_param_grads=num_param_grad_values,
            param_grad_output_start=1,
            device=stage.device,
        )
        if fuse_wgrad_accumulation:
            fuse_wgrad_accumulation_pass(fsdp_full.compute_module)

    modules = _ScheduledJointGraphModules(
        full_fwd_bwd=fsdp_full.compute_module,
        unshard=fsdp_full.unshard_module,
        reduce_grad=fsdp_grads.reduce_grad_module,
    )
    for gm, callable_name, action_name in (
        (modules.full_fwd_bwd, "full_fwd_bwd", "FULL_FORWARD_BACKWARD"),
        (modules.unshard, "unshard", "UNSHARD"),
        (modules.reduce_grad, "reduce_grad", "REDUCE_GRAD"),
    ):
        if gm is not None:
            _annotate_graph_pp_graph(
                gm,
                stage_index=stage.stage_index,
                callable_name=callable_name,
                action_name=action_name,
            )
    modules = _ScheduledJointGraphModules(
        full_fwd_bwd=_compile_graph_pp_module(
            modules.full_fwd_bwd,
            compile_config=compile_config,
            graph_name="stage_0_full_fwd_bwd",
        ),
        unshard=(
            None
            if modules.unshard is None
            else _compile_graph_pp_module(
                modules.unshard,
                compile_config=compile_config,
                graph_name="stage_0_unshard",
            )
        ),
        reduce_grad=(
            None
            if modules.reduce_grad is None
            else _compile_graph_pp_module(
                modules.reduce_grad,
                compile_config=compile_config,
                graph_name="stage_0_reduce_grad",
            )
        ),
    )
    stage.graphs = GraphTrainerScheduledJointStageGraphs(
        modules=modules,
        meta=_ScheduledJointGraphMeta(
            num_param_grad_values=num_param_grad_values,
            num_flat_param_values=num_flat_param_values,
            param_grad_values=param_grad_values,
            full_input_names=fsdp_full.compute_input_names,
            full_flat_input_indices=fsdp_full.compute_flat_input_indices,
            full_output_names=full_output_names,
            reduce_grad_input_names=fsdp_grads.reduce_grad_input_names,
            unshard_flat_param_indices=fsdp_full.unshard_flat_param_indices,
            num_full_param_inputs=fsdp_full.num_compute_param_inputs,
        ),
        grad_accumulators=grad_accumulators,
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
    model_config: BaseModel.Config | None = None,
    parallelism: ParallelismConfig | None = None,
    compile_graphs: bool = True,
    extract_fsdp_param_unshard: bool = True,
    extract_fsdp_grad_reduction: bool = True,
    accumulate_gradients_in_graph: bool = False,
    fuse_wgrad_accumulation: bool = False,
) -> None:
    """Trace one stage-local train step and attach bound GraphPP graphs."""
    maybe_register_blockmask_pytree_node()
    if compile_config.enable_fsdp_dense_region_overlap and (
        extract_fsdp_param_unshard or extract_fsdp_grad_reduction
    ):
        raise ValueError(
            "FSDP dense-region overlap requires parameter all-gathers and "
            "gradient reductions to remain inside the forward/backward graphs"
        )

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

        def stage_step(stage_args, stage_kwargs, target, loss_kwargs):
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
        )
        backward_only_indices = ()
    else:
        output_grads = stage_builder._flat_output_grads_from_stage_metadata(stage)

        def stage_step(stage_args, stage_kwargs, output_grads_from_next):
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
    _apply_graph_pp_pre_partition_or_extraction_passes(
        stage,
        traced,
        compile_config=compile_config,
        model_config=model_config,
        parallelism=parallelism,
        extract_fsdp_param_unshard=extract_fsdp_param_unshard,
        extract_fsdp_grad_reduction=extract_fsdp_grad_reduction,
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
    grad_accumulators: tuple[Any, ...] = ()
    if accumulate_gradients_in_graph:
        grad_accumulators = insert_graph_gradient_accumulation(
            fsdp_bw.compute_module,
            num_param_grads=num_param_grad_values,
            device=stage.device,
        )
        if didw_split is not None:
            insert_graph_gradient_accumulation(
                didw_split.bw_dw_module,
                num_param_grads=num_param_grad_values,
                device=stage.device,
                accumulators=grad_accumulators,
            )
        if fuse_wgrad_accumulation:
            fuse_wgrad_accumulation_pass(fsdp_bw.compute_module)
            if didw_split is not None:
                fuse_wgrad_accumulation_pass(didw_split.bw_dw_module)
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
        fwd_input_names=fsdp_fw.compute_input_names,
        fwd_flat_input_indices=fsdp_fw.compute_flat_input_indices,
        bw_no_fsdp_output_names=fsdp_bw.compute_output_names,
        reduce_grad_input_names=fsdp_bw.reduce_grad_input_names,
        unshard_flat_param_indices=fsdp_fw.unshard_flat_param_indices,
        num_fw_param_inputs=fsdp_fw.num_compute_param_inputs,
        is_last_stage=stage.is_last,
    )
    stage.graphs = GraphTrainerStageGraphs(
        modules=graph_modules,
        meta=graph_meta,
        grad_accumulators=grad_accumulators,
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
        annotate_graph=_annotate_graph_pp_graph,
        compile_graph_module=_compile_graph_pp_module,
        execute_graph_module=_execute_graph_module,
    )


def _trace_kwargs_from_context(ctx: _PipelineContext) -> dict[str, Any]:
    if ctx.kwarg_mbs is None:
        return {}
    return ctx.kwarg_mbs[0]


@dataclasses.dataclass(slots=True)
class GraphTrainerStageGraphProvider:
    """Build bound GraphPP stage graphs with GraphTrainer tracing and passes.

    Args:
        loss_fn: Loss function used to trace last-stage loss and backward.
        compile_config: GraphTrainer compile configuration.
        model_config: Model config consumed by GraphTrainer compile passes, or
            ``None`` when compile passes are disabled in tests.
        parallelism: Parallelism config consumed by GraphTrainer compile passes,
            or ``None`` when compile passes are disabled in tests.
        extract_fsdp_param_unshard: Whether to extract FSDP parameter all-gathers
            from forward into a separately scheduled graph.
        extract_fsdp_grad_reduction: Whether to extract FSDP gradient reduction
            from backward into a separately scheduled graph.
        accumulate_gradients_in_graph: Whether backward graphs accumulate raw
            gradients into persistent stage-owned buffers.
        fuse_wgrad_accumulation: Whether compatible WGrad producers write
            directly into those buffers.
        trainer_config: Full Trainer configuration for PP=1, or ``None`` for
            PP>1 schedules.
    """

    loss_fn: Callable
    compile_config: GraphTrainerCompileConfig
    model_config: BaseModel.Config | None
    parallelism: ParallelismConfig | None
    extract_fsdp_param_unshard: bool = True
    extract_fsdp_grad_reduction: bool = True
    accumulate_gradients_in_graph: bool = False
    fuse_wgrad_accumulation: bool = False
    trainer_config: "GraphTrainer.Config | None" = None
    parallel_dims: ParallelDims | None = None
    _warned_cuda_graph: bool = False
    # Calling convention:
    # key = (forward_stage_index, backward_stage_index); the graph is reused
    # across microbatches for that stage pair.
    _overlap_graphs: dict[tuple[int, int], OverlapStageGraphs] | None = None

    def _warn_if_cuda_graph_pass_requested(self) -> None:
        if self._warned_cuda_graph:
            return
        if self.compile_config.mode is None or not self.compile_config.enable_passes:
            return
        if "cuda_graph_pass" in self.compile_config.disable_passes:
            return
        warnings.warn(
            "GraphPP compiles extracted stage graphs with use_cuda_graph=False "
            "even though cuda_graph_pass is enabled. CUDA graph capture needs "
            "a separate GraphPP runtime integration. Pass "
            "--compile.disable_passes cuda_graph_pass to silence this warning.",
            stacklevel=3,
        )
        self._warned_cuda_graph = True

    def prepare_graphs(
        self,
        schedule: _PipelineScheduleRuntime,
        ctx: _PipelineContext,
        *,
        loss_kwargs: dict[str, Any],
    ) -> dict[tuple[int, int], OverlapStageGraphs]:
        """Build, multiplex, and compile all local GraphPP graphs for one step."""
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
        if self.trainer_config is not None:
            if len(graph_stages) != 1 or self.parallel_dims is None:
                raise ValueError(
                    "Joint forward/backward requires one stage and parallel dims"
                )
            stage = graph_stages[0]
            if stage.graphs is None:
                _build_joint_stage_graph(
                    stage,
                    stage_builder._trace_args_for_stage(stage, trace_ctx),
                    _trace_kwargs_from_context(trace_ctx),
                    stage_builder._trace_target_from_context(stage, trace_ctx),
                    loss_kwargs,
                    loss_fn=self.loss_fn,
                    compile_config=self.compile_config,
                    trainer_config=self.trainer_config,
                    parallel_dims=self.parallel_dims,
                    extract_fsdp_param_unshard=self.extract_fsdp_param_unshard,
                    extract_fsdp_grad_reduction=self.extract_fsdp_grad_reduction,
                    accumulate_gradients_in_graph=self.accumulate_gradients_in_graph,
                    fuse_wgrad_accumulation=self.fuse_wgrad_accumulation,
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
                compile_config=self.compile_config,
                model_config=self.model_config,
                parallelism=self.parallelism,
                compile_graphs=False,
                extract_fsdp_param_unshard=self.extract_fsdp_param_unshard,
                extract_fsdp_grad_reduction=self.extract_fsdp_grad_reduction,
                accumulate_gradients_in_graph=self.accumulate_gradients_in_graph,
                fuse_wgrad_accumulation=self.fuse_wgrad_accumulation,
            )

        required_overlap_pairs = stage_builder._required_multiplex_pairs(schedule)
        if not required_overlap_pairs:
            self._overlap_graphs = {}
            overlap_graphs: dict[tuple[int, int], OverlapStageGraphs] = {}
        elif self._overlap_graphs is None:
            self._overlap_graphs = _build_graph_pp_overlap_graphs(
                schedule,
                compile_config=self.compile_config,
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
            _compile_stage_graphs(stage, compile_config=self.compile_config)
        return overlap_graphs
