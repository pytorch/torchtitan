# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""GraphPP stage graph construction and runtime action handlers.

ABI and wrapping contract:
1. GraphPP extracted graphs execute on the flat value ABI produced by
   ``minimal_fx_tracer``. Tensor subclasses are unwrapped into plain leaves by
   the tracer before FX execution.
2. Only values that cross the PP/runtime boundary are rewrapped: stage forward
   outputs, input gradients sent to the previous stage, and parameter gradients
   before assigning to live ``param.grad``.
3. Internal graph ABI values stay flat because they never escape GraphPP graph
   execution: saved-for-backward values, unsharded FSDP params, raw grad leaves,
   reduce-grad inputs, and multiplexed intermediate outputs.
4. DTensor and other traceable tensor subclasses use the existing tracer layout
   metadata. GraphPP must not add a separate DTensor-specific wrapping path.
"""

import dataclasses
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.fx as fx
import torch.nn as nn
import torch.utils._pytree as pytree
from torch.distributed.pipelining.microbatch import _split_tensor
from torch.distributed.pipelining.schedules import (
    _Action,
    _PipelineContext,
    _PipelineScheduleRuntime,
    _TARGET_CHUNK_SPEC,
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
from torch.distributed.pipelining.stage import (
    _normalize_model_output_as_tuple,
    PipelineStage,
)

from torchtitan.experiments.graph_trainer.common_utils import (
    accumulate_param_grads_,
    compute_annotated_loss,
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.graph_pp.fsdp import (
    split_backward_fsdp_collectives,
    split_forward_fsdp_collectives,
)
from torchtitan.experiments.graph_trainer.graph_pp.graph_multiplex import (
    multiplex_fw_bw_graph,
)
from torchtitan.experiments.graph_trainer.graph_pp.partition import (
    GraphMeta as PartitionGraphMeta,
    GraphPPInputSource,
    partition_joint_graph,
)
from torchtitan.experiments.graph_trainer.graph_pp.split_di_dw import (
    GraphPPDiDwSplit,
    split_di_dw_graph,
)
from torchtitan.experiments.graph_trainer.graph_pp.utils import (
    flatten_graph_values,
    graph_pp_value_spec,
    GraphPPValueSpec,
    normalize_graph_pp_microbatch_inputs,
    preserve_module_buffer_state,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    extract_module_state,
    minimal_fx_tracer,
    TracedResult,
)
from torchtitan.experiments.graph_trainer.passes import (
    annotate_flex_attention_for_regional_inductor_pass,
    apply_graph_passes,
    compile_time_passes,
    full_inductor_compilation_pass,
    regional_inductor_pass,
)
from torchtitan.tools.logging import logger


@dataclasses.dataclass(slots=True)
class GraphCallables:
    fw: fx.GraphModule
    full_bw: fx.GraphModule
    bw_di: fx.GraphModule | None = None
    bw_dw: fx.GraphModule | None = None
    unshard: fx.GraphModule | None = None
    reduce_grad: fx.GraphModule | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class StageGraphMeta:
    # Counts ending in ``_outputs``/``_values`` refer to the raw tensor ABI of
    # the extracted FX graphs. Counts ending in ``_leaves``/``_grads`` refer to
    # semantic pytree leaves before minimal_fx_tracer unwraps tensor subclasses.
    num_user_outputs: int
    num_user_output_leaves: int
    num_saved_for_backward: int
    num_bwd_runtime_inputs: int
    num_params: int
    num_param_grad_values: int
    num_buffers: int
    num_input_grads: int
    num_input_grad_values: int
    fwd_output_values: GraphPPValueSpec
    param_grad_values: GraphPPValueSpec
    input_grad_values: GraphPPValueSpec
    partition: PartitionGraphMeta
    fwd_input_sources: tuple[GraphPPInputSource, ...]
    bw_no_fsdp_output_names: tuple[str, ...] = ()
    reduce_grad_input_names: tuple[str, ...] = ()
    unshard_input_sources: tuple[GraphPPInputSource, ...] = ()
    unshard_output_names: tuple[str, ...] = ()


@dataclasses.dataclass(slots=True)
class StageTraceSpec:
    output_spec: pytree.TreeSpec | None = None
    output_grad_spec: pytree.TreeSpec | None = None


@dataclasses.dataclass(slots=True)
class GraphPPStageRuntimeState:
    sharded_params: list[Any] = dataclasses.field(default_factory=list)
    unsharded_params: list[Any] = dataclasses.field(default_factory=list)
    buffers: list[Any] = dataclasses.field(default_factory=list)
    sharded_grads: list[Any] = dataclasses.field(default_factory=list)
    unsharded_grads: list[Any] = dataclasses.field(default_factory=list)
    trainable_params: list[torch.Tensor] = dataclasses.field(default_factory=list)

    def clear(self) -> None:
        self.sharded_params = []
        self.unsharded_params = []
        self.buffers = []
        self.sharded_grads = []
        self.unsharded_grads = []
        self.trainable_params = []


@dataclasses.dataclass(frozen=True, slots=True)
class _GraphPPPassModelSpec:
    model: Any


@dataclasses.dataclass(frozen=True, slots=True)
class _GraphPPPassConfig:
    compile: GraphTrainerCompileConfig
    parallelism: Any
    model_spec: _GraphPPPassModelSpec


MultiplexFwBwGraphPass = Callable[[fx.GraphModule, fx.GraphModule], fx.GraphModule]


def _execute_graph(graph: fx.GraphModule, args: list[Any]) -> tuple[Any, ...]:
    with torch.no_grad():
        outputs = graph(*args)
    if isinstance(outputs, tuple):
        return outputs
    if isinstance(outputs, list):
        return tuple(outputs)
    return (outputs,)


def _requires_grad_like(value: Any) -> Any:
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        value = value.detach().requires_grad_(True)
    return value


def _unwrapped_flat_count(values: list[Any]) -> int:
    return len(flatten_graph_values(values))


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


def _example_inputs_from_placeholders(gm: fx.GraphModule) -> tuple[Any, ...]:
    example_inputs = []
    for node in gm.graph.find_nodes(op="placeholder"):
        if "val" not in node.meta:
            raise ValueError(
                "GraphPP cannot compile graph without placeholder metadata: "
                f"{node.name}"
            )
        example_inputs.append(node.meta["val"])
    return tuple(example_inputs)


def _compile_graph_pp_module(
    gm: fx.GraphModule | None,
    *,
    compile_config: GraphTrainerCompileConfig,
    graph_name: str,
) -> fx.GraphModule | None:
    """Compile one extracted GraphPP callable.

    Contract:
      Input is an already partitioned FX graph whose placeholders carry
      ``meta["val"]`` example inputs. The graph remains unchanged when compile
      passes are disabled.

    Pseudocode:
      if graph is missing or compile passes are disabled: return graph
      collect placeholder example inputs
      run regional or full Inductor according to compile_config
      return compiled GraphModule
    """
    if gm is None or not compile_config.enable or not compile_config.enable_passes:
        return gm

    example_inputs = _example_inputs_from_placeholders(gm)
    if compile_config.inductor_compilation == "regional":
        from torchtitan.models.common.attention import FlexAttention

        gm = annotate_flex_attention_for_regional_inductor_pass(
            gm,
            example_inputs,
            flex_compile_config=FlexAttention.inductor_configs,
        )
        if compile_config.numerics_changing_optim:
            from torchtitan.experiments.graph_trainer.performance_passes import (
                annotate_rmsnorm_for_regional_inductor_pass,
            )

            gm = annotate_rmsnorm_for_regional_inductor_pass(gm, example_inputs)
        gm = regional_inductor_pass(gm, example_inputs)
    elif compile_config.inductor_compilation == "full":
        gm = full_inductor_compilation_pass(gm, example_inputs)
    else:
        raise ValueError(
            "GraphPP supports --compile.inductor_compilation regional or full, "
            f"got {compile_config.inductor_compilation!r}"
        )
    logger.info(
        "GraphPP compiled %s with %s inductor",
        graph_name,
        compile_config.inductor_compilation,
    )
    return gm


def _compile_graph_pp_callables(
    callables: GraphCallables,
    *,
    compile_config: GraphTrainerCompileConfig,
    stage_index: int,
) -> GraphCallables:
    """Compile the independent callables for one stage graph bundle.

    Contract:
      Each callable has already been extracted and annotated. Compilation is
      rank-local and synchronous; all PP ranks build their owned stages, but a
      single rank compiles its local callables one at a time.

    Pseudocode:
      for each callable slot in the stage bundle:
        compile it with _compile_graph_pp_module
      return a bundle with the same callable layout
    """
    return GraphCallables(
        fw=_compile_graph_pp_module(
            callables.fw,
            compile_config=compile_config,
            graph_name=f"stage_{stage_index}_fw",
        ),
        full_bw=_compile_graph_pp_module(
            callables.full_bw,
            compile_config=compile_config,
            graph_name=f"stage_{stage_index}_full_bw",
        ),
        bw_di=_compile_graph_pp_module(
            callables.bw_di,
            compile_config=compile_config,
            graph_name=f"stage_{stage_index}_bw_di",
        ),
        bw_dw=_compile_graph_pp_module(
            callables.bw_dw,
            compile_config=compile_config,
            graph_name=f"stage_{stage_index}_bw_dw",
        ),
        unshard=_compile_graph_pp_module(
            callables.unshard,
            compile_config=compile_config,
            graph_name=f"stage_{stage_index}_unshard",
        ),
        reduce_grad=_compile_graph_pp_module(
            callables.reduce_grad,
            compile_config=compile_config,
            graph_name=f"stage_{stage_index}_reduce_grad",
        ),
    )


def _compile_stage_graph_bundle(stage: "GraphPipelineStage") -> None:
    if stage._graph_pp_callables_compiled:
        return
    if stage.graph_callables is None:
        raise ValueError(
            "GraphPP cannot compile a missing graph bundle for "
            f"stage {stage.stage_index}."
        )
    if not stage.compile_config.enable or not stage.compile_config.enable_passes:
        return
    stage.graph_callables = _compile_graph_pp_callables(
        stage.graph_callables,
        compile_config=stage.compile_config,
        stage_index=stage.stage_index,
    )
    stage._graph_pp_callables_compiled = True


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


def _annotate_graph_pp_callables(
    callables: GraphCallables,
    *,
    stage_index: int,
) -> None:
    graph_specs = (
        (callables.fw, "fw", "FORWARD"),
        (callables.full_bw, "full_bw", "FULL_BACKWARD"),
        (callables.bw_di, "bw_di", "BACKWARD_INPUT"),
        (callables.bw_dw, "bw_dw", "BACKWARD_WEIGHT"),
        (callables.unshard, "unshard", "UNSHARD"),
        (callables.reduce_grad, "reduce_grad", "REDUCE_GRAD"),
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
    stage: "GraphPipelineStage",
    traced: TracedResult,
) -> None:
    """Apply metadata-preserving GraphTrainer passes before GraphPP partition.

    Contract:
      GraphPP runs the normal GraphTrainer compile-time pass pipeline on each
      stage-local train-step FX graph before partitioning. Only the final
      compilation path is excluded because GraphPP compiles extracted
      forward/backward/FSDP/dI/dW callables after partitioning.

    Pseudocode:
      build GraphTrainer compile_time_passes(include_inductor=False)
      apply those passes to traced.gm
      leave the graph in FX form for GraphPP partition/extraction
    """
    compile_config = stage.compile_config
    if not compile_config.enable_passes:
        return

    model_config = stage.model_config
    parallelism = stage.parallelism
    if model_config is None or parallelism is None:
        raise ValueError(
            "GraphPP requires model_config and parallelism on each stage when "
            "compile passes are enabled."
        )

    passes = compile_time_passes(
        traced,
        _GraphPPPassConfig(
            compile=compile_config,
            parallelism=parallelism,
            model_spec=_GraphPPPassModelSpec(model=model_config),
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


def _example_output_grads_from_stage_metadata(
    stage: "GraphPipelineStage",
    output_example: Any,
) -> Any:
    flat_outputs, output_spec = pytree.tree_flatten(output_example)
    # NOTE: PipelineStage metadata entries are typed, but GraphPP still reaches
    # through private fields/methods until PP exposes a public access surface.
    stage_meta = getattr(stage, "_stage_meta", None)
    output_grad_metas = None if stage_meta is None else stage_meta.output_grads
    if output_grad_metas is None:
        raise ValueError(
            "GraphPP requires eager PP backward metadata before tracing a "
            f"non-last stage. Missing output_grads for stage {stage.stage_index}."
        )
    if len(output_grad_metas) != len(flat_outputs):
        raise ValueError(
            "GraphPP output grad metadata does not match stage output structure: "
            f"{len(output_grad_metas)} metadata entries for "
            f"{len(flat_outputs)} output leaves"
        )
    output_grads = [
        None if meta is None else stage._to_tensor(meta) for meta in output_grad_metas
    ]
    return pytree.tree_unflatten(output_grads, output_spec)


def _build_stage_graph_bundle(
    stage: "GraphPipelineStage",
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    target: Any,
    loss_kwargs: dict[str, Any],
    *,
    compile_callables: bool = True,
) -> None:
    """Trace one stage-local train step and build its GraphPP graph bundle.

    Contract:
      The trace function must return forward user outputs first, then parameter
      gradients, then input gradients. ``minimal_fx_tracer`` owns module state
      flattening and tensor-subclass unwrap/rewrap metadata; GraphPP only slices
      that metadata into stage-output, param-grad, and input-grad views.

    Pseudocode:
      prepare representative microbatch args with requires_grad
      if last stage: trace forward + annotated loss + autograd.grad(loss, ...)
      else: trace forward + autograd.grad(outputs, ..., output_grads)
      apply metadata-preserving pre-partition passes
      partition into forward/backward
      extract FSDP collective graphs and optional dI/dW graphs
      annotate each callable and optionally run the final compilation step
      store callables and ABI metadata on the stage
    """
    maybe_register_blockmask_pytree_node()
    stage_args = pytree.tree_map_only(torch.Tensor, _requires_grad_like, args)
    stage_kwargs = pytree.tree_map_only(torch.Tensor, _requires_grad_like, kwargs)

    state_params = [p for _, p in stage.submod.named_parameters(remove_duplicate=False)]
    grad_params = [p for p in state_params if p.requires_grad]
    buffers = [b for _, b in stage.submod.named_buffers(remove_duplicate=False)]
    num_state_param_values = _unwrapped_flat_count(state_params)
    num_grad_params = len(grad_params)
    num_buffers = len(buffers)
    num_input_grad_leaves = len(_grad_input_leaves(stage_args, stage_kwargs))
    trace_spec = StageTraceSpec()

    with preserve_module_buffer_state(stage.submod):
        if stage.is_last:
            with torch.no_grad():
                pred_example = stage.submod(*args, **kwargs)
                loss_example = compute_annotated_loss(
                    stage.loss_fn,
                    pred_example,
                    target,
                    loss_kwargs,
                )
                loss_grad = torch.ones_like(loss_example)
            _, fwd_output_spec = pytree.tree_flatten(loss_example)

            def stage_step(
                stage_args,
                stage_kwargs,
                target,
                loss_kwargs,
                loss_grad,
            ):
                # Last-stage ABI:
                #   inputs: stage args/kwargs, target, loss kwargs, scalar tangent
                #   outputs: loss, then parameter grads, then input grads.
                pred = stage.submod(*stage_args, **stage_kwargs)
                loss = compute_annotated_loss(stage.loss_fn, pred, target, loss_kwargs)
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
                    grad_outputs=loss_grad,
                    allow_unused=True,
                )
                return [loss, *grads]

            traced = minimal_fx_tracer(stage_step, module=stage.submod)(
                stage_args,
                stage_kwargs,
                target,
                loss_kwargs,
                loss_grad,
            )
            num_fwd_output_leaves = 1
            backward_only_indices = (len(traced.example_inputs) - 1,)
        else:
            with torch.no_grad():
                output_example = stage.submod(*args, **kwargs)
            output_grads = _example_output_grads_from_stage_metadata(
                stage, output_example
            )
            _, fwd_output_spec = pytree.tree_flatten(output_example)
            trace_spec.output_spec = fwd_output_spec
            _, trace_spec.output_grad_spec = pytree.tree_flatten(output_grads)

            def stage_step(stage_args, stage_kwargs, output_grads):
                # Non-last-stage ABI:
                #   inputs: stage args/kwargs plus backward-only output grads
                #   outputs: flat stage outputs, then parameter grads, then input grads.
                output = stage.submod(*stage_args, **stage_kwargs)
                flat_outputs, trace_spec.output_spec = pytree.tree_flatten(output)
                flat_output_grads, trace_spec.output_grad_spec = pytree.tree_flatten(
                    output_grads
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
                    flat_outputs,
                    grad_inputs,
                    grad_outputs=flat_output_grads,
                    allow_unused=True,
                )
                return [*flat_outputs, *grads]

            traced = minimal_fx_tracer(stage_step, module=stage.submod)(
                stage_args,
                stage_kwargs,
                output_grads,
            )
            num_fwd_output_leaves = len(pytree.tree_leaves(output_example))
            state_flat, _ = pytree.tree_flatten(extract_module_state(stage.submod))
            prefix_user_flat, _ = pytree.tree_flatten(((stage_args, stage_kwargs), {}))
            backward_only_start = _unwrapped_flat_count(
                [*state_flat, *prefix_user_flat]
            )
            backward_only_count = _unwrapped_flat_count(
                pytree.tree_leaves(output_grads)
            )
            backward_only_indices = tuple(
                range(backward_only_start, backward_only_start + backward_only_count)
            )

    _apply_graph_pp_pre_partition_passes(stage, traced)
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
    fw_module, bw_module, partition_meta = partition_joint_graph(
        traced,
        num_fwd_outputs=num_fwd_output_values,
        backward_only_input_indices=backward_only_indices,
    )
    fsdp_fw = split_forward_fsdp_collectives(
        fw_module,
        num_params=num_state_param_values,
        fwd_input_sources=partition_meta.fwd_input_sources,
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
    graph_callables = GraphCallables(
        fw=fsdp_fw.fw_no_fsdp_module,
        full_bw=fsdp_bw.bw_no_fsdp_module,
        bw_di=None if didw_split is None else didw_split.bw_di_module,
        bw_dw=None if didw_split is None else didw_split.bw_dw_module,
        unshard=fsdp_fw.unshard_module,
        reduce_grad=fsdp_bw.reduce_grad_module,
    )
    _annotate_graph_pp_callables(
        graph_callables,
        stage_index=stage.stage_index,
    )
    stage.graph_callables = graph_callables
    stage._graph_pp_callables_compiled = False
    stage.graph_meta = StageGraphMeta(
        num_user_outputs=partition_meta.num_fwd_user_outputs,
        num_user_output_leaves=num_fwd_output_leaves,
        num_saved_for_backward=partition_meta.num_saved_for_backward,
        num_bwd_runtime_inputs=partition_meta.num_bwd_runtime_inputs,
        num_params=num_grad_params,
        num_param_grad_values=num_param_grad_values,
        num_buffers=num_buffers,
        num_input_grads=num_input_grad_leaves,
        num_input_grad_values=num_input_grad_values,
        fwd_output_values=fwd_output_values,
        param_grad_values=param_grad_values,
        input_grad_values=input_grad_values,
        partition=partition_meta,
        fwd_input_sources=fsdp_fw.fw_no_fsdp_input_sources,
        bw_no_fsdp_output_names=fsdp_bw.bw_no_fsdp_output_names,
        reduce_grad_input_names=fsdp_bw.reduce_grad_input_names,
        unshard_input_sources=fsdp_fw.unshard_input_sources,
        unshard_output_names=fsdp_fw.unshard_output_names,
    )
    stage.trace_spec = trace_spec
    logger.info(
        "GraphPP traced stage %s: fwd_outputs=%s saved=%s bwd_runtime_inputs=%s",
        stage.stage_index,
        stage.graph_meta.num_user_outputs,
        stage.graph_meta.num_saved_for_backward,
        stage.graph_meta.num_bwd_runtime_inputs,
    )
    if compile_callables:
        _compile_stage_graph_bundle(stage)


def _run_fw_module(
    fw_module: fx.GraphModule,
    graph_meta: StageGraphMeta,
    fw_args: list[Any],
) -> tuple[Any, tuple[Any, ...]]:
    fw_placeholders = fw_module.graph.find_nodes(op="placeholder")
    if len(fw_args) != len(fw_placeholders):
        raise ValueError(
            "GraphPP forward graph input mismatch: "
            f"expected {len(fw_placeholders)} args, got {len(fw_args)}. "
            "Placeholders: "
            f"{[node.name for node in fw_placeholders]}. "
            f"Graph meta inputs: {[source.name for source in graph_meta.fwd_input_sources]}."
        )
    fw_outputs = _execute_graph(fw_module, fw_args)
    user_outputs = fw_outputs[: graph_meta.num_user_outputs]
    saved_start = graph_meta.num_user_outputs
    saved_end = saved_start + graph_meta.num_saved_for_backward
    saved_intermediates = tuple(fw_outputs[saved_start:saved_end])
    output = graph_meta.fwd_output_values.unflatten(user_outputs)
    return output, saved_intermediates


def _run_full_bw_module(
    bw_module: fx.GraphModule,
    graph_meta: StageGraphMeta,
    bw_args: list[Any],
) -> tuple[list[Any], list[Any]]:
    bw_outputs = _execute_graph(bw_module, bw_args)
    num_param_grads = graph_meta.num_param_grad_values
    param_grads = list(bw_outputs[:num_param_grads])
    input_grads = graph_meta.input_grad_values.wrap_flat_values(
        bw_outputs[num_param_grads:]
    )
    return input_grads, param_grads


def _run_di_bw_module(
    bw_di_module: fx.GraphModule,
    graph_meta: StageGraphMeta,
    bw_args: list[Any],
) -> tuple[list[Any], list[Any]]:
    outputs = _execute_graph(bw_di_module, bw_args)
    return (
        graph_meta.input_grad_values.wrap_flat_values(
            outputs[: graph_meta.num_input_grad_values]
        ),
        list(outputs[graph_meta.num_input_grad_values :]),
    )


def _run_dw_bw_module(
    bw_dw_module: fx.GraphModule,
    graph_meta: StageGraphMeta,
    bw_args: list[Any],
) -> list[Any]:
    return list(_execute_graph(bw_dw_module, bw_args))


def _raw_unsharded_grads(stage: "GraphPipelineStage") -> list[Any]:
    assert stage.graph_meta is not None
    raw_grads = list(stage.state.unsharded_grads)
    if len(raw_grads) != stage.graph_meta.num_param_grad_values:
        raise ValueError(
            "GraphPP raw unsharded grad count mismatch: "
            f"expected {stage.graph_meta.num_param_grad_values}, "
            f"got {len(raw_grads)}"
        )
    return raw_grads


def _prepare_reduce_grad_args(stage: "GraphPipelineStage") -> list[Any]:
    assert stage.graph_meta is not None
    raw_grads = _raw_unsharded_grads(stage)
    grad_values_by_name = dict(
        zip(
            stage.graph_meta.bw_no_fsdp_output_names[
                : stage.graph_meta.num_param_grad_values
            ],
            raw_grads,
            strict=True,
        )
    )
    return [
        grad_values_by_name[name] for name in stage.graph_meta.reduce_grad_input_names
    ]


def _scale_grad_values_(grads: list[Any], grad_scale_factor: int) -> None:
    if grad_scale_factor == 1:
        return
    for grad in grads:
        if isinstance(grad, torch.Tensor):
            grad.div_(grad_scale_factor)


def _scale_graph_pp_sharded_grads(
    stage: "GraphPipelineStage",
    schedule: _PipelineScheduleRuntime,
) -> None:
    if stage._graph_pp_grads_scaled:
        return
    grad_scale_factor = schedule._n_microbatches if schedule.scale_grads else 1
    _scale_grad_values_(stage.state.sharded_grads, grad_scale_factor)
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


def _require_graph_bundle(stage: "GraphPipelineStage", action_name: str) -> None:
    if stage.graph_callables is None or stage.graph_meta is None:
        raise ValueError(
            "GraphPP graph bundle must be built before runtime execution. "
            f"Missing graph bundle for stage {stage.stage_index} action {action_name}."
        )


class GraphPipelineStage(PipelineStage):
    def __init__(
        self,
        submodule: nn.Module,
        *,
        stage_index: int,
        num_stages: int,
        device: torch.device,
        loss_fn: Callable,
        compile_config: GraphTrainerCompileConfig | None = None,
        model_config: Any = None,
        parallelism: Any = None,
        input_args: Any = None,
        output_args: Any = None,
        group: torch.distributed.ProcessGroup | None = None,
        get_mesh: Callable | None = None,
    ) -> None:
        super().__init__(
            submodule,
            stage_index,
            num_stages,
            device,
            input_args=input_args,
            output_args=output_args,
            group=group,
            get_mesh=get_mesh,
        )
        self.loss_fn = loss_fn
        self.compile_config = compile_config or GraphTrainerCompileConfig()
        self.model_config = model_config
        self.parallelism = parallelism
        self.graph_callables: GraphCallables | None = None
        self._graph_pp_callables_compiled = False
        self.graph_meta: StageGraphMeta | None = None
        self.trace_spec = StageTraceSpec()
        self.state = GraphPPStageRuntimeState()
        self.bwd_activation_cache: dict[int, tuple[Any, ...]] = {}
        self._graph_pp_grads_scaled = False

    def _ensure_unsharded_params(self) -> None:
        assert self.graph_callables is not None and self.graph_meta is not None
        if self.state.unsharded_params:
            return
        if self.graph_callables.unshard is None:
            self.state.unsharded_params = list(self.state.sharded_params)
        else:
            self.state.unsharded_params = list(
                _execute_graph(
                    self.graph_callables.unshard,
                    [
                        self.state.sharded_params[source.index]
                        for source in self.graph_meta.unshard_input_sources
                    ],
                )
            )

    def _accumulate_stage_unsharded_grads(self, grads: list[Any]) -> None:
        _accumulate_flat_grad_values_(
            self.state.unsharded_grads,
            grads,
            label="unsharded",
        )

    def scale_grads(self, grad_scale_factor: int) -> None:
        grads = (
            self.state.sharded_grads
            if self.state.sharded_grads
            else self.state.unsharded_grads
        )
        _scale_grad_values_(grads, grad_scale_factor)


def _get_stage_from_action(
    action: _Action,
    ctx: _PipelineContext,
) -> tuple[_PipelineScheduleRuntime, dict[int, GraphPipelineStage], GraphPipelineStage]:
    schedule = ctx.schedule_ref
    assert isinstance(schedule, _PipelineScheduleRuntime)
    stage_index_to_stage = {
        stage.stage_index: cast(GraphPipelineStage, stage) for stage in schedule._stages
    }
    return schedule, stage_index_to_stage, stage_index_to_stage[action.stage_index]


def _prepare_fwd_common(action: _Action, ctx: _PipelineContext):
    schedule, stage_index_to_stage, stage = _get_stage_from_action(action, ctx)
    mb_index = action.microbatch_index
    assert mb_index is not None
    is_next_stage_on_this_rank = stage.stage_index + 1 in stage_index_to_stage
    is_prev_stage_on_this_rank = stage.stage_index - 1 in stage_index_to_stage
    if not stage.is_first and not is_prev_stage_on_this_rank:
        fwd_recv_ops = schedule.fwd_recv_ops
        assert (stage.stage_index, mb_index) in fwd_recv_ops
        _wait_batch_p2p(fwd_recv_ops.pop((stage.stage_index, mb_index)))
    return (
        schedule,
        stage_index_to_stage,
        stage,
        mb_index,
        is_next_stage_on_this_rank,
    )


def _prepare_fwd_user_args(
    stage: GraphPipelineStage,
    mb_index: int,
    ctx: _PipelineContext,
) -> tuple[tuple[Any, ...], dict[str, Any], Any]:
    arg_mbs = ctx.arg_mbs
    kwarg_mbs = ctx.kwarg_mbs
    assert arg_mbs is not None and kwarg_mbs is not None
    kwargs = kwarg_mbs[mb_index]
    if stage.is_first:
        args = arg_mbs[mb_index]
    else:
        args = _normalize_model_output_as_tuple(
            stage._retrieve_recv_activations(mb_index)
        )
    target = ctx.target_mbs[mb_index] if stage.is_last and ctx.target_mbs else None
    return tuple(args), kwargs, target


def _flatten_stage_inputs(
    stage: GraphPipelineStage,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    target: Any,
    loss_kwargs: dict[str, Any],
) -> list[Any]:
    state_flat, _ = pytree.tree_flatten(extract_module_state(stage.submod))
    if stage.is_last:
        flat_user_inputs, _ = pytree.tree_flatten(
            ((args, kwargs, target, loss_kwargs), {})
        )
    else:
        flat_user_inputs, _ = pytree.tree_flatten(((args, kwargs), {}))
    return flatten_graph_values([*state_flat, *flat_user_inputs])


def _prepare_fwd_args_from_sources(
    stage: GraphPipelineStage,
    fwd_input_sources: tuple[GraphPPInputSource, ...],
    flat_inputs: list[Any],
) -> list[Any]:
    assert stage.graph_meta is not None
    fw_args = []
    for source in fwd_input_sources:
        if source.kind == "flat_input":
            if source.index >= len(flat_inputs):
                raise ValueError(
                    "GraphPP forward placeholder index is out of range: "
                    f"{source.name} indexes {source.index}, but runtime has "
                    f"{len(flat_inputs)} flattened inputs"
                )
            fw_args.append(flat_inputs[source.index])
        elif source.kind == "unsharded_param":
            if source.index >= len(stage.state.unsharded_params):
                raise ValueError(
                    "GraphPP unsharded parameter index is out of range: "
                    f"{source.name} indexes {source.index}, but runtime has "
                    f"{len(stage.state.unsharded_params)} unsharded params"
                )
            fw_args.append(stage.state.unsharded_params[source.index])
        else:
            raise ValueError(f"Unknown GraphPP forward input source {source}")
    return fw_args


def _prepare_fwd_graph_args(
    stage: GraphPipelineStage,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    target: Any,
    loss_kwargs: dict[str, Any],
) -> list[Any]:
    stage._ensure_unsharded_params()
    return _prepare_fwd_args_from_sources(
        stage,
        stage.graph_meta.fwd_input_sources,
        _flatten_stage_inputs(stage, args, kwargs, target, loss_kwargs),
    )


def _post_fwd_common(
    stage: GraphPipelineStage,
    mb_index: int,
    output: Any,
    saved_intermediates: tuple[Any, ...],
    schedule: _PipelineScheduleRuntime,
    stage_index_to_stage: dict[int, GraphPipelineStage],
    ctx: _PipelineContext,
    is_next_stage_on_this_rank: bool,
) -> None:
    output_tuple = _normalize_model_output_as_tuple(output)
    if stage.is_last:
        stage.output_chunks.append(output)
        if ctx.losses is not None:
            ctx.losses.append(output)
        schedule._internal_losses.append(output)
    stage.fwd_cache[mb_index] = (output_tuple, saved_intermediates)
    if is_next_stage_on_this_rank:
        stage_index_to_stage[stage.stage_index + 1].set_local_fwd_input(
            output, mb_index
        )


def stage_forward(action: _Action, ctx: _PipelineContext) -> None:
    (
        schedule,
        stage_index_to_stage,
        stage,
        mb_index,
        is_next_stage_on_this_rank,
    ) = _prepare_fwd_common(action, ctx)
    args, kwargs, target = _prepare_fwd_user_args(stage, mb_index, ctx)
    loss_kwargs = getattr(schedule, "_graph_pp_loss_kwargs", {})
    _require_graph_bundle(stage, "FORWARD")
    assert stage.graph_callables is not None and stage.graph_meta is not None
    output, saved_intermediates = _run_fw_module(
        stage.graph_callables.fw,
        stage.graph_meta,
        _prepare_fwd_graph_args(stage, args, kwargs, target, loss_kwargs),
    )
    _post_fwd_common(
        stage,
        mb_index,
        output,
        saved_intermediates,
        schedule,
        stage_index_to_stage,
        ctx,
        is_next_stage_on_this_rank,
    )


def _prepare_backward_common(action: _Action, ctx: _PipelineContext):
    schedule, stage_index_to_stage, stage = _get_stage_from_action(action, ctx)
    mb_index = action.microbatch_index
    assert mb_index is not None
    is_next_stage_on_this_rank = stage.stage_index + 1 in stage_index_to_stage
    is_prev_stage_on_this_rank = stage.stage_index - 1 in stage_index_to_stage
    if not stage.is_last and not is_next_stage_on_this_rank:
        bwd_recv_ops = schedule.bwd_recv_ops
        assert (stage.stage_index, mb_index) in bwd_recv_ops
        _wait_batch_p2p(bwd_recv_ops.pop((stage.stage_index, mb_index)))
    schedule.backward_counter[stage.stage_index] += 1
    return schedule, stage_index_to_stage, stage, mb_index, is_prev_stage_on_this_rank


def _prepare_backward_args(stage: GraphPipelineStage, mb_index: int) -> list[Any]:
    stage_output, saved_intermediates = stage.fwd_cache.pop(mb_index)
    assert stage.graph_meta is not None
    if stage.is_last:
        if len(stage_output) != 1:
            raise ValueError(
                "GraphPP last stage backward expects the traced forward graph to "
                f"return one loss tensor, got {len(stage_output)} outputs."
            )
        runtime_bwd_inputs = (torch.ones_like(stage_output[0]),)
    else:
        runtime_bwd_inputs = _normalize_model_output_as_tuple(
            stage._retrieve_recv_grads(mb_index)
        )
    raw_runtime_bwd_inputs = flatten_graph_values(list(runtime_bwd_inputs))
    saved_by_name = dict(
        zip(
            stage.graph_meta.partition.saved_for_backward_names,
            saved_intermediates,
            strict=True,
        )
    )
    runtime_by_name = dict(
        zip(
            stage.graph_meta.partition.bwd_runtime_input_names,
            [
                raw_runtime_bwd_inputs[index]
                for index in stage.graph_meta.partition.bwd_runtime_input_indices
            ],
            strict=True,
        )
    )
    bwd_args = []
    for name in stage.graph_meta.partition.bwd_input_names:
        if name in saved_by_name:
            bwd_args.append(saved_by_name[name])
        elif name in runtime_by_name:
            bwd_args.append(runtime_by_name[name])
        else:
            raise ValueError(f"Missing backward input {name}")
    return bwd_args


def _post_backward_common(
    stage: GraphPipelineStage,
    mb_index: int,
    input_grads: list[Any],
    stage_index_to_stage: dict[int, GraphPipelineStage],
    is_prev_stage_on_this_rank: bool,
) -> None:
    stage.bwd_cache[mb_index] = tuple(input_grads)
    if is_prev_stage_on_this_rank:
        stage_index_to_stage[stage.stage_index - 1].set_local_bwd_input(
            stage.get_local_bwd_output(mb_index),
            mb_index,
        )


def stage_full_backward(action: _Action, ctx: _PipelineContext) -> None:
    (
        _schedule,
        stage_index_to_stage,
        stage,
        mb_index,
        is_prev_stage_on_this_rank,
    ) = _prepare_backward_common(action, ctx)
    if not stage.has_backward:
        return
    assert stage.graph_callables is not None and stage.graph_meta is not None
    input_grads, param_grads = _run_full_bw_module(
        stage.graph_callables.full_bw,
        stage.graph_meta,
        _prepare_backward_args(stage, mb_index),
    )
    stage._accumulate_stage_unsharded_grads(param_grads)
    _post_backward_common(
        stage,
        mb_index,
        input_grads,
        stage_index_to_stage,
        is_prev_stage_on_this_rank,
    )


def stage_backward_input(action: _Action, ctx: _PipelineContext) -> None:
    _, _, stage = _get_stage_from_action(action, ctx)
    assert stage.graph_callables is not None
    if stage.graph_callables.bw_di is None:
        logger.debug("GraphPP skipping BACKWARD_INPUT for stage %s", stage.stage_index)
        return
    (
        schedule,
        stage_index_to_stage,
        stage,
        mb_index,
        is_prev_stage_on_this_rank,
    ) = _prepare_backward_common(action, ctx)
    if not stage.has_backward:
        return
    assert stage.graph_meta is not None and stage.graph_callables.bw_di is not None
    input_grads, activations = _run_di_bw_module(
        stage.graph_callables.bw_di,
        stage.graph_meta,
        _prepare_backward_args(stage, mb_index),
    )
    stage.bwd_activation_cache[mb_index] = tuple(activations)
    _post_backward_common(
        stage,
        mb_index,
        input_grads,
        stage_index_to_stage,
        is_prev_stage_on_this_rank,
    )


def stage_backward_weight(action: _Action, ctx: _PipelineContext) -> None:
    _schedule, _, stage = _get_stage_from_action(action, ctx)
    mb_index = action.microbatch_index
    assert mb_index is not None
    assert stage.graph_callables is not None and stage.graph_meta is not None
    if stage.graph_callables.bw_dw is None:
        new_action = _Action(
            action.stage_index,
            FULL_BACKWARD,
            action.microbatch_index,
            action.sub_actions,
        )
        stage_full_backward(new_action, ctx)
        return
    if not stage.has_backward:
        return
    activations = stage.bwd_activation_cache.pop(mb_index)
    param_grads = _run_dw_bw_module(
        stage.graph_callables.bw_dw,
        stage.graph_meta,
        list(activations),
    )
    stage._accumulate_stage_unsharded_grads(param_grads)


def stage_unshard(action: _Action, ctx: _PipelineContext) -> None:
    _, _, stage = _get_stage_from_action(action, ctx)
    _require_graph_bundle(stage, "UNSHARD")
    stage._ensure_unsharded_params()


def stage_reshard(action: _Action, ctx: _PipelineContext) -> None:
    _, _, stage = _get_stage_from_action(action, ctx)
    _require_graph_bundle(stage, "RESHARD")
    stage.state.unsharded_params = []


def stage_reduce_grad(action: _Action, ctx: _PipelineContext) -> None:
    schedule, _, stage = _get_stage_from_action(action, ctx)
    _require_graph_bundle(stage, "REDUCE_GRAD")
    assert stage.graph_callables is not None and stage.graph_meta is not None
    if stage.graph_callables.reduce_grad is None:
        stage.state.sharded_grads = _raw_unsharded_grads(stage)
    else:
        stage.state.sharded_grads = list(
            _execute_graph(
                stage.graph_callables.reduce_grad,
                _prepare_reduce_grad_args(stage),
            )
        )
    _scale_graph_pp_sharded_grads(stage, schedule)


def get_multiplexed_graph_callables(
    stage_graphs: dict[int, GraphCallables],
    multiplex_fw_bw_graph_pass: MultiplexFwBwGraphPass,
) -> dict[tuple[int, int], fx.GraphModule]:
    multiplexed = {}
    for bw_stage_idx, bw_graphs in stage_graphs.items():
        for fw_stage_idx, fw_graphs in stage_graphs.items():
            if bw_stage_idx == fw_stage_idx:
                continue
            multiplexed[(fw_stage_idx, bw_stage_idx)] = multiplex_fw_bw_graph_pass(
                fw_graphs.fw,
                bw_graphs.full_bw,
            )
    return multiplexed


def _example_args_from_stage_metadata(stage: GraphPipelineStage) -> tuple[Any, ...]:
    # NOTE: See _example_output_grads_from_stage_metadata for the private
    # PipelineStage metadata dependency.
    if stage._stage_meta.inputs is None:
        raise ValueError(
            "GraphPP stage metadata was not initialized before graph construction: "
            f"stage {stage.stage_index}"
        )
    return tuple(stage._to_tensor(meta) for meta in stage._stage_meta.inputs)


def _overlap_fw_bw_sub_actions(action: _Action) -> tuple[_Action, _Action]:
    if action.sub_actions is None or len(action.sub_actions) != 2:
        raise ValueError(f"GraphPP OVERLAP_F_B action is malformed: {action}")
    fw_action, bw_action = action.sub_actions
    if fw_action.computation_type != FORWARD:
        raise ValueError(f"GraphPP OVERLAP_F_B first sub-action must be F: {action}")
    if bw_action.computation_type == BACKWARD_INPUT:
        raise NotImplementedError(
            "GraphPP OVERLAP_F_B with BACKWARD_INPUT is not implemented. "
            "Current multiplexed graphs support FORWARD + FULL_BACKWARD only."
        )
    if bw_action.computation_type != FULL_BACKWARD:
        raise ValueError(
            "GraphPP OVERLAP_F_B second sub-action must be FULL_BACKWARD: " f"{action}"
        )
    return fw_action, bw_action


def _required_multiplex_pairs(
    schedule: _PipelineScheduleRuntime,
) -> set[tuple[int, int]]:
    pipeline_order = getattr(schedule, "pipeline_order_with_comms", None)
    if pipeline_order is None:
        return set()
    required_pairs: set[tuple[int, int]] = set()
    for action in pipeline_order.get(schedule.rank, []):
        if action.computation_type != OVERLAP_F_B:
            continue
        fw_action, bw_action = _overlap_fw_bw_sub_actions(action)
        required_pairs.add((fw_action.stage_index, bw_action.stage_index))
    return required_pairs


def _build_graph_pp_multiplexed_graph_bundles(
    schedule: _PipelineScheduleRuntime,
) -> None:
    stage_index_to_stage = {
        stage.stage_index: cast(GraphPipelineStage, stage) for stage in schedule._stages
    }
    # The upstream runtime schedule has no typed extension slot for custom
    # callable caches, so GraphPP keeps multiplexed graphs on a private field.
    multiplexed_graphs = getattr(schedule, "_graph_pp_multiplexed_graphs", {})
    for fw_stage_idx, bw_stage_idx in _required_multiplex_pairs(schedule):
        pair = (fw_stage_idx, bw_stage_idx)
        if pair in multiplexed_graphs:
            continue
        fw_stage = stage_index_to_stage[fw_stage_idx]
        bw_stage = stage_index_to_stage[bw_stage_idx]
        _require_graph_bundle(fw_stage, "OVERLAP_F_B")
        _require_graph_bundle(bw_stage, "OVERLAP_F_B")
        if fw_stage.compile_config != bw_stage.compile_config:
            raise ValueError(
                "GraphPP multiplexed graph requires matching compile configs for "
                f"forward stage {fw_stage_idx} and backward stage {bw_stage_idx}."
            )
        if (
            fw_stage._graph_pp_callables_compiled
            or bw_stage._graph_pp_callables_compiled
        ):
            raise ValueError(
                "GraphPP multiplexed graphs must be built before stage callables "
                "are compiled."
            )
        fw_graphs = fw_stage.graph_callables
        bw_graphs = bw_stage.graph_callables
        assert fw_graphs is not None and bw_graphs is not None
        multiplexed_graph = multiplex_fw_bw_graph(
            fw_graphs.fw,
            bw_graphs.full_bw,
        )
        _annotate_graph_pp_graph(
            multiplexed_graph,
            stage_index=fw_stage_idx,
            callable_name="multiplex",
            action_name="OVERLAP_F_B",
        )
        compiled_graph = _compile_graph_pp_module(
            multiplexed_graph,
            compile_config=fw_stage.compile_config,
            graph_name=f"stage_{fw_stage_idx}_fw_stage_{bw_stage_idx}_bw_multiplex",
        )
        assert compiled_graph is not None
        multiplexed_graphs[pair] = compiled_graph
    schedule._graph_pp_multiplexed_graphs = multiplexed_graphs


def build_graph_pp_graph_bundles(
    schedule: _PipelineScheduleRuntime,
    *args: Any,
    target: Any = None,
    loss_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    """Build all stage-local GraphPP callables before schedule execution.

    Contract:
      This function is the only tracing/partitioning/compilation entrypoint for
      GraphPP training. It runs before ``GraphPPRunner.step`` so the runner can
      remain a pure PP action execution engine.

    Pseudocode:
      split the full batch with the upstream PP microbatch splitter
      normalize trace-reused BlockMask inputs
      run eager PP metadata inference on representative microbatch 0
      build missing stage graph bundles
      build required multiplexed graphs for OVERLAP_F_B actions
    """
    graph_stages = [cast(GraphPipelineStage, stage) for stage in schedule._stages]
    if all(stage.graph_callables is not None for stage in graph_stages):
        _build_graph_pp_multiplexed_graph_bundles(schedule)
        for stage in graph_stages:
            _compile_stage_graph_bundle(stage)
        return

    maybe_register_blockmask_pytree_node()
    args_split, kwargs_split = schedule._split_inputs(args, kwargs)
    args_split, kwargs_split = normalize_graph_pp_microbatch_inputs(
        args_split,
        kwargs_split,
    )
    targets_split = (
        list(_split_tensor(target, _TARGET_CHUNK_SPEC, schedule._n_microbatches))
        if target is not None
        else None
    )
    arg_mbs, kwarg_mbs = schedule._check_inputs(
        args_split,
        kwargs_split,
        targets_split,
        None,
    )
    maybe_first_target = targets_split[0] if targets_split is not None else None
    loss_kwargs = loss_kwargs or {}
    original_loss_fn = schedule._loss_fn
    # TODO(sanketpurandare): requires upstream change: PP metadata
    # initialization should accept the loss function explicitly instead of
    # reading schedule._loss_fn.
    schedule._loss_fn = graph_stages[0].loss_fn
    try:
        schedule._initialize_stages(
            arg_mbs[0],
            kwarg_mbs[0],
            maybe_first_target,
            loss_kwargs,
        )
    finally:
        schedule._loss_fn = original_loss_fn

    for stage in graph_stages:
        if stage.graph_callables is not None:
            continue
        stage_args = (
            tuple(arg_mbs[0])
            if stage.is_first
            else _example_args_from_stage_metadata(stage)
        )
        stage_target = maybe_first_target if stage.is_last else None
        _build_stage_graph_bundle(
            stage,
            stage_args,
            kwarg_mbs[0],
            stage_target,
            loss_kwargs,
            compile_callables=False,
        )
    _build_graph_pp_multiplexed_graph_bundles(schedule)
    for stage in graph_stages:
        _compile_stage_graph_bundle(stage)


def overlap_fw_bw(
    action: _Action,
    ctx: _PipelineContext,
) -> None:
    fw_action, bw_action = _overlap_fw_bw_sub_actions(action)

    (
        schedule,
        stage_index_to_stage,
        fw_stage,
        fw_mb_index,
        fw_is_next_stage_on_this_rank,
    ) = _prepare_fwd_common(fw_action, ctx)
    (
        _,
        _,
        bw_stage,
        bw_mb_index,
        bw_is_prev_stage_on_this_rank,
    ) = _prepare_backward_common(bw_action, ctx)
    if not bw_stage.has_backward:
        return

    args, kwargs, target = _prepare_fwd_user_args(fw_stage, fw_mb_index, ctx)
    loss_kwargs = getattr(schedule, "_graph_pp_loss_kwargs", {})
    _require_graph_bundle(fw_stage, "OVERLAP_F_B")
    _require_graph_bundle(bw_stage, "OVERLAP_F_B")
    assert fw_stage.graph_callables is not None and fw_stage.graph_meta is not None
    assert bw_stage.graph_callables is not None and bw_stage.graph_meta is not None

    multiplexed_graphs = getattr(schedule, "_graph_pp_multiplexed_graphs", None)
    pair = (fw_action.stage_index, bw_action.stage_index)
    bw_args = _prepare_backward_args(bw_stage, bw_mb_index)
    fw_args = _prepare_fwd_graph_args(fw_stage, args, kwargs, target, loss_kwargs)
    if multiplexed_graphs is None or pair not in multiplexed_graphs:
        raise ValueError(
            "GraphPP multiplexed graph must be built before OVERLAP_F_B runtime "
            f"execution for pair {pair}."
        )

    multiplexed_outputs = _execute_graph(
        multiplexed_graphs[pair],
        [*bw_args, *fw_args],
    )

    num_param_grads = bw_stage.graph_meta.num_param_grad_values
    num_bw_outputs = num_param_grads + bw_stage.graph_meta.num_input_grad_values
    bw_outputs = multiplexed_outputs[:num_bw_outputs]
    param_grads = list(bw_outputs[:num_param_grads])
    input_grads = bw_stage.graph_meta.input_grad_values.wrap_flat_values(
        bw_outputs[num_param_grads:]
    )
    fw_outputs = multiplexed_outputs[num_bw_outputs:]
    output = fw_stage.graph_meta.fwd_output_values.unflatten(
        fw_outputs[: fw_stage.graph_meta.num_user_outputs],
    )
    saved_start = fw_stage.graph_meta.num_user_outputs
    saved_end = saved_start + fw_stage.graph_meta.num_saved_for_backward
    saved_intermediates = tuple(fw_outputs[saved_start:saved_end])

    bw_stage._accumulate_stage_unsharded_grads(param_grads)
    _post_fwd_common(
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


class GraphPPRunner:
    def __init__(self, schedule: _PipelineScheduleRuntime) -> None:
        self.schedule = schedule
        self.schedule._has_backward = True
        for stage in schedule._stages:
            if not isinstance(stage, GraphPipelineStage):
                raise TypeError(
                    "GraphPPRunner requires GraphPipelineStage instances, got "
                    f"{type(stage).__name__}"
                )

    def _require_prebuilt_graph_bundles(self) -> None:
        for stage in self.schedule._stages:
            _require_graph_bundle(cast(GraphPipelineStage, stage), "step")

    def _populate_stage_states(self, stage: GraphPipelineStage) -> None:
        sharded_params = []
        trainable_params = []
        for _, value in stage.submod.named_parameters(remove_duplicate=False):
            sharded_params.extend(flatten_graph_values([value]))
            if value.requires_grad:
                trainable_params.append(value)
        buffer_values = [
            value for _, value in stage.submod.named_buffers(remove_duplicate=False)
        ]
        buffers = flatten_graph_values(buffer_values)
        stage.state.sharded_params = sharded_params
        stage.state.buffers = buffers
        stage.state.trainable_params = trainable_params
        graph_meta = stage.graph_meta
        num_unsharded_grads = (
            graph_meta.num_param_grad_values
            if graph_meta is not None
            else len(trainable_params)
        )
        stage.state.unsharded_grads = [None] * num_unsharded_grads
        stage.state.sharded_grads = []
        stage._graph_pp_grads_scaled = False

    def _ensure_reduced_grads(self, stage: GraphPipelineStage) -> None:
        if stage.state.sharded_grads:
            return
        if not any(grad is not None for grad in stage.state.unsharded_grads):
            return
        if stage.graph_callables is None:
            return
        if stage.graph_callables.reduce_grad is None:
            if stage.graph_meta is None:
                stage.state.sharded_grads = list(stage.state.unsharded_grads)
            else:
                stage.state.sharded_grads = _raw_unsharded_grads(stage)
        else:
            if stage.graph_meta is None:
                raise ValueError("GraphPP reduce-grad graph requires graph metadata")
            stage.state.sharded_grads = list(
                _execute_graph(
                    stage.graph_callables.reduce_grad,
                    _prepare_reduce_grad_args(stage),
                )
            )
        _scale_graph_pp_sharded_grads(stage, self.schedule)

    def _accumulate_stage_sharded_grads(self, stage: GraphPipelineStage) -> None:
        self._ensure_reduced_grads(stage)
        if not stage.state.sharded_grads:
            return
        if stage.graph_meta is None:
            param_grads = list(stage.state.sharded_grads)
        else:
            param_grads = stage.graph_meta.param_grad_values.wrap_flat_values(
                stage.state.sharded_grads
            )
        accumulate_param_grads_(stage.state.trainable_params, param_grads)

    def step(self, *args, **kwargs) -> None:
        self._require_prebuilt_graph_bundles()
        loss_kwargs = kwargs.get("loss_kwargs") or {}
        self.schedule._graph_pp_loss_kwargs = loss_kwargs
        for stage in self.schedule._stages:
            self._populate_stage_states(cast(GraphPipelineStage, stage))
        original_split_inputs = self.schedule._split_inputs

        def graph_pp_split_inputs(args, kwargs=None):
            args_split, kwargs_split = original_split_inputs(args, kwargs)
            return normalize_graph_pp_microbatch_inputs(args_split, kwargs_split)

        # TODO(sanketpurandare): requires upstream change: runtime PP schedules
        # should expose a microbatch input normalization hook used by both
        # tracing and execution.
        self.schedule._split_inputs = graph_pp_split_inputs  # type: ignore[method-assign]
        try:
            self.schedule.step(*args, **kwargs)
        finally:
            self.schedule._split_inputs = original_split_inputs  # type: ignore[method-assign]
        for stage in self.schedule._stages:
            graph_stage = cast(GraphPipelineStage, stage)
            self._accumulate_stage_sharded_grads(graph_stage)
            graph_stage.state.clear()

    def eval(self, *args, **kwargs):
        return self.schedule.eval(*args, **kwargs)


def register_graph_pp_schedule(schedule: _PipelineScheduleRuntime) -> GraphPPRunner:
    schedule.register_custom_function(FORWARD, stage_forward)
    schedule.register_custom_function(FULL_BACKWARD, stage_full_backward)
    schedule.register_custom_function(UNSHARD, stage_unshard)
    schedule.register_custom_function(RESHARD, stage_reshard)
    schedule.register_custom_function(REDUCE_GRAD, stage_reduce_grad)
    schedule.register_custom_function(BACKWARD_INPUT, stage_backward_input)
    schedule.register_custom_function(BACKWARD_WEIGHT, stage_backward_weight)
    schedule.register_custom_function(OVERLAP_F_B, overlap_fw_bw)
    return GraphPPRunner(schedule)
