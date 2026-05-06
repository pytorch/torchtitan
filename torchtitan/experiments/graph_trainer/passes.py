# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Compiler passes for graph_trainer training.

This module provides various compiler passes that can be applied to graph modules
during compilation. Passes can be selected and configured via job config.

Pass Types:
- Joint custom passes: Applied to the joint forward-backward graph before partitioning
- Compiler passes: Applied to the partitioned forward/backward graphs
"""

from __future__ import annotations

import functools
import operator
import time
from collections import defaultdict
from collections.abc import Callable

import torch
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.fx_passes.bucketing import (
    is_all_gather_into_tensor as is_all_gather,
)
from torch._inductor.fx_passes.overlap_manual_scheduling import manual_overlap_bucketing
from torch._inductor.fx_passes.overlap_scheduling import schedule_overlap_bucketing
from torch._logging import trace_structured
from torch.fx.passes.regional_inductor import regional_inductor
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.distributed.activation_checkpoint import _get_save_ops
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.experiments.graph_trainer.bucketing import (
    joint_transformer_block_bucketing_reordering_pass,
)
from torchtitan.experiments.graph_trainer.common_utils import (
    _get_layer_id,
    _is_backward_node,
    _NOT_IN_LAYERS,
)
from torchtitan.experiments.graph_trainer.cpu_offload import (
    apply_cpu_offload_pass,
    tag_all_offloadable_activations,
)
from torchtitan.experiments.graph_trainer.custom_codegen import custom_codegen_pass
from torchtitan.experiments.graph_trainer.debug_utils import (
    log_graph_diff,
    snapshot_graph,
)
from torchtitan.experiments.graph_trainer.log_activation_memory_policy import (
    log_activation_memory_policy,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import TracedResult
from torchtitan.experiments.graph_trainer.remove_noop_passes import (
    remove_detach_pass,
    remove_identity_slice_pass,
    remove_identity_view_pass,
)
from torchtitan.experiments.graph_trainer.reshard_after_forward import (
    annotate_fsdp_all_gather,
    is_wait_tensor_from_fsdp,
)
from torchtitan.tools.logging import logger

aten = torch.ops.aten
c10d = torch.ops._c10d_functional


def normalize_view_ops_as_reshape(
    gm: torch.fx.GraphModule,
    example_inputs=None,
) -> torch.fx.GraphModule:
    """Replace aten.view and aten._unsafe_view with aten.reshape.

    Downstream passes expect aten.reshape.default for pattern matching.
    """
    view_targets = {aten.view.default, aten._unsafe_view.default}
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in view_targets:
            node.target = aten.reshape.default
    gm.graph.lint()
    gm.recompile()
    return gm


def async_tensor_parallel_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
) -> torch.fx.GraphModule:
    """Pipeline TP collectives with matmuls via symmetric memory.

    Fuses all-gather + matmul into ``symm_mem.fused_all_gather_matmul``
    and matmul + reduce-scatter into
    ``symm_mem.fused_matmul_reduce_scatter``.
    """
    import warnings

    from torch._inductor.fx_passes.micro_pipeline_tp import micro_pipeline_tp_pass
    from torch._inductor.fx_passes.overlap_scheduling import get_group_name
    from torch.distributed._symmetric_memory import enable_symm_mem_for_group

    # Ensure symmetric memory is registered for every collective PG in
    # the graph.  The upstream API is deprecated but the auto-registration
    # it promises has not landed yet, so the explicit call is still needed.
    collective_targets = {
        c10d.all_gather_into_tensor.default,
        c10d.reduce_scatter_tensor.default,
    }
    registered: set[str] = set()
    for node in gm.graph.nodes:
        if node.target not in collective_targets:
            continue
        pg = get_group_name(node)
        if pg not in registered:
            registered.add(pg)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                enable_symm_mem_for_group(pg)

    micro_pipeline_tp_pass(gm.graph)
    gm.graph.lint()
    gm.recompile()
    return gm


def compile_time_passes(
    traced_result: "TracedResult",
    config: "GraphTrainer.Config",
    *,
    use_cudagraph: bool = False,
) -> list[Callable]:
    """Cleanup, FlexAttention annotation, and regional_inductor passes.

    If precompile is enabled, these are applied before serialization so
    that compiled Triton kernels are baked into the artifact. Otherwise
    they run at trace time via ``construct_default_graph_passes``.

    cudagraph is excluded because it needs to re-capture the graph into
    an in-memory CUDA graph at runtime.

    ``overlap_fsdp_ag_rs_pass`` runs immediately before
    ``joint_transformer_block_bucketing_reordering_pass`` so that
    forward+backward all-gathers end up on a separate CUDA stream from
    reduce-scatters (enabling AG/RS overlap in backward). It is a no-op
    when the graph contains no FSDP all-gathers.
    """
    from torchtitan.experiments.graph_trainer.common_utils import (
        get_default_transformer_block_buckets,
    )

    from torchtitan.experiments.graph_trainer.preserve_ops_order import (
        preserve_ops_order,
    )
    from torchtitan.models.common.attention import FlexAttention

    n_layers = len(config.model_spec.model.layers)
    passes: list[Callable] = [
        remove_detach_pass,
        remove_identity_view_pass,
        remove_identity_slice_pass,
        normalize_view_ops_as_reshape,
        functools.partial(
            tag_with_memory_policy_pass,
            config=config,
        ),
        # TODO: currently either SAC or CPU offload is used, not both at the
        # same time. Composability between these two passes is untested.
        apply_cpu_offload_pass,
        selective_activation_remat_pass,
        overlap_fsdp_ag_rs_pass,
        # The bucketing pass topologically reorders ops to overlap collectives
        # with compute, but doesn't track tensor aliasing/mutation deps.
        # ChunkedCELoss's per-chunk in-place writes into a grad accumulator
        # buffer (and the downstream readers of that buffer) live under
        # ``module_fqn`` ``loss``/``lm_head``; without pinning their order,
        # the bucketing pass moves the readers past the writes and decoder
        # grads come out all-zero. ``preserve_ops_order`` chains the loss
        # region through ``control_deps`` HOPs so the bucketing pass cannot
        # break the read-after-write ordering.
        preserve_ops_order(["loss", "lm_head"])(
            functools.partial(
                joint_transformer_block_bucketing_reordering_pass,
                module_bucket_plans=get_default_transformer_block_buckets(n_layers),
            )
        ),
    ]
    if config.parallelism.enable_async_tensor_parallel:
        passes.append(async_tensor_parallel_pass)

    inductor_compilation = config.compile.inductor_compilation
    if inductor_compilation == "full":
        # Compile the entire graph into optimized Triton kernels. Must
        # be terminal — the FX graph is no longer authoritative after
        # this pass, so custom_codegen_pass and
        # insert_kernel_annotations_pass cannot follow.
        passes.append(full_inductor_compilation_pass)
    if inductor_compilation == "regional":
        # FlexAttention HOPs must be compiled (via regional_inductor) to
        # produce bitwise identical results to the eager Trainer path.
        # When left uncompiled, flex_attention still runs correctly but
        # produces different numerical results.
        passes.append(
            functools.partial(
                annotate_flex_attention_for_regional_inductor_pass,
                flex_compile_config=FlexAttention.inductor_configs,
            )
        )
        # Performance passes that may change numerics.
        if config.compile.numerics_changing_optim:
            from torchtitan.experiments.graph_trainer.performance_passes import (
                annotate_rmsnorm_for_regional_inductor_pass,
            )

            passes.append(annotate_rmsnorm_for_regional_inductor_pass)
        passes.append(regional_inductor_pass)
        if use_cudagraph:
            # Must run before custom_codegen_pass (last in pre_passes)
            # which replaces the GraphModule's forward().
            # Also must run before cudagraph_pass.
            passes.append(insert_kernel_annotations_pass)
        # TODO: Switch to upstream PyTorch implementation when
        # https://github.com/pytorch/pytorch/pull/178246 lands.
        # custom_codegen_pass saves the FX graph to disk for:
        # 1. Debugging: inspect the generated graph code directly
        # 2. Profiling provenance: dual-path codegen with _RecordFunctionFast
        #    gives fine-grained operator-level attribution in profiler traces
        # 3. User-editable codegen: users can directly modify the generated
        #    program on disk for fine-grain scheduling optimizations, with
        #    hot-reload picking up changes at runtime
        passes.append(custom_codegen_pass)
    return passes


def construct_default_graph_passes(
    traced_result: "TracedResult",
    config: "GraphTrainer.Config",
) -> list[Callable]:
    """Build the pass list for the aot_fx_trace path.

    When ``precompile_artifact_dir`` is unset, returns the full list: cleanup,
    FlexAttention annotation, regional_inductor, and cudagraph.

    When ``precompile_artifact_dir`` is set, the artifact has graph
    transformed during precompile phase, so only cudagraph is returned.
    """
    from torchtitan.experiments.graph_trainer.cudagraph import is_cudagraph_compatible

    use_cudagraph = config.compile.enable_cudagraph and is_cudagraph_compatible(
        traced_result.gm
    )

    has_precompile_artifact = bool(config.compile.precompile_artifact_dir)

    passes: list[Callable] = []
    if not has_precompile_artifact:
        passes.extend(
            compile_time_passes(traced_result, config, use_cudagraph=use_cudagraph)
        )

    # cudagraph should be the last pass.
    if use_cudagraph:
        static_input_indices = list(range(traced_result.num_static_inputs))
        passes.append(
            functools.partial(
                cudagraph_pass,
                is_forward=True,
                static_input_indices=static_input_indices,
                tensor_input_indices=traced_result.tensor_input_indices,
            )
        )
    return passes


def apply_graph_passes(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
    passes: list[Callable],
    *,
    compile_config: "GraphTrainerCompileConfig | None" = None,
) -> torch.fx.GraphModule:
    """Apply graph passes to the traced fwd+bwd graph.

    Args:
        gm: The traced forward+backward graph module.
        example_inputs: Example (fake) inputs matching the graph signature.
        passes: Ordered list of pass callables, each with signature
            ``(gm, example_inputs, **kwargs) -> gm``.
        compile_config: Optional compile config. When provided and
            ``debug_graph_passes`` is True, logs timing, op-count diffs,
            and before/after graphs to tlparse for each pass.
    """
    debug = compile_config is not None and compile_config.debug_graph_passes
    tlparse_log_graph_pass(gm, graph_name="make_fx_graph_traced", debug=debug)
    for pass_fn in passes:
        pass_name = (
            pass_fn.func.__name__
            if isinstance(pass_fn, functools.partial)
            else pass_fn.__name__
        )
        if debug:
            tlparse_log_graph_pass(gm, graph_name=f"before_{pass_name}", debug=debug)
            before_snapshot = snapshot_graph(gm)
            start = time.perf_counter()
        gm = pass_fn(gm, example_inputs)
        assert isinstance(
            gm, torch.fx.GraphModule
        ), f"Pass {pass_name} returned {type(gm).__name__}, expected GraphModule"
        if debug:
            elapsed = time.perf_counter() - start
            logger.info(f"Pass {pass_name} took {elapsed:.3f}s")
            tlparse_log_graph_pass(gm, graph_name=f"after_{pass_name}", debug=debug)
            after_snapshot = snapshot_graph(gm)
            log_graph_diff(before_snapshot, after_snapshot, pass_name)
    return gm


def autobucketing_reordering_pass(
    gm: torch.fx.GraphModule, example_inputs: tuple | None = None
) -> torch.fx.GraphModule:
    """
    Apply autobucketing and reordering optimization.

    This pass applies schedule_overlap_bucketing with collective_bucketing enabled
    to optimize comm/compute overlap patterns in the graph.
    """
    schedule_overlap_bucketing(gm, collective_bucketing=True)
    gm.recompile()
    return gm


def transformer_block_bucketing_reordering_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    fsdp_manual_buckets,
) -> torch.fx.GraphModule:
    """
    Apply aten-level manual bucketing and reordering optimization.
    """
    manual_overlap_bucketing(
        gm, module_bucket_plans=fsdp_manual_buckets, insert_overlap_deps=False
    )
    gm.recompile()
    return gm


def _ops_filter_with_distributed(name: str) -> bool:
    """Ops filter that allows distributed collective ops for serialization.

    The default GraphPickler ops filter only allows aten and fbgemm ops.
    SimpleFSDP uses _c10d_functional collectives that must also be
    allowed for the graph to serialize correctly.  The device_mesh ops
    (e.g. _get_submesh) appear in the backward graph when DTensor
    reconstructs submeshes from tracked ancestor meshes.
    """
    return name.startswith(
        (
            "torch.ops.aten",
            "torch.ops.fbgemm",
            "torch.ops._c10d_functional",
            "torch.ops._dtensor",
            "torch.ops.device_mesh",
        )
    )


def _node_metadata_key_filter_distributed(key: str) -> bool:
    """Metadata key filter for regional_inductor with distributed ops.

    Distributed ops (e.g. _get_submesh, mesh_get_process_group) produce
    opaque values (DeviceMesh, ProcessGroup) in node.meta["val"] and
    node.meta["eager_input_vals"] that cannot be pickled.  We strip
    both — they are not needed at runtime.
    """
    if key in ("val", "eager_input_vals"):
        return False
    return key not in ["source_fn_stack", "nn_module_stack", "fwd_source_fn_stack"]


def regional_inductor_pass(
    gm: torch.fx.GraphModule, example_inputs: tuple, *, serializable: bool = False
) -> torch.fx.GraphModule:
    """Compile tagged graph regions with ``regional_inductor``.

    Scans the graph for nodes whose ``node.meta["custom"]`` contains a
    ``compile_with_inductor`` key and compiles those regions with
    TorchInductor.  Nodes without this tag are left unchanged.  If no
    nodes are tagged the pass is a no-op.

    Inductor is configured for bitwise-equal numerics so that the
    compiled regions match eager execution exactly.

    Args:
        gm: The graph module to compile.
        example_inputs: Example inputs for shape propagation.
        serializable: When True (precompile mode), sets
            ``force_autograd_cache`` so that ``regional_inductor`` wraps
            its output in ``RegionalOutputCode``, and overrides the ops
            filter to allow distributed collective ops.
    """
    import torch._inductor.config as ic
    from torch._subclasses.fake_tensor import FakeTensor

    def _get_fake_mode_from_gm(gm: torch.fx.GraphModule):
        """Extract the FakeTensorMode from a graph module's placeholder metadata."""
        for node in gm.graph.nodes:
            if node.op == "placeholder" and "val" in node.meta:
                val = node.meta["val"]
                if isinstance(val, FakeTensor):
                    return val.fake_mode
        return None

    # Ensure inductor produces bitwise-equal numerics vs eager.
    ic.eager_numerics.division_rounding = True
    # Recommended by inductor team — uncomment as needed:
    # ic.emulate_precision_casts = True
    # ic.eager_numerics.disable_ftz = True
    # ic.eager_numerics.use_pytorch_libdevice = True
    # ic.fallback_random = True

    # regional_inductor calls standalone_compile with
    # dynamic_shapes="from_tracing_context", which requires an active
    # TracingContext with a FakeTensorMode.  When this pass is called
    # outside torch.compile (e.g. after make_fx tracing in graph_trainer),
    # no TracingContext exists, so we create one from the graph's fake
    # tensor metadata.
    fake_mode = _get_fake_mode_from_gm(gm)
    tracing_ctx = torch._guards.TracingContext(fake_mode)

    if serializable:
        with (
            torch._guards.tracing(tracing_ctx),
            torch._functorch.config.patch("force_autograd_cache", True),
        ):
            result = regional_inductor(gm, example_inputs)
        from torch._inductor.output_code import RegionalOutputCode

        # Override the ops filter after compilation so that
        # serialization (which happens later) allows distributed
        # collective ops like _c10d_functional through GraphPickler.
        if isinstance(result, RegionalOutputCode):
            result._ops_filter = _ops_filter_with_distributed
            result._node_metadata_key_filter = _node_metadata_key_filter_distributed
        else:
            logger.warning(
                "regional_inductor with serializable=True did not produce "
                "RegionalOutputCode; distributed ops may not serialize correctly."
            )
        return result

    with torch._guards.tracing(tracing_ctx):
        gm = regional_inductor(gm, example_inputs)

    # regional_inductor may switch to boxed calling convention; reset to
    # default so the graph can be called with positional args as usual.
    gm.graph.set_codegen(torch.fx.graph.CodeGen())
    gm.recompile()
    return gm


def insert_kernel_annotations_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    """Insert mark_kernels() calls at module boundaries in the FX graph.

    Reads ``node.meta["custom"]["module_fqn"]`` (set via
    ``annotate_module_fqns``) and inserts enter/exit calls so that
    CUDA graph capture records the annotations.

    Requires ``cuda-python`` package and CUDA toolkit/driver >= 13.1
    (or cuda-compat >= 13.1).  Returns the graph unchanged when unavailable.

    Also enables annotation capture on :class:`CUDAGraphWrapper` so that
    ``enable_annotations=True`` is passed to ``torch.cuda.graph()``.

    Alternative approaches:

    1. **fx.Interpreter**: During cudagraph capture, run the graph via an
       ``fx.Interpreter`` subclass that reads ``module_fqn`` metadata and
       calls ``mark_kernels`` enter/exit around each node — avoids mutating
       the graph.
    2. **Custom CodeGen**: Use a custom ``torch.fx.graph.CodeGen`` to emit
       enter/exit lines (or ``with`` blocks) directly in the generated
       Python code.

    The current graph-pass approach is the least invasive.
    """
    from torch.cuda._graph_annotations import _is_tools_id_unavailable

    from torchtitan.experiments.graph_trainer.common_utils import _MODULE_FQN
    from torchtitan.experiments.graph_trainer.cudagraph import (
        enable_cudagraph_annotations,
    )

    def _enter(annotation: dict) -> object:
        from torch.cuda._graph_annotations import mark_kernels

        ctx = mark_kernels(annotation)
        ctx.__enter__()
        return ctx

    def _exit(ctx: object) -> None:
        ctx.__exit__(None, None, None)  # type: ignore[union-attr]

    if _is_tools_id_unavailable():
        return gm

    enable_cudagraph_annotations()

    graph = gm.graph
    current_fqn: str | None = None
    current_ctx_node = None

    for node in list(graph.nodes):
        fqn = (node.meta.get("custom") or {}).get(_MODULE_FQN)

        if fqn != current_fqn:
            # Close previous scope
            if current_ctx_node is not None:
                with graph.inserting_before(node):
                    exit_node = graph.call_function(_exit, (current_ctx_node,))
                    exit_node.meta["custom"] = {}
                current_ctx_node = None

            # Open new scope
            if fqn is not None:
                with graph.inserting_before(node):
                    enter_node = graph.call_function(
                        _enter,
                        ({_MODULE_FQN: fqn},),
                    )
                    enter_node.meta["custom"] = {}
                current_ctx_node = enter_node

            current_fqn = fqn

    # Close any trailing scope (before output/return)
    if current_ctx_node is not None:
        output_nodes = [n for n in graph.nodes if n.op == "output"]
        if output_nodes:
            with graph.inserting_before(output_nodes[0]):
                exit_node = graph.call_function(_exit, (current_ctx_node,))
                exit_node.meta["custom"] = {}

    graph.lint()
    gm.recompile()
    return gm


def cudagraph_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
    *,
    is_forward: bool,
    static_input_indices: list[int] | None = None,
    tensor_input_indices: list[int] | None = None,
) -> torch.fx.GraphModule:
    """
    Apply cudagraph.

    This pass wraps the forward function with cudagraph during compilation and does
    not record cudagraph until runtime.
    - For the first run, it will warm up operators such as nccl.
    - For the second run, it will record cudagraph and replay cudagraph.
    - For the following runs, it will replay cudagraph.

    Args:
        gm: The graph module to wrap.
        example_inputs: Example inputs for warmup/recording.
        is_forward: Whether this is a forward graph (True) or backward graph
            (False). Used to infer which inputs have stable tensor addresses
            when ``static_input_indices`` is not provided.
        static_input_indices: Explicit list of input indices with stable tensor
            addresses. When provided, ``is_forward`` is not used for inference.
        tensor_input_indices: Indices of graph inputs that are tensors (as
            opposed to opaque values like DeviceMesh). Used to compute which
            inputs need copying for cudagraph replay. When not provided, this
            is inferred from ``example_inputs``.
    """
    if not isinstance(gm, torch.fx.GraphModule):
        raise TypeError(
            f"cudagraph_pass requires a GraphModule but got {type(gm).__name__}. "
            f"Ensure cudagraph is not combined with passes that replace the "
            f"GraphModule (e.g. full_inductor_compilation)."
        )

    # Lazy import: cudagraph.py runs init_global_graph_pool() at import time,
    # which must happen after torch.cuda.set_device(local_rank).
    from torchtitan.experiments.graph_trainer.cudagraph import (
        CUDAGraphWrapper,
        get_static_input_indices,
    )

    if static_input_indices is None:
        static_input_indices = get_static_input_indices(gm, is_forward)
    gm.forward = CUDAGraphWrapper(
        gm.forward,
        example_inputs,
        static_input_indices,
        tensor_input_indices=tensor_input_indices,
    )
    logger.info("Applied cudagraph pass.")
    return gm


def annotate_flex_attention_for_regional_inductor_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    flex_compile_config: dict | None,
    mask_compile_config: dict | None = None,
) -> torch.fx.GraphModule:
    """Tag flex attention HOPs with compile_with_inductor for regional_inductor.

    Annotates three sets of nodes so that regional_inductor correctly
    scoops and compiles flex attention regions:
    1. The HOP node itself (flex_attention / flex_attention_backward)
    2. The get_attr nodes referencing score_mod / mask_mod submodules.
    3. All nodes inside those submodule graphs.

    Args:
        gm: The graph module to annotate.
        example_inputs: Example inputs (unused, required by pass interface).
        flex_compile_config: Inductor config dict for flex attention HOP
            nodes and their get_attr submodule references. When provided,
            wrapped as ``{"inductor_configs": flex_compile_config}``.
            When None, nodes are tagged with an empty annotation.
        mask_compile_config: Inductor config dict for nodes inside mask_mod
            subgraphs. When provided, wrapped as
            ``{"inductor_configs": mask_compile_config}``.
            When None, nodes are tagged with an empty annotation.
    """
    flex_compile_annotation: dict = (
        {"inductor_configs": flex_compile_config}
        if flex_compile_config is not None
        else {}
    )
    mask_compile_annotation: dict = (
        {"inductor_configs": mask_compile_config}
        if mask_compile_config is not None
        else {}
    )

    for node in gm.graph.nodes:
        if node.target not in {
            torch.ops.higher_order.flex_attention,
            torch.ops.higher_order.flex_attention_backward,
        }:
            continue
        node.meta.setdefault("custom", {})[
            "compile_with_inductor"
        ] = flex_compile_annotation
        for inp in node.all_input_nodes:
            if inp.op != "get_attr":
                continue
            submod = getattr(gm, inp.target, None)
            if not isinstance(submod, torch.fx.GraphModule):
                continue
            inp.meta.setdefault("custom", {})[
                "compile_with_inductor"
            ] = flex_compile_annotation

            # Following are the nodes in mask_mod subgraph
            for sub_node in submod.graph.nodes:
                sub_node.meta.setdefault("custom", {})[
                    "compile_with_inductor"
                ] = mask_compile_annotation
    return gm


def _make_default_memory_policy(
    save_ops: set | None = None,
    *,
    fsdp_reshard_after_forward: bool = True,
) -> Callable:
    """Create a SAC policy function from a set of op targets to save."""
    if save_ops is None:
        save_ops = _get_save_ops()
        if not fsdp_reshard_after_forward:
            save_ops.add(torch.ops._c10d_functional.all_gather_into_tensor.default)

    def policy_fn(node: torch.fx.Node) -> CheckpointPolicy:
        if node.target in save_ops:
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return policy_fn


def _make_eager_memory_policy(save_ops: set | None = None) -> Callable:
    """Eager-compatible SAC policy that alternates mm ops between save/recompute.

    Matches the behavior of torchtitan.distributed.activation_checkpoint:
    every second mm/linear op is marked PREFER_RECOMPUTE instead of MUST_SAVE.
    """
    if save_ops is None:
        save_ops = _get_save_ops()
    mm_ops = {torch.ops.aten.mm.default, torch.ops.aten.linear.default}
    mm_count = 0

    def policy_fn(node: torch.fx.Node) -> CheckpointPolicy:
        nonlocal mm_count
        if node.target in mm_ops:
            mm_count += 1
            if node.target in save_ops and mm_count % 2 == 0:
                return CheckpointPolicy.PREFER_RECOMPUTE
        if node.target in save_ops:
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return policy_fn


def apply_sac_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    policy_fn: Callable[[torch.fx.Node], CheckpointPolicy] | None = None,
) -> torch.fx.GraphModule:
    """Apply selective activation checkpointing on the joint graph.

    Annotates forward ``call_function`` nodes with a ``CheckpointPolicy``
    determined by ``policy_fn``. After tagging, a boundary pass forces
    ``MUST_SAVE`` on recomputable nodes whose output crosses a layer
    boundary (layer N → layer N+1), since recomputing them would require
    rerunning the entire preceding layer.

    ``getitem`` / ``wait_tensor`` nodes inherit the parent's tag.

    The model must have been annotated with ``annotate_module_fqns`` before
    tracing so that nodes carry ``module_fqn`` metadata.

    Args:
        gm: The joint forward-backward graph module.
        policy_fn: Callable that takes a node and returns a CheckpointPolicy.
            Defaults to ``_make_default_memory_policy()`` if None.

    Returns:
        The annotated graph module
    """
    if policy_fn is None:
        policy_fn = _make_default_memory_policy()

    layer_stats: dict[int, dict[str, int]] = defaultdict(
        lambda: {"save": 0, "recompute": 0}
    )

    # Pass 1: Tag each forward node with a recompute policy.
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue

        # Skip backward nodes — they must not carry recompute tags,
        # otherwise the remat pass would try to duplicate backward ops.
        if _is_backward_node(node):
            continue

        if node.target in (
            operator.getitem,
            torch.ops._c10d_functional.wait_tensor.default,
        ):
            # Propagate from parent: getitem extracts tuple elements,
            # wait_tensor is tied to its async collective — both must
            # share the parent's save/recompute decision.
            parent = node.args[0]
            if isinstance(parent, torch.fx.Node) and "recompute" in parent.meta:
                node.meta["recompute"] = parent.meta["recompute"]
            continue

        layer_id = _get_layer_id(node)

        # AC only applies inside transformer layers, matching eager apply_ac
        # which wraps TransformerBlock only. In particular with ChunkedCELoss
        # the chunked region (lm_head + per-chunk ce_loss) must not be tagged
        # for recompute: per-chunk autograd backward dispatches create N
        # disjoint backward regions, and the upstream remat pass errors with
        # "Detected N disjoint backward regions that require recomputation,
        # but remat only supports one such region."
        if layer_id == _NOT_IN_LAYERS:
            continue

        # NOTE: The eager SAC policy (activation_checkpoint.py) alternates
        # mm ops between MUST_SAVE and PREFER_RECOMPUTE. We omit that here
        # because the alternating heuristic is arbitrary.
        node.meta["recompute"] = policy_fn(node)
        key = (
            "save"
            if node.meta["recompute"] == CheckpointPolicy.MUST_SAVE
            else "recompute"
        )
        layer_stats[layer_id][key] += 1

    # Pass 2: Force MUST_SAVE at layer boundaries. If a recomputable node
    # feeds into a node in a higher layer, saving it is cheaper than
    # recomputing the entire preceding layer.
    def _is_recomputable(n: torch.fx.Node) -> bool:
        return n.meta.get("recompute") in (
            CheckpointPolicy.PREFER_RECOMPUTE,
            CheckpointPolicy.MUST_RECOMPUTE,
        )

    boundary_saves = 0
    for node in gm.graph.nodes:
        if _is_backward_node(node) or not _is_recomputable(node):
            continue
        node_layer_id = _get_layer_id(node)
        for user in node.users:
            if (
                not _is_backward_node(user)
                and _is_recomputable(user)
                and _get_layer_id(user) > node_layer_id
            ):
                node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
                boundary_saves += 1
                break

    gm.recompile()
    logger.info("Applied selective activation checkpointing (SAC) graph pass.")
    if boundary_saves:
        logger.info(f"  Forced {boundary_saves} nodes to MUST_SAVE at layer boundaries")
    for layer_id in sorted(layer_stats):
        stats = layer_stats[layer_id]
        label = "non-layer" if layer_id == _NOT_IN_LAYERS else str(layer_id)
        logger.info(
            f"  Layer {label}: "
            f"{stats['save']} MUST_SAVE, "
            f"{stats['recompute']} PREFER_RECOMPUTE"
        )
    return gm


def tag_with_memory_policy_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    config: "GraphTrainer.Config",
) -> torch.fx.GraphModule:
    """Tag forward nodes with MUST_SAVE, PREFER_RECOMPUTE, or MUST_CPU_OFFLOAD.

    The ``config.compile.memory_policy`` selects the tagging strategy:
        default: SAC with all compute-intensive ops saved.
        eager: SAC alternating mm ops between save/recompute.
        cpu_offload_all: tag all eligible activations for CPU offload.

    Other memory policies combining SAC and CPU offload can be added here.
    """
    memory_policy = config.compile.memory_policy
    if memory_policy == "default":
        fsdp_reshard_after_forward = get_fsdp_reshard_after_forward_policy(
            config.parallelism.fsdp_reshard_after_forward,
            pp_enabled=config.parallelism.pipeline_parallel_degree > 1,
        )
        default_policy_fn = functools.partial(
            _make_default_memory_policy,
            fsdp_reshard_after_forward=fsdp_reshard_after_forward,
        )
        apply_sac_pass(gm, policy_fn=default_policy_fn())
    elif memory_policy == "eager":
        apply_sac_pass(gm, policy_fn=_make_eager_memory_policy())
    elif memory_policy == "cpu_offload_all":
        tag_all_offloadable_activations(gm)
    else:
        raise ValueError(f"Unknown memory_policy: {memory_policy!r}")

    log_activation_memory_policy(gm)
    return gm


def selective_activation_remat_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    """Duplicate recompute nodes for backward use, then DCE unused forward versions.

    Wraps ``remat_using_tags_for_fwd_loss_bwd_graph`` with the graph pass
    signature ``(gm, example_inputs)``.
    """
    # TODO: remove this wrapper when upstream remat_using_tags_for_fwd_loss_bwd_graph
    # accepts example_inputs (matching the graph pass signature).
    from torch._functorch._activation_checkpointing.remat_using_tags_for_fwd_loss_bwd_graph_pass import (
        remat_using_tags_for_fwd_loss_bwd_graph,
    )

    return remat_using_tags_for_fwd_loss_bwd_graph(gm)


# Apply activation checkpointing on joint graph before partitioner
def fsdp_reshard_after_fwd_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    reshard_after_forward: bool,
) -> torch.fx.GraphModule:
    # this pass implements simplefsdp's fsdp_reshard_after_forward behavior
    # when fsdp_reshard_after_forward set to True, it will annotate simple_fsdp AG
    #   to CheckpointPolicy.MUST_RECOMPUTE.
    # when fsdp_reshard_after_forward set to False, it will annotate simple_fsdp AG
    #   to CheckpointPolicy.MUST_SAVE.
    gm = annotate_fsdp_all_gather(gm, reshard_after_forward)
    gm.recompile()
    return gm


def full_inductor_compilation_pass(
    gm: torch.fx.GraphModule, example_inputs: tuple
) -> torch.fx.GraphModule:
    """Apply full Inductor compilation with code generation.

    Applies inductor decompositions (e.g. ``aten.t`` → ``aten.permute``),
    then compiles the graph into optimized Triton/C++ kernels via
    ``compile_fx_inner`` and replaces the GraphModule's ``forward``
    with the compiled callable.

    Must be the **terminal** pass — no FX-graph-level passes (e.g.
    ``custom_codegen_pass``, ``insert_kernel_annotations_pass``) can
    run after this because the FX graph is no longer authoritative.
    """

    def _apply_decompositions(
        gm: torch.fx.GraphModule, example_inputs: tuple
    ) -> torch.fx.GraphModule:
        """Retrace with ``select_decomp_table()`` so that ops like ``aten.t``
        are decomposed before ``compile_fx_inner``."""
        from torch._inductor.decomposition import select_decomp_table
        from torch._subclasses.fake_tensor import FakeTensor
        from torch.fx.experimental.proxy_tensor import make_fx

        decomp_table = select_decomp_table()

        fake_mode = None
        for inp in example_inputs:
            if isinstance(inp, FakeTensor):
                fake_mode = inp.fake_mode
                break

        if fake_mode is not None:
            with fake_mode:
                gm = make_fx(
                    gm,
                    decomposition_table=decomp_table,
                    _allow_non_fake_inputs=True,
                )(*example_inputs)

        return gm

    gm = _apply_decompositions(gm, example_inputs)
    output_code = compile_fx_inner(gm, example_inputs)

    # compile_fx_inner returns OutputCode with boxed calling convention
    # (single list arg). Adapt to positional args so the graph trainer's
    # execution path (gm(*flat_inputs)) works unchanged.
    def _compiled_forward(*args):
        return output_code(list(args))

    gm.forward = _compiled_forward
    return gm


# Maps an FSDP group_name to an extra group_name created by this pass.
# Each NCCL PG gets its own CUDA stream, so the extra PG is what enables
# AG/RS overlap in backward.
_EXTRA_FSDP_PG_REGISTRY: dict[str, str] = {}


def _get_or_create_extra_fsdp_pg(source_pg_name: str) -> str:
    """Return the extra PG name for ``source_pg_name``, creating it once.

    The extra PG is a new NCCL process group with the same ranks as the source
    FSDP PG but a different communicator (and therefore a different CUDA stream).
    """
    import torch.distributed as dist

    if source_pg_name in _EXTRA_FSDP_PG_REGISTRY:
        return _EXTRA_FSDP_PG_REGISTRY[source_pg_name]

    source_pg = dist.distributed_c10d._resolve_process_group(source_pg_name)
    ranks = dist.get_process_group_ranks(source_pg)
    extra_pg = dist.new_group(
        ranks=ranks, group_desc="fsdp_extra", use_local_synchronization=True
    )
    _EXTRA_FSDP_PG_REGISTRY[source_pg_name] = extra_pg.group_name
    logger.info(
        f"Created extra FSDP PG (source: {source_pg_name}, "
        f"extra: {extra_pg.group_name})"
    )
    return extra_pg.group_name


def overlap_fsdp_ag_rs_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    """
    Reassign FSDP all-gather nodes to an extra NCCL process group for
    AG/RS overlap in backward.

    Discovers the FSDP PG by inspecting the graph, creates an extra
    NCCL PG over the same ranks (giving it a separate CUDA stream),
    and rewrites every all-gather using that source PG to the extra PG.
    This separates all-gathers from reduce-scatters onto different streams,
    enabling AG/RS overlap in backward.

    No-op when the graph has no FSDP all-gathers. Must be applied BEFORE
    bucketing passes so bucketed all-gathers inherit the new PG name.
    """
    source_pg_name: str | None = None
    for node in gm.graph.nodes:
        if is_wait_tensor_from_fsdp(node):
            ag_node = node.args[0]
            source_pg_name = ag_node.args[2]
            break

    if source_pg_name is None:
        return gm

    target_pg_name = _get_or_create_extra_fsdp_pg(source_pg_name)

    count = 0
    for node in gm.graph.nodes:
        if is_all_gather(node) and node.args[2] == source_pg_name:
            # AG args: (input_tensor, group_size, group_name)
            node.args = (node.args[0], node.args[1], target_pg_name)
            count += 1
    if count > 0:
        logger.info(
            f"Rewrote {count} all-gather node(s) from PG {source_pg_name} "
            f"to PG {target_pg_name}"
        )
    gm.recompile()
    return gm


def tlparse_log_graph_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    graph_name: str,
    debug: bool = False,
) -> torch.fx.GraphModule:
    """Log the transformed graph to tlparse via trace_structured.

    This pass should be added as the last transform in fwd/bwd_transforms
    so that the logged graph reflects all prior transformations.

    Args:
        gm: The graph module to log.
        example_inputs: The example inputs (unused, required by protocol).
        graph_name: The name for this graph artifact
            (e.g. "aot_forward_graph_transformed").
        debug: When True, include additional metadata in the printed nodes.

    Returns:
        The graph module unchanged.
    """
    additional_meta = ["autograd_backward"]
    if debug:
        additional_meta.append("seq_nr")

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": graph_name,
            "encoding": "string",
        },
        payload_fn=lambda: gm.print_readable(
            print_output=False,
            include_stride=True,
            include_device=True,
            expanded_def=True,
            additional_meta=additional_meta,
        ),
        expect_trace_id=False,
    )

    return gm


# Registry mapping pass names to pass functions (for AOT mode fwd/bwd passes)
AVAILABLE_COMPILER_PASSES = {
    "auto_bucketing": autobucketing_reordering_pass,
    "transformer_block_bucketing": transformer_block_bucketing_reordering_pass,
    "regional_inductor": regional_inductor_pass,
    "custom_codegen": custom_codegen_pass,
    "cudagraph": cudagraph_pass,
    "full_inductor_compilation": full_inductor_compilation_pass,
}

# Registry for joint custom passes (applied before partitioning, AOT mode only)
AVAILABLE_JOINT_PASSES = {
    "fsdp_reshard_after_fwd": fsdp_reshard_after_fwd_pass,
    "apply_sac": apply_sac_pass,
}
