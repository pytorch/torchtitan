# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Compiler passes for graph_trainer training.

This module provides pass orchestration: building the pass list, applying passes
in order, and the pass registries.  Individual passes live in dedicated modules:

- ``memory_policy.py`` — SAC tagging and memory policy dispatch
- ``inductor_passes.py`` — regional and full Inductor compilation
- ``cudagraph.py`` — cudagraph wrapping and kernel annotations
- ``fsdp_passes.py`` — FSDP bucketing and resharding
- ``remove_noop_passes.py`` — graph cleanup bundled as ``canonicalize_graph_pass``
  (detach, identity view/slice, back-to-back transpose, view→reshape normalization)
- ``performance_passes.py`` — opt-in numerics-changing optimizations
- ``selective_activation_remat.py`` — activation rematerialization
- ``cpu_offload.py`` — CPU offload insertion
- ``custom_codegen.py`` — custom code generation for profiling/debugging
"""

from __future__ import annotations

import functools
import time
import warnings
from collections.abc import Callable

import torch
from torch._logging import trace_structured

from torchtitan.experiments.graph_trainer.cpu_offload import apply_cpu_offload_pass
from torchtitan.experiments.graph_trainer.cudagraph import (
    cudagraph_pass,
    insert_kernel_annotations_pass,
)
from torchtitan.experiments.graph_trainer.custom_codegen import custom_codegen_pass
from torchtitan.experiments.graph_trainer.debug_utils import (
    log_graph_diff,
    snapshot_graph,
)
from torchtitan.experiments.graph_trainer.fsdp_passes import (
    joint_transformer_block_bucketing_reordering_pass,
)
from torchtitan.experiments.graph_trainer.inductor_passes import (
    annotate_flex_attention_for_regional_inductor_pass,
    full_inductor_compilation_pass,
    regional_inductor_pass,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import TracedResult
from torchtitan.experiments.graph_trainer.memory_policy import (
    tag_with_memory_policy_pass,
)
from torchtitan.experiments.graph_trainer.remove_noop_passes import (
    canonicalize_graph_pass,
    eliminate_dead_code_pass,
)
from torchtitan.experiments.graph_trainer.selective_activation_remat import (
    selective_activation_remat_pass,
)
from torchtitan.tools.logging import logger

c10d = torch.ops._c10d_functional


def async_tensor_parallel_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
) -> torch.fx.GraphModule:
    """Pipeline TP collectives with matmuls via symmetric memory.

    Fuses all-gather + matmul into ``symm_mem.fused_all_gather_matmul``
    and matmul + reduce-scatter into
    ``symm_mem.fused_matmul_reduce_scatter``.
    """
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

    ``joint_transformer_block_bucketing_reordering_pass`` optionally runs
    ``overlap_fsdp_ag_rs_pass`` first (gated by
    ``compile.enable_fsdp_ag_rs_overlap``) so that forward+backward
    all-gathers end up on a separate CUDA stream from reduce-scatters
    (enabling AG/RS overlap in backward).
    """
    from torchtitan.experiments.graph_trainer.common_utils import (
        get_default_transformer_block_buckets,
    )
    from torchtitan.models.common.attention import FlexAttention

    n_layers = len(config.model_spec.model.layers)
    passes: list[Callable] = [
        eliminate_dead_code_pass,
        canonicalize_graph_pass,
        functools.partial(
            tag_with_memory_policy_pass,
            config=config,
        ),
        functools.partial(
            apply_cpu_offload_pass,
            prefetch_lookahead=config.compile.cpu_offload_prefetch_n_layers,
            defer_n_layers=config.compile.cpu_offload_defer_n_layers,
        ),
        selective_activation_remat_pass,
        functools.partial(
            joint_transformer_block_bucketing_reordering_pass,
            module_bucket_plans=get_default_transformer_block_buckets(n_layers),
            enable_fsdp_ag_rs_overlap=config.compile.enable_fsdp_ag_rs_overlap,
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
    want_cudagraph = "cudagraph_pass" not in config.compile.disable_passes

    has_precompile_artifact = bool(config.compile.precompile_artifact_dir)

    passes: list[Callable] = []
    if not has_precompile_artifact:
        passes.extend(
            compile_time_passes(traced_result, config, use_cudagraph=want_cudagraph)
        )

    if want_cudagraph:
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


def _get_pass_name(pass_fn: Callable) -> str:
    return (
        pass_fn.func.__name__
        if isinstance(pass_fn, functools.partial)
        else pass_fn.__name__
    )


def _filter_disabled_passes(
    passes: list[Callable], disable_names: list[str]
) -> list[Callable]:
    """Remove passes whose names exactly match any entry in ``disable_names``."""
    disable_set = set(disable_names)
    filtered = []
    skipped = []
    for pass_fn in passes:
        name = _get_pass_name(pass_fn)
        if name in disable_set:
            skipped.append(name)
        else:
            filtered.append(pass_fn)
    if skipped:
        logger.info(f"Disabled {len(skipped)} graph passes: {skipped}")
    return filtered


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
    disable_patterns = (
        compile_config.disable_passes if compile_config is not None else []
    )
    if disable_patterns:
        passes = _filter_disabled_passes(passes, disable_patterns)
    pass_names = [_get_pass_name(pass_fn) for pass_fn in passes]
    pass_list = "\n  ".join(f"{i}. {name}" for i, name in enumerate(pass_names, 1))
    logger.info(f"Applying {len(passes)} graph passes:\n  {pass_list}")
    all_passes_start = time.perf_counter()
    tlparse_log_graph_pass(gm, graph_name="make_fx_graph_traced", debug=debug)
    for pass_fn in passes:
        pass_name = _get_pass_name(pass_fn)
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
    all_passes_elapsed = time.perf_counter() - all_passes_start
    logger.info(f"All {len(passes)} graph passes took {all_passes_elapsed:.3f}s")
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
