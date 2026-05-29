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
- ``xpugraph.py`` —  xpugraph wrapping
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

from torchtitan.experiments.graph_trainer.configs import (
    MOE_BLOCK_FQN,
    validate_ep_overlap_config,
)

from torchtitan.experiments.graph_trainer.cpu_offload import apply_cpu_offload_pass
from torchtitan.experiments.graph_trainer.cudagraph import (
    cudagraph_pass,
    insert_kernel_annotations_pass,
)
from torchtitan.experiments.graph_trainer.debug_utils import (
    log_graph_diff,
    snapshot_graph,
    tlparse_log_graph_pass,
)
from torchtitan.experiments.graph_trainer.ep_chunk_pass import (
    ep_overlap_chunk_pass,
    populate_chunk_dim_metadata_pass,
)
from torchtitan.experiments.graph_trainer.ep_eager_chunk import (
    populate_eager_chunk_metadata_pass,
)
from torchtitan.experiments.graph_trainer.ep_overlap_pass import (
    ep_overlap_schedule_pass,
)
from torchtitan.experiments.graph_trainer.ep_pass_utils import (
    concretize_ep_chunk_symbolic_shapes_pass,
)
from torchtitan.experiments.graph_trainer.ep_process_group_pass import (
    isolate_ep_process_group_pass,
)
from torchtitan.experiments.graph_trainer.fsdp_passes import (
    get_fsdp_param_module_order,
    get_transformer_block_bucket_counts,
    joint_transformer_block_bucketing_reordering_pass,
    reassign_collective_pgs_pass,
    schedule_fsdp_comms_to_dense_regions_pass,
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

def _get_graph_capture_backend() -> str | None:
    """Return the active device backend for graph capture."""
    if hasattr(torch, "xpu") and torch.xpu.is_available() and torch.xpu.is_initialized():
        return "xpu"
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        return "cuda"
    return None

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


def _tensor_parallel_degree(config, parallel_dims=None) -> int:
    """Return TP degree from ``ParallelDims`` when available, else config."""
    if parallel_dims is not None and hasattr(parallel_dims, "tp"):
        return int(parallel_dims.tp)
    return int(getattr(config.parallelism, "tensor_parallel_degree", 1))


def compile_time_passes(
    traced_result: "TracedResult",
    config: "GraphTrainer.Config",
    *,
    use_cudagraph: bool = False,
    parallel_dims=None,
) -> list[Callable]:
    """Cleanup, FlexAttention annotation, and regional_inductor passes.

    If precompile is enabled, these are applied before serialization so
    that compiled Triton kernels are baked into the artifact. Otherwise
    they run at trace time via ``construct_default_graph_passes``.

    cudagraph is excluded because it needs to re-capture the graph into
    an in-memory CUDA graph at runtime.

    ``reassign_collective_pgs_pass`` runs just before bucketing to place
    collectives on dedicated process groups / streams (bucketing then inherits
    the new PGs). Disable with
    ``--compile.disable_passes reassign_collective_pgs_pass``.
    """
    from torchtitan.components.loss import ChunkedCELoss
    from torchtitan.experiments.graph_trainer.common_utils import (
        get_default_transformer_block_buckets,
    )
    from torchtitan.models.common.attention import FlexAttention

    n_layers = len(config.model_spec.model.layers)
    loss_config = getattr(config, "loss", None)
    uses_chunked_loss = isinstance(loss_config, ChunkedCELoss.Config)
    moe_layer_ids = frozenset(
        i
        for i, layer_cfg in enumerate(config.model_spec.model.layers)
        if getattr(layer_cfg, "moe", None) is not None
    )
    ep_overlap_enabled = config.compile.ep_overlap.enabled
    if parallel_dims is not None and hasattr(parallel_dims, "get_optional_mesh"):
        efsdp_mesh = parallel_dims.get_optional_mesh("efsdp")
        efsdp_degree = 1 if efsdp_mesh is None else efsdp_mesh.size()
    else:
        dp_shard = max(1, getattr(config.parallelism, "data_parallel_shard_degree", 1))
        cp_degree = getattr(config.parallelism, "context_parallel_degree", 1)
        tp_degree = getattr(config.parallelism, "tensor_parallel_degree", 1)
        ep_degree = max(1, getattr(config.parallelism, "expert_parallel_degree", 1))
        efsdp_degree = max(1, (dp_shard * cp_degree * tp_degree) // ep_degree)
    module_bucket_plans = get_default_transformer_block_buckets(
        n_layers,
        chunked_loss_enabled=uses_chunked_loss,
        moe_layer_ids=moe_layer_ids,
        split_moe_expert_buckets=efsdp_degree > 1,
    )

    passes: list[Callable] = [
        eliminate_dead_code_pass,
        canonicalize_graph_pass,
    ]
    ep_overlap_chunk_passes: list[Callable] = []
    ep_overlap_module_fqn: str | None = None
    ep_overlap_chunk_strategy: str | None = None
    if ep_overlap_enabled:
        (
            overlap_dim,
            ep_overlap_chunk_strategy,
            ep_overlap_module_fqn,
        ) = validate_ep_overlap_config(config.compile.ep_overlap)
        if (
            ep_overlap_chunk_strategy == "graph"
            and _tensor_parallel_degree(config, parallel_dims) > 1
        ):
            # After DTensor lowering, the FX graph contains physical TP-local
            # tensors and TP/SP layout helpers. Splitting those values is not
            # proven equivalent to eager DTensor-level chunking.
            raise ValueError(
                "Graph EP chunking does not support tensor_parallel_degree > 1. "
                "Use tensor_parallel_degree=1 or eager chunking for this "
                "configuration."
            )
        if ep_overlap_chunk_strategy == "eager":
            ep_overlap_chunk_passes.append(populate_eager_chunk_metadata_pass)
        if ep_overlap_chunk_strategy == "graph":
            ep_overlap_chunk_passes.extend(
                [
                    functools.partial(
                        populate_chunk_dim_metadata_pass,
                        mode=overlap_dim,
                    ),
                    functools.partial(
                        ep_overlap_chunk_pass,
                        mode=overlap_dim,
                        module_pattern=ep_overlap_module_fqn,
                        num_static_inputs=traced_result.num_static_inputs,
                        optimize_grad_live_out=not (
                            config.compile.ep_overlap.disable_early_grad_accumulation
                        ),
                        require_all_to_all=(
                            getattr(config.parallelism, "expert_parallel_degree", 1) > 1
                        ),
                    ),
                ]
            )

    passes.extend(
        [
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
        ]
    )
    if ep_overlap_enabled:
        passes.extend(ep_overlap_chunk_passes)
        passes.append(isolate_ep_process_group_pass)
        passes.append(eliminate_dead_code_pass)

    if config.compile.enable_fsdp_ag_rs_overlap:
        passes.append(reassign_collective_pgs_pass)
    passes.append(
        functools.partial(
            joint_transformer_block_bucketing_reordering_pass,
            module_bucket_plans=module_bucket_plans,
            # FSDP2 packs buckets in managed parameter order. The traced state
            # FQNs preserve that registration order, unlike graph execution order.
            fsdp_param_module_order=get_fsdp_param_module_order(
                traced_result.state_fqns
            ),
        )
    )

    if ep_overlap_enabled:
        assert ep_overlap_module_fqn is not None
        passes.append(
            functools.partial(
                ep_overlap_schedule_pass,
                module_pattern=ep_overlap_module_fqn,
                require_all_to_all=(
                    getattr(config.parallelism, "expert_parallel_degree", 1) > 1
                ),
                pair_first_token_exchange=ep_overlap_module_fqn == MOE_BLOCK_FQN,
            )
        )
        passes.append(concretize_ep_chunk_symbolic_shapes_pass)

    enable_fsdp_dense_region_overlap = config.compile.enable_fsdp_dense_region_overlap
    if (
        enable_fsdp_dense_region_overlap
        and ep_overlap_enabled
        and (
            ep_overlap_module_fqn != MOE_BLOCK_FQN
            or ep_overlap_chunk_strategy != "graph"
        )
    ):
        warnings.warn(
            "--compile.enable_fsdp_dense_region_overlap is ignored when "
            "--compile.ep_overlap.enabled is set unless graph chunking is "
            "applied to layers.*.moe. The dense FSDP scheduler can be used "
            "standalone when ep_overlap is disabled.",
            stacklevel=2,
        )
        enable_fsdp_dense_region_overlap = False

    if enable_fsdp_dense_region_overlap:
        # Move FSDP comm launches into neighboring transformer dense regions.
        # This is useful both as an EP-overlap companion and as a standalone
        # FSDP scheduling ablation, so it is controlled by its explicit flag.
        passes.append(
            functools.partial(
                schedule_fsdp_comms_to_dense_regions_pass,
                moe_layer_ids=moe_layer_ids,
                n_layers=n_layers,
                transformer_bucket_counts_by_layer=get_transformer_block_bucket_counts(
                    module_bucket_plans,
                    n_layers=n_layers,
                ),
                strict=True,
            )
        )

    if config.parallelism.enable_async_tensor_parallel:
        passes.append(async_tensor_parallel_pass)

    inductor_compilation = config.compile.inductor_compilation
    if inductor_compilation == "full":
        # Compile the entire graph into optimized Triton kernels. Must be
        # terminal; the FX graph is no longer authoritative after this pass.
        passes.append(full_inductor_compilation_pass)
    if inductor_compilation == "regional":
        # FlexAttention HOPs must be compiled (via regional_inductor) to
        # produce bitwise identical results to the eager Trainer path.
        passes.append(
            functools.partial(
                annotate_flex_attention_for_regional_inductor_pass,
                flex_compile_config=FlexAttention.inductor_configs,
            )
        )
        if config.compile.numerics_changing_optim:
            from torchtitan.experiments.graph_trainer.performance_passes import (
                annotate_rmsnorm_for_regional_inductor_pass,
            )

            passes.append(annotate_rmsnorm_for_regional_inductor_pass)
        passes.append(regional_inductor_pass)
        if use_cudagraph:
            passes.append(insert_kernel_annotations_pass)
    return passes


def construct_default_graph_passes(
    traced_result: "TracedResult",
    config: "GraphTrainer.Config",
    *,
    parallel_dims=None,
) -> list[Callable]:
    """Build the pass list for the aot_fx_trace path.

    When ``precompile_artifact_dir`` is unset, returns the full list: cleanup,
    FlexAttention annotation, regional_inductor, and graph capture.

    When ``precompile_artifact_dir`` is set, the artifact has graph
    transformed during precompile phase, so only graph capture is returned.
    """
    want_xpugraph = (
        config.compile.enable_xpugraph
        and "xpugraph_pass" not in config.compile.disable_passes
    )
    want_cudagraph = (
        config.compile.enable_cudagraph
        and "cudagraph_pass" not in config.compile.disable_passes
    )

    if want_xpugraph and want_cudagraph:
        raise ValueError(
            "Only one graph capture backend can be enabled. "
            "Set either compile.enable_xpugraph or compile.enable_cudagraph, not both."
        )

    has_precompile_artifact = bool(config.compile.precompile_artifact_dir)

    passes: list[Callable] = []
    if not has_precompile_artifact:
        passes.extend(
            compile_time_passes(
                traced_result,
                config,
                use_cudagraph=want_cudagraph,
                parallel_dims=parallel_dims,
            )
        )

    if want_xpugraph:
        graph_capture_pass = xpugraph_pass
    elif want_cudagraph:
        graph_capture_pass = cudagraph_pass
    else:
        graph_capture_pass = None

    # Graph capture should be the last pass.
    if graph_capture_pass is not None:
        static_input_indices = list(range(traced_result.num_static_inputs))
        passes.append(
        functools.partial(
            graph_capture_pass,
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
    # Some passes intentionally change placeholder shape metadata. Keep the
    # pass-local fake inputs in sync so later compiler passes see the same
    # static/dynamic contract as the FX graph.
    pass_example_inputs = list(example_inputs)
    for pass_fn in passes:
        pass_name = _get_pass_name(pass_fn)
        if debug:
            tlparse_log_graph_pass(gm, graph_name=f"before_{pass_name}", debug=debug)
            before_snapshot = snapshot_graph(gm)
            start = time.perf_counter()
        gm = pass_fn(gm, pass_example_inputs)
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
