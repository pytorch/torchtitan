# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Compiler passes for graph_trainer training.

This module provides pass orchestration: building the pass list, applying passes
in order, and the pass registries.  Individual passes live in dedicated modules:

- ``memory_policy.py`` - SAC and min-cut policy tagging and dispatch
- ``decompositions.py`` — standalone graph decomposition
- ``inductor_passes.py`` — regional and full Inductor compilation
- ``cuda_graph.py`` — opt-in CUDA graph wrapping and kernel annotations
- ``fsdp_passes.py`` — FSDP bucketing and resharding
- ``remove_noop_passes.py`` — mandatory gradient-marker cleanup plus graph
  cleanup bundled as ``canonicalize_graph_pass`` (detach, identity view/slice,
  back-to-back transpose, view→reshape normalization)
- ``performance_passes.py`` — opt-in numerics-changing optimizations
- ``subgraph_regions.py`` — region annotation, invoke_subgraph outlining, and
  shared region prologue extraction
- ``selective_activation_remat.py`` — activation rematerialization
- ``cpu_offload.py`` — CPU offload insertion
- ``custom_codegen.py`` — custom code generation for profiling/debugging
"""

from __future__ import annotations

import functools
import logging
import time
import warnings
from collections.abc import Callable

import torch

from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    MOE_BLOCK_FQN,
    validate_ep_overlap_config,
)

from torchtitan.experiments.graph_trainer.cpu_offload import apply_cpu_offload_pass
from torchtitan.experiments.graph_trainer.cuda_graph import (
    cuda_graph_pass as _cuda_graph_pass,
    insert_kernel_annotations_pass as _insert_kernel_annotations_pass,
)
from torchtitan.experiments.graph_trainer.debug_utils import (
    log_graph_diff,
    snapshot_graph,
    tlparse_log_graph_pass,
)
from torchtitan.experiments.graph_trainer.ep_eager_chunk import (
    populate_eager_chunk_metadata_pass,
)
from torchtitan.experiments.graph_trainer.ep_overlap_pass import (
    ep_overlap_schedule_pass,
)
from torchtitan.experiments.graph_trainer.ep_process_group_pass import (
    isolate_ep_process_group_pass,
)
from torchtitan.experiments.graph_trainer.fsdp_passes import (
    deduplicate_fsdp_unshard_chains_pass,
    get_fsdp_param_module_order,
    get_transformer_block_bucket_counts,
    get_transformer_block_layer_ids,
    joint_transformer_block_bucketing_reordering_pass,
    reassign_collective_pgs_pass,
    schedule_fsdp_comms_to_dense_regions_pass,
)
from torchtitan.experiments.graph_trainer.graph_pp.split_fsdp_collectives import (
    coalesce_fsdp_reduce_grad_add_pass,
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
    remove_parameter_gradient_markers_pass,
)
from torchtitan.experiments.graph_trainer.selective_activation_remat import (
    selective_activation_remat_pass,
)


cuda_graph_pass = _cuda_graph_pass
insert_kernel_annotations_pass = _insert_kernel_annotations_pass

logger = logging.getLogger(__name__)


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


def construct_mandatory_graph_passes() -> list[Callable]:
    """Return correctness passes that run even when optional passes are disabled."""
    return [
        remove_parameter_gradient_markers_pass,
        coalesce_fsdp_reduce_grad_add_pass,
    ]


def compile_time_passes(
    traced_result: "TracedResult",
    config: "GraphTrainer.Config",
    *,
    parallelism_context=None,
    include_inductor: bool = True,
    include_mandatory_normalization: bool = True,
) -> list[Callable]:
    """Cleanup, FlexInnerAttention annotation, and regional_inductor passes.

    If precompile is enabled, these are applied before serialization so
    that compiled Triton kernels are baked into the artifact. Otherwise
    they run at trace time via ``construct_default_graph_passes``.

    The opt-in passes in ``cuda_graph.py`` are not part of this list. Callers
    can use ``construct_cuda_graph_passes`` to apply them explicitly at runtime.

    ``reassign_collective_pgs_pass`` runs just before bucketing to place
    collectives on dedicated process groups / streams (bucketing then inherits
    the new PGs). Disable with
    ``compile.disable_passes=["reassign_collective_pgs_pass"]``.

    ``include_inductor=False`` leaves the graph in FX form after the
    metadata-preserving passes. GraphPP uses that mode before it calls its
    standalone partitioner and compiles the extracted graphs.

    ``include_mandatory_normalization=False`` lets GraphPP run required
    normalization unconditionally and then append only the optional passes
    controlled by ``enable_passes``.
    """
    from torchtitan.components.loss import ChunkedLossWrapper
    from torchtitan.experiments.graph_trainer.common_utils import (
        get_default_transformer_block_buckets,
    )

    n_layers = len(config.model.layers)
    num_mtp_layers = len(getattr(config.model, "mtp_layers", ()) or ())
    loss_config = getattr(config, "loss", None)
    uses_chunked_loss = isinstance(loss_config, ChunkedLossWrapper.Config)
    moe_layer_ids = frozenset(
        i
        for i, layer_cfg in enumerate(config.model.layers)
        if getattr(layer_cfg, "moe", None) is not None
    )
    ep_overlap_enabled = config.compile.ep_overlap.enabled
    if parallelism_context is not None and hasattr(
        parallelism_context, "get_optional_mesh"
    ):
        edp_shard_mesh = parallelism_context.get_optional_mesh("edp_shard")
        edp_shard_degree = 1 if edp_shard_mesh is None else edp_shard_mesh.size()
    else:
        dp_shard = max(1, getattr(config.parallelism, "data_parallel_shard_degree", 1))
        cp_degree = getattr(config.parallelism, "context_parallel_degree", 1)
        tp_degree = getattr(config.parallelism, "tensor_parallel_degree", 1)
        ep_degree = max(1, getattr(config.parallelism, "expert_parallel_degree", 1))
        edp_shard_degree = max(1, (dp_shard * cp_degree * tp_degree) // ep_degree)
    module_bucket_plans = get_default_transformer_block_buckets(
        n_layers,
        num_mtp_layers=num_mtp_layers,
        chunked_loss_enabled=uses_chunked_loss,
        moe_layer_ids=moe_layer_ids,
        split_moe_expert_buckets=edp_shard_degree > 1,
    )

    passes = construct_mandatory_graph_passes()
    if include_mandatory_normalization:
        passes.extend(
            [
                eliminate_dead_code_pass,
                canonicalize_graph_pass,
                deduplicate_fsdp_unshard_chains_pass,
            ]
        )
    ep_overlap_module_fqn: str | None = None
    if ep_overlap_enabled:
        _, ep_overlap_module_fqn = validate_ep_overlap_config(config.compile.ep_overlap)

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
        passes.append(populate_eager_chunk_metadata_pass)
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

    enable_fsdp_dense_region_overlap = config.compile.enable_fsdp_dense_region_overlap
    if enable_fsdp_dense_region_overlap and ep_overlap_enabled:
        warnings.warn(
            "compile.enable_fsdp_dense_region_overlap is ignored when "
            "compile.ep_overlap.enabled is set. The dense FSDP scheduler can "
            "standalone when ep_overlap is disabled.",
            stacklevel=2,
        )
        enable_fsdp_dense_region_overlap = False

    if enable_fsdp_dense_region_overlap:
        require_backward_all_gathers = get_fsdp_reshard_after_forward_policy(
            config.parallelism.fsdp_reshard_after_forward,
            pp_enabled=config.parallelism.pipeline_parallel_degree > 1,
        )
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
                local_layer_ids=get_transformer_block_layer_ids(
                    traced_result.state_fqns,
                    n_layers=n_layers,
                ),
                require_backward_all_gathers=require_backward_all_gathers,
                strict=True,
            )
        )

    if config.compile.enable_async_tensor_parallel:
        passes.append(async_tensor_parallel_pass)

    if not include_inductor:
        return passes

    passes.extend(
        final_inductor_compile_passes(
            config.compile,
        )
    )
    return passes


def final_inductor_compile_passes(
    compile_config: GraphTrainerCompileConfig,
    *,
    boxed_codegen: bool = False,
) -> list[Callable]:
    """Return the terminal Inductor passes for a traced graph.

    GraphTrainer applies these to the full train-step graph. GraphPP applies
    the same pass list to each extracted stage callable after its PP-specific
    partitioning has chosen the callable boundary. Terminal Inductor selection
    only depends on compile config; model- and parallelism-aware rewrites stay
    in ``compile_time_passes``.
    """
    from torchtitan.models.common.attention import FlexInnerAttention

    passes: list[Callable] = []
    inductor_compilation = compile_config.inductor_compilation
    if inductor_compilation == "full":
        # Compile the entire graph into optimized Triton kernels. Must be
        # terminal; the FX graph is no longer authoritative after this pass.
        passes.append(
            functools.partial(
                full_inductor_compilation_pass,
                boxed_codegen=boxed_codegen,
            )
        )
    elif inductor_compilation == "regional":
        # FlexInnerAttention HOPs must be compiled (via regional_inductor) to
        # produce bitwise identical results to the eager Trainer path.
        passes.append(
            functools.partial(
                annotate_flex_attention_for_regional_inductor_pass,
                flex_compile_config=FlexInnerAttention.inductor_configs,
            )
        )
        if compile_config.numerics_changing_optim:
            from torchtitan.experiments.graph_trainer.performance_passes import (
                annotate_rmsnorm_for_regional_inductor_pass,
            )

            passes.append(annotate_rmsnorm_for_regional_inductor_pass)
        passes.append(
            functools.partial(
                regional_inductor_pass,
                boxed_codegen=boxed_codegen,
            )
        )
    else:
        raise ValueError(
            "compile.inductor_compilation must be 'regional' or 'full', "
            f"got {inductor_compilation!r}"
        )
    return passes


def construct_default_graph_passes(
    traced_result: "TracedResult",
    config: "GraphTrainer.Config",
    *,
    parallelism_context=None,
) -> list[Callable]:
    """Build the pass list for the aot_fx_trace path.

    When ``precompile_artifact_dir`` is unset, returns the full list: cleanup,
    FlexInnerAttention annotation, and regional_inductor.

    When ``precompile_artifact_dir`` is set, the artifact has graph
    transformed during precompile phase, so no passes are returned.

    This default list never includes ``cuda_graph_pass``. GraphTrainer uses
    client-owned outer capture, while custom callers can use
    ``construct_cuda_graph_passes`` from ``cuda_graph.py`` themselves.
    """
    if config.compile.precompile_artifact_dir:
        return []
    return compile_time_passes(
        traced_result,
        config,
        parallelism_context=parallelism_context,
    )


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
    respect_disable_passes: bool = True,
) -> torch.fx.GraphModule:
    """Apply graph passes to the traced fwd+bwd graph.

    Args:
        gm: The traced forward+backward graph module.
        example_inputs: Example (fake) inputs matching the graph signature.
        passes: Ordered list of pass callables, each with signature
            ``(gm, example_inputs, **kwargs) -> gm``.
        compile_config: Optional compile config, used for ``disable_passes``.
        respect_disable_passes: Whether ``compile_config.disable_passes`` may
            remove passes from this invocation. GraphPP sets this to ``False``
            for mandatory pre-partition normalization.
    """
    disable_patterns = (
        compile_config.disable_passes if compile_config is not None else []
    )
    if respect_disable_passes and disable_patterns:
        passes = _filter_disabled_passes(passes, disable_patterns)
    pass_names = [_get_pass_name(pass_fn) for pass_fn in passes]
    pass_list = "\n  ".join(f"{i}. {name}" for i, name in enumerate(pass_names, 1))
    logger.info(f"Applying {len(passes)} graph passes:\n  {pass_list}")
    all_passes_start = time.perf_counter()
    tlparse_log_graph_pass(gm, graph_name="make_fx_graph_traced")
    # Some passes intentionally change placeholder shape metadata. Keep the
    # pass-local fake inputs in sync so later compiler passes see the same
    # static/dynamic contract as the FX graph.
    pass_example_inputs = list(example_inputs)
    for pass_fn in passes:
        pass_name = _get_pass_name(pass_fn)
        tlparse_log_graph_pass(gm, graph_name=f"before_{pass_name}")
        before_snapshot = snapshot_graph(gm)
        start = time.perf_counter()
        gm = pass_fn(gm, pass_example_inputs)
        assert isinstance(
            gm, torch.fx.GraphModule
        ), f"Pass {pass_name} returned {type(gm).__name__}, expected GraphModule"
        elapsed = time.perf_counter() - start
        logger.info(f"Pass {pass_name} took {elapsed:.3f}s")
        tlparse_log_graph_pass(gm, graph_name=f"after_{pass_name}")
        after_snapshot = snapshot_graph(gm)
        log_graph_diff(before_snapshot, after_snapshot, pass_name)
    all_passes_elapsed = time.perf_counter() - all_passes_start
    logger.info(f"All {len(passes)} graph passes took {all_passes_elapsed:.3f}s")
    return gm
