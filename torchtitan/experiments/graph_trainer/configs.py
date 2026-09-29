# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from typing import Literal

from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.config.configs import CompileConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.graph_trainer.chunked_loss import (
    ChunkedLossWrapperWithParamGrads,
)
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer

EpOverlapChunkDim = Literal["batch", "seq"]
EpOverlapChunkStrategy = Literal["eager", "graph"]

TRANSFORMER_BLOCK_FQN = "layers.*"
MOE_BLOCK_FQN = "layers.*.moe"
SUPPORTED_EP_OVERLAP_MODULE_FQNS = frozenset({TRANSFORMER_BLOCK_FQN, MOE_BLOCK_FQN})


@dataclass(kw_only=True, slots=True)
class EpOverlapConfig:
    enabled: bool = False
    """Enable EP-overlap support for the selected chunking and scheduling mode."""

    chunk_dim: EpOverlapChunkDim = "batch"
    """Logical input dimension to split for EP-overlap chunking.

    Sequence chunking is only supported for MoE block roots because attention
    requires full K/V context.
    """

    strategy: EpOverlapChunkStrategy = "graph"
    """How selected EP-overlap regions are chunked before scheduling.

    ``eager`` wraps module forwards before tracing. ``graph`` traces the
    unmodified model and chunks selected regions with an FX graph pass.
    """

    module_fqn: str = TRANSFORMER_BLOCK_FQN
    """Single module FQN pattern chunked for EP overlap.

    v1 supports all transformer blocks (``layers.*``) or all MoE blocks
    (``layers.*.moe``). The overlap scheduler consumes the common chunk metadata
    produced by either eager or graph chunking.
    """

    disable_early_grad_accumulation: bool = False
    """Disable graph chunking's early parameter-gradient accumulation.

    Early accumulation is the performant default: graph chunking materializes
    parameter-gradient live-outs before distributed grad cast/communication
    when legal. This flag preserves eager chunking's cast/reduction order for
    strict bitwise tests.
    """


@dataclass(kw_only=True, slots=True)
class GraphTrainerCompileConfig(CompileConfig):
    mode: Literal["jit", "aot_fx_trace"] | None = "aot_fx_trace"
    """
    Compilation mode. Options:
        aot_fx_trace: non-strict tracing of fwd+loss+bwd via make_fx
        jit: standard torch.compile() with custom backend (deprecated)
    """

    backend: str = "aot_eager"

    passes: list[str] = field(default_factory=list)
    """
    Additional compiler pass names to apply.
    In JIT mode: applied as graph passes (e.g., auto_bucketing, transformer_block_bucketing)
    """

    enable_passes: bool = True
    """When False, skip optional graph passes (both default and user-configured).

    GraphPP still runs mandatory pre-partition normalization passes because its
    partitioning contracts depend on canonical graph structure.
    """

    disable_passes: list[str] = field(default_factory=list)
    """Pass names to selectively disable for debugging and ablation
    studies. A pass is skipped if its name exactly matches any entry.
    Example: --compile.disable_passes custom_codegen_pass,cudagraph_pass"""

    debug_graph_passes: bool = False
    """Log timing, op-count diffs, and before/after graphs for each pass to tlparse."""

    memory_policy: Literal[
        "default", "full", "eager", "sac_and_offload", "sac_and_paged_stash"
    ] = "default"
    """
    Memory optimization policy for activation management (SAC, offload, stash).
        default: SAC — save all compute-intensive ops and FSDP all_gathers.
        full: full recompute, saving layer outputs and operations selected by
            full_recompute_save_ops. With no selectors, this mirrors eager's
            full AC (checkpoint_wrapper with no context_fn).
        eager: SAC alternating mm ops between save/recompute, matching the
            eager AC policy in torchtitan.distributed.activation_checkpoint.
        sac_and_offload: SAC + CPU offload — apply default SAC first,
            then offload surviving MUST_SAVE activations to CPU within
            the cpu_offload_budget_gb budget.
        sac_and_paged_stash: SAC + MoE paged stashing — apply default SAC
            first, then page capacity-padded routed-expert activations into
            fixed-size pages so only the live rows stay resident. Requires a
            token dispatcher with a static capacity factor (e.g. HybridEP with
            non_blocking_capacity_factor set).
    """

    full_recompute_save_ops: str = ""
    """Operations to save instead of recomputing under the ``full`` policy.

    Each selector has the form ``MODULE_FQN_PATTERN::OP``. Separate multiple
    selectors with ``|`` and quote the full argument in the shell. For example:
    ``layers.*.moe.router.gate::aten.mm.default | layers.*.attention.wkv_a::aten.mm.default``.
    """

    pass_pipeline: str = "default"
    """Pass pipeline selection. Controls which graph pass pipeline, post-init
    hooks, and pre-train-step hooks are activated."""

    inductor_compilation: Literal["regional", "full"] = "regional"
    """
    Inductor compilation strategy. Mutually exclusive options:
        regional: compile tagged regions (e.g. FlexAttention HOPs) with
            regional_inductor while leaving the rest interpreted.
        full: compile the entire graph with inductor into optimized
            Triton kernels. Provides better performance but may change
            bitwise numerics compared to regional/interpreted execution.
    """

    numerics_changing_optim: bool = False
    """Enable passes that improve performance but may change numerics
    compared to the uncompiled path (e.g. RMSNorm Inductor fusion)."""

    cpu_offload_prefetch_n_layers: int = 1
    """Prefetch reloads this many layers ahead in the backward graph
    to overlap H2D transfers with compute."""

    cpu_offload_defer_n_layers: int = 1
    """Defer forward wait_tensor ops this many layers past the last consumer
    to overlap D2H transfers with compute."""

    cpu_offload_budget_gb: float = 100.0
    """Maximum CPU memory budget (in GB per rank) for offloaded activations.
    Tensors are selected largest-first until the budget is exhausted."""

    paged_stash_page_size: int = 64
    """Tokens per paged-stash page. Smaller pages waste less on stashes that do
    not fill a page, at the cost of a longer page record per activation."""

    paged_stash_buffer_size_factor_cuda: float = 1.10
    """Headroom multiplier on the CUDA paged-stash buffers, over the pages the
    measured step and the pipeline schedule say are needed. Matches Megatron's
    moe_paged_stash_buffer_size_factor_cuda."""

    paged_stash_buffer_size_factor_cpu: float = 0.0
    """Headroom multiplier for an optional pinned-host spill buffer, using the
    same page basis as the CUDA factor. 0 disables host spilling, so a full
    CUDA stash goes straight to overflow. Matches Megatron's
    moe_paged_stash_buffer_size_factor_cpu."""

    paged_stash_prefetch_n_layers: int = 1
    """Issue each paged-stash reload this many backward layers early so the page
    reads overlap with backward compute."""

    paged_stash_page_recomputed: bool = False
    """Page declared activations that SAC would otherwise recompute, notably the
    BF16 FC1 output. Megatron pages this tensor (Transformer Engine saves it);
    our SAC rebuilds it instead. Turning this on trades a grouped GEMM of
    recompute for a stash round trip and the pages to hold it -- the FC1 output
    is several times larger than the quantized operands beside it, so it is off
    by default and worth measuring on your model before enabling."""

    paged_stash_skip_immediate_backward: bool = True
    """Keep an activation resident instead of stashing it when the pipeline
    schedule runs that microbatch's backward next, so the stash would be written
    and read straight back with no bubble to hide it in. Applies to the last
    paged layer, the only one whose backward can be the next scheduled compute.
    Matches Megatron's ``remove_paged_tensor_from_stash``. Automatically
    disabled under CUDA graph capture, where a per-microbatch decision cannot be
    replayed."""

    paged_stash_overflow_check: Literal["assert", "blocking", "deferred"] = "assert"
    """How a paged-stash overflow verdict is acted on. Overflow means backward
    read activations that were never written back, so the step's gradients are
    invalid.
        assert: enqueue a device-side assertion on the all-reduced flag. Costs
            no host sync, and stream ordering places it before every optimizer
            kernel, so an invalid step can never be applied. Fatal: a fired
            device assertion ends the job.
        blocking: read the flag with .item() at every step boundary, as Megatron
            does, and rerun an overflowing step with larger buffers. Also never
            applies an invalid step, and unlike assert it recovers in process,
            but costs one device sync per step.
        deferred: read the flag from a pinned host mirror on a later step. No
            sync and no job loss, but the verdict arrives after the step's
            gradients were applied, so this is the only mode that can train on
            invalid gradients. Opt in only where losing the run to a transient
            overflow is worse than absorbing one bad step."""

    paged_stash_module_fqn: str = "layers.*.moe.routed_experts"
    """Module FQN prefix pattern selecting which activations are eligible for
    paged stashing. Matched against the leading FQN components, so the default
    covers the whole routed-expert subtree including inner_experts."""

    enable_fsdp_ag_rs_overlap: bool = False
    """When True, run ``overlap_fsdp_ag_rs_pass``. The pass moves backward
    FSDP all-gathers onto a separate CUDA stream from reduce-scatters so the
    two collectives can overlap. It is a no-op when the graph contains no
    FSDP all-gathers."""

    enable_fsdp_dense_region_overlap: bool = False
    """When True, schedule FSDP AG/RS buckets into neighboring dense regions.

    This is disabled by default because it changes FSDP collective placement
    and is intended for performance/integration validation, not
    bitwise-equivalence tests. When EP overlap is also enabled, this scheduler
    only composes with graph chunking rooted at ``layers.*.moe``; otherwise the
    explicit request is skipped with a warning. Without EP overlap, it can run
    as a standalone FSDP scheduling ablation.
    """

    ep_overlap: EpOverlapConfig = field(default_factory=EpOverlapConfig)
    """Configuration for EP-overlap chunking and scheduling."""

    precompile_artifact_dir: str = ""
    """
    Directory for precompiled artifacts. Setting this enables precompile:
    precompile_main.py saves the artifact here, and training loads it from
    here to skip compilation. For multi-node setups use a shared filesystem
    path.
    """

    enable_autoparallel: bool = False
    """Use AutoParallelGraph (ILP solver-based SPMD sharding) instead of
    manual TP/FSDP/EP. Forces the AOT compilation path internally."""


def validate_autoparallel_config(
    compile_config: GraphTrainerCompileConfig,
) -> None:
    if compile_config.enable_autoparallel and compile_config.mode != "aot_fx_trace":
        raise ValueError(
            "AutoParallel graph_trainer integration only supports "
            "--compile.mode aot_fx_trace"
        )


def validate_ep_overlap_config(
    ep_overlap_config: EpOverlapConfig,
) -> tuple[EpOverlapChunkDim, EpOverlapChunkStrategy, str]:
    chunk_dim = ep_overlap_config.chunk_dim
    if chunk_dim not in ("batch", "seq"):
        raise ValueError(
            "--compile.ep_overlap.chunk_dim must be 'batch' or 'seq' when "
            "--compile.ep_overlap.enabled is set"
        )

    chunk_strategy = ep_overlap_config.strategy
    if chunk_strategy not in ("eager", "graph"):
        raise ValueError(
            "--compile.ep_overlap.strategy must be 'eager' or 'graph' when "
            "--compile.ep_overlap.enabled is set"
        )

    module_fqn = ep_overlap_config.module_fqn
    if module_fqn not in SUPPORTED_EP_OVERLAP_MODULE_FQNS:
        raise ValueError(
            "--compile.ep_overlap.module_fqn must be either 'layers.*' "
            "or 'layers.*.moe' for ep_overlap"
        )
    if chunk_dim == "seq" and module_fqn != MOE_BLOCK_FQN:
        raise ValueError(
            "--compile.ep_overlap.chunk_dim seq is only supported with "
            "--compile.ep_overlap.module_fqn layers.*.moe"
        )

    return chunk_dim, chunk_strategy, module_fqn


def trace_input_preparer_keys(
    compile_config: GraphTrainerCompileConfig,
) -> list[str]:
    """Return feature names whose trace-input hooks should run.

    ``compile.passes`` remains the escape hatch for standalone graph passes.
    EP overlap has structured config because enabling it also controls eager
    module wrapping and later scheduling behavior.
    """
    names = list(compile_config.passes)
    if compile_config.ep_overlap.enabled:
        names.append("ep_overlap")
    return list(dict.fromkeys(names))


def to_graph_trainer_config(
    base_config: Trainer.Config,
    model_registry: Callable[[str], ModelSpec],
) -> "GraphTrainer.Config":
    """Convert a base Trainer.Config to a GraphTrainer.Config.

    Copies all fields from the base config and replaces the model_spec with one
    from the graph_trainer model_registry. The compile field is removed and
    left as the GraphTrainer.Config default; callers should explicitly set it.
    """
    from .trainer import GraphTrainer

    d = {f.name: getattr(base_config, f.name) for f in fields(base_config)}
    d["parallelism"] = replace(
        base_config.parallelism,
        spmd_backend="spmd_types",
    )
    graph_spec = model_registry(base_config.model_spec.flavor)
    # Wrap the base model config in the graph_trainer's model config class
    # (e.g. GraphTrainerQwen3Model.Config) while preserving all field values
    # (including moe_comm_backend etc.).
    graph_model_cls = type(graph_spec.model)
    graph_model = graph_model_cls(
        **{
            f.name: getattr(base_config.model_spec.model, f.name)
            for f in fields(base_config.model_spec.model)
        }
    )
    d["model_spec"] = replace(
        base_config.model_spec,
        parallelize_fn=graph_spec.parallelize_fn,
        pipelining_fn=graph_spec.pipelining_fn,
        model=graph_model,
    )
    d.pop("compile")

    # graph_trainer uses graph-based SAC instead of eager AC. Override any
    # enabled AC policy with the default selective one so callers don't need
    # per-config fixups.
    ac = d.get("activation_checkpoint")
    if ac is not None:
        d["activation_checkpoint"] = SelectiveAC.Config()

    # graph_trainer's tracer requires explicit autograd outputs for lm_head
    # params instead of relying on .grad side effects from chunk_loss.backward().
    loss_cfg = d.get("loss")
    if isinstance(loss_cfg, ChunkedLossWrapper.Config):
        d["loss"] = ChunkedLossWrapperWithParamGrads.Config(
            **{f.name: getattr(loss_cfg, f.name) for f in fields(loss_cfg)}
        )

    return GraphTrainer.Config(**d)
