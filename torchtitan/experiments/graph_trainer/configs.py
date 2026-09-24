# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field, fields
from typing import Literal

from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.config.configs import CompileConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.graph_trainer.chunked_loss import (
    ChunkedLossWrapperWithParamGrads,
)
from torchtitan.protocols.model import BaseModel
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

    GraphPP still runs mandatory pre-partition or pre-extraction normalization
    passes because its partitioning and extraction contracts depend on
    canonical graph structure.
    """

    fsdp_param_unshard_mode: Literal[
        "auto", "in_graph", "extracted_in_schedule_stage"
    ] = "auto"
    """Choose where FSDP parameter all-gathers run.

    - ``auto``
        - PP=1 without gradient accumulation: all-gathers inside
          ``FULL_FORWARD_BACKWARD``
        - PP=1 with gradient accumulation: explicit ``UNSHARD``
        - PP>1: explicit ``UNSHARD``
    - ``in_graph``
        - PP=1: all-gathers inside ``FULL_FORWARD_BACKWARD``
        - PP>1: error
        - Keep all-gathers inside ``FULL_FORWARD_BACKWARD`` to be able to
          immediately deallocate them after their last use and get lower peak
          memory
    - ``extracted_in_schedule_stage``
        - PP=1 and PP>1: explicit ``UNSHARD``
        - Commonly used for gradient accumulation and PP to run ``UNSHARD``
          once at the first microbatch. This is achieved by extracting
          ``UNSHARD`` (all-gathers) into a schedule stage and running it once
          in GraphRuntime
    """

    fsdp_gradient_sync_mode: Literal[
        "auto", "in_graph", "deferred_as_schedule_stage"
    ] = "auto"
    """Choose where FSDP gradient reduction runs.

    - ``auto``
        - PP=1 without gradient accumulation: reduction inside
          ``FULL_FORWARD_BACKWARD``
        - PP=1 with gradient accumulation: explicit ``REDUCE_GRAD``
        - PP>1: explicit ``REDUCE_GRAD``
    - ``in_graph``
        - PP=1: gradient reduction inside ``FULL_FORWARD_BACKWARD``
        - PP>1: error
        - Keep reduce-scatters inside ``FULL_FORWARD_BACKWARD`` to be able to
          immediately deallocate them after their last use and get lower peak
          memory
    - ``deferred_as_schedule_stage``
        - PP=1 and PP>1: explicit ``REDUCE_GRAD``
        - Commonly used for gradient accumulation and PP to run
          ``REDUCE_GRAD`` once at the last microbatch. This is achieved by
          extracting ``REDUCE_GRAD`` (reduce-scatters) into a schedule stage
          and running it once in GraphRuntime
    """

    gradient_accumulation_mode: Literal["auto", "runtime", "in_graph"] = "auto"
    """Choose where gradients accumulate across schedule microbatches.

    - ``auto``
        - PP=1: in-graph for WGrad fusion or supported multi-microbatch schedules
        - PP>1: runtime
    - ``runtime``
        - PP=1 and PP>1: accumulate backward outputs in ``GraphRuntime``
    - ``in_graph``
        - PP=1: accumulate into persistent graph inputs
        - PP>1: error
    """

    gradient_accum_in_wgrad_fusion: Literal["auto", "disabled", "enabled"] = "auto"
    """Control fusion of WGrad producers with gradient accumulation.

    - ``auto``
        - In-graph accumulation with ``numerics_changing_optim``: fuse
          supported WGrad producers
        - Otherwise: explicit accumulation
    - ``disabled``
        - Keep explicit accumulation
    - ``enabled``
        - PP=1: enable in-graph accumulation and fuse supported WGrad producers
        - PP>1: error
    """

    disable_passes: list[str] = field(default_factory=list)
    """Pass names to selectively disable for debugging and ablation
    studies. A pass is skipped if its name exactly matches any entry.
    Example: --compile.disable_passes custom_codegen_pass,cuda_graph_pass"""

    debug_graph_passes: bool = False
    """Log timing, op-count diffs, and before/after graphs for each pass to tlparse."""

    memory_policy: Literal[
        "none", "default", "full", "eager", "min_cut", "sac_and_offload"
    ] = "default"
    """
    Memory optimization policy for activation management (SAC, offload).
        none: save forward activations without rematerialization.
        default: SAC — save all compute-intensive ops and FSDP all_gathers.
        full: full recompute, saving layer outputs and operations selected by
            full_recompute_save_ops. With no selectors, this mirrors eager's
            full AC (checkpoint_wrapper with no context_fn).
        eager: SAC alternating mm ops between save/recompute, matching the
            eager AC policy in torchtitan.distributed.activation_checkpoint.
        min_cut: choose saved activations with the min-cut partitioner.
        sac_and_offload: SAC + CPU offload — apply default SAC first,
            then offload surviving MUST_SAVE activations to CPU within
            the cpu_offload_budget_gb budget.
    """

    full_recompute_save_ops: str = ""
    """Operations to save instead of recomputing under the ``full`` policy.

    Each selector has the form ``MODULE_FQN_PATTERN::OP``. Separate multiple
    selectors with ``|`` and quote the full argument in the shell. For example:
    ``layers.*.moe.router.gate::aten.mm.dtype | layers.*.attention.wkv_a::aten.mm.default``.
    """

    pass_pipeline: str = "default"
    """Pass pipeline selection. Controls which graph pass pipeline, post-init
    hooks, and pre-train-step hooks are activated."""

    inductor_compilation: Literal["regional", "full"] = "regional"
    """
    Inductor compilation strategy. Mutually exclusive options:
        regional: compile tagged regions (e.g. FlexInnerAttention HOPs) with
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
    model_config_cls: type[BaseModel.Config],
) -> "GraphTrainer.Config":
    """Convert a base Trainer.Config to a GraphTrainer.Config.

    Copies all fields from the base config and converts its model config to the
    GraphTrainer model config class. The compile field is removed and left as
    the GraphTrainer.Config default; callers should explicitly set it.
    """
    from .trainer import GraphTrainer

    d = {f.name: getattr(base_config, f.name) for f in fields(base_config)}
    graph_model = model_config_cls(
        **{
            f.name: getattr(base_config.model, f.name)
            for f in fields(base_config.model)
        }
    )
    d["model"] = graph_model
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
