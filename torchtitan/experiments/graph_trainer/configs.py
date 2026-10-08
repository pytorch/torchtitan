# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field, fields
from typing import Literal

from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.graph_trainer.chunked_loss import (
    ChunkedLossWrapperWithParamGrads,
)
from torchtitan.protocols.model import BaseModel
from torchtitan.trainer import Trainer

EpOverlapChunkDim = Literal["batch", "seq"]

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

    module_fqn: str = TRANSFORMER_BLOCK_FQN
    """Single module FQN pattern chunked for EP overlap.

    v1 supports all transformer blocks (``layers.*``) or all MoE blocks
    (``layers.*.moe``). Selected module forwards are wrapped with eager chunking
    before tracing, and the overlap scheduler consumes the resulting chunk
    metadata.
    """


@dataclass(kw_only=True, slots=True)
class SPMDGradientAccumulationConfig:
    """Settings for SPMD with gradient accumulation.

    SPMD with gradient accumulation runs more than one microbatch per step
    without pipeline parallelism. The ``fsdp_*`` fields additionally require
    FSDP. Otherwise these settings are ignored with a warning: SPMD without
    gradient accumulation keeps FSDP collectives inside
    ``FULL_FORWARD_BACKWARD``, and PP always runs them as explicit
    ``UNSHARD`` and ``REDUCE_GRAD`` schedule actions without WGrad
    accumulation fusion.
    """

    fsdp_param_unshard_mode: Literal[
        "every_microbatch", "first_microbatch"
    ] = "first_microbatch"
    """Choose where FSDP parameter all-gathers run.

    - ``every_microbatch``
        - All-gathers inside each joint microbatch graph, so unsharded
          parameters can be freed after their last use for lower peak memory
    - ``first_microbatch``
        - All-gathers inside the first ``FORWARD_BACKWARD_FIRST_WITH_UNSHARD``
          graph; later microbatches reuse the unsharded parameters

    ``first_microbatch`` keeps parameters unsharded until the end of the step,
    which implies ``parallelism.fsdp_reshard_after_forward`` = ``never`` for
    the compiled graphs.
    """

    fsdp_grad_reduce_mode: Literal[
        "every_microbatch", "last_microbatch"
    ] = "last_microbatch"
    """Choose where FSDP gradient reduction runs.

    - ``every_microbatch``
        - Reduce-scatters inside each joint microbatch graph, so unsharded
          gradients can be freed immediately for lower peak memory
    - ``last_microbatch``
        - Reduction inside the last ``FORWARD_BACKWARD_LAST_WITH_REDUCE_GRAD``
          graph after accumulating all microbatches

    ``first_microbatch`` unsharding cannot be combined with
    ``every_microbatch`` reduction, and ``every_microbatch`` unsharding cannot
    be combined with ``last_microbatch`` reduction.
    """

    fuse_wgrad_accumulation: Literal["auto", "disabled", "enabled"] = "auto"
    """Control fusion of WGrad producers with gradient accumulation.

    - ``auto``
        - Fuse supported WGrad producers when
          ``compile.numerics_changing_optim`` is set; otherwise keep explicit
          accumulation
    - ``disabled``
        - Keep explicit accumulation
    - ``enabled``
        - Fuse supported WGrad producers

    With FSDP, fusion requires ``fsdp_grad_reduce_mode`` = ``last_microbatch``.
    """


@dataclass(kw_only=True, slots=True)
class GraphTrainerCompileConfig:
    enable_async_tensor_parallel: bool = False
    """Whether to pipeline tensor-parallel collectives with matrix multiplications."""

    passes: list[str] = field(default_factory=list)
    """
    Additional compiler pass names to apply or prepare inputs for.
    """

    enable_passes: bool = True
    """When False, skip optional graph passes (both default and user-configured).

    GraphPP still runs mandatory pre-partition or pre-extraction normalization
    passes because its partitioning and extraction contracts depend on
    canonical graph structure.
    """

    spmd_gradient_accumulation: SPMDGradientAccumulationConfig = field(
        default_factory=SPMDGradientAccumulationConfig
    )
    """Settings for SPMD with gradient accumulation."""

    disable_passes: list[str] = field(default_factory=list)
    """Pass names to selectively disable for debugging and ablation
    studies. A pass is skipped if its name exactly matches any entry.
    Example: ``["custom_codegen_pass", "cuda_graph_pass"]``."""

    memory_policy: Literal[
        "none", "default", "full", "eager", "min_cut", "sac_and_offload"
    ] = "default"
    """
    Memory optimization policy for activation management (SAC, offload).
        none: save forward activations without rematerialization.
        default: SAC — save all compute-intensive ops and FSDP all_gathers.
        full: full recompute, saving layer outputs and operations selected by
            full_recompute_save_ops. With no selectors, this mirrors eager's
            the former eager full-AC implementation.
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
    """Pass pipeline selection. Selects a graph pass pipeline registered in
    ``PASS_PIPELINE_REGISTRY``."""

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
    bitwise-equivalence tests. It does not compose with EP overlap: when
    ``ep_overlap.enabled`` is set, the explicit request is skipped with a
    warning. Without EP overlap, it can run as a standalone FSDP scheduling
    ablation.
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
    manual TP/FSDP/EP."""


def validate_ep_overlap_config(
    ep_overlap_config: EpOverlapConfig,
) -> tuple[EpOverlapChunkDim, str]:
    chunk_dim = ep_overlap_config.chunk_dim
    if chunk_dim not in ("batch", "seq"):
        raise ValueError(
            "compile.ep_overlap.chunk_dim must be 'batch' or 'seq' when "
            "compile.ep_overlap.enabled is set"
        )

    module_fqn = ep_overlap_config.module_fqn
    if module_fqn not in SUPPORTED_EP_OVERLAP_MODULE_FQNS:
        raise ValueError(
            "compile.ep_overlap.module_fqn must be either 'layers.*' "
            "or 'layers.*.moe' for ep_overlap"
        )
    if chunk_dim == "seq" and module_fqn != MOE_BLOCK_FQN:
        raise ValueError(
            "compile.ep_overlap.chunk_dim='seq' is only supported with "
            "compile.ep_overlap.module_fqn='layers.*.moe'"
        )

    return chunk_dim, module_fqn


def to_graph_trainer_config(
    base_config: Trainer.Config,
    model_config_cls: type[BaseModel.Config],
) -> "GraphTrainer.Config":
    """Convert a base Trainer.Config to a GraphTrainer.Config.

    Copies all fields from the base config and converts its model config to the
    GraphTrainer model config class, without local compile regions because
    GraphTrainer traces the whole step. The ``compile`` field keeps the
    GraphTrainer.Config default; callers should explicitly set it.
    """
    from .trainer import GraphTrainer

    d = {f.name: getattr(base_config, f.name) for f in fields(base_config)}
    graph_model = model_config_cls(
        **{
            f.name: getattr(base_config.model, f.name)
            for f in fields(base_config.model)
        }
    )
    # GraphTrainer compiles the whole step (config.compile), so it drops the model's local compile regions.
    graph_model.local_compile_regions = []
    d["model"] = graph_model

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
