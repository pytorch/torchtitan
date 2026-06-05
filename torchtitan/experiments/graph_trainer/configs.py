# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from typing import Literal

from torchtitan.components.loss import ChunkedCELoss, CrossEntropyLoss
from torchtitan.config import ActivationCheckpointConfig
from torchtitan.config.configs import CompileConfig
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer


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
    Compiler pass names to apply.
    In JIT mode: applied as graph passes (e.g., auto_bucketing, transformer_block_bucketing)
    """

    enable_passes: bool = True
    """When False, skip all graph passes (both default and user-configured)."""

    disable_passes: list[str] = field(default_factory=list)
    """Pass names to selectively disable for debugging and ablation
    studies. A pass is skipped if its name exactly matches any entry.
    Example: --compile.disable_passes custom_codegen_pass,cudagraph_pass"""

    debug_graph_passes: bool = False
    """Log timing, op-count diffs, and before/after graphs for each pass to tlparse."""

    memory_policy: Literal["default", "full", "eager", "sac_and_offload"] = "default"
    """
    Memory optimization policy for activation management (SAC, offload).
        default: SAC — save all compute-intensive ops and FSDP all_gathers.
        full: full recompute — only layer outputs are saved. Mirrors
            eager's full AC (checkpoint_wrapper with no context_fn).
        eager: SAC alternating mm ops between save/recompute, matching the
            eager AC policy in torchtitan.distributed.activation_checkpoint.
        sac_and_offload: SAC + CPU offload — apply default SAC first,
            then offload surviving MUST_SAVE activations to CPU within
            the cpu_offload_budget_gb budget.
    """

    pass_pipeline: str = "default"
    """Pass pipeline selection. Controls which graph pass pipeline, post-init
    hooks, and pre-train-step hooks are activated."""

    inductor_compilation: Literal["regional", "full"] = "regional"
    """
    Inductor compilation strategy — how ops are *code-generated* (orthogonal to
    ``cudagraph_mode``, which controls how the result is *captured* for replay):
        regional: compile tagged regions (e.g. FlexAttention HOPs) with
            regional_inductor while leaving the rest interpreted. The FX graph
            stays partitionable, so all ``cudagraph_mode`` options apply.
        full: compile the entire graph with inductor into optimized Triton
            kernels (one opaque program). Better performance but may change
            bitwise numerics. CUDA graphs for this path are handled by **Inductor
            internally** (it has its own cudagraph support), so ``cudagraph_mode``
            is ignored here — there is no FX graph left for us to wrap or
            partition.
    """

    cudagraph_mode: Literal["off", "auto", "full"] = "auto"
    """
    CUDA graph capture strategy for the ``inductor_compilation=regional`` path
    (ignored when ``inductor_compilation=full``, which uses Inductor's own
    cudagraphs):
        off: no CUDA graph.
        auto (default): best-effort. Capture the whole joint graph as one CUDA
            graph if it is cudagraph-compatible; otherwise partition it and capture
            the compatible regions (running incompatible regions such as the MoE
            block eagerly between replays); if nothing is capturable, run without
            CUDA graph. This is what enables CUDA graph for DeepSeek-V3 on H100
            (the MoE block is captured piecewise). Note: ``auto`` does not silently
            fall back to eager if a region deemed capturable nonetheless fails to
            capture at runtime (a gap in the compatibility checks) — that surfaces
            as an error; fix the check or use ``off``.
        full: capture the entire joint fwd+bwd graph as one CUDA graph. **Raises**
            if the graph is not fully cudagraph-compatible (e.g. MoE _grouped_mm on
            sm_90 / H100, EP token-routing splits, unpinned CPU<->CUDA copies,
            data-dependent shapes). Use this to assert/require a single full
            capture.

    Mental model: a full CUDA graph is just the special case where the whole graph
    is one capturable piece; ``auto`` produces that when possible and otherwise
    falls back to capturing the safe pieces.
    """

    cudagraph_min_capture_size: int = 8
    """Piecewise cudagraph only: a capturable region with fewer than this many ops
    runs eager instead of being captured. A standalone CUDA graph (private memory
    pool + per-step input copy-in) is not worth it for a handful of ops -- e.g. a
    lone op stranded between two eager MoE blocks. Raising it captures fewer, larger
    regions (less per-segment overhead/memory); 1 captures every safe region."""

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


def to_graph_trainer_config(
    base_config: Trainer.Config,
    model_registry: Callable[[str], ModelSpec],
) -> "GraphTrainer.Config":
    """Convert a base Trainer.Config to a GraphTrainer.Config.

    Copies all fields from the base config and replaces the model_spec with one
    from the graph_trainer model_registry. The compile field is removed and
    left as the GraphTrainer.Config default; callers should explicitly set it.
    """
    from .cudagraph import cudagraph_annotate_trace_post_processor
    from .trainer import GraphTrainer

    d = {f.name: getattr(base_config, f.name) for f in fields(base_config)}
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
        model=graph_model,
    )
    d.pop("compile")

    # graph_trainer uses graph-based SAC instead of eager AC. Override any
    # non-"none" AC mode to "selective" so callers don't need per-config fixups.
    ac = d.get("activation_checkpoint")
    if ac is not None and ac.mode != "none":
        d["activation_checkpoint"] = ActivationCheckpointConfig(mode="selective")

    # TODO: graph_trainer doesn't yet support ChunkedCELoss
    if isinstance(d.get("loss"), ChunkedCELoss.Config):
        d["loss"] = CrossEntropyLoss.Config()

    # Merge CUDA graph kernel annotations into profiler traces when profiling
    # is active.  No-op otherwise (and no-op when requirements aren't met).
    # It's also a no-op if there is CUDA graph is not enabled.
    profiler = d.get("profiler")
    if profiler is not None:
        profiler.trace_post_processor = cudagraph_annotate_trace_post_processor()

    return GraphTrainer.Config(**d)
