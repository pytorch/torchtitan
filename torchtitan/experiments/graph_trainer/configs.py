# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from typing import Annotated, Literal, Union

import tyro

from torchtitan.components.loss import ChunkedCELoss, CrossEntropyLoss
from torchtitan.config import ActivationCheckpointConfig
from torchtitan.config.configs import CompileConfig
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer


# ---------------------------------------------------------------------------
# Memory policy config dataclasses (built-in)
# ---------------------------------------------------------------------------


@dataclass(kw_only=True, slots=True)
class DefaultMemoryPolicy:
    """Default SAC: save all compute-intensive ops and FSDP all_gathers."""

    pass


@dataclass(kw_only=True, slots=True)
class EagerMemoryPolicy:
    """Eager SAC: alternate mm ops between save/recompute."""

    pass


@dataclass(kw_only=True, slots=True)
class CpuOffloadAllMemoryPolicy:
    """Offload all eligible activations to CPU (WIP)."""

    pass


# ---------------------------------------------------------------------------
# Memory policy config registration
# ---------------------------------------------------------------------------

_MEMORY_POLICY_TYPES: list = [
    Annotated[DefaultMemoryPolicy, tyro.conf.subcommand("default")],
    Annotated[EagerMemoryPolicy, tyro.conf.subcommand("eager")],
    Annotated[CpuOffloadAllMemoryPolicy, tyro.conf.subcommand("cpu_offload_all")],
]


def register_memory_policy_config(name: str):
    """Register a memory policy config dataclass for CLI selection.

    Use as a decorator on the policy dataclass. Call this before tyro parses
    CLI args (i.e., at module import time).
    The registered type becomes available as ``compile.memory_policy:<name>``.
    """

    def decorator(cls: type) -> type:
        _MEMORY_POLICY_TYPES.append(Annotated[cls, tyro.conf.subcommand(name)])
        # Patch the annotation on GraphTrainerCompileConfig so tyro sees the
        # updated Union when it parses CLI args.
        GraphTrainerCompileConfig.__annotations__["memory_policy"] = Union[
            tuple(_MEMORY_POLICY_TYPES)
        ]
        return cls

    return decorator


@dataclass(kw_only=True, slots=True)
class GraphTrainerCompileConfig(CompileConfig):
    mode: Literal["jit", "aot", "aot_fx_trace"] | None = "aot_fx_trace"
    """
    Compilation mode. Options:
        aot_fx_trace: non-strict tracing of fwd+loss+bwd via make_fx
        jit: standard torch.compile() with custom backend (deprecated)
        aot: explicit joint graph export + custom graph passes (deprecated)
    """

    backend: str = "aot_eager"

    passes: list[str] = field(default_factory=list)
    """
    Compiler pass names to apply.
    In JIT mode: applied as graph passes (e.g., auto_bucketing, transformer_block_bucketing)
    In AOT mode: applied to the partitioned forward/backward graphs
    """

    joint_passes: list[str] = field(default_factory=list)
    """Joint graph pass names to apply on the joint forward-backward
    graph before partitioning. Only used in AOT mode."""

    enable_passes: bool = True
    """When False, skip all graph passes (both default and user-configured)."""

    debug_graph_passes: bool = False
    """Log timing, op-count diffs, and before/after graphs for each pass to tlparse."""

    memory_policy: Union[tuple(_MEMORY_POLICY_TYPES)] = field(
        default_factory=DefaultMemoryPolicy
    )
    """
    Memory optimization policy for activation management (SAC, offload, etc.).

    Built-in policies:
        default: save all compute-intensive ops and FSDP all_gathers.
        eager: alternate mm ops between save/recompute, matching the eager
            AC policy in torchtitan.distributed.activation_checkpoint.
        cpu_offload_all: offload all eligible activations to CPU.
            Work in progress — for development and testing only.

    Additional policies can be registered via ``register_memory_policy_config``
    in ``graph_trainer/configs.py``. Use tyro subcommand syntax to select::

        compile.memory_policy:default
        compile.memory_policy:paged_stash
        compile.memory_policy:paged_stash --compile.memory_policy.buffer_size_factor 0.3
    """

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

    enable_cudagraph: bool = True
    """When False, skip the cudagraph pass even if the graph is compatible."""

    precompile_artifact_dir: str = ""
    """
    Directory for precompiled artifacts. Setting this enables precompile:
    precompile_main.py saves the artifact here, and training loads it from
    here to skip compilation. For multi-node setups use a shared filesystem
    path.
    """


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
