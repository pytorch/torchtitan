# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from typing import Literal

from torchtitan.config import ActivationCheckpointConfig
from torchtitan.config.configs import CompileConfig
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer


@dataclass(kw_only=True, slots=True)
class GraphTrainerCompileConfig(CompileConfig):
    mode: Literal["jit", "aot", "aot_fx_trace"] | None = "aot"
    """
    Compilation mode. Options:
        jit: standard torch.compile() with custom backend
        aot: explicit joint graph export + custom graph passes
        aot_fx_trace: non-strict tracing of fwd+loss+bwd via make_fx
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

    enable_cudagraph: bool = True
    """When False, skip the cudagraph pass even if the graph is compatible."""

    memory_budget: float = 0.5
    """
    Memory budget for ILP-based SAC (selective activation checkpointing).
    Only used when "apply_ilp_sac" is in compile.joint_passes. Controls
    the fraction of total activation memory that can be saved (0.0-1.0).
    0.0 = recompute everything, 1.0 = save everything.
    """

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

    # Merge CUDA graph kernel annotations into profiler traces when profiling
    # is active.  No-op otherwise (and no-op when requirements aren't met).
    # It's also a no-op if there is CUDA graph is not enabled.
    profiler = d.get("profiler")
    if profiler is not None:
        profiler.trace_post_processor = cudagraph_annotate_trace_post_processor()

    return GraphTrainer.Config(**d)
