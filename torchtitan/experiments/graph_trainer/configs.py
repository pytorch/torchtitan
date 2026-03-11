# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import Literal

from torchtitan.config.configs import CompileConfig
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer


@dataclass(kw_only=True, slots=True)
class GraphTrainerCompileConfig(CompileConfig):
    mode: Literal["jit", "aot"] | None = "aot"
    """
    Compilation mode. Options:
        jit: standard torch.compile() with custom backend
        aot: explicit joint graph export + custom graph passes
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


@dataclass(kw_only=True, slots=True)
class GraphTrainerConfig(Trainer.Config):
    compile: GraphTrainerCompileConfig = field(
        default_factory=GraphTrainerCompileConfig
    )


def to_graph_trainer_config(
    base_config: Trainer.Config,
    model_registry: Callable[[str], ModelSpec],
) -> GraphTrainerConfig:
    """Convert a base Trainer.Config to a GraphTrainerConfig.

    Copies all fields from the base config and replaces the model_spec with one
    from the graph_trainer model_registry. The compile field is removed and
    left as the GraphTrainerConfig default; callers should explicitly set it.
    """
    from .trainer import GraphTrainer

    d = {f.name: getattr(base_config, f.name) for f in fields(base_config)}
    d["model_spec"] = model_registry(base_config.model_spec.flavor)
    d.pop("compile")

    return GraphTrainer.Config(**d)
