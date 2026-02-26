# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass, field, fields

from torchtitan.config.configs import CompileConfig
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer


@dataclass(kw_only=True, slots=True)
class CompilerToolkitCompileConfig(CompileConfig):
    joint_passes: list[str] = field(default_factory=list)
    """Joint graph pass names to apply on the joint forward-backward
    graph before partitioning."""

    passes: list[str] = field(default_factory=list)
    """Compiler pass names to apply to the partitioned forward/backward graphs."""


def to_compiler_toolkit_config(
    base_config: Trainer.Config,
    model_registry: Callable[[str], ModelSpec],
):
    """Convert a base Trainer.Config to a CompilerToolkitTrainer.Config.

    Copies all fields from the base config and replaces the model_spec with one
    from the compiler_toolkit model_registry. The compile field is removed and
    left as the default; callers should explicitly set it.
    """
    from torchtitan.experiments.compiler_toolkit.trainer import CompilerToolkitTrainer

    d = {f.name: getattr(base_config, f.name) for f in fields(base_config)}
    d["model_spec"] = model_registry(base_config.model_spec.flavor)
    d.pop("compile")

    return CompilerToolkitTrainer.Config(**d)
