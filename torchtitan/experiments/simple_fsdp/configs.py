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
class SimpleFSDPCompileConfig(CompileConfig):
    graph_passes: Literal["auto_bucketing", "transformer_block_bucketing"] | None = None
    """
    Bucketing and overlapping passes in simplefsdp. Additional passes include:
        auto_bucketing, transformer_block_bucketing
    """


@dataclass(kw_only=True, slots=True)
class SimpleFSDPConfig(Trainer.Config):
    compile: SimpleFSDPCompileConfig = field(default_factory=SimpleFSDPCompileConfig)


def to_simple_fsdp_config(
    base_config: Trainer.Config,
    model_registry: Callable[[str], ModelSpec],
) -> SimpleFSDPConfig:
    """Convert a base Trainer.Config to a SimpleFSDPConfig.

    Copies all fields from the base config and replaces the model_spec with one
    from the simple_fsdp model_registry. The compile field is removed and left
    as the SimpleFSDPConfig default; callers should explicitly set it.
    """
    d = {f.name: getattr(base_config, f.name) for f in fields(base_config)}
    d["model_spec"] = model_registry(base_config.model_spec.flavor)
    d.pop("compile")

    return SimpleFSDPConfig(**d)
