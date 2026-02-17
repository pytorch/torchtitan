# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Literal

from torchtitan.config.configs import CompileConfig
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
