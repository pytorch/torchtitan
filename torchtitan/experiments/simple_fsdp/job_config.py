# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Compile:
    graph_passes: Literal["auto_bucketing", "transformer_block_bucketing"] | None = None
    """
    Bucketing and overlapping passes in simplefsdp. Additional passes include:
        auto_bucketing, transformer_block_bucketing
    """


@dataclass
class JobConfig:
    compile: Compile = field(default_factory=Compile)
