# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from torchtitan.config.configs import CompileConfig
from torchtitan.trainer import Trainer


@dataclass(kw_only=True, slots=True)
class AutoParallelCompileConfig(CompileConfig):
    comms_bucket_reorder_strategy: str = "aten"
    """Options: "aten" (default), "inductor", "none" """

    autop_force_bf16: bool = False


@dataclass(kw_only=True, slots=True)
class AutoParallelConfig(Trainer.Config):
    compile: AutoParallelCompileConfig = field(
        default_factory=AutoParallelCompileConfig
    )
