# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
from dataclasses import dataclass, field

from torchtitan.config.configs import CompileConfig
from torchtitan.trainer import Trainer


@dataclass(kw_only=True, slots=True)
class CompilerToolkitCompileConfig(CompileConfig):
    joint_passes: list[str] = field(default_factory=list)
    """Joint graph pass names to apply on the joint forward-backward
    graph before partitioning."""

    passes: list[str] = field(default_factory=list)
    """Compiler pass names to apply to the partitioned forward/backward graphs."""


class CompilerToolkitTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        compile: CompilerToolkitCompileConfig = field(
            default_factory=CompilerToolkitCompileConfig
        )

    def close(self) -> None:
        super().close()

        # Note [explicit cudagraph close]
        # cudagraph holds reference to nccl which prevents destroy nccl
        # group. so we need to explicitly delete cudagraph which is held
        # in joint_graph_module. An explicit gc.collect() is necessary
        # to clean up reference cycles.
        for part in self.model_parts:
            if hasattr(part, "joint_graph_module"):
                part.joint_graph_module = None
        gc.collect()
