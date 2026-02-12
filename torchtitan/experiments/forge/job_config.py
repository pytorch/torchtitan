# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict, dataclass, field
from typing import Any

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    ActivationCheckpointConfig,
    CommConfig,
    CompileConfig,
    DebugConfig,
    ModelConfig,
    ParallelismConfig,
    TrainingConfig,
)


# Parity w/ TorchTitan commit: 8ec37d2bca7ac9d3f7517ba70ac10e75e22a7bcb
@dataclass
class ForgeJobConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizersContainer.Config = field(
        default_factory=OptimizersContainer.Config
    )
    lr_scheduler: LRSchedulersContainer.Config = field(
        default_factory=LRSchedulersContainer.Config
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    checkpoint: CheckpointManager.Config = field(
        default_factory=CheckpointManager.Config
    )
    activation_checkpoint: ActivationCheckpointConfig = field(
        default_factory=ActivationCheckpointConfig
    )
    compile: CompileConfig = field(default_factory=CompileConfig)
    comm: CommConfig = field(default_factory=CommConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
