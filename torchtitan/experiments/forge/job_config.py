# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict, dataclass, field
from typing import Any

from torchtitan.config.job_config import (
    ActivationCheckpoint,
    Checkpoint,
    Comm,
    Compile,
    FP8LinearConfig,
    LRScheduler,
    Model,
    Optimizer,
    Parallelism,
    Training,
)


@dataclass
class ForgeJobConfig:
    model: Model = field(default_factory=Model)
    optimizer: Optimizer = field(default_factory=Optimizer)
    lr_scheduler: LRScheduler = field(default_factory=LRScheduler)
    training: Training = field(default_factory=Training)
    parallelism: Parallelism = field(default_factory=Parallelism)
    checkpoint: Checkpoint = field(default_factory=Checkpoint)
    activation_checkpoint: ActivationCheckpoint = field(
        default_factory=ActivationCheckpoint
    )
    compile: Compile = field(default_factory=Compile)
    float8: FP8LinearConfig = field(default_factory=FP8LinearConfig)
    comm: Comm = field(default_factory=Comm)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
