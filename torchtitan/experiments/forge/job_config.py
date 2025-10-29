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
    Debug,
    Job,
    LRScheduler,
    MemoryEstimation,
    Model,
    Optimizer,
    Parallelism,
    Quantize,
    Training,
)


# Parity w/ TorchTitan commit: 8ec37d2bca7ac9d3f7517ba70ac10e75e22a7bcb
@dataclass
class ForgeJobConfig:
    job: Job = field(default_factory=Job)
    # profiling: Profiling = field(default_factory=Profiling)
    # metrics: Metrics = field(default_factory=Metrics)
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
    quantize: Quantize = field(default_factory=Quantize)
    comm: Comm = field(default_factory=Comm)
    memory_estimation: MemoryEstimation = field(default_factory=MemoryEstimation)
    # fault_tolerance: FaultTolerance = field(default_factory=FaultTolerance)
    # experimental: Experimental = field(default_factory=Experimental)
    # validation: Validation = field(default_factory=Validation)
    debug: Debug = field(default_factory=Debug)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
