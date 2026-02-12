# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: this config is a preset for 64 A100 GPUs.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    ActivationCheckpointConfig,
    JobConfig,
    ModelConfig,
    ParallelismConfig,
    TrainingConfig,
    ValidationConfig,
)
from torchtitan.tools.profiling import ProfilingConfig
from torchtitan.trainer import Trainer

default_config = Trainer.Config(
    job=JobConfig(
        description="Llama 3 70B training",
    ),
    profiling=ProfilingConfig(
        enable_profiling=True,
        profile_freq=100,
    ),
    metrics=MetricsProcessor.Config(
        enable_tensorboard=True,
    ),
    model=ModelConfig(
        name="llama3",
        flavor="70B",
        hf_assets_path="./assets/hf/Llama-3.1-70B",
    ),
    optimizer=OptimizersContainer.Config(lr=1.5e-4),
    training=TrainingConfig(
        local_batch_size=8,
        seq_len=8192,
        steps=1000,
        dataset="c4",
    ),
    parallelism=ParallelismConfig(
        tensor_parallel_degree=8,
    ),
    checkpoint=CheckpointManager.Config(interval=500),
    activation_checkpoint=ActivationCheckpointConfig(mode="full"),
    validation=ValidationConfig(
        dataset="c4_validation",
        freq=500,
        steps=1200,
    ),
)
