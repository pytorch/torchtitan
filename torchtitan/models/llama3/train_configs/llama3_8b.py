# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    ActivationCheckpointConfig,
    JobConfig,
    ModelConfig,
    TrainingConfig,
    ValidationConfig,
)
from torchtitan.tools.profiling import ProfilingConfig
from torchtitan.trainer import Trainer

default_config = Trainer.Config(
    job=JobConfig(
        description="Llama 3 8B training",
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
        flavor="8B",
        hf_assets_path="./assets/hf/Llama-3.1-8B",
    ),
    optimizer=OptimizersContainer.Config(lr=3e-4),
    training=TrainingConfig(
        local_batch_size=1,
        seq_len=8192,
        steps=1000,
        dataset="c4",
    ),
    checkpoint=CheckpointManager.Config(interval=500),
    activation_checkpoint=ActivationCheckpointConfig(
        mode="selective",
        selective_ac_option="op",
    ),
    validation=ValidationConfig(
        dataset="c4_validation",
        freq=500,
        steps=1200,
    ),
)
