# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    ActivationCheckpointConfig,
    JobConfig,
    ModelConfig,
    TrainingConfig,
)
from torchtitan.trainer import Trainer

default_config = Trainer.Config(
    job=JobConfig(description="Qwen 3 0.6B training"),
    metrics=MetricsProcessor.Config(log_freq=1),
    model=ModelConfig(
        name="qwen3",
        flavor="0.6B",
        hf_assets_path="./assets/hf/Qwen3-0.6B",
    ),
    optimizer=OptimizersContainer.Config(lr=3e-4),
    lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
    training=TrainingConfig(
        local_batch_size=4,
        seq_len=4096,
        steps=10,
        dataset="c4",
    ),
    checkpoint=CheckpointManager.Config(
        interval=500,
        last_save_model_only=False,
        export_dtype="float16",
    ),
    activation_checkpoint=ActivationCheckpointConfig(
        mode="selective",
        selective_ac_option="op",
    ),
)
