# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: this config is a preset for 8 H100 GPUs (with 96GiB memory).

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    ActivationCheckpointConfig,
    JobConfig,
    ModelConfig,
    TrainingConfig,
)
from torchtitan.trainer import Trainer

default_config = Trainer.Config(
    job=JobConfig(description="Qwen3 32B training"),
    model=ModelConfig(
        name="qwen3",
        flavor="32B",
        hf_assets_path="./assets/hf/Qwen3-32B",
    ),
    optimizer=OptimizersContainer.Config(lr=8e-4),
    lr_scheduler=LRSchedulersContainer.Config(warmup_steps=600),
    training=TrainingConfig(
        local_batch_size=2,
        seq_len=4096,
        steps=3000,
        dataset="c4",
    ),
    checkpoint=CheckpointManager.Config(
        interval=500,
        last_save_model_only=False,
        export_dtype="float16",
    ),
    activation_checkpoint=ActivationCheckpointConfig(
        mode="full",
        selective_ac_option="op",
    ),
)
