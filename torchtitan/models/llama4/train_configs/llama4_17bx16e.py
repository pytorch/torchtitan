# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: this config is a preset for 64 H100 GPUs.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.quantization.float8 import Float8LinearConverter
from torchtitan.components.quantization.mx import MXLinearConverter
from torchtitan.config import (
    ActivationCheckpointConfig,
    JobConfig,
    ModelConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.trainer import Trainer

default_config = Trainer.Config(
    job=JobConfig(description="Llama 4 Scout 17Bx16E training"),
    model=ModelConfig(
        name="llama4",
        flavor="17bx16e",
        hf_assets_path="./assets/hf/Llama-4-Scout-17B-16E",
    ),
    optimizer=OptimizersContainer.Config(lr=4e-3, eps=1e-15),
    lr_scheduler=LRSchedulersContainer.Config(
        warmup_steps=600,
        min_lr_factor=0.1,
    ),
    training=TrainingConfig(
        local_batch_size=8,
        seq_len=8192,
        steps=3000,
        dataset="c4",
    ),
    parallelism=ParallelismConfig(
        tensor_parallel_degree=8,
        expert_parallel_degree=1,
        expert_tensor_parallel_degree=8,
    ),
    checkpoint=CheckpointManager.Config(interval=500),
    activation_checkpoint=ActivationCheckpointConfig(mode="full"),
    model_converters=ModelConvertersContainer.Config(
        converters=[
            Float8LinearConverter.Config(filter_fqns=["output", "router.gate"]),
            MXLinearConverter.Config(filter_fqns=["output", "router.gate"]),
        ],
    ),
)
