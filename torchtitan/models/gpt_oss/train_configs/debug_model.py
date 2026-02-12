# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.quantization.float8 import (
    Float8GroupedMMConverter,
    Float8LinearConverter,
)
from torchtitan.config import (
    ActivationCheckpointConfig,
    JobConfig,
    ModelConfig,
    ParallelismConfig,
    TrainingConfig,
    ValidationConfig,
)
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.trainer import Trainer

default_config = Trainer.Config(
    job=JobConfig(description="Gpt-oss debug training"),
    metrics=MetricsProcessor.Config(log_freq=1),
    model=ModelConfig(
        name="gpt_oss",
        flavor="debugmodel",
        hf_assets_path="./tests/assets/tokenizer",
    ),
    optimizer=OptimizersContainer.Config(lr=8e-4),
    lr_scheduler=LRSchedulersContainer.Config(
        warmup_steps=2,
        decay_ratio=0.8,
        decay_type="linear",
        min_lr_factor=0.0,
    ),
    training=TrainingConfig(
        local_batch_size=8,
        seq_len=2048,
        steps=10,
        dataset="c4_test",
    ),
    parallelism=ParallelismConfig(
        expert_parallel_degree=1,
        expert_tensor_parallel_degree=1,
    ),
    checkpoint=CheckpointManager.Config(
        interval=10,
        last_save_model_only=False,
    ),
    activation_checkpoint=ActivationCheckpointConfig(
        mode="none",
        selective_ac_option="2",
    ),
    model_converters=ModelConvertersContainer.Config(
        converters=[
            Float8LinearConverter.Config(filter_fqns=["output", "router.gate"]),
            Float8GroupedMMConverter.Config(fqns=["experts"]),
        ],
    ),
    validation=ValidationConfig(
        dataset="c4_validation",
        freq=5,
        steps=10,
    ),
)
