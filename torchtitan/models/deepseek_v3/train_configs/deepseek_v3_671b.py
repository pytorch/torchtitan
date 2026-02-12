# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.quantization.float8 import (
    Float8GroupedMMConverter,
    Float8LinearConverter,
)
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    JobConfig,
    ModelConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.trainer import Trainer

default_config = Trainer.Config(
    job=JobConfig(description="DeepSeek-V3 671B model training"),
    model=ModelConfig(
        name="deepseek_v3",
        flavor="671B",
        hf_assets_path="./assets/hf/DeepSeek-V3.1-Base",
    ),
    optimizer=OptimizersContainer.Config(lr=2.2e-4),
    lr_scheduler=LRSchedulersContainer.Config(
        warmup_steps=2000,
        decay_ratio=0.8,
        decay_type="cosine",
        min_lr_factor=0.1,
    ),
    training=TrainingConfig(
        local_batch_size=4,
        seq_len=4096,
        steps=10000,
        dataset="c4",
    ),
    parallelism=ParallelismConfig(
        pipeline_parallel_schedule="Interleaved1F1B",
        expert_parallel_degree=1,
        expert_tensor_parallel_degree=1,
    ),
    checkpoint=CheckpointManager.Config(interval=500),
    activation_checkpoint=ActivationCheckpointConfig(
        mode="selective",
        selective_ac_option="op",
    ),
    compile=CompileConfig(enable=True, components=["loss"]),
    model_converters=ModelConvertersContainer.Config(
        converters=[
            Float8LinearConverter.Config(filter_fqns=["output", "router.gate"]),
            Float8GroupedMMConverter.Config(fqns=["experts"]),
        ],
    ),
)
