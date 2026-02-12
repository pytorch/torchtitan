# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: this config is a preset for 128 H100 GPUs.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.quantization.float8 import Float8LinearConverter
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    JobConfig,
    ModelConfig,
    ParallelismConfig,
    TrainingConfig,
    ValidationConfig,
)
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.profiling import ProfilingConfig
from torchtitan.trainer import Trainer

default_config = Trainer.Config(
    job=JobConfig(
        description="Llama 3 405B training",
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
        flavor="405B",
        hf_assets_path="./assets/hf/Llama-3.1-405B",
    ),
    model_converters=ModelConvertersContainer.Config(
        converters=[
            Float8LinearConverter.Config(
                enable_fsdp_float8_all_gather=True,
                precompute_float8_dynamic_scale_for_fsdp=True,
                filter_fqns=["output"],
            ),
        ],
    ),
    optimizer=OptimizersContainer.Config(lr=8e-5),
    lr_scheduler=LRSchedulersContainer.Config(warmup_steps=600),
    training=TrainingConfig(
        local_batch_size=2,
        seq_len=8192,
        steps=3000,
        dataset="c4",
    ),
    parallelism=ParallelismConfig(
        tensor_parallel_degree=8,
        enable_async_tensor_parallel=True,
    ),
    checkpoint=CheckpointManager.Config(interval=500),
    activation_checkpoint=ActivationCheckpointConfig(mode="full"),
    compile=CompileConfig(enable=True),
    validation=ValidationConfig(
        dataset="c4_validation",
        freq=500,
        steps=1200,
    ),
)
