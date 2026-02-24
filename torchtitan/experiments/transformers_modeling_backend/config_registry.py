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
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments.transformers_modeling_backend.configs import (
    TransformersBackendConfig,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.tools.profiling import ProfilingConfig

from . import model_registry


def transformers_modeling_backend_debugmodel() -> TransformersBackendConfig:
    return TransformersBackendConfig(
        hf_assets_path="./tests/assets/tokenizer",
        hf_model="Qwen/Qwen3-4B-Instruct-2507",
        debug=DebugConfig(print_config=True),
        model_spec=model_registry("debugmodel"),
        profiling=ProfilingConfig(profile_freq=5),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=2048,
            steps=10,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(pipeline_parallel_schedule="1F1B"),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="2",
        ),
    )


def transformers_modeling_backend_full() -> TransformersBackendConfig:
    return TransformersBackendConfig(
        hf_model="Qwen/Qwen3-4B-Instruct-2507",
        debug=DebugConfig(print_config=True),
        model_spec=model_registry("full"),
        profiling=ProfilingConfig(profile_freq=5),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=2048,
            steps=10,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(pipeline_parallel_schedule="1F1B"),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="2",
        ),
    )
