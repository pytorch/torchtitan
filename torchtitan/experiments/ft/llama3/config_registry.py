# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.validate import Validator
from torchtitan.config import ActivationCheckpointConfig, CommConfig, TrainingConfig
from torchtitan.experiments.ft.config.job_config import FaultTolerance
from torchtitan.experiments.ft.optimizer import FTOptimizersContainer
from torchtitan.experiments.ft.trainer import FaultTolerantTrainer
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.tools.profiling import ProfilingConfig

from . import model_registry


def llama3_ft_debugmodel() -> FaultTolerantTrainer.Config:
    return FaultTolerantTrainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        profiling=ProfilingConfig(
            enable_profiling=True,
            profile_freq=10,
            profiler_active=10,
            profiler_warmup=0,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("debugmodel"),
        optimizer=FTOptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=100,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="2",
        ),
        comm=CommConfig(train_timeout_seconds=15),
        fault_tolerance=FaultTolerance(
            enable=True,
            semi_sync_method="diloco",
            process_group="nccl",
            process_group_timeout_ms=10000,
            sync_steps=10,
            num_fragments=2,
        ),
        validator=Validator.Config(
            freq=5,
            steps=10,
        ),
    )
