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
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.trainer import Trainer

from . import model_registry


def qwen3_5_moe_debugmodel() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("debugmodel"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizersContainer.Config(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=2048,
            steps=10,
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
            mode="selective",
            selective_ac_option="op",
        ),
    )


def qwen3_5_moe_35b_a3b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3.5-35B-A3B",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("35b-a3b"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizersContainer.Config(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            tensor_parallel_degree=2,
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="op",
        ),
    )


def qwen3_5_moe_35b_a3b_sdpa() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3.5-35B-A3B",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("35b-a3b"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizersContainer.Config(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            tensor_parallel_degree=2,
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="op",
        ),
    )


def qwen3_5_moe_35b_a3b_varlen() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3.5-35B-A3B",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("35b-a3b-varlen"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizersContainer.Config(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            tensor_parallel_degree=2,
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="op",
        ),
    )
