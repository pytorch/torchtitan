# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
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
from torchtitan.hf_datasets.text_datasets import ChatDataLoader, HuggingFaceTextDataLoader
from torchtitan.tools.profiler import Profiler

from . import model_registry


def transformers_modeling_backend_debugmodel() -> TransformersBackendConfig:
    return TransformersBackendConfig(
        loss=CrossEntropyLoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        hf_model="Qwen/Qwen3-4B-Instruct-2507",
        debug=DebugConfig(print_config=True),
        model_spec=model_registry("debugmodel"),
        profiler=Profiler.Config(profile_freq=5),
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
        ),
    )


def transformers_modeling_backend_full() -> TransformersBackendConfig:
    return TransformersBackendConfig(
        loss=CrossEntropyLoss.Config(),
        hf_model="Qwen/Qwen3-4B-Instruct-2507",
        debug=DebugConfig(print_config=True),
        model_spec=model_registry("full"),
        profiler=Profiler.Config(profile_freq=5),
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
        ),
    )


def transformers_modeling_backend_sft_full() -> TransformersBackendConfig:
    """SFT config with real HF pretrained weights loaded via initial_load_in_hf."""

    def process_sample(sample):
        return [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
        ]

    return TransformersBackendConfig(
        loss=CrossEntropyLoss.Config(),
        hf_assets_path="./tests/assets/qwen3_0.6b",
        hf_model="Qwen/Qwen3-0.6B",
        model_spec=model_registry("sft_full"),
        optimizer=OptimizersContainer.Config(lr=2e-5),
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
        dataloader=ChatDataLoader.Config(
            dataset_path="json",
            load_dataset_kwargs={
                "data_files": "tests/assets/sft_test/data.json",
                "split": "train",
            },
            sample_processor=process_sample,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        checkpoint=CheckpointManager.Config(
            enable=True,
            initial_load_in_hf=True,
            initial_load_model_only=True,
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )


def transformers_modeling_backend_sft_debugmodel() -> TransformersBackendConfig:
    """SFT debug config for the transformers backend using ChatDataLoader."""

    def process_sample(sample):
        return [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
        ]

    return TransformersBackendConfig(
        loss=CrossEntropyLoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        hf_model="Qwen/Qwen3-4B-Instruct-2507",
        model_spec=model_registry("sft_debugmodel"),
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
        ),
        dataloader=ChatDataLoader.Config(
            dataset_path="json",
            load_dataset_kwargs={
                "data_files": "tests/assets/sft_test/data.json",
                "split": "train",
            },
            sample_processor=process_sample,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )
