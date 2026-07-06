# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.transformers_modeling_backend.configs import (
    TransformersBackendConfig,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.tools.profiler import Profiler
from . import model_registry


def transformers_modeling_backend_debugmodel() -> TransformersBackendConfig:
    model_spec = model_registry("debugmodel")
    return TransformersBackendConfig(
        loss=CrossEntropyLoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        hf_model="Qwen/Qwen3-4B-Instruct-2507",
        debug=DebugConfig(print_config=True),
        model_spec=model_spec,
        profiler=Profiler.Config(profile_freq=5),
        optimizer=default_adamw(lr=8e-4),
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
        activation_checkpoint=SelectiveAC.Config(),
    )


def transformers_modeling_backend_debugmodel_flex() -> TransformersBackendConfig:
    """Dense debug config with flex attention enabled (use_flex_attn=True).

    Exercises the packed/document-mask flex path (get_attention_masks -> trainer
    capability gate -> forward -> local_map under TP). Same as debugmodel but
    with the debugmodel_flex flavor.
    """
    return TransformersBackendConfig(
        loss=CrossEntropyLoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        hf_model="Qwen/Qwen3-4B-Instruct-2507",
        debug=DebugConfig(print_config=True),
        model_spec=model_registry("debugmodel_flex"),
        profiler=Profiler.Config(profile_freq=5),
        optimizer=default_adamw(lr=8e-4),
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
        activation_checkpoint=SelectiveAC.Config(),
    )


def transformers_modeling_backend_debugmodel_moe() -> TransformersBackendConfig:
    return TransformersBackendConfig(
        loss=CrossEntropyLoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        hf_model="Qwen/Qwen3-30B-A3B",
        debug=DebugConfig(print_config=True),
        model_spec=model_registry("debugmodel_moe"),
        profiler=Profiler.Config(profile_freq=5),
        optimizer=default_adamw(lr=8e-4),
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
        activation_checkpoint=SelectiveAC.Config(),
    )


def transformers_modeling_backend_debugmodel_moe_flex() -> TransformersBackendConfig:
    """MoE debug config with flex attention enabled (use_flex_attn=True).

    Exercises the MoE backend (EP/TP) together with the packed/document-mask
    flex path, including flex + context parallelism.
    """
    return TransformersBackendConfig(
        loss=CrossEntropyLoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        hf_model="Qwen/Qwen3-30B-A3B",
        debug=DebugConfig(print_config=True),
        model_spec=model_registry("debugmodel_moe_flex"),
        profiler=Profiler.Config(profile_freq=5),
        optimizer=default_adamw(lr=8e-4),
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
        activation_checkpoint=SelectiveAC.Config(),
    )


def transformers_modeling_backend_full_moe() -> TransformersBackendConfig:
    return TransformersBackendConfig(
        hf_model="Qwen/Qwen3-30B-A3B",
        debug=DebugConfig(print_config=True),
        model_spec=model_registry("full_moe"),
        profiler=Profiler.Config(profile_freq=5),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=200,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=2048,
            steps=1000,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        metrics=MetricsProcessor.Config(log_freq=10),
        parallelism=ParallelismConfig(pipeline_parallel_schedule="1F1B"),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def transformers_modeling_backend_full() -> TransformersBackendConfig:
    model_spec = model_registry("full")
    return TransformersBackendConfig(
        loss=CrossEntropyLoss.Config(),
        hf_model="Qwen/Qwen3-4B-Instruct-2507",
        debug=DebugConfig(print_config=True),
        model_spec=model_spec,
        profiler=Profiler.Config(profile_freq=5),
        optimizer=default_adamw(lr=8e-4),
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
        activation_checkpoint=SelectiveAC.Config(),
    )
