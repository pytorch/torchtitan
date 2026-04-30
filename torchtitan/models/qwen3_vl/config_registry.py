# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedCELoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.tokenizer import MultiModalTokenizer

from torchtitan.config import (
    ActivationCheckpointConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.hf_datasets.multimodal.mm_datasets import MMDataLoader
from torchtitan.trainer import Trainer

from . import model_registry, QWEN3_VL_SPECIAL_TOKENS


def _qwen3_vl_dataloader(dataset: str, **kwargs) -> MMDataLoader.Config:
    return MMDataLoader.Config(
        dataset=dataset,
        max_images_per_batch=128,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        min_pixels=65536,
        max_pixels=16777216,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        **kwargs,
    )


def qwen3_vl_debugmodel() -> Trainer.Config:
    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_VL_SPECIAL_TOKENS),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("debugmodel"),
        dataloader=_qwen3_vl_dataloader("cc12m-test"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=512,
            steps=10,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )


def qwen3_vl_debugmodel_moe() -> Trainer.Config:
    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_VL_SPECIAL_TOKENS),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("debugmodel_moe"),
        dataloader=_qwen3_vl_dataloader("cc12m-test"),
        optimizer=OptimizersContainer.Config(lr=3e-3),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=512,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=4,
            expert_parallel_degree=4,
            tensor_parallel_degree=2,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )


def qwen3_vl_2b() -> Trainer.Config:
    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./assets/hf/Qwen3-VL-2B-Instruct",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_VL_SPECIAL_TOKENS),
        model_spec=model_registry("2B"),
        dataloader=_qwen3_vl_dataloader("cc12m"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=4096,
            steps=1000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            enable=False,
            interval=50,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="full",
        ),
    )


def qwen3_vl_8b() -> Trainer.Config:
    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./assets/hf/Qwen3-VL-8B-Instruct",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_VL_SPECIAL_TOKENS),
        model_spec=model_registry("8B"),
        dataloader=_qwen3_vl_dataloader("cc12m"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=1000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            enable=False,
            interval=50,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="full",
        ),
    )


def qwen3_vl_30b_a3b() -> Trainer.Config:
    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./assets/hf/Qwen3-VL-30B-A3B-Instruct",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_VL_SPECIAL_TOKENS),
        model_spec=model_registry("30B-A3B"),
        dataloader=_qwen3_vl_dataloader("cc12m"),
        optimizer=OptimizersContainer.Config(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=1000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            expert_parallel_degree=8,
        ),
        checkpoint=CheckpointManager.Config(
            enable=False,
            interval=500,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="full",
        ),
    )
