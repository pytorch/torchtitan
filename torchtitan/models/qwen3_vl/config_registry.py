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
from torchtitan.trainer import Trainer

from . import model_registry
from .datasets.mm_datasets import MMDataLoader


def qwen3_vl_debugmodel() -> Trainer.Config:
    spec = model_registry("debugmodel")
    encoder = spec.model.encoder
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-VL-2B-Instruct",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=spec,
        dataloader=MMDataLoader.Config(
            dataset="cc12m-test",
            patch_size=encoder.patch_size,
            temporal_patch_size=encoder.temporal_patch_size,
            spatial_merge_size=encoder.spatial_merge_size,
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
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="2",
        ),
    )


def qwen3_vl_debugmodel_video() -> Trainer.Config:
    spec = model_registry("debugmodel")
    encoder = spec.model.encoder
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-VL-2B-Instruct",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=spec,
        dataloader=MMDataLoader.Config(
            dataset="nemotron-video",
            dataset_subset="nextqa",
            video_dir="./assets/videos",
            patch_size=encoder.patch_size,
            temporal_patch_size=encoder.temporal_patch_size,
            spatial_merge_size=encoder.spatial_merge_size,
            video_fps=0.5,
            video_max_frames=8,
            max_pixels=200000,
            min_pixels=784,
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
            seq_len=4096,
            steps=10,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="2",
        ),
    )


def qwen3_vl_debugmodel_moe() -> Trainer.Config:
    spec = model_registry("debugmodel_moe")
    encoder = spec.model.encoder
    return Trainer.Config(
        hf_assets_path="../hf_models/Qwen/Qwen3-VL-2B-Instruct",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=spec,
        dataloader=MMDataLoader.Config(
            dataset="cc12m-test",
            patch_size=encoder.patch_size,
            temporal_patch_size=encoder.temporal_patch_size,
            spatial_merge_size=encoder.spatial_merge_size,
        ),
        optimizer=OptimizersContainer.Config(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
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


def qwen3_vl_2b() -> Trainer.Config:
    spec = model_registry("2B")
    encoder = spec.model.encoder
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-VL-2B-Instruct",
        model_spec=spec,
        dataloader=MMDataLoader.Config(
            dataset="cc12m",
            patch_size=encoder.patch_size,
            temporal_patch_size=encoder.temporal_patch_size,
            spatial_merge_size=encoder.spatial_merge_size,
        ),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=4096,
            steps=100,
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
    spec = model_registry("8B")
    encoder = spec.model.encoder
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-VL-8B-Instruct",
        model_spec=spec,
        dataloader=MMDataLoader.Config(
            dataset="cc12m",
            patch_size=encoder.patch_size,
            temporal_patch_size=encoder.temporal_patch_size,
            spatial_merge_size=encoder.spatial_merge_size,
        ),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=100,
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
    spec = model_registry("30B-A3B")
    encoder = spec.model.encoder
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-VL-30B-A3B-Instruct",
        model_spec=spec,
        dataloader=MMDataLoader.Config(
            dataset="cc12m",
            patch_size=encoder.patch_size,
            temporal_patch_size=encoder.temporal_patch_size,
            spatial_merge_size=encoder.spatial_merge_size,
        ),
        optimizer=OptimizersContainer.Config(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=600),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=100,
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
