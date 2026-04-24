# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import ActivationCheckpointConfig, ParallelismConfig, TrainingConfig
from torchtitan.models.flux2.configs import Encoder, Inference, Validation
from torchtitan.models.flux2.flux2_datasets import Flux2DataLoader
from torchtitan.models.flux2.trainer import Flux2Trainer

from . import model_registry


_DATA_CONSOLIDATION_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "DataConsolidation"
    / "config"
    / "joint_train_dataset.yaml"
)

_DATA_CONSOLIDATION_PREFETCH_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "DataConsolidation"
    / "config"
    / "joint_train_dataset_prefetch.yaml"
)


def flux2_debugmodel() -> Flux2Trainer.Config:
    encoder = Encoder(model_name="flux.2-dev", text_encoder_device="cuda")
    return Flux2Trainer.Config(
        metrics=MetricsProcessor.Config(
            log_freq=1,    
            enable_tensorboard=True,
            save_tb_folder="tb",
        ),
        model_spec=model_registry("flux2-dev"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=1,
            decay_ratio=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            max_norm=2.0,
            steps=10,
        ),
        dataloader=Flux2DataLoader.Config(
            dataset="cc12m-test",
            classifier_free_guidance_prob=0.0,
            img_size=256,
        ),
        encoder=encoder,
        parallelism=ParallelismConfig(
            context_parallel_degree=1,
            data_parallel_shard_degree=8,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        validation=Validation(
            enable_classifier_free_guidance=False,
            denoising_steps=4,
        ),
        inference=Inference(
            save_img_folder="inference_results",
            local_batch_size=1,
        ),
    )


def flux2_dev() -> Flux2Trainer.Config:
    encoder = Encoder(model_name="flux.2-klein-4b", text_encoder_device="cuda")
    return Flux2Trainer.Config(
        metrics=MetricsProcessor.Config(
            log_freq=100,    
            enable_tensorboard=True,
            save_tb_folder="tb",
        ),
        model_spec=model_registry("flux2-klein-4b"),
        optimizer=OptimizersContainer.Config(lr=1e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=3000,
            decay_ratio=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=128,
            steps=1000,
        ),
        # parallelism=ParallelismConfig(
        #     context_parallel_degree=2,
        #     # data_parallel_shard_degree=4,
        # ),
        dataloader=Flux2DataLoader.Config(
            dataset="cc12m-test",
            classifier_free_guidance_prob=0.0,
            img_size=256,
        ),
        encoder=encoder,
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        checkpoint=CheckpointManager.Config(interval=1000),
    )


def flux2_dev_dataconsolidation() -> Flux2Trainer.Config:
    encoder = Encoder(model_name="flux.2-klein-4b", text_encoder_device="cuda")
    return Flux2Trainer.Config(
        metrics=MetricsProcessor.Config(
            log_freq=100,
            enable_tensorboard=True,
            save_tb_folder="tb",
        ),
        model_spec=model_registry("flux2-klein-4b"),
        optimizer=OptimizersContainer.Config(lr=1e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=3000,
            decay_ratio=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=128,
            steps=1000,
        ),
        dataloader=Flux2DataLoader.Config(
            dataset="data-consolidation",
            dataset_path=str(_DATA_CONSOLIDATION_CONFIG),
            classifier_free_guidance_prob=0.0,
            img_size=256,
            shuffle=True,
        ),
        encoder=encoder,
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        checkpoint=CheckpointManager.Config(interval=1000),
    )


def flux2_dev_dataconsolidation_prefetch() -> Flux2Trainer.Config:
    encoder = Encoder(model_name="flux.2-klein-4b", text_encoder_device="cuda")
    return Flux2Trainer.Config(
        metrics=MetricsProcessor.Config(
            log_freq=100,
            enable_tensorboard=True,
            save_tb_folder="tb",
        ),
        model_spec=model_registry("flux2-klein-4b"),
        optimizer=OptimizersContainer.Config(lr=1e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=3000,
            decay_ratio=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=128,
            steps=1000,
        ),
        dataloader=Flux2DataLoader.Config(
            dataset="data-consolidation",
            dataset_path=str(_DATA_CONSOLIDATION_PREFETCH_CONFIG),
            classifier_free_guidance_prob=0.0,
            img_size=256,
            shuffle=True,
        ),
        encoder=encoder,
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        checkpoint=CheckpointManager.Config(interval=1000),
    )
