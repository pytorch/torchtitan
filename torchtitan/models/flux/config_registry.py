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
from torchtitan.models.flux.configs import Encoder, Inference, Validation
from torchtitan.models.flux.flux_datasets import FluxDataLoader
from torchtitan.models.flux.trainer import FluxTrainer
from torchtitan.models.flux.validate import FluxValidator

from . import model_registry


def flux_debugmodel() -> FluxTrainer.Config:
    encoder = Encoder(
        t5_encoder="google/t5-v1_1-xxl",
        clip_encoder="openai/clip-vit-large-patch14",
        max_t5_encoding_len=256,
        autoencoder_path="assets/hf/FLUX.1-dev/ae.safetensors",
    )
    hf_assets_path = "tests/assets/tokenizer"
    return FluxTrainer.Config(
        hf_assets_path=hf_assets_path,
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("flux-debug"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=1,
            decay_ratio=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            max_norm=2.0,
            steps=10,
        ),
        dataloader=FluxDataLoader.Config(
            classifier_free_guidance_prob=0.447,
            img_size=256,
            encoder=encoder,
            hf_assets_path=hf_assets_path,
        ),
        encoder=encoder,
        parallelism=ParallelismConfig(context_parallel_degree=1),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        validation=Validation(
            enable_classifier_free_guidance=True,
            classifier_free_guidance_scale=5.0,
            denoising_steps=4,
        ),
        validator=FluxValidator.Config(
            freq=5,
            steps=48,
            dataloader=FluxDataLoader.Config(
                dataset="coco-validation",
                classifier_free_guidance_prob=0.447,
                img_size=256,
                generate_timesteps=True,
            ),
            save_img_count=1,
            save_img_folder="img",
            all_timesteps=False,
        ),
        inference=Inference(
            save_img_folder="inference_results",
            prompts_path="./torchtitan/models/flux/inference/prompts.txt",
            local_batch_size=2,
        ),
    )


def flux_dev() -> FluxTrainer.Config:
    encoder = Encoder(
        t5_encoder="google/t5-v1_1-xxl",
        clip_encoder="openai/clip-vit-large-patch14",
        max_t5_encoding_len=512,
        autoencoder_path="assets/hf/FLUX.1-dev/ae.safetensors",
    )
    return FluxTrainer.Config(
        metrics=MetricsProcessor.Config(log_freq=100),
        model_spec=model_registry("flux-dev"),
        optimizer=OptimizersContainer.Config(lr=1e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=3000,
            decay_ratio=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=32,
            steps=30000,
        ),
        dataloader=FluxDataLoader.Config(
            dataset="cc12m-wds",
            classifier_free_guidance_prob=0.447,
            img_size=256,
            encoder=encoder,
        ),
        encoder=encoder,
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        checkpoint=CheckpointManager.Config(interval=1000),
        validation=Validation(
            enable_classifier_free_guidance=True,
            classifier_free_guidance_scale=5.0,
            denoising_steps=50,
        ),
        validator=FluxValidator.Config(
            freq=1000,
            steps=12,
            dataloader=FluxDataLoader.Config(
                dataset="coco-validation",
                classifier_free_guidance_prob=0,
                img_size=256,
                generate_timesteps=True,
            ),
            save_img_count=50,
            save_img_folder="img",
            all_timesteps=False,
        ),
    )


def flux_schnell() -> FluxTrainer.Config:
    encoder = Encoder(
        t5_encoder="google/t5-v1_1-xxl",
        clip_encoder="openai/clip-vit-large-patch14",
        max_t5_encoding_len=256,
        autoencoder_path="assets/hf/FLUX.1-dev/ae.safetensors",
    )
    return FluxTrainer.Config(
        metrics=MetricsProcessor.Config(log_freq=100),
        model_spec=model_registry("flux-schnell"),
        optimizer=OptimizersContainer.Config(lr=1e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=3000,
            decay_ratio=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=64,
            steps=30000,
        ),
        dataloader=FluxDataLoader.Config(
            dataset="cc12m-wds",
            classifier_free_guidance_prob=0.447,
            img_size=256,
            encoder=encoder,
        ),
        encoder=encoder,
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        checkpoint=CheckpointManager.Config(interval=1000),
        validation=Validation(
            enable_classifier_free_guidance=True,
            classifier_free_guidance_scale=5.0,
            denoising_steps=50,
        ),
        validator=FluxValidator.Config(
            freq=1000,
            steps=6,
            dataloader=FluxDataLoader.Config(
                dataset="coco-validation",
                classifier_free_guidance_prob=0,
                img_size=256,
                generate_timesteps=True,
            ),
            save_img_count=50,
            save_img_folder="img",
            all_timesteps=False,
        ),
    )
