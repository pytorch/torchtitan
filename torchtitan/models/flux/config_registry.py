# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import ConfigManager
from torchtitan.models.flux.flux_datasets import FluxDataLoader
from torchtitan.models.flux.job_config import JobConfig as FluxJobConfig
from torchtitan.models.flux.validate import FluxValidator
from torchtitan.trainer import Trainer

from . import model_registry

MergedConfig = ConfigManager._merge_configs(Trainer.Config, FluxJobConfig)


def flux_debugmodel():
    config = MergedConfig()

    config.job.description = "Flux debug model"
    config.job.hf_assets_path = "tests/assets/tokenizer"

    config.metrics.log_freq = 1

    config.model_spec = model_registry("flux-debug")

    config.optimizer.lr = 8e-4

    config.lr_scheduler.warmup_steps = 1
    config.lr_scheduler.decay_ratio = 0.0

    config.training.local_batch_size = 4
    config.training.max_norm = 2.0
    config.training.steps = 10
    config.dataloader = FluxDataLoader.Config()
    config.training.classifier_free_guidance_prob = 0.447
    config.training.img_size = 256

    config.encoder.t5_encoder = "google/t5-v1_1-xxl"
    config.encoder.clip_encoder = "openai/clip-vit-large-patch14"
    config.encoder.max_t5_encoding_len = 256
    config.encoder.autoencoder_path = "assets/hf/FLUX.1-dev/ae.safetensors"

    config.parallelism.context_parallel_degree = 1

    config.activation_checkpoint.mode = "full"

    config.checkpoint.interval = 10
    config.checkpoint.last_save_model_only = False

    config.validation.enable_classifier_free_guidance = True
    config.validation.classifier_free_guidance_scale = 5.0
    config.validation.denoising_steps = 4

    config.validator = FluxValidator.Config(
        freq=5,
        steps=48,
        dataloader=FluxDataLoader.Config(
            dataset="coco-validation",
            generate_timesteps=True,
        ),
        save_img_count=1,
        save_img_folder="img",
        all_timesteps=False,
    )

    config.inference.save_img_folder = "inference_results"
    config.inference.prompts_path = "./torchtitan/models/flux/inference/prompts.txt"
    config.inference.local_batch_size = 2

    return config


def flux_dev():
    config = MergedConfig()

    config.job.description = "Flux-dev model"

    config.metrics.log_freq = 100

    config.model_spec = model_registry("flux-dev")

    config.optimizer.lr = 1e-4

    config.lr_scheduler.warmup_steps = 3000
    config.lr_scheduler.decay_ratio = 0.0

    config.training.local_batch_size = 32
    config.training.steps = 30000
    config.dataloader = FluxDataLoader.Config(dataset="cc12m-wds")
    config.training.classifier_free_guidance_prob = 0.447
    config.training.img_size = 256

    config.encoder.t5_encoder = "google/t5-v1_1-xxl"
    config.encoder.clip_encoder = "openai/clip-vit-large-patch14"
    config.encoder.max_t5_encoding_len = 512
    config.encoder.autoencoder_path = "assets/hf/FLUX.1-dev/ae.safetensors"

    config.activation_checkpoint.mode = "full"

    config.checkpoint.interval = 1000

    config.validation.enable_classifier_free_guidance = True
    config.validation.classifier_free_guidance_scale = 5.0
    config.validation.denoising_steps = 50

    config.validator = FluxValidator.Config(
        freq=1000,
        steps=12,
        dataloader=FluxDataLoader.Config(
            dataset="coco-validation",
            generate_timesteps=True,
        ),
        save_img_count=50,
        save_img_folder="img",
        all_timesteps=False,
    )

    return config


def flux_schnell():
    config = MergedConfig()

    config.job.description = "Flux-schnell model"

    config.metrics.log_freq = 100

    config.model_spec = model_registry("flux-schnell")

    config.optimizer.lr = 1e-4

    config.lr_scheduler.warmup_steps = 3000
    config.lr_scheduler.decay_ratio = 0.0

    config.training.local_batch_size = 64
    config.training.steps = 30000
    config.dataloader = FluxDataLoader.Config(dataset="cc12m-wds")
    config.training.classifier_free_guidance_prob = 0.447
    config.training.img_size = 256

    config.encoder.t5_encoder = "google/t5-v1_1-xxl"
    config.encoder.clip_encoder = "openai/clip-vit-large-patch14"
    config.encoder.max_t5_encoding_len = 256
    config.encoder.autoencoder_path = "assets/hf/FLUX.1-dev/ae.safetensors"

    config.activation_checkpoint.mode = "full"

    config.checkpoint.interval = 1000

    config.validation.enable_classifier_free_guidance = True
    config.validation.classifier_free_guidance_scale = 5.0
    config.validation.denoising_steps = 50

    config.validator = FluxValidator.Config(
        freq=1000,
        steps=6,
        dataloader=FluxDataLoader.Config(
            dataset="coco-validation",
            generate_timesteps=True,
        ),
        save_img_count=50,
        save_img_folder="img",
        all_timesteps=False,
    )

    return config
