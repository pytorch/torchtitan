# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import ConfigManager
from torchtitan.models.flux.job_config import JobConfig as FluxJobConfig
from torchtitan.trainer import Trainer

MergedConfig = ConfigManager._merge_configs(Trainer.Config, FluxJobConfig)

# Start from defaults, then override
default_config = MergedConfig()

# [training]
default_config.job.description = "Flux debug model"

# [metrics]
default_config.metrics.log_freq = 1

# [model]
default_config.model.name = "flux"
default_config.model.flavor = "flux-debug"
default_config.model.hf_assets_path = "tests/assets/tokenizer"

# [optimizer]
default_config.optimizer.lr = 8e-4

# [lr_scheduler]
default_config.lr_scheduler.warmup_steps = 1
default_config.lr_scheduler.decay_ratio = 0.0

# [training]
default_config.training.local_batch_size = 4
default_config.training.max_norm = 2.0
default_config.training.steps = 10
default_config.training.dataset = "cc12m-test"
default_config.training.classifier_free_guidance_prob = 0.447
default_config.training.img_size = 256

# [encoder]
default_config.encoder.t5_encoder = "google/t5-v1_1-xxl"
default_config.encoder.clip_encoder = "openai/clip-vit-large-patch14"
default_config.encoder.max_t5_encoding_len = 256
default_config.encoder.autoencoder_path = "assets/hf/FLUX.1-dev/ae.safetensors"

# [parallelism]
default_config.parallelism.context_parallel_degree = 1

# [activation_checkpoint]
default_config.activation_checkpoint.mode = "full"

# [checkpoint]
default_config.checkpoint.interval = 10
default_config.checkpoint.last_save_model_only = False

# [validation]
default_config.validation.dataset = "coco-validation"
default_config.validation.freq = 5
default_config.validation.local_batch_size = 8
default_config.validation.steps = 48
default_config.validation.enable_classifier_free_guidance = True
default_config.validation.classifier_free_guidance_scale = 5.0
default_config.validation.denoising_steps = 4
default_config.validation.save_img_count = 1
default_config.validation.save_img_folder = "img"
default_config.validation.all_timesteps = False

# [inference]
default_config.inference.save_img_folder = "inference_results"
default_config.inference.prompts_path = "./torchtitan/models/flux/inference/prompts.txt"
default_config.inference.local_batch_size = 2
