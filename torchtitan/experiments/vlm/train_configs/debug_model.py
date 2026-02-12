# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import ConfigManager
from torchtitan.experiments.vlm.job_config import JobConfig as VLMJobConfig
from torchtitan.trainer import Trainer

MergedConfig = ConfigManager._merge_configs(Trainer.Config, VLMJobConfig)

# Start from defaults, then override
default_config = MergedConfig()

# [training]
default_config.job.description = "Llama 3 Siglip2 VLM debug training"

# [metrics]
default_config.metrics.log_freq = 1

# [model]
default_config.model.name = "vlm"
default_config.model.flavor = "debugmodel"

# [optimizer]
default_config.optimizer.lr = 8e-4

# [lr_scheduler]
default_config.lr_scheduler.warmup_steps = 2
default_config.lr_scheduler.decay_ratio = 0.8
default_config.lr_scheduler.decay_type = "linear"
default_config.lr_scheduler.min_lr_factor = 0.0

# [training]
default_config.training.local_batch_size = 32
default_config.training.seq_len = 4096
default_config.training.steps = 10
default_config.training.dataset = "cc12m"

# [data]
default_config.data.max_patches_per_image = 1024
default_config.data.max_images_per_batch = 64

# [activation_checkpoint]
default_config.activation_checkpoint.mode = "selective"
default_config.activation_checkpoint.selective_ac_option = "2"
