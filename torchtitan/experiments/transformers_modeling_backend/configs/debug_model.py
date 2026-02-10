# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import ConfigManager, JobConfig
from torchtitan.experiments.transformers_modeling_backend.job_config import (
    JobConfig as TMBJobConfig,
)

MergedJobConfig = ConfigManager._merge_configs(JobConfig, TMBJobConfig)

# Start from defaults, then override
default_config = MergedJobConfig()

# [job]
default_config.job.description = "Qwen 3 debug training"
default_config.job.print_config = True

# [profiling]
default_config.profiling.profile_freq = 5

# [metrics]
default_config.metrics.log_freq = 1

# [model]
default_config.model.name = "transformers_modeling_backend"
default_config.model.flavor = "debugmodel"

# [hf_transformers]
default_config.hf_transformers.model = "Qwen/Qwen3-4B-Instruct-2507"

# [optimizer]
default_config.optimizer.lr = 8e-4

# [lr_scheduler]
default_config.lr_scheduler.warmup_steps = 2
default_config.lr_scheduler.decay_ratio = 0.8
default_config.lr_scheduler.decay_type = "linear"
default_config.lr_scheduler.min_lr_factor = 0.0

# [training]
default_config.training.local_batch_size = 2
default_config.training.steps = 10
default_config.training.dataset_path = "./tests/assets/c4_test"

# [parallelism]
default_config.parallelism.pipeline_parallel_schedule = "1F1B"

# [activation_checkpoint]
default_config.activation_checkpoint.mode = "selective"
