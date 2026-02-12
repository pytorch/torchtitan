# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.ft.config.job_config import JobConfig as FTJobConfig
from torchtitan.config import ConfigManager
from torchtitan.trainer import Trainer

MergedConfig = ConfigManager._merge_configs(Trainer.Config, FTJobConfig)

# Start from defaults, then override
default_config = MergedConfig()

# [training]
default_config.job.description = "Llama 3 fault-tolerant debug training"

# [profiling]
default_config.profiling.enable_profiling = True
default_config.profiling.profile_freq = 10
default_config.profiling.profiler_active = 10
default_config.profiling.profiler_warmup = 0

# [metrics]
default_config.metrics.log_freq = 1

# [model]
default_config.model.name = "llama3"
default_config.model.flavor = "debugmodel"
default_config.model.hf_assets_path = "./tests/assets/tokenizer"

# [optimizer]
default_config.optimizer.lr = 8e-4

# [lr_scheduler]
default_config.lr_scheduler.warmup_steps = 2
default_config.lr_scheduler.decay_ratio = 0.8
default_config.lr_scheduler.decay_type = "linear"
default_config.lr_scheduler.min_lr_factor = 0.0

# [training]
default_config.training.local_batch_size = 8
default_config.training.seq_len = 2048
default_config.training.steps = 100
default_config.training.dataset = "c4_test"

# [checkpoint]
default_config.checkpoint.interval = 10
default_config.checkpoint.last_save_model_only = False

# [activation_checkpoint]
default_config.activation_checkpoint.mode = "selective"
default_config.activation_checkpoint.selective_ac_option = "2"

# [comm]
default_config.comm.train_timeout_seconds = 15

# [fault_tolerance]
default_config.fault_tolerance.enable = True
default_config.fault_tolerance.semi_sync_method = "diloco"
default_config.fault_tolerance.process_group = "nccl"
default_config.fault_tolerance.process_group_timeout_ms = 10000
default_config.fault_tolerance.sync_steps = 10
default_config.fault_tolerance.num_fragments = 2

# [validation]
default_config.validation.dataset = "c4_validation"
default_config.validation.freq = 5
default_config.validation.steps = 10
