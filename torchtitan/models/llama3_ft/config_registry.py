# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.ft.config.job_config import JobConfig as FTJobConfig
from torchtitan.config import ConfigManager
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.trainer import Trainer

from . import model_registry

MergedConfig = ConfigManager._merge_configs(Trainer.Config, FTJobConfig)


def llama3_ft_debugmodel():
    config = MergedConfig()

    config.job.description = "Llama 3 fault-tolerant debug training"
    config.job.hf_assets_path = "./tests/assets/tokenizer"

    config.profiling.enable_profiling = True
    config.profiling.profile_freq = 10
    config.profiling.profiler_active = 10
    config.profiling.profiler_warmup = 0

    config.metrics.log_freq = 1

    config.model_spec = model_registry("debugmodel")

    config.optimizer.lr = 8e-4

    config.lr_scheduler.warmup_steps = 2
    config.lr_scheduler.decay_ratio = 0.8
    config.lr_scheduler.decay_type = "linear"
    config.lr_scheduler.min_lr_factor = 0.0

    config.training.local_batch_size = 8
    config.training.seq_len = 2048
    config.training.steps = 100
    config.dataloader = HuggingFaceTextDataLoader.Config()

    config.checkpoint.interval = 10
    config.checkpoint.last_save_model_only = False

    config.activation_checkpoint.mode = "selective"
    config.activation_checkpoint.selective_ac_option = "2"

    config.comm.train_timeout_seconds = 15

    config.fault_tolerance.enable = True
    config.fault_tolerance.semi_sync_method = "diloco"
    config.fault_tolerance.process_group = "nccl"
    config.fault_tolerance.process_group_timeout_ms = 10000
    config.fault_tolerance.sync_steps = 10
    config.fault_tolerance.num_fragments = 2

    config.validator.dataloader.dataset = "c4_validation"
    config.validator.freq = 5
    config.validator.steps = 10

    return config
