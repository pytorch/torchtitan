# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: This module is kept for backward compatibility with experiment
# base_config.py files that import apply_base_settings. New tests should
# use --model/--config instead.


def apply_base_settings(config):
    """Apply the standard base integration test settings to a config instance.

    This can be called on either a plain JobConfig or a merged config type
    to set the common non-default values used by integration tests.
    """
    config.job.dump_folder = "./outputs"
    config.job.description = "model debug training for integration tests"
    config.metrics.log_freq = 1
    config.model_spec.flavor = "debugmodel"
    config.optimizer.lr = 8e-4
    config.lr_scheduler.warmup_steps = 2
    config.lr_scheduler.decay_ratio = 0.8
    config.lr_scheduler.min_lr_factor = 0.0
    config.training.steps = 10
    config.activation_checkpoint.mode = "selective"
