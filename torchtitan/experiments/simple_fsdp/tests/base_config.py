# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import ConfigManager
from torchtitan.experiments.simple_fsdp.job_config import JobConfig as CustomJobConfig
from torchtitan.trainer import Trainer

from tests.integration_tests.base_config import apply_base_settings

MergedConfig = ConfigManager._merge_configs(Trainer.Config, CustomJobConfig)
default_config = MergedConfig()
apply_base_settings(default_config)
