# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from tests.integration_tests.base_config import apply_base_settings
from torchtitan.config import ConfigManager
from torchtitan.experiments.transformers_modeling_backend.job_config import (
    JobConfig as CustomJobConfig,
)
from torchtitan.trainer import Trainer

MergedConfig = ConfigManager._merge_configs(Trainer.Config, CustomJobConfig)
default_config = MergedConfig()
apply_base_settings(default_config)
