# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import ConfigManager, JobConfig
from torchtitan.experiments.autoparallel.job_config import JobConfig as CustomJobConfig

from tests.integration_tests.base_config import apply_base_settings

MergedJobConfig = ConfigManager._merge_configs(JobConfig, CustomJobConfig)
default_config = MergedJobConfig()
apply_base_settings(default_config)
