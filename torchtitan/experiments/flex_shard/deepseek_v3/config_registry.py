# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import ParallelismConfig
from torchtitan.models.deepseek_v3.config_registry import deepseek_v3_debugmodel
from torchtitan.trainer import Trainer

from . import model_registry


def flex_shard_deepseek_v3_debugmodel() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    config.model_spec = model_registry("debugmodel")
    return config


def flex_shard_deepseek_v3_debugmodel_dp8_ep4() -> Trainer.Config:
    config = flex_shard_deepseek_v3_debugmodel()
    config.parallelism = ParallelismConfig(
        data_parallel_shard_degree=8,
        expert_parallel_degree=4,
    )
    return config
