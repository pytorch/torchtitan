# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import ParallelismConfig
from torchtitan.models.deepseek_v3.config_registry import deepseek_v3_debugmodel

from . import model_registry


def deepseek_v3_2_debugmodel_full_dtensor():
    config = deepseek_v3_debugmodel()
    config.model_spec = model_registry("debugmodel")
    config.parallelism = ParallelismConfig(
        tensor_parallel_degree=2,
        spmd_backend="full_dtensor",
        enable_sequence_parallel=True,
    )
    return config
