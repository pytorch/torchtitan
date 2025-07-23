# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

from torchtitan.config.job_config import JobConfig
from torchtitan.config.manager import ConfigManager

__all__ = ["JobConfig", "ConfigManager", "TORCH_DTYPE_MAP"]
