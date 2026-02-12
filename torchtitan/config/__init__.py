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

from .configs import (
    ActivationCheckpointConfig,
    CommConfig,
    CompileConfig,
    DataLoaderConfig,
    DebugConfig,
    FaultToleranceConfig,
    JobConfig,
    ModelConfig,
    ParallelismConfig,
    TrainingConfig,
    ValidationConfig,
)

from .configurable import Configurable, Module
from .manager import ConfigManager

__all__ = [
    "ConfigManager",
    "Configurable",
    "Module",
    "TORCH_DTYPE_MAP",
    # Config dataclasses
    "JobConfig",
    "ModelConfig",
    "ActivationCheckpointConfig",
    "CompileConfig",
    "FaultToleranceConfig",
    "ParallelismConfig",
    "CommConfig",
    "TrainingConfig",
    "DataLoaderConfig",
    "ValidationConfig",
    "DebugConfig",
]
