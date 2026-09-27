# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared configuration dataclasses for torchtitan.

Some configs live near their owner instead of here:
  - Profiler.Config                 (in observability/profiler.py)
  - OptimizersContainer.Config      (in components/optimizer/optimizer.py)
  - LRSchedulersContainer.Config    (in components/optimizer/lr_scheduler.py)
  - MetricsProcessor.Config         (in observability/metrics.py)
  - CheckpointManager.Config        (in components/checkpointer/dcp.py)

Configs without a clear single owner (or with circular-import constraints)
live here.

Most knobs belong to a component or to the model, not here. But some options
have no suitable home, e.g. the training token-budget settings, and those can
be placed here. Discuss with the maintainers first if you intend to add one.

The command-line surface is frozen either way, so annotate a new field with
``tyro.conf.Suppress``, as ``Trainer.Config.model`` does. See
``torchtitan/config/README.md``.
"""

import torch

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

from .comm import CommConfig
from .compile import CompileConfig
from .configurable import Configurable
from .debug import DebugConfig
from .function import Function
from .manager import ConfigManager
from .override import (
    apply_overrides,
    clear_overrides,
    derive,
    Override,
    override,
    OverrideConfig,
)
from .training import TrainingConfig

__all__ = [
    "ConfigManager",
    "Configurable",
    "Function",
    "TORCH_DTYPE_MAP",
    # Config dataclasses
    "CompileConfig",
    "CommConfig",
    "TrainingConfig",
    "DebugConfig",
    # Override mechanism
    "OverrideConfig",
    "Override",
    "override",
    "derive",
    "apply_overrides",
    "clear_overrides",
]
