# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

@dataclass
class Optimizer:

    global_lr: float = 0.7
    """Learning rate to use"""

    global_momentum: float = 0.9
    """Momentum hyperparameters to use"""

    num_local_steps: int = 10
    """Number of steps of inner optimizer"""

    global_nesterov: bool = True
    """Nesterov momentum"""

@dataclass
class JobConfig:
    optimizer: Optimizer = field(default_factory=Optimizer)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
