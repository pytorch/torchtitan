# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch.nn as nn
from torchtitan.protocols.train_spec import TrainSpec


FragmentFunction: TypeAlias = Callable[..., list[nn.Module]]


@dataclass
class FaultTolerantTrainSpec(TrainSpec):
    fragment_fn: FragmentFunction | None = None
