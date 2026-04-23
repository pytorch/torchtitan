# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import torch.nn as nn
from torch.distributed.pipelining.schedules import _PipelineSchedule

from torchtitan.components.loss import LossFunction
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter

# Type aliases for ModelSpec callables
ParallelizeFunction: TypeAlias = Callable[..., nn.Module]
PipeliningFunction: TypeAlias = Callable[
    ..., tuple[_PipelineSchedule, list[nn.Module], bool, bool]
]
LossFunctionBuilder: TypeAlias = Callable[..., LossFunction]
FragmentFunction: TypeAlias = Callable[..., list[nn.Module]]
PostOptimizerBuildFn: TypeAlias = Callable[..., None]


@dataclass
class ModelSpec:
    """Per-model bundle. Contains already-selected arch config + callables."""

    name: str
    flavor: str
    model: BaseModel.Config
    # TODO: improve the serializability of ModelSpec by refactoring the following
    #       fields, e.g. by having their own classes, or hard-coding into trainer
    # NOTE: Callable fields use bare ``Callable`` instead of the parameterised
    # TypeAliases (e.g. ``ParallelizeFunction``) because tyro's type-parameter
    # resolver does not handle ``Callable[..., X]`` (Ellipsis as param spec).
    # The detailed TypeAliases above are still available for use in function
    # signatures elsewhere in the codebase.
    build_loss_fn: Callable
    parallelize_fn: Callable
    pipelining_fn: Callable | None
    post_optimizer_build_fn: Callable | None
    state_dict_adapter: type[BaseStateDictAdapter] | None


@dataclass
class FaultTolerantModelSpec(ModelSpec):
    fragment_fn: Callable | None
