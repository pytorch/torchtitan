# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from dataclasses import dataclass
from typing import Callable, Protocol, Type, TypeAlias

import torch
import torch.nn as nn
from torch.distributed.pipelining.schedules import _PipelineSchedule

from torchtitan.config_manager import JobConfig
from torchtitan.dataloader import DataLoaderBuilder
from torchtitan.datasets.tokenizer import Tokenizer
from torchtitan.optimizer import LRSchedulersContainer, OptimizersContainer


@dataclass
class BaseModelArgs:
    """All ModelArgs should inherit from this class.

    The only usage of this class is type checking but allows us to extend common
    arguments to all models in the future.
    """

    _enforced: str = "This field is used to enforce all fields have defaults."


class ModelProtocol(Protocol):
    """Defines the interface for a model class.

    This is used to enforce that all model classes have some methods that are
    required by the TorchTitan trainer.
    """

    @classmethod
    def from_model_args(cls, args: BaseModelArgs) -> nn.Module:
        ...


OptimizersBuilder: TypeAlias = Callable[
    [list[nn.Module], JobConfig], OptimizersContainer
]
LRSchedulersBuilder: TypeAlias = Callable[[OptimizersContainer], LRSchedulersContainer]
LossFunction: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class TrainSpec:
    name: str
    cls: Type[nn.Module]
    config: dict[str, BaseModelArgs]
    parallelize_fn: Callable[[nn.Module], None]
    pipelining_fn: Callable[
        [nn.Module], tuple[_PipelineSchedule, list[nn.Module], bool, bool]
    ]
    build_optimizers_fn: OptimizersBuilder
    build_lr_schedulers_fn: LRSchedulersBuilder
    build_dataloader_fn: DataLoaderBuilder
    tokenizer_cls: Type[Tokenizer]
    loss_fn: LossFunction

    # TODO: Add a FQN convert fn to allow users to load checkpoints from
    # HuggingFace or other sources that have different FQN conventions.


_train_specs = {}


def register_train_spec(train_spec: TrainSpec) -> None:
    global _train_specs
    if train_spec.name in _train_specs:
        raise ValueError(f"Model {train_spec.name} is already registered.")

    _train_specs[train_spec.name] = train_spec


def get_train_spec(name: str) -> TrainSpec:
    global _train_specs
    if name not in _train_specs:
        raise ValueError(f"Model {name} is not registered.")
    return _train_specs[name]


def apply_to_train_specs(func: Callable[[TrainSpec], TrainSpec]) -> None:
    global _train_specs
    for name, train_spec in _train_specs.items():
        _train_specs[name] = func(train_spec)
