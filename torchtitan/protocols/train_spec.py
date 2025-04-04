# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from abc import abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, TypeAlias

import torch
import torch.nn as nn
from torch.distributed.pipelining.schedules import _PipelineSchedule

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.ft import FTManager
from torchtitan.components.loss import LossFunction
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig

DeviceType = int | str | torch.device


@dataclass
class BaseModelArgs:
    """All ModelArgs should inherit from this class.

    The only usage of this class is type checking but allows us to extend common
    arguments to all models in the future.
    """

    _enforced: str = "This field is used to enforce all fields have defaults."

    @abstractmethod
    def update_from_config(self, job_config: JobConfig, tokenizer: Tokenizer) -> None:
        pass

    @abstractmethod
    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, float]:
        pass


class ModelProtocol(Protocol):
    """Defines the interface for a model class.

    This is used to enforce that all model classes have some methods that are
    required by the TorchTitan trainer.
    """

    @classmethod
    def from_model_args(cls, args: BaseModelArgs) -> nn.Module:
        ...


ParallelizeFunction: TypeAlias = Callable[..., nn.Module]
PipeliningFunction: TypeAlias = Callable[
    ..., tuple[_PipelineSchedule, list[nn.Module], bool, bool]
]
DataLoaderBuilder: TypeAlias = Callable[..., BaseDataLoader]
TokenizerBuilder: TypeAlias = Callable[..., Tokenizer]
MetricsProcessorBuilder: TypeAlias = Callable[..., MetricsProcessor]
OptimizersBuilder: TypeAlias = Callable[
    [list[nn.Module], JobConfig, FTManager], OptimizersContainer
]
LRSchedulersBuilder: TypeAlias = Callable[
    [OptimizersContainer, JobConfig], LRSchedulersContainer
]
LossFunctionBuilder: TypeAlias = Callable[..., LossFunction]


@dataclass
class TrainSpec:
    name: str
    cls: type[nn.Module]
    config: Mapping[str, BaseModelArgs]
    parallelize_fn: ParallelizeFunction
    pipelining_fn: PipeliningFunction | None
    build_optimizers_fn: OptimizersBuilder
    build_lr_schedulers_fn: LRSchedulersBuilder
    build_dataloader_fn: DataLoaderBuilder
    build_tokenizer_fn: TokenizerBuilder | None
    build_loss_fn: LossFunctionBuilder
    build_metrics_processor_fn: MetricsProcessorBuilder | None = None


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
