# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass
from typing import Mapping, TypeAlias

import torch.nn as nn
from torch.distributed.pipelining.schedules import _PipelineSchedule

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.loss import LossFunction
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.components.validate import BaseValidator
from torchtitan.config import LRScheduler

from .model import BaseModelArgs, ModelProtocol
from .state_dict_adapter import BaseStateDictAdapter


ParallelizeFunction: TypeAlias = Callable[..., nn.Module]
PipeliningFunction: TypeAlias = Callable[
    ..., tuple[_PipelineSchedule, list[nn.Module], bool, bool]
]
DataLoaderBuilder: TypeAlias = Callable[..., BaseDataLoader]
TokenizerBuilder: TypeAlias = Callable[..., BaseTokenizer]
MetricsProcessorBuilder: TypeAlias = Callable[..., MetricsProcessor]
OptimizersBuilder: TypeAlias = Callable[..., OptimizersContainer]
LRSchedulersBuilder: TypeAlias = Callable[
    [OptimizersContainer, LRScheduler, int], LRSchedulersContainer
]
LossFunctionBuilder: TypeAlias = Callable[..., LossFunction]
ValidatorBuilder: TypeAlias = Callable[..., BaseValidator]


@dataclass
class TrainSpec:
    name: str
    model_cls: type[ModelProtocol]
    model_args: Mapping[str, BaseModelArgs]
    parallelize_fn: ParallelizeFunction
    pipelining_fn: PipeliningFunction | None
    build_optimizers_fn: OptimizersBuilder
    build_lr_schedulers_fn: LRSchedulersBuilder
    build_dataloader_fn: DataLoaderBuilder
    build_tokenizer_fn: TokenizerBuilder | None
    build_loss_fn: LossFunctionBuilder
    build_validator_fn: ValidatorBuilder | None = None
    build_metrics_processor_fn: MetricsProcessorBuilder | None = None
    state_dict_adapter: type[BaseStateDictAdapter] | None = None


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
