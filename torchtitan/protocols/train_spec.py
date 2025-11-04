# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
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


_extra_train_specs: dict[str, TrainSpec] = {}


def register_train_spec(name: str, train_spec: TrainSpec) -> None:
    global _extra_train_specs
    if name in _extra_train_specs:
        raise ValueError(f"TrainSpec {name} is already registered.")

    # user can define a TrainSpec from outside of torchtitan
    _extra_train_specs[name] = train_spec


def get_train_spec(name: str) -> TrainSpec:
    # user-defined TrainSpec has higher priority
    global _extra_train_specs
    if name in _extra_train_specs:
        return _extra_train_specs[name]

    from torchtitan.experiments import _supported_experiments
    from torchtitan.models import _supported_models

    if name in _supported_models:
        module = import_module(f"torchtitan.models.{name}")
        return module.get_train_spec()
    elif name in _supported_experiments:
        module = import_module(f"torchtitan.experiments.{name}")
        return module.get_train_spec()

    raise ValueError(f"TrainSpec {name} is not registered.")
