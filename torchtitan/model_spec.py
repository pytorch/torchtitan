# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


from dataclasses import dataclass
from typing import Callable, Dict, List, Protocol, Tuple, Type

import torch.nn as nn
from torch.distributed.pipelining.schedules import _PipelineSchedule
from torchtitan.config_manager import JobConfig
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

    @staticmethod
    def from_model_args(self, args: BaseModelArgs) -> nn.Module: ...


@dataclass
class ModelSpec:
    name: str
    cls: Type[nn.Module]
    config: Dict[str, BaseModelArgs]
    # TODO: Add a ``build_dataloader_fn``
    # As for now, this is a string. So it will have to be built-in to the
    # TorchTitan library. A better way would be to have a dataloader class
    # and a ``build_dataloader`` function that take job_config to consume
    # the different dataloader and tokenizer configs.
    tokenizer: str
    parallelize_fn: Callable[[nn.Module], None]
    pipelining_fn: Callable[[nn.Module], Tuple[_PipelineSchedule, List[nn.Module]]]
    build_optimizers_fn: Callable[[List[nn.Module], JobConfig], OptimizersContainer]
    build_lr_schedulers_fn: Callable[
        [List[nn.Module], JobConfig], LRSchedulersContainer
    ]

    # TODO: Add a FQN convert fn to allow users to load checkpoints from
    # HuggingFace or other sources that have different FQN conventions.


_model_specs = {}


def register_model_spec(model_spec: ModelSpec) -> None:
    global _model_specs
    if model_spec.name in _model_specs:
        raise ValueError(f"Model {model_spec.name} is already registered.")
    _model_specs[model_spec.name] = model_spec


def get_model_spec(name: str) -> ModelSpec:
    global _model_specs
    if name not in _model_specs:
        raise ValueError(f"Model {name} is not registered.")
    return _model_specs[name]
