# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from importlib import import_module
from typing import Mapping

from torchtitan.protocols import BaseModelArgs, BaseStateDictAdapter, ModelProtocol
from torchtitan.protocols.train_spec import (
    LossFunctionBuilder,
    LRSchedulersBuilder,
    OptimizersBuilder,
    ParallelizeFunction,
    PipeliningFunction,
    TrainSpec,
)


@dataclass
class ForgeTrainSpec:
    model_cls: type[ModelProtocol]
    model_args: Mapping[str, BaseModelArgs]
    parallelize_fn: ParallelizeFunction
    pipelining_fn: PipeliningFunction | None
    build_optimizers_fn: OptimizersBuilder
    build_lr_schedulers_fn: LRSchedulersBuilder
    build_loss_fn: LossFunctionBuilder
    state_dict_adapter: type[BaseStateDictAdapter] | None = None


_extra_train_specs: dict[str, ForgeTrainSpec] = {}


def _transform_train_spec(original_spec: TrainSpec):
    """Transform the original train spec to ForgeTrainSpec format."""
    # Create a new TrainSpec with only the fields we need in forge
    return ForgeTrainSpec(
        model_cls=original_spec.model_cls,
        model_args=original_spec.model_args,
        parallelize_fn=original_spec.parallelize_fn,
        pipelining_fn=original_spec.pipelining_fn,
        build_optimizers_fn=original_spec.build_optimizers_fn,
        build_lr_schedulers_fn=original_spec.build_lr_schedulers_fn,
        build_loss_fn=original_spec.build_loss_fn,
        state_dict_adapter=original_spec.state_dict_adapter,
    )


def register_train_spec(name: str, train_spec: ForgeTrainSpec) -> None:
    global _extra_train_specs
    if name in _extra_train_specs:
        raise ValueError(f"ForgeTrainSpec {name} is already registered.")

    # user can define a ForgeTrainSpec from outside of torchtitan
    _extra_train_specs[name] = train_spec


def get_train_spec(name: str) -> ForgeTrainSpec:
    # user-defined ForgeTrainSpec has higher priority
    global _extra_train_specs
    if name in _extra_train_specs:
        return _extra_train_specs[name]

    from torchtitan.experiments import _supported_experiments
    from torchtitan.models import _supported_models

    assert _supported_models.isdisjoint(_supported_experiments)

    if name in _supported_models:
        module = import_module(f"torchtitan.models.{name}")
        return _transform_train_spec(module.get_train_spec())
    elif name in _supported_experiments:
        module = import_module(f"torchtitan.experiments.{name}")
        return _transform_train_spec(module.get_train_spec())

    raise ValueError(f"ForgeTrainSpec {name} is not registered.")
