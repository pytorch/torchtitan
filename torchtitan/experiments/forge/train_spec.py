# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

# Import torchtitan.models to ensure all train specs are registered
import torchtitan.models  # noqa: F401

from torchtitan.protocols.train_spec import (
    _train_specs,
    BaseModelArgs,
    LossFunctionBuilder,
    LRSchedulersBuilder,
    ModelProtocol,
    OptimizersBuilder,
    ParallelizeFunction,
    PipeliningFunction,
    TokenizerBuilder,
    TrainSpec,
)


@dataclass
class ForgeTrainSpec:
    name: str
    model_cls: type[ModelProtocol]
    model_args: dict[str, BaseModelArgs]
    parallelize_fn: ParallelizeFunction
    pipelining_fn: PipeliningFunction | None
    build_optimizers_fn: OptimizersBuilder
    build_lr_schedulers_fn: LRSchedulersBuilder
    build_tokenizer_fn: TokenizerBuilder | None
    build_loss_fn: LossFunctionBuilder


# Copy and transform train specs from torchtitan.protocols.train_spec._train_specs
# This happens during import after all models have been registered
_forge_train_specs = {}


def register_train_spec(train_spec: ForgeTrainSpec) -> None:
    global _forge_train_specs
    if train_spec.name in _forge_train_specs:
        raise ValueError(f"Model {train_spec.name} is already registered.")

    _forge_train_specs[train_spec.name] = train_spec


def get_train_spec(name: str) -> ForgeTrainSpec:
    global _forge_train_specs
    if name not in _forge_train_specs:
        raise ValueError(f"Model {name} is not registered.")
    return _forge_train_specs[name]


def _transform_train_spec(original_spec: TrainSpec):
    """Transform the original train spec to ForgeTrainSpec format."""
    # Create a new TrainSpec with only the fields we need in forge
    return ForgeTrainSpec(
        name=original_spec.name,
        model_cls=original_spec.model_cls,
        model_args=original_spec.model_args,
        parallelize_fn=original_spec.parallelize_fn,
        pipelining_fn=original_spec.pipelining_fn,
        build_optimizers_fn=original_spec.build_optimizers_fn,
        build_lr_schedulers_fn=original_spec.build_lr_schedulers_fn,
        build_tokenizer_fn=original_spec.build_tokenizer_fn,
        build_loss_fn=original_spec.build_loss_fn,
    )


# Populate _forge_train_specs with transformed specs
for name, spec in _train_specs.items():
    register_train_spec(_transform_train_spec(spec))
