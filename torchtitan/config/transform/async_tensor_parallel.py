# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Asynchronous tensor-parallel model transform."""

from dataclasses import dataclass
from typing import cast

from torchtitan.models.common.async_linear import (
    AsyncColumnParallelLinear,
    AsyncRowParallelLinear,
)
from torchtitan.models.common.linear import (
    ColumnParallelLinear,
    Linear,
    parallel_linear_role,
    RowParallelLinear,
)
from torchtitan.protocols.module import Module

from .base import convert_config_type, ModelConfigTransform
from .lora import LoRATransform

__all__ = ["AsyncTensorParallelTransform"]


def _convert_linear(
    config: Linear.Config,
    replacement: type[ColumnParallelLinear] | type[RowParallelLinear],
    *,
    projection_name: str,
) -> ColumnParallelLinear.Config | RowParallelLinear.Config:
    expected = (
        ColumnParallelLinear.Config
        if replacement is AsyncColumnParallelLinear
        else RowParallelLinear.Config
    )
    if type(config) is expected:
        return cast(
            ColumnParallelLinear.Config | RowParallelLinear.Config,
            convert_config_type(config, replacement),
        )
    raise ValueError(
        "Async tensor parallelism does not support converted "
        f"{projection_name} projections"
    )


def _transform_parallel_linears(model: Module.Config) -> Module.Config:
    """Replace synchronous TP projection configs with async implementations."""
    for fqn, traversed, parent, attr in list(model.traverse(Linear.Config)):
        role = parallel_linear_role(traversed)
        if role is None:
            continue
        replacement = (
            AsyncColumnParallelLinear
            if role is ColumnParallelLinear
            else AsyncRowParallelLinear
        )
        converted = _convert_linear(
            traversed,
            replacement,
            projection_name=fqn or type(traversed).__qualname__,
        )
        if parent is None:
            model = cast(Module.Config, converted)
        elif isinstance(parent, list):
            assert isinstance(attr, int)
            parent[attr] = converted
        else:
            assert isinstance(attr, str)
            setattr(parent, attr, converted)
    return model


@dataclass(kw_only=True, slots=True)
class AsyncTensorParallelTransform(ModelConfigTransform):
    """Replace synchronous tensor-parallel projections with async versions."""

    def transform(self, model: Module.Config) -> Module.Config:
        return _transform_parallel_linears(model)


# Async kernels call their fused autograd functions directly instead of the
# projection's ``_linear`` method, so they would silently omit LoRA computation.
AsyncTensorParallelTransform.conflicts_with = (LoRATransform,)
