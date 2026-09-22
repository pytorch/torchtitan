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
    RowParallelLinear,
)
from torchtitan.protocols.module import Module

from .base import convert_config_type, ModelConfigTransform
from .lora import LoRATransform

__all__ = ["AsyncTensorParallelTransform"]


@dataclass(kw_only=True, slots=True)
class AsyncTensorParallelTransform(ModelConfigTransform):
    """Replace synchronous dense tensor-parallel projections with async versions."""

    enable_sequence_parallel: bool

    def transform(self, model: Module.Config) -> Module.Config:
        if not self.enable_sequence_parallel:
            raise ValueError("Async tensor parallelism requires sequence parallelism.")

        for fqn, config, parent, attr in list(model.traverse(Linear.Config)):
            owner = config._owner
            assert owner is not None
            if issubclass(owner, ColumnParallelLinear):
                parallel_cls = ColumnParallelLinear
            elif issubclass(owner, RowParallelLinear):
                parallel_cls = RowParallelLinear
            else:
                continue
            if type(config) is not parallel_cls.Config:
                projection_name = fqn or type(config).__qualname__
                raise ValueError(
                    "Async tensor parallelism does not support converted "
                    f"{projection_name} projections"
                )
            replacement = (
                AsyncColumnParallelLinear
                if parallel_cls is ColumnParallelLinear
                else AsyncRowParallelLinear
            )
            converted = cast(
                ColumnParallelLinear.Config | RowParallelLinear.Config,
                convert_config_type(config, replacement),
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


# Async kernels call their fused autograd functions directly instead of the
# projection's ``_linear`` method, so they would silently omit LoRA computation.
# TODO: Add quantization transforms to this conflict list when quantization
# migrates from ModelConfigConverter to ModelConfigTransform.
AsyncTensorParallelTransform.conflicts_with = (LoRATransform,)
