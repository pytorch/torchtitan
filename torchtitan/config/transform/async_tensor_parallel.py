# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Asynchronous tensor-parallel model transform."""

import logging
from dataclasses import dataclass
from typing import cast

from torchtitan.models.common.async_linear import (
    AsyncColumnParallelLinear,
    AsyncRowParallelLinear,
)
from torchtitan.models.common.linear import (
    ColumnParallelLinear,
    get_parallel_linear_cls,
    Linear,
    RowParallelLinear,
)
from torchtitan.protocols.module import Module

from .base import convert_config_type, ModelConfigTransform
from .lora import LoRATransform

__all__ = ["AsyncTensorParallelTransform"]

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class AsyncTensorParallelTransform(ModelConfigTransform):
    """Replace synchronous tensor-parallel projections with async versions."""

    enable_sequence_parallel: bool

    def transform(self, model: Module.Config) -> Module.Config:
        if not self.enable_sequence_parallel:
            logger.warning(
                "Async tensor parallelism requires sequence parallelism; "
                "leaving synchronous tensor-parallel projections unchanged."
            )
            return model

        for fqn, config, parent, attr in list(model.traverse(Linear.Config)):
            parallel_cls = get_parallel_linear_cls(config)
            if parallel_cls is None:
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
AsyncTensorParallelTransform.conflicts_with = (LoRATransform,)
