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

from torchtitan.models.common.attention import GQAttention, QKVLinear
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import (
    ColumnParallelLinear,
    Linear,
    RowParallelLinear,
)
from torchtitan.protocols.module import Module

from .base import convert_config_type, ModelConfigTransform
from .lora import LoRATransform

__all__ = ["AsyncTensorParallelTransform"]


def _convert_linear(
    config: Linear.Config,
    replacement: type[Linear],
    *,
    projection_name: str,
) -> Linear.Config:
    expected = (
        ColumnParallelLinear.Config
        if replacement is AsyncColumnParallelLinear
        else RowParallelLinear.Config
    )
    if type(config) is expected and not config._module_decorators:
        return cast(Linear.Config, convert_config_type(config, replacement))
    raise ValueError(
        "Async tensor parallelism does not support converted "
        f"{projection_name} projections"
    )


def _transform_attention(model: Module.Config) -> Module.Config:
    """Replace common GQA synchronous projections with async implementations."""
    for _, traversed, parent, attr in model.traverse(GQAttention.Config):
        existing = cast(GQAttention.Config, traversed)
        if existing._owner is not GQAttention:
            continue

        if existing.qkv_linear._owner is not QKVLinear:
            raise ValueError(
                "Async tensor parallelism requires the common QKVLinear "
                "implementation"
            )
        existing.qkv_linear.wqkv = _convert_linear(
            existing.qkv_linear.wqkv,
            AsyncColumnParallelLinear,
            projection_name="QKV",
        )
        existing.wo = _convert_linear(
            existing.wo,
            AsyncRowParallelLinear,
            projection_name="attention output",
        )
    return model


def _transform_feed_forward(
    model: Module.Config,
) -> Module.Config:
    """Replace common dense FFN synchronous projections with async versions.

    Configs stored as ``feed_forward`` fields are transformed, along with a
    root FFN config. Shared experts remain synchronous because their input
    layout depends on EP.
    """
    for _, traversed, parent, attr in model.traverse(FeedForward.Config):
        is_root = parent is None
        is_dense_block = attr == "feed_forward"
        if not is_root and not is_dense_block:
            continue
        existing = cast(FeedForward.Config, traversed)
        if existing._owner is not FeedForward:
            continue

        existing.w13 = _convert_linear(
            existing.w13,
            AsyncColumnParallelLinear,
            projection_name="w13",
        )
        existing.w2 = _convert_linear(
            existing.w2,
            AsyncRowParallelLinear,
            projection_name="w2",
        )
    return model


@dataclass(kw_only=True, slots=True)
class AsyncTensorParallelTransform(ModelConfigTransform):
    """Select async tensor-parallel attention and dense FFN projections."""

    def transform(self, model: Module.Config) -> Module.Config:
        model = _transform_attention(model)
        return _transform_feed_forward(model)


# Async kernels call their fused autograd functions directly instead of the
# wrapped projection's forward, so they would silently omit LoRA computation.
AsyncTensorParallelTransform.conflicts_with = (LoRATransform,)
