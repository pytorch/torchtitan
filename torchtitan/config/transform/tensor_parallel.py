# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tensor-parallel model transforms."""

from dataclasses import dataclass
from typing import cast

from torchtitan.models.common.attention import GQAttention, QKVLinear
from torchtitan.models.common.dist_gemm import (
    AsyncColumnParallelLinear,
    AsyncRowParallelLinear,
)
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import (
    ColumnParallelLinear,
    Linear,
    RowParallelLinear,
)
from torchtitan.models.common.tensor_parallel import TensorParallelFeedForward
from torchtitan.protocols.module import Module

from .base import convert_config_type, ModelConfigTransform

__all__ = ["AsyncTensorParallelTransform", "TensorParallelTransform"]


def _convert_linear(
    config: Linear.Config,
    replacement: type[Linear],
    *,
    async_tp: bool,
    projection_name: str,
) -> Linear.Config:
    if type(config) is Linear.Config:
        return cast(Linear.Config, convert_config_type(config, replacement))
    if async_tp:
        raise ValueError(
            "Async tensor parallelism does not support converted "
            f"{projection_name} projections"
        )
    return config


def _transform_attention(model: Module.Config, *, async_tp: bool) -> None:
    """Select TP projection roles for common GQA blocks.

    Synchronous TP keeps the inner QKV projection unchanged because the
    ``QKVLinear`` boundary owns its input redistribution. Async TP replaces
    that projection so the all-gather can overlap its GEMM. Both modes mark
    the output projection with the corresponding row-parallel implementation.
    Model-specific ``GQAttention`` subclasses retain their existing behavior.
    """
    output_linear = AsyncRowParallelLinear if async_tp else RowParallelLinear
    for _, traversed, _, _ in model.traverse(GQAttention.Config):
        attention = cast(GQAttention.Config, traversed)
        if attention._owner is not GQAttention:
            continue

        # Synchronous TP attaches the input redistribution to QKVLinear, so
        # its inner wqkv receives replicated input and remains a normal Linear.
        # Async TP must replace wqkv because it fuses that all-gather with the
        # projection GEMM itself.
        if async_tp:
            if attention.qkv_linear._owner is not QKVLinear:
                raise ValueError(
                    "Async tensor parallelism requires the common QKVLinear "
                    "implementation"
                )
            if (
                type(attention.qkv_linear) is not QKVLinear.Config
                or type(attention.qkv_linear.wqkv) is not Linear.Config
            ):
                raise ValueError(
                    "Async tensor parallelism does not support converted "
                    "QKV projections"
                )
            attention.qkv_linear.wqkv = _convert_linear(
                attention.qkv_linear.wqkv,
                AsyncColumnParallelLinear,
                async_tp=True,
                projection_name="QKV",
            )
        attention.wo = _convert_linear(
            attention.wo,
            output_linear,
            async_tp=async_tp,
            projection_name="attention output",
        )


def _transform_feed_forward(
    model: Module.Config,
    *,
    async_tp: bool,
) -> Module.Config:
    """Select TP projection roles for transformer-block dense FFNs.

    Only configs stored as ``feed_forward`` fields, plus a root FFN config, are
    transformed; shared-expert FFNs are intentionally left unchanged. The
    outer ``TensorParallelFeedForward`` marker records that projection leaves
    own the collectives even when a converter preserves a different Linear
    implementation for ``w13`` or ``w2``.
    """
    input_linear = AsyncColumnParallelLinear if async_tp else ColumnParallelLinear
    output_linear = AsyncRowParallelLinear if async_tp else RowParallelLinear
    for _, traversed, parent, attr in model.traverse(FeedForward.Config):
        is_root = parent is None
        if not is_root and attr != "feed_forward":
            continue
        existing = cast(FeedForward.Config, traversed)
        if type(existing) is not FeedForward.Config:
            if async_tp and existing._owner is FeedForward:
                raise ValueError(
                    "Async tensor parallelism does not support converted "
                    "FeedForward configs"
                )
            continue

        replacement = convert_config_type(existing, TensorParallelFeedForward)
        assert isinstance(replacement, TensorParallelFeedForward.Config)
        replacement.w13 = _convert_linear(
            replacement.w13,
            input_linear,
            async_tp=async_tp,
            projection_name="w13",
        )
        replacement.w2 = _convert_linear(
            replacement.w2,
            output_linear,
            async_tp=async_tp,
            projection_name="w2",
        )

        if is_root:
            model = replacement
        else:
            assert parent is not None
            assert isinstance(attr, str)
            setattr(parent, attr, replacement)
    return model


def _transform_tensor_parallel(
    model: Module.Config,
    *,
    async_tp: bool,
) -> Module.Config:
    _transform_attention(model, async_tp=async_tp)
    return _transform_feed_forward(model, async_tp=async_tp)


@dataclass(kw_only=True, slots=True)
class TensorParallelTransform(ModelConfigTransform):
    """Select synchronous TP projections for common attention and dense FFNs."""

    def transform(self, model: Module.Config) -> Module.Config:
        return _transform_tensor_parallel(model, async_tp=False)


@dataclass(kw_only=True, slots=True)
class AsyncTensorParallelTransform(ModelConfigTransform):
    """Select async tensor-parallel attention and dense FFN projections."""

    def transform(self, model: Module.Config) -> Module.Config:
        return _transform_tensor_parallel(model, async_tp=True)
