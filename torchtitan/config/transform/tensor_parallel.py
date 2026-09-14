# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tensor-parallel model transforms."""

import logging
from dataclasses import dataclass
from typing import cast

from torchtitan.models.common.attention import GQAttention, QKVLinear
from torchtitan.models.common.decoder_sharding import colwise_config, rowwise_config
from torchtitan.models.common.dist_gemm import (
    AsyncAllGatherLinear,
    AsyncAllGatherQKVLinear,
    AsyncLinearReduceScatter,
)
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import AllGatherLinear, Linear, LinearReduceScatter
from torchtitan.models.common.tensor_parallel import TensorParallelFeedForward
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import ShardingConfig

from .base import convert_config_type, ModelConfigTransform

logger = logging.getLogger(__name__)

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


def _transform_attention(model: Module.Config, *, async_tp: bool) -> int:
    qkv_linear = AsyncAllGatherQKVLinear if async_tp else QKVLinear
    output_linear = AsyncLinearReduceScatter if async_tp else LinearReduceScatter
    num_replaced = 0

    for _, traversed, _, _ in model.traverse(GQAttention.Config):
        attention = cast(GQAttention.Config, traversed)
        if attention._owner is not GQAttention:
            continue

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
            attention.qkv_linear = cast(
                QKVLinear.Config,
                convert_config_type(attention.qkv_linear, qkv_linear),
            )
        attention.wo = _convert_linear(
            attention.wo,
            output_linear,
            async_tp=async_tp,
            projection_name="attention output",
        )
        num_replaced += 1

    return num_replaced


def _transform_feed_forward(
    model: Module.Config,
    *,
    async_tp: bool,
) -> tuple[Module.Config, int]:
    input_linear = AsyncAllGatherLinear if async_tp else AllGatherLinear
    output_linear = AsyncLinearReduceScatter if async_tp else LinearReduceScatter
    num_replaced = 0

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

        replacement.w13.sharding_config = colwise_config()
        w2_sharding = rowwise_config()
        replacement.w2.sharding_config = ShardingConfig(
            state_shardings=w2_sharding.state_shardings,
            out_src_shardings=w2_sharding.out_src_shardings,
        )
        if is_root:
            model = replacement
        else:
            assert parent is not None
            assert isinstance(attr, str)
            setattr(parent, attr, replacement)
        num_replaced += 1

    return model, num_replaced


def _transform_tensor_parallel(
    model: Module.Config,
    *,
    async_tp: bool,
) -> tuple[Module.Config, int]:
    num_replaced = _transform_attention(model, async_tp=async_tp)
    model, num_feed_forwards = _transform_feed_forward(model, async_tp=async_tp)
    num_replaced += num_feed_forwards
    return model, num_replaced


@dataclass(kw_only=True, slots=True)
class TensorParallelTransform(ModelConfigTransform):
    """Select synchronous TP projections for common attention and dense FFNs."""

    def transform(self, model: Module.Config) -> Module.Config:
        model, num_replaced = _transform_tensor_parallel(model, async_tp=False)
        if num_replaced == 0:
            logger.warning(
                "%s did not find any supported attention or feed-forward configs.",
                type(self).__qualname__,
            )
        return model


@dataclass(kw_only=True, slots=True)
class AsyncTensorParallelTransform(ModelConfigTransform):
    """Select async tensor-parallel attention and dense FFN projections."""

    def transform(self, model: Module.Config) -> Module.Config:
        model, num_replaced = _transform_tensor_parallel(model, async_tp=True)
        if num_replaced == 0:
            logger.warning(
                "%s did not find any supported attention or feed-forward configs.",
                type(self).__qualname__,
            )
        return model
