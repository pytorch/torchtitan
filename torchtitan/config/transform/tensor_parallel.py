# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tensor-parallel model transforms."""

import logging
from dataclasses import dataclass
from typing import cast

from torchtitan.models.common.attention import (
    AllGatherQKVLinear,
    GQAttention,
    QKVLinear,
)
from torchtitan.models.common.decoder_sharding import colwise_config, rowwise_config
from torchtitan.models.common.dist_gemm import (
    AsyncAllGatherQKVLinear,
    AsyncLinearReduceScatter,
    DistGEMMFeedForward,
)
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear, LinearReduceScatter
from torchtitan.models.common.tensor_parallel import TensorParallelFeedForward
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import ShardingConfig

from .base import convert_config_type, ModelConfigTransform

logger = logging.getLogger(__name__)

__all__ = ["AsyncTensorParallelTransform", "TensorParallelTransform"]


@dataclass(kw_only=True, slots=True)
class TensorParallelTransform(ModelConfigTransform):
    """Select tensor-parallel attention projections and dense FFNs.

    Only the common ``GQAttention`` and configs stored in a ``feed_forward``
    field are replaced. Attention subclasses may share communication across
    projections, while MoE shared experts reduce their partial output only
    after combining it with the routed-expert output.
    """

    qkv_linear: type[AllGatherQKVLinear] = AllGatherQKVLinear
    output_linear: type[LinearReduceScatter] = LinearReduceScatter
    feed_forward: type[TensorParallelFeedForward] = TensorParallelFeedForward

    def __post_init__(self) -> None:
        if not issubclass(self.qkv_linear, AllGatherQKVLinear):
            raise ValueError(
                f"{self.qkv_linear.__qualname__} must inherit AllGatherQKVLinear."
            )
        if not issubclass(self.output_linear, LinearReduceScatter):
            raise ValueError(
                f"{self.output_linear.__qualname__} must inherit "
                "LinearReduceScatter."
            )
        if not issubclass(self.feed_forward, TensorParallelFeedForward):
            raise ValueError(
                f"{self.feed_forward.__qualname__} must inherit "
                "TensorParallelFeedForward."
            )

    def transform(self, model: Module.Config) -> Module.Config:
        num_replaced = self._transform_attention(model)
        model, num_feed_forwards = self._transform_feed_forward(model)
        num_replaced += num_feed_forwards
        if num_replaced == 0:
            logger.warning(
                "%s did not find any supported attention or feed-forward configs.",
                type(self).__qualname__,
            )
        return model

    def _transform_attention(self, model: Module.Config) -> int:
        num_replaced = 0
        for _, traversed, _, _ in model.traverse(GQAttention.Config):
            attention = cast(GQAttention.Config, traversed)
            if attention._owner is not GQAttention:
                continue

            if attention.qkv_linear._owner is QKVLinear:
                attention.qkv_linear = cast(
                    QKVLinear.Config,
                    convert_config_type(attention.qkv_linear, self.qkv_linear),
                )
                num_replaced += 1
            if attention.wo._owner is Linear:
                attention.wo = cast(
                    Linear.Config,
                    convert_config_type(attention.wo, self.output_linear),
                )
                num_replaced += 1
        return num_replaced

    def _transform_feed_forward(
        self, model: Module.Config
    ) -> tuple[Module.Config, int]:
        num_replaced = 0
        for _, traversed, parent, attr in model.traverse(FeedForward.Config):
            is_root = parent is None
            if not is_root and attr != "feed_forward":
                continue
            existing = cast(FeedForward.Config, traversed)
            if existing._owner is not FeedForward:
                continue

            replacement = convert_config_type(existing, self.feed_forward)
            assert isinstance(replacement, TensorParallelFeedForward.Config)

            replacement.w13.sharding_config = colwise_config()
            w2_sharding = rowwise_config()
            replacement.w2.sharding_config = ShardingConfig(
                state_shardings=w2_sharding.state_shardings,
                out_src_shardings=(
                    None
                    if issubclass(self.feed_forward, DistGEMMFeedForward)
                    else w2_sharding.out_src_shardings
                ),
            )
            if is_root:
                model = replacement
            else:
                assert parent is not None
                assert isinstance(attr, str)
                setattr(parent, attr, replacement)
            num_replaced += 1
        return model, num_replaced


@dataclass(kw_only=True, slots=True)
class AsyncTensorParallelTransform(ModelConfigTransform):
    """Select async attention projections and the dist-GEMM dense FFN."""

    def transform(self, model: Module.Config) -> Module.Config:
        return TensorParallelTransform(
            qkv_linear=AsyncAllGatherQKVLinear,
            output_linear=AsyncLinearReduceScatter,
            feed_forward=DistGEMMFeedForward,
        ).transform(model)
