# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model transform selecting asynchronous tensor-parallel linears."""

from dataclasses import dataclass
from typing import cast

from torchtitan.models.common.attention import GQAttention
from torchtitan.models.common.dist_gemm import (
    AsyncAllGatherLinear,
    AsyncLinearReduceScatter,
)
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module

from .base import convert_config_type, ModelConfigTransform

__all__ = ["AsyncTensorParallelTransform"]


@dataclass(kw_only=True, slots=True)
class AsyncTensorParallelTransform(ModelConfigTransform):
    """Replace common GQA and FFN TP projections with async variants.

    Model-specific subclasses are left unchanged until their communication
    boundaries have been audited.
    """

    def transform(self, model: Module.Config) -> Module.Config:
        for _, traversed, _, _ in model.traverse(GQAttention.Config):
            if traversed._owner is not GQAttention:
                continue
            attention = cast(GQAttention.Config, traversed)
            attention.qkv_linear.wqkv = cast(
                Linear.Config,
                convert_config_type(
                    attention.qkv_linear.wqkv,
                    AsyncAllGatherLinear,
                ),
            )
            attention.wo = cast(
                Linear.Config,
                convert_config_type(
                    attention.wo,
                    AsyncLinearReduceScatter,
                ),
            )

        for _, traversed, _, _ in model.traverse(FeedForward.Config):
            if traversed._owner is not FeedForward:
                continue
            feed_forward = cast(FeedForward.Config, traversed)
            feed_forward.w1 = cast(
                Linear.Config,
                convert_config_type(
                    feed_forward.w1,
                    AsyncAllGatherLinear,
                ),
            )
            feed_forward.w3 = cast(
                Linear.Config,
                convert_config_type(
                    feed_forward.w3,
                    AsyncAllGatherLinear,
                ),
            )
            feed_forward.w2 = cast(
                Linear.Config,
                convert_config_type(
                    feed_forward.w2,
                    AsyncLinearReduceScatter,
                ),
            )
        return model
