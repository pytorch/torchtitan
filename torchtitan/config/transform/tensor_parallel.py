# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tensor-parallel model transforms."""

import logging
from dataclasses import dataclass
from typing import cast

from torchtitan.models.common.decoder_sharding import colwise_config, rowwise_config
from torchtitan.models.common.dist_gemm import DistGEMMFeedForward
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.tensor_parallel import TensorParallelFeedForward
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import ShardingConfig

from .base import convert_config_type, ModelConfigTransform

logger = logging.getLogger(__name__)

__all__ = ["TensorParallelFeedForwardTransform"]


@dataclass(kw_only=True, slots=True)
class TensorParallelFeedForwardTransform(ModelConfigTransform):
    """Replace transformer-block dense FFNs with a TP implementation.

    Only configs stored in a ``feed_forward`` field are replaced. MoE shared
    experts are intentionally excluded because their partial output is reduced
    only after it is combined with the routed-expert output.
    """

    feed_forward: type[TensorParallelFeedForward] = TensorParallelFeedForward

    def __post_init__(self) -> None:
        if not issubclass(self.feed_forward, TensorParallelFeedForward):
            raise ValueError(
                f"{self.feed_forward.__qualname__} must inherit "
                "TensorParallelFeedForward."
            )

    def transform(self, model: Module.Config) -> Module.Config:
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

        if num_replaced == 0:
            logger.warning(
                "%s did not find any transformer-block dense feed-forward configs.",
                type(self).__qualname__,
            )
        return model
