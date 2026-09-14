# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model components selected by tensor-parallel transforms."""

from dataclasses import dataclass

from torchtitan.models.common.feed_forward import FeedForward


class TensorParallelFeedForward(FeedForward):
    """Dense FFN whose projection modules own the TP collectives.

    The subclass marks transformer-block FFNs selected by
    ``TensorParallelTransform``. Sharding setup attaches the input collective
    to ``w13`` and the output collective to ``w2``, so their existing remat
    regions remain communication-complete.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        pass


__all__ = ["TensorParallelFeedForward"]
