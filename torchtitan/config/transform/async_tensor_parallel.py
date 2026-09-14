# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model transform selecting asynchronous tensor-parallel implementations."""

from dataclasses import dataclass

from torchtitan.models.common.dist_gemm import (
    AsyncAllGatherQKVLinear,
    AsyncLinearReduceScatter,
    DistGEMMFeedForward,
)
from torchtitan.protocols.module import Module

from .base import ModelConfigTransform
from .tensor_parallel import TensorParallelTransform

__all__ = ["AsyncTensorParallelTransform"]


@dataclass(kw_only=True, slots=True)
class AsyncTensorParallelTransform(ModelConfigTransform):
    """Select async attention projections and the dist-GEMM dense FFN."""

    def transform(self, model: Module.Config) -> Module.Config:
        return TensorParallelTransform(
            qkv_linear=AsyncAllGatherQKVLinear,
            output_linear=AsyncLinearReduceScatter,
            feed_forward=DistGEMMFeedForward,
        ).transform(model)
