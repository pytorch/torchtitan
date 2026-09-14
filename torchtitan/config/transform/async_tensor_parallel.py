# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model transform selecting asynchronous tensor-parallel implementations."""

from dataclasses import dataclass

from torchtitan.models.common.attention import AllGatherQKVLinear

from torchtitan.models.common.dist_gemm import (
    AsyncAllGatherQKVLinear,
    AsyncLinearReduceScatter,
    DistGEMMFeedForward,
)
from torchtitan.models.common.linear import LinearReduceScatter
from torchtitan.protocols.module import Module

from .base import convert_config_type, ModelConfigTransform
from .tensor_parallel import TensorParallelFeedForwardTransform

__all__ = ["AsyncTensorParallelTransform"]


@dataclass(kw_only=True, slots=True)
class AsyncTensorParallelTransform(ModelConfigTransform):
    """Select async attention projections and the dist-GEMM dense FFN."""

    def transform(self, model: Module.Config) -> Module.Config:
        TensorParallelFeedForwardTransform(feed_forward=DistGEMMFeedForward).transform(
            model
        )
        for source, replacement in (
            (AllGatherQKVLinear, AsyncAllGatherQKVLinear),
            (LinearReduceScatter, AsyncLinearReduceScatter),
        ):
            for _, config, parent, attr in list(
                model.traverse(source.Config, recurse=True)
            ):
                if config._owner is not source:
                    continue
                assert isinstance(config, Module.Config)
                assert parent is not None
                converted = convert_config_type(config, replacement)
                if isinstance(parent, list):
                    assert isinstance(attr, int)
                    parent[attr] = converted
                else:
                    assert isinstance(attr, str)
                    setattr(parent, attr, converted)
        return model
