# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model transform selecting asynchronous tensor-parallel linears."""

from dataclasses import dataclass

from torchtitan.models.common.attention import AllGatherQKVLinear

from torchtitan.models.common.dist_gemm import (
    AsyncAllGatherLinear,
    AsyncAllGatherQKVLinear,
    AsyncLinearReduceScatter,
)
from torchtitan.models.common.linear import AllGatherLinear, LinearReduceScatter
from torchtitan.protocols.module import Module

from .base import convert_config_type, ModelConfigTransform

__all__ = ["AsyncTensorParallelTransform"]


@dataclass(kw_only=True, slots=True)
class AsyncTensorParallelTransform(ModelConfigTransform):
    """Replace synchronous TP projection roles with async variants."""

    def transform(self, model: Module.Config) -> Module.Config:
        for source, replacement in (
            (AllGatherQKVLinear, AsyncAllGatherQKVLinear),
            (AllGatherLinear, AsyncAllGatherLinear),
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
