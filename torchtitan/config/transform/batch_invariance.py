# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model transform for batch-invariant FlexAttention settings."""

from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import FlexInnerAttention

from .base import ModelConfigTransform

__all__ = ["BatchInvariantFlexTransform"]


class BatchInvariantFlexTransform(ModelConfigTransform):
    """Pin FlexAttention kernel options for batch-invariant mode.

    Sets fixed BLOCK_M/BLOCK_N=16 and BACKEND=TRITON on all
    FlexInnerAttention layers.

    BACKEND=TRITON avoids the flex_decode kernel.
    """

    # The Triton BLOCK_N tile size must be pinned for stable numerics.
    _BLOCK_M = 16
    _BLOCK_N = 16

    def transform(self, model: Module.Config) -> Module.Config:
        for _, config, _, _ in model.traverse(FlexInnerAttention.Config):
            config.kernel_options["BACKEND"] = "TRITON"
            config.kernel_options["BLOCK_M"] = self._BLOCK_M
            config.kernel_options["BLOCK_N"] = self._BLOCK_N
        return model
