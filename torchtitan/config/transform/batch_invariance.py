# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-config converter for batch-invariant FlexAttention settings."""

from dataclasses import dataclass

from torchtitan.models.common.attention import FlexInnerAttention

from .converter import ModelConfigConverter

__all__ = ["BatchInvariantFlexConverter"]


class BatchInvariantFlexConverter(ModelConfigConverter):
    """Pin FlexAttention kernel options for batch-invariant mode.

    Sets fixed BLOCK_M/BLOCK_N=16 and BACKEND=TRITON on all
    FlexInnerAttention layers.

    BACKEND=TRITON avoids the flex_decode kernel.
    """

    # The Triton BLOCK_N tile size must be pinned for stable numerics.
    _BLOCK_M = 16
    _BLOCK_N = 16

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        pass

    def __init__(self, config: Config):
        pass

    def convert(self, model_config):
        for layer_cfg in model_config.layers:
            inner = layer_cfg.attention.inner_attention
            if isinstance(inner, FlexInnerAttention.Config):
                inner.kernel_options["BACKEND"] = "TRITON"
                inner.kernel_options["BLOCK_M"] = self._BLOCK_M
                inner.kernel_options["BLOCK_N"] = self._BLOCK_N
        return model_config
