# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from torch import nn

from torchtitan.models.flux.model.autoencoder import AutoEncoderParams

from torchtitan.protocols import BaseModelArgs
from torchtitan.tools.logging import logger


@dataclass
class FluxModelArgs(BaseModelArgs):
    in_channels: int = 64
    out_channels: int = 64
    vec_in_dim: int = 768
    context_in_dim: int = 512
    hidden_size: int = 3072
    mlp_ratio: float = 4.0
    num_heads: int = 24
    depth: int = 19
    depth_single_blocks: int = 38
    axes_dim: tuple = (16, 56, 56)
    theta: int = 10_000
    qkv_bias: bool = True
    autoencoder_params: AutoEncoderParams = field(default_factory=AutoEncoderParams)

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        # TODO(jianiw): Add the number of flops for the autoencoder
        nparams = sum(p.numel() for p in model.parameters())
        logger.warning("FLUX model haven't implement get_nparams_and_flops() function")
        return nparams, 1
