# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.protocols.module import Module

from .utils import trunc_normal_


def compute_ffn_hidden_dim(
    dim: int,
    *,
    multiple_of: int = 1,
    ffn_dim_multiplier: float | None = None,
) -> int:
    """Compute the SwiGLU hidden dimension for Llama3/4-style models.

    This applies the 2/3 scaling, optional multiplier, and rounds up to multiple_of.
    """
    hidden_dim = int(2 * 4 * dim / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


class FeedForward(Module):
    """SwiGLU feed-forward module shared across models.

    Config takes the **final** hidden_dim (no internal 2/3 scaling).
    Use compute_ffn_hidden_dim() for Llama3/4-style dim computation.
    Runtime ``dim`` is passed as a build() kwarg.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_dim: int
        bias: bool = False

    def __init__(self, config: Config, *, dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, config.hidden_dim, bias=config.bias)
        self.w2 = nn.Linear(config.hidden_dim, dim, bias=config.bias)
        self.w3 = nn.Linear(dim, config.hidden_dim, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float = 0.02, **kwargs):
        trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        if self.w1.bias is not None:
            nn.init.zeros_(self.w1.bias)
        for linear in (self.w2, self.w3):
            trunc_normal_(linear.weight, mean=0.0, std=init_std)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
