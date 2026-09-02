# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch_remat as remat

from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module

__all__ = ["FeedForward", "SigmoidGatedFeedForward", "compute_ffn_hidden_dim"]


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
    """

    AVAILABLE_REMAT_SAVE_REGIONS: tuple[str, ...] = ("w1", "w3", "w2")

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        w1: Linear.Config
        w2: Linear.Config
        w3: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.w1 = config.w1.build()
        self.w2 = config.w2.build()
        self.w3 = config.w3.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_out = remat.region(
            self.w1,
            self.remat_region_name("w1"),
            recompute=self.should_recompute_remat_region("w1"),
        )(x)
        w3_out = remat.region(
            self.w3,
            self.remat_region_name("w3"),
            recompute=self.should_recompute_remat_region("w3"),
        )(x)
        remat.recompute_needs_tensor(w1_out, w3_out)
        out = remat.region(
            self.w2,
            self.remat_region_name("w2"),
            recompute=self.should_recompute_remat_region("w2"),
        )(F.silu(w1_out) * w3_out)
        remat.recompute_needs_tensor(out)
        return out


class SigmoidGatedFeedForward(FeedForward):
    """SwiGLU feed-forward with a per-token sigmoid gate.

    The output is ``sigmoid(gate(x)) * ffn(x)``. Inherits ``w1/w2/w3`` from
    FeedForward so weight FQNs are flat (no nested ``ffn.`` level), which keeps
    the weights directly shardable.
    """

    AVAILABLE_REMAT_SAVE_REGIONS = (
        *FeedForward.AVAILABLE_REMAT_SAVE_REGIONS,
        "gate",
    )

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        gate: Linear.Config

    def __init__(self, config: Config):
        super().__init__(config)
        self.gate = config.gate.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        gate_out = remat.region(
            self.gate,
            self.remat_region_name("gate"),
            recompute=self.should_recompute_remat_region("gate"),
        )(x)
        remat.recompute_needs_tensor(gate_out)
        return torch.sigmoid(gate_out) * out
