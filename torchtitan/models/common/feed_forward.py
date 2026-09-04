# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import spmd_types as spmd

import torch
import torch.nn.functional as F

from torchtitan.distributed.spmd_types import sp_enabled, spmd_mesh_group
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
        tp_group = spmd_mesh_group("tp")
        if tp_group is not None:
            x = spmd.redistribute(
                x,
                tp_group,
                src=spmd.S(0) if sp_enabled() else spmd.I,
                dst=spmd.R,
                backward_options={"op_dtype": x.dtype},
            )
        out = self.w2(F.silu(self.w1(x)) * self.w3(x))
        if tp_group is not None:
            out = spmd.redistribute(
                out,
                tp_group,
                src=spmd.P,
                dst=spmd.S(0) if sp_enabled() else spmd.I,
                backward_options={"op_dtype": out.dtype},
            )
        return out


class SigmoidGatedFeedForward(FeedForward):
    """SwiGLU feed-forward with a per-token sigmoid gate.

    The output is ``sigmoid(gate(x)) * ffn(x)``. Inherits ``w1/w2/w3`` from
    FeedForward so weight FQNs are flat (no nested ``ffn.`` level), which keeps
    the weights directly shardable.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        gate: Linear.Config

    def __init__(self, config: Config):
        super().__init__(config)
        self.gate = config.gate.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.w2(F.silu(self.w1(x)) * self.w3(x))
        return torch.sigmoid(self.gate(x)) * out
