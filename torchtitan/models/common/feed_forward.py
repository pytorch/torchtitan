# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module
from torchtitan.protocols.state_initializer import StateInitializer

__all__ = ["FeedForward", "FeedForwardStateInitializer", "compute_ffn_hidden_dim"]


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


class FeedForwardStateInitializer(StateInitializer):
    @dataclass(kw_only=True, slots=True)
    class Config(StateInitializer.Config):
        init_std: float = 0.02

    def init_states(self, module, *, buffer_device=None) -> None:
        module.w1.init_states()
        module.w2.init_states()
        module.w3.init_states()


class FeedForward(Module):
    """SwiGLU feed-forward module shared across models.

    Config takes the **final** hidden_dim (no internal 2/3 scaling).
    Use compute_ffn_hidden_dim() for Llama3/4-style dim computation.
    Runtime ``dim`` is passed as a build() kwarg.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_dim: int
        w1: Linear.Config
        w2: Linear.Config
        w3: Linear.Config
        dim: int = field(init=False)
        state_initializer: StateInitializer.Config = field(
            default_factory=FeedForwardStateInitializer.Config
        )

    def __init__(self, config: Config):
        super().__init__(config)
        # Read init_std from our state_initializer config for output projections
        assert isinstance(config.state_initializer, FeedForwardStateInitializer.Config)
        init_std = config.state_initializer.init_std

        self.w1 = config.w1.build(
            in_features=config.dim, out_features=config.hidden_dim
        )
        w2_cfg = config.w2.replace_state_init_field(init_std=init_std)
        self.w2 = w2_cfg.build(in_features=config.hidden_dim, out_features=config.dim)
        w3_cfg = config.w3.replace_state_init_field(init_std=init_std)
        self.w3 = w3_cfg.build(in_features=config.dim, out_features=config.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
