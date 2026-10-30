# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gated residual streams used by Qwen3.8-Flash-Next."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch_remat as remat
from torch import nn

from torchtitan.models.base.module import Module
from torchtitan.models.common import Linear

# T = packed tokens, S = residual streams, D = model dimension,
# C = S * D, R = mixing rank.


class GroupedOffsetRMSNorm(Module):
    """Offset RMSNorm independently normalizing each residual stream."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        group_size: int
        eps: float = 1e-6

        def __post_init__(self):
            if self.group_size <= 0 or self.dim % self.group_size:
                raise ValueError("group_size must be positive and divide dim")

    def __init__(self, config: Config):
        super().__init__()
        self.group_size = config.group_size
        self.eps = config.eps
        self.weight = nn.Parameter(torch.empty(config.dim))

    def forward(self, x_TC: torch.Tensor) -> torch.Tensor:
        x_TSD = x_TC.float().unflatten(-1, (-1, self.group_size))
        x_TSD = x_TSD * torch.rsqrt(x_TSD.square().mean(-1, keepdim=True) + self.eps)
        return (x_TSD.flatten(-2) * (1 + self.weight.float())).to(x_TC.dtype)


class GatedResidual(Module):
    """Read residual streams and optionally produce per-stream write gates."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        num_streams: int
        hc_norm: GroupedOffsetRMSNorm.Config
        input_mix_weight_down: Linear.Config
        input_mix_weight_up: Linear.Config
        block_inject_weight: Linear.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.num_streams = config.num_streams
        self.hc_norm = config.hc_norm.build()
        self.input_mix_weight_down = config.input_mix_weight_down.build()
        self.input_mix_weight_up = config.input_mix_weight_up.build()
        self.block_inject_weight = (
            config.block_inject_weight.build()
            if config.block_inject_weight is not None
            else None
        )

    def forward(
        self, x_TC: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_norm_TC = self.hc_norm(x_TC)
        remat.recompute_needs_tensor(x_norm_TC)
        mix_TR = F.silu(self.input_mix_weight_down(x_norm_TC) / self.num_streams)
        remat.recompute_needs_tensor(mix_TR)
        read_TSD = (
            self.input_mix_weight_up(mix_TR)
            .sigmoid()
            .unflatten(-1, (self.num_streams, self.dim))
        )
        mixed_TD = (
            read_TSD * x_norm_TC.unflatten(-1, (self.num_streams, self.dim))
        ).mean(-2)
        if self.block_inject_weight is None:
            return mixed_TD
        write_TS = (
            2 * (self.block_inject_weight(x_norm_TC) / self.num_streams).sigmoid()
        )
        return mixed_TD, x_TC, write_TS


def inject_residual(
    update_TD: torch.Tensor, residual_TC: torch.Tensor, write_TS: torch.Tensor
) -> torch.Tensor:
    return residual_TC + (update_TD.unsqueeze(-2) * write_TS.unsqueeze(-1)).flatten(-2)
