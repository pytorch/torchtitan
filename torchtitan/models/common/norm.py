# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared normalization modules."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.distributed.local_compile import local_compile
from torchtitan.models.base.module import Module
from torchtitan.models.common.activation import UnaryActivationFn


class GatedRMSNorm(Module):
    """Apply RMSNorm followed by a unary gate activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float
        activation_fn: UnaryActivationFn.Config

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.activation_fn = config.activation_fn.build()
        self.weight = nn.Parameter(torch.empty(config.dim))

    @local_compile("gated_rmsnorm", batch_invariant=False)
    def forward(
        self,
        input: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = input.dtype
        normalized = F.rms_norm(
            input.float(),
            (input.shape[-1],),
            self.weight.float(),
            self.eps,
        )
        return (normalized * self.activation_fn(gate.float())).to(input_dtype)


class OffsetRMSNorm(Module):
    """RMSNorm with offset: ``(1 + weight) * norm(x)``.

    Weight is zero-initialized so the norm starts as identity-scaled.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.empty(config.dim))

    @local_compile("offset_rmsnorm", batch_invariant=False, recompile_limit=16)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Upcast to float32 for numerical stability in pow/rsqrt
        input_dtype = input.dtype
        input = input.float()
        variance = input.pow(2).mean(-1, keepdim=True)
        input = input * torch.rsqrt(variance + self.eps)
        return ((1.0 + self.weight.float()) * input).to(input_dtype)


__all__ = ["GatedRMSNorm", "OffsetRMSNorm"]
