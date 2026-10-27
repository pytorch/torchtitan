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

# T = tokens, H = heads, D = features per head.


class GatedRMSNorm(Module):
    """Apply RMSNorm followed by a unary gate activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float
        activation_fn: UnaryActivationFn.Config
        cast_before_gate: bool = False
        """Round the unweighted norm to the input dtype before weight and gate."""

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.cast_before_gate = config.cast_before_gate
        self.activation_fn = config.activation_fn.build()
        self.weight = nn.Parameter(torch.empty(config.dim))

    @local_compile("gated_rmsnorm", batch_invariant=False)
    def forward(
        self,
        input: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = input.dtype
        normalized_THD = F.rms_norm(
            input.float(),
            (input.shape[-1],),
            None if self.cast_before_gate else self.weight.float(),
            self.eps,
        )
        if self.cast_before_gate:
            normalized_THD = normalized_THD.to(input_dtype) * self.weight
        return (normalized_THD * self.activation_fn(gate.float())).to(input_dtype)


__all__ = ["GatedRMSNorm"]
