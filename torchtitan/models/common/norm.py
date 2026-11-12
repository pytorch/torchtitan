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
from torchtitan.models.common.nn_modules import RMSNorm

# T = tokens, D = model dimension.


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


@local_compile("residual_rmsnorm", batch_invariant=False)
def _residual_rms_norm(
    residual_TD: torch.Tensor,
    delta_TD: torch.Tensor,
    weight_D: torch.Tensor | None,
    normalized_shape: tuple[int, ...],
    eps: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    h_TD = residual_TD + delta_TD
    return h_TD, F.rms_norm(h_TD, normalized_shape, weight_D, eps)


def residual_rms_norm(
    residual_TD: torch.Tensor,
    delta_TD: torch.Tensor,
    norm: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(residual + delta, norm(residual + delta))``.

    For a plain ``RMSNorm``, the add and the norm form one ``residual_rmsnorm``
    local compile region, so enabling the region fuses them into one kernel.
    With the region disabled this is the same eager math as the module. Any
    other norm (a subclass or a config override) keeps its own ``forward``.

    Example:
        x, ffn_input = residual_rms_norm(x, attn_out, self.ffn_norm)
    """
    if type(norm) is not RMSNorm:
        h_TD = residual_TD + delta_TD
        return h_TD, norm(h_TD)
    return _residual_rms_norm(
        residual_TD, delta_TD, norm.weight, norm.normalized_shape, norm.eps
    )


__all__ = ["GatedRMSNorm", "residual_rms_norm"]
