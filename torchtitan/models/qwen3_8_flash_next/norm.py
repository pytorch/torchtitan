# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flash-Next normalization conventions and rounding boundaries."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torchtitan.distributed.local_compile import local_compile
from torchtitan.models.common.norm import GatedRMSNorm
from torchtitan.models.qwen3_5.model import OffsetRMSNorm

# T = tokens, S = residual streams, D = features, C = S * D, H = heads.


class GroupedOffsetRMSNorm(OffsetRMSNorm):
    """Apply (1 + weight) RMSNorm independently to each residual stream."""

    @dataclass(kw_only=True, slots=True)
    class Config(OffsetRMSNorm.Config):
        group_size: int

        def __post_init__(self):
            if self.group_size <= 0 or self.dim % self.group_size:
                raise ValueError("group_size must be positive and divide dim")

    def __init__(self, config: Config):
        super().__init__(config)
        self.group_size = config.group_size

    @local_compile("offset_rmsnorm", batch_invariant=False, recompile_limit=16)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_TC = input
        x_TSD = x_TC.float().unflatten(-1, (-1, self.group_size))
        x_TSD = x_TSD * torch.rsqrt(x_TSD.square().mean(-1, keepdim=True) + self.eps)
        return (x_TSD.flatten(-2) * (1 + self.weight.float())).to(x_TC.dtype)


class InputDtypeGatedRMSNorm(GatedRMSNorm):
    """Round the unweighted RMSNorm before the learned scale and FP32 gate.

    This ordering matches the released Qwen4Exp reference in BF16.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GatedRMSNorm.Config):
        pass

    @local_compile("gated_rmsnorm", batch_invariant=False)
    def forward(self, input: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_THD, gate_THD = input, gate
        normalized_THD = F.rms_norm(
            input_THD.float(), (input_THD.shape[-1],), eps=self.eps
        )
        normalized_THD = normalized_THD.to(input_THD.dtype) * self.weight
        return (normalized_THD * self.activation_fn(gate_THD.float())).to(
            input_THD.dtype
        )
