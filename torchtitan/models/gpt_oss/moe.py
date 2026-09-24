# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from torchtitan.models.common.activation import BinaryActivationFn
from torchtitan.models.common.linear import GroupedLinear


class GptOssSwiGLU(BinaryActivationFn):
    """GPT-OSS clamped SwiGLU activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(BinaryActivationFn.Config):
        swiglu_limit: float = 7.0

    def __init__(self, config: Config):
        self.swiglu_limit = config.swiglu_limit

    def __call__(
        self,
        gate_RF: torch.Tensor,
        up_RF: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        gate_RF = gate_RF.clamp(max=self.swiglu_limit)
        up_RF = up_RF.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        silu_RF = gate_RF * torch.sigmoid(1.702 * gate_RF)
        return torch.addcmul(silu_RF, silu_RF, up_RF)


class GptOssGroupedLinear(GroupedLinear):
    """Grouped linear with GPT-OSS per-expert bias."""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedLinear.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.bias = nn.Parameter(torch.empty(self.weight.shape[:-1]))

    def forward(self, input_RI: torch.Tensor, offsets_E: torch.Tensor) -> torch.Tensor:
        output_RO = super().forward(input_RI, offsets_E)
        bias_RO = self._expand_grouped_bias(
            self.bias.flatten(1), offsets_E, output_RO.shape[0]
        ).reshape_as(output_RO)
        return self._add_grouped_bias(output_RO, bias_RO)

    @staticmethod
    def _expand_grouped_bias(
        bias_EO: torch.Tensor,
        offsets_E: torch.Tensor,
        output_rows: int,
    ) -> torch.Tensor:
        """Expand expert bias across routed rows and zero-valued tail padding."""
        counts_E = torch.diff(torch.cat((offsets_E.new_zeros(1), offsets_E)))
        tail_count = (output_rows - offsets_E[-1]).unsqueeze(0).to(counts_E.dtype)
        padded_bias = torch.cat((bias_EO, bias_EO.new_zeros(1, bias_EO.shape[-1])))
        return padded_bias.repeat_interleave(
            torch.cat((counts_E, tail_count)).long(),
            dim=0,
            output_size=output_rows,
        )

    def _add_grouped_bias(
        self,
        output_RO: torch.Tensor,
        bias_RO: torch.Tensor,
    ) -> torch.Tensor:
        return output_RO + bias_RO.to(output_RO.dtype)
