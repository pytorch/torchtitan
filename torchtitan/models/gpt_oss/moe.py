# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

from dataclasses import dataclass

import torch
import torch_remat as remat
from torch import nn

from torchtitan.models.common.linear import GroupedLinear

# Shape suffix legend: R = routed rows, E = experts, I = input features,
# O = flattened output features, including stacked projections.


class GptOssGroupedLinear(GroupedLinear):
    """Grouped linear with GPT-OSS per-expert bias."""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedLinear.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.bias = nn.Parameter(torch.empty(self.weight.shape[:-1]))

    def forward(
        self,
        input_RI: torch.Tensor,
        offsets_E: torch.Tensor,
        *,
        routed_scores_R: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_RO = super().forward(
            input_RI, offsets_E, routed_scores_R=routed_scores_R
        )
        if routed_scores_R is None:
            # The bias add below reads the grouped_mm output with bare ops.
            remat.recompute_needs_tensor(output_RO)
        bias_RO = self._expand_grouped_bias(
            self.bias.flatten(1), offsets_E, output_RO.shape[0]
        )
        if routed_scores_R is not None:
            bias_RO = bias_RO.to(output_RO.dtype) * routed_scores_R.to(
                output_RO.dtype
            ).reshape(-1, 1)
            # Saving the add avoids pinning the grouped_mm output for bias replay.
            return remat.region(
                self._add_grouped_bias,
                self.remat_region_name("bias_add"),
                recompute=self.remat_should_recompute("bias_add"),
            )(output_RO, bias_RO.reshape_as(output_RO))
        bias_RO = bias_RO.reshape_as(output_RO)
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
