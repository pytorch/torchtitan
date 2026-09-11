# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gemma-4 Mixture-of-Experts (MoE) module for Gemma-4 26B-A4B."""

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from torchtitan.models.common.moe import (
    GroupedExperts,
    MoE,
    RoutedExperts,
    TokenChoiceTopKRouter,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.feed_forward import FeedForward
from .model import Gemma4FeedForward


class Gemma4GroupedExperts(GroupedExperts):
    """Gemma-4 GroupedExperts with GeGLU (tanh approximation) activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        pass

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.w1_EFD, DTensor):
            w1_EFD = self.w1_EFD.to_local()
            assert isinstance(self.w2_EDF, DTensor)
            w2_EDF = self.w2_EDF.to_local()
            assert isinstance(self.w3_EFD, DTensor)
            w3_EFD = self.w3_EFD.to_local()
        else:
            w1_EFD = self.w1_EFD
            w2_EDF = self.w2_EDF
            w3_EFD = self.w3_EFD

        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)

        gate_RF = self._grouped_mm(
            A=x_RD.bfloat16(),
            weight_EOI=w1_EFD,
            offs=offsets_E,
        )
        up_RF = self._grouped_mm(
            A=x_RD.bfloat16(),
            weight_EOI=w3_EFD,
            offs=offsets_E,
        )

        h_RF = F.gelu(gate_RF, approximate="tanh") * up_RF

        return self._grouped_mm(
            A=h_RF,
            weight_EOI=w2_EDF,
            offs=offsets_E,
        ).type_as(x_RD)


class Gemma4MoE(MoE):
    """Gemma-4 MoE layer with top-k routed experts and optional shared expert."""

    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        pass
