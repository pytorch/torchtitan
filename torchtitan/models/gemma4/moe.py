# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gemma-4 Mixture-of-Experts (MoE) module for Gemma-4 26B-A4B."""

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
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
from torchtitan.models.common.nn_modules import RMSNorm
from .model import Gemma4FeedForward


class Gemma4GroupedExperts(GroupedExperts):
    """Gemma-4 GroupedExperts with GeGLU (tanh approximation) activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        moe_ffn_norm: RMSNorm.Config | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.moe_ffn_norm = config.moe_ffn_norm.build() if config.moe_ffn_norm is not None else None

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

        if self.moe_ffn_norm is not None:
            x_RD = self.moe_ffn_norm(x_RD)

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


class Gemma4TokenChoiceTopKRouter(TokenChoiceTopKRouter):
    """Gemma-4 router with unscaled RMSNorm and learned scales."""

    @dataclass(kw_only=True, slots=True)
    class Config(TokenChoiceTopKRouter.Config):
        dim: int

    def __init__(self, config: Config):
        super().__init__(config)
        self.dim = config.dim
        self.scale = nn.Parameter(torch.ones(self.dim))
        self.per_expert_scale = nn.Parameter(torch.ones(self.num_experts))

    def _select_experts(
        self,
        scores_TE: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
        **router_kwargs,
    ) -> torch.Tensor:
        # Standard top-k selection (Gemma-4 does not use expert_bias_E)
        return torch.topk(scores_TE, k=self.top_k, dim=-1, sorted=True).indices

    def forward(
        self,
        x_TD: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
        **router_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Gemma-4 applies an unscaled RMSNorm and then a learned scale
        with torch.autocast(device_type=x_TD.device.type, dtype=torch.float32):
            variance = x_TD.pow(2).mean(-1, keepdim=True)
            x_TD_normed = x_TD * torch.rsqrt(variance + 1e-6)
        
        # Apply the learned router scale and the constant scaling factor
        x_TD_normed = (x_TD_normed.type_as(x_TD) * self.scale) * (self.dim ** -0.5)

        with torch.autocast(device_type=x_TD.device.type, dtype=torch.float32):
            scores_TE = self.gate(x_TD_normed)

            if self.score_func == "softmax":
                scores_TE = F.softmax(scores_TE, dim=-1)
            else:
                raise NotImplementedError(f"Gemma-4 router only supports softmax, got {self.score_func}")

        topk_expert_ids_TK = self._select_experts(scores_TE, expert_bias_E, **router_kwargs)
        topk_scores_TK = scores_TE.gather(dim=-1, index=topk_expert_ids_TK)

        if self.route_norm:
            denominator = topk_scores_TK.sum(dim=-1, keepdim=True) + 1e-20
            topk_scores_TK = topk_scores_TK / denominator
            
        # Apply the per-expert scaling to the normalized weights
        per_expert_scale_TK = self.per_expert_scale[topk_expert_ids_TK]
        topk_scores_TK = topk_scores_TK * per_expert_scale_TK

        return topk_scores_TK, topk_expert_ids_TK, scores_TE


class Gemma4MoE(MoE):
    """Gemma-4 MoE layer with top-k routed experts and optional shared expert."""

    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        pass
