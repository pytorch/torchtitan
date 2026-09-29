# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SiTU feed-forward and latent MoE modules for Kimi K3."""

from dataclasses import dataclass

import torch
from torch.distributed.tensor import DTensor

from torchtitan.models.common import Linear
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.moe import GroupedExperts, MoE
from torchtitan.models.common.nn_modules import RMSNorm

# Shape suffixes:
# T = packed tokens, D = model dimension, E = experts,
# F = expert hidden dimension, R = routed tokens, K = selected experts per token.


def _situ_glu(
    gate: torch.Tensor,
    up: torch.Tensor,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
    """Kimi's SiTU-GLU activation, evaluated in FP32."""
    input_dtype = gate.dtype
    gate = gate.float()
    up = up.float()
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (gate * up).to(input_dtype)


class KimiFeedForward(FeedForward):
    """FeedForward with Kimi's SiTU activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        beta: float = 1.0
        linear_beta: float | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.beta = config.beta
        self.linear_beta = config.linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(
            _situ_glu(self.w1(x), self.w3(x), self.beta, self.linear_beta),
        )


class KimiGroupedExperts(GroupedExperts):
    """``common/moe.py::GroupedExperts`` with Kimi's SiTU activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        beta: float = 1.0
        linear_beta: float | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.beta = config.beta
        self.linear_beta = config.linear_beta

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

        h_RF = _situ_glu(gate_RF, up_RF, self.beta, self.linear_beta)

        return self._grouped_mm(
            A=h_RF,
            weight_EOI=w2_EDF,
            offs=offsets_E,
        ).type_as(x_RD)


class KimiLatentMoE(MoE):
    """``common/moe.py::MoE`` with Kimi's latent routed-expert path."""

    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        routed_down: Linear.Config
        routed_norm: RMSNorm.Config
        routed_up: Linear.Config

    def __init__(self, config: Config):
        super().__init__(config)
        self.routed_down = config.routed_down.build()
        self.routed_norm = config.routed_norm.build()
        self.routed_up = config.routed_up.build()

    def forward(self, x_TD: torch.Tensor, **router_kwargs) -> torch.Tensor:
        weights_TK, expert_ids_TK, scores_TE = self.router(
            x_TD, self.expert_bias_E, **router_kwargs
        )
        routing_map_TE = torch.zeros_like(scores_TE, dtype=torch.bool).scatter_(
            -1, expert_ids_TK, True
        )
        num_tokens_per_expert_E = routing_map_TE.sum(dim=0)
        if self.training:
            with torch.no_grad():
                self.tokens_per_expert_E.add_(num_tokens_per_expert_E)

        routed_TD = self.routed_experts(
            self.routed_down(x_TD),
            weights_TK,
            expert_ids_TK,
            num_tokens_per_expert_E,
        )
        out_TD = self.routed_up(self.routed_norm(routed_TD))
        if self.shared_experts is not None:
            out_TD = out_TD + self.shared_experts(x_TD)
        return out_TD
