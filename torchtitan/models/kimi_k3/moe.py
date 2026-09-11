# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Latent MoE modules for Kimi K3."""

from dataclasses import dataclass

import torch

from torchtitan.models.common import Linear
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.nn_modules import RMSNorm

# Shape suffixes:
# T = packed tokens, D = model dimension, E = experts,
# F = expert hidden dimension, R = routed tokens, K = selected experts per token.


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
        weights_TK, expert_ids_TK, routing_map_TE = self.router(
            x_TD, self.expert_bias_E, **router_kwargs
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
