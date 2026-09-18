# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Latent MoE modules for Kimi K3."""

from dataclasses import dataclass

import spmd_types as spmd
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
        if config.load_balance_coeff is not None:
            raise ValueError(
                "KimiLatentMoE cannot combine sign-based and quantile balancing."
            )
        super().__init__(config)
        del self.expert_bias_E
        self.register_buffer(
            "expert_bias_E",
            torch.zeros(config.num_experts, dtype=torch.float32),
            persistent=True,
        )
        self.routed_down = config.routed_down.build()
        self.routed_norm = config.routed_norm.build()
        self.routed_up = config.routed_up.build()

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        if buffer_device is None:
            buffer_device = self.router.tokens_per_expert_E.device
        super()._init_self_buffers(buffer_device=buffer_device)
        with torch.device(buffer_device):
            self.expert_bias_E = torch.zeros(
                self.routed_experts.inner_experts.num_experts,
                dtype=torch.float32,
            )

    def forward(
        self,
        x_TD: torch.Tensor,
        *,
        padding_mask_T: torch.Tensor | None = None,
        **router_kwargs,
    ) -> torch.Tensor:
        x_TD, padding_mask_T = self._prepare_tp_inputs(x_TD, padding_mask_T)
        weights_TK, expert_ids_TK, routing_map_TE = self.router(
            x_TD,
            self.expert_bias_E,
            padding_mask_T=padding_mask_T,
            **router_kwargs,
        )
        num_tokens_per_expert_E = routing_map_TE.sum(dim=0)

        routed_TD = self.routed_experts(
            self.routed_down(x_TD),
            weights_TK,
            expert_ids_TK,
            num_tokens_per_expert_E,
        )
        out_TD = self.routed_up(self.routed_norm(routed_TD))
        shared_out_TD = self._forward_shared_experts(x_TD)
        if shared_out_TD is not None:
            out_TD = out_TD + shared_out_TD
        return self._reduce_tp_output(out_TD)

    def _combined_output_tp_type(
        self, output_tp_type: spmd.PerMeshAxisSpmdType
    ) -> spmd.PerMeshAxisSpmdType:
        # The latent routed path reduces before its nonlinear norm and returns
        # to the token-sharded layout whenever SP is enabled.
        return output_tp_type if isinstance(output_tp_type, spmd.Shard) else spmd.P
