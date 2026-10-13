# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Latent MoE modules for Kimi K3."""

from dataclasses import dataclass

import torch
import torch_remat as remat

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
                self.router.num_experts,
                dtype=torch.float32,
            )

    def forward(
        self,
        x_TD: torch.Tensor,
        *,
        padding_mask_T: torch.Tensor | None = None,
        routed_expert_ids_TK: torch.Tensor | None = None,
        **router_kwargs,
    ) -> torch.Tensor:
        (
            routed_x_TD,
            routed_padding_mask_T,
            routed_expert_ids_TK,
        ) = self._maybe_shard_routed_branch_inputs_across_tp(
            x_TD, padding_mask_T, routed_expert_ids_TK
        )

        weights_TK, expert_ids_TK, routing_map_TE = self.router(
            routed_x_TD,
            self.expert_bias_E,
            padding_mask_T=routed_padding_mask_T,
            routed_expert_ids_TK=routed_expert_ids_TK,
            **router_kwargs,
        )
        num_tokens_per_expert_E = routing_map_TE.sum(dim=0)

        routed_down_TD = self.routed_down(routed_x_TD)
        # The token dispatcher reads the routed_down projection output with bare ops.
        remat.recompute_needs_tensor(routed_down_TD)
        routed_TD = self.routed_experts(
            routed_down_TD,
            weights_TK,
            expert_ids_TK,
            num_tokens_per_expert_E,
            padding_mask_T=routed_padding_mask_T,
        )
        # routed_norm reads the routed experts' combined output with bare ops.
        remat.recompute_needs_tensor(routed_TD)
        out_TD = self.routed_up(self.routed_norm(routed_TD))
        # The TP zero-fill and the shared-expert add read the routed_up projection
        # output with bare ops.
        remat.recompute_needs_tensor(out_TD)
        out_TD = self._maybe_zero_fill_routed_output_to_tp_partial(out_TD)
        if self.shared_experts is not None:
            shared_TD = self.shared_experts(x_TD)
            # Trailing add, always saved: it saves nothing for backward, so replay skips
            # it and its inputs need no persisting, matching checkpoint early stop.
            out_TD = remat.region(
                torch.add, self.remat_region_name("shared_add"), recompute=False
            )(out_TD, shared_TD)
        return self._maybe_all_reduce_moe_output_across_tp(out_TD)
