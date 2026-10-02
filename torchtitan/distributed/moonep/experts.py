# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Routed experts for MoonEP, which moves expert copies as well as tokens."""

from dataclasses import dataclass

import torch

from torchtitan.distributed.moonep.ops import routed_experts
from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh
from torchtitan.models.common.moe import RoutedExperts


class MoonEPRoutedExperts(RoutedExperts):
    """Routed experts whose grouped GEMMs also cover the expert copies MoonEP
    prefetches into this rank's slots."""

    @dataclass(kw_only=True, slots=True)
    class Config(RoutedExperts.Config):
        pass

    def _compute(
        self, x_RD: torch.Tensor, rows: dict[str, torch.Tensor], offsets: torch.Tensor
    ) -> torch.Tensor:
        gate_RF = self.w13._grouped_mm(
            input_RI=x_RD, weight_EOI=rows["gate"], offsets_E=offsets
        )
        up_RF = self.w13._grouped_mm(
            input_RI=x_RD, weight_EOI=rows["up"], offsets_E=offsets
        )
        hidden_RF = self.activation_fn(gate_RF, up_RF, offsets=offsets)
        return self.w2._grouped_mm(
            input_RI=hidden_RF, weight_EOI=rows["down"], offsets_E=offsets
        )

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        routed_input_RD, _, metadata = self.token_dispatcher.dispatch(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
        )
        with maybe_set_sparse_mesh():
            routed_output_RD = routed_experts(
                self._compute,
                routed_input_RD,
                self.w13.weight,
                self.w2.weight,
                metadata.cu_seqlens,
                metadata.plan_id,
            )
            if self.output_postprocess is not None:
                routed_output_RD = self.output_postprocess(routed_output_RD)
        return self.token_dispatcher.combine(routed_output_RD, metadata, x_TD)
