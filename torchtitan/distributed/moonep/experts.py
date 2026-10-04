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
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import RoutedExperts


class MoonEPRoutedExperts(RoutedExperts):
    """Routed experts whose grouped GEMMs also cover the expert copies MoonEP
    prefetches into this rank's slots."""

    @dataclass(kw_only=True, slots=True)
    class Config(RoutedExperts.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        if type(self.w13) is not GroupedLinear or type(self.w2) is not GroupedLinear:
            raise ValueError(
                "MoonEP runs the expert GEMMs and their backward itself, so it supports "
                f"plain GroupedLinear experts only, not {type(self.w13).__qualname__}."
            )
        if self.output_postprocess is not None:
            raise ValueError(
                "MoonEP scales the expert outputs by their routing weights inside the "
                "expert op, so it does not support an output postprocess."
            )
        gate_up = (config.w13.out_features, config.w13.in_features)
        self._pool_shapes: dict[str, tuple[int, ...]] = {
            "gate": gate_up,
            "up": gate_up,
            "down": (config.w2.out_features, config.w2.in_features),
        }

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        super()._init_self_buffers(buffer_device=buffer_device)
        from torchtitan.distributed.moonep.moonep import allocate_pools, current_buffer

        _, group = current_buffer()
        allocate_pools(
            group, self.token_dispatcher.num_experts // group.size(), self._pool_shapes
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
                self.activation_fn,
                routed_input_RD,
                metadata.weights_N,
                self.w13.weight,
                self.w2.weight,
                metadata.cu_seqlens,
                metadata.plan_id,
            )
        return self.token_dispatcher.combine(routed_output_RD, metadata, x_TD)
