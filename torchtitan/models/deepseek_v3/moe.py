# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch

from torchtitan.models.common.moe import TokenChoiceTopKRouter

# Shape suffix legend:
#   T = num tokens, E = num experts, G = num expert groups,
#   P = num experts per group, Q = two experts, L = num selected groups


class DeepSeekV3Router(TokenChoiceTopKRouter):
    """DeepSeek V3 router with optional group-limited expert selection."""

    @dataclass(kw_only=True, slots=True)
    class Config(TokenChoiceTopKRouter.Config):
        num_expert_groups: int | None = None
        num_limited_groups: int | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.num_expert_groups = config.num_expert_groups
        self.num_limited_groups = config.num_limited_groups

    def _select_experts(
        self,
        scores_TE: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
        **router_kwargs,
    ) -> torch.Tensor:
        if self.num_expert_groups is None:
            return super()._select_experts(
                scores_TE,
                expert_bias_E,
                **router_kwargs,
            )
        if self.num_limited_groups is None:
            raise ValueError(
                "num_limited_groups must be set when num_expert_groups is set"
            )
        if self.num_experts % self.num_expert_groups != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by "
                f"num_expert_groups ({self.num_expert_groups})"
            )

        num_experts_per_group = self.num_experts // self.num_expert_groups
        if num_experts_per_group < 2:
            raise ValueError(
                f"num_experts_per_group ({num_experts_per_group}) must be >= 2"
            )

        scores_for_choice_TE = (
            scores_TE if expert_bias_E is None else scores_TE + expert_bias_E
        )
        scores_TGP = scores_for_choice_TE.unflatten(
            -1, (self.num_expert_groups, num_experts_per_group)
        )
        top2_scores_TGQ = scores_TGP.topk(2, dim=-1).values
        group_scores_TG = top2_scores_TGQ.sum(dim=-1)
        selected_group_ids_TL = torch.topk(
            group_scores_TG,
            k=self.num_limited_groups,
            dim=-1,
            sorted=False,
        ).indices
        unselected_groups_TG = torch.ones_like(group_scores_TG, dtype=torch.bool)
        unselected_groups_TG.scatter_(-1, selected_group_ids_TL, False)
        scores_for_choice_TE = scores_TGP.masked_fill(
            unselected_groups_TG.unsqueeze(-1), float("-inf")
        ).flatten(-2)
        return torch.topk(
            scores_for_choice_TE,
            k=self.top_k,
            dim=-1,
            sorted=False,
        ).indices
