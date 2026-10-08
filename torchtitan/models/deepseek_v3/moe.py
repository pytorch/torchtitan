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


def _topk_ids_by_argmax(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Return the ids of the ``k`` largest entries on the last dim; ties go to the lower id.

    Compiled replacement for ``torch.topk(...).indices``: Inductor lowers ``topk``
    to an ATen fallback (a radix-select kernel), while ``k`` masked argmax rounds
    fuse with the surrounding pointwise code into one kernel.

    Example:
        >>> _topk_ids_by_argmax(torch.tensor([[0.1, 0.9, 0.5, 0.9]]), 2)
        tensor([[1, 3]])
    """
    ids = torch.arange(scores.size(-1), device=scores.device)
    topk_ids = []
    for _ in range(k):
        top_id = scores.argmax(dim=-1, keepdim=True)
        topk_ids.append(top_id)
        scores = scores.masked_fill(ids == top_id, float("-inf"))
    return torch.cat(topk_ids, dim=-1)


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
        if self.num_expert_groups is not None and self.num_limited_groups is not None:
            # Under compile the argmax rounds in _select_experts would repeat group ids
            # where eager topk raises (more limited groups than groups), or repeat
            # expert ids where eager picks masked experts (fewer than top_k experts in
            # the selected groups).
            if self.num_limited_groups > self.num_expert_groups:
                raise ValueError(
                    f"num_limited_groups ({self.num_limited_groups}) must be <= "
                    f"num_expert_groups ({self.num_expert_groups})"
                )
            num_candidates = self.num_limited_groups * (
                self.num_experts // self.num_expert_groups
            )
            if num_candidates < self.top_k:
                raise ValueError(
                    f"num_limited_groups ({self.num_limited_groups}) groups of "
                    f"{self.num_experts // self.num_expert_groups} experts hold fewer "
                    f"than top_k ({self.top_k}) experts"
                )

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
        if torch.compiler.is_compiling():
            # Same group scores; ties pick the lower id (topk's order is unspecified).
            first_id_TG1 = scores_TGP.argmax(dim=-1, keepdim=True)
            second_scores_TG = scores_TGP.masked_fill(
                torch.arange(num_experts_per_group, device=scores_TGP.device)
                == first_id_TG1,
                float("-inf"),
            ).amax(dim=-1)
            group_scores_TG = scores_TGP.amax(dim=-1) + second_scores_TG
            selected_group_ids_TL = _topk_ids_by_argmax(
                group_scores_TG, self.num_limited_groups
            )
        else:
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
        if torch.compiler.is_compiling():
            return _topk_ids_by_argmax(scores_for_choice_TE, self.top_k)
        return torch.topk(
            scores_for_choice_TE,
            k=self.top_k,
            dim=-1,
            sorted=False,
        ).indices
