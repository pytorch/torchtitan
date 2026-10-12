# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unbiased sequence balancing used by the DeepSeek-V3 MLPerf recipes."""

from dataclasses import dataclass

import torch

from torchtitan.models.common.moe import MicrobatchWiseLoadBalanceLoss

__all__ = ["UnbiasedTopKLoadBalanceLoss"]

# T = tokens, E = experts, K = experts selected per token.


def _unbiased_routing_map(scores_TE, top_k, *, padding_mask_T=None):
    """Select plain top-k after NVIDIA's unbiased positive-score normalization."""
    normalized_TE = scores_TE / (scores_TE.sum(-1, keepdim=True) + 1e-20)
    expert_ids_TK = normalized_TE.topk(top_k, dim=-1, sorted=False).indices
    routing_map_TE = torch.zeros_like(scores_TE, dtype=torch.bool).scatter_(
        -1, expert_ids_TK, True
    )
    return (
        routing_map_TE
        if padding_mask_T is None
        else routing_map_TE & ~padding_mask_T.unsqueeze(-1)
    )


class UnbiasedTopKLoadBalanceLoss(MicrobatchWiseLoadBalanceLoss):
    """Compute sequence balancing from scores, not dispatch bias or forced routes."""

    @dataclass(kw_only=True, slots=True)
    class Config(MicrobatchWiseLoadBalanceLoss.Config):
        top_k: int
        """Number of unbiased expert choices counted per valid token."""

        def __post_init__(self):
            if self.top_k <= 0:
                raise ValueError("top_k must be positive")

    def __init__(self, config: Config):
        super().__init__(config)
        self.top_k = config.top_k

    def _compute_local_stats(self, scores_TE, routing_map_TE, padding_mask_T):
        """Use NVIDIA's positive-score denominator, including tiny sigmoid rows."""
        counts_E = routing_map_TE.to(scores_TE.dtype).sum(0)
        normalized_TE = scores_TE / (scores_TE.sum(-1, keepdim=True) + 1e-20)
        if padding_mask_T is not None:
            normalized_TE = normalized_TE * ~padding_mask_T.unsqueeze(-1)
        return counts_E, normalized_TE.sum(0)

    def forward(
        self, scores_TE, routing_map_TE, *, carrier, padding_mask_T=None, denominator
    ):
        """Reuse upstream normalization/injection with the independent top-k map."""
        del routing_map_TE
        unbiased_map_TE = _unbiased_routing_map(
            scores_TE, self.top_k, padding_mask_T=padding_mask_T
        )
        return super().forward(
            scores_TE,
            unbiased_map_TE,
            carrier=carrier,
            padding_mask_T=padding_mask_T,
            denominator=denominator,
        )
