# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from torchtitan.models.common.config_utils import (
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
)
from torchtitan.models.common.moe import TokenChoiceTopKRouter


class _PassthroughRoutedExperts(nn.Module):
    def forward(
        self,
        x_TD,
        topk_scores_TK,
        topk_expert_ids_TK,
        num_local_tokens_per_expert_E,
    ):
        return x_TD


class _FixedRouter(nn.Module):
    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x_TD, expert_bias_E):
        num_tokens = x_TD.shape[0]
        topk_scores_TK = x_TD.new_ones(num_tokens, self.top_k)
        topk_expert_ids_TK = torch.zeros(
            num_tokens, self.top_k, dtype=torch.int64, device=x_TD.device
        )
        routing_map_TE = torch.zeros(
            num_tokens, self.num_experts, dtype=torch.bool, device=x_TD.device
        ).scatter_(-1, topk_expert_ids_TK, True)
        return topk_scores_TK, topk_expert_ids_TK, routing_map_TE


class _CustomNumericsRouter(TokenChoiceTopKRouter):
    def _compute_scores(self, x_TD: torch.Tensor) -> torch.Tensor:
        return x_TD

    def _normalize_topk_scores(
        self,
        topk_scores_TK: torch.Tensor,
    ) -> torch.Tensor:
        return topk_scores_TK / topk_scores_TK.amax(dim=-1, keepdim=True)


class TestMoE(unittest.TestCase):
    def test_token_choice_router_uses_custom_numerics(self):
        x_TD = torch.tensor(
            [
                [0.1, 0.9, 0.4, 0.8],
                [0.7, 0.2, 0.6, 0.3],
            ]
        )
        route_scale = 2.5

        for route_norm in (False, True):
            with self.subTest(route_norm=route_norm):
                config = make_router_config(
                    dim=4,
                    num_experts=4,
                    gate_param_init={"weight": nn.init.zeros_},
                    top_k=2,
                    route_norm=route_norm,
                    route_scale=route_scale,
                )
                router = _CustomNumericsRouter(config)
                with torch.no_grad():
                    router.gate.weight.copy_(-torch.eye(4))

                (
                    actual_topk_scores_TK,
                    actual_topk_expert_ids_TK,
                    actual_routing_map_TE,
                ) = router(x_TD)

                (expected_topk_scores_TK, expected_topk_expert_ids_TK,) = torch.topk(
                    x_TD,
                    k=2,
                    dim=-1,
                    sorted=False,
                )
                if route_norm:
                    expected_topk_scores_TK = (
                        expected_topk_scores_TK
                        / expected_topk_scores_TK.amax(dim=-1, keepdim=True)
                    )
                expected_topk_scores_TK = expected_topk_scores_TK * route_scale
                expected_routing_map_TE = torch.zeros(
                    x_TD.shape[0],
                    router.num_experts,
                    dtype=torch.bool,
                    device=x_TD.device,
                ).scatter_(
                    dim=-1,
                    index=expected_topk_expert_ids_TK,
                    value=True,
                )

                torch.testing.assert_close(
                    actual_topk_scores_TK,
                    expected_topk_scores_TK,
                    rtol=0,
                    atol=0,
                )
                self.assertTrue(
                    torch.equal(
                        actual_topk_expert_ids_TK,
                        expected_topk_expert_ids_TK,
                    )
                )
                self.assertTrue(
                    torch.equal(actual_routing_map_TE, expected_routing_map_TE)
                )

    def test_eval_forward_does_not_accumulate_tokens_per_expert(self):
        num_experts = 2
        dim = 4
        top_k = 1
        moe = make_moe_config(
            num_experts=num_experts,
            router=make_router_config(
                dim=dim,
                num_experts=num_experts,
                gate_param_init={"weight": nn.init.zeros_},
                top_k=top_k,
            ),
            routed_experts=make_routed_experts_config(
                dim=dim,
                hidden_dim=8,
                num_experts=num_experts,
                top_k=top_k,
                param_init={},
                comm_backend="standard",
            ),
        ).build()
        moe.router = _FixedRouter(num_experts, top_k)
        moe.routed_experts = _PassthroughRoutedExperts()

        x_TD = torch.randn(6, dim)
        moe.train()
        moe(x_TD)
        torch.testing.assert_close(
            moe.tokens_per_expert_E,
            moe.tokens_per_expert_E.new_tensor([2 * 3 * top_k, 0]),
        )
        training_counts = moe.tokens_per_expert_E.clone()

        moe.eval()
        with torch.no_grad():
            moe(x_TD)

        torch.testing.assert_close(
            moe.tokens_per_expert_E,
            training_counts,
        )


if __name__ == "__main__":
    unittest.main()
