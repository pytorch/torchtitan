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


class _PassthroughRoutedExperts(nn.Module):
    def forward(
        self,
        x_BLD,
        topk_scores_BLK,
        topk_expert_ids_BLK,
        num_local_tokens_per_expert_E,
    ):
        return x_BLD


class _FixedRouter(nn.Module):
    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x_BLD, expert_bias_E):
        B, L, _ = x_BLD.shape
        topk_scores_BLK = x_BLD.new_ones(B, L, self.top_k)
        topk_expert_ids_BLK = torch.zeros(
            B, L, self.top_k, dtype=torch.int64, device=x_BLD.device
        )
        scores_BLE = x_BLD.new_zeros(B, L, self.num_experts)
        return topk_scores_BLK, topk_expert_ids_BLK, scores_BLE


class TestMoE(unittest.TestCase):
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

        x_BLD = torch.randn(2, 3, dim)
        moe.train()
        moe(x_BLD)
        torch.testing.assert_close(
            moe.tokens_per_expert_E,
            moe.tokens_per_expert_E.new_tensor([2 * 3 * top_k, 0]),
        )
        training_counts = moe.tokens_per_expert_E.clone()

        moe.eval()
        with torch.no_grad():
            moe(x_BLD)

        torch.testing.assert_close(
            moe.tokens_per_expert_E,
            training_counts,
        )


if __name__ == "__main__":
    unittest.main()
