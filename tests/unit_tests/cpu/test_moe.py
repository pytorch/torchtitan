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
from torchtitan.models.common.linear import GroupedLinear


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
        scores_TE = x_TD.new_zeros(num_tokens, self.num_experts)
        return topk_scores_TK, topk_expert_ids_TK, scores_TE


class TestMoE(unittest.TestCase):
    def test_grouped_linear_structured_output(self):
        grouped = GroupedLinear.Config(
            num_groups=2,
            in_features=8,
            out_features=(2, 8),
        ).build()
        with torch.no_grad():
            identity = torch.eye(8)
            grouped.weight[0].copy_(torch.stack((identity, identity)))
            grouped.weight[1].copy_(torch.stack((2 * identity, 2 * identity)))

        input = torch.arange(24, dtype=torch.bfloat16).reshape(3, 8)
        output = grouped(
            input,
            torch.tensor([2, 3], dtype=torch.int32),
        )

        self.assertEqual(output.shape, (3, 2, 8))
        torch.testing.assert_close(
            output,
            torch.stack((input, input), dim=1)
            * input.new_tensor([1, 1, 2]).reshape(-1, 1, 1),
        )

    def test_routed_experts_own_structured_linears(self):
        init = {
            "gate": nn.init.zeros_,
            "up": nn.init.ones_,
            "down": nn.init.zeros_,
        }
        config = make_routed_experts_config(
            dim=4,
            hidden_dim=8,
            num_experts=2,
            top_k=1,
            param_init=init,
            comm_backend="standard",
        )
        routed_experts = config.build()

        self.assertEqual(routed_experts.w13.weight.shape, (2, 2, 8, 4))
        self.assertEqual(routed_experts.w2.weight.shape, (2, 4, 8))
        self.assertEqual(set(routed_experts.state_dict()), {"w13.weight", "w2.weight"})

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
                param_init={
                    "gate": nn.init.zeros_,
                    "up": nn.init.ones_,
                    "down": nn.init.zeros_,
                },
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
