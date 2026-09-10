# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from torchtitan.models.common.activation import ActivationFn, SiTUGLU
from torchtitan.models.common.config_utils import (
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
)
from torchtitan.models.common.moe import GroupedExperts


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
    def test_grouped_experts_use_configured_activation(self):
        activation_fn = ActivationFn.Config(
            fn=SiTUGLU(beta=4.0, linear_beta=25.0)  # pyrefly: ignore[bad-argument-type]
        )

        experts = GroupedExperts.Config(
            dim=4,
            hidden_dim=8,
            num_experts=2,
            activation_fn=activation_fn,
        ).build()
        gate_RF = torch.randn(3, 8)
        up_RF = torch.randn(3, 8)

        expected_RF = activation_fn.build()(gate_RF, up_RF)
        actual_RF = experts._activation(gate_RF, up_RF, torch.tensor([1, 3]))
        torch.testing.assert_close(actual_RF, expected_RF)

    def test_grouped_experts_use_fused_gate_up_parameter(self):
        experts = GroupedExperts.Config(
            dim=4,
            hidden_dim=8,
            num_experts=2,
        ).build()

        self.assertEqual(
            {name for name, _ in experts.named_parameters(recurse=False)},
            {"w13", "w2_EDF"},
        )
        self.assertEqual(tuple(experts.w13.shape), (2, 8, 2, 4))

    def test_grouped_experts_checkpoint_uses_logical_projection_keys(self):
        source = GroupedExperts.Config(
            dim=4,
            hidden_dim=8,
            num_experts=2,
        ).build()
        with torch.no_grad():
            source.w13.copy_(torch.randn_like(source.w13))
            source.w2_EDF.copy_(torch.randn_like(source.w2_EDF))

        state_dict = source.state_dict()
        self.assertEqual(set(state_dict), {"w1_EFD", "w2_EDF", "w3_EFD"})

        target = GroupedExperts.Config(
            dim=4,
            hidden_dim=8,
            num_experts=2,
        ).build()
        target.load_state_dict(state_dict)
        torch.testing.assert_close(target.w13, source.w13)
        torch.testing.assert_close(target.w2_EDF, source.w2_EDF)

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
