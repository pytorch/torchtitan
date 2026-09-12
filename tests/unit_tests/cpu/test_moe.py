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


class _IdentityDispatcher(nn.Module):
    """Return dispatched and combined rows without changing their order."""

    def dispatch(
        self,
        x_TD,
        topk_scores_TK,
        topk_expert_ids_TK,
        num_local_tokens_per_expert_E,
    ):
        """Return the input rows and supplied per-expert counts."""
        del topk_scores_TK, topk_expert_ids_TK
        return x_TD, num_local_tokens_per_expert_E, None

    def combine(self, routed_output_RD, metadata, x_TD):
        """Return the routed rows presented to the combine boundary."""
        del metadata, x_TD
        return routed_output_RD


class _AddOneExperts(nn.Module):
    """Apply a visible transformation at the expert-compute boundary."""

    def forward(self, x_RD, num_tokens_per_expert_E):
        """Increment every route row by one."""
        del num_tokens_per_expert_E
        return x_RD + 1


class TestMoE(unittest.TestCase):
    def test_routed_experts_postprocesses_routes_before_combine(self):
        """The optional callback transforms expert rows before combine."""
        config = make_routed_experts_config(
            dim=4,
            hidden_dim=8,
            num_experts=2,
            top_k=1,
            param_init={},
            comm_backend="standard",
        )
        routed_experts = config.build()
        routed_experts.inner_experts = _AddOneExperts()
        routed_experts.token_dispatcher = _IdentityDispatcher()
        scale = nn.Parameter(torch.tensor(3.0))
        x = torch.arange(8, dtype=torch.float32).reshape(2, 4).requires_grad_()

        output = routed_experts(
            x,
            torch.ones(2, 1),
            torch.zeros(2, 1, dtype=torch.int64),
            torch.tensor([2, 0]),
            expert_output_postprocess=lambda value: value * scale,
        )
        torch.testing.assert_close(output, (x + 1) * scale)

        output.sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 3.0))
        torch.testing.assert_close(scale.grad, (x.detach() + 1).sum())

    def test_routed_experts_expose_parameter_owner(self):
        config = make_routed_experts_config(
            dim=4,
            hidden_dim=8,
            num_experts=2,
            top_k=1,
            param_init={},
            comm_backend="standard",
        )
        routed_experts = config.build()
        original_owner = routed_experts.expert_parameters_module()
        wrapped_owner = nn.Sequential(original_owner)

        result = routed_experts.replace_expert_parameters_module(wrapped_owner)

        self.assertIs(result, routed_experts)
        self.assertIs(routed_experts.expert_parameters_module(), wrapped_owner)

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
