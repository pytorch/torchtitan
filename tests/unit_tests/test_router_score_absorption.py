# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import cast

import torch

from torchtitan.models.common.config_utils import make_routed_experts_config
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.models.deepseek_v3 import model_registry
from torchtitan.models.deepseek_v3.model import DeepSeekV3Model
from torchtitan.models.gpt_oss.moe import GptOssGroupedExperts
from torchtitan.models.kimi_k2_7 import model_registry as kimi_model_registry
from torchtitan.models.kimi_k2_7.model import KimiK25Model


class TestRouterScoreAbsorption(unittest.TestCase):
    def test_deepseek_v3_enables_router_score_absorption(self):
        model_config = cast(DeepSeekV3Model.Config, model_registry("debugmodel").model)
        routed_expert_configs = [
            layer.moe.routed_experts
            for layer in model_config.layers
            if layer.moe is not None
        ]

        self.assertTrue(routed_expert_configs)
        self.assertTrue(
            all(config.absorb_router_scores for config in routed_expert_configs)
        )

    def test_kimi_keeps_post_combine_router_scoring(self):
        model_config = cast(
            KimiK25Model.Config, kimi_model_registry("debugmodel").model
        )
        routed_expert_configs = [
            layer.moe.routed_experts
            for layer in model_config.layers
            if layer.moe is not None
        ]

        self.assertTrue(routed_expert_configs)
        self.assertTrue(
            all(not config.absorb_router_scores for config in routed_expert_configs)
        )

    def test_local_dispatch_returns_expert_aligned_scores(self):
        dispatcher = AllToAllTokenDispatcher(
            AllToAllTokenDispatcher.Config(num_experts=3, top_k=2)
        )
        x_TD = torch.arange(48, dtype=torch.bfloat16).reshape(3, 16)
        topk_scores_TK = torch.tensor(
            [[0.25, 0.5], [0.75, 1.0], [0.125, 0.625]], requires_grad=True
        )
        topk_expert_ids_TK = torch.tensor([[2, 0], [1, 2], [0, 1]])
        num_tokens_per_expert_E = torch.bincount(
            topk_expert_ids_TK.reshape(-1), minlength=3
        )

        routed_input_RD, _, _, routed_scores_R = dispatcher.dispatch(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_tokens_per_expert_E,
            absorb_router_scores=True,
        )

        self.assertIsNotNone(routed_scores_R)
        expert_order_N = torch.argsort(topk_expert_ids_TK.reshape(-1), stable=True)
        expected_token_indices_N = expert_order_N // 2
        torch.testing.assert_close(routed_input_RD, x_TD[expected_token_indices_N])
        torch.testing.assert_close(
            routed_scores_R, topk_scores_TK.reshape(-1)[expert_order_N]
        )

        routed_scores_R.sum().backward()
        torch.testing.assert_close(topk_scores_TK.grad, torch.ones_like(topk_scores_TK))

    def test_score_free_combine_matches_postweighted_combine(self):
        dispatcher = AllToAllTokenDispatcher(
            AllToAllTokenDispatcher.Config(num_experts=2, top_k=2)
        )
        x_TD = torch.zeros(3, 16, dtype=torch.bfloat16)
        topk_scores_TK = torch.tensor([[0.25, 0.5], [0.75, 1.0], [0.125, 0.625]])
        topk_expert_ids_TK = torch.tensor([[0, 1], [1, 0], [0, 1]])
        num_tokens_per_expert_E = torch.tensor([3, 3])

        _, _, base_metadata, _ = dispatcher.dispatch(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_tokens_per_expert_E,
        )
        _, _, absorbed_metadata, routed_scores_R = dispatcher.dispatch(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_tokens_per_expert_E,
            absorb_router_scores=True,
        )
        self.assertIsNotNone(routed_scores_R)
        routed_output_RD = torch.randn(6, 16, dtype=torch.bfloat16)
        preweighted_output_RD = (
            routed_output_RD.float() * routed_scores_R.reshape(-1, 1)
        ).to(routed_output_RD.dtype)

        expected_TD = dispatcher.combine(
            routed_output_RD,
            base_metadata,
            x_TD,
            num_local_tokens_after_padding=3,
            local_seq_len_after_padding=3,
        )
        actual_TD = dispatcher.combine(
            preweighted_output_RD,
            absorbed_metadata,
            x_TD,
            num_local_tokens_after_padding=3,
            local_seq_len_after_padding=3,
            router_scores_applied=True,
        )
        torch.testing.assert_close(actual_TD, expected_TD)

    def test_routed_experts_matches_postweighted_reference(self):
        torch.manual_seed(42)
        base = make_routed_experts_config(
            dim=16,
            hidden_dim=16,
            num_experts=2,
            top_k=2,
            param_init={},
            comm_backend="standard",
        ).build()
        absorbed = make_routed_experts_config(
            dim=16,
            hidden_dim=16,
            num_experts=2,
            top_k=2,
            param_init={},
            comm_backend="standard",
            absorb_router_scores=True,
        ).build()
        with torch.no_grad():
            for parameter in base.parameters():
                parameter.normal_(std=0.1)
            absorbed.load_state_dict(base.state_dict())

        x_base_BLD = torch.randn(1, 4, 16, dtype=torch.bfloat16, requires_grad=True)
        x_absorbed_BLD = x_base_BLD.detach().clone().requires_grad_()
        scores_base_BLK = torch.tensor(
            [[[0.25, 0.75], [0.5, 1.0], [0.125, 0.625], [0.875, 0.375]]],
            requires_grad=True,
        )
        scores_absorbed_BLK = scores_base_BLK.detach().clone().requires_grad_()
        expert_ids_BLK = torch.tensor([[[0, 1], [1, 0], [0, 1], [1, 0]]])
        num_tokens_per_expert_E = torch.tensor([4, 4])

        expected_BLD = base(
            x_base_BLD,
            scores_base_BLK,
            expert_ids_BLK,
            num_tokens_per_expert_E,
            num_local_tokens_after_seq_dim_padding=4,
        )
        actual_BLD = absorbed(
            x_absorbed_BLD,
            scores_absorbed_BLK,
            expert_ids_BLK,
            num_tokens_per_expert_E,
            num_local_tokens_after_seq_dim_padding=4,
        )

        torch.testing.assert_close(actual_BLD, expected_BLD, rtol=0.02, atol=0.02)

        expected_BLD.float().sum().backward()
        actual_BLD.float().sum().backward()
        torch.testing.assert_close(
            scores_absorbed_BLK.grad,
            scores_base_BLK.grad,
            rtol=0.02,
            atol=0.02,
        )
        torch.testing.assert_close(
            x_absorbed_BLD.grad,
            x_base_BLD.grad,
            rtol=0.02,
            atol=0.02,
        )

    def test_gpt_oss_experts_scale_projection_and_bias(self):
        torch.manual_seed(42)
        experts = GptOssGroupedExperts(
            GptOssGroupedExperts.Config(dim=16, hidden_dim=16, num_experts=2)
        )
        with torch.no_grad():
            for parameter in experts.parameters():
                parameter.normal_(std=0.1)

        x_RD = torch.randn(4, 16, dtype=torch.bfloat16)
        num_tokens_per_expert_E = torch.tensor([2, 2])
        routed_scores_R = torch.tensor([0.25, 0.5, 0.75, 1.0])

        base_RD = experts(x_RD, num_tokens_per_expert_E)
        absorbed_RD = experts(
            x_RD,
            num_tokens_per_expert_E,
            routed_scores_R=routed_scores_R,
        )
        expected_RD = (base_RD.float() * routed_scores_R.reshape(-1, 1)).bfloat16()

        torch.testing.assert_close(absorbed_RD, expected_RD, rtol=0.02, atol=0.02)

    def test_score_free_combine_does_not_save_router_scores(self):
        dispatcher = AllToAllTokenDispatcher(
            AllToAllTokenDispatcher.Config(num_experts=2, top_k=2)
        )
        x_TD = torch.zeros(2, 16, dtype=torch.bfloat16)
        topk_scores_TK = torch.tensor([[0.25, 0.75], [0.5, 1.0]])
        topk_expert_ids_TK = torch.tensor([[0, 1], [1, 0]])
        num_tokens_per_expert_E = torch.tensor([2, 2])
        _, _, metadata, _ = dispatcher.dispatch(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_tokens_per_expert_E,
        )

        def saved_combine_tensors(*, router_scores_applied: bool):
            routed_output_RD = torch.randn(
                4, 16, dtype=torch.bfloat16, requires_grad=True
            )
            saved = []
            with torch.autograd.graph.saved_tensors_hooks(
                lambda tensor: saved.append(tensor) or tensor,
                lambda tensor: tensor,
            ):
                dispatcher.combine(
                    routed_output_RD,
                    metadata,
                    x_TD,
                    num_local_tokens_after_padding=2,
                    local_seq_len_after_padding=2,
                    router_scores_applied=router_scores_applied,
                )
            return saved

        base_saved = saved_combine_tensors(router_scores_applied=False)
        absorbed_saved = saved_combine_tensors(router_scores_applied=True)

        self.assertTrue(
            any(
                tensor.shape == (4, 1) and tensor.dtype == torch.float32
                for tensor in base_saved
            )
        )
        self.assertFalse(
            any(
                tensor.shape == (4, 1) and tensor.dtype == torch.float32
                for tensor in absorbed_saved
            )
        )


if __name__ == "__main__":
    unittest.main()
