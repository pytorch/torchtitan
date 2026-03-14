# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe.moe import GroupedExperts, MoE, TokenChoiceTopKRouter
from torchtitan.protocols.module import Module


class TestGroupedExperts(unittest.TestCase):
    """Tests for GroupedExperts Config/build pattern."""

    def test_config_build(self):
        """GroupedExperts.Config.build() creates a working instance."""
        config = GroupedExperts.Config()
        experts = config.build(dim=32, hidden_dim=64, num_experts=4)
        self.assertIsInstance(experts, GroupedExperts)
        self.assertIsInstance(experts, Module)
        self.assertEqual(experts.w1.shape, torch.Size([4, 64, 32]))
        self.assertEqual(experts.w2.shape, torch.Size([4, 32, 64]))
        self.assertEqual(experts.w3.shape, torch.Size([4, 64, 32]))
        self.assertEqual(experts.num_experts, 4)
        self.assertTrue(experts.use_grouped_mm)

    def test_config_build_no_grouped_mm(self):
        """GroupedExperts.Config(use_grouped_mm=False) is respected."""
        config = GroupedExperts.Config(use_grouped_mm=False)
        experts = config.build(dim=16, hidden_dim=32, num_experts=2)
        self.assertFalse(experts.use_grouped_mm)

    def test_config_build_without_fields_raises(self):
        """build() raises when required fields are not provided."""
        config = GroupedExperts.Config()
        with self.assertRaises(TypeError):
            config.build()

    def test_config_build_partial_fields_raises(self):
        """build() raises when only some required fields are provided."""
        config = GroupedExperts.Config()
        with self.assertRaises(TypeError):
            config.build(dim=32)

    def test_init_weights(self):
        """init_weights re-initializes weight tensors."""
        config = GroupedExperts.Config()
        experts = config.build(dim=16, hidden_dim=32, num_experts=2)

        with torch.no_grad():
            torch.nn.init.zeros_(experts.w1)
            torch.nn.init.zeros_(experts.w2)
            torch.nn.init.zeros_(experts.w3)
            self.assertTrue(torch.all(experts.w1 == 0))
            experts.init_weights(init_std=0.02)
            self.assertFalse(torch.all(experts.w1 == 0))
            self.assertFalse(torch.all(experts.w2 == 0))
            self.assertFalse(torch.all(experts.w3 == 0))

    def test_init_weights_requires_init_std(self):
        """init_weights raises when init_std is not provided."""
        config = GroupedExperts.Config()
        experts = config.build(dim=16, hidden_dim=32, num_experts=2)
        with self.assertRaises(AssertionError):
            experts.init_weights()

    def test_forward_for_loop(self):
        """Forward pass works with for-loop implementation."""
        config = GroupedExperts.Config(use_grouped_mm=False)
        experts = config.build(dim=16, hidden_dim=32, num_experts=4)
        experts.init_weights(init_std=0.02)

        num_tokens_per_expert = torch.tensor([3, 2, 4, 1])
        total_tokens = num_tokens_per_expert.sum().item()
        x = torch.randn(total_tokens, 16)
        out = experts(x, num_tokens_per_expert)
        self.assertEqual(out.shape, torch.Size([total_tokens, 16]))

    def test_shared_config_builds_independent_instances(self):
        """A single Config can build multiple independent instances."""
        config = GroupedExperts.Config()
        e1 = config.build(dim=16, hidden_dim=32, num_experts=4)
        e2 = config.build(dim=32, hidden_dim=64, num_experts=8)
        self.assertIsNot(e1, e2)
        self.assertEqual(e1.w1.shape, torch.Size([4, 32, 16]))
        self.assertEqual(e2.w1.shape, torch.Size([8, 64, 32]))

    def test_default_use_grouped_mm_true(self):
        """GroupedExperts.Config defaults to use_grouped_mm=True."""
        config = GroupedExperts.Config()
        self.assertTrue(config.use_grouped_mm)


class TestTokenChoiceTopKRouter(unittest.TestCase):
    """Tests for TokenChoiceTopKRouter Config/build pattern."""

    def test_config_build(self):
        """TokenChoiceTopKRouter.Config.build() creates a working instance."""
        config = TokenChoiceTopKRouter.Config()
        router = config.build(dim=32, num_experts=8)
        self.assertIsInstance(router, TokenChoiceTopKRouter)
        self.assertIsInstance(router, Module)
        self.assertEqual(router.num_experts, 8)
        self.assertEqual(router.top_k, 1)
        self.assertEqual(router.score_func, "sigmoid")
        self.assertFalse(router.route_norm)
        self.assertEqual(router.route_scale, 1.0)

    def test_config_build_with_custom_params(self):
        """Config respects custom routing parameters."""
        config = TokenChoiceTopKRouter.Config(
            top_k=4,
            score_func="softmax",
            route_norm=True,
            route_scale=2.5,
        )
        router = config.build(dim=64, num_experts=16)
        self.assertEqual(router.top_k, 4)
        self.assertEqual(router.score_func, "softmax")
        self.assertTrue(router.route_norm)
        self.assertEqual(router.route_scale, 2.5)

    def test_config_build_with_gate_bias(self):
        """gate=Linear.Config(bias=True) creates a gate with bias."""
        config = TokenChoiceTopKRouter.Config(
            gate=Linear.Config(bias=True),
        )
        router = config.build(dim=32, num_experts=8)
        self.assertIsNotNone(router.gate.bias)
        self.assertEqual(router.gate.bias.shape, torch.Size([8]))

    def test_config_build_default_gate_no_bias(self):
        """Default gate config has no bias."""
        config = TokenChoiceTopKRouter.Config()
        router = config.build(dim=32, num_experts=8)
        self.assertIsNone(router.gate.bias)

    def test_config_build_without_fields_raises(self):
        """build() raises when required fields are not provided."""
        config = TokenChoiceTopKRouter.Config()
        with self.assertRaises(TypeError):
            config.build()

    def test_gate_shape(self):
        """Gate linear layer has correct shape (dim -> num_experts)."""
        config = TokenChoiceTopKRouter.Config()
        router = config.build(dim=64, num_experts=16)
        self.assertIsInstance(router.gate, Linear)
        self.assertEqual(router.gate.weight.shape, torch.Size([16, 64]))

    def test_node_limited_routing_config(self):
        """Config correctly passes node-limited routing parameters."""
        config = TokenChoiceTopKRouter.Config(
            num_expert_groups=4,
            num_limited_groups=2,
        )
        router = config.build(dim=32, num_experts=16)
        self.assertEqual(router.num_expert_groups, 4)
        self.assertEqual(router.num_limited_groups, 2)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_forward(self):
        """Forward pass returns correct shapes."""
        config = TokenChoiceTopKRouter.Config(top_k=2, score_func="softmax")
        router = config.build(dim=32, num_experts=8).cuda()
        router.init_weights(init_std=0.02)

        x = torch.randn(10, 32, device="cuda")
        top_scores, selected_experts, num_tokens_per_expert = router(x)
        self.assertEqual(top_scores.shape, torch.Size([10, 2]))
        self.assertEqual(selected_experts.shape, torch.Size([10, 2]))
        self.assertEqual(num_tokens_per_expert.shape, torch.Size([8]))

    def test_init_weights(self):
        """init_weights delegates to gate.init_weights."""
        config = TokenChoiceTopKRouter.Config()
        router = config.build(dim=16, num_experts=4)

        with torch.no_grad():
            torch.nn.init.zeros_(router.gate.weight)
            self.assertTrue(torch.all(router.gate.weight == 0))
            router.init_weights(init_std=0.02)
            self.assertFalse(torch.all(router.gate.weight == 0))

    def test_shared_config_builds_independent_instances(self):
        """A single Config can build multiple independent routers."""
        config = TokenChoiceTopKRouter.Config(top_k=2)
        r1 = config.build(dim=32, num_experts=8)
        r2 = config.build(dim=64, num_experts=16)
        self.assertIsNot(r1, r2)
        self.assertEqual(r1.gate.weight.shape, torch.Size([8, 32]))
        self.assertEqual(r2.gate.weight.shape, torch.Size([16, 64]))


class TestMoEConfig(unittest.TestCase):
    """Tests for MoE with nested GroupedExperts and TokenChoiceTopKRouter configs."""

    def test_config_build_defaults(self):
        """MoE.Config.build() with defaults creates a working MoE."""
        config = MoE.Config(hidden_dim=64)
        moe = config.build(dim=32)
        self.assertIsInstance(moe, MoE)
        self.assertIsInstance(moe, Module)
        self.assertIsInstance(moe.experts, GroupedExperts)
        self.assertIsInstance(moe.router, TokenChoiceTopKRouter)

    def test_config_with_nested_router(self):
        """MoE.Config with custom router config is respected."""
        config = MoE.Config(
            hidden_dim=64,
            num_experts=16,
            router=TokenChoiceTopKRouter.Config(
                top_k=4,
                score_func="softmax",
                route_norm=True,
            ),
        )
        moe = config.build(dim=32)
        self.assertEqual(moe.router.top_k, 4)
        self.assertEqual(moe.router.score_func, "softmax")
        self.assertTrue(moe.router.route_norm)
        self.assertEqual(moe.experts.num_experts, 16)

    def test_config_with_nested_experts(self):
        """MoE.Config with custom experts config is respected."""
        config = MoE.Config(
            hidden_dim=64,
            experts=GroupedExperts.Config(use_grouped_mm=False),
        )
        moe = config.build(dim=32)
        self.assertFalse(moe.experts.use_grouped_mm)

    def test_config_with_gate_bias(self):
        """MoE.Config with gate bias via nested router config."""
        config = MoE.Config(
            hidden_dim=64,
            router=TokenChoiceTopKRouter.Config(
                gate=Linear.Config(bias=True),
            ),
        )
        moe = config.build(dim=32)
        self.assertIsNotNone(moe.router.gate.bias)

    def test_num_experts_propagated(self):
        """num_experts from MoE.Config propagates to experts and router."""
        config = MoE.Config(hidden_dim=64, num_experts=16)
        moe = config.build(dim=32)
        self.assertEqual(moe.experts.num_experts, 16)
        self.assertEqual(moe.router.num_experts, 16)
        self.assertEqual(moe.experts.w1.shape[0], 16)

    def test_hidden_dim_propagated(self):
        """hidden_dim from MoE.Config propagates to experts."""
        config = MoE.Config(hidden_dim=128, num_experts=4)
        moe = config.build(dim=32)
        self.assertEqual(moe.experts.w1.shape, torch.Size([4, 128, 32]))
        self.assertEqual(moe.experts.w2.shape, torch.Size([4, 32, 128]))

    def test_init_weights(self):
        """MoE.init_weights initializes experts and router weights."""
        config = MoE.Config(hidden_dim=64, num_experts=4)
        moe = config.build(dim=32)

        with torch.no_grad():
            torch.nn.init.zeros_(moe.experts.w1)
            torch.nn.init.zeros_(moe.router.gate.weight)
            self.assertTrue(torch.all(moe.experts.w1 == 0))
            self.assertTrue(torch.all(moe.router.gate.weight == 0))

            moe.init_weights(init_std=0.02, buffer_device=torch.device("cpu"))
            self.assertFalse(torch.all(moe.experts.w1 == 0))
            self.assertFalse(torch.all(moe.router.gate.weight == 0))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_forward(self):
        """MoE forward pass produces correct output shape."""
        config = MoE.Config(
            hidden_dim=64,
            num_experts=4,
            num_shared_experts=0,
            experts=GroupedExperts.Config(use_grouped_mm=False),
        )
        moe = config.build(dim=32).cuda()
        moe.init_weights(init_std=0.02, buffer_device=torch.device("cuda"))

        x = torch.randn(2, 8, 32, device="cuda")
        out = moe(x)
        self.assertEqual(out.shape, torch.Size([2, 8, 32]))

    def test_shared_experts(self):
        """MoE with shared experts creates FeedForward."""
        config = MoE.Config(hidden_dim=64, num_shared_experts=2)
        moe = config.build(dim=32)
        self.assertIsNotNone(moe.shared_experts)

    def test_no_shared_experts(self):
        """MoE with num_shared_experts=0 has no shared experts."""
        config = MoE.Config(hidden_dim=64, num_shared_experts=0)
        moe = config.build(dim=32)
        self.assertIsNone(moe.shared_experts)


if __name__ == "__main__":
    unittest.main()
