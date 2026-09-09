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
    """Router that returns deterministic routing decisions for testing."""

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


def _build_moe(num_experts=4, dim=8, top_k=2):
    """Build a minimal MoE module with passthrough experts for testing."""
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
            hidden_dim=16,
            num_experts=num_experts,
            top_k=top_k,
            param_init={},
            comm_backend="standard",
        ),
    ).build()
    moe.router = _FixedRouter(num_experts, top_k)
    moe.routed_experts = _PassthroughRoutedExperts()
    return moe


class TestAnticipatoryRouting(unittest.TestCase):
    def test_cache_mode_populates_cache_and_returns_zeros(self):
        """In cache-routing mode, forward() should populate the cache dict
        with routing decisions and return a zero tensor."""
        moe = _build_moe()
        x_TD = torch.randn(6, 8)

        cache = {}
        moe.anticipatory_cache = cache

        with torch.no_grad():
            out = moe(x_TD)

        # Output should be zeros (no expert computation in cache mode)
        torch.testing.assert_close(out, torch.zeros_like(x_TD))

        # Cache should be populated with this module's routing decisions
        self.assertIn(id(moe), cache)
        topk_scores, topk_expert_ids, scores = cache[id(moe)]
        self.assertEqual(topk_scores.shape, (6, 2))  # (T, K)
        self.assertEqual(topk_expert_ids.shape, (6, 2))  # (T, K)
        self.assertEqual(scores.shape, (6, 4))  # (T, E)

        # Cache mode should still be set (cleared externally by the trainer)
        self.assertIsNotNone(moe.anticipatory_cache)

        # Clean up
        moe.anticipatory_cache = None

    def test_cache_mode_does_not_accumulate_tokens_per_expert(self):
        """Cache-routing mode returns before the tokens_per_expert_E
        accumulation, so expert counts should remain zero."""
        moe = _build_moe()
        moe.train()
        x_TD = torch.randn(6, 8)

        cache = {}
        moe.anticipatory_cache = cache

        with torch.no_grad():
            moe(x_TD)

        torch.testing.assert_close(
            moe.tokens_per_expert_E,
            torch.zeros(4, dtype=torch.float32),
        )
        moe.anticipatory_cache = None

    def test_replay_mode_uses_provided_indices(self):
        """In replay mode, forward() should use the provided indices
        instead of calling the router, and produce valid output."""
        moe = _build_moe()
        x_TD = torch.randn(6, 8)

        # Create fake routing indices
        topk_scores = torch.ones(6, 2)
        topk_expert_ids = torch.zeros(6, 2, dtype=torch.int64)
        scores = torch.zeros(6, 4)

        moe.anticipatory_indices = (topk_scores, topk_expert_ids, scores)

        out = moe(x_TD)

        # With passthrough experts, output should equal input
        torch.testing.assert_close(out, x_TD)

        # Replay mode should be cleared after use (one-shot)
        self.assertIsNone(moe.anticipatory_indices)

    def test_replay_mode_is_one_shot(self):
        """After one replay-mode forward(), the indices should be cleared
        and the next forward() should use the normal router."""
        moe = _build_moe()
        x_TD = torch.randn(6, 8)

        topk_scores = torch.ones(6, 2)
        topk_expert_ids = torch.zeros(6, 2, dtype=torch.int64)
        scores = torch.zeros(6, 4)

        moe.anticipatory_indices = (topk_scores, topk_expert_ids, scores)

        # First call: replay mode
        moe(x_TD)
        self.assertIsNone(moe.anticipatory_indices)

        # Second call: normal mode (should not error)
        out = moe(x_TD)
        self.assertEqual(out.shape, x_TD.shape)

    def test_round_trip_cache_then_replay(self):
        """Cache routing decisions with one input, then replay them with
        a different input. The routing decisions should be the same as
        what was cached."""
        moe = _build_moe()
        x1_TD = torch.randn(6, 8)
        x2_TD = torch.randn(6, 8)

        # Step 1: Cache routing decisions for x1
        cache = {}
        moe.anticipatory_cache = cache
        with torch.no_grad():
            moe(x1_TD)
        moe.anticipatory_cache = None

        cached_scores, cached_ids, cached_all_scores = cache[id(moe)]

        # Step 2: Replay cached routing with x2
        moe.anticipatory_indices = (
            cached_scores.clone(),
            cached_ids.clone(),
            cached_all_scores.clone(),
        )
        out = moe(x2_TD)

        # With passthrough experts, output should be x2 (current features)
        # but using x1's routing decisions
        torch.testing.assert_close(out, x2_TD)

    def test_normal_mode_unchanged(self):
        """When neither cache nor indices are set, forward() should work
        exactly as before (normal routing)."""
        moe = _build_moe()
        x_TD = torch.randn(6, 8)

        self.assertIsNone(moe.anticipatory_cache)
        self.assertIsNone(moe.anticipatory_indices)

        moe.train()
        out = moe(x_TD)

        # With passthrough experts, output equals input
        torch.testing.assert_close(out, x_TD)

        # tokens_per_expert_E should have been updated in normal mode
        self.assertTrue(moe.tokens_per_expert_E.sum() > 0)


if __name__ == "__main__":
    unittest.main()
