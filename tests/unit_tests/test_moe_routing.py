# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.models.common.moe.moe import TokenChoiceTopKRouter, TokenReorderer


class TestMoERoutingCounts(unittest.TestCase):
    """num_tokens_per_expert must equal the reference histc and sum to N*top_k.

    Guards against regressions of the histc -> bincount swap, which was made
    so that recomputation under selective activation checkpointing produces
    identical expert assignments on backends where histc is non-deterministic.
    """

    def _reference_counts(
        self, selected_experts_indices: torch.Tensor, num_experts: int
    ) -> torch.Tensor:
        return torch.histc(
            selected_experts_indices.view(-1).float(),
            bins=num_experts,
            min=0,
            max=num_experts,
        ).to(torch.int64)

    def test_router_counts(self):
        torch.manual_seed(0)
        dim, num_experts, top_k, n_tokens = 32, 8, 2, 64

        cfg = TokenChoiceTopKRouter.Config(top_k=top_k, score_func="softmax")
        cfg.dim = dim
        cfg.num_experts = num_experts
        router = cfg.build(dim=dim, num_experts=num_experts)

        x = torch.randn(n_tokens, dim)
        _, selected, counts = router(x, expert_bias=None)

        self.assertEqual(counts.numel(), num_experts)
        self.assertEqual(int(counts.sum().item()), n_tokens * top_k)
        torch.testing.assert_close(
            counts.to(torch.int64),
            self._reference_counts(selected, num_experts),
        )

    def test_reorderer_counts(self):
        num_experts, top_k, n_tokens = 8, 2, 64
        reorderer = TokenReorderer(num_experts=num_experts, top_k=top_k)

        torch.manual_seed(1)
        selected = torch.randint(0, num_experts, (n_tokens, top_k))
        scores = torch.randn(n_tokens, top_k)

        _, _, counts = reorderer(scores, selected)

        self.assertEqual(int(counts.sum().item()), n_tokens * top_k)
        torch.testing.assert_close(
            counts.to(torch.int64),
            self._reference_counts(selected, num_experts),
        )

    def test_recompute_consistency(self):
        """Running the router twice on the same input must produce identical counts."""
        torch.manual_seed(2)
        dim, num_experts, top_k, n_tokens = 32, 16, 4, 128

        cfg = TokenChoiceTopKRouter.Config(top_k=top_k, score_func="softmax")
        cfg.dim = dim
        cfg.num_experts = num_experts
        router = cfg.build(dim=dim, num_experts=num_experts)

        x = torch.randn(n_tokens, dim)
        _, sel1, c1 = router(x, expert_bias=None)
        _, sel2, c2 = router(x, expert_bias=None)

        torch.testing.assert_close(sel1, sel2)
        torch.testing.assert_close(c1, c2)


if __name__ == "__main__":
    unittest.main()
