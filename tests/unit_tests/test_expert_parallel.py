# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher


class TestPermute(unittest.TestCase):
    """Test AllToAllTokenDispatcher._permute which reorders tokens from rank-major to expert-major layout.

    Input layout:  (e0,r0), (e1,r0), ..., (e0,r1), (e1,r1), ...  (rank-major)
    Output layout: (e0,r0), (e0,r1), ..., (e1,r0), (e1,r1), ...  (expert-major)
    """

    def _make_dispatcher(self) -> AllToAllTokenDispatcher:
        """Create a minimal AllToAllTokenDispatcher for testing _permute."""
        cfg = AllToAllTokenDispatcher.Config(
            num_experts=1, top_k=1, score_before_experts=True
        )
        return AllToAllTokenDispatcher(cfg)

    def _permute(self, tokens_per_expert_group, experts_per_rank, num_ranks):
        """Helper that calls _permute and returns (permuted_indices, num_tokens_per_expert)."""
        dispatcher = self._make_dispatcher()
        total = tokens_per_expert_group.sum().item()
        dummy_input = torch.zeros(total, 1)
        _, _, permuted_indices, num_tokens_per_expert = dispatcher._permute(
            dummy_input, tokens_per_expert_group, num_ranks, experts_per_rank
        )
        return permuted_indices, num_tokens_per_expert

    def test_basic_2ranks_2experts(self):
        # 2 ranks, 2 experts per rank
        # tokens_per_expert_group: [r0e0, r0e1, r1e0, r1e1] = [2, 3, 1, 4]
        tokens_per_expert_group = torch.tensor([2, 3, 1, 4])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=2, num_ranks=2
        )

        # Expert-major layout: e0r0(2), e0r1(1), e1r0(3), e1r1(4)
        # Input positions:
        #   r0e0: [0, 1], r0e1: [2, 3, 4], r1e0: [5], r1e1: [6, 7, 8, 9]
        # Output order:
        #   e0r0: [0, 1], e0r1: [5], e1r0: [2, 3, 4], e1r1: [6, 7, 8, 9]
        expected_indices = torch.tensor([0, 1, 5, 2, 3, 4, 6, 7, 8, 9])
        torch.testing.assert_close(permuted_indices, expected_indices)

        # num_tokens_per_expert: sum across ranks for each expert
        # e0: r0e0 + r1e0 = 2 + 1 = 3, e1: r0e1 + r1e1 = 3 + 4 = 7
        expected_num_tokens = torch.tensor([3, 7])
        torch.testing.assert_close(num_tokens_per_expert, expected_num_tokens)

    def test_single_rank(self):
        # 1 rank, 3 experts: no reordering needed
        tokens_per_expert_group = torch.tensor([4, 2, 5])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=3, num_ranks=1
        )

        expected_indices = torch.arange(11)
        torch.testing.assert_close(permuted_indices, expected_indices)
        torch.testing.assert_close(num_tokens_per_expert, tokens_per_expert_group)

    def test_single_expert(self):
        # 3 ranks, 1 expert per rank: no reordering needed
        tokens_per_expert_group = torch.tensor([3, 5, 2])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=1, num_ranks=3
        )

        expected_indices = torch.arange(10)
        torch.testing.assert_close(permuted_indices, expected_indices)

        # Single expert gets all tokens
        expected_num_tokens = torch.tensor([10])
        torch.testing.assert_close(num_tokens_per_expert, expected_num_tokens)

    def test_zero_tokens_for_some_experts(self):
        # 2 ranks, 2 experts, some with zero tokens
        # [r0e0, r0e1, r1e0, r1e1] = [0, 3, 2, 0]
        tokens_per_expert_group = torch.tensor([0, 3, 2, 0])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=2, num_ranks=2
        )

        # Expert-major: e0r0(0), e0r1(2), e1r0(3), e1r1(0)
        # Input positions: r0e0: [], r0e1: [0, 1, 2], r1e0: [3, 4], r1e1: []
        # Output order: e0r0: [], e0r1: [3, 4], e1r0: [0, 1, 2], e1r1: []
        expected_indices = torch.tensor([3, 4, 0, 1, 2])
        torch.testing.assert_close(permuted_indices, expected_indices)

        expected_num_tokens = torch.tensor([2, 3])
        torch.testing.assert_close(num_tokens_per_expert, expected_num_tokens)

    def test_all_zero_tokens(self):
        tokens_per_expert_group = torch.tensor([0, 0, 0, 0])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=2, num_ranks=2
        )

        self.assertEqual(permuted_indices.numel(), 0)
        expected_num_tokens = torch.tensor([0, 0])
        torch.testing.assert_close(num_tokens_per_expert, expected_num_tokens)

    def test_uniform_distribution(self):
        # 3 ranks, 2 experts, uniform token counts
        # [r0e0, r0e1, r1e0, r1e1, r2e0, r2e1] = [2, 2, 2, 2, 2, 2]
        tokens_per_expert_group = torch.tensor([2, 2, 2, 2, 2, 2])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=2, num_ranks=3
        )

        # Expert-major: e0r0(2), e0r1(2), e0r2(2), e1r0(2), e1r1(2), e1r2(2)
        # Input positions:
        #   r0e0: [0,1], r0e1: [2,3], r1e0: [4,5], r1e1: [6,7], r2e0: [8,9], r2e1: [10,11]
        # Output: e0r0[0,1], e0r1[4,5], e0r2[8,9], e1r0[2,3], e1r1[6,7], e1r2[10,11]
        expected_indices = torch.tensor([0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11])
        torch.testing.assert_close(permuted_indices, expected_indices)

        expected_num_tokens = torch.tensor([6, 6])
        torch.testing.assert_close(num_tokens_per_expert, expected_num_tokens)

    def test_permutation_is_valid(self):
        # The output should be a permutation of [0, total)
        tokens_per_expert_group = torch.tensor([3, 1, 4, 1, 5, 9])
        permuted_indices, _ = self._permute(
            tokens_per_expert_group, experts_per_rank=3, num_ranks=2
        )

        total = tokens_per_expert_group.sum().item()
        self.assertEqual(permuted_indices.numel(), total)
        self.assertEqual(
            set(permuted_indices.tolist()),
            set(range(total)),
        )


class TestAllToAllDispatcherOddTokens(unittest.TestCase):
    """Verify ``AllToAllTokenDispatcher`` handles an odd ``bs * slen`` not
    divisible by ``sp_size`` via the SP pad/unpad path (see
    ``docs/moe_sp_padding.md``).

    Drives dispatch + identity-expert + combine on a single-rank EP mesh once
    per ``sp_rank``; summing the per-rank outputs must recover the original
    tokens (the pad row at the tail is masked out by combine).
    """

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "AllToAllTokenDispatcher.dispatch uses torch.histc on the "
                "expert indices, which is not implemented for Long on CPU."
            )
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="tcp://localhost:12361",
                world_size=1,
                rank=0,
            )

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_dispatch_combine_odd_tokens(self):
        original_num_tokens = 7  # not divisible by sp_size=4
        sp_size = 4
        num_experts = 4
        top_k = 1
        dim = 3

        device = torch.device("cuda")
        ep_mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("ep",))

        torch.manual_seed(0)
        x = torch.randn(original_num_tokens, dim, device=device)
        top_scores = torch.ones(original_num_tokens, top_k, device=device)
        # Route every token to expert 0 — keeps the per-rank scatter target
        # deterministic so summing across SP ranks reconstructs x cleanly.
        selected_experts_indices = torch.zeros(
            original_num_tokens, top_k, dtype=torch.long, device=device
        )

        cfg = AllToAllTokenDispatcher.Config(
            num_experts=num_experts, top_k=top_k, score_before_experts=True
        )

        summed = torch.zeros_like(x)
        for sp_rank in range(sp_size):
            dispatcher = AllToAllTokenDispatcher(cfg)
            dispatcher.ep_mesh = ep_mesh
            dispatcher.sp_size = sp_size
            dispatcher.sp_rank = sp_rank

            routed_input, _, metadata = dispatcher.dispatch(
                x.clone(), top_scores.clone(), selected_experts_indices.clone()
            )
            # Identity expert.
            out = dispatcher.combine(routed_input, metadata, x, shared_experts=None)
            self.assertEqual(out.shape, x.shape)
            summed = summed + out

        torch.testing.assert_close(summed, x)


if __name__ == "__main__":
    unittest.main()
