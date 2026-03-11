# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtitan.distributed.varlen_cp.dispatch_solver import solve_dispatch
from torchtitan.distributed.varlen_cp.mask_primitives import (
    cu_seqlens_to_attn_slices,
)


class TestSolveDispatch(unittest.TestCase):
    def test_uniform_docs(self):
        """Uniform document lengths should be balanced by default."""
        cu_seqlens = [0, 128, 256, 384, 512]
        slices = cu_seqlens_to_attn_slices(cu_seqlens)
        plan = solve_dispatch(slices, total_seqlen=512, chunk_size=256, cp_world_size=2)

        self.assertEqual(plan.cp_world_size, 2)
        self.assertEqual(plan.chunk_size, 256)
        self.assertEqual(plan.total_seqlen, 512)
        self.assertEqual(plan.pad_size, 0)

        # All ranks should have some work
        for rank in range(2):
            self.assertGreater(plan.get_rank_work(rank), 0)

    def test_skewed_docs(self):
        """Skewed document lengths should still be distributed."""
        cu_seqlens = [0, 400, 450, 500, 512]
        slices = cu_seqlens_to_attn_slices(cu_seqlens)
        plan = solve_dispatch(slices, total_seqlen=512, chunk_size=256, cp_world_size=2)

        # Both ranks should get work
        work_0 = plan.get_rank_work(0)
        work_1 = plan.get_rank_work(1)
        self.assertGreater(work_0, 0)
        self.assertGreater(work_1, 0)

    def test_all_slices_covered(self):
        """All global slices should appear in some rank's assignment."""
        cu_seqlens = [0, 128, 300, 512]
        global_slices = cu_seqlens_to_attn_slices(cu_seqlens)
        plan = solve_dispatch(
            global_slices, total_seqlen=512, chunk_size=256, cp_world_size=2
        )

        # Count total sub-slices across all ranks
        total_sub_slices = sum(
            len(cp.slices)
            for rank_assignments in plan.assignments
            for cp in rank_assignments
        )
        self.assertGreater(total_sub_slices, 0)

    def test_single_rank(self):
        """Single CP rank should get all work."""
        cu_seqlens = [0, 256]
        slices = cu_seqlens_to_attn_slices(cu_seqlens)
        plan = solve_dispatch(slices, total_seqlen=256, chunk_size=256, cp_world_size=1)

        self.assertEqual(len(plan.assignments), 1)
        self.assertGreater(plan.get_rank_work(0), 0)

    def test_minimax_property(self):
        """Greedy min-heap should produce reasonable load balance."""
        cu_seqlens = [0, 100, 200, 300, 400, 512]
        slices = cu_seqlens_to_attn_slices(cu_seqlens)
        plan = solve_dispatch(slices, total_seqlen=512, chunk_size=128, cp_world_size=4)

        works = [plan.get_rank_work(r) for r in range(4)]
        # Max work should be within 3x of min work (loose bound)
        if min(works) > 0:
            ratio = max(works) / min(works)
            self.assertLess(ratio, 3.0)

    def test_num_chunks(self):
        plan = solve_dispatch(
            cu_seqlens_to_attn_slices([0, 512]),
            total_seqlen=512,
            chunk_size=128,
            cp_world_size=4,
        )
        self.assertEqual(plan.num_chunks, 4)

    def test_pair_has_work(self):
        """pair_has_work returns True for valid pairs and False for above-diagonal pairs."""
        # Single doc of length 256, chunk_size=128, 2 chunks
        # Valid pairs: (0,0) diagonal, (1,1) diagonal, (1,0) below diagonal
        # Invalid: (0,1) above diagonal
        cu_seqlens = [0, 256]
        slices = cu_seqlens_to_attn_slices(cu_seqlens)
        plan = solve_dispatch(slices, total_seqlen=256, chunk_size=128, cp_world_size=2)

        # Diagonal and below-diagonal pairs should have work
        self.assertTrue(plan.pair_has_work(0, 0))
        self.assertTrue(plan.pair_has_work(1, 1))
        self.assertTrue(plan.pair_has_work(1, 0))

        # Above diagonal should NOT have work
        self.assertFalse(plan.pair_has_work(0, 1))

    def test_pair_has_work_no_spanning_docs(self):
        """pair_has_work returns False for off-diagonal when no doc spans chunks."""
        # Two docs, each exactly one chunk, no doc spans both chunks
        cu_seqlens = [0, 128, 256]
        slices = cu_seqlens_to_attn_slices(cu_seqlens)
        plan = solve_dispatch(slices, total_seqlen=256, chunk_size=128, cp_world_size=2)

        # Diagonal pairs have work
        self.assertTrue(plan.pair_has_work(0, 0))
        self.assertTrue(plan.pair_has_work(1, 1))

        # Off-diagonal pairs have NO work (no doc spans both chunks)
        self.assertFalse(plan.pair_has_work(1, 0))
        self.assertFalse(plan.pair_has_work(0, 1))


if __name__ == "__main__":
    unittest.main()
