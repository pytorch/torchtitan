# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the selective-gather plan builders and plan transpose.

This is pure tensor code: shapes, rank-major flattening, -1 padding, num_valid,
the out-of-range guard, and consumer/slot assignment. The transport numerics
need real collectives and live in ``tests/unit_tests/gpu/test_selective_gather.py``.
"""

import unittest

import torch
from torchtitan.distributed.context_parallel.selective_gather import (
    BlockGatherPlan,
    full_plan,
    sliding_window_plan,
)
from torchtitan.distributed.context_parallel.selective_gather.topology import (
    _consumer_slot_map,
    _validate_plans,
)

CPU = torch.device("cpu")


class _FakeGroup:
    """Minimal stand-in for a ProcessGroup: only size()/rank() are used by the
    pure plan-transpose logic, so no real distributed init is needed."""

    def __init__(self, size: int, rank: int):
        self._size, self._rank = size, rank

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank


class TestFullPlan(unittest.TestCase):
    def test_shapes_and_rank_major_order(self):
        # full_plan must reproduce the DTensor Replicate all-gather layout:
        # every block from every rank, in rank-major (concat) order.
        plan = full_plan(
            batch_size=2, cp_size=4, blocks_per_rank=3, block_numel=8, device=CPU
        )
        self.assertEqual(plan.block_numel, 8)
        self.assertEqual(plan.batch_size, 2)
        self.assertEqual(plan.capacity, 12)
        expected_rank = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        expected_block = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        for b in range(2):
            self.assertEqual(plan.src_rank[b].tolist(), expected_rank)
            self.assertEqual(plan.src_block[b].tolist(), expected_block)
        # No padding: every entry is valid.
        self.assertEqual(plan.num_valid.tolist(), [12, 12])
        self.assertTrue((plan.src_rank >= 0).all())


class TestSlidingWindowPlan(unittest.TestCase):
    def test_own_plus_previous_rank_window(self):
        plan = sliding_window_plan(
            cp_rank=2,
            cp_size=4,
            blocks_per_rank=4,
            block_numel=8,
            window_blocks=2,
            device=CPU,
        )
        # own blocks first, then the last window_blocks of rank r-1.
        self.assertEqual(plan.src_rank[0].tolist(), [2, 2, 2, 2, 1, 1])
        self.assertEqual(plan.src_block[0].tolist(), [0, 1, 2, 3, 2, 3])
        self.assertEqual(plan.num_valid.tolist(), [6])

    def test_rank0_has_no_predecessor_and_is_padded(self):
        plan = sliding_window_plan(
            cp_rank=0,
            cp_size=4,
            blocks_per_rank=4,
            block_numel=8,
            window_blocks=2,
            device=CPU,
        )
        # capacity is fixed across ranks; rank 0's window slots are -1 padding.
        self.assertEqual(plan.capacity, 6)
        self.assertEqual(plan.src_rank[0].tolist(), [0, 0, 0, 0, -1, -1])
        self.assertEqual(plan.src_block[0].tolist(), [0, 1, 2, 3, -1, -1])
        self.assertEqual(plan.num_valid.tolist(), [4])

    def test_include_own_false_keeps_only_remote(self):
        plan = sliding_window_plan(
            cp_rank=2,
            cp_size=4,
            blocks_per_rank=4,
            block_numel=8,
            window_blocks=2,
            device=CPU,
            include_own=False,
        )
        self.assertEqual(plan.capacity, 2)
        self.assertEqual(plan.src_rank[0].tolist(), [1, 1])
        self.assertEqual(plan.src_block[0].tolist(), [2, 3])
        self.assertEqual(plan.num_valid.tolist(), [2])

    def test_rejects_window_larger_than_shard(self):
        with self.assertRaises(ValueError):
            sliding_window_plan(
                cp_rank=1,
                cp_size=4,
                blocks_per_rank=4,
                block_numel=8,
                window_blocks=5,
                device=CPU,
            )

    def test_rejects_negative_window(self):
        with self.assertRaises(ValueError):
            sliding_window_plan(
                cp_rank=1,
                cp_size=4,
                blocks_per_rank=4,
                block_numel=8,
                window_blocks=-1,
                device=CPU,
            )


class TestValidatePlans(unittest.TestCase):
    """Plan validation, cp_size 2 and 2 blocks per rank.

    Rank 0 owns both its blocks; rank 1 owns both of its own plus rank 0's
    block 1, and pads the unused slot.
    """

    RANKS = [[[0, 0, -1]], [[1, 1, 0]]]
    BLOCKS = [[[0, 1, -1]], [[0, 1, 1]]]
    NUM_VALID = [[2], [3]]

    def _run(self, ranks=None, blocks=None, num_valid=None):
        _validate_plans(
            ranks or self.RANKS,
            blocks or self.BLOCKS,
            all_num_valid=num_valid or self.NUM_VALID,
            cp_size=2,
            blocks_per_rank=2,
        )

    def test_accepts_a_well_formed_plan(self):
        self._run()

    def test_rejects_duplicate_remote_block(self):
        # Rank 1 asks for rank 0's block 1 twice.
        with self.assertRaises(ValueError):
            self._run(
                blocks=[[[0, 1, -1]], [[0, 1, 1]]], ranks=[[[0, 0, -1]], [[1, 0, 0]]]
            )

    def test_rejects_duplicate_own_block(self):
        with self.assertRaises(ValueError):
            self._run(
                ranks=[[[0, 0, -1]], [[1, 1, 0]]], blocks=[[[0, 0, -1]], [[0, 1, 1]]]
            )

    def test_rejects_rank_out_of_range(self):
        with self.assertRaises(ValueError):
            self._run(ranks=[[[0, 0, -1]], [[1, 1, 2]]])

    def test_rejects_block_out_of_range(self):
        with self.assertRaises(ValueError):
            self._run(blocks=[[[0, 1, -1]], [[0, 1, 2]]])

    def test_rejects_half_padded_entry(self):
        # A -1 rank paired with a real block id is not padding. num_valid counts
        # it, so only the range check can reject this one.
        with self.assertRaises(ValueError):
            self._run(blocks=[[[0, 1, 1]], [[0, 1, 1]]], num_valid=[[3], [3]])

    def test_rejects_num_valid_mismatch(self):
        with self.assertRaises(ValueError):
            self._run(num_valid=[[3], [3]])

    def test_checks_num_valid_of_every_rank_not_just_the_caller(self):
        # Only rank 1's count is wrong. Every rank must still reject it, or the
        # good ranks walk into a collective the bad one never joins.
        with self.assertRaises(ValueError):
            self._run(num_valid=[[2], [2]])


class TestConsumerSlotMap(unittest.TestCase):
    """The plan-transpose (backward staging map) computed by _consumer_slot_map.

    Broadcast pattern, cp=3: rank 0's block is read by both ranks 1 and 2, so
    it has two consumers that must land in distinct staging slots.
    """

    def _all_src_rank(self):
        return [
            torch.tensor([[0, -1]], dtype=torch.int32),  # rank 0: own only
            torch.tensor([[1, 0]], dtype=torch.int32),  # rank 1: own + rank 0
            torch.tensor([[2, 0]], dtype=torch.int32),  # rank 2: own + rank 0
        ]

    def _run(self, my: int):
        all_src_rank = self._all_src_rank()
        src_rank = all_src_rank[my]
        plan = BlockGatherPlan(
            block_numel=8,
            src_rank=src_rank,
            src_block=torch.zeros_like(src_rank),
            num_valid=torch.tensor([2], dtype=torch.int32),
        )
        return _consumer_slot_map(all_src_rank, _FakeGroup(3, my), plan)

    def test_staging_depth_is_max_consumers(self):
        # rank 0 has two consumers -> staging depth 2 for every rank.
        for my in range(3):
            _, max_consumers, _ = self._run(my)
            self.assertEqual(max_consumers, 2)

    def test_consumer_lists(self):
        self.assertEqual(self._run(0)[2].tolist(), [1, 2])  # rank 0 read by 1,2
        self.assertEqual(self._run(1)[2].tolist(), [])
        self.assertEqual(self._run(2)[2].tolist(), [])

    def test_distinct_slots_for_shared_producer(self):
        # For rank 0's shared block (entry index 1 in each reader's plan),
        # consumer rank 1 writes slot 0 and consumer rank 2 writes slot 1.
        self.assertEqual(self._run(1)[0][0].tolist(), [0, 0])
        self.assertEqual(self._run(2)[0][0].tolist(), [0, 1])
        # A rank that consumes no remote block stays at slot 0 everywhere.
        self.assertEqual(self._run(0)[0][0].tolist(), [0, 0])


if __name__ == "__main__":
    unittest.main()
