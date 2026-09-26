# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The remote activation pool's allocator, without ranks or a transfer engine.

Parked tensors are freed in whatever order backward reaches them, not in
allocation order, so the free list has to coalesce; a bump pointer would run a
long step out of pool while most of it is free.
"""

import unittest

from torchtitan.distributed.activation_storage import _PoolAllocator

_ALIGN = 512
_CAPACITY = 1 << 20


class TestPoolAllocator(unittest.TestCase):
    def test_allocations_do_not_overlap(self):
        pool = _PoolAllocator(0, _CAPACITY)
        spans = []
        for nbytes in (1000, 2000, 4000, 8000):
            offset = pool.alloc(nbytes)
            self.assertIsNotNone(offset)
            spans.append((offset, offset + nbytes))
        spans.sort()
        for (_, end), (start, _) in zip(spans, spans[1:], strict=False):
            self.assertLessEqual(end, start, f"spans overlap: {spans}")

    def test_offsets_are_aligned(self):
        pool = _PoolAllocator(0, _CAPACITY)
        for nbytes in (1, 513, 1000):
            self.assertEqual(pool.alloc(nbytes) % _ALIGN, 0)

    def test_a_freed_span_is_reused(self):
        pool = _PoolAllocator(0, _CAPACITY)
        first = pool.alloc(1000)
        pool.alloc(2000)
        pool.free(first, 1000)
        self.assertEqual(pool.alloc(512), first)

    def test_out_of_order_frees_coalesce(self):
        """Backward frees in its own order; the pool has to come back whole."""
        pool = _PoolAllocator(0, _CAPACITY)
        spans = [(pool.alloc(nbytes), nbytes) for nbytes in (4096, 8192, 4096)]
        for offset, nbytes in reversed(spans):
            pool.free(offset, nbytes)
        # The whole pool is one span again, so the largest request fits.
        self.assertEqual(pool.alloc(_CAPACITY), 0)

    def test_a_full_pool_refuses_rather_than_overlapping(self):
        pool = _PoolAllocator(0, _CAPACITY)
        self.assertEqual(pool.alloc(_CAPACITY), 0)
        self.assertIsNone(
            pool.alloc(_ALIGN), "a full pool handed out a span it does not own"
        )

    def test_two_sources_never_share_an_offset(self):
        """Each source owns a span of the one pool, so its offsets are its own."""
        share = _CAPACITY // 2
        first = _PoolAllocator(0, share)
        second = _PoolAllocator(share, share)
        mine = set()
        theirs = set()
        while (offset := first.alloc(4096)) is not None:
            mine.add(offset)
        while (offset := second.alloc(4096)) is not None:
            theirs.add(offset)
        self.assertEqual(mine & theirs, set())
        self.assertLess(max(mine), min(theirs))
        self.assertLess(max(theirs) + 4096, _CAPACITY + 1)

    def test_a_second_free_of_one_span_raises(self):
        """A second free would hand the span out while another tensor sits in it."""
        pool = _PoolAllocator(0, _CAPACITY)
        offset = pool.alloc(4096)
        pool.free(offset, 4096)
        with self.assertRaisesRegex(RuntimeError, "freed twice"):
            pool.free(offset, 4096)


if __name__ == "__main__":
    unittest.main()
