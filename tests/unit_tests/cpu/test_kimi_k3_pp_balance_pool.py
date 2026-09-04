# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The remote pool's allocator, without ranks or a transfer engine.

Parked tensors are freed in whatever order backward reaches them, not in
allocation order, so the free list has to coalesce; a bump pointer would run a
long step out of pool while most of it is free.
"""

import unittest

from torchtitan.models.kimi_k3.pp_balance import _PoolAllocator

_ALIGN = 512
_CAPACITY = 1 << 20


class TestPoolAllocator(unittest.TestCase):
    def test_allocations_do_not_overlap(self):
        pool = _PoolAllocator(_CAPACITY)
        spans = []
        for nbytes in (1000, 2000, 4000, 8000):
            offset = pool.alloc(nbytes)
            self.assertIsNotNone(offset)
            spans.append((offset, offset + nbytes))
        spans.sort()
        for (_, end), (start, _) in zip(spans, spans[1:], strict=False):
            self.assertLessEqual(end, start, f"spans overlap: {spans}")

    def test_offsets_are_aligned(self):
        pool = _PoolAllocator(_CAPACITY)
        for nbytes in (1, 513, 1000):
            self.assertEqual(pool.alloc(nbytes) % _ALIGN, 0)

    def test_a_freed_span_is_reused(self):
        pool = _PoolAllocator(_CAPACITY)
        first = pool.alloc(1000)
        pool.alloc(2000)
        pool.free(first, 1000)
        self.assertEqual(pool.alloc(512), first)

    def test_out_of_order_frees_coalesce(self):
        """Backward frees in its own order; the pool has to come back whole."""
        pool = _PoolAllocator(_CAPACITY)
        spans = [(pool.alloc(nbytes), nbytes) for nbytes in (4096, 8192, 4096)]
        for offset, nbytes in reversed(spans):
            pool.free(offset, nbytes)
        # The whole pool is one span again, so the largest request fits.
        self.assertEqual(pool.alloc(_CAPACITY), 0)

    def test_a_full_pool_refuses_rather_than_overlapping(self):
        pool = _PoolAllocator(_CAPACITY)
        self.assertEqual(pool.alloc(_CAPACITY), 0)
        self.assertIsNone(
            pool.alloc(_ALIGN), "a full pool handed out a span it does not own"
        )


if __name__ == "__main__":
    unittest.main()
