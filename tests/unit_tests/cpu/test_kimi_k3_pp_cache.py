# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The rank cache releases a micro-batch's blocks before its gradient slots."""

import unittest

import torch

from torchtitan.models.kimi_k3.pipeline_adapter import RankLocalCache


class TestRankCacheRelease(unittest.TestCase):
    def test_release_frees_blocks_and_keeps_the_slots(self):
        cache = RankLocalCache()
        cache.append(0, torch.zeros(4, 2), (0, 0, 0))
        cache.append(0, torch.ones(4, 2), (1, 1, 0))
        cache.append(1, torch.ones(4, 2), (0, 0, 0))
        cache.capture_grad((0, 0, 0), torch.full((4, 2), 2.0))

        cache.release_blocks(0)

        # The rank's forwards for micro-batch 0 are done: its blocks are gone,
        # micro-batch 1 is untouched, and the slot the producer's backward will
        # pop is still there.
        self.assertEqual(cache.get_blocks(0), [])
        self.assertEqual(cache.get_meta(0), [])
        self.assertEqual(len(cache.get_blocks(1)), 1)
        self.assertTrue(cache.has_captured_for_mb(0))
        grad, count = cache.pop_grad((0, 0, 0))
        self.assertEqual(count, 1)
        self.assertTrue(torch.equal(grad, torch.full((4, 2), 2.0)))

    def test_drop_still_sweeps_everything(self):
        cache = RankLocalCache()
        cache.append(0, torch.zeros(4, 2), (0, 0, 0))
        cache.capture_grad((0, 0, 0), torch.ones(4, 2))
        cache.drop(0)
        self.assertEqual(cache.get_blocks(0), [])
        self.assertFalse(cache.has_captured_for_mb(0))


if __name__ == "__main__":
    unittest.main()
