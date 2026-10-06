# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The pure planners of the vision encoder's dynamic context parallelism."""

import unittest

from torchtitan.models.kimi_k3.vit_cp_plan import (
    balance_images,
    classify,
    key_runs,
    merged_tokens,
    plan_dynamic_cp,
    row_partition,
    subgroup_layout,
)


class TestVitCpPlan(unittest.TestCase):
    def test_row_partition_cuts_on_merge_blocks_and_keeps_every_frame(self):
        shards = row_partition(2, 12, 8, kh=2, group_size=4)
        self.assertEqual([s.row_start for s in shards], [0, 4, 8, 12])
        self.assertEqual([s.row_end for s in shards], [4, 8, 12, 12])
        for s in shards:
            self.assertEqual((s.row_end - s.row_start) % 2, 0)
            self.assertEqual(len(s.ranges), 2)
        # the last rank is empty: the deficit sits on the trailing ranks
        self.assertEqual(shards[-1].grid, (2, 0, 8))
        self.assertEqual(shards[0].ranges, ((0, 32), (96, 128)))

    def test_row_partition_refuses_an_indivisible_height(self):
        with self.assertRaises(ValueError):
            row_partition(1, 7, 8, kh=2, group_size=2)

    def test_merged_tokens_ignores_time(self):
        self.assertEqual(merged_tokens(12, 8, 2, 2), 24)

    def test_subgroup_layout_is_the_largest_divisor_below_the_image_count(self):
        self.assertEqual(subgroup_layout(1, 8), (1, 8))
        self.assertEqual(subgroup_layout(4, 8), (4, 2))
        self.assertEqual(subgroup_layout(3, 8), (2, 4))
        self.assertEqual(subgroup_layout(0, 8), (1, 8))

    def test_balance_images_is_longest_first(self):
        self.assertEqual(balance_images([100, 10, 10, 10], 2), [0, 1, 1, 1])

    def test_classify_uses_the_image_threshold(self):
        self.assertEqual(classify([300, 100, 256], 4, min_patches=256), [0, 2])
        self.assertEqual(classify([300], 1, min_patches=256), [])

    def test_plan_splits_large_images_over_balanced_subgroups(self):
        grids = [[1, 8, 8], [1, 4, 4], [1, 12, 8], [1, 7, 10]]
        plan = plan_dynamic_cp(grids, cp_size=4, kh=2, min_patches=64)
        assert plan is not None
        # 7 rows cannot take whole merge blocks, so that image stays whole.
        self.assertEqual(plan.whole, (1, 3))
        self.assertEqual(plan.split, (0, 2))
        self.assertEqual((plan.num_subgroups, plan.subgroup_size), (2, 2))
        self.assertEqual(plan.images_of(0), [2])
        self.assertEqual(plan.images_of(1), [0])
        self.assertIsNone(plan_dynamic_cp([[1, 4, 4]], cp_size=4, kh=2, min_patches=64))

    def test_key_runs_mask_each_frames_padding(self):
        # Two frames of 6 x 2 patches over two ranks: rank 0 holds rows 0-3, rank 1 rows 4-5 and two padding rows.
        runs = key_runs([[1, 2, 2], [2, 6, 2]], [0], [1], kh=2, group_size=2)
        self.assertEqual(
            runs,
            [(0, 4), (1, 8), (1, 8), (1, 4), (-1, 4), (1, 4), (-1, 4)],
        )


if __name__ == "__main__":
    unittest.main()
