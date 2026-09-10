# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The pure planners of the vision encoder's dynamic context parallelism."""

import unittest
from unittest.mock import patch

import torch

from torchtitan.models.kimi_k3 import vision_encoder as ve
from torchtitan.models.kimi_k3.vit_cp_plan import (
    balance_images,
    classify,
    merged_tokens,
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

    def test_padded_key_mask_is_not_a_prefix_for_videos(self):
        # Two frames, group of 2, band 2 rows of width 2: rank 1 holds one real row.
        plan = ve.CPPatchPlan(
            group=object(),
            valid_total=12,
            full_grid=(2, 3, 2),
            row_start=0,
            band=2,
            real_rows=2,
        )
        with patch.object(ve.dist, "get_world_size", return_value=2):
            keep = ve._padded_key_keep(plan, 16, torch.device("cpu"))
        # gathered stream: rank0 [f0 rows 0-1, f1 rows 0-1], rank1 [f0 row 2, pad, f1 row 2, pad]
        expected = [True] * 8 + [True, True, False, False, True, True, False, False]
        self.assertEqual(keep.tolist(), expected)


if __name__ == "__main__":
    unittest.main()
