# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dynamic CP scheduling: sub-group layout, load balance, merge-aligned cuts."""

import unittest

import torch

from torchtitan.models.kimi_k3.vit_cp_plan import (
    balance_images,
    classify,
    row_partition,
    subgroup_layout,
)


class TestRowPartition(unittest.TestCase):
    def test_bands_are_multiples_of_the_merge_height(self):
        shards = row_partition(1, 8, 3, kh=2, group_size=2)
        self.assertEqual([(s.row_start, s.row_end) for s in shards], [(0, 4), (4, 8)])
        for s in shards:
            self.assertEqual(s.row_start % 2, 0)
            self.assertEqual((s.row_end - s.row_start) % 2, 0)

    def test_still_image_is_one_contiguous_range_covering_everything(self):
        shards = row_partition(1, 8, 3, kh=2, group_size=2)
        self.assertEqual([s.ranges for s in shards], [((0, 12),), ((12, 24),)])
        self.assertEqual(shards[-1].ranges[-1][1], 8 * 3)

    def test_video_keeps_every_frame_and_strides_the_band(self):
        """Time is collapsed by the projector's mean over frames, so a rank must
        hold ALL frames of its rows. Splitting by frames gives each rank the mean
        of its own frames -- t times too many tokens, measured as a 100% mismatch."""
        t, h, w, kh = 3, 4, 5, 2
        shards = row_partition(t, h, w, kh=kh, group_size=2)
        self.assertEqual([s.grid for s in shards], [(3, 2, 5), (3, 2, 5)])
        # One range per frame, each covering this rank's rows within that frame.
        self.assertEqual(len(shards[0].ranges), t)
        self.assertEqual(shards[0].ranges, ((0, 10), (20, 30), (40, 50)))
        self.assertEqual(shards[1].ranges, ((10, 20), (30, 40), (50, 60)))
        # Every patch of every frame is covered exactly once.
        covered = sorted(i for s in shards for a, b in s.ranges for i in range(a, b))
        self.assertEqual(covered, list(range(t * h * w)))

    def test_uneven_split_leaves_the_deficit_on_trailing_ranks(self):
        shards = row_partition(1, 6, 2, kh=2, group_size=2)
        bands = [s.row_end - s.row_start for s in shards]
        self.assertEqual(bands, [4, 2])
        self.assertEqual(bands, sorted(bands, reverse=True))
        shards = row_partition(1, 2, 2, kh=2, group_size=4)
        self.assertEqual([s.row_end - s.row_start for s in shards], [2, 0, 0, 0])

    def test_video_with_an_uneven_split_keeps_every_frame_on_every_rank(self):
        """t > 1 AND blocks % group_size != 0 -- the combination with no coverage.

        The two cases above each hold one variable still: the video test splits
        evenly, the uneven test uses a single frame. Their intersection is where
        the padding is interleaved PER FRAME rather than trailing, which is
        exactly what the gather-KV mask got wrong.
        """
        t, h, w, kh = 2, 6, 2, 2
        shards = row_partition(t, h, w, kh=kh, group_size=2)
        bands = [s.row_end - s.row_start for s in shards]
        # 3 merge blocks over 2 ranks: 2 blocks then 1, so 4 rows then 2.
        self.assertEqual(bands, [4, 2])
        self.assertEqual(bands, sorted(bands, reverse=True))
        # Each rank still holds all t frames of its own rows, one range each.
        for shard in shards:
            self.assertEqual(len(shard.ranges), t)
            self.assertEqual(shard.grid, (t, shard.row_end - shard.row_start, w))
        covered = sorted(i for s in shards for a, b in s.ranges for i in range(a, b))
        self.assertEqual(covered, list(range(t * h * w)))

    def test_height_not_divisible_by_the_kernel_is_refused(self):
        with self.assertRaises(ValueError):
            row_partition(1, 7, 2, kh=2, group_size=2)


class TestMergedTokens(unittest.TestCase):
    def test_time_is_collapsed_so_t_does_not_appear(self):
        from torchtitan.models.kimi_k3.vit_cp_plan import merged_tokens

        self.assertEqual(merged_tokens(8, 4, 2, 2), 8)
        # patch_count // (kh*kw) would give 16 for t=2, which is the bug this
        # helper exists to stop.
        self.assertEqual(merged_tokens(8, 4, 2, 2), (2 * 8 * 4) // 4 // 2)


class TestSubgroupLayout(unittest.TestCase):
    def test_one_large_image_uses_the_whole_group(self):
        self.assertEqual(subgroup_layout(1, 8), (1, 8))

    def test_four_large_images_on_eight_ranks_pair_up(self):
        self.assertEqual(subgroup_layout(4, 8), (4, 2))

    def test_sub_group_count_divides_the_cp_size(self):
        # 3 large images on 8 ranks: 3 does not divide 8, so 2 groups of 4.
        self.assertEqual(subgroup_layout(3, 8), (2, 4))

    def test_more_images_than_ranks_caps_at_one_rank_each(self):
        self.assertEqual(subgroup_layout(20, 8), (8, 1))


class TestBalance(unittest.TestCase):
    def test_lpt_beats_round_robin_on_a_skewed_batch(self):
        sizes = [100, 10, 10, 10]
        g = balance_images(sizes, 2)
        loads = [sum(s for s, gg in zip(sizes, g) if gg == i) for i in range(2)]
        self.assertEqual(sorted(loads), [30, 100])
        rr = [i % 2 for i in range(4)]
        rr_loads = [sum(s for s, gg in zip(sizes, rr) if gg == i) for i in range(2)]
        self.assertEqual(sorted(rr_loads), [20, 110])
        self.assertLess(max(loads), max(rr_loads))

    def test_single_group_is_a_no_op(self):
        self.assertEqual(balance_images([5, 1, 3], 1), [0, 0, 0])


class TestClassify(unittest.TestCase):
    def test_threshold_is_on_the_image_not_the_batch(self):
        self.assertEqual(classify([1000, 10, 2000], 4, min_patches=512), [0, 2])

    def test_no_cp_means_nothing_to_partition(self):
        self.assertEqual(classify([1000, 2000], 1, min_patches=512), [])


if __name__ == "__main__":
    unittest.main()


class TestStageExchange(unittest.TestCase):
    """The ViT/text PP boundary (DEP). See vit_cp_plan's section comment."""

    def test_lengths_carry_no_frame_count(self):
        from torchtitan.models.kimi_k3.vit_cp_plan import stage_exchange_lengths

        # A 4-frame video and a still with the same spatial grid send the same
        # number of tokens, because the projector's temporal mean collapses t.
        self.assertEqual(
            stage_exchange_lengths([(4, 8, 4), (1, 8, 4)], kh=2, kw=2), [8, 8]
        )

    def test_capacity_comes_from_configured_maxima_not_a_batch(self):
        from torchtitan.models.kimi_k3.vit_cp_plan import stage_exchange_capacity

        # PP sizes its P2P buffers once, so a batch-derived shape breaks on the
        # first later batch that carries more image tokens.
        self.assertEqual(stage_exchange_capacity(16, 16, 3, kh=2, kw=2), 3 * 8 * 8)

    def test_pack_then_unpack_round_trips(self):
        from torchtitan.models.kimi_k3.vit_cp_plan import (
            pack_stage_features,
            unpack_stage_features,
        )

        a, b = torch.randn(8, 5), torch.randn(4, 5)
        packed = pack_stage_features([a, b], capacity=20)
        self.assertEqual(tuple(packed.shape), (20, 5))
        got = unpack_stage_features(packed, [8, 4])
        torch.testing.assert_close(got[0], a)
        torch.testing.assert_close(got[1], b)

    def test_overflow_raises_instead_of_truncating(self):
        from torchtitan.models.kimi_k3.vit_cp_plan import pack_stage_features

        with self.assertRaises(ValueError):
            pack_stage_features([torch.randn(21, 5)], capacity=20)

    def test_unpack_refuses_a_layout_the_buffer_cannot_hold(self):
        from torchtitan.models.kimi_k3.vit_cp_plan import unpack_stage_features

        with self.assertRaises(ValueError):
            unpack_stage_features(torch.randn(10, 5), [8, 4])
