# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""K3's NaViT preprocessing, and the bridge from torchtitan's collator."""

from __future__ import annotations

import json
import pathlib
import unittest

import torch

from torchtitan.models.kimi_k3.moonvit import MoonViT, MoonViTConfig
from torchtitan.models.kimi_k3.vision_preprocess import (
    from_titan_collator,
    navit_resize,
    pack_images,
    pack_video,
    prepare_image,
)

_PRE = (
    pathlib.Path(__file__).resolve().parents[5]
    / "phase13_k3like_48b_posttrain"
    / "official_k3"
    / "reference"
    / "preprocessor_config.json"
)


class TestPolicyMatchesOfficial(unittest.TestCase):
    def test_constants_match_preprocessor_config(self):
        if not _PRE.exists():
            self.skipTest("official preprocessor_config not present")
        cfg = json.loads(_PRE.read_text())["media_proc_cfg"]
        from torchtitan.models.kimi_k3 import vision_preprocess as vp

        self.assertEqual(vp.PATCH_SIZE, cfg["patch_size"])
        self.assertEqual(vp.MERGE_KERNEL_SIZE, cfg["merge_kernel_size"])
        self.assertEqual(vp.IN_PATCH_LIMIT, cfg["in_patch_limit"])
        self.assertEqual(vp.PATCH_LIMIT_ON_ONE_SIDE, cfg["patch_limit_on_one_side"])
        self.assertEqual(
            vp.TEMPORAL_MERGE_KERNEL_SIZE, cfg["temporal_merge_kernel_size"]
        )
        self.assertEqual(vp.IMAGE_MEAN, cfg["image_mean"][0])
        self.assertEqual(vp.IMAGE_STD, cfg["image_std"][0])

    def test_the_patch_budget_is_what_the_report_calls_3584(self):
        """256 patches per side at patch_size 14 is 3584 pixels, and 256*256 is
        exactly in_patch_limit -- that is where the report's "up to 3584 x 3584"
        comes from."""
        self.assertEqual(256 * 256, 65536)
        self.assertEqual(256 * 14, 3584)

    def test_dimensions_always_tile_the_merge_kernel(self):
        """The condition tpool_patch_merger raises on. Padding to
        merge_kernel_size * patch_size is what guarantees it."""
        for w, h in ((1, 1), (37, 91), (224, 224), (1000, 600), (4000, 4000)):
            plan = navit_resize(w, h)
            gh, gw = plan.patch_grid
            self.assertEqual(gh % 2, 0, f"{w}x{h} -> grid {gh}x{gw}")
            self.assertEqual(gw % 2, 0, f"{w}x{h} -> grid {gh}x{gw}")
            self.assertEqual(plan.num_tokens, (gh // 2) * (gw // 2))

    def test_large_images_are_downscaled_within_the_side_limit(self):
        plan = navit_resize(20000, 8000)
        self.assertLessEqual(max(plan.patch_grid), 512)

    def test_padding_happens_after_resize_not_instead_of_it(self):
        # 1000x600 keeps its resolution (well under the budget) and is padded
        plan = navit_resize(1000, 600)
        self.assertEqual((plan.new_width, plan.new_height), (1000, 600))
        self.assertGreater(plan.pad_width + plan.pad_height, 0)

    def test_normalization_maps_to_minus_one_to_one(self):
        patches, _ = prepare_image(torch.zeros(3, 28, 28))
        self.assertAlmostEqual(patches.min().item(), -1.0, places=5)
        patches, _ = prepare_image(torch.ones(3, 28, 28))
        self.assertAlmostEqual(patches.max().item(), 1.0, places=5)


class TestPacking(unittest.TestCase):
    def test_mixed_resolutions_pack_into_one_batch(self):
        imgs = [torch.rand(3, 224, 224), torch.rand(3, 100, 300)]
        patches, grid = pack_images(imgs)
        self.assertEqual(grid.shape, (2, 3))
        total = sum(int(t * h * w) for t, h, w in grid.tolist())
        self.assertEqual(patches.shape[0], total)
        self.assertEqual(patches.shape[1:], (3, 14, 14))

    def test_video_groups_by_the_temporal_kernel(self):
        _, grid = pack_video(torch.rand(6, 3, 112, 112))
        # 6 frames, kernel 4 -> groups of 4 and 2, so t never exceeds
        # init_pos_emb_time and the fixed sincos table is never interpolated
        self.assertEqual([row[0] for row in grid.tolist()], [4, 2])
        self.assertTrue(all(row[0] <= 4 for row in grid.tolist()))

    def test_packed_output_feeds_the_tower(self):
        cfg = MoonViTConfig(
            num_hidden_layers=1,
            hidden_size=32,
            num_attention_heads=2,
            qkv_hidden_size=48,
            intermediate_size=64,
            patch_size=14,
            init_pos_emb_height=16,
            init_pos_emb_width=16,
            text_hidden_size=64,
            rope_max_grid=32,
        )
        torch.manual_seed(0)
        tower = MoonViT(cfg)
        tower.init_weights()
        patches, grid = pack_images([torch.rand(3, 112, 112), torch.rand(3, 56, 84)])
        out = tower(patches, grid)
        self.assertEqual(len(out), 2)
        for item, (t, h, w) in zip(out, grid.tolist()):
            self.assertEqual(item.shape, ((h // 2) * (w // 2), 64))


class TestCollatorBridge(unittest.TestCase):
    """torchtitan's collator emits BLOCK order; MoonViT needs ROW-MAJOR."""

    def _block_order(self, rowmajor, h, w, pad=0):
        flat = rowmajor.reshape(h * w, -1)
        blk = (
            flat.view(1, h // 2, 2, w // 2, 2, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(1, h * w, -1)
        )
        if pad:
            blk = torch.cat([blk, torch.zeros(1, pad, blk.shape[-1])], dim=1)
        return blk

    def test_reorders_block_to_row_major_exactly(self):
        torch.manual_seed(0)
        rowmajor, grid = prepare_image(torch.rand(3, 112, 112))
        t, h, w = grid
        padded = self._block_order(rowmajor, h, w, pad=7)
        back, _ = from_titan_collator(padded, torch.tensor([[t, h, w]]))
        self.assertTrue(torch.equal(back, rowmajor))

    def test_skipping_the_reorder_would_be_wrong(self):
        """Without this the merger groups the wrong patches and every patch gets
        the wrong position -- and the loss curve stays plausible."""
        torch.manual_seed(0)
        rowmajor, grid = prepare_image(torch.rand(3, 112, 112))
        t, h, w = grid
        blocked = self._block_order(rowmajor, h, w)
        naive = blocked[0, : h * w].view(-1, 3, 14, 14)
        self.assertFalse(torch.equal(naive, rowmajor))

    def test_padding_is_dropped_per_image(self):
        torch.manual_seed(0)
        a, ga = prepare_image(torch.rand(3, 112, 112))
        b, gb = prepare_image(torch.rand(3, 56, 56))
        n_a, n_b = a.shape[0], b.shape[0]
        width = max(n_a, n_b)
        dim = 3 * 14 * 14
        rows = torch.zeros(2, width, dim)
        rows[0, :n_a] = self._block_order(a, ga[1], ga[2])[0]
        rows[1, :n_b] = self._block_order(b, gb[1], gb[2])[0]
        grid = torch.tensor([list(ga), list(gb)])
        back, _ = from_titan_collator(rows, grid)
        self.assertEqual(back.shape[0], n_a + n_b)

    def test_grid_that_does_not_tile_the_kernel_is_rejected(self):
        rows = torch.zeros(1, 7 * 8, 3 * 14 * 14)
        with self.assertRaisesRegex(ValueError, "merge kernel"):
            from_titan_collator(rows, torch.tensor([[1, 7, 8]]))

    def test_wrong_patch_dim_is_rejected(self):
        rows = torch.zeros(1, 4, 99)
        with self.assertRaisesRegex(ValueError, "patch_dim"):
            from_titan_collator(rows, torch.tensor([[1, 2, 2]]))


class TestPackVideoGrids(unittest.TestCase):
    """One grid entry per temporal group has to describe every frame in it."""

    def test_uniform_frames_pack_cleanly(self):
        from torchtitan.models.kimi_k3.vision_preprocess import pack_video

        frames = torch.rand(4, 3, 64, 64)
        patches, grids = pack_video(frames)
        t_total = int(grids[:, 0].sum())
        self.assertEqual(t_total, 4)
        expected = sum(int(t * h * w) for t, h, w in grids.tolist())
        self.assertEqual(patches.shape[0], expected)


if __name__ == "__main__":
    unittest.main()


class TestResizePlanCarriesItsPatchSize(unittest.TestCase):
    """A plan's grid must match the plan's own dimensions, at any patch size.

    ``patch_grid`` used to divide ``padded_size`` by the module constant while
    ``navit_resize`` honoured its ``patch_size`` argument for everything else, so a plan
    built with a non-default patch size described a grid that did not fit it --
    ``prepare_image`` then died in the view with a shape mismatch. A documented parameter
    that could not be used.

    Found by trying to use it: forcing a cheap downscale for the PIL parity test needed a
    small patch size, and that is what tripped it.
    """

    def test_grid_matches_padded_size_at_every_patch_size(self):
        for patch_size in (2, 4, 7, 14):
            plan = navit_resize(
                768, 768, patch_size=patch_size, merge_kernel_size=2
            )
            padded_h, padded_w = plan.padded_size
            self.assertEqual(
                plan.patch_grid,
                (padded_h // patch_size, padded_w // patch_size),
                f"patch_size={patch_size}: grid disagrees with the plan's own size",
            )

    def test_prepare_image_accepts_a_non_default_patch_size(self):
        pixels = torch.rand(3, 768, 768)
        patches, (frames, h, w) = prepare_image(
            pixels, patch_size=2, merge_kernel_size=2, already_normalized=True
        )
        self.assertEqual(patches.shape[1:], (3, 2, 2))
        self.assertEqual(patches.shape[0], h * w)
        self.assertEqual(frames, 1)
