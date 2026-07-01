# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU unit tests for the multimodal dataset image preprocessing.

``resize_to_patch_budget`` is the NaViT patch-budget protocol: cap raw patches
at ``max_patches``, then pad to a ``patch_size * merge_size`` multiple. These
pin that pure-geometry behavior.
"""

import math
import unittest

import torch
from PIL import Image

from torchtitan.hf_datasets.multimodal.utils.image import (
    process_image,
    resize_to_patch_budget,
    vision_to_patches,
)


def _patch_budget_geometry(h, w, *, patch_size, merge_size, max_patches):
    """Reference geometry for the patch-budget resize: the final padded (H, W)
    in pixels."""
    nh, nw = h, w
    if (nw // patch_size) * (nh // patch_size) > max_patches:
        scale = math.sqrt(max_patches / ((nw // patch_size) * (nh // patch_size)))
        nh, nw = int(nh * scale), int(nw * scale)
    factor = merge_size * patch_size
    pad_h = (factor - nh % factor) % factor
    pad_w = (factor - nw % factor) % factor
    return nh + pad_h, nw + pad_w


class TestResizeToPatchBudget(unittest.TestCase):
    PS, MERGE, LIMIT, SIDE = 14, 2, 4096, 512

    def _final(self, h, w):
        rh, rw, ph, pw = resize_to_patch_budget(
            h,
            w,
            patch_size=self.PS,
            merge_size=self.MERGE,
            max_patches=self.LIMIT,
            max_patches_per_side=self.SIDE,
        )
        return rh + ph, rw + pw

    def test_matches_patch_budget_geometry(self):
        for h, w in [(600, 800), (336, 336), (224, 448), (1000, 500), (101, 173)]:
            got = self._final(h, w)
            want = _patch_budget_geometry(
                h,
                w,
                patch_size=self.PS,
                merge_size=self.MERGE,
                max_patches=self.LIMIT,
            )
            self.assertEqual(got, want, f"{h}x{w}: {got} != {want}")

    def test_output_is_factor_multiple(self):
        factor = self.PS * self.MERGE
        for h, w in [(600, 800), (101, 173), (1400, 1400)]:
            fh, fw = self._final(h, w)
            self.assertEqual(fh % factor, 0)
            self.assertEqual(fw % factor, 0)

    def test_caps_patches_at_limit(self):
        # 1400x1400 -> 100*100 = 10000 raw patches, well over the 4096 cap.
        self.assertEqual((1400 // self.PS) * (1400 // self.PS), 10000)  # sanity
        fh, fw = self._final(1400, 1400)
        # Scaled down to a square grid at the cap; this size needs no padding,
        # so the count must not exceed the limit at all.
        self.assertEqual(fh % (self.PS * self.MERGE), 0)
        self.assertLessEqual((fh // self.PS) * (fw // self.PS), self.LIMIT)

    def test_small_image_not_upscaled(self):
        # below the cap -> only padded, never scaled up.
        rh, rw, ph, pw = resize_to_patch_budget(
            30,
            30,
            patch_size=self.PS,
            merge_size=self.MERGE,
            max_patches=self.LIMIT,
            max_patches_per_side=self.SIDE,
        )
        self.assertEqual((rh, rw), (30, 30))

    def test_per_side_cap_raises(self):
        # An extreme aspect ratio that pushes one side past max_patches_per_side
        # (with a high max_patches so the patch cap doesn't scale it down).
        with self.assertRaises(ValueError):
            resize_to_patch_budget(
                14 * 600,
                28,
                patch_size=self.PS,
                merge_size=self.MERGE,
                max_patches=10**9,
                max_patches_per_side=512,
            )


class TestProcessImagePatchBudget(unittest.TestCase):
    def test_navit_pads_to_factor_multiple(self):
        ps, merge = 14, 2
        factor = ps * merge
        # 100x173 is not a factor multiple -> navit must pad to one.
        img = Image.fromarray((torch.rand(100, 173, 3) * 255).to(torch.uint8).numpy())
        out = process_image(
            img,
            patch_size=ps,
            merge_size=merge,
            resize_fn=resize_to_patch_budget,
            max_patches=4096,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
        )
        self.assertIsNotNone(out)
        # (1, H, W, C)
        _, H, W, C = out.shape
        self.assertEqual(C, 3)
        self.assertEqual(H % factor, 0)
        self.assertEqual(W % factor, 0)
        want_h, want_w = _patch_budget_geometry(
            100, 173, patch_size=ps, merge_size=merge, max_patches=4096
        )
        self.assertEqual((H, W), (want_h, want_w))


class TestVisionToPatchesOrder(unittest.TestCase):
    """Patch sequence layout: 'block' vs 'raster'."""

    def test_block_to_raster_permutation(self):
        # 2x4 patch grid (h=2, w=4), merge_size=2. Distinct per-patch values so
        # the two orderings are a pure permutation of each other.
        img = torch.arange(1 * 28 * 56 * 3, dtype=torch.float32).reshape(1, 28, 56, 3)
        block, grid = vision_to_patches(img, 14, 1, 2, patch_order="block")
        raster, _ = vision_to_patches(img, 14, 1, 2, patch_order="raster")

        self.assertEqual(grid.tolist(), [1, 2, 4])
        self.assertEqual(block.shape, raster.shape)
        # block slot b corresponds to raster slot block_to_raster_idx[b].
        block_to_raster_idx = [0, 1, 4, 5, 2, 3, 6, 7]
        for b, r in enumerate(block_to_raster_idx):
            self.assertTrue(torch.equal(block[b], raster[r]))

    def test_invalid_patch_order_raises(self):
        img = torch.zeros(1, 28, 28, 3)
        with self.assertRaises(ValueError):
            vision_to_patches(img, 14, 1, 2, patch_order="bogus")


if __name__ == "__main__":
    unittest.main()
