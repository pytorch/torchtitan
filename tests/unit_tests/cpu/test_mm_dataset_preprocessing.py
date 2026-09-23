# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU unit tests for the multimodal dataset image preprocessing.

``resize_to_navit_patch_grid`` follows Kimi's NaViT image geometry: apply total
and per-side patch limits before padding to a ``patch_size * merge_size`` grid.
These tests pin that pure-geometry behavior.
"""

import math
import unittest
from unittest.mock import patch

import torch
from PIL import Image

from torchtitan.hf_datasets.multimodal.mm_datasets import _process_mm_sample
from torchtitan.hf_datasets.multimodal.utils.image import (
    calculate_vision_tokens,
    process_image,
    resize_to_navit_patch_grid,
    resize_to_pixel_budget,
    vision_to_patches,
)


def _navit_patch_grid_geometry(h, w, *, patch_size, merge_size, max_patches):
    """Reference geometry for the NaViT resize: the final padded (H, W)
    in pixels."""
    nh, nw = h, w
    if (nw // patch_size) * (nh // patch_size) > max_patches:
        scale = math.sqrt(max_patches / ((nw // patch_size) * (nh // patch_size)))
        nh, nw = int(nh * scale), int(nw * scale)
    factor = merge_size * patch_size
    pad_h = (factor - nh % factor) % factor
    pad_w = (factor - nw % factor) % factor
    return nh + pad_h, nw + pad_w


class TestResizeToNavitPatchGrid(unittest.TestCase):
    PS, MERGE, LIMIT, SIDE = 14, 2, 4096, 512

    def _final(self, h, w):
        rh, rw, ph, pw = resize_to_navit_patch_grid(
            h,
            w,
            patch_size=self.PS,
            merge_size=self.MERGE,
            max_patches=self.LIMIT,
            max_patches_per_side=self.SIDE,
        )
        return rh + ph, rw + pw

    def test_matches_navit_patch_grid_geometry(self):
        for h, w in [(600, 800), (336, 336), (224, 448), (1000, 500), (101, 173)]:
            got = self._final(h, w)
            want = _navit_patch_grid_geometry(
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

    def test_padding_can_exceed_pre_padding_patch_budget(self):
        # The 4096-patch budget produces a 901x901 resize before padding.
        # NaViT then pads to 924x924, which contains 66*66=4356 patches.
        fh, fw = self._final(1000, 1000)
        self.assertEqual((fh, fw), (924, 924))
        self.assertGreater((fh // self.PS) * (fw // self.PS), self.LIMIT)

    def test_small_image_not_upscaled(self):
        # below the cap -> only padded, never scaled up.
        rh, rw, ph, pw = resize_to_navit_patch_grid(
            30,
            30,
            patch_size=self.PS,
            merge_size=self.MERGE,
            max_patches=self.LIMIT,
            max_patches_per_side=self.SIDE,
        )
        self.assertEqual((rh, rw), (30, 30))

    def test_per_side_cap_scales_down(self):
        # The per-side limit scales an extreme aspect ratio instead of dropping it.
        final_h, final_w = self._final(self.PS * 600, self.PS * 2)
        self.assertEqual(final_h // self.PS, self.SIDE)
        self.assertEqual(final_w % (self.PS * self.MERGE), 0)

    def test_per_side_cap_is_inclusive(self):
        height, width = self.PS * self.SIDE, self.PS * self.MERGE
        rh, rw, ph, pw = resize_to_navit_patch_grid(
            height,
            width,
            patch_size=self.PS,
            merge_size=self.MERGE,
            max_patches=self.LIMIT,
            max_patches_per_side=self.SIDE,
        )
        self.assertEqual((rh, rw, ph, pw), (height, width, 0, 0))


class TestProcessImageNavitPatchGrid(unittest.TestCase):
    def test_navit_pads_to_factor_multiple(self):
        ps, merge = 14, 2
        factor = ps * merge
        # 100x173 is not a factor multiple -> navit must pad to one.
        img = Image.fromarray((torch.rand(100, 173, 3) * 255).to(torch.uint8).numpy())
        out = process_image(
            img,
            patch_size=ps,
            merge_size=merge,
            resize_fn=resize_to_navit_patch_grid,
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
        want_h, want_w = _navit_patch_grid_geometry(
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


class _FakeMMTokenizer:
    """Minimal tokenizer for exercising ``_process_mm_sample`` on CPU."""

    vision_start_token = "S"
    image_token = "I"
    vision_end_token = "E"
    eos_token = "X"
    vision_start_id = 1
    vision_end_id = 2
    image_id = 3
    video_id = 4

    _token_ids = {"S": 1, "I": 3, "E": 2, "X": 5}

    def encode(self, text: str) -> list[int]:
        return [self._token_ids.get(ch, 10 + (ord(ch) % 50)) for ch in text]


_IMAGE_TOKEN_GEOM = dict(
    height=64,
    width=64,
    patch_size=16,
    spatial_merge_size=2,
)


def _synthetic_rgb_image(height=64, width=64) -> Image.Image:
    return Image.fromarray((torch.rand(height, width, 3) * 255).to(torch.uint8).numpy())


def _process_synthetic_image(*, temporal_patch_size: int):
    return _process_mm_sample(
        texts=[None, "hi"],
        images=[_synthetic_rgb_image(), None],
        tokenizer=_FakeMMTokenizer(),
        patch_size=_IMAGE_TOKEN_GEOM["patch_size"],
        temporal_patch_size=temporal_patch_size,
        spatial_merge_size=_IMAGE_TOKEN_GEOM["spatial_merge_size"],
        min_pixels=1,
        max_pixels=1_000_000,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        resize_fn=resize_to_pixel_budget,
        max_patches=4096,
        max_patches_per_side=512,
    )


class TestCalculateVisionTokens(unittest.TestCase):
    def test_image_token_count_independent_of_temporal_patch_size(self):
        image_tps1 = calculate_vision_tokens(
            num_frames=1, temporal_patch_size=1, **_IMAGE_TOKEN_GEOM
        )
        image_tps2 = calculate_vision_tokens(
            num_frames=1, temporal_patch_size=2, **_IMAGE_TOKEN_GEOM
        )
        self.assertEqual(image_tps1, image_tps2)
        for tps in (1, 2, 3, 8):
            self.assertEqual(
                calculate_vision_tokens(
                    num_frames=1, temporal_patch_size=tps, **_IMAGE_TOKEN_GEOM
                ),
                image_tps1,
            )

    def test_two_frames_scales_with_temporal_patch_size(self):
        image_count, _, _ = calculate_vision_tokens(
            num_frames=1, temporal_patch_size=1, **_IMAGE_TOKEN_GEOM
        )
        two_frames_tps2, _, _ = calculate_vision_tokens(
            num_frames=2, temporal_patch_size=2, **_IMAGE_TOKEN_GEOM
        )
        two_frames_tps1, _, _ = calculate_vision_tokens(
            num_frames=2, temporal_patch_size=1, **_IMAGE_TOKEN_GEOM
        )
        self.assertEqual(two_frames_tps2, image_count)
        self.assertEqual(two_frames_tps1, 2 * image_count)


class TestProcessMmSampleTemporalPatchSize(unittest.TestCase):
    def test_image_placeholders_match_across_temporal_patch_sizes(self):
        expected_tokens, _, _ = calculate_vision_tokens(
            num_frames=1, temporal_patch_size=1, **_IMAGE_TOKEN_GEOM
        )
        counts = []
        for tps in (1, 2, 3, 8):
            sample = _process_synthetic_image(temporal_patch_size=tps)
            self.assertIsNotNone(sample)
            counts.append(int((sample["input_ids"] == _FakeMMTokenizer.image_id).sum()))
        self.assertTrue(all(count == expected_tokens for count in counts))

    def test_process_mm_sample_forwards_configured_temporal_patch_size(self):
        for tps in (2, 7):
            with patch(
                "torchtitan.hf_datasets.multimodal.mm_datasets.calculate_vision_tokens",
                wraps=calculate_vision_tokens,
            ) as mock_calc:
                sample = _process_synthetic_image(temporal_patch_size=tps)
            self.assertIsNotNone(sample)
            mock_calc.assert_called_once()
            self.assertEqual(mock_calc.call_args.kwargs["temporal_patch_size"], tps)
            self.assertEqual(mock_calc.call_args.kwargs["num_frames"], 1)


if __name__ == "__main__":
    unittest.main()
