# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU unit tests for the multimodal dataset image preprocessing.

``resize_to_navit_patch_grid`` follows Kimi's NaViT image geometry: apply total
and per-side patch limits before padding to a ``patch_size * merge_size`` grid.
These tests pin that pure-geometry behavior, and the ``ImageProcessor`` /
``VideoProcessor`` stages built on the resize rules.
"""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2.functional as TVF
from PIL import Image
from torchtitan.components.data.multimodal.image_processor import ImageProcessor
from torchtitan.components.data.multimodal.mm_datasets import _process_mm_sample
from torchtitan.components.data.multimodal.utils.image import (
    calculate_vision_tokens,
    vision_to_patches,
)
from torchtitan.components.data.multimodal.utils.resize import (
    resize_to_navit_patch_grid,
    resize_to_pixel_budget,
)
from torchtitan.components.data.multimodal.utils.video import load_npy_video_frames
from torchtitan.components.data.multimodal.video_processor import VideoProcessor

from torchtitan.components.loss import IGNORE_INDEX


def _image_processor_config(**overrides) -> ImageProcessor.Config:
    fields = dict(
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        resize_fn=resize_to_pixel_budget,
        patch_order="block",
    )
    fields.update(overrides)
    return ImageProcessor.Config(**fields)


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


class TestImageProcessorNavitPatchGrid(unittest.TestCase):
    def test_navit_pads_to_factor_multiple(self):
        ps, merge = 14, 2
        factor = ps * merge
        # 100x173 is not a factor multiple -> navit must pad to one.
        img = Image.fromarray((torch.rand(100, 173, 3) * 255).to(torch.uint8).numpy())
        processor = _image_processor_config(
            patch_size=ps,
            temporal_patch_size=1,
            spatial_merge_size=merge,
            resize_fn=resize_to_navit_patch_grid,
            patch_order="raster",
            max_patches=4096,
        ).build()
        out = processor.decode_and_resize(img)
        # (1, H, W, C) uint8
        _, H, W, C = out.shape
        self.assertEqual(out.dtype, torch.uint8)
        self.assertEqual(C, 3)
        self.assertEqual(H % factor, 0)
        self.assertEqual(W % factor, 0)
        want_h, want_w = _navit_patch_grid_geometry(
            100, 173, patch_size=ps, merge_size=merge, max_patches=4096
        )
        self.assertEqual((H, W), (want_h, want_w))


class TestImageProcessorInterpolationMode(unittest.TestCase):
    def test_lanczos_matches_torchvision_tensor_resize(self):
        image_CHW = torch.arange(3 * 5 * 7, dtype=torch.uint8).reshape(3, 5, 7)
        pil_image = Image.fromarray(image_CHW.permute(1, 2, 0).numpy())
        expected_CHW = TVF.resize(
            image_CHW,
            [3, 4],
            interpolation=TVF.InterpolationMode.LANCZOS,
            antialias=True,
        )

        def fixed_geometry(*args, **kwargs):
            return 3, 4, 0, 0

        processor = _image_processor_config(
            resize_fn=fixed_geometry,
            image_interpolation_mode=TVF.InterpolationMode.LANCZOS,
        ).build()
        actual = processor.decode_and_resize(pil_image)

        self.assertTrue(torch.equal(actual, expected_CHW.permute(1, 2, 0).unsqueeze(0)))


class TestImageProcessorStages(unittest.TestCase):
    def test_two_stages_match_single_pass_preprocessing(self):
        # decode_and_resize + normalize_and_patchify must equal decode -> resize ->
        # normalize -> patchify in one pass, bitwise: the split only moves where
        # steps run.
        image_HWC = (torch.rand(100, 173, 3) * 255).to(torch.uint8)
        pil_image = Image.fromarray(image_HWC.numpy())
        for patch_order in ("block", "raster"):
            config = _image_processor_config(patch_order=patch_order)
            processor = config.build()

            pixel_values, grid_thw = processor.normalize_and_patchify(
                [processor.decode_and_resize(pil_image)]
            )

            resize_h, resize_w, _, _ = resize_to_pixel_budget(
                100,
                173,
                patch_size=16,
                merge_size=2,
                min_pixels=config.min_pixels,
                max_pixels=config.max_pixels,
            )
            expected_CHW = TVF.resize(
                image_HWC.permute(2, 0, 1),
                [resize_h, resize_w],
                interpolation=TVF.InterpolationMode.BICUBIC,
                antialias=True,
            )
            expected_CHW = TVF.to_dtype(expected_CHW, torch.float32, scale=True)
            expected_CHW = TVF.normalize(expected_CHW, [0.5] * 3, [0.5] * 3)
            expected, expected_grid = vision_to_patches(
                expected_CHW.permute(1, 2, 0).unsqueeze(0),
                16,
                2,
                2,
                patch_order=patch_order,
            )
            self.assertTrue(torch.equal(pixel_values, expected))
            self.assertEqual(grid_thw.tolist(), [expected_grid.tolist()])

    def test_preprocess_concatenates_items(self):
        processor = _image_processor_config(min_pixels=1).build()
        small = torch.zeros(1, 32, 64, 3, dtype=torch.uint8)
        large = torch.zeros(1, 64, 64, 3, dtype=torch.uint8)

        pixel_values, grid_thw = processor.normalize_and_patchify([small, large])

        self.assertEqual(grid_thw.tolist(), [[1, 2, 4], [1, 4, 4]])
        self.assertEqual(pixel_values.shape, (8 + 16, 3 * 2 * 16 * 16))

    def test_num_placeholder_tokens(self):
        processor = _image_processor_config().build()
        # (64 / 32) * (96 / 32) merged tokens for one frame.
        self.assertEqual(processor.num_placeholder_tokens(1, 64, 96), 6)

    def test_resized_size_matches_decode_and_resize(self):
        # A renderer counts placeholder tokens from the size in the file header with
        # resized_size; it must be the size decode_and_resize produces, for a
        # pixel-budget resize and for a NaViT resize that pads.
        kimi_style = dict(
            patch_size=14,
            temporal_patch_size=1,
            resize_fn=resize_to_navit_patch_grid,
            patch_order="raster",
            max_patches=256,
            max_patches_per_side=16,
        )
        for overrides in (dict(min_pixels=32 * 32, max_pixels=256 * 256), kimi_style):
            processor = _image_processor_config(**overrides).build()
            for height, width in ((200, 300), (48, 64), (900, 40)):
                image = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
                _, resized_h, resized_w, _ = processor.decode_and_resize(image).shape
                self.assertEqual(
                    processor.resized_size(height, width), (resized_h, resized_w)
                )

    def test_normalize_and_patchify_rejects_float_items(self):
        # In-place normalization would otherwise overwrite the caller's tensor.
        processor = _image_processor_config().build()
        with self.assertRaisesRegex(ValueError, "uint8"):
            processor.normalize_and_patchify([torch.zeros(1, 32, 32, 3)])


class TestLoadNpyVideoFrames(unittest.TestCase):
    def test_loads_c_contiguous_uint8_thwc_array(self):
        frames_THWC = np.arange(2 * 3 * 4 * 3, dtype=np.uint8).reshape(2, 3, 4, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frames.npy"
            np.save(path, frames_THWC)

            actual_THWC = load_npy_video_frames(path)

        self.assertTrue(actual_THWC.is_contiguous())
        self.assertTrue(torch.equal(actual_THWC, torch.from_numpy(frames_THWC)))


class TestVideoProcessorResize(unittest.TestCase):
    def test_custom_resize_padding_and_interpolation(self):
        video_TCHW = torch.arange(1 * 3 * 2 * 2, dtype=torch.uint8).reshape(1, 3, 2, 2)
        video_THWC = video_TCHW.permute(0, 2, 3, 1)
        resize_calls = []

        def custom_resize(height, width, **kwargs):
            resize_calls.append((height, width, kwargs))
            return 2, 3, 1, 2

        expected_TCHW = TVF.resize(
            video_TCHW,
            [2, 3],
            interpolation=TVF.InterpolationMode.NEAREST,
            antialias=True,
        )
        expected_TCHW = TVF.pad(expected_TCHW, [0, 0, 2, 1])

        processor = VideoProcessor.Config(
            patch_size=4,
            temporal_patch_size=2,
            spatial_merge_size=2,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            resize_fn=custom_resize,
            patch_order="block",
            max_pixels=200,
            min_pixels=20,
            image_interpolation_mode=TVF.InterpolationMode.NEAREST,
            max_patches=12,
            max_patches_per_side=5,
        ).build()
        actual_THWC = processor.resize_video(video_THWC)

        self.assertEqual(
            resize_calls,
            [
                (
                    2,
                    2,
                    {
                        "patch_size": 4,
                        "merge_size": 2,
                        "min_pixels": 20,
                        "max_pixels": 200,
                        "max_patches": 12,
                        "max_patches_per_side": 5,
                    },
                )
            ],
        )
        self.assertEqual(actual_THWC.shape, (1, 3, 5, 3))
        self.assertTrue(torch.equal(actual_THWC, expected_TCHW.permute(0, 2, 3, 1)))


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

    TOKEN_FIELDS = ("image", "video", "vision_start", "vision_end", "pad")
    LOSS_MASK_TOKEN_FIELDS = ("image", "video", "vision_start", "vision_end")

    vision_start_token = "S"
    image_token = "I"
    vision_end_token = "E"
    eos_token = "X"
    vision_start_id = 1
    vision_end_id = 2
    image_id = 3
    video_id = 4
    eos_id = 5
    pad_id = 6

    _token_ids = {"S": 1, "I": 3, "E": 2, "V": 4, "X": 5, "P": 6}

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


def _sample_image_processor(*, temporal_patch_size: int) -> ImageProcessor:
    return _image_processor_config(
        patch_size=_IMAGE_TOKEN_GEOM["patch_size"],
        temporal_patch_size=temporal_patch_size,
        spatial_merge_size=_IMAGE_TOKEN_GEOM["spatial_merge_size"],
        min_pixels=1,
        max_pixels=1_000_000,
    ).build()


def _process_synthetic_image(*, temporal_patch_size: int):
    return _process_mm_sample(
        texts=[None, "hi"],
        images=[_synthetic_rgb_image(), None],
        tokenizer=_FakeMMTokenizer(),
        image_processor=_sample_image_processor(
            temporal_patch_size=temporal_patch_size
        ),
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


class TestProcessMmSampleLossMasking(unittest.TestCase):
    def test_only_media_special_token_labels_are_masked(self):
        sample = _process_mm_sample(
            texts=["a", None, "VP"],
            images=[None, _synthetic_rgb_image(), None],
            tokenizer=_FakeMMTokenizer(),
            image_processor=_sample_image_processor(temporal_patch_size=1),
        )

        self.assertIsNotNone(sample)
        target_token_ids = sample["input_ids"][1:]
        target_labels = sample["labels"][:-1]
        for field in _FakeMMTokenizer.LOSS_MASK_TOKEN_FIELDS:
            token_id = getattr(_FakeMMTokenizer, f"{field}_id")
            self.assertTrue(torch.any(target_token_ids == token_id))
            self.assertTrue(
                torch.all(target_labels[target_token_ids == token_id] == IGNORE_INDEX)
            )
        self.assertEqual(sample["labels"][-2].item(), _FakeMMTokenizer.pad_id)
        self.assertEqual(sample["labels"][-1].item(), _FakeMMTokenizer.eos_id)


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


class TestProcessMmSampleImageErrors(unittest.TestCase):
    def test_unreadable_image_drops_sample(self):
        sample = _process_mm_sample(
            texts=[None, "hi"],
            images=[b"not an image", None],
            tokenizer=_FakeMMTokenizer(),
            image_processor=_sample_image_processor(temporal_patch_size=1),
        )

        self.assertIsNone(sample)

    def test_image_sample_without_image_processor_raises(self):
        with self.assertRaisesRegex(ValueError, "image_processor"):
            _process_mm_sample(
                texts=[None, "hi"],
                images=[_synthetic_rgb_image(), None],
                tokenizer=_FakeMMTokenizer(),
                image_processor=None,
            )


if __name__ == "__main__":
    unittest.main()
