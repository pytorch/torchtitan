# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU unit tests for the multimodal dataset image preprocessing.

``NaViTPatchGridResize`` follows Kimi's NaViT image geometry: apply total and
per-side patch limits before padding to a ``patch_size * merge_size`` grid.
These tests pin that pure-geometry behavior, and the ``VisionProcessor`` image
and video preprocessing built on the resize rules.
"""

import math
import unittest
from dataclasses import dataclass

import torch
import torchvision.transforms.v2.functional as TVF
from PIL import Image
from torchtitan.components.data.multimodal.mm_datasets import _process_mm_sample
from torchtitan.components.data.multimodal.vision_processor import (
    NaViTPatchGridResize,
    PixelBudgetResize,
    ResizeRule,
    VisionProcessor,
)

from torchtitan.components.loss import IGNORE_INDEX


def _vision_processor_config(**overrides) -> VisionProcessor.Config:
    fields = dict(
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        resize=PixelBudgetResize.Config(min_pixels=65_536, max_pixels=16_777_216),
        patch_order="block",
    )
    fields.update(overrides)
    return VisionProcessor.Config(**fields)


class _FixedResize(ResizeRule):
    """Resizes to ``size`` and pads by ``padding``, recording each call's
    ``(height, width, patch_size, merge_size)``."""

    @dataclass(kw_only=True, slots=True)
    class Config(ResizeRule.Config):
        size: tuple[int, int]
        padding: tuple[int, int] = (0, 0)

    def __init__(self, config: Config) -> None:
        self.size = config.size
        self.padding = config.padding
        self.calls: list[tuple[int, int, int, int]] = []

    def __call__(
        self, height: int, width: int, *, patch_size: int, merge_size: int
    ) -> tuple[int, int, int, int]:
        self.calls.append((height, width, patch_size, merge_size))
        return (*self.size, *self.padding)


def _reference_patches(
    resized_TCHW: torch.Tensor, config: VisionProcessor.Config
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize and patchify resized uint8 frames step by step."""
    pixels_TCHW = TVF.to_dtype(resized_TCHW, torch.float32, scale=True)
    pixels_TCHW = TVF.normalize(
        pixels_TCHW, list(config.image_mean), list(config.image_std)
    )
    return config.build().patchify(pixels_TCHW.permute(0, 2, 3, 1))


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


class TestNaViTPatchGridResize(unittest.TestCase):
    PS, MERGE, LIMIT, SIDE = 14, 2, 4096, 512

    def _resize(self, h, w):
        resize = NaViTPatchGridResize.Config(
            max_patches=self.LIMIT, max_patches_per_side=self.SIDE
        ).build()
        return resize(h, w, patch_size=self.PS, merge_size=self.MERGE)

    def _final(self, h, w):
        rh, rw, ph, pw = self._resize(h, w)
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
        rh, rw, ph, pw = self._resize(30, 30)
        self.assertEqual((rh, rw), (30, 30))

    def test_per_side_cap_scales_down(self):
        # The per-side limit scales an extreme aspect ratio instead of dropping it.
        final_h, final_w = self._final(self.PS * 600, self.PS * 2)
        self.assertEqual(final_h // self.PS, self.SIDE)
        self.assertEqual(final_w % (self.PS * self.MERGE), 0)

    def test_per_side_cap_is_inclusive(self):
        height, width = self.PS * self.SIDE, self.PS * self.MERGE
        rh, rw, ph, pw = self._resize(height, width)
        self.assertEqual((rh, rw, ph, pw), (height, width, 0, 0))


class TestVisionProcessorNaViTPatchGrid(unittest.TestCase):
    def test_navit_pads_to_factor_multiple(self):
        ps, merge = 14, 2
        factor = ps * merge
        # 100x173 is not a factor multiple -> navit must pad to one.
        img = Image.fromarray((torch.rand(100, 173, 3) * 255).to(torch.uint8).numpy())
        processor = _vision_processor_config(
            patch_size=ps,
            temporal_patch_size=1,
            spatial_merge_size=merge,
            resize=NaViTPatchGridResize.Config(
                max_patches=4096, max_patches_per_side=512
            ),
            patch_order="raster",
        ).build()
        pixels, grid_thw = processor.resize_and_normalize(img)
        pixel_values, _ = processor.patchify(pixels)
        t, h, w = grid_thw.tolist()
        H, W = h * ps, w * ps
        self.assertEqual(H % factor, 0)
        self.assertEqual(W % factor, 0)
        want_h, want_w = _navit_patch_grid_geometry(
            100, 173, patch_size=ps, merge_size=merge, max_patches=4096
        )
        self.assertEqual((H, W), (want_h, want_w))
        self.assertEqual(pixel_values.shape, (t * h * w, 3 * ps * ps))


class TestVisionProcessorInterpolationMode(unittest.TestCase):
    def test_lanczos_matches_torchvision_tensor_resize(self):
        image_CHW = torch.arange(3 * 5 * 7, dtype=torch.uint8).reshape(3, 5, 7)
        pil_image = Image.fromarray(image_CHW.permute(1, 2, 0).numpy())
        expected_CHW = TVF.resize(
            image_CHW,
            [4, 6],
            interpolation=TVF.InterpolationMode.LANCZOS,
            antialias=True,
        )
        config = _vision_processor_config(
            patch_size=2,
            spatial_merge_size=1,
            resize=_FixedResize.Config(size=(4, 6)),
            image_interpolation_mode="lanczos",
        )
        processor = config.build()
        actual, _ = processor.patchify(processor.resize_and_normalize(pil_image)[0])

        expected, _ = _reference_patches(expected_CHW.unsqueeze(0), config)
        self.assertTrue(torch.equal(actual, expected))


class TestVisionProcessorPreprocess(unittest.TestCase):
    def test_matches_step_by_step_preprocessing(self):
        # decode -> resize -> normalize -> patchify, bitwise, in both patch orders.
        image_HWC = (torch.rand(100, 173, 3) * 255).to(torch.uint8)
        pil_image = Image.fromarray(image_HWC.numpy())
        for patch_order in ("block", "raster"):
            config = _vision_processor_config(patch_order=patch_order)

            processor = config.build()
            pixels, grid_thw = processor.resize_and_normalize(pil_image)
            pixel_values, patch_grid_thw = processor.patchify(pixels)
            self.assertEqual(patch_grid_thw.tolist(), grid_thw.tolist())

            resize_h, resize_w, _, _ = config.resize.build()(
                100, 173, patch_size=16, merge_size=2
            )
            expected_CHW = TVF.resize(
                image_HWC.permute(2, 0, 1),
                [resize_h, resize_w],
                interpolation=TVF.InterpolationMode.BICUBIC,
                antialias=True,
            )
            expected, expected_grid = _reference_patches(
                expected_CHW.unsqueeze(0), config
            )
            self.assertTrue(torch.equal(pixel_values, expected))
            self.assertEqual(grid_thw.tolist(), expected_grid.tolist())

    def test_num_placeholder_tokens(self):
        processor = _vision_processor_config().build()
        # A (1, 4, 6) patch grid merges 2x2 patches into 2 * 3 tokens.
        self.assertEqual(processor.num_placeholder_tokens((1, 4, 6)), 6)


class TestVisionProcessorVideo(unittest.TestCase):
    def test_video_resize_padding_and_interpolation(self):
        video_TCHW = torch.arange(1 * 3 * 2 * 2, dtype=torch.uint8).reshape(1, 3, 2, 2)
        video_THWC = video_TCHW.permute(0, 2, 3, 1)

        expected_TCHW = TVF.resize(
            video_TCHW,
            [4, 6],
            interpolation=TVF.InterpolationMode.NEAREST,
            antialias=True,
        )
        expected_TCHW = TVF.pad(expected_TCHW, [0, 0, 2, 4])

        # Frames use ``video_resize``, not the image rule.
        config = _vision_processor_config(
            patch_size=4,
            resize=_FixedResize.Config(size=(8, 8)),
            video_resize=_FixedResize.Config(size=(4, 6), padding=(4, 2)),
            image_interpolation_mode="nearest",
        )
        processor = config.build()
        pixels, grid_thw = processor.resize_and_normalize_video(video_THWC)
        pixel_values, patch_grid_thw = processor.patchify(pixels)
        self.assertEqual(patch_grid_thw.tolist(), grid_thw.tolist())

        self.assertEqual(processor.video_resize.calls, [(2, 2, 4, 2)])
        self.assertEqual(processor.resize.calls, [])
        expected, expected_grid = _reference_patches(expected_TCHW, config)
        # Resized to 4 x 6, padded to 8 x 8: a (1, 2, 2) grid of 4 x 4 patches.
        self.assertEqual(grid_thw.tolist(), [1, 2, 2])
        self.assertEqual(expected_grid.tolist(), [1, 2, 2])
        self.assertTrue(torch.equal(pixel_values, expected))

    def test_video_without_video_resize_raises(self):
        processor = _vision_processor_config().build()
        frames_THWC = torch.zeros(2, 32, 32, 3, dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, "takes no video"):
            processor.resize_and_normalize_video(frames_THWC)


class TestVisionToPatchesOrder(unittest.TestCase):
    """Patch sequence layout: 'block' vs 'raster'."""

    def test_block_to_raster_permutation(self):
        # 2x4 patch grid (h=2, w=4), merge_size=2. Distinct per-patch values so
        # the two orderings are a pure permutation of each other.
        img = torch.arange(1 * 28 * 56 * 3, dtype=torch.float32).reshape(1, 28, 56, 3)
        moonvit_style = dict(patch_size=14, temporal_patch_size=1, spatial_merge_size=2)
        block, grid = (
            _vision_processor_config(**moonvit_style, patch_order="block")
            .build()
            .patchify(img)
        )
        raster, _ = (
            _vision_processor_config(**moonvit_style, patch_order="raster")
            .build()
            .patchify(img)
        )

        self.assertEqual(grid.tolist(), [1, 2, 4])
        self.assertEqual(block.shape, raster.shape)
        # block slot b corresponds to raster slot block_to_raster_idx[b].
        block_to_raster_idx = [0, 1, 4, 5, 2, 3, 6, 7]
        for b, r in enumerate(block_to_raster_idx):
            self.assertTrue(torch.equal(block[b], raster[r]))

    def test_invalid_patch_order_raises(self):
        with self.assertRaisesRegex(ValueError, "patch_order"):
            _vision_processor_config(patch_order="bogus")


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


def _sample_vision_processor(*, temporal_patch_size: int) -> VisionProcessor:
    return _vision_processor_config(
        patch_size=_IMAGE_TOKEN_GEOM["patch_size"],
        temporal_patch_size=temporal_patch_size,
        spatial_merge_size=_IMAGE_TOKEN_GEOM["spatial_merge_size"],
        resize=PixelBudgetResize.Config(min_pixels=1, max_pixels=1_000_000),
    ).build()


def _process_synthetic_image(*, temporal_patch_size: int):
    return _process_mm_sample(
        texts=[None, "hi"],
        images=[_synthetic_rgb_image(), None],
        tokenizer=_FakeMMTokenizer(),
        vision_processor=_sample_vision_processor(
            temporal_patch_size=temporal_patch_size
        ),
    )


class TestProcessMmSampleLossMasking(unittest.TestCase):
    def test_only_media_special_token_labels_are_masked(self):
        sample = _process_mm_sample(
            texts=["a", None, "VP"],
            images=[None, _synthetic_rgb_image(), None],
            tokenizer=_FakeMMTokenizer(),
            vision_processor=_sample_vision_processor(temporal_patch_size=1),
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
        # A 64 x 64 image in 32 x 32-pixel merged tokens, for any temporal patch size.
        expected_tokens = 4
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
            vision_processor=_sample_vision_processor(temporal_patch_size=1),
        )

        self.assertIsNone(sample)

    def test_image_sample_without_vision_processor_raises(self):
        with self.assertRaisesRegex(ValueError, "vision_processor"):
            _process_mm_sample(
                texts=[None, "hi"],
                images=[_synthetic_rgb_image(), None],
                tokenizer=_FakeMMTokenizer(),
                vision_processor=None,
            )


if __name__ == "__main__":
    unittest.main()
