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
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torchvision.transforms.v2.functional as TVF
from PIL import Image
from torchtitan.components.data.multimodal.mm_datasets import (
    _process_chunked_video_text_sample,
    _process_mm_sample,
)
from torchtitan.components.data.multimodal.utils.image import (
    calculate_vision_tokens,
    process_image,
    resize_to_navit_patch_grid,
    resize_to_pixel_budget,
    vision_to_patches,
)
from torchtitan.components.data.multimodal.utils.video import (
    format_video_timestamp,
    load_npy_video_frames,
    load_video,
    load_video_frames,
    process_video,
    sample_frame_indices,
)

from torchtitan.components.loss import IGNORE_INDEX


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


class TestProcessImageInterpolationMode(unittest.TestCase):
    def test_lanczos_matches_torchvision_tensor_resize(self):
        image_CHW = torch.arange(3 * 5 * 7, dtype=torch.uint8).reshape(3, 5, 7)
        pil_image = Image.fromarray(image_CHW.permute(1, 2, 0).numpy())
        expected_CHW = TVF.resize(
            image_CHW,
            [3, 4],
            interpolation=TVF.InterpolationMode.LANCZOS,
            antialias=True,
        )
        expected_CHW = TVF.to_dtype(expected_CHW, torch.float32, scale=True)
        expected_CHW = TVF.normalize(expected_CHW, [0.5] * 3, [0.5] * 3)

        def fixed_geometry(*args, **kwargs):
            return 3, 4, 0, 0

        actual = process_image(
            pil_image,
            resize_fn=fixed_geometry,
            image_interpolation_mode=TVF.InterpolationMode.LANCZOS,
        )

        self.assertIsNotNone(actual)
        self.assertTrue(torch.equal(actual, expected_CHW.permute(1, 2, 0).unsqueeze(0)))


class TestLoadNpyVideoFrames(unittest.TestCase):
    def test_loads_c_contiguous_uint8_thwc_array(self):
        frames_THWC = np.arange(2 * 3 * 4 * 3, dtype=np.uint8).reshape(2, 3, 4, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frames.npy"
            np.save(path, frames_THWC)

            actual_THWC = load_npy_video_frames(path)

        self.assertTrue(actual_THWC.is_contiguous())
        self.assertTrue(torch.equal(actual_THWC, torch.from_numpy(frames_THWC)))


class TestProcessVideoResize(unittest.TestCase):
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
        expected_TCHW = TVF.to_dtype(expected_TCHW, torch.float32, scale=True)
        expected_TCHW = TVF.normalize(expected_TCHW, [0.5] * 3, [0.5] * 3)

        actual_THWC = process_video(
            video_THWC,
            patch_size=4,
            merge_size=2,
            max_pixels=200,
            min_pixels=20,
            resize_fn=custom_resize,
            image_interpolation_mode=TVF.InterpolationMode.NEAREST,
            max_patches=12,
            max_patches_per_side=5,
        )

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


class TestProcessMmSampleLossMasking(unittest.TestCase):
    def test_only_media_special_token_labels_are_masked(self):
        sample = _process_mm_sample(
            texts=["a", None, "VP"],
            images=[None, _synthetic_rgb_image(), None],
            tokenizer=_FakeMMTokenizer(),
            patch_size=_IMAGE_TOKEN_GEOM["patch_size"],
            temporal_patch_size=1,
            spatial_merge_size=_IMAGE_TOKEN_GEOM["spatial_merge_size"],
            min_pixels=1,
            max_pixels=1_000_000,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            resize_fn=resize_to_pixel_budget,
            max_patches=4096,
            max_patches_per_side=512,
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

    def test_process_mm_sample_forwards_configured_temporal_patch_size(self):
        for tps in (2, 7):
            with patch(
                "torchtitan.components.data.multimodal.mm_datasets.calculate_vision_tokens",
                wraps=calculate_vision_tokens,
            ) as mock_calc:
                sample = _process_synthetic_image(temporal_patch_size=tps)
            self.assertIsNotNone(sample)
            mock_calc.assert_called_once()
            self.assertEqual(mock_calc.call_args.kwargs["temporal_patch_size"], tps)
            self.assertEqual(mock_calc.call_args.kwargs["num_frames"], 1)


class _FakeSharedMediaTokenizer(_FakeMMTokenizer):
    video_token = _FakeMMTokenizer.image_token
    video_id = _FakeMMTokenizer.image_id


def _process_chunked_video(
    tokenizer, *, num_frames, temporal_patch_size=1, frame_times=None
):
    if frame_times is None:
        frame_times = [i / 8.0 for i in range(num_frames)]
    return _process_chunked_video_text_sample(
        text="cap",
        frames=torch.zeros(num_frames, 28, 56, 3, dtype=torch.uint8),
        tokenizer=tokenizer,
        patch_size=14,
        temporal_patch_size=temporal_patch_size,
        spatial_merge_size=2,
        min_pixels=1,
        max_pixels=1_000_000,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        resize_fn=resize_to_navit_patch_grid,
        max_patches=4096,
        max_patches_per_side=512,
        frame_times=frame_times,
        frames_per_chunk=4,
        chunk_header="video<|media_content|>",
    )


def _write_test_video(path, *, codec, pix_fmt, num_frames=12, fps=30):
    av = pytest.importorskip("av")
    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=fps)
        stream.width, stream.height, stream.pix_fmt = 16, 16, pix_fmt
        for i in range(num_frames):
            frame = np.full((16, 16, 3), 10 * i, dtype=np.uint8)
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(frame, format="rgb24")
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class TestFormatVideoTimestamp(unittest.TestCase):
    def test_formats_hours_minutes_seconds_and_millis(self):
        self.assertEqual(format_video_timestamp(0.0), "00:00:00.000")
        self.assertEqual(format_video_timestamp(0.5), "00:00:00.500")
        self.assertEqual(format_video_timestamp(3725.25), "01:02:05.250")

    def test_spells_float_times_as_the_kimi_processor_does(self):
        self.assertEqual(format_video_timestamp(288 / 30), "00:00:09.599")
        self.assertEqual(format_video_timestamp(1.9999999), "00:00:02.999")


class TestSampleFrameIndices(unittest.TestCase):
    def test_keeps_the_kimi_processor_frames(self):
        self.assertEqual(sample_frame_indices(12, 30.0, 8.0), [0, 6, 11])
        self.assertEqual(sample_frame_indices(10, 5.0, 8.0), list(range(10)))
        self.assertEqual(sample_frame_indices(1, 30.0, 8.0), [0])
        self.assertEqual(sample_frame_indices(17, 30000 / 1001, 8.0), [0, 4, 8, 12, 16])
        indices = sample_frame_indices(300, 30.0, 8.0)
        self.assertEqual(len(indices), 80)
        self.assertEqual(indices[:9], [0, 4, 8, 11, 15, 19, 23, 26, 30])
        self.assertEqual(indices[-3:], [291, 295, 299])


class TestLoadVideo(unittest.TestCase):
    def test_load_video_frames_returns_sampled_frames_and_times(self):
        for suffix, codec, pix_fmt in (
            ("mov", "png", "rgb24"),
            ("mkv", "ffv1", "bgr0"),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / f"clip.{suffix}"
                _write_test_video(path, codec=codec, pix_fmt=pix_fmt)
                loaded = load_video_frames(str(path), sample_fps=8.0)
            self.assertIsNotNone(loaded)
            frames, times = loaded
            self.assertEqual(frames[:, 0, 0, 0].tolist(), [0, 60, 110])
            self.assertEqual(times, [0.0, 6 / 30, 11 / 30])

    def test_load_video_keeps_its_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mov"
            _write_test_video(path, codec="png", pix_fmt="rgb24")
            frames = load_video(str(path), fps=2.0)
        self.assertEqual(frames[:, 0, 0, 0].tolist(), [0, 30, 70, 110])


class TestProcessChunkedVideoTextSample(unittest.TestCase):
    def test_chunks_follow_their_timestamps_in_order(self):
        tokenizer = _FakeSharedMediaTokenizer()
        sample = _process_chunked_video(tokenizer, num_frames=10)

        self.assertIsNotNone(sample)
        self.assertEqual(
            [tuple(chunk.shape) for chunk in sample["pixel_values"]],
            [(4, 28, 56, 3), (4, 28, 56, 3), (2, 28, 56, 3)],
        )
        chunk = "Svideo<|media_content|>" + "I" * 2 + "E"
        expected = tokenizer.encode(
            f"00:00:00.000{chunk}00:00:00.500{chunk}00:00:01.000{chunk}capX"
        )
        self.assertEqual(sample["input_ids"].tolist(), expected[:-1])

    def test_chunk_timestamps_are_their_first_frame_times(self):
        tokenizer = _FakeSharedMediaTokenizer()
        sample = _process_chunked_video(
            tokenizer, num_frames=6, frame_times=[0.0, 0.1, 0.2, 0.3, 9.6, 9.7]
        )

        self.assertIsNotNone(sample)
        chunk = "Svideo<|media_content|>" + "I" * 2 + "E"
        expected = tokenizer.encode(f"00:00:00.000{chunk}00:00:09.599{chunk}capX")
        self.assertEqual(sample["input_ids"].tolist(), expected[:-1])

    def test_rejects_frame_times_of_another_length(self):
        with self.assertRaises(ValueError):
            _process_chunked_video(
                _FakeSharedMediaTokenizer(), num_frames=4, frame_times=[0.0]
            )

    def test_rejects_a_separate_video_placeholder(self):
        tokenizer = _FakeSharedMediaTokenizer()
        tokenizer.video_token = "V"
        with self.assertRaises(ValueError):
            _process_chunked_video(tokenizer, num_frames=4)

    def test_rejects_temporal_patches(self):
        with self.assertRaises(ValueError):
            _process_chunked_video(
                _FakeSharedMediaTokenizer(), num_frames=4, temporal_patch_size=2
            )


if __name__ == "__main__":
    unittest.main()
