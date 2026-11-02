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

import io
import json
import math
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import grain.python as grain
import numpy as np
import pytest
import torch
import torchvision.transforms.v2.functional as TVF
from PIL import Image
from renderers import CustomRendererConfig
from torchtitan.components.data.dataset import SingleDatasetConfig
from torchtitan.components.data.multimodal.mm_collator import MultiModalCollator
from torchtitan.components.data.multimodal.mm_datasets import (
    _process_mm_sample,
    MMSamplePackingConfig,
    MultiModalChatProcessor,
)
from torchtitan.components.data.multimodal.utils.image import (
    calculate_vision_tokens,
    process_image,
    resize_to_muse_glimmer_patch_grid,
    resize_to_navit_patch_grid,
    resize_to_pixel_budget,
    vision_to_patches,
)
from torchtitan.components.data.multimodal.utils.text import expand_image_placeholders
from torchtitan.components.data.multimodal.utils.video import (
    load_npy_video_frames,
    process_video,
)
from torchtitan.components.data.types import DatasetBuildContext, DatasetIterationPolicy

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.renderer import from_renderers
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.models.muse_spark.renderer import MuseSparkRendererConfig


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


# Grids match the HF MuseGlimmerImageProcessor. 4000x3000 would be 20,000 merged
# tokens, so it takes the 4096-token cap path.
@pytest.mark.parametrize(
    "height, width, expected",
    [
        (256, 256, (280, 280, 0, 0)),
        (240, 240, (224, 224, 0, 0)),
        (4000, 3000, (2044, 1540, 0, 0)),
    ],
)
def test_resize_to_muse_glimmer_patch_grid(height, width, expected):
    assert (
        resize_to_muse_glimmer_patch_grid(height, width, patch_size=14, merge_size=2)
        == expected
    )


def test_expand_image_placeholders_rejects_count_mismatch():
    with pytest.raises(ValueError, match="2 image placeholders for 1 images"):
        expand_image_placeholders(
            [3, 3], [False, False], [1], image_id=3, image_start_id=1, image_end_id=2
        )


_MUSE_SPECIAL_TOKENS = {
    "<|begin_of_text|>": 200000,
    "<|end_of_text|>": 200001,
    "<|eom|>": 200007,
    "<|eot|>": 200008,
    "<|finetune_right_pad|>": 200018,
    "<|start|>": 200022,
    "<|message|>": 200023,
    "<|image_start|>": 200080,
    "<|image_end|>": 200081,
    "<|video|>": 200091,
    "<|patch|>": 200092,
}


@pytest.fixture
def muse_tokenizer(tmp_path):
    # Character tokens with the Muse special-token ids, without checkpoint assets.
    vocab = {chr(index): index for index in range(128)} | _MUSE_SPECIAL_TOKENS
    (tmp_path / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "model": {"type": "BPE", "vocab": vocab, "merges": []},
                "decoder": {"type": "Fuse"},
                "added_tokens": [
                    {
                        "id": token_id,
                        "content": token,
                        "single_word": False,
                        "lstrip": False,
                        "rstrip": False,
                        "normalized": False,
                        "special": True,
                    }
                    for token, token_id in _MUSE_SPECIAL_TOKENS.items()
                ],
            }
        )
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"bos_token": "<|begin_of_text|>", "eos_token": "<|end_of_text|>"})
    )
    return MultiModalTokenizer.Config(
        image_token="<|patch|>",
        video_token="<|video|>",
        vision_start_token="<|image_start|>",
        vision_end_token="<|image_end|>",
        pad_token="<|finetune_right_pad|>",
    ).build(tokenizer_path=str(tmp_path))


def _png_bytes(height: int, width: int) -> bytes:
    buffer = io.BytesIO()
    _synthetic_rgb_image(height, width).save(buffer, format="PNG")
    return buffer.getvalue()


def _vqa_messages(sample: dict) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": sample["question"]},
            ],
        },
        {"role": "assistant", "content": sample["answer"]},
    ]


def _chat_processor(renderer, tokenizer, max_context_length=2048):
    # 56 x 84 pixels stay unresized: a 2 x 3 grid of 28-pixel merged patches.
    return MultiModalChatProcessor.Config(
        messages_fn=_vqa_messages,
        renderer=renderer,
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
        min_pixels=784,
        max_pixels=1_000_000,
    ).build(
        context=DatasetBuildContext(
            tokenizer=tokenizer,
            max_context_length=max_context_length,
            num_tokens_per_microbatch=max_context_length,
            read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
        )
    )


@pytest.mark.parametrize(
    "renderer, answer_targets",
    [
        (
            MuseSparkRendererConfig(current_date="2026-01-15"),
            " to=user<|message|>yes<|eot|>",
        ),
        (
            from_renderers(
                CustomRendererConfig(
                    import_path=(
                        "torchtitan.models.muse_glimmer.renderer.MuseGlimmerRenderer"
                    )
                )
            ),
            "yes<|eot|>",
        ),
    ],
)
def test_chat_processor_expands_image_and_masks_prompt(
    muse_tokenizer, renderer, answer_targets
):
    processor = _chat_processor(renderer, muse_tokenizer)
    sample = processor(
        {"image": _png_bytes(56, 84), "question": "Is it a lung?", "answer": "yes"},
        np.random.default_rng(0),
    )

    input_ids = sample["input_ids"].tolist()
    image_start = input_ids.index(muse_tokenizer.vision_start_id)
    assert input_ids[image_start : image_start + 8] == [
        muse_tokenizer.vision_start_id,
        *[muse_tokenizer.image_id] * 6,
        muse_tokenizer.vision_end_id,
    ]
    assert input_ids.count(muse_tokenizer.image_id) == 6
    assert [image.shape for image in sample["pixel_values"]] == [(1, 56, 84, 3)]
    labels = sample["labels"]
    assert len(labels) == len(input_ids)
    targets = labels[labels != IGNORE_INDEX].tolist()
    assert muse_tokenizer.decode(targets, skip_special_tokens=False) == answer_targets


def test_chat_loss_mask_survives_packing_and_collation(muse_tokenizer):
    processor = _chat_processor(
        MuseSparkRendererConfig(current_date="2026-01-15"), muse_tokenizer
    )
    samples = [
        processor(
            {"image": _png_bytes(56, 84), "question": question, "answer": answer},
            np.random.default_rng(0),
        )
        for question, answer in (("Lung?", "yes"), ("Organ?", "liver"))
    ]
    num_tokens = sum(len(sample["input_ids"]) for sample in samples) + 3

    @dataclass(frozen=True)
    class _SamplesSourceConfig:
        def build(self):
            return samples

    context = DatasetBuildContext(
        tokenizer=muse_tokenizer,
        max_context_length=num_tokens,
        num_tokens_per_microbatch=num_tokens,
        read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
    )
    packed = next(
        iter(
            MMSamplePackingConfig(
                dataset=SingleDatasetConfig(source=_SamplesSourceConfig())
            ).build(
                context=context,
                dataset_iteration_policy=DatasetIterationPolicy(
                    seed=0,
                    shuffle=False,
                    repeat=False,
                    dp_rank=0,
                    dp_world_size=1,
                    streaming_shuffle_buffer_size=1,
                ),
            )
        )
    )

    expected_labels = torch.cat([sample["labels"] for sample in samples])
    assert torch.equal(packed["labels"][: len(expected_labels)], expected_labels)
    assert (packed["labels"][len(expected_labels) :] == IGNORE_INDEX).all()
    assert len(packed["pixel_values"]) == 2

    collator = MultiModalCollator.Config(
        patch_size=14, temporal_patch_size=2, spatial_merge_size=2
    ).build(context=context)
    microbatch = collator([packed])
    assert torch.equal(microbatch.labels, packed["labels"])
    assert microbatch.model_kwargs["grid_thw"].tolist() == [[1, 4, 6], [1, 4, 6]]


if __name__ == "__main__":
    unittest.main()
