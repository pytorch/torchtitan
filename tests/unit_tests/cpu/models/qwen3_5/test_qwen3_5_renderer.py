# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The image-reference Qwen3.5 renderer renders the same tokens as the library's.

Downloads the Qwen3.5 tokenizer from the Hugging Face Hub; set ``QWEN3_5_TOKENIZER`` to
use a local copy instead.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import transformers
from PIL import Image
from renderers import CustomRendererConfig, Qwen35RendererConfig
from renderers.qwen35 import Qwen35Renderer

from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.components.renderer import from_renderers, hash_image_file
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.models.qwen3_5 import QWEN3_5_VISION_PROCESSOR_KWARGS
from torchtitan.models.qwen3_5.renderer import (
    Qwen35ImageRefRenderer,
    Qwen35ImageRefRendererConfig,
)


@pytest.fixture(scope="module")
def tokenizer():
    return transformers.AutoTokenizer.from_pretrained(
        os.environ.get("QWEN3_5_TOKENIZER", "Qwen/Qwen3.5-0.8B")
    )


@pytest.fixture
def image_paths(tmp_path):
    rng = np.random.default_rng(0)
    paths = []
    for name, (width, height) in (("a.png", (300, 200)), ("b.png", (640, 512))):
        path = tmp_path / name
        pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        paths.append(str(path))
    return paths


@pytest.fixture
def vision_processor() -> VisionProcessor.Config:
    return VisionProcessor.Config(**QWEN3_5_VISION_PROCESSOR_KWARGS)


def _hashes(image_paths):
    return [hash_image_file(Path(path).read_bytes()) for path in image_paths]


def _hf_image_processor():
    """The Hugging Face processor Qwen3.5 checkpoints ship (preprocessor_config.json)."""
    return transformers.Qwen2VLImageProcessorFast(
        size={"longest_edge": 16777216, "shortest_edge": 65536},
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )


def _messages(image_paths):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": image_paths[0]},
                {"type": "text", "text": "Compare this image"},
                {"type": "image", "path": image_paths[1]},
                {"type": "text", "text": "with this one."},
            ],
        }
    ]


def test_matches_library_renderer(tokenizer, vision_processor, image_paths) -> None:
    hf_processor = _hf_image_processor()
    library = Qwen35Renderer(
        tokenizer,
        Qwen35RendererConfig(enable_thinking=False),
        processor=SimpleNamespace(image_processor=hf_processor),
    )
    renderer = Qwen35ImageRefRenderer(
        tokenizer,
        Qwen35ImageRefRendererConfig(
            enable_thinking=False, vision_processor=vision_processor
        ),
    )

    expected = library.render(_messages(image_paths), add_generation_prompt=True)
    actual = renderer.render(_messages(image_paths), add_generation_prompt=True)

    assert actual.token_ids == expected.token_ids
    assert actual.message_indices == expected.message_indices
    expected_mm, actual_mm = expected.multi_modal_data, actual.multi_modal_data
    assert actual_mm.mm_placeholders == expected_mm.mm_placeholders
    assert actual_mm.mm_hashes["image"] == _hashes(image_paths)
    for actual_item, expected_item in zip(
        actual_mm.mm_items["image"], expected_mm.mm_items["image"]
    ):
        assert actual_item["pixel_values"] is None
        np.testing.assert_array_equal(
            actual_item["image_grid_thw"], expected_item["image_grid_thw"]
        )

    # The image processor's pixel values are what the HF processor feeds the encoder
    # (equal up to float32 rounding of the normalization).
    processor = vision_processor.build()
    pixel_values = torch.cat(
        [
            processor.patchify(
                processor.resize_and_normalize(Path(path).read_bytes())[0]
            )[0]
            for path in image_paths
        ]
    )
    hf_pixel_values = np.concatenate(
        [item["pixel_values"] for item in expected_mm.mm_items["image"]]
    )
    torch.testing.assert_close(pixel_values, torch.from_numpy(hf_pixel_values))


def test_builds_through_custom_renderer_config_with_the_vision_processor(
    tokenizer, vision_processor, image_paths, tmp_path
) -> None:
    # As the controller builds it: the image processor is a build argument, like the
    # tokenizer, not a field of the renderer config.
    tokenizer.save_pretrained(tmp_path)
    renderer = from_renderers(
        CustomRendererConfig(
            import_path="torchtitan.models.qwen3_5.renderer.Qwen35ImageRefRenderer",
            enable_thinking=False,
        )
    ).build(
        tokenizer=HuggingFaceTokenizer(tokenizer_path=str(tmp_path)),
        vision_processor=vision_processor.build(),
    )
    assert isinstance(renderer, Qwen35ImageRefRenderer)
    rendered = renderer.render(_messages(image_paths), add_generation_prompt=True)
    assert rendered.multi_modal_data.mm_hashes["image"] == _hashes(image_paths)


def test_hashes_an_image_by_content_from_any_path(
    tokenizer, vision_processor, image_paths, tmp_path, monkeypatch
) -> None:
    renderer = Qwen35ImageRefRenderer(
        tokenizer, Qwen35ImageRefRendererConfig(vision_processor=vision_processor)
    )
    moved = tmp_path / "elsewhere" / "renamed.png"
    moved.parent.mkdir()
    shutil.copy(image_paths[0], moved)
    monkeypatch.chdir(moved.parent)

    def image_hash(path):
        message = {"role": "user", "content": [{"type": "image", "path": path}]}
        rendered = renderer.render([message], add_generation_prompt=True)
        return rendered.multi_modal_data.mm_hashes["image"][0]

    assert image_hash("renamed.png") == image_hash(image_paths[0])
    assert image_hash(image_paths[0]) != image_hash(image_paths[1])


def test_needs_a_vision_processor(tokenizer) -> None:
    with pytest.raises(ValueError, match="needs a vision processor"):
        Qwen35ImageRefRenderer(tokenizer, Qwen35ImageRefRendererConfig())


def test_rejects_image_cache_max() -> None:
    with pytest.raises(ValueError, match="image_cache_max"):
        Qwen35ImageRefRendererConfig(image_cache_max=1)


def test_rejects_image_parts_without_a_path(tokenizer, vision_processor) -> None:
    renderer = Qwen35ImageRefRenderer(
        tokenizer, Qwen35ImageRefRendererConfig(vision_processor=vision_processor)
    )
    message = {"role": "user", "content": [{"type": "image", "image": "y.png"}]}
    with pytest.raises(ValueError, match="needs image parts with a 'path'"):
        renderer.render([message], add_generation_prompt=True)
