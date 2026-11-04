# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from torchtitan.components.data.multimodal.utils.image import (
    decode_image_files,
    ImageSpec,
    largest_image_grid,
    lookup_image,
    num_image_tokens,
)
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.rl.generator.vllm.multimodal import (
    _ImageProcessingInfo,
    _ImageProcessor,
    image_prompt,
    mrope_hf_config,
    validate_image_model,
    VLLMMultimodalModelWrapper,
)
from torchtitan.rl.generator.vllm.registry import model_config_to_hf_config_dict
from torchtitan.rl.rollout.types import Prompt, PromptImage
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.transformers_utils.config import PretrainedConfig, uses_mrope

SPEC = ImageSpec(min_pixels=32 * 32, max_pixels=128 * 128)


@pytest.fixture
def image_paths(tmp_path) -> list[str]:
    rng = np.random.default_rng(0)
    paths = []
    for name, (width, height) in (("a.png", (64, 96)), ("b.png", (128, 64))):
        path = tmp_path / name
        Image.fromarray(
            rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        ).save(path)
        paths.append(str(path))
    return paths


def _prompt_images(image_paths) -> list[PromptImage]:
    images, offset = [], 2
    for path in image_paths:
        info = lookup_image(path, SPEC)
        length = num_image_tokens(info.grid_thw, SPEC)
        images.append(
            PromptImage(
                key=info.key,
                path=path,
                grid_thw=info.grid_thw,
                offset=offset,
                length=length,
                data=info.data,
            )
        )
        offset += length + 3
    return images


def test_image_prompt_places_each_image_on_its_placeholders(image_paths):
    # The mm hash, salt and pixels are checked through the engine in test_generator.
    images = _prompt_images(image_paths)
    num_tokens = images[-1].offset + images[-1].length + 2

    prompt = image_prompt(
        Prompt(token_ids=list(range(num_tokens)), images=tuple(images)),
        decode_image_files([image.data for image in images], SPEC),
    )

    assert [(p.offset, p.length) for p in prompt["mm_placeholders"]["image"]] == [
        (image.offset, image.length) for image in images
    ]
    for image, item in zip(images, prompt["mm_kwargs"]["image"], strict=True):
        assert item["image_grid_thw"].data.tolist() == list(image.grid_thw)


def test_mrope_input_positions_follow_placeholders():
    # 2 text tokens, a 1 x 4 x 4 image (merge 2: 2 x 2 = 4 tokens), 2 text tokens.
    image = PromptImage(
        key="k", path="k.png", grid_thw=(1, 4, 4), offset=2, length=4, data=b"k"
    )
    prompt = image_prompt(
        Prompt(token_ids=[0] * 8, images=(image,)),
        [torch.zeros(3, 64, 64, dtype=torch.uint8)],
    )
    features = [
        MultiModalFeatureSpec(
            data=item,
            modality="image",
            identifier=key,
            mm_position=placeholder,
            mm_hash=key,
        )
        for item, key, placeholder in zip(
            prompt["mm_kwargs"]["image"],
            prompt["mm_hashes"]["image"],
            prompt["mm_placeholders"]["image"],
        )
    ]
    wrapper = SimpleNamespace(_image_spec=SPEC)

    positions, delta = VLLMMultimodalModelWrapper.get_mrope_input_positions(
        wrapper, [0] * 8, features
    )

    assert positions.tolist() == [
        [0, 1, 2, 2, 2, 2, 4, 5],
        [0, 1, 2, 2, 3, 3, 4, 5],
        [0, 1, 2, 3, 2, 3, 4, 5],
    ]
    # Token 8 (the first after the prompt) has position 8 + delta = 6.
    assert delta == -2


def test_profiling_uses_the_specs_largest_grid():
    info = _ImageProcessingInfo(
        SimpleNamespace(model_config=SimpleNamespace(max_model_len=4096)),
        image_spec=SPEC,
    )
    largest = largest_image_grid(SPEC, 4096)
    num_tokens = largest[0] * largest[1] * largest[2] // SPEC.merge_size**2

    assert info.largest_grid(4096) == largest
    assert info.get_mm_max_tokens_per_item(4096, {"image": 1}) == {"image": num_tokens}

    processor = _ImageProcessor.__new__(_ImageProcessor)
    processor.info = info
    inputs = SimpleNamespace(
        mm_data_items=SimpleNamespace(get_count=lambda modality, strict: 2)
    )
    dummy = processor.apply(inputs, timing_ctx=None)

    [first, second] = dummy["mm_kwargs"]["image"]
    assert first["image_grid_thw"].data.tolist() == list(largest)
    assert first["images"].data.shape == (
        3,
        largest[1] * SPEC.patch_size,
        largest[2] * SPEC.patch_size,
    )
    assert len(dummy["prompt_token_ids"]) == 2 * num_tokens
    assert dummy["mm_hashes"]["image"][0] != dummy["mm_hashes"]["image"][1]


def test_hf_config_declares_mrope_only_for_image_prompts():
    config = build_model_config(
        "0.8B", seq_len=256, attn_backend="varlen", parallelism=ParallelismConfig()
    )

    text = PretrainedConfig.from_dict(model_config_to_hf_config_dict(config))
    image = PretrainedConfig.from_dict(
        {**model_config_to_hf_config_dict(config), **mrope_hf_config(config)}
    )

    assert not uses_mrope(text)
    assert uses_mrope(image)
    assert image.rope_parameters["mrope_section"] == list(
        config.first_full_attention.rope.mrope_section
    )


def test_validate_image_model_rejects_a_mismatched_cache_or_no_encoder():
    config = build_model_config(
        "0.8B", seq_len=256, attn_backend="varlen", parallelism=ParallelismConfig()
    )
    validate_image_model(config, SPEC)

    with pytest.raises(ValueError, match="Use the model's preprocessing"):
        validate_image_model(config, dataclasses.replace(SPEC, patch_size=14))
    with pytest.raises(ValueError, match="no vision encoder"):
        validate_image_model(dataclasses.replace(config, vision_encoder=None), SPEC)
