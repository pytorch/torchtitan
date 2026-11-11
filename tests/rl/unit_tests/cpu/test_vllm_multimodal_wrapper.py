# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import dataclasses
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from renderers import PlaceholderRange

from torchtitan.components.data.multimodal.vision_processor import (
    NaViTPatchGridResize,
    PixelBudgetResize,
    VisionProcessor,
)
from torchtitan.components.renderer import hash_image_file
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.qwen3_5 import (
    build_model_config,
    QWEN3_5_VISION_PROCESSOR_KWARGS,
)
from torchtitan.rl.generator.vllm.multimodal_wrapper import (
    _ImageMultiModalProcessor,
    _ImageProcessingInfo,
    _largest_patch_grid,
    image_prompt_to_engine_input,
    VLLMMultimodalModelWrapper,
)
from torchtitan.rl.generator.vllm.registry import model_config_to_hf_config_dict
from torchtitan.rl.rollout.types import Prompt, PromptImage
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.transformers_utils.config import PretrainedConfig, uses_mrope

QWEN3_5_SMALL_IMAGES = dataclasses.replace(
    VisionProcessor.Config(**QWEN3_5_VISION_PROCESSOR_KWARGS),
    resize=PixelBudgetResize.Config(min_pixels=32 * 32, max_pixels=128 * 128),
)
PROCESSOR = QWEN3_5_SMALL_IMAGES.build()


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


def test_embed_input_ids_writes_image_embeddings_over_placeholders() -> None:
    """Image embeddings replace the embeddings of the placeholder tokens that vLLM's
    (CPU) mask marks, in order; the text tokens keep theirs."""
    wrapper = object.__new__(VLLMMultimodalModelWrapper)
    tok_embeddings = torch.nn.Embedding(8, 4)
    object.__setattr__(wrapper, "model", SimpleNamespace(tok_embeddings=tok_embeddings))
    object.__setattr__(
        wrapper,
        "parallelism_context",
        SimpleNamespace(activate_spmd=contextlib.nullcontext),
    )
    input_ids = torch.tensor([1, 7, 7, 2, 7, 3])
    images = [torch.full((2, 4), 10.0), torch.full((1, 4), 20.0)]

    with torch.no_grad():
        embeds = wrapper.embed_input_ids(
            input_ids, images, is_multimodal=input_ids == 7
        )
        expected = tok_embeddings(input_ids)
    expected[[1, 2]] = 10.0
    expected[4] = 20.0
    assert torch.equal(embeds, expected)


def _image_prompt(image_paths) -> Prompt:
    """Each image's placeholders, 2 tokens in and 3 tokens apart, with 2 tokens after."""
    images, placeholders, offset = [], [], 2
    for path in image_paths:
        data = Path(path).read_bytes()
        width, height = Image.open(path).size
        grid_thw = PROCESSOR.grid_thw(height, width)
        length = PROCESSOR.num_placeholder_tokens(grid_thw)
        images.append(
            PromptImage(
                hash_value=hash_image_file(data),
                source_path=path,
                grid_thw=grid_thw,
                data=data,
            )
        )
        placeholders.append(PlaceholderRange(offset=offset, length=length))
        offset += length + 3
    return Prompt(
        token_ids=list(range(offset - 1)),
        images=tuple(images),
        image_placeholders=tuple(placeholders),
    )


def test_image_prompt_places_each_image_on_its_placeholders(image_paths):
    # The mm hash, salt and pixels are checked through the engine in test_generator.
    prompt = _image_prompt(image_paths)
    images = prompt.images
    pixel_values = [
        PROCESSOR.patchify(PROCESSOR.resize_and_normalize(image.data)[0])[0]
        for image in images
    ]

    engine_input = image_prompt_to_engine_input(prompt, pixel_values)

    assert [(p.offset, p.length) for p in engine_input["mm_placeholders"]["image"]] == [
        (p.offset, p.length) for p in prompt.image_placeholders
    ]
    for image, image_pixel_values, item in zip(
        images, pixel_values, engine_input["mm_kwargs"]["image"], strict=True
    ):
        assert item["image_grid_thw"].data.tolist() == list(image.grid_thw)
        assert item["pixel_values"].data is image_pixel_values
        # The grid the renderer computed from the header matches the decoded image's.
        assert image_pixel_values.shape[0] == math.prod(image.grid_thw)


def test_mrope_input_positions_follow_placeholders():
    # 2 text tokens, a 1 x 4 x 4 image (merge 2: 2 x 2 = 4 tokens), 2 text tokens.
    image = PromptImage(
        hash_value="k", source_path="k.png", grid_thw=(1, 4, 4), data=b"k"
    )
    prompt = image_prompt_to_engine_input(
        Prompt(
            token_ids=[0] * 8,
            images=(image,),
            image_placeholders=(PlaceholderRange(offset=2, length=4),),
        ),
        [torch.zeros(16, 1536)],
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
    wrapper = SimpleNamespace(_vision_processor=PROCESSOR)

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


def test_largest_patch_grid() -> None:
    # 128 x 128 pixels in 32-pixel merged tokens: at most 4 x 4 = 16 tokens.
    assert _largest_patch_grid(PROCESSOR, 10_000) == (1, 8, 8)
    assert _largest_patch_grid(PROCESSOR, 10) == (1, 6, 6)
    with pytest.raises(ValueError, match="fits"):
        _largest_patch_grid(PROCESSOR, 0)
    # Not always square: 1,003,520 pixels are exactly 28 x 35 = 980 tokens of 32 x 32
    # pixels; the largest square has 31 x 31 = 961.
    non_square = dataclasses.replace(
        QWEN3_5_SMALL_IMAGES,
        resize=PixelBudgetResize.Config(min_pixels=32 * 32, max_pixels=1_003_520),
    )
    assert _largest_patch_grid(non_square.build(), 10_000) == (1, 56, 70)
    # NaViT: 256 patches at most, 16 x 16 square.
    navit = VisionProcessor.Config(
        patch_size=14,
        temporal_patch_size=1,
        spatial_merge_size=2,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        resize=NaViTPatchGridResize.Config(max_patches=256, max_patches_per_side=16),
        patch_order="raster",
    )
    assert _largest_patch_grid(navit.build(), 10_000) == (1, 16, 16)


def test_profiling_uses_the_vision_processors_largest_grid():
    info = _ImageProcessingInfo(
        SimpleNamespace(model_config=SimpleNamespace(max_model_len=4096)),
        vision_processor=PROCESSOR,
    )
    largest = _largest_patch_grid(PROCESSOR, 4096)
    num_tokens = PROCESSOR.num_placeholder_tokens(largest)

    assert info.largest_grid(4096) == largest
    assert info.get_mm_max_tokens_per_item(4096, {"image": 1}) == {"image": num_tokens}

    processor = _ImageMultiModalProcessor.__new__(_ImageMultiModalProcessor)
    processor.info = info
    inputs = SimpleNamespace(
        mm_data_items=SimpleNamespace(get_count=lambda modality, strict: 2)
    )
    dummy = processor.apply(inputs, timing_ctx=None)

    [first, second] = dummy["mm_kwargs"]["image"]
    assert first["image_grid_thw"].data.tolist() == list(largest)
    config = PROCESSOR.config
    assert first["pixel_values"].data.shape == (
        math.prod(largest),
        config.temporal_patch_size * config.patch_size**2 * 3,
    )
    assert len(dummy["prompt_token_ids"]) == 2 * num_tokens
    assert dummy["mm_hashes"]["image"][0] != dummy["mm_hashes"]["image"][1]


def test_hf_config_declares_mrope_only_for_image_prompts():
    config = build_model_config(
        "0.8B_mm", seq_len=256, attn_backend="varlen", parallelism=ParallelismConfig()
    )

    text = PretrainedConfig.from_dict(model_config_to_hf_config_dict(config))
    image = PretrainedConfig.from_dict(
        model_config_to_hf_config_dict(config, serves_images=True)
    )

    assert not uses_mrope(text)
    assert uses_mrope(image)
    assert image.rope_parameters["mrope_section"] == list(
        config.first_full_attention.rope.mrope_section
    )


def test_controller_rejects_a_vision_processor_without_a_vision_encoder():
    from torchtitan_recipes.tests.rl.alphabet_sort import rl_grpo_qwen3_5_debug_varlen

    config = rl_grpo_qwen3_5_debug_varlen(seq_len=128)
    without_encoder = dataclasses.replace(config.generator.model, vision_encoder=None)
    with pytest.raises(ValueError, match="multimodal model with a vision encoder"):
        dataclasses.replace(
            config,
            generator=dataclasses.replace(config.generator, model=without_encoder),
            trainer=dataclasses.replace(
                config.trainer,
                model=dataclasses.replace(config.trainer.model, vision_encoder=None),
            ),
            vision_processor=VisionProcessor.Config(**QWEN3_5_VISION_PROCESSOR_KWARGS),
        )
