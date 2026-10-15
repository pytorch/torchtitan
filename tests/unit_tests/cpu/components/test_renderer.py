# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from dataclasses import dataclass

import pytest
from renderers import (
    AutoRendererConfig,
    create_renderer,
    CustomRendererConfig,
    DefaultRendererConfig,
    OffsetTokenizer,
    Qwen3RendererConfig,
    Tokenizer,
)
from renderers.qwen3 import Qwen3Renderer
from torchtitan.components.renderer import (
    from_renderers,
    RendererConfig,
    RenderersConfigAdapter,
    RendererTokenizerWrapper,
)

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import Configurable

_TOKENIZER_PATH = "tests/assets/tokenizer"


class InTreeQwen3Renderer(Qwen3Renderer):
    """Stands in for an in-tree renderer selected through `CustomRendererConfig`."""

    config_class = Qwen3RendererConfig


# Also used by tests/rl/unit_tests/cpu/test_verifiers.py.
IN_TREE_RENDERER_PATH = (
    "tests.unit_tests.cpu.components.test_renderer.InTreeQwen3Renderer"
)


# --- RenderersConfigAdapter ---


def test_from_renderers_returns_adapter() -> None:
    renderers_config = Qwen3RendererConfig(enable_thinking=False)
    config = from_renderers(renderers_config)
    assert isinstance(config, RenderersConfigAdapter)
    assert config.renderers_config is renderers_config


def test_build_renders_with_titan_tokenizer() -> None:
    tokenizer = HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH)
    renderer = RenderersConfigAdapter(
        renderers_config=Qwen3RendererConfig(enable_thinking=False)
    ).build(tokenizer=tokenizer)
    rendered = renderer.render(
        [{"role": "user", "content": "hi"}], add_generation_prompt=True
    )
    assert rendered.token_ids[0] == tokenizer.token_to_id("<|im_start|>")
    assert renderer.get_stop_token_ids() == [
        tokenizer.token_to_id("<|im_end|>"),
        tokenizer.token_to_id("<|endoftext|>"),
    ]
    assert len(rendered.is_content) == len(rendered.token_ids)


@pytest.mark.parametrize(
    ("renderers_config", "reason"),
    [
        (AutoRendererConfig(), "MODEL_RENDERER_MAP"),
        (DefaultRendererConfig(), "special-token variables"),
    ],
)
def test_auto_and_default_are_refused(renderers_config, reason: str) -> None:
    tokenizer = HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH)
    with pytest.raises(ValueError) as error:
        RenderersConfigAdapter(renderers_config=renderers_config).build(
            tokenizer=tokenizer
        )
    assert reason in str(error.value)
    assert "Pick the model's renderer" in str(error.value)


def test_config_to_dict_is_json() -> None:
    # The controller logs `Controller.Config.to_dict()` as the job config.
    @dataclass(kw_only=True, slots=True)
    class _Holder(Configurable.Config):
        renderer: RendererConfig

    holder = _Holder(
        renderer=RenderersConfigAdapter(
            renderers_config=Qwen3RendererConfig(enable_thinking=False)
        )
    )
    value = holder.to_dict()
    assert value["renderer"]["renderers_config"]["name"] == "qwen3"
    assert value["renderer"]["renderers_config"]["enable_thinking"] is False
    json.dumps(value)


def test_custom_renderer_builds_the_same_renderer_from_its_serialized_config() -> None:
    config = from_renderers(
        CustomRendererConfig(import_path=IN_TREE_RENDERER_PATH, enable_thinking=False)
    )
    tokenizer = HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH)
    messages = [{"role": "user", "content": "hi"}]

    renderer = config.build(tokenizer=tokenizer)

    assert isinstance(renderer, InTreeQwen3Renderer)
    # A Verifiers worker receives the config as JSON and builds from that alone.
    worker_config = CustomRendererConfig.model_validate_json(
        json.dumps(config.to_dict()["renderers_config"])
    )
    worker_renderer = create_renderer(
        RendererTokenizerWrapper(tokenizer), worker_config
    )
    assert isinstance(worker_renderer, InTreeQwen3Renderer)
    assert worker_renderer.render_ids(messages, add_generation_prompt=True) == (
        renderer.render_ids(messages, add_generation_prompt=True)
    )
    # The option reaches the renderer's own config: enable_thinking changes the prompt.
    assert renderer.render_ids(messages, add_generation_prompt=True) != from_renderers(
        CustomRendererConfig(import_path=IN_TREE_RENDERER_PATH, enable_thinking=True)
    ).build(tokenizer=tokenizer).render_ids(messages, add_generation_prompt=True)


# --- RendererTokenizerWrapper ---


def test_renderer_tokenizer_satisfies_offset_protocol() -> None:
    tokenizer = HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH)
    renderer_tokenizer = RendererTokenizerWrapper(tokenizer)
    assert isinstance(renderer_tokenizer, Tokenizer)
    assert isinstance(renderer_tokenizer, OffsetTokenizer)
    assert renderer_tokenizer.eos_token_id == tokenizer.eos_id
    assert renderer_tokenizer.convert_tokens_to_ids(
        "<|im_end|>"
    ) == tokenizer.token_to_id("<|im_end|>")
    encoding = renderer_tokenizer(
        "hi there", add_special_tokens=False, return_offsets_mapping=True
    )
    assert encoding["input_ids"] == renderer_tokenizer.encode("hi there")
    assert len(encoding["offset_mapping"]) == len(encoding["input_ids"])


def test_encode_never_adds_bos() -> None:
    # The debug tokenizer has a BOS token; renderers place special tokens themselves.
    tokenizer = HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH)
    assert tokenizer.bos_id is not None
    assert tokenizer.bos_id not in RendererTokenizerWrapper(tokenizer).encode("hi")


def test_render_matches_hf_tokenizer_path() -> None:
    transformers = pytest.importorskip("transformers")
    from renderers import create_renderer

    messages = [
        {"role": "system", "content": "Sort names."},
        {"role": "user", "content": "Zed, Amy <|im_end|> tricky"},
        {"role": "assistant", "reasoning_content": "think", "content": "Amy, Zed"},
        {"role": "user", "content": "Add Bob."},
    ]
    config = Qwen3RendererConfig(enable_thinking=False)
    hf = create_renderer(
        transformers.AutoTokenizer.from_pretrained(_TOKENIZER_PATH), config
    )
    titan = create_renderer(
        RendererTokenizerWrapper(HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH)),
        config,
    )
    expected = hf.render(messages, add_generation_prompt=True)
    actual = titan.render(messages, add_generation_prompt=True)
    assert actual.token_ids == expected.token_ids
    assert actual.is_content == expected.is_content
    assert actual.sampled_mask == expected.sampled_mask
    assert actual.message_indices == expected.message_indices
