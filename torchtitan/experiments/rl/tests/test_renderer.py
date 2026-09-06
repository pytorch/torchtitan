# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal

import pytest
from renderers import (
    AutoRendererConfig,
    DefaultRendererConfig,
    OffsetTokenizer,
    Qwen3RendererConfig,
    Tokenizer,
)

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.experiments.rl.renderer import (
    build_renderer,
    RendererTokenizer,
    TorchTitanRendererConfig,
)

_TOKENIZER_PATH = "tests/assets/tokenizer"


# --- build_renderer ---


def test_build_renders_with_titan_tokenizer() -> None:
    tokenizer = HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH)
    renderer = build_renderer(
        tokenizer=tokenizer, config=Qwen3RendererConfig(enable_thinking=False)
    )
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
    ("config", "reason"),
    [
        (AutoRendererConfig(), "MODEL_RENDERER_MAP"),
        (DefaultRendererConfig(), "special-token variables"),
    ],
)
def test_auto_and_default_are_refused(config, reason: str) -> None:
    tokenizer = HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH)
    with pytest.raises(ValueError) as error:
        build_renderer(tokenizer=tokenizer, config=config)
    assert reason in str(error.value)
    assert "Pick the model's renderer" in str(error.value)


def test_build_uses_the_torchtitan_renderer_class() -> None:
    class _FakeRenderer:
        def __init__(self, tokenizer, config):
            self.tokenizer, self.config = tokenizer, config

    class _FakeConfig(TorchTitanRendererConfig):
        name: Literal["fake"] = "fake"
        renderer_cls = _FakeRenderer

    tokenizer = HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH)
    renderer = build_renderer(tokenizer=tokenizer, config=_FakeConfig())
    assert isinstance(renderer, _FakeRenderer)
    assert isinstance(renderer.tokenizer, RendererTokenizer)
    assert isinstance(renderer.config, _FakeConfig)


# --- RendererTokenizer ---


def test_renderer_tokenizer_satisfies_offset_protocol() -> None:
    tokenizer = HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH)
    renderer_tokenizer = RendererTokenizer(tokenizer)
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
    assert tokenizer.bos_id not in RendererTokenizer(tokenizer).encode("hi")


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
        RendererTokenizer(HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH)),
        config,
    )
    expected = hf.render(messages, add_generation_prompt=True)
    actual = titan.render(messages, add_generation_prompt=True)
    assert actual.token_ids == expected.token_ids
    assert actual.is_content == expected.is_content
    assert actual.sampled_mask == expected.sampled_mask
    assert actual.message_indices == expected.message_indices
