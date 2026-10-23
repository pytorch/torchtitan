# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3 XTML rendering and parsing with a byte-level stand-in tokenizer.

Token-level equality with the checkpoint's ``apply_chat_template`` is checked by
``scripts/checkpoint_conversion/convert_kimi_k3_tokenizer.py``, which needs the
released tokenizer; these tests pin the message structure without it.
"""

import pytest

from torchtitan.models.kimi_k3.renderer import KimiK3Renderer, KimiK3RendererConfig

_SPECIAL_IDS = {
    "<|open|>": 1000,
    "<|close|>": 1001,
    "<|sep|>": 1002,
    "<|end_of_msg|>": 1003,
    "[EOS]": 1004,
}
_EFFORT_TEXT = (
    "`thinking_effort` guides on how much to think in your thinking channel (not "
    "including the response channel), supported values include `low`, `medium`, "
    "`high`, and `max`.\nNow the system is invoked with `thinking_effort={}`."
)


class _ByteTokenizer:
    """Encodes text as UTF-8 bytes and the K3 chat markers as fixed ids."""

    unk_token_id = None

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return _SPECIAL_IDS.get(token)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode())

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        markers = {token_id: token for token, token_id in _SPECIAL_IDS.items()}
        text, pending = [], bytearray()
        for token_id in token_ids:
            if token_id in markers:
                text.append(pending.decode())
                pending.clear()
                text.append(markers[token_id])
            else:
                pending.append(token_id)
        text.append(pending.decode())
        return "".join(text)


def _message(attrs: str, body: str) -> str:
    return f"<|open|>message {attrs}<|sep|>{body}<|close|>message<|sep|><|end_of_msg|>"


def _render(messages, **config) -> str:
    tokenizer = _ByteTokenizer()
    renderer = KimiK3Renderer(tokenizer, KimiK3RendererConfig(**config))
    return tokenizer.decode(renderer.render_ids(messages, add_generation_prompt=True))


def test_thinking_prompt_follows_the_xtml_layout():
    rendered = _render(
        [
            {"role": "system", "content": "Be careful."},
            {"role": "user", "content": "1+1?"},
            {"role": "assistant", "content": "2", "reasoning_content": "Add."},
            {"role": "user", "content": "2+2?"},
        ],
        thinking=True,
        thinking_effort="low",
    )

    assert rendered == (
        _message('role="system" type="thinking-effort"', _EFFORT_TEXT.format("low"))
        + _message('role="system"', "Be careful.")
        + _message('role="user"', "1+1?")
        + _message(
            'role="assistant"',
            "<|open|>think<|sep|>Add.<|close|>think<|sep|>"
            "<|open|>response<|sep|>2<|close|>response<|sep|>",
        )
        + _message('role="user"', "2+2?")
        + '<|open|>message role="assistant"<|sep|><|open|>think<|sep|>'
    )


def test_non_thinking_prompt_opens_the_response_channel():
    rendered = _render(
        [
            {"role": "user", "content": "1+1?"},
            {"role": "assistant", "content": "2", "reasoning_content": "Add."},
            {"role": "user", "content": "2+2?"},
        ],
        thinking=False,
    )

    # No effort message and no think channel, even for an assistant turn with
    # reasoning.
    assert rendered == (
        _message('role="user"', "1+1?")
        + _message(
            'role="assistant"', "<|open|>response<|sep|>2<|close|>response<|sep|>"
        )
        + _message('role="user"', "2+2?")
        + '<|open|>message role="assistant"<|sep|><|open|>response<|sep|>'
    )


def test_only_assistant_output_is_marked_sampled():
    tokenizer = _ByteTokenizer()
    renderer = KimiK3Renderer(tokenizer, KimiK3RendererConfig(thinking_effort=None))
    messages = [
        {"role": "user", "content": "1+1?"},
        {"role": "assistant", "content": "2", "reasoning_content": "Add."},
    ]

    rendered = renderer.render(messages)

    sampled = [
        token
        for token, is_sampled in zip(rendered.token_ids, rendered.sampled_mask)
        if is_sampled
    ]
    # Everything after the assistant role header, through its end-of-message.
    assert tokenizer.decode(sampled) == (
        "<|open|>think<|sep|>Add.<|close|>think<|sep|>"
        "<|open|>response<|sep|>2<|close|>response<|sep|>"
        "<|close|>message<|sep|><|end_of_msg|>"
    )
    assert rendered.message_roles == ["user", "assistant"]


def test_parse_response_splits_reasoning_from_the_answer():
    tokenizer = _ByteTokenizer()
    renderer = KimiK3Renderer(tokenizer, KimiK3RendererConfig())
    completion = tokenizer.encode("x is 2.")
    completion += [_SPECIAL_IDS["<|close|>"], *tokenizer.encode("think")]
    completion += [_SPECIAL_IDS["<|sep|>"], _SPECIAL_IDS["<|open|>"]]
    completion += [*tokenizer.encode("response"), _SPECIAL_IDS["<|sep|>"]]
    completion += [*tokenizer.encode("\\boxed{2}"), _SPECIAL_IDS["<|close|>"]]
    completion += [*tokenizer.encode("response"), _SPECIAL_IDS["<|sep|>"]]
    completion += [_SPECIAL_IDS["<|close|>"], *tokenizer.encode("message")]
    completion += [_SPECIAL_IDS["<|sep|>"], _SPECIAL_IDS["<|end_of_msg|>"]]

    parsed = renderer.parse_response(completion)

    assert parsed.reasoning_content == "x is 2."
    assert parsed.content == "\\boxed{2}"
    assert renderer.get_stop_token_ids() == [
        _SPECIAL_IDS["<|end_of_msg|>"],
        _SPECIAL_IDS["[EOS]"],
    ]
    # A completion cut off while thinking keeps its reasoning and has no answer.
    truncated = renderer.parse_response(tokenizer.encode("still thinking"))
    assert truncated.reasoning_content == "still thinking"
    assert truncated.content == ""


def test_tools_are_rejected():
    renderer = KimiK3Renderer(_ByteTokenizer(), KimiK3RendererConfig())

    with pytest.raises(ValueError, match="tools"):
        renderer.render_ids([{"role": "user", "content": "hi"}], tools=[{}])
