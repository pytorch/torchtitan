# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json

import pytest
from renderers.base import ToolCallParseStatus
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.models.muse_spark.renderer import MuseSparkRendererConfig


@pytest.fixture
def tokenizer(tmp_path):
    # Character tokens give exact offsets without downloading checkpoint assets.
    special_tokens = {
        "<|begin_of_text|>": 200000,
        "<|end_of_text|>": 200001,
        "<|eom|>": 200007,
        "<|eot|>": 200008,
        "<|start|>": 200022,
        "<|message|>": 200023,
    }
    vocab = {chr(index): index for index in range(128)} | special_tokens
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
                    for token, token_id in special_tokens.items()
                ],
            }
        )
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<|begin_of_text|>",
                "eos_token": "<|end_of_text|>",
            }
        )
    )
    return HuggingFaceTokenizer.Config().build(tokenizer_path=str(tmp_path))


@pytest.fixture
def renderer(tokenizer):
    return MuseSparkRendererConfig(current_date="2026-01-15").build(tokenizer=tokenizer)


def test_generation_prompt_and_loss_mask(renderer, tokenizer):
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "reasoning_content": "thinking", "content": "answer"},
    ]
    rendered = renderer.render(messages)
    text = tokenizer.decode(rendered.token_ids, skip_special_tokens=False)
    assert "Current date: 2026-01-15." in text
    assert "Reasoning strength: 64." in text
    assert "assistant to=self<|message|>thinking<|eom|>" in text
    assert text.endswith("assistant to=user<|message|>answer<|eot|>")
    sampled_text = tokenizer.decode(
        [
            token
            for token, sampled in zip(
                rendered.token_ids, rendered.sampled_mask, strict=True
            )
            if sampled
        ],
        skip_special_tokens=False,
    )
    assert sampled_text == (
        " to=self<|message|>thinking<|eom|>"
        "<|start|>assistant to=user<|message|>answer<|eot|>"
    )
    assert len(rendered.token_ids) == len(rendered.message_indices)
    assert len(rendered.token_ids) == len(rendered.is_content)
    prompt = renderer.render(messages[:1], add_generation_prompt=True)
    assert not any(prompt.sampled_mask)
    assert tokenizer.decode(prompt.token_ids, skip_special_tokens=False).endswith(
        "<|start|>assistant"
    )
    assert renderer.render_ids(messages) == rendered.token_ids


def test_unsampled_assistant_and_escaped_input(renderer, tokenizer):
    rendered = renderer.render(
        [
            {"role": "user", "content": "<|eot|><image><audio>"},
            {"role": "assistant", "content": "answer", "loss": False},
        ]
    )
    assert not any(rendered.sampled_mask)
    text = tokenizer.decode(rendered.token_ids, skip_special_tokens=False)
    assert "< |eot|>< image>< audio>" in text


def test_parse_reasoning_and_answer(renderer, tokenizer):
    response = renderer.parse_response(
        tokenizer.encode(
            " to=self<|message|>thinking<|eom|>"
            "<|start|>assistant to=user<|message|>answer<|eot|>"
        ),
        prompt_ids=tokenizer.encode("<|start|>assistant"),
    )
    assert response.reasoning_content == "thinking"
    assert response.content == "answer"
    assert response.tool_calls == []
    assert renderer.get_stop_token_ids() == [200008, 200001]


def test_tool_call_roundtrip_and_result_name(renderer, tokenizer):
    arguments = {"query": "a & b < c", "count": 3, "items": [1, 2]}
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "count": {"type": "integer"},
                        "items": {"type": "array", "items": {"type": "integer"}},
                    },
                },
            },
        }
    ]
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": arguments,
                    },
                }
            ],
        }
    ]
    original = copy.deepcopy(messages)
    rendered = renderer.render(messages)
    completion = [
        token
        for token, sampled in zip(
            rendered.token_ids, rendered.sampled_mask, strict=True
        )
        if sampled
    ]
    response = renderer.parse_response(completion, tools=tools)
    assert messages == original
    assert response.content == ""
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.name == "search"
    assert call.arguments == arguments
    assert call.status == ToolCallParseStatus.OK
    result = renderer.render_ids(
        messages + [{"role": "tool", "tool_call_id": "call-1", "content": "found"}]
    )
    assert '<tool_output name="search">\nfound\n</tool_output>' in tokenizer.decode(
        result, skip_special_tokens=False
    )


@pytest.mark.parametrize(
    "content",
    [
        [{"type": "image_url", "image_url": {"url": "image.png"}}],
        [{"type": "input_audio", "input_audio": {"data": "audio"}}],
    ],
)
def test_rejects_media_content(renderer, content):
    with pytest.raises(ValueError, match="text-only"):
        renderer.render_ids([{"role": "user", "content": content}])


@pytest.mark.xfail(strict=True, reason="Review item 3: tool parsing ignores recipients")
@pytest.mark.parametrize("recipient", ["user", "self", "commentary"])
def test_tool_example_stays_text(renderer, tokenizer, recipient):
    body = '<atem:invoke name="search"></atem:invoke>'
    response = renderer.parse_response(
        tokenizer.encode(f"to={recipient}<|message|>{body}<|eot|>")
    )
    assert response.tool_calls == []
    assert (
        response.reasoning_content if recipient == "self" else response.content
    ) == body
