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
    assert "Reasoning strength: 32." in text
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
    "effort, strength", [("low", 8), ("medium", 32), ("high", 256), ("xhigh", 512)]
)
def test_reasoning_strength_matches_published_template(tokenizer, effort, strength):
    renderer = MuseSparkRendererConfig(
        current_date="2026-01-15", reasoning_effort=effort
    ).build(tokenizer=tokenizer)
    text = tokenizer.decode(
        renderer.render_ids([{"role": "user", "content": "hi"}]),
        skip_special_tokens=False,
    )
    assert f"Reasoning strength: {strength}." in text


def test_rejects_reasoning_effort_missing_from_published_template():
    with pytest.raises(ValueError, match="reasoning_effort"):
        MuseSparkRendererConfig(reasoning_effort="minimal")


def test_function_schemas_match_published_template(renderer, tokenizer):
    def function(name, **fields):
        return {"type": "function", "function": {"name": name, **fields}}

    tools = [
        function(
            "search",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        ),
        function(
            "open",
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            },
        ),
        function("now"),
    ]
    text = tokenizer.decode(
        renderer.render_ids([{"role": "user", "content": "hi"}], tools=tools),
        skip_special_tokens=False,
    )
    # A schema without additionalProperties gets it appended; one that has it is
    # kept; missing parameters become an empty object schema.
    assert (
        '"parameters": {"type": "object", "properties": {"q": {"type": "string"}}, '
        '"additionalProperties": false}}'
    ) in text
    assert (
        '"parameters": {"type": "object", "properties": {}, '
        '"additionalProperties": true}}'
    ) in text
    assert (
        '{"name": "now", "description": "", "parameters": {"type": "object", '
        '"properties": {}, "additionalProperties": false}}'
    ) in text


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


def _tool_call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _sampled_turn(renderer, history, assistant):
    prompt = renderer.render_ids(history, add_generation_prompt=True)
    with_assistant = renderer.render_ids([*history, assistant])
    assert with_assistant[: len(prompt)] == prompt
    return prompt, with_assistant[len(prompt) :]


def test_bridge_matches_full_render_for_canonical_completion(renderer):
    history = [{"role": "user", "content": "List the files."}]
    assistant = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "I should run ls.",
        "tool_calls": [_tool_call("c1", "bash", {"command": "ls"})],
    }
    tool = {"role": "tool", "tool_call_id": "c1", "content": "a.py"}
    prompt, completion = _sampled_turn(renderer, history, assistant)

    bridged = renderer.bridge_to_next_turn(prompt, completion, [tool])

    assert bridged is not None
    assert bridged.token_ids == renderer.render_ids(
        [*history, assistant, tool], add_generation_prompt=True
    )
    assert len(bridged.message_indices) == len(bridged.token_ids)
    assert len(bridged.is_content) == len(bridged.token_ids)
    assert not any(bridged.sampled_mask)


def test_bridge_matches_full_render_for_tool_results_and_user(renderer):
    history = [{"role": "user", "content": "Inspect two files."}]
    assistant = {
        "role": "assistant",
        "content": "Reading both.",
        "tool_calls": [
            _tool_call("c1", "read", {"path": "a.py"}),
            _tool_call("c2", "bash", {"command": "cat b.py"}),
        ],
    }
    tail = [
        {"role": "tool", "tool_call_id": "c1", "content": "print(1)"},
        {"role": "tool", "tool_call_id": "c2", "name": "bash", "content": "x = 2"},
        {"role": "user", "content": "Now summarize."},
    ]
    prompt, completion = _sampled_turn(renderer, history, assistant)

    bridged = renderer.bridge_to_next_turn(prompt, completion, tail)

    assert bridged is not None
    assert bridged.token_ids == renderer.render_ids(
        [*history, assistant, *tail], add_generation_prompt=True
    )


def test_bridge_keeps_non_canonical_completion_verbatim(renderer, tokenizer):
    history = [{"role": "user", "content": "List the files."}]
    assistant = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "run ls",
        "tool_calls": [_tool_call("c1", "bash", {"command": "ls"})],
    }
    tool = {"role": "tool", "tool_call_id": "c1", "content": "a.py"}
    prompt, canonical = _sampled_turn(renderer, history, assistant)
    # Sampled with spaces around the reasoning, which the parser strips.
    text = tokenizer.decode(canonical, skip_special_tokens=False)
    completion = tokenizer.encode(
        text.replace("<|message|>run ls<|eom|>", "<|message|>  run ls  <|eom|>")
    )
    parsed = renderer.parse_response(completion)
    rerendered = renderer.render_ids(
        [
            *history,
            {
                "role": "assistant",
                "content": parsed.content or None,
                "reasoning_content": parsed.reasoning_content,
                "tool_calls": [
                    _tool_call("c1", call.name, call.arguments)
                    for call in parsed.tool_calls
                ],
            },
            tool,
        ],
        add_generation_prompt=True,
    )

    bridged = renderer.bridge_to_next_turn(prompt, completion, [tool])

    assert bridged is not None
    sampled_turn = prompt + completion
    assert bridged.token_ids[: len(sampled_turn)] == sampled_turn
    # A re-render loses the sampled spaces, so the next prompt would not extend the
    # sampled turn.
    assert rerendered[: len(sampled_turn)] != sampled_turn


def test_bridge_falls_back_to_re_render(renderer):
    history = [{"role": "user", "content": "List the files."}]
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [_tool_call("c1", "bash", {"command": "ls"})],
    }
    tool = {"role": "tool", "tool_call_id": "c1", "content": "a.py"}
    prompt, completion = _sampled_turn(renderer, history, assistant)

    cases = {
        "no previous turn": ([], completion, [tool]),
        "truncated completion": (prompt, completion[:-1], [tool]),
        "more tool results than calls": (
            prompt,
            completion,
            [{"role": "user", "content": "?"}, tool, tool],
        ),
        "tool name disagrees": (prompt, completion, [{**tool, "name": "read"}]),
        "assistant in the tail": (prompt, completion, [assistant]),
    }
    for case, (previous_prompt, previous_completion, tail) in cases.items():
        assert (
            renderer.bridge_to_next_turn(previous_prompt, previous_completion, tail)
            is None
        ), case
