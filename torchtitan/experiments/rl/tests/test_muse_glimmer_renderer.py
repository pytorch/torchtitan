# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parity between the native Muse Glimmer renderer and the published chat template.

The renderer builds harmony/ATEM prompts in Python instead of calling
``apply_chat_template``. These tests are the spec for that: every case asserts the
native render is byte-identical to what the checkpoint's own ``chat_template.jinja``
produces, so a template change upstream fails here first.

Needs a Muse Glimmer tokenizer; point ``MUSE_GLIMMER_TOKENIZER`` at a local checkpoint
directory or leave it unset to pull the public one from the Hub.

  pytest torchtitan/experiments/rl/tests/test_muse_glimmer_renderer.py -v
"""

from __future__ import annotations

import os

import pytest

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.experiments.rl.models.muse_glimmer.renderer import (
    EOM_ID,
    EOT_ID,
    MESSAGE_ID,
    MuseGlimmerRenderer,
    MuseGlimmerRendererConfig,
    START_ID,
)

DEFAULT_TOKENIZER = "meta-models/Muse-Glimmer-30B"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Run a web search",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "topk": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    }
]
# Namespaced names exercise the "# Valid recipients" and "// Tool metadata" grouping.
NAMESPACED_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web.search",
            "description": 'Search with a "quoted" phrase',
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web.fetch",
            "description": "Fetch a URL",
            "parameters": {"type": "object", "properties": {"url": {"type": "string"}}},
        },
    },
]


def _tool_call(name="search", **arguments):
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


# (id, messages, tools) -- each rendered both with and without a generation prompt.
CASES = [
    ("user_only", [{"role": "user", "content": "who wrote Blade Runner"}], None),
    ("user_with_tools", [{"role": "user", "content": "who wrote Blade Runner"}], TOOLS),
    (
        "namespaced_tools",
        [{"role": "user", "content": "search please"}],
        NAMESPACED_TOOLS,
    ),
    (
        "explicit_system",
        [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "hi"},
        ],
        TOOLS,
    ),
    (
        "system_declaring_reasoning",
        [
            {"role": "system", "content": "Be terse.\n\nReasoning strength: low."},
            {"role": "user", "content": "hi"},
        ],
        None,
    ),
    (
        "system_saying_reasoning_effort",
        [
            {"role": "system", "content": "Be terse. Reasoning effort: low."},
            {"role": "user", "content": "hi"},
        ],
        None,
    ),
    (
        "assistant_answer",
        [
            {"role": "user", "content": "who wrote Blade Runner"},
            {"role": "assistant", "content": "Philip K. Dick"},
        ],
        None,
    ),
    (
        "assistant_reasoning_then_answer",
        [
            {"role": "user", "content": "2+2?"},
            {
                "role": "assistant",
                "reasoning_content": "simple arithmetic",
                "content": "4",
            },
        ],
        None,
    ),
    (
        "tool_call_then_output",
        [
            {"role": "user", "content": "who wrote Blade Runner"},
            {
                "role": "assistant",
                "reasoning_content": "I should search.",
                "tool_calls": [_tool_call(query="Blade Runner author", topk=3)],
            },
            {"role": "tool", "name": "search", "content": "Philip K. Dick"},
            {"role": "assistant", "content": "Philip K. Dick"},
        ],
        TOOLS,
    ),
    (
        "parallel_tool_calls",
        [
            {"role": "user", "content": "two things"},
            {
                "role": "assistant",
                "tool_calls": [_tool_call(query="a"), _tool_call(query="b")],
            },
        ],
        TOOLS,
    ),
    (
        "tool_args_coercion",
        [
            {"role": "user", "content": "coerce"},
            {
                "role": "assistant",
                "tool_calls": [
                    _tool_call(
                        flag=True,
                        off=False,
                        nothing=None,
                        obj={"b": 1, "a": 2},
                        arr=[1, "x"],
                        text="plain",
                    )
                ],
            },
        ],
        TOOLS,
    ),
    (
        "consecutive_same_role",
        [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ],
        None,
    ),
    (
        "unicode_content",
        [{"role": "user", "content": "café naïve 你好"}],
        None,
    ),
    (
        "multimodal_parts",
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe "},
                    {"type": "image"},
                    {"type": "text", "text": " please"},
                ],
            }
        ],
        None,
    ),
]

CONFIGS = [
    ("defaults", {}),
    ("low_reasoning", {"reasoning_strength": "low"}),
    ("pinned_dates", {"knowledge_cutoff": "2025-01-01", "current_date": "2026-02-03"}),
]


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    path = os.environ.get("MUSE_GLIMMER_TOKENIZER", DEFAULT_TOKENIZER)
    try:
        tok = transformers.AutoTokenizer.from_pretrained(path)
    except Exception as exc:  # offline, or no access to the Hub
        pytest.skip(f"Muse Glimmer tokenizer unavailable at {path!r}: {exc}")
    if not tok.chat_template:
        pytest.skip(f"tokenizer at {path!r} ships no chat template to compare against")
    return tok


def _renderer(tokenizer, **overrides):
    return MuseGlimmerRenderer(tokenizer, MuseGlimmerRendererConfig(**overrides))


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"thinking_retention": "everything"}, ValueError),
        ({"retain_reasoning_in_history": "false"}, TypeError),
        ({"answer_from_reasoning_fallback": 1}, TypeError),
        ({"reasoning_strength": 123}, TypeError),
    ],
)
def test_config_rejects_invalid_values(kwargs, error):
    with pytest.raises(error):
        MuseGlimmerRendererConfig(**kwargs)


def test_build_snapshots_config():
    path = os.environ.get("MUSE_GLIMMER_TOKENIZER", DEFAULT_TOKENIZER)
    if not os.path.isdir(path):
        pytest.skip("HuggingFaceTokenizer needs a local tokenizer directory")
    config = MuseGlimmerRendererConfig(retain_reasoning_in_history=True)
    renderer = config.build(tokenizer=HuggingFaceTokenizer(tokenizer_path=path))
    config.retain_reasoning_in_history = False
    assert renderer._config.retain_reasoning_in_history is True
    assert renderer.effective_thinking_retention == "all"


def test_config_build_matches_hf_tokenizer_path(tokenizer):
    path = os.environ.get("MUSE_GLIMMER_TOKENIZER", DEFAULT_TOKENIZER)
    if not os.path.isdir(path):
        pytest.skip("HuggingFaceTokenizer needs a local tokenizer directory")
    titan = MuseGlimmerRendererConfig(reasoning_strength="low").build(
        tokenizer=HuggingFaceTokenizer(tokenizer_path=path)
    )
    assert isinstance(titan, MuseGlimmerRenderer)
    hf = _renderer(tokenizer, reasoning_strength="low")
    messages = [{"role": "user", "content": "search for bob"}]
    assert titan.render_ids(
        messages, tools=TOOLS, add_generation_prompt=True
    ) == hf.render_ids(messages, tools=TOOLS, add_generation_prompt=True)


def _template_kwargs(config: MuseGlimmerRendererConfig) -> dict:
    """Only forward knobs the caller set, so the template applies its own defaults."""
    return {
        name: value
        for name, value in (
            ("reasoning_strength", config.reasoning_strength),
            ("knowledge_cutoff", config.knowledge_cutoff),
            ("current_date", config.current_date),
        )
        if value is not None
    }


@pytest.mark.parametrize("config_id,overrides", CONFIGS, ids=[c[0] for c in CONFIGS])
@pytest.mark.parametrize("add_generation_prompt", [False, True])
@pytest.mark.parametrize("case_id,messages,tools", CASES, ids=[c[0] for c in CASES])
def test_matches_chat_template(
    tokenizer, case_id, messages, tools, add_generation_prompt, config_id, overrides
):
    """The native render is byte-identical to apply_chat_template."""
    renderer = _renderer(tokenizer, **overrides)
    # current_date defaults to today in both implementations; pin it so a run that
    # straddles midnight cannot produce a spurious mismatch.
    config = renderer._config
    kwargs = _template_kwargs(config)
    if config.current_date is None:
        import datetime

        kwargs["current_date"] = datetime.datetime.now().strftime("%Y-%m-%d")
        renderer = _renderer(
            tokenizer, **overrides, current_date=kwargs["current_date"]
        )

    expected = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
        **kwargs,
    )
    actual = renderer._render_text(
        messages, tools=tools, add_generation_prompt=add_generation_prompt
    )
    assert actual == expected


@pytest.mark.parametrize("case_id,messages,tools", CASES, ids=[c[0] for c in CASES])
def test_render_ids_match_encoding_whole_string(tokenizer, case_id, messages, tools):
    """Per-span encoding concatenates to the same ids as encoding the text at once.

    This is what makes the loss mask trustworthy: splitting on special tokens must not
    change tokenization.
    """
    renderer = _renderer(tokenizer, current_date="2026-02-03")
    text = renderer._render_text(messages, tools=tools, add_generation_prompt=True)
    assert renderer.render_ids(
        messages, tools=tools, add_generation_prompt=True
    ) == tokenizer.encode(text, add_special_tokens=False)


@pytest.mark.parametrize("case_id,messages,tools", CASES, ids=[c[0] for c in CASES])
def test_render_mask_aligns_with_ids(tokenizer, case_id, messages, tools):
    """render() returns one mask entry per token and the same ids as render_ids()."""
    renderer = _renderer(tokenizer, current_date="2026-02-03")
    rendered = renderer.render(messages, tools=tools, add_generation_prompt=False)
    assert rendered.token_ids == renderer.render_ids(
        messages, tools=tools, add_generation_prompt=False
    )
    n = len(rendered.token_ids)
    assert len(rendered.sampled_mask) == n
    assert len(rendered.message_indices) == n
    assert len(rendered.is_content) == n


def test_mask_excludes_scaffolding_and_covers_assistant_body(tokenizer):
    """Scaffolding is never trained on; the assistant body and terminator are."""
    renderer = _renderer(tokenizer, current_date="2026-02-03")
    messages = [
        {"role": "user", "content": "who wrote Blade Runner"},
        {"role": "assistant", "content": "Philip K. Dick"},
    ]
    rendered = renderer.render(messages)
    sampled = [
        tid for tid, keep in zip(rendered.token_ids, rendered.sampled_mask) if keep
    ]

    assert sampled, "the assistant turn should contribute trainable tokens"
    # Header scaffolding must never be trainable.
    assert START_ID not in sampled
    assert MESSAGE_ID not in sampled
    # The terminator the model itself emits is trainable.
    assert sampled[-1] == EOT_ID
    # The body round-trips to the assistant text.
    assert "Philip K. Dick" in tokenizer.decode(sampled)
    # Nothing before the assistant turn is trainable.
    first = rendered.sampled_mask.index(True)
    assert not any(rendered.sampled_mask[:first])


def test_is_content_covers_non_assistant_bodies(tokenizer):
    """is_content marks caller-supplied bodies on every role, not just assistant.

    It is deliberately NOT a copy of sampled_mask: a user or tool message's body is
    content the caller supplied even though the model never sampled it. Only header
    scaffolding is excluded on every role.
    """
    renderer = _renderer(tokenizer, current_date="2026-02-03")
    messages = [
        {"role": "user", "content": "who wrote Blade Runner"},
        {
            "role": "assistant",
            "tool_calls": [_tool_call(query="Blade Runner author")],
        },
        {"role": "tool", "name": "search", "content": "Philip K. Dick wrote it"},
        {"role": "assistant", "content": "Philip K. Dick"},
    ]
    rendered = renderer.render(messages, tools=TOOLS)
    content = [
        tid for tid, keep in zip(rendered.token_ids, rendered.is_content) if keep
    ]
    text = tokenizer.decode(content)

    # Bodies from every role are present...
    assert "who wrote Blade Runner" in text
    assert "Philip K. Dick wrote it" in text
    # ...and header scaffolding is not content on any role.
    assert START_ID not in content
    assert MESSAGE_ID not in content
    # is_content must differ from sampled_mask now (non-assistant bodies included).
    assert rendered.is_content != rendered.sampled_mask
    # Every sampled token is still content (assistant bodies + their terminators).
    assert all(c for c, s in zip(rendered.is_content, rendered.sampled_mask) if s)


def test_bridge_declines_when_retention_requires_rerender(tokenizer):
    """thinking_retention='tool_cycle' must stop the bridge at a new user query.

    The default is 'all' (the template renders prior reasoning unconditionally), so
    the bridge extends; an explicit tool_cycle policy has to decline instead.
    """
    messages = [{"role": "user", "content": "q"}]
    keep_all = _renderer(tokenizer, current_date="2026-02-03")
    assert keep_all.effective_thinking_retention == "all"

    prompt_ids = keep_all.render_ids(messages, add_generation_prompt=True)
    new_user = [{"role": "user", "content": "a second question"}]
    tool_only = [{"role": "tool", "name": "search", "content": "x"}]

    # Default policy: a new user query is still bridgeable.
    assert keep_all.bridge_to_next_turn(prompt_ids, [EOT_ID], new_user) is not None

    cycle = _renderer(
        tokenizer, current_date="2026-02-03", thinking_retention="tool_cycle"
    )
    assert cycle.effective_thinking_retention == "tool_cycle"
    # A new user query crosses the boundary -> must re-render.
    assert cycle.bridge_to_next_turn(prompt_ids, [EOT_ID], new_user) is None
    # Staying inside the tool cycle is still fine.
    assert cycle.bridge_to_next_turn(prompt_ids, [EOT_ID], tool_only) is not None


def test_reasoning_channel_is_trainable_and_ends_with_eom(tokenizer):
    """A to=self channel is sampled and closes with <|eom|>, not <|eot|>."""
    renderer = _renderer(tokenizer, current_date="2026-02-03")
    rendered = renderer.render(
        [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "reasoning_content": "arithmetic", "content": "4"},
        ]
    )
    sampled = [
        tid for tid, keep in zip(rendered.token_ids, rendered.sampled_mask) if keep
    ]
    assert EOM_ID in sampled, "the reasoning channel terminator should be trainable"
    assert sampled[-1] == EOT_ID


def test_drop_reasoning_history(tokenizer):
    """retain_reasoning_in_history=False removes the to=self channel from history."""
    messages = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "reasoning_content": "arithmetic", "content": "4"},
    ]
    kept = _renderer(tokenizer, current_date="2026-02-03")._render_text(messages)
    dropped = _renderer(
        tokenizer, current_date="2026-02-03", retain_reasoning_in_history=False
    )._render_text(messages)
    assert "to=self" in kept
    assert "to=self" not in dropped


def test_bridge_extends_without_retokenizing_the_completion(tokenizer):
    """The bridge preserves sampled ids verbatim and appends the new turn."""
    renderer = _renderer(tokenizer, current_date="2026-02-03")
    prompt_ids = renderer.render_ids(
        [{"role": "user", "content": "who wrote Blade Runner"}],
        tools=TOOLS,
        add_generation_prompt=True,
    )
    completion_ids = tokenizer.encode(
        ' to=search<|message|><atem:function_calls>\n<atem:invoke name="search">\n'
        '<atem:parameter name="query">Blade Runner</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls><|eot|>",
        add_special_tokens=False,
    )
    bridged = renderer.bridge_to_next_turn(
        prompt_ids,
        completion_ids,
        [{"role": "tool", "name": "search", "content": "Philip K. Dick"}],
        tools=TOOLS,
    )
    assert bridged is not None
    # The caller reads `.token_ids`, so the bridge must return RenderedTokens, not a list.
    carried = len(prompt_ids) + len(completion_ids)
    assert bridged.token_ids[:carried] == prompt_ids + completion_ids
    suffix = tokenizer.decode(bridged.token_ids[carried:])
    assert suffix.startswith("<|start|>tool search<|message|>")
    assert suffix.endswith("<|start|>assistant")
    # The bridge must not re-emit the system preamble already in prompt_ids.
    assert "Valid recipients" not in suffix

    # A bridge produces a prompt, so nothing in it is trainable, and every parallel
    # array must line up with token_ids or the trainer will index past the end.
    n = len(bridged.token_ids)
    assert len(bridged.sampled_mask) == n
    assert len(bridged.message_indices) == n
    assert len(bridged.is_content) == n
    assert not any(bridged.sampled_mask)
    # -1 over the carried prefix and the trailing generation prompt; 0 for new_messages[0].
    assert bridged.message_indices[:carried] == [-1] * carried
    assert set(bridged.message_indices[carried:]) <= {0, -1}
    assert bridged.message_indices[-1] == -1
    assert bridged.message_roles == ["tool"]


def test_bridge_returns_none_when_it_cannot_extend_safely(tokenizer):
    """Cases the bridge must decline so the caller re-renders instead."""
    renderer = _renderer(tokenizer, current_date="2026-02-03")
    prompt_ids = renderer.render_ids(
        [{"role": "user", "content": "q"}], add_generation_prompt=True
    )
    tool_msg = [{"role": "tool", "name": "search", "content": "x"}]

    assert renderer.bridge_to_next_turn([], [EOT_ID], tool_msg) is None
    assert renderer.bridge_to_next_turn(prompt_ids, [EOT_ID], []) is None
    # An assistant turn's terminator depends on the *next* message's role, which the
    # bridge cannot see, so it must decline rather than guess.
    assert (
        renderer.bridge_to_next_turn(
            prompt_ids, [EOT_ID], [{"role": "assistant", "content": "hi"}]
        )
        is None
    )


def test_bridge_synthesizes_a_close_for_a_truncated_completion(tokenizer):
    """A completion cut off at max_tokens has no terminator; the bridge adds one."""
    renderer = _renderer(tokenizer, current_date="2026-02-03")
    prompt_ids = renderer.render_ids(
        [{"role": "user", "content": "q"}], add_generation_prompt=True
    )
    truncated = tokenizer.encode(
        " to=user<|message|>cut off mid", add_special_tokens=False
    )
    bridged = renderer.bridge_to_next_turn(
        prompt_ids, truncated, [{"role": "tool", "name": "search", "content": "x"}]
    )
    assert bridged is not None
    carried = len(prompt_ids) + len(truncated)
    assert bridged.token_ids[:carried] == prompt_ids + truncated
    assert bridged.token_ids[carried] == EOT_ID  # synthesized turn close


def test_rejects_incompatible_tokenizer(tokenizer):
    """Special-token ids are load-bearing, so a mismatch fails loudly."""

    class Shifted:
        def __init__(self, inner):
            self._inner = inner
            self.bos_token = inner.bos_token
            self.bos_token_id = inner.bos_token_id

        def convert_tokens_to_ids(self, token):
            return 1 + self._inner.convert_tokens_to_ids(token)

    with pytest.raises(ValueError, match="not compatible"):
        MuseGlimmerRenderer(Shifted(tokenizer), MuseGlimmerRendererConfig())
