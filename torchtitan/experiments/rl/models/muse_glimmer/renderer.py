# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muse Glimmer renderer: chat messages <-> tokens for TorchTitan RL.

RL needs two directions. ``render_ids`` turns messages + tool definitions into token
ids; ``parse_response`` turns generated tokens back into ``(content,
reasoning_content, tool_calls)`` so the rollout loop can tell "call a tool and
continue" apart from "final answer, score it".

Muse Glimmer uses the harmony chat format -- an assistant turn is a sequence of
``to=<recipient><|message|><body>`` channels, where ``to=self`` is private reasoning and
other recipients carry user-visible content -- and expresses tool calls as ATEM XML
inside those channels (see ``atem.py``).

The format is built natively in Python rather than by calling
``tokenizer.apply_chat_template``. That matches how the ``renderers`` library implements
every model-specific renderer (only ``DefaultRenderer`` wraps Jinja), and it buys two
things a template wrapper cannot: per-token loss attribution is exact rather than
recovered by diffing prefixes, and ``bridge_to_next_turn`` can extend a sampled
completion without re-rendering it.

``_render_text`` is kept byte-exact against the published ``chat_template.jinja``; the
renderer test asserts equality with ``apply_chat_template`` across roles, tool shapes and
reasoning states. Treat that test as the spec -- if the template changes upstream, it
fails first.

Implements the ``renderers.Renderer`` Protocol. Its config is a ``TorchTitanRendererConfig``
naming this class, so ``build_renderer`` constructs it directly instead of looking it up in
the ``renderers`` registry (Muse Glimmer is not in the library yet).

TODO: upstream this to PrimeIntellect-ai/renderers (renderer -> renderers/muse_glimmer.py,
atem.py -> a tool parser in renderers/parsers.py), then delete both files and the
``renderer_cls`` assignment at the bottom of this module.

It lives under ``experiments/rl`` rather than ``torchtitan/models/muse_glimmer`` because
RL is its only consumer and ``renderers`` is an RL-only optional dependency; keeping it
here leaves the core model package importable without it.
"""

from __future__ import annotations

import datetime
import json
import re
from typing import ClassVar, Literal, NamedTuple

from renderers.base import (
    extract_message_tool_names,
    ParsedResponse,
    ParsedToolCall,
    reject_assistant_in_extension,
    RenderedTokens,
    resolve_thinking_retention,
    should_rerender_for_thinking_retention,
    trim_to_turn_close,
)

from torchtitan.experiments.rl.renderer import TorchTitanRendererConfig

from .atem import parse_atem_tool_calls, render_atem_tool_call


class MuseGlimmerRendererConfig(TorchTitanRendererConfig):
    """Muse Glimmer (harmony chat format + ATEM tool calls) renderer config."""

    name: Literal["muse_glimmer"] = "muse_glimmer"

    # renderers validates in BaseRendererConfig.__pydantic_init_subclass__ that every
    # non-base field is classified as either a chat-template kwarg or a renderer-internal
    # knob; the two sets must be disjoint and together cover all of them. Declared
    # unconditionally -- versions without the validator ignore these ClassVars, so this
    # is compatible with both. The template fields mirror kwargs the published
    # chat_template.jinja reads, which is what the library's parity matrix varies.
    _template_fields: ClassVar[frozenset[str]] = frozenset(
        {"reasoning_strength", "knowledge_cutoff", "current_date"}
    )
    _internal_fields: ClassVar[frozenset[str]] = frozenset(
        {"retain_reasoning_in_history", "answer_from_reasoning_fallback"}
    )

    reasoning_strength: str | None = None
    """Sizes the reasoning budget, rendered as ``Reasoning strength: <value>.``

    ``None`` (default) uses the template's own default of ``"high"``. Set ``"low"`` for
    agentic tasks with a tight token budget: at high strength the model can spend the
    whole budget reasoning and get truncated before it emits a tool call or an answer,
    which scores as a failed rollout.
    """

    knowledge_cutoff: str | None = None
    """Knowledge-cutoff date in the default system prompt. ``None`` uses the template's
    own default. Only rendered when the caller supplies no system message."""

    current_date: str | None = None
    """Pins ``Current date:`` in the default system prompt.

    ``None`` reproduces the template's behaviour of substituting today's date, which
    makes the rendered prompt change from one day to the next. Pin it for runs that need
    to be reproducible across days.
    """

    retain_reasoning_in_history: bool = True
    """Whether prior assistant turns keep their ``reasoning_content`` when re-rendered.

    A ``to=self`` channel is emitted for any assistant message carrying
    ``reasoning_content``, so history grows with every turn's reasoning. Harmony-style
    models are often trained with prior analysis *dropped* from context (gpt-oss does
    this via ``auto_drop_analysis``); set this False to match that convention and to
    keep multi-turn prompts short.

    Defaults True to preserve the template's own behaviour -- flip it only if you have
    checked what the model was trained against.
    """

    answer_from_reasoning_fallback: bool = False
    """When the model produces only reasoning channels -- no user-facing content and no
    tool call -- treat the last non-empty reasoning line as the answer.

    Off by default: it promotes private reasoning to user-visible content, which is
    usually not what you want. Useful for outcome-scored RL, where a rollout with an
    empty ``content`` is unscoreable and the answer is often the final reasoning line.
    """


# Muse Glimmer special tokens. The ids are checked against the tokenizer in __init__
# rather than trusted, since they are baked into parse_response and the loss mask.
START_STR, MESSAGE_STR = "<|start|>", "<|message|>"
EOM_STR, EOT_STR = "<|eom|>", "<|eot|>"

START_ID = 200022  # <|start|> begins a harmony message header
MESSAGE_ID = 200023  # <|message|> ends the header, body follows
EOT_ID = 200008  # <|eot|> end of turn
EOM_ID = 200007  # <|eom|> end of message
EOS_ID = 200001  # <|end_of_text|>

_DEFAULT_REASONING_STRENGTH = "high"
_DEFAULT_KNOWLEDGE_CUTOFF = "2026-01-04"

_FUNCTION_CALLS_BLOCK = re.compile(
    r"<atem:function_calls>.*?</atem:function_calls>", re.DOTALL
)
# One harmony channel: "to=<recipient><|message|><body>" up to <|eom|>/<|eot|>/next channel.
_CHANNEL = re.compile(
    r"(?:to=(?P<rcpt>[^\s<|]+))?\s*<\|message\|>(?P<body>.*?)"
    r"(?=<\|eom\|>|<\|eot\|>|<\|end_of_text\|>|<\|start\|>|\Z)",
    re.DOTALL,
)

# The template normalises "reasoning effort" to "reasoning strength" in a caller-supplied
# system prompt. Jinja has no case-insensitive replace, so it spells out four casings;
# reproduce exactly those four, or renders diverge on any other casing.
_EFFORT_TO_STRENGTH = (
    ("Reasoning effort", "Reasoning strength"),
    ("Reasoning Effort", "Reasoning Strength"),
    ("reasoning effort", "reasoning strength"),
    ("REASONING EFFORT", "REASONING STRENGTH"),
)


def _tojson(value) -> str:
    """Jinja's ``tojson`` as transformers configures it: insertion order, raw unicode."""
    return json.dumps(value, ensure_ascii=False)


def _render_content(content) -> str:
    """A message body: a plain string, or a list of typed multimodal parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    out = []
    for part in content:
        kind = part.get("type")
        if kind == "image":
            out.append("<|patch|>")
        elif kind == "video":
            out.append("<|video|>")
        elif kind == "text":
            out.append(part["text"])
    return "".join(out)


def _tool_fn(tool) -> dict:
    """Tools arrive either OpenAI-nested (``{"function": {...}}``) or flat."""
    if isinstance(tool, dict) and tool.get("function") is not None:
        return tool["function"]
    return tool


def _tool_namespaces(tools) -> list[str]:
    """Leading dotted segment of each tool name, first-seen order, deduplicated."""
    seen: list[str] = []
    for tool in tools:
        namespace = _tool_fn(tool)["name"].split(".")[0]
        if namespace not in seen:
            seen.append(namespace)
    return seen


def _render_tool_defs(tools) -> str:
    """The tool-definition block injected into the system message."""
    parts = [
        "In this environment you have access to a set of tools you can use to answer "
        "the user's question.\n\n",
        'You can invoke a function by writing a "<atem:function_calls>" block like the '
        "following:\n",
        '<atem:function_calls>\n<atem:invoke name="$FUNCTION_NAME">\n'
        '<atem:parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</atem:parameter>\n'
        "...\n</atem:invoke>\n</atem:function_calls>\n\n",
        "String and scalar parameters should be specified as is, while lists and "
        "objects should use JSON format. Note that spaces for string values are not "
        "stripped. The output is not expected to be valid XML and is parsed with "
        "regular expressions.\n",
        "Here are the functions available in JSONSchema format:\n",
        "// Tool metadata\n",
    ]
    for namespace in _tool_namespaces(tools):
        parts.append(
            f'{{"name": {_tojson(namespace)}, "description": {_tojson("")}}}\n'
        )
    parts.append("// Function schemas")
    for tool in tools:
        fn = _tool_fn(tool)
        parts.append(
            f'\n{{"name": {_tojson(fn["name"])}, '
            f'"description": {_tojson(fn.get("description"))}, '
            f'"parameters": {_tojson(fn.get("parameters"))}}}'
        )
    parts.append(
        "\n\nHere's an example of how to call a function in the tool set:\n"
        "(If the tool namespace is not specified, invoke the function directly as "
        "`example_function_name` rather than "
        "`example_tool_name.example_function_name`)\n\n"
        "to=example_tool_name.example_function_name\n\n"
        "<atem:function_calls>\n"
        '<atem:invoke name="example_tool_name.example_function_name">\n'
        '<atem:parameter name="example_parameter_1">value_1</atem:parameter>\n'
        '<atem:parameter name="example_parameter_2">This is the value for the second '
        'parameter\nthat can span\n"multiple" lines\n</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls>"
    )
    return "".join(parts)


def _render_system_meta(tools) -> str:
    """The ``# Valid recipients:`` line closing every system message."""
    recipients = ['"self"']
    if tools:
        recipients += [f'"{ns}.*"' for ns in _tool_namespaces(tools)]
    recipients.append('"user"')
    return "# Valid recipients: " + ", ".join(recipients) + "."


def _render_atem(tool_call) -> str:
    """An assistant tool call as an ATEM block."""
    fn = tool_call["function"]
    args = fn.get("arguments")
    if not isinstance(args, dict):
        raise ValueError(
            "Muse Glimmer tool_call.function.arguments must be a dict, got "
            f"{type(args).__name__}. A JSON string cannot be rendered."
        )
    return render_atem_tool_call(fn["name"], args)


def _strip_reasoning_history(messages):
    """Drop ``reasoning_content`` from assistant messages (see the config field)."""
    out = []
    for m in messages:
        if (
            isinstance(m, dict)
            and m.get("role") == "assistant"
            and m.get("reasoning_content")
        ):
            m = {k: v for k, v in m.items() if k != "reasoning_content"}
        out.append(m)
    return out


def _normalize_tool_calls(messages):
    """Our ParsedToolCall is flat (.name/.arguments); rendering expects OpenAI-nested."""
    out = []
    for m in messages:
        tcs = m.get("tool_calls") if isinstance(m, dict) else None
        if not tcs:
            out.append(m)
            continue
        norm = []
        for tc in tcs:
            if isinstance(tc, dict) and "function" in tc:
                norm.append(tc)
                continue
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            args = (
                tc.get("arguments")
                if isinstance(tc, dict)
                else getattr(tc, "arguments", None)
            )
            norm.append(
                {"type": "function", "function": {"name": name, "arguments": args}}
            )
        out.append({**m, "tool_calls": norm})
    return out


def _parse_channels(text: str) -> list[tuple[str, str]]:
    """Split a Muse Glimmer assistant completion into (recipient, body) harmony channels."""
    return [
        (m.group("rcpt") or "user", m.group("body").strip())
        for m in _CHANNEL.finditer(text)
    ]


class _Piece(NamedTuple):
    """One span of the render: either a literal special token or text to encode.

    ``sampled`` marks spans the model itself produced, which is what the trainer's loss
    mask keys off. Template scaffolding (``<|start|>``, the role header, ``<|message|>``)
    is never sampled, even inside an assistant turn.

    ``is_body`` marks the message body specifically, which the bridge reports as content
    even on non-assistant roles that are never sampled. ``is_generation_prompt`` marks
    the trailing ``<|start|>assistant``, which belongs to no message.
    """

    text: str
    token_id: int | None
    sampled: bool
    msg_idx: int
    is_body: bool = False
    is_generation_prompt: bool = False


class MuseGlimmerRenderer:
    def __init__(self, tokenizer, config: MuseGlimmerRendererConfig | None = None):
        # Match the `(tokenizer, config)` constructor used by library renderers.
        self._tok = tokenizer
        self._config = config or MuseGlimmerRendererConfig()
        self._bos = tokenizer.bos_token or ""
        # BaseRendererConfig.thinking_retention is the library-wide knob every renderer
        # is expected to honour in its bridge. Muse Glimmer's published chat template
        # renders reasoning_content for every assistant turn unconditionally -- no
        # query-boundary drop like gpt-oss's auto_drop_analysis or Qwen3's think-block
        # stripping -- so "all" is the template-faithful implied policy. An explicit
        # thinking_retention on the config overrides it.
        self.effective_thinking_retention = resolve_thinking_retention(
            self._config,
            "all" if self._config.retain_reasoning_in_history else "tool_cycle",
        )
        self._verify_special_ids()

    def _verify_special_ids(self) -> None:
        """The special-token ids are baked into parse_response and the loss mask.

        A tokenizer that disagrees would mask the wrong spans and mis-split channels,
        both of which corrupt training silently, so check rather than assume.
        """
        for token, expected in (
            (START_STR, START_ID),
            (MESSAGE_STR, MESSAGE_ID),
            (EOM_STR, EOM_ID),
            (EOT_STR, EOT_ID),
        ):
            actual = self._tok.convert_tokens_to_ids(token)
            if actual != expected:
                raise ValueError(
                    f"Tokenizer maps {token} to id {actual}, but the muse_glimmer "
                    f"renderer expects {expected}. This tokenizer is not compatible."
                )

    # ---------------------------------------------------------------- rendering

    def _prepare(self, messages):
        """Apply the history policy, then the tool-call shape rendering expects."""
        if not self._config.retain_reasoning_in_history:
            messages = _strip_reasoning_history(messages)
        return _normalize_tool_calls(messages)

    def _reasoning_line(self) -> str:
        strength = self._config.reasoning_strength or _DEFAULT_REASONING_STRENGTH
        return f"Reasoning strength: {strength}."

    def _default_system_body(self, tools) -> str:
        """System block synthesised when the caller supplies no system message."""
        cutoff = self._config.knowledge_cutoff or _DEFAULT_KNOWLEDGE_CUTOFF
        date = self._config.current_date or datetime.datetime.now().strftime("%Y-%m-%d")
        body = (
            "You are a helpful AI assistant."
            f"\nKnowledge cutoff: {cutoff}."
            f"\nCurrent date: {date}."
            f"\n\n{self._reasoning_line()}"
        )
        if tools:
            body += "\n\n" + _render_tool_defs(tools)
        return body + "\n\n" + _render_system_meta(tools)

    def _explicit_system_body(self, message, tools) -> str:
        text = _render_content(message.get("content"))
        for old, new in _EFFORT_TO_STRENGTH:
            text = text.replace(old, new)
        body = text
        if "reasoning strength" not in text.lower():
            body += "\n\n" + self._reasoning_line()
        if tools:
            body += "\n\n" + _render_tool_defs(tools)
        return body + "\n\n" + _render_system_meta(tools)

    @staticmethod
    def _tool_name(message, messages) -> str:
        """Tool messages name their tool directly, or via the call id that produced them."""
        name = message.get("name")
        if name:
            return name
        call_id = message.get("tool_call_id")
        resolved = call_id if call_id else ""
        for m in messages:
            for tc in m.get("tool_calls") or ():
                if call_id is not None and tc.get("id") == call_id:
                    resolved = tc["function"]["name"]
        return resolved

    def _build(self, messages, *, tools, add_generation_prompt) -> list[_Piece]:
        """Render to spans. Concatenating ``.text`` reproduces the chat template exactly.

        One assistant message can expand into several harmony blocks -- a ``to=self``
        reasoning channel plus one per tool call -- which is why attribution is tracked
        per span rather than per message.
        """
        pieces: list[_Piece] = []

        def emit(
            text: str,
            *,
            token_id: int | None = None,
            sampled=False,
            idx=0,
            is_body=False,
            is_generation_prompt=False,
        ):
            if text:
                pieces.append(
                    _Piece(text, token_id, sampled, idx, is_body, is_generation_prompt)
                )

        def block(header: str, body: str, end: str, *, idx: int, sampled: bool):
            emit(START_STR, token_id=START_ID, idx=idx)
            emit(header, idx=idx)
            emit(MESSAGE_STR, token_id=MESSAGE_ID, idx=idx)
            emit(body, sampled=sampled, idx=idx, is_body=True)
            # The terminator counts as content only on assistant turns, where the model
            # emits its own stop token; on history roles it is template scaffolding.
            emit(
                end,
                token_id=EOT_ID if end == EOT_STR else EOM_ID,
                sampled=sampled,
                idx=idx,
                is_body=sampled,
            )

        emit(self._bos, token_id=self._tok.bos_token_id)

        if not any(m.get("role") == "system" for m in messages):
            block(
                "system",
                self._default_system_body(tools),
                EOT_STR,
                idx=0,
                sampled=False,
            )

        for i, message in enumerate(messages):
            role = message.get("role")
            # The template picks the terminator by looking at the NEXT message's role:
            # two consecutive same-role messages are joined with <|eom|>.
            same_role_next = (
                i + 1 < len(messages) and messages[i + 1].get("role") == role
            )
            end_token = EOM_STR if same_role_next else EOT_STR

            if role == "system":
                block(
                    "system",
                    self._explicit_system_body(message, tools),
                    EOT_STR,
                    idx=i,
                    sampled=False,
                )
            elif role == "user":
                block(
                    "user",
                    _render_content(message.get("content")),
                    EOT_STR,
                    idx=i,
                    sampled=False,
                )
            elif role == "tool":
                name = self._tool_name(message, messages)
                body = (
                    f'<tool_output name="{name}">\n'
                    f'{_render_content(message.get("content"))}\n</tool_output>'
                )
                block(f"tool {name}", body, EOT_STR, idx=i, sampled=False)
            elif role == "assistant":
                if message.get("reasoning_content"):
                    block(
                        "assistant to=self",
                        message["reasoning_content"],
                        EOM_STR,
                        idx=i,
                        sampled=True,
                    )
                tool_calls = message.get("tool_calls")
                if tool_calls:
                    for j, tc in enumerate(tool_calls):
                        last = j == len(tool_calls) - 1
                        block(
                            f'assistant to={tc["function"]["name"]}',
                            _render_atem(tc),
                            end_token if last else EOM_STR,
                            idx=i,
                            sampled=True,
                        )
                else:
                    recipient = message.get("recipient") or "user"
                    end_turn = message.get("end_turn")
                    if end_turn is None:
                        end_turn = recipient == "user"
                    block(
                        f"assistant to={recipient}",
                        _render_content(message.get("content")),
                        EOT_STR if end_turn else EOM_STR,
                        idx=i,
                        sampled=True,
                    )

        if add_generation_prompt:
            idx = max(len(messages) - 1, 0)
            emit(START_STR, token_id=START_ID, idx=idx, is_generation_prompt=True)
            emit("assistant", idx=idx, is_generation_prompt=True)

        return pieces

    def _render_text(self, messages, *, tools=None, add_generation_prompt=False) -> str:
        """The rendered prompt as text. Byte-exact against chat_template.jinja."""
        pieces = self._build(
            self._prepare(messages),
            tools=tools,
            add_generation_prompt=add_generation_prompt,
        )
        return "".join(p.text for p in pieces)

    def _encode(self, piece: _Piece) -> list[int]:
        if piece.token_id is not None:
            return [piece.token_id]
        return self._tok.encode(piece.text, add_special_tokens=False)

    def get_stop_token_ids(self) -> list[int]:
        # NOT <|eom|> (200007): it ends the *reasoning* channel, after which the model emits
        # the tool call / final answer. Stopping at eom truncates before the answer.
        return [EOT_ID, EOS_ID]

    def render_ids(
        self, messages, *, tools=None, add_generation_prompt: bool = False
    ) -> list[int]:
        pieces = self._build(
            self._prepare(messages),
            tools=tools,
            add_generation_prompt=add_generation_prompt,
        )
        ids: list[int] = []
        for piece in pieces:
            ids += self._encode(piece)
        return ids

    def render(
        self, messages, *, tools=None, add_generation_prompt: bool = False
    ) -> RenderedTokens:
        """Render with per-token attribution for the trainer loss mask.

        Attribution is exact: spans are emitted already labelled, so assistant bodies and
        their terminators are marked sampled while the surrounding scaffolding is not.
        Splits only ever fall on special-token boundaries, which are atomic in the
        tokenizer, so encoding per span concatenates to the same ids as encoding the
        whole string at once.
        """
        pieces = self._build(
            self._prepare(messages),
            tools=tools,
            add_generation_prompt=add_generation_prompt,
        )

        token_ids: list[int] = []
        message_indices: list[int] = []
        sampled_mask: list[bool] = []
        is_content: list[bool] = []
        for piece in pieces:
            encoded = self._encode(piece)
            token_ids += encoded
            message_indices += [piece.msg_idx] * len(encoded)
            sampled_mask += [piece.sampled] * len(encoded)
            # is_content is NOT sampled_mask: a user or tool message's body is content
            # the caller supplied even though the model never sampled it. Only the
            # header scaffolding is excluded on every role.
            is_content += [piece.is_body] * len(encoded)

        return RenderedTokens(
            token_ids=token_ids,
            message_indices=message_indices,
            sampled_mask=sampled_mask,
            is_content=is_content,
            message_roles=[m.get("role") for m in messages],
            message_tool_names=[m.get("name") for m in messages],
            multi_modal_data=None,
        )

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages,
        *,
        tools=None,
    ) -> RenderedTokens | None:
        """Extend prompt + sampled completion with the next turn, without re-rendering.

        Re-rendering the previous turn would round-trip the completion through parse and
        back, which can change its tokenization. Keeping the sampled ids verbatim and
        appending only the new messages avoids that drift, so the completion the trainer
        sees stays bitwise what the generator produced.

        Returns ``None`` -- the caller then re-renders -- when the extension cannot be
        appended safely: no prior prompt, nothing to add, an assistant turn in the
        extension (its terminator depends on the following message's role, which the
        bridge cannot see), or a prior turn with no ``<|eot|>`` to attach to.

        The output is a prompt, so nothing in it is sampled. ``message_indices`` follows
        the library convention: -1 over the carried-forward prefix and the trailing
        generation prompt, and the index into ``new_messages`` elsewhere.
        """
        if not previous_prompt_ids or not new_messages:
            return None
        if reject_assistant_in_extension(new_messages):
            return None
        # Under a retention policy that drops history at user-query boundaries, the next
        # prompt is not a suffix of this one, so the bridge cannot extend it.
        if should_rerender_for_thinking_retention(
            self.effective_thinking_retention, new_messages
        ):
            return None

        # A completion truncated at max_tokens has no terminator; synthesizing <|eot|>
        # closes the turn the same way the template would.
        previous_ids = trim_to_turn_close(
            previous_prompt_ids,
            previous_completion_ids,
            {EOT_ID},
            synthesize_close=EOT_ID,
        )
        if previous_ids is None:
            return None

        prepared = self._prepare(list(new_messages))
        # _build always emits the leading bos, and synthesises a default system block
        # when no system message is present. Both already exist in previous_ids, so the
        # extension starts after them.
        pieces = self._build(prepared, tools=tools, add_generation_prompt=True)
        skip_synthesized_system = not any(m.get("role") == "system" for m in prepared)

        ext: list[int] = []
        ext_indices: list[int] = []
        ext_content: list[bool] = []
        blocks_seen = 0
        for piece in pieces:
            if piece.token_id == START_ID:
                blocks_seen += 1
            if piece.token_id == self._tok.bos_token_id and piece.text == self._bos:
                continue
            if skip_synthesized_system and blocks_seen == 1:
                continue
            encoded = self._encode(piece)
            ext += encoded
            ext_indices += [-1 if piece.is_generation_prompt else piece.msg_idx] * len(
                encoded
            )
            ext_content += [piece.is_body] * len(encoded)

        total = len(previous_ids) + len(ext)
        return RenderedTokens(
            token_ids=previous_ids + ext,
            message_indices=[-1] * len(previous_ids) + ext_indices,
            sampled_mask=[False] * total,
            is_content=[False] * len(previous_ids) + ext_content,
            message_roles=[m.get("role") or "" for m in new_messages],
            message_tool_names=extract_message_tool_names(new_messages),
            multi_modal_data=None,
        )

    # ------------------------------------------------------------------ parsing

    def parse_response(self, token_ids, *, tools=None) -> ParsedResponse:
        """Parse a Muse Glimmer completion into (content, reasoning_content, tool_calls).

        Muse Glimmer emits harmony channels `to=<recipient><|message|><body>`:
          - recipient == "self"      -> reasoning
          - body has an ATEM block   -> tool call(s) (env.step reads these)
          - otherwise ("user", ...)  -> final answer content
        Validated against real generations.
        """
        text = self._tok.decode(token_ids, skip_special_tokens=False)

        reasoning_parts, content_parts, tool_calls = [], [], []
        for recipient, body in _parse_channels(text):
            atem = parse_atem_tool_calls(body)
            if atem:
                tool_calls += [
                    ParsedToolCall(
                        raw=render_atem_tool_call(c["name"], c["arguments"]),
                        name=c["name"],
                        arguments=c["arguments"],
                    )
                    for c in atem
                ]
            elif recipient == "self":
                reasoning_parts.append(body)
            else:
                # strip any stray ATEM remnants; keep the plain answer text
                content_parts.append(_FUNCTION_CALLS_BLOCK.sub("", body).strip())

        content = "\n".join(p for p in content_parts if p).strip()
        reasoning = "\n".join(reasoning_parts).strip() or None
        # Opt-in: recover an answer from reasoning when the model produced no
        # user-facing channel and no tool call. See the config field for why this is
        # off by default.
        if (
            self._config.answer_from_reasoning_fallback
            and not content
            and not tool_calls
            and reasoning
        ):
            lines = [ln.strip() for ln in reasoning.splitlines() if ln.strip()]
            if lines:
                content = lines[-1]
        return ParsedResponse(
            content=content,
            reasoning_content=reasoning,
            tool_calls=tool_calls,
        )


# TODO: upstream Muse Glimmer to PrimeIntellect-ai/renderers, then make the config a plain
# BaseRendererConfig again and delete this line (the renderer class is defined above).
MuseGlimmerRendererConfig.renderer_cls = MuseGlimmerRenderer
