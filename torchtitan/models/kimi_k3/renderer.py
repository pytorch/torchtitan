# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3 renderer: chat messages <-> tokens in K3's XTML chat encoding.

K3 marks message structure with four special tokens, ``<|open|>`` and ``<|close|>``
around a tag (``message``, ``think``, ``response``) followed by ``<|sep|>``, and
``<|end_of_msg|>`` after each message::

    <|open|>message role="user"<|sep|>{content}<|close|>message<|sep|><|end_of_msg|>
    <|open|>message role="assistant"<|sep|><|open|>think<|sep|>{reasoning}
    <|close|>think<|sep|><|open|>response<|sep|>{content}<|close|>response<|sep|>
    <|close|>message<|sep|><|end_of_msg|>

This follows the checkpoint's ``encoding_k3.build_chat_segments`` for system, user and
assistant messages (markers are special tokens, everything else ordinary text);
``scripts/checkpoint_conversion/convert_kimi_k3_tokenizer.py`` checks the token ids
against the checkpoint's ``apply_chat_template``. Tools and tool messages are not
supported yet and raise.

Implements the ``renderers.Renderer`` protocol; selected with ``CustomRendererConfig``
(``import_path="torchtitan.models.kimi_k3.renderer.KimiK3Renderer"``).
"""

from __future__ import annotations

from typing import Literal, NamedTuple

from renderers.base import (
    ParsedResponse,
    reject_assistant_in_extension,
    RenderedTokens,
    trim_to_turn_close,
)
from renderers.configs import BaseRendererConfig

OPEN = "<|open|>"
CLOSE = "<|close|>"
SEP = "<|sep|>"
END_OF_MSG = "<|end_of_msg|>"
EOS = "[EOS]"


class KimiK3RendererConfig(BaseRendererConfig):
    """Kimi K3 (XTML chat encoding) renderer config."""

    name: Literal["kimi_k3"] = "kimi_k3"
    thinking: bool = True
    """Open the ``think`` channel in the generation prompt (thinking mode); otherwise
    the model answers directly in the ``response`` channel."""

    thinking_effort: Literal["low", "high", "max"] | None = "max"
    """K3's internal ``thinking-effort`` system message (thinking mode only); the
    checkpoint's ``apply_chat_template`` defaults it to ``"max"``. ``None`` omits it."""

    # Both mirror kwargs of K3's apply_chat_template (encoding_k3.build_chat_segments).
    _template_fields = frozenset({"thinking", "thinking_effort"})


class _Piece(NamedTuple):
    """One span: a special marker (``special``) or ordinary text."""

    text: str
    special: bool
    msg_idx: int
    sampled: bool = False
    is_body: bool = False


class KimiK3Renderer:
    config_class = KimiK3RendererConfig

    @property
    def is_prefix_stable(self) -> bool:
        return True

    def __init__(self, tokenizer, config: KimiK3RendererConfig | None = None):
        self._tok = tokenizer
        self._config = config if config is not None else KimiK3RendererConfig()
        self._special_ids = {}
        for marker in (OPEN, CLOSE, SEP, END_OF_MSG, EOS):
            token_id = tokenizer.convert_tokens_to_ids(marker)
            if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
                raise ValueError(f"Tokenizer has no special token {marker}")
            self._special_ids[marker] = token_id

    # ---------------------------------------------------------------- rendering

    def _build(
        self, messages, *, tools, add_generation_prompt, effort_message: bool = True
    ) -> list[_Piece]:
        if tools:
            raise ValueError("KimiK3Renderer does not support tools yet")
        pieces: list[_Piece] = []

        def control(marker: str, idx: int, sampled: bool = False) -> None:
            pieces.append(_Piece(marker, True, idx, sampled))

        def text(value: str, idx: int, sampled: bool = False, body: bool = False):
            if value:
                pieces.append(_Piece(value, False, idx, sampled, body))

        def open_tag(tag: str, idx: int, attrs=(), sampled: bool = False) -> None:
            control(OPEN, idx, sampled)
            text(tag, idx, sampled)
            for key, value in attrs:
                escaped = str(value).replace("&", "&amp;").replace('"', "&quot;")
                text(f" {key}", idx, sampled)
                text('="', idx, sampled)
                text(escaped, idx, sampled)
                text('"', idx, sampled)
            control(SEP, idx, sampled)

        def close_tag(tag: str, idx: int, sampled: bool = False) -> None:
            control(CLOSE, idx, sampled)
            text(tag, idx, sampled)
            control(SEP, idx, sampled)

        if (
            effort_message
            and self._config.thinking
            and self._config.thinking_effort is not None
        ):
            open_tag("message", -1, [("role", "system"), ("type", "thinking-effort")])
            text(
                "`thinking_effort` guides on how much to think in your thinking "
                "channel (not including the response channel), supported values "
                "include `low`, `medium`, `high`, and `max`.\n"
                "Now the system is invoked with "
                f"`thinking_effort={self._config.thinking_effort}`.",
                -1,
            )
            close_tag("message", -1)
            control(END_OF_MSG, -1)

        for idx, message in enumerate(messages):
            role = message.get("role")
            if role not in ("system", "user", "assistant"):
                raise ValueError(f"KimiK3Renderer does not support role {role!r} yet")
            if message.get("tools") or message.get("tool_calls"):
                raise ValueError("KimiK3Renderer does not support tools yet")
            attrs = [("role", role)]
            if message.get("name"):
                attrs.append(("name", message["name"]))
            open_tag("message", idx, attrs)
            if role == "assistant":
                # Everything after the role header is what the model produced.
                if self._config.thinking:
                    reasoning = (
                        message.get("reasoning_content")
                        or message.get("reasoning")
                        or ""
                    )
                    open_tag("think", idx, sampled=True)
                    text(str(reasoning), idx, sampled=True)
                    close_tag("think", idx, sampled=True)
                open_tag("response", idx, sampled=True)
                text(_content_text(message.get("content")), idx, True, True)
                close_tag("response", idx, sampled=True)
                close_tag("message", idx, sampled=True)
                control(END_OF_MSG, idx, sampled=True)
            else:
                text(_content_text(message.get("content")), idx, body=True)
                close_tag("message", idx)
                control(END_OF_MSG, idx)

        if add_generation_prompt:
            open_tag("message", -1, [("role", "assistant")])
            open_tag("think" if self._config.thinking else "response", -1)
        return pieces

    def _encode(self, piece: _Piece) -> list[int]:
        if piece.special:
            return [self._special_ids[piece.text]]
        return self._tok.encode(piece.text, add_special_tokens=False)

    def get_stop_token_ids(self) -> list[int]:
        return [self._special_ids[END_OF_MSG], self._special_ids[EOS]]

    def render_ids(
        self, messages, *, tools=None, add_generation_prompt: bool = False
    ) -> list[int]:
        ids: list[int] = []
        for piece in self._build(
            messages, tools=tools, add_generation_prompt=add_generation_prompt
        ):
            ids += self._encode(piece)
        return ids

    def _encode_pieces(
        self, pieces: list[_Piece]
    ) -> tuple[list[int], list[int], list[bool], list[bool]]:
        """Token ids with their message index, sampled flag and body flag."""
        token_ids: list[int] = []
        message_indices: list[int] = []
        sampled_mask: list[bool] = []
        is_content: list[bool] = []
        for piece in pieces:
            encoded = self._encode(piece)
            token_ids += encoded
            message_indices += [piece.msg_idx] * len(encoded)
            sampled_mask += [piece.sampled] * len(encoded)
            is_content += [piece.is_body] * len(encoded)
        return token_ids, message_indices, sampled_mask, is_content

    def render(
        self, messages, *, tools=None, add_generation_prompt: bool = False
    ) -> RenderedTokens:
        token_ids, message_indices, sampled_mask, is_content = self._encode_pieces(
            self._build(
                messages, tools=tools, add_generation_prompt=add_generation_prompt
            )
        )
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
        """Extend the previous prompt + sampled completion with the next turn.

        A re-render rebuilds the previous assistant turn from ``parse_response``'s
        stripped text, so it can drop or re-tokenize sampled ids. The next prompt then
        no longer extends the previous one: the generator re-prefills from the first
        changed token and the trainer sees a history the generator never sampled.
        Rendering concatenates per-message pieces, so appending the new messages'
        pieces to the sampled ids equals a re-render that keeps the sampled turn.

        Returns ``None``, and the caller re-renders, when there is nothing to extend,
        the new messages include an assistant turn or tools, or the completion did
        not end its message with ``<|end_of_msg|>`` (cut at the token limit or
        stopped at ``[EOS]``).
        """
        if not previous_prompt_ids or not new_messages or tools:
            return None
        if reject_assistant_in_extension(new_messages):
            return None
        previous_ids = trim_to_turn_close(
            previous_prompt_ids,
            previous_completion_ids,
            {self._special_ids[END_OF_MSG]},
        )
        if previous_ids is None:
            return None
        # The effort message opens the conversation and is already in previous_ids.
        token_ids, message_indices, _, is_content = self._encode_pieces(
            self._build(
                new_messages,
                tools=None,
                add_generation_prompt=True,
                effort_message=False,
            )
        )
        num_previous = len(previous_ids)
        return RenderedTokens(
            token_ids=previous_ids + token_ids,
            message_indices=[-1] * num_previous + message_indices,
            sampled_mask=[False] * (num_previous + len(token_ids)),
            is_content=[False] * num_previous + is_content,
            message_roles=[m.get("role") for m in new_messages],
            message_tool_names=[m.get("name") for m in new_messages],
            multi_modal_data=None,
        )

    # ------------------------------------------------------------------ parsing

    def parse_response(
        self,
        token_ids: list[int],
        *,
        tools=None,
        prompt_ids: list[int] | None = None,
    ) -> ParsedResponse:
        """Split a completion into ``think`` (reasoning) and ``response`` (content).

        The generation prompt opens ``think`` (thinking mode) or ``response``, so the
        completion starts inside that channel. A completion cut at the token limit
        keeps what was generated.
        """
        del prompt_ids
        text = self._tok.decode(token_ids, skip_special_tokens=False)
        close_think = f"{CLOSE}think{SEP}"
        open_response = f"{OPEN}response{SEP}"
        close_response = f"{CLOSE}response{SEP}"
        reasoning = None
        if self._config.thinking:
            reasoning, found, text = text.partition(close_think)
            if not found:
                return ParsedResponse(
                    content="",
                    reasoning_content=reasoning.strip() or None,
                    tool_calls=[],
                )
            _, _, text = text.partition(open_response)
        content = text.split(close_response, 1)[0]
        for marker in (END_OF_MSG, EOS):
            content = content.split(marker, 1)[0]
        return ParsedResponse(
            content=content.strip(),
            reasoning_content=(reasoning or "").strip() or None,
            tool_calls=[],
        )


def _content_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if part.get("type") != "text":
            raise ValueError("KimiK3Renderer supports text content only")
        parts.append(part.get("text", ""))
    return "".join(parts)
