# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Text-only Muse Spark Open renderer for supervised fine-tuning and generation."""

from __future__ import annotations

import datetime
import html
import json
import re
from dataclasses import dataclass
from typing import Any, cast, NamedTuple

from renderers import Message, Renderer, ToolSpec
from renderers.base import (
    attribute_text_segments,
    extract_message_tool_names,
    ParsedResponse,
    ParsedToolCall,
    reject_assistant_in_extension,
    RenderedTokens,
    ToolCallParseStatus,
    trim_to_turn_close,
)
from renderers.parsing import _build_param_type_index, _coerce_arg_value

from torchtitan.components.renderer import RendererConfig, RendererTokenizerWrapper
from torchtitan.components.tokenizer import HuggingFaceTokenizer


START_STR, MESSAGE_STR = "<|start|>", "<|message|>"
EOM_STR, EOT_STR = "<|eom|>", "<|eot|>"

START_ID = 200022
MESSAGE_ID = 200023
EOM_ID = 200007
EOT_ID = 200008
EOS_ID = 200001

_DEFAULT_CUTOFF_DATE = "2026-01-04"
_REASONING_EFFORT = {
    "minimal": 1,
    "low": 16,
    "medium": 64,
    "high": 256,
    "xhigh": 512,
}
_EFFORT_TO_STRENGTH = (
    ("Reasoning effort", "Reasoning strength"),
    ("Reasoning Effort", "Reasoning Strength"),
    ("reasoning effort", "reasoning strength"),
    ("REASONING EFFORT", "REASONING STRENGTH"),
)
_CHANNEL = re.compile(
    r"(?:to=(?P<recipient>[^\s<|]+))?\s*<\|message\|>(?P<body>.*?)"
    r"(?=<\|eom\|>|<\|eot\|>|<\|end_of_text\|>|<\|start\|>|\Z)",
    re.DOTALL,
)
_ATEM_CALL = re.compile(
    r'<atem:invoke name="(?P<name>[^"]+)">(?P<body>.*?)</atem:invoke>',
    re.DOTALL,
)
_ATEM_PARAMETER = re.compile(
    r'<atem:parameter name="(?P<name>[^"]+)">(?P<value>.*?)' r"</atem:parameter>",
    re.DOTALL,
)


@dataclass(kw_only=True, slots=True)
class MuseSparkRendererConfig(RendererConfig):
    reasoning_effort: str = "medium"
    cutoff_date: str = _DEFAULT_CUTOFF_DATE
    current_date: str | None = None

    def __post_init__(self) -> None:
        if self.reasoning_effort not in _REASONING_EFFORT:
            raise ValueError(
                "reasoning_effort must be one of "
                f"{tuple(_REASONING_EFFORT)}, got {self.reasoning_effort!r}"
            )
        if not isinstance(self.cutoff_date, str):
            raise TypeError("cutoff_date must be a string")
        if self.current_date is not None and not isinstance(self.current_date, str):
            raise TypeError("current_date must be a string or None")

    def build(self, *, tokenizer: HuggingFaceTokenizer) -> Renderer:
        return MuseSparkRenderer(RendererTokenizerWrapper(tokenizer), self)


class _Piece(NamedTuple):
    text: str
    token_id: int | None
    sampled: bool
    message_index: int
    is_content: bool = False
    prompt_prefix_chars: int = 0


def _normalize_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Accept nested/flat dicts and ParsedToolCall objects without mutating messages."""
    normalized_messages = []
    for message in messages:
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            normalized_messages.append(message)
            continue
        normalized_calls: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                if "function" in tool_call:
                    normalized_calls.append(tool_call)
                    continue
                name = tool_call.get("name")
                arguments = tool_call.get("arguments")
                call_id = tool_call.get("id")
            else:
                name = tool_call.name
                arguments = tool_call.arguments
                call_id = tool_call.id
            normalized_call = {
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
            if call_id is not None:
                normalized_call["id"] = call_id
            normalized_calls.append(normalized_call)
        normalized_messages.append({**message, "tool_calls": normalized_calls})
    return normalized_messages


def _escape(value: Any) -> str:
    return (
        str(value)
        .replace("<|", "< |")
        .replace("<image>", "< image>")
        .replace("<video>", "< video>")
        .replace("<audio>", "< audio>")
    )


def _escape_header(value: Any) -> str:
    return _escape(value).replace("\r", " ").replace("\n", " ")


def _escape_attribute(value: Any) -> str:
    return html.escape(_escape_header(value), quote=True).replace("&#x27;", "&#39;")


def _render_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return _escape(content)
    rendered: list[str] = []
    for part in content:
        kind = part.get("type", "")
        if kind in {
            "image",
            "input_image",
            "image_url",
            "video",
            "input_video",
            "video_url",
            "audio",
            "input_audio",
            "audio_url",
        }:
            raise ValueError("Muse Spark SFT rendering supports text-only content.")
        if kind in {"text", "input_text", "output_text"}:
            rendered.append(_escape(part.get("text", "")))
    return "".join(rendered)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _tool_function(tool: dict[str, Any]) -> dict[str, Any]:
    return tool.get("function") or tool


def _render_tool_definitions(tools: list[dict[str, Any]]) -> str:
    parts = [
        "In this environment you have access to a set of tools you can use to "
        "answer the user's question.\n\n",
        'You can invoke a function by writing a "<atem:function_calls>" block '
        "like the following as part of your reply to the user:\n",
        '<atem:function_calls>\n<atem:invoke name="$FUNCTION_NAME">\n'
        '<atem:parameter name="$PARAMETER_NAME">$PARAMETER_VALUE'
        "</atem:parameter>\n...\n</atem:invoke>\n</atem:function_calls>\n\n",
        "String and scalar parameters should be specified as is, while lists and "
        "objects should use JSON format. Note that spaces for string values are "
        "not stripped. The output is not expected to be valid XML and is parsed "
        "with regular expressions.\n",
        "Here are the functions available in JSONSchema format:\n",
        '// Tool metadata\n{"name": "", "description": ""}\n',
        "// Function schemas",
    ]
    for tool in tools:
        function = _tool_function(tool)
        parts.append(
            "\n"
            + _json(
                {
                    "name": function["name"],
                    "description": function.get("description") or "",
                    "parameters": function.get("parameters") or {},
                }
            )
        )
    parts.extend(
        [
            "\n\nHere are examples of how to call functions in the tool set. To make "
            "parallel tool calls, emit each call as its own message (one tool call "
            "per message) in consecutive messages, as shown below:\n",
            "(If the tool namespace is not specified, invoke the function directly "
            "as `example_function_name_1` rather than "
            "`example_tool_name_1.example_function_name_1`)\n\n",
            "to=example_tool_name_1.example_function_name_1\n\n",
            '<atem:function_calls>\n<atem:invoke name="example_tool_name_1.'
            'example_function_name_1">\n<atem:parameter name="example_parameter_1">'
            "value_1</atem:parameter>\n</atem:invoke>\n</atem:function_calls>\n\n",
            "to=example_tool_name_2.example_function_name_2\n\n",
            '<atem:function_calls>\n<atem:invoke name="example_tool_name_2.'
            'example_function_name_2">\n<atem:parameter name="example_parameter_2">'
            'This is the value for the second parameter\nthat can span\n"multiple" '
            'lines\n</atem:parameter>\n<atem:parameter name="example_parameter_3">'
            "value_3</atem:parameter>\n</atem:invoke>\n</atem:function_calls>",
        ]
    )
    return "".join(parts)


def _render_valid_recipients(tools: list[dict[str, Any]] | None) -> str:
    lines = [
        "Use the appropriate recipient for each message:\n",
        '\n- "self": private reasoning and tool planning.',
        '\n- "commentary": user-visible intermediate updates while the assistant will '
        "continue working, including messages sent to the user before or between "
        "tool calls.",
    ]
    recipients = ['"self"', '"commentary"']
    if tools:
        first_name = _escape_header(_tool_function(tools[0])["name"])
        lines.append('\n- "tool" or "[tool name]": the name of the tool to call.')
        recipients.append(f'"[func_name] (e.g., {first_name})"')
    lines.append(
        '\n- "user": messages that end the assistant turn, such as a completed '
        "response or clarification question that waits for the user's reply; do "
        "not use for pre-tool updates or partial responses.\n\n"
    )
    recipients.append('"user"')
    lines.append(f"# Valid recipients: {', '.join(recipients)}.")
    return "".join(lines)


def _render_atem(tool_call: dict[str, Any]) -> str:
    function = tool_call["function"]
    arguments = function.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError("tool_call.function.arguments must be a dict")
    parts = [
        '<atem:function_calls>\n<atem:invoke name="',
        _escape_attribute(function["name"]),
        '">\n',
    ]
    for name, value in arguments.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif value is None:
            rendered = "null"
        elif isinstance(value, (dict, list, tuple)):
            rendered = _escape(_json(value))
        else:
            rendered = _escape(value)
        rendered = html.escape(rendered, quote=False)
        parts.extend(
            [
                '<atem:parameter name="',
                _escape_attribute(name),
                '">',
                rendered,
                "</atem:parameter>\n",
            ]
        )
    parts.append("</atem:invoke>\n</atem:function_calls>")
    return "".join(parts)


class MuseSparkRenderer:
    is_prefix_stable: bool = False

    def __init__(
        self,
        tokenizer: Any,
        config: MuseSparkRendererConfig | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        config = config or MuseSparkRendererConfig()
        self._cutoff_date = config.cutoff_date
        self._current_date = config.current_date
        self._reasoning_effort = config.reasoning_effort
        self._bos = tokenizer.bos_token or ""
        self._verify_special_ids()

    def _verify_special_ids(self) -> None:
        for token, expected in (
            (START_STR, START_ID),
            (MESSAGE_STR, MESSAGE_ID),
            (EOM_STR, EOM_ID),
            (EOT_STR, EOT_ID),
        ):
            actual = self._tokenizer.convert_tokens_to_ids(token)
            if actual != expected:
                raise ValueError(
                    f"Tokenizer maps {token} to id {actual}, but Muse Spark expects "
                    f"{expected}."
                )

    def _system_body(
        self,
        instructions: str | None,
        tools: list[dict[str, Any]] | None,
    ) -> str:
        parts = [
            "You are a helpful AI assistant.",
            f"Knowledge cutoff: {self._cutoff_date}.",
        ]
        current_date = self._current_date
        if current_date is None:
            current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        if current_date:
            parts.append(f"Current date: {current_date}.")
        if instructions:
            for source, replacement in _EFFORT_TO_STRENGTH:
                instructions = instructions.replace(source, replacement)
            parts.append(instructions)
        if tools:
            parts.append(_render_tool_definitions(tools))
        if not instructions or "reasoning strength:" not in instructions.lower():
            strength = _REASONING_EFFORT[self._reasoning_effort]
            parts.append(f"Reasoning strength: {strength}.")
        parts.append(_render_valid_recipients(tools))
        return "\n\n".join(parts)

    @staticmethod
    def _tool_name(message: dict[str, Any], messages: list[dict[str, Any]]) -> str:
        if message.get("name"):
            return message["name"]
        call_id = message.get("tool_call_id")
        for candidate in messages:
            for tool_call in candidate.get("tool_calls") or ():
                if call_id is not None and tool_call.get("id") == call_id:
                    return tool_call["function"]["name"]
        raise ValueError("tool message has no name and no matching tool_call_id")

    def _build(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        add_generation_prompt: bool,
    ) -> list[_Piece]:
        messages = _normalize_tool_calls(messages)
        pieces: list[_Piece] = []
        generation_active = False

        def emit(
            text: str,
            token_id: int | None,
            sampled: bool,
            message_index: int,
            is_content: bool = False,
            prompt_prefix_chars: int = 0,
        ) -> None:
            if text:
                pieces.append(
                    _Piece(
                        text,
                        token_id,
                        sampled,
                        message_index,
                        is_content,
                        prompt_prefix_chars,
                    )
                )

        def block(
            header: str,
            body: str,
            end: str,
            *,
            message_index: int,
            sampled: bool,
            assistant_output: bool = False,
        ) -> None:
            nonlocal generation_active
            if assistant_output:
                starts_generation = not generation_active
                emit(
                    START_STR,
                    START_ID,
                    sampled and not starts_generation,
                    message_index,
                    is_content=sampled and not starts_generation,
                )
                emit(
                    header,
                    None,
                    sampled,
                    message_index,
                    is_content=sampled,
                    prompt_prefix_chars=(len("assistant") if starts_generation else 0),
                )
                emit(
                    MESSAGE_STR,
                    MESSAGE_ID,
                    sampled,
                    message_index,
                    is_content=sampled,
                )
            else:
                generation_active = False
                emit(START_STR, START_ID, False, message_index)
                emit(header, None, False, message_index)
                emit(MESSAGE_STR, MESSAGE_ID, False, message_index)
            emit(body, None, sampled, message_index, is_content=True)
            emit(
                end,
                EOT_ID if end == EOT_STR else EOM_ID,
                sampled,
                message_index,
                is_content=sampled,
            )
            if assistant_output:
                generation_active = end == EOM_STR

        emit(self._bos, self._tokenizer.bos_token_id, False, 0)
        rest = messages
        instructions = None
        if messages and messages[0].get("role") in {"system", "developer"}:
            instructions = _render_content(messages[0].get("content"))
            rest = messages[1:]
        block(
            "system",
            self._system_body(instructions, tools),
            EOT_STR,
            message_index=0,
            sampled=False,
        )
        self._build_messages(rest, messages, pieces, block)
        if add_generation_prompt:
            index = max(len(messages) - 1, 0)
            emit(START_STR, START_ID, False, index)
            emit("assistant", None, False, index)
        return pieces

    def _build_messages(
        self,
        rest: list[dict[str, Any]],
        all_messages: list[dict[str, Any]],
        pieces: list[_Piece],
        block: Any,
    ) -> None:
        del pieces
        offset = len(all_messages) - len(rest)
        for position, message in enumerate(rest):
            index = offset + position
            role = message.get("role")
            same_role_next = (
                position + 1 < len(rest) and rest[position + 1].get("role") == role
            )
            end = EOM_STR if same_role_next else EOT_STR
            if role in {"system", "developer", "user"}:
                self._build_plain_message(role, message, index, block)
            elif role == "tool":
                self._build_tool_message(message, all_messages, index, block)
            elif role == "assistant":
                self._build_assistant_message(message, end, index, block)
            else:
                raise ValueError(f"Unsupported Muse Spark chat role: {role!r}")

    @staticmethod
    def _build_plain_message(
        role: str,
        message: dict[str, Any],
        index: int,
        block: Any,
    ) -> None:
        body = _render_content(message.get("content"))
        if (
            role == "user"
            and message.get("render_timestamp")
            and message.get("timestamp")
        ):
            timestamp = message["timestamp"]
            if isinstance(timestamp, str):
                body = f"[{_escape_header(timestamp)}] {body}"
            elif hasattr(timestamp, "strftime"):
                body = f"[{timestamp.strftime('%a %Y-%m-%d %H:%M %Z').strip()}] {body}"
        block(role, body, EOT_STR, message_index=index, sampled=False)

    def _build_tool_message(
        self,
        message: dict[str, Any],
        all_messages: list[dict[str, Any]],
        index: int,
        block: Any,
    ) -> None:
        name = self._tool_name(message, all_messages)
        body = (
            f'<tool_output name="{_escape_attribute(name)}">\n'
            f"{_render_content(message.get('content'))}\n</tool_output>"
        )
        block(
            f"tool {_escape_header(name)}",
            body,
            EOT_STR,
            message_index=index,
            sampled=False,
        )

    @staticmethod
    def _build_assistant_message(
        message: dict[str, Any],
        end: str,
        index: int,
        block: Any,
    ) -> None:
        loss = message.get("loss", True)
        if type(loss) is not bool:
            raise ValueError("assistant message loss must be a bool when provided")
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if reasoning:
            block(
                "assistant to=self",
                _escape(reasoning),
                EOM_STR,
                message_index=index,
                sampled=loss,
                assistant_output=True,
            )
        body = _render_content(message.get("content"))
        tool_calls = message.get("tool_calls")
        if tool_calls:
            if body:
                block(
                    "assistant to=commentary",
                    body,
                    EOM_STR,
                    message_index=index,
                    sampled=loss,
                    assistant_output=True,
                )
            for call_index, tool_call in enumerate(tool_calls):
                call_end = end if call_index == len(tool_calls) - 1 else EOM_STR
                name = tool_call["function"]["name"]
                block(
                    f"assistant to={_escape_header(name)}",
                    _render_atem(tool_call),
                    call_end,
                    message_index=index,
                    sampled=loss,
                    assistant_output=True,
                )
            return
        if body or not reasoning:
            recipient = message.get("recipient") or "user"
            if recipient == "user":
                end_turn = True
            elif recipient in {"self", "commentary"}:
                end_turn = False
            elif message.get("end_turn") is not None:
                end_turn = message["end_turn"]
            else:
                end_turn = False
            block(
                f"assistant to={_escape_header(recipient)}",
                body,
                EOT_STR if end_turn else EOM_STR,
                message_index=index,
                sampled=loss,
                assistant_output=True,
            )

    def _encode(self, piece: _Piece) -> tuple[list[int], list[bool]]:
        if piece.token_id is not None:
            return [piece.token_id], [piece.sampled]
        if not piece.prompt_prefix_chars:
            token_ids = self._tokenizer.encode(piece.text, add_special_tokens=False)
            return token_ids, [piece.sampled] * len(token_ids)

        attributed = attribute_text_segments(
            self._tokenizer,
            [
                (piece.text[: piece.prompt_prefix_chars], False),
                (piece.text[piece.prompt_prefix_chars :], piece.sampled),
            ],
        )
        return (
            [token_id for token_id, _ in attributed],
            [sampled for _, sampled in attributed],
        )

    def _render_text(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = False,
    ) -> str:
        return "".join(
            piece.text
            for piece in self._build(
                messages,
                tools=tools,
                add_generation_prompt=add_generation_prompt,
            )
        )

    def get_stop_token_ids(self) -> list[int]:
        return [EOT_ID, EOS_ID]

    def render_ids(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        return [
            token_id
            for piece in self._build(
                cast(list[dict[str, Any]], messages),
                tools=cast(list[dict[str, Any]] | None, tools),
                add_generation_prompt=add_generation_prompt,
            )
            for token_id in self._encode(piece)[0]
        ]

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        pieces = self._build(
            cast(list[dict[str, Any]], messages),
            tools=cast(list[dict[str, Any]] | None, tools),
            add_generation_prompt=add_generation_prompt,
        )
        token_ids: list[int] = []
        message_indices: list[int] = []
        sampled_mask: list[bool] = []
        is_content: list[bool] = []
        for piece in pieces:
            encoded, piece_sampled_mask = self._encode(piece)
            token_ids.extend(encoded)
            message_indices.extend([piece.message_index] * len(encoded))
            sampled_mask.extend(piece_sampled_mask)
            if piece.prompt_prefix_chars and piece.is_content:
                is_content.extend(piece_sampled_mask)
            else:
                is_content.extend([piece.is_content] * len(encoded))
        return RenderedTokens(
            token_ids=token_ids,
            message_indices=message_indices,
            sampled_mask=sampled_mask,
            is_content=is_content,
            message_roles=[cast(str, message.get("role")) for message in messages],
            message_tool_names=[message.get("name") for message in messages],
            multi_modal_data=None,
        )

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> RenderedTokens | None:
        """Extend the previous turn's exact tokens with new tool or user messages.

        A full re-render replaces the sampled completion with its parsed, canonical
        form: stripped whitespace, merged reasoning channels, re-serialized tool
        arguments. When the completion was not canonical, the next prompt then stops
        extending it, and training starts a new sample that repeats the history.
        Keeping the sampled tokens verbatim avoids that. The new messages render as in
        ``render``, piece by piece, followed by the generation prompt.

        Returns None, so the caller re-renders, when there is no previous turn, the
        completion did not close its turn with <|eot|>, or the tool results do not
        match the completion's tool calls one to one. Muse Spark tool calls carry no
        ids, so tool results are matched to the calls in order, and a result's own
        ``name``, when given, must agree.
        """
        if (
            not previous_prompt_ids
            or not new_messages
            or reject_assistant_in_extension(new_messages)
        ):
            return None
        previous_ids = trim_to_turn_close(
            previous_prompt_ids, previous_completion_ids, {EOT_ID}
        )
        if previous_ids is None:
            return None

        tool_results = [m for m in new_messages if m.get("role") == "tool"]
        tool_names: list[str] = []
        if tool_results:
            calls = self.parse_response(previous_completion_ids, tools=tools).tool_calls
            if len(calls) != len(tool_results):
                return None
            for message, call in zip(tool_results, calls):
                if call.name is None or message.get("name") not in (None, call.name):
                    return None
                tool_names.append(call.name)

        pieces: list[_Piece] = []

        def block(
            header: str,
            body: str,
            end: str,
            *,
            message_index: int,
            sampled: bool,
            assistant_output: bool = False,
        ) -> None:
            del sampled, assistant_output
            for text, token_id, is_content in (
                (START_STR, START_ID, False),
                (header, None, False),
                (MESSAGE_STR, MESSAGE_ID, False),
                (body, None, True),
                (end, EOT_ID if end == EOT_STR else EOM_ID, False),
            ):
                if text:
                    pieces.append(
                        _Piece(text, token_id, False, message_index, is_content, 0)
                    )

        names = iter(tool_names)
        for index, message in enumerate(cast(list[dict[str, Any]], new_messages)):
            role = message.get("role")
            if role == "tool":
                self._build_tool_message(
                    {**message, "name": next(names)}, [], index, block
                )
            elif role in {"system", "developer", "user"}:
                self._build_plain_message(role, message, index, block)
            else:
                return None
        pieces.append(_Piece(START_STR, START_ID, False, -1, False, 0))
        pieces.append(_Piece("assistant", None, False, -1, False, 0))

        token_ids = list(previous_ids)
        message_indices = [-1] * len(previous_ids)
        is_content = [False] * len(previous_ids)
        for piece in pieces:
            encoded, _ = self._encode(piece)
            token_ids.extend(encoded)
            message_indices.extend([piece.message_index] * len(encoded))
            is_content.extend([piece.is_content] * len(encoded))
        return RenderedTokens(
            token_ids=token_ids,
            message_indices=message_indices,
            sampled_mask=[False] * len(token_ids),
            is_content=is_content,
            message_roles=[m.get("role") or "" for m in new_messages],
            message_tool_names=extract_message_tool_names(new_messages),
            multi_modal_data=None,
        )

    def parse_response(
        self,
        token_ids: list[int],
        *,
        tools: list[ToolSpec] | None = None,
        prompt_ids: list[int] | None = None,
    ) -> ParsedResponse:
        # The generation prompt ends after "assistant", so the completion contains
        # the recipient and channel body without needing prompt context.
        del prompt_ids
        text = self._tokenizer.decode(token_ids, skip_special_tokens=False)
        param_index = _build_param_type_index(tools)
        reasoning: list[str] = []
        content: list[str] = []
        tool_calls: list[ParsedToolCall] = []
        for channel in _CHANNEL.finditer(text):
            recipient = channel.group("recipient") or "user"
            body = channel.group("body").strip()
            calls = _parse_atem(body, param_index=param_index)
            if calls:
                tool_calls.extend(calls)
            elif recipient == "self":
                reasoning.append(body)
            else:
                content.append(body)
        return ParsedResponse(
            content="\n".join(filter(None, content)).strip(),
            reasoning_content="\n".join(filter(None, reasoning)).strip() or None,
            tool_calls=tool_calls,
        )


def _parse_atem(
    text: str,
    *,
    param_index: dict[str, dict[str, dict[str, Any]]],
) -> list[ParsedToolCall]:
    parsed: list[ParsedToolCall] = []
    for match in _ATEM_CALL.finditer(text):
        name = html.unescape(match.group("name"))
        params = param_index.get(name, {})
        arguments: dict[str, Any] = {}
        any_json_fallback = False
        for parameter in _ATEM_PARAMETER.finditer(match.group("body")):
            parameter_name = html.unescape(parameter.group("name"))
            value, used_fallback = _coerce_arg_value(
                html.unescape(parameter.group("value")),
                params.get(parameter_name),
            )
            arguments[parameter_name] = value
            any_json_fallback = any_json_fallback or used_fallback
        parsed.append(
            ParsedToolCall(
                raw=match.group(0),
                name=name,
                arguments=arguments,
                status=(
                    ToolCallParseStatus.INVALID_JSON
                    if any_json_fallback
                    else ToolCallParseStatus.OK
                ),
            )
        )
    return parsed
