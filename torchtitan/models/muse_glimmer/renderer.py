# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muse Glimmer renderer: chat messages <-> tokens for TorchTitan RL.

RL needs two directions. ``render_ids`` turns messages + tool definitions into token
ids via the model's chat template; ``parse_response`` turns generated tokens back into
``(content, reasoning_content, tool_calls)`` so the rollout loop can tell "call a tool
and continue" apart from "final answer, score it".

Muse Glimmer uses the harmony chat format -- an assistant turn is a sequence of
``to=<recipient><|message|><body>`` channels, where ``to=self`` is private reasoning and
other recipients carry user-visible content -- and expresses tool calls as ATEM XML
inside those channels (see ``atem.py``).

Implements the ``renderers.Renderer`` Protocol. ``register()`` installs it into the
``renderers`` library's public registry (``RENDERER_REGISTRY`` / ``_CONFIG_BY_NAME``),
which is that library's supported extension path -- no fork or upstream change needed.

Every other TorchTitan model resolves to a renderer that lives in
PrimeIntellect-ai/renderers. This one ships here because Muse Glimmer is not in that
library yet. Once it is upstreamed, delete ``register()`` and keep only the
``_RENDERER_BY_MODEL`` entry.
"""

from __future__ import annotations

import re
from typing import ClassVar, Literal

from renderers.base import ParsedResponse, ParsedToolCall, RenderedTokens
from renderers.configs import BaseRendererConfig

from .atem import parse_atem_tool_calls, render_atem_tool_call  # noqa: F401

RENDERER_NAME = "muse_glimmer"


class MuseGlimmerRendererConfig(BaseRendererConfig):
    """Muse Glimmer (harmony chat format + ATEM tool calls) renderer config."""

    name: Literal["muse_glimmer"] = RENDERER_NAME

    # renderers validates in BaseRendererConfig.__pydantic_init_subclass__ that every
    # non-base field is classified as either a chat-template kwarg or a renderer-internal
    # knob; the two sets must be disjoint and together cover all of them. Declared
    # unconditionally -- versions without the validator ignore these ClassVars, so this
    # is compatible with both. ``reasoning_strength`` belongs in _template_fields, not
    # _internal_fields: it is forwarded to the chat template verbatim, so it is exactly
    # the kind of field the library's template-parity matrix is meant to cover.
    _template_fields: ClassVar[frozenset[str]] = frozenset({"reasoning_strength"})
    _internal_fields: ClassVar[frozenset[str]] = frozenset(
        {
            "chat_template",
            "retain_reasoning_in_history",
            "answer_from_reasoning_fallback",
        }
    )

    chat_template: str | None = None
    """Jinja chat template to render with: a path to a ``.jinja`` file, or the template
    source inline. ``None`` (default) uses the template the tokenizer ships.

    Required when the tokenizer has no template of its own. Not every Muse Glimmer
    checkpoint carries one -- see ``MuseGlimmerRenderer.__init__`` for why this is a
    hard error rather than a fallback.
    """

    reasoning_strength: str | None = None
    """Passed through to the chat template to size the reasoning budget.

    ``None`` (default) leaves the template's own default alone. Set ``"low"`` for
    agentic tasks with a tight token budget: at the default strength the model can
    spend the whole budget reasoning and get truncated before it emits a tool call or
    an answer, which scores as a failed rollout.
    """

    retain_reasoning_in_history: bool = True
    """Whether prior assistant turns keep their ``reasoning_content`` when re-rendered.

    The chat template emits a ``to=self`` channel for any assistant message carrying
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


# Muse Glimmer special-token ids (from the GGUF metadata)
START_ID = 200022  # <|start|> begins a harmony message header
MESSAGE_ID = 200023  # <|message|> ends the header, body follows
EOT_ID = 200008  # <|eot|> end of turn
EOM_ID = 200007  # <|eom|> end of message
EOS_ID = 200001  # <|end_of_text|>

_FUNCTION_CALLS_BLOCK = re.compile(
    r"<atem:function_calls>.*?</atem:function_calls>", re.DOTALL
)
# One harmony channel: "to=<recipient><|message|><body>" up to <|eom|>/<|eot|>/next channel.
_CHANNEL = re.compile(
    r"(?:to=(?P<rcpt>[^\s<|]+))?\s*<\|message\|>(?P<body>.*?)"
    r"(?=<\|eom\|>|<\|eot\|>|<\|end_of_text\|>|<\|start\|>|\Z)",
    re.DOTALL,
)


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
    """Muse Glimmer's chat template renders assistant tool_calls as `tc.function.name/.arguments`
    (OpenAI-nested). Our ParsedToolCall is flat (.name/.arguments) -> convert so re-rendering
    prior turns doesn't crash the template."""
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
    out = []
    for m in _CHANNEL.finditer(text):
        out.append((m.group("rcpt") or "user", m.group("body").strip()))
    return out


class MuseGlimmerRenderer:
    def __init__(self, tokenizer, config: MuseGlimmerRendererConfig | None = None):
        # (tokenizer, config) is the renderers-library constructor contract, so
        # ``create_renderer`` can instantiate this from RENDERER_REGISTRY.
        self._tok = tokenizer
        self._config = config or MuseGlimmerRendererConfig()
        # The controller reads renderer._tokenizer (e.g. for pad_id=eos_token_id).
        self._tokenizer = tokenizer
        self._chat_template = self._resolve_chat_template()

    def _resolve_chat_template(self) -> str | None:
        """Return the template source to render with, or None to use the tokenizer's.

        Raises when neither is available. Muse Glimmer checkpoints do not all ship a
        chat template -- some carry it as a separate ``chat_template.jinja``, others
        have none at all -- and without one ``apply_chat_template`` raises deep inside
        the first rollout, which the rollout loop reports as a generic per-rollout
        ERROR. Failing here instead turns "every rollout errors" into one actionable
        message before training starts.
        """
        configured = self._config.chat_template
        if configured is not None:
            if configured.endswith(".jinja") or configured.endswith(".j2"):
                with open(configured) as f:
                    return f.read()
            return configured
        if getattr(self._tok, "chat_template", None):
            return None  # tokenizer has one; let apply_chat_template find it
        raise ValueError(
            f"The tokenizer at {getattr(self._tok, 'name_or_path', '<unknown>')!r} has "
            "no chat template, so the muse_glimmer renderer cannot render prompts. Not "
            "every Muse Glimmer checkpoint ships one. Point the renderer at the "
            "template for your checkpoint, e.g. "
            "MuseGlimmerRendererConfig(chat_template='/path/to/chat_template.jinja'), "
            "or use a checkpoint whose tokenizer carries a chat template."
        )

    def _template_arg(self) -> dict:
        """``chat_template=`` kwarg, omitted when deferring to the tokenizer's."""
        return (
            {}
            if self._chat_template is None
            else {"chat_template": self._chat_template}
        )

    def _prepare(self, messages):
        """Apply history policy, then the tool-call shape the template expects."""
        if not self._config.retain_reasoning_in_history:
            messages = _strip_reasoning_history(messages)
        return _normalize_tool_calls(messages)

    def _template_kwargs(self) -> dict:
        """Extra kwargs for ``apply_chat_template``, omitting unset knobs."""
        if self._config.reasoning_strength is None:
            return {}
        return {"reasoning_strength": self._config.reasoning_strength}

    def get_stop_token_ids(self) -> list[int]:
        # NOT <|eom|> (200007): it ends the *reasoning* channel, after which the model emits
        # the tool call / final answer. Stopping at eom truncates before the answer.
        return [EOT_ID, EOS_ID]

    def render_ids(
        self, messages, *, tools=None, add_generation_prompt: bool = False
    ) -> list[int]:
        # apply_chat_template already emits <|begin_of_text|>, so don't re-add specials.
        text = self._tok.apply_chat_template(
            self._prepare(messages),
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            **self._template_arg(),
            **self._template_kwargs(),
        )
        return self._tok.encode(text, add_special_tokens=False)

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

    @staticmethod
    def _lcp_len(a: list[int], b: list[int]) -> int:
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n

    def _encode_template(self, messages, *, tools, add_generation_prompt) -> list[int]:
        text = self._tok.apply_chat_template(
            self._prepare(messages),
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            **self._template_arg(),
            **self._template_kwargs(),
        )
        return self._tok.encode(text, add_special_tokens=False)

    def render(
        self, messages, *, tools=None, add_generation_prompt: bool = False
    ) -> RenderedTokens:
        """Render with per-token attribution for the trainer loss mask.

        Attribution is computed by growing the prefix one message at a time and
        diffing token ids. Assistant tokens are marked sampled + content (trainable);
        everything else (system/user/tool envelope + generation prompt) is not.

        Known limitation: Muse Glimmer's template picks the message end token by looking at
        the *next* message's role (<|eom|> vs <|eot|>), so two consecutive same-role
        messages (e.g. multiple tool outputs in one step) shift one boundary token.
        Handled via longest-common-prefix diffing; validate byte-exactness against a
        real multi-tool generation before trusting the mask there.
        """
        message_roles = [m["role"] for m in messages]
        message_tool_names = [m.get("name") for m in messages]

        token_ids: list[int] = []
        message_indices: list[int] = []
        sampled_mask: list[bool] = []
        is_content: list[bool] = []

        prev: list[int] = []
        for i in range(len(messages)):
            cur = self._encode_template(
                messages[: i + 1], tools=tools, add_generation_prompt=False
            )
            cpl = self._lcp_len(prev, cur)
            # Re-attribute any prev tail that changed (end-token flip) to message i.
            for _ in range(len(token_ids) - cpl):
                token_ids.pop()
                message_indices.pop()
                sampled_mask.pop()
                is_content.pop()
            sampled = message_roles[i] == "assistant"
            # Split this message's span into template scaffolding vs body. A harmony
            # block is ``<|start|> role [to=recipient] <|message|> body <|eom|>/<|eot|>``;
            # everything up to and including ``<|message|>`` is header the template
            # injects, which the model never sampled -- marking it sampled would put
            # gradient on scaffolding. One assistant message can expand into SEVERAL
            # blocks (a to=self reasoning channel plus one per tool call), so track
            # header state across the whole span instead of splitting once.
            in_header = False
            for tok_id in cur[cpl:]:
                if tok_id == START_ID:
                    in_header = True
                is_scaffold = in_header
                if tok_id == MESSAGE_ID:
                    in_header = False  # header ends *including* this token
                token_ids.append(tok_id)
                message_indices.append(i)
                sampled_mask.append(sampled and not is_scaffold)
                is_content.append(sampled and not is_scaffold)
            prev = cur

        if add_generation_prompt:
            full = self._encode_template(
                messages, tools=tools, add_generation_prompt=True
            )
            for tok_id in full[len(prev) :]:
                token_ids.append(tok_id)
                message_indices.append(len(messages) - 1 if messages else 0)
                sampled_mask.append(False)  # generation prompt is not sampled/content
                is_content.append(False)

        return RenderedTokens(
            token_ids=token_ids,
            message_indices=message_indices,
            sampled_mask=sampled_mask,
            is_content=is_content,
            message_roles=message_roles,
            message_tool_names=message_tool_names,
            multi_modal_data=None,
        )

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages,
        *,
        tools=None,
    ):
        """Extend prev_prompt + sampled_completion with the next turn's tokens.

        Returns None (safe fallback -> caller re-renders) when the sampled completion
        can't be extended byte-exactly: Muse Glimmer wraps each turn in <|start|>...<|eot|>
        and injects <|start|>tool ...<tool_output>..., and getting that continuation
        byte-identical to a fresh full render requires the full message history +
        validation against a real generation. Returning None keeps training correct
        (re-render) at the cost of the extension optimization.

        TODO: implement the byte-exact extension (append turn-close + rendered
        new_messages + next generation prompt) and verify the prefix contract
        against a real generation, then drop the None fallback.
        """
        return None


def register() -> None:
    """Install the muse_glimmer renderer into the ``renderers`` library registry.

    Uses the library's public extension surface -- implement the ``Renderer``
    Protocol, then add the class to ``RENDERER_REGISTRY`` and its config to
    ``_CONFIG_BY_NAME`` -- so ``create_renderer(config_from_name("muse_glimmer"))``
    resolves it. Also maps the ``muse_glimmer`` TorchTitan model name to it, which is what
    ``RendererConfig(name="muse_glimmer")`` looks up.

    Idempotent. Delete this once the renderer is upstreamed to
    PrimeIntellect-ai/renderers (only the _RENDERER_BY_MODEL entry stays).
    """
    from renderers import base as renderers_base, configs as renderers_configs

    from torchtitan.experiments.rl.renderer import _RENDERER_BY_MODEL

    # Populate the library's built-ins first: _populate_registry() early-returns if
    # RENDERER_REGISTRY is already non-empty, so registering before it runs would
    # suppress every built-in renderer.
    renderers_base._populate_registry()

    renderers_configs._CONFIG_BY_NAME.setdefault(
        RENDERER_NAME, MuseGlimmerRendererConfig
    )
    renderers_base.RENDERER_REGISTRY[RENDERER_NAME] = MuseGlimmerRenderer
    _RENDERER_BY_MODEL["muse_glimmer"] = RENDERER_NAME
