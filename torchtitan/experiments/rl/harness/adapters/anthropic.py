# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Anthropic Messages adapter that turns an external CLI agent into on-policy RL data.

The adapter exposes an Anthropic ``/v1/messages`` endpoint that the unmodified
agent is pointed at. Per turn it renders the agent's message history into prompt
token ids (Token-In-Token-Out via ``renderer.bridge_to_next_turn``, reusing prior
turns' exact tokens so a multi-turn trajectory packs into ONE episode), samples via
``generate_fn``, and records a ``CapturedTurn`` (prompt/completion ids + logprobs).
``finish_session`` drains the recorded turns for ``rollout_to_training_samples``.

Token legend (this module): ``ids`` = token id lists; a *turn* is one
agent<->model HTTP round trip; a *session* is one agent run (one rollout sibling).
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import secrets
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from aiohttp import web
from renderers import Message, Renderer, ToolSpec
from renderers.base import ToolCallParseStatus

from torchtitan.experiments.rl.rollout.types import GenerateFn

if TYPE_CHECKING:
    # Type-only: importing the generator module pulls in vLLM at import time.
    from torchtitan.experiments.rl.actors.generator import SamplingConfig

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class CapturedTurn:
    """Exact token snapshot for one agent<->model turn (one HTTP round trip)."""

    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    completion_logprobs: list[float]
    # Async RL splits a turn's policy version into a (min, max) span (weights can
    # update mid-decode); a single-shot completion sets both to the same value.
    min_policy_version: int | None
    max_policy_version: int | None
    finish_reason: str | None
    # Whether this turn's prompt continues the previous turn's prompt+completion
    # (TITO-bridged). False marks a history rewrite (compaction) -> episode branch.
    extends_previous: bool


@dataclass
class _Session:
    """Per-rollout adapter state (one Claude Code run)."""

    generate_fn: GenerateFn
    sampling: "SamplingConfig"
    routing_session_id: str
    max_context_tokens: int = 0
    tools: list[ToolSpec] | None = None
    # TITO continuation state.
    last_prompt_ids: list[int] = field(default_factory=list)
    last_completion_ids: list[int] = field(default_factory=list)
    prev_msg_hashes: list[str] = field(default_factory=list)
    prev_system_hash: str = ""
    req_count: int = 0
    turns: list[CapturedTurn] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# Aiohttp app keys.
_ADAPTER_KEY = web.AppKey("adapter", object)


# ---------------------------------------------------------------------------
# Hashing + Anthropic <-> renderers.Message translation (pure helpers)
# ---------------------------------------------------------------------------
def _strip_cache_control(obj: Any) -> Any:
    """Drop ``cache_control`` markers Claude Code sprinkles into blocks; they vary
    turn-to-turn and would defeat prefix-stable hashing."""
    if isinstance(obj, dict):
        return {
            k: _strip_cache_control(v) for k, v in obj.items() if k != "cache_control"
        }
    if isinstance(obj, list):
        return [_strip_cache_control(x) for x in obj]
    return obj


def _hash(obj: Any) -> str:
    payload = json.dumps(
        _strip_cache_control(obj), sort_keys=True, ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _flatten(content: Any) -> str:
    """Anthropic content -> plain text: text / tool_result(content) joined by
    newline; images replaced with a placeholder."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for b in content:
        if isinstance(b, dict):
            t = b.get("type")
            if t == "text":
                parts.append(b.get("text", ""))
            elif t == "tool_result":
                parts.append(_flatten(b.get("content")))
            elif t == "image":
                parts.append("[image omitted]")
        elif isinstance(b, str):
            parts.append(b)
    return "\n".join(p for p in parts if p)


def _anthropic_tools_to_tool_specs(
    anth_tools: list[dict] | None,
) -> list[ToolSpec] | None:
    """Anthropic tool defs -> renderers ``ToolSpec`` (OpenAI function schema)."""
    if not anth_tools:
        return None
    specs: list[ToolSpec] = []
    for t in anth_tools:
        if not isinstance(t, dict) or "name" not in t:
            continue
        specs.append(
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema")
                or t.get("parameters")
                or {"type": "object", "properties": {}},
            }
        )
    return specs or None


def _translate_messages(anth_messages: list[dict], system: Any | None) -> list[Message]:
    """Anthropic messages (+ optional top-level system) -> renderers Messages.

    System content is merged into ONE leading system message (some chat templates,
    e.g. Qwen3.x, hard-require system-first). user tool_result blocks become
    ``role="tool"`` messages; assistant tool_use blocks become OpenAI
    ``tool_calls``. Reasoning is carried as ``reasoning_content``.
    """
    system_parts: list[str] = []
    if system:
        system_parts.append(_flatten(system))
    out: list[Message] = []
    for m in anth_messages:
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), m.get("content")
        if role == "user":
            blocks = (
                content
                if isinstance(content, list)
                else [{"type": "text", "text": _flatten(content)}]
            )
            for b in blocks:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    out.append({"role": "tool", "content": _flatten(b.get("content"))})
                elif isinstance(b, dict) and b.get("type") == "text":
                    out.append({"role": "user", "content": b.get("text", "")})
                else:
                    out.append({"role": "user", "content": _flatten(b)})
        elif role == "assistant":
            texts: list[str] = []
            thinkings: list[str] = []
            tool_calls: list[dict] = []
            blocks = (
                content
                if isinstance(content, list)
                else [{"type": "text", "text": _flatten(content)}]
            )
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "text":
                    texts.append(b.get("text", ""))
                elif bt == "thinking":
                    thinkings.append(b.get("thinking", ""))
                elif bt == "tool_use":
                    tool_calls.append(
                        {
                            "type": "function",
                            "id": b.get("id", f"toolu_{secrets.token_hex(6)}"),
                            "function": {
                                "name": b.get("name", "tool"),
                                "arguments": b.get("input") or {},
                            },
                        }
                    )
            msg: Message = {"role": "assistant", "content": "".join(texts)}
            if thinkings:
                msg["reasoning_content"] = "".join(thinkings)
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)
        elif role == "system":
            system_parts.append(_flatten(content))
    if system_parts:
        out.insert(
            0, {"role": "system", "content": "\n".join(p for p in system_parts if p)}
        )
    return out


# ---------------------------------------------------------------------------
# Completion -> Anthropic content blocks
# ---------------------------------------------------------------------------
def _tool_input(arguments: Any) -> dict:
    """Coerce a parsed tool-call argument value into a JSON object for claude."""
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"_raw": arguments}
    return {}


def _completion_to_blocks(
    renderer: Renderer, token_ids: list[int], tools: list[ToolSpec] | None
) -> tuple[list[dict], str]:
    """Parse completion tokens -> (Anthropic content blocks, stop_reason hint).

    stop_reason hint is ``"tool_use"`` when any OK tool call was parsed, else
    ``"end_turn"``; the caller upgrades it to ``"max_tokens"`` on a length finish.
    """
    parsed = renderer.parse_response(token_ids=token_ids, tools=tools)
    blocks: list[dict] = []
    if parsed.reasoning_content:
        blocks.append({"type": "thinking", "thinking": parsed.reasoning_content})
    if parsed.content:
        blocks.append({"type": "text", "text": parsed.content})
    has_tool = False
    for tc in parsed.tool_calls:
        if tc.status != ToolCallParseStatus.OK or not tc.name:
            continue
        has_tool = True
        blocks.append(
            {
                "type": "tool_use",
                "id": tc.id or f"toolu_{secrets.token_hex(8)}",
                "name": tc.name,
                "input": _tool_input(tc.arguments),
            }
        )
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    return blocks, ("tool_use" if has_tool else "end_turn")


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------
class AnthropicAdapter:
    """Anthropic Messages-compatible HTTP server, backed by a TorchTitan ``generate_fn``.

    Lifecycle::

        adapter = AnthropicAdapter(renderer=renderer)
        await adapter.start()                       # binds host:port on this loop
        adapter.open_session(sid, generate_fn=gf, sampling=sp, routing_session_id=sid)
        ...  # the in-sandbox agent dials /v1/messages with Bearer <sid>
        turns = await adapter.finish_session(sid)   # list[CapturedTurn]
        await adapter.stop()
    """

    def __init__(
        self,
        *,
        renderer: Renderer,
        host: str = "127.0.0.1",
        port: int = 18001,
    ) -> None:
        self.renderer = renderer
        self.host = host
        self.port = port
        self.store: dict[str, _Session] = {}
        self.closed: set[str] = set()
        self.app = web.Application(client_max_size=256 * 1024 * 1024)
        self.app[_ADAPTER_KEY] = self
        self.app.router.add_post("/v1/messages", _handle_messages)
        self.app.router.add_post("/v1/messages/count_tokens", _count_tokens)
        self.app.router.add_get("/healthz", _ok)
        self.app.router.add_get("/v1/models", _ok)
        self._runner: web.AppRunner | None = None

    async def start(self) -> "AnthropicAdapter":
        self._runner = web.AppRunner(self.app, handler_cancellation=True)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        logger.info("[anthropic_adapter] serving on http://%s:%d", self.host, self.port)
        return self

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def open_session(
        self,
        sid: str,
        *,
        generate_fn: GenerateFn,
        sampling: "SamplingConfig",
        routing_session_id: str | None = None,
        max_context_tokens: int = 0,
    ) -> None:
        if sid in self.store:
            raise ValueError(
                f"session_id {sid!r} already open; sids must be unique per run"
            )
        self.closed.discard(sid)
        self.store[sid] = _Session(
            generate_fn=generate_fn,
            sampling=sampling,
            routing_session_id=routing_session_id or sid,
            max_context_tokens=int(max_context_tokens or 0),
        )

    async def finish_session(self, sid: str) -> list[CapturedTurn]:
        """Close the session and return its captured turns (idempotent)."""
        self.closed.add(sid)
        session = self.store.pop(sid, None)
        if session is None:
            return []
        return list(session.turns)


# ---------------------------------------------------------------------------
# Request handling
# ---------------------------------------------------------------------------
def _request_session_id(request: web.Request) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        sid = auth[7:].strip()
        if sid:
            return sid
    api_key = request.headers.get("X-Api-Key")
    if api_key:
        return api_key.strip()
    return "default"


def _plan_prompt(
    adapter: AnthropicAdapter, session: _Session, body: dict
) -> tuple[list[int], bool]:
    """Decide this turn's prompt token ids and whether it extends the previous turn.

    append (TITO bridge): the request continues the last turn's history (same
    system + matching message prefix + an echoed assistant turn), so extend
    ``last_prompt_ids + last_completion_ids`` with only the new tool/user messages.
    new/wipe (full render): first turn or a history rewrite (compaction); render
    the whole conversation -- a wipe deliberately breaks the prefix so
    ``rollout_to_training_samples`` opens a new training-sample branch.
    """
    anth_messages = body.get("messages") or []
    system = body.get("system")
    msg_hashes = [_hash(m) for m in anth_messages]
    system_hash = _hash(system) if system is not None else session.prev_system_hash

    if session.tools is None:
        session.tools = _anthropic_tools_to_tool_specs(body.get("tools"))

    seen = len(session.prev_msg_hashes)
    can_append = (
        session.req_count > 0
        and bool(session.last_completion_ids)
        and system_hash == session.prev_system_hash
        and len(msg_hashes) > seen
        and msg_hashes[:seen] == session.prev_msg_hashes
        and isinstance(anth_messages[seen], dict)
        and anth_messages[seen].get("role") == "assistant"
    )

    if can_append:
        new_messages = _translate_messages(anth_messages[seen + 1 :], system=None)
        # bridge_to_next_turn refuses assistant messages; the tail after an echoed
        # assistant turn is tool/user only, but guard anyway.
        if all(m.get("role") != "assistant" for m in new_messages):
            bridged = adapter.renderer.bridge_to_next_turn(
                previous_prompt_ids=session.last_prompt_ids,
                previous_completion_ids=session.last_completion_ids,
                new_messages=new_messages,
                tools=session.tools,
            )
            if bridged is not None:
                session.prev_msg_hashes = msg_hashes
                session.prev_system_hash = system_hash
                return list(bridged.token_ids), True

    # new / wipe: full re-render.
    chat = _translate_messages(anth_messages, system=system)
    prompt_ids = adapter.renderer.render_ids(
        chat, tools=session.tools, add_generation_prompt=True
    )
    session.prev_msg_hashes = msg_hashes
    session.prev_system_hash = system_hash
    # extends_previous False only when there WAS a previous turn (a real rewrite);
    # the very first turn is the episode's natural start, not a branch.
    return prompt_ids, session.req_count == 0


async def _handle_messages(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    sid = _request_session_id(request)
    adapter: AnthropicAdapter = request.app[_ADAPTER_KEY]
    if sid in adapter.closed:
        return web.Response(status=503, text="session closed")
    session = adapter.store.get(sid)
    if session is None:
        return web.Response(status=404, text=f"unknown session {sid!r}")

    async with session.lock:
        prompt_ids, extends_previous = _plan_prompt(adapter, session, body)

        # Per-turn generation cap: respect the model context budget so a long
        # prompt does not overflow max_model_len mid-trajectory.
        sampling = session.sampling
        if session.max_context_tokens > 0:
            remaining = session.max_context_tokens - len(prompt_ids)
            if remaining <= 0:
                # Prompt already over budget: end the trajectory cleanly with an
                # empty completion (recorded so the merge sees a length finish).
                session.turns.append(
                    CapturedTurn(
                        prompt_token_ids=list(prompt_ids),
                        completion_token_ids=[],
                        completion_logprobs=[],
                        min_policy_version=None,
                        max_policy_version=None,
                        finish_reason="length",
                        extends_previous=extends_previous,
                    )
                )
                # Streaming clients (stream=True) abort with "Stream ended ..." on a
                # one-shot JSON body, so emit SSE for them; JSON otherwise.
                empty_blocks = [{"type": "text", "text": ""}]
                if body.get("stream") is True or "text/event-stream" in (
                    request.headers.get("Accept", "")
                ):
                    return await _stream_response(
                        request, empty_blocks, "max_tokens", len(prompt_ids), 0
                    )
                return web.json_response(
                    _message_response(
                        body, empty_blocks, "max_tokens", len(prompt_ids), 0
                    )
                )
            if remaining < sampling.max_tokens:
                sampling = dataclasses.replace(sampling, max_tokens=remaining)

        # Per-call nonce: a slow one-shot reply can make the client retry the same
        # turn; with handler_cancellation the first handler is cancelled before it
        # appends, so a stable id would collide as "already in flight". The nonce
        # makes each submission distinct.
        request_id = (
            f"{session.routing_session_id}/turn={len(session.turns)}"
            f"/{secrets.token_hex(4)}"
        )
        completion = await session.generate_fn(
            prompt_token_ids=prompt_ids,
            request_id=request_id,
            routing_session_id=session.routing_session_id,
            sampling_config=sampling,
        )
        if completion is None:
            return web.Response(status=502, text="generator returned no completion")

        session.turns.append(
            CapturedTurn(
                prompt_token_ids=list(prompt_ids),
                completion_token_ids=list(completion.token_ids),
                completion_logprobs=list(completion.token_logprobs),
                min_policy_version=completion.min_policy_version,
                max_policy_version=completion.max_policy_version,
                finish_reason=completion.finish_reason,
                extends_previous=extends_previous,
            )
        )
        session.last_prompt_ids = list(prompt_ids)
        session.last_completion_ids = list(completion.token_ids)
        session.req_count += 1

        logger.info(
            "[anthropic_adapter] %s turn=%d: prompt=%d max_tokens=%d out=%d finish=%s",
            session.routing_session_id,
            len(session.turns) - 1,
            len(prompt_ids),
            sampling.max_tokens,
            len(completion.token_ids),
            completion.finish_reason,
        )

        blocks, stop = _completion_to_blocks(
            adapter.renderer, completion.token_ids, session.tools
        )
        if completion.finish_reason == "length":
            stop = "max_tokens"
        in_tok, out_tok = len(prompt_ids), len(completion.token_ids)

    if body.get("stream") is True or "text/event-stream" in request.headers.get(
        "Accept", ""
    ):
        return await _stream_response(request, blocks, stop, in_tok, out_tok)
    return web.json_response(_message_response(body, blocks, stop, in_tok, out_tok))


def _message_response(
    body: dict, blocks: list[dict], stop_reason: str, in_tok: int, out_tok: int
) -> dict:
    return {
        "id": f"msg_{secrets.token_hex(12)}",
        "type": "message",
        "role": "assistant",
        "model": body.get("model", "titan-actor"),
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }


async def _stream_response(
    request: web.Request,
    blocks: list[dict],
    stop_reason: str,
    in_tok: int,
    out_tok: int,
) -> web.StreamResponse:
    """Emit the reply as an Anthropic Messages SSE stream (message_start,
    per-block start/delta/stop, message_delta, message_stop). The adapter only
    streams after the full generation, so this is one-shot SSE."""
    out = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await out.prepare(request)

    async def send(event: str, data: dict) -> None:
        await out.write(
            f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()
        )

    await send(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": f"msg_{secrets.token_hex(12)}",
                "type": "message",
                "role": "assistant",
                "model": "titan-actor",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": in_tok, "output_tokens": 0},
            },
        },
    )
    for idx, block in enumerate(blocks):
        bt = block["type"]
        if bt == "thinking":
            start = {"type": "thinking", "thinking": ""}
            delta = {"type": "thinking_delta", "thinking": block["thinking"]}
        elif bt == "text":
            start = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        else:  # tool_use
            start = {
                "type": "tool_use",
                "id": block["id"],
                "name": block["name"],
                "input": {},
            }
            delta = {
                "type": "input_json_delta",
                "partial_json": json.dumps(block["input"], ensure_ascii=False),
            }
        await send(
            "content_block_start",
            {"type": "content_block_start", "index": idx, "content_block": start},
        )
        await send(
            "content_block_delta",
            {"type": "content_block_delta", "index": idx, "delta": delta},
        )
        await send("content_block_stop", {"type": "content_block_stop", "index": idx})

    await send(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
        },
    )
    await send("message_stop", {"type": "message_stop"})
    return out


# count_tokens runs every turn; claude uses it as a hint, not a hard budget.
async def _count_tokens(request: web.Request) -> web.Response:
    return web.json_response({"input_tokens": 0})


async def _ok(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})
