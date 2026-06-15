# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

import aiohttp

from renderers import Message, ToolSpec
from renderers.base import ParsedToolCall

from torchtitan.experiments.rl.environment import (
    MessageEnv,
    MessageEnvInitOutput,
    MessageEnvStepOutput,
)
from torchtitan.experiments.rl.examples.search_r1.data import SearchR1Sample

logger = logging.getLogger(__name__)


# The `search` tool the model calls (OpenAI function-calling schema). The renderer
# injects this into the prompt and parses the model's `<tool_call>` back into
# `completion_message["tool_calls"]`; the model uses native thinking + tool calling.
SEARCH_TOOL: ToolSpec = {
    "name": "search",
    "description": (
        "Search a Wikipedia-derived knowledge base and return the top passages for "
        "a query. Use it whenever you need external facts to answer the question."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The natural-language search query.",
            },
        },
        "required": ["query"],
    },
}

INSTRUCTION = (
    "Answer the question. Use the search tool to look up any facts you are unsure of. "
    "When you know the answer, reply with only the answer itself — just the entity or "
    'phrase, with no explanation or full sentence. For example: "Beijing".'
    "\n\nQuestion: "
)


def _passages_to_string(docs: list[dict]) -> str:
    """Format retrieved docs into the block returned to the model.

    Each doc's ``contents`` is ``"<title>\\n<body>"``. Example::

        [{"contents": "Eiffel Tower\\nA tower in Paris."}]
        -> "Doc 1(Title: Eiffel Tower) A tower in Paris.\\n"
    """
    out = ""
    for i, doc in enumerate(docs):
        contents = doc.get("contents", "") if isinstance(doc, dict) else ""
        lines = contents.split("\n")
        title = lines[0] if lines else ""
        text = "\n".join(lines[1:])
        out += f"Doc {i + 1}(Title: {title}) {text}\n"
    return out


async def _search(query: str, *, url: str, topk: int, timeout_s: float = 60.0) -> str:
    """Retrieve the top-``topk`` passages for ``query``, formatted for the model.

    POSTs ``{"queries": [query], "topk": topk}`` and reads back
    ``{"result": [[{"contents": ...}, ...]]}``. On any transport/server error
    returns ``""`` instead of raising, so one flaky request doesn't crash the rollout.
    """
    payload = {"queries": [query], "topk": topk, "return_scores": False}
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
    except Exception as exc:
        # Warn (don't raise) so one flaky request doesn't crash the rollout — but a
        # persistent failure is visible rather than silently training on empty results.
        logger.warning("search request to %s failed: %s", url, exc)
        return ""
    results = data.get("result") or [[]]
    return _passages_to_string(results[0])


def _query_from_tool_call(tool_call: ParsedToolCall) -> str:
    """Pull the ``query`` argument out of a renderer-parsed ``search`` tool call."""
    arguments = tool_call.arguments
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return ""
    if isinstance(arguments, dict):
        query = arguments.get("query", "")
        return query if isinstance(query, str) else ""
    return ""


class SearchR1Env(MessageEnv):
    """Multi-turn open-domain QA env with a ``search`` tool.

    Each assistant turn either calls the ``search`` tool — which the env answers with
    a ``role="tool"`` message containing the retrieved passages — or stops calling
    tools, which ends the rollout with the assistant's reply as the final answer. The
    model uses native thinking and tool calling (the renderer renders the tool schema
    and parses the tool calls); the per-rollout turn and token budget is enforced by
    ``TokenEnv``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MessageEnv.Config):
        search_url: str = "http://127.0.0.1:8000/retrieve"
        """URL of the local dense retrieval server."""

        topk: int = 3
        """Number of passages to retrieve per search query."""

        timeout_s: float = 60.0
        """Per-request timeout for a retrieval call, in seconds."""

    def __init__(self, config: Config, *, env_input: SearchR1Sample) -> None:
        self._question = env_input.question
        self._search_url = config.search_url
        self._topk = config.topk
        self._timeout_s = config.timeout_s

    async def init(self) -> MessageEnvInitOutput:
        return MessageEnvInitOutput(
            init_prompt_messages=[
                {"role": "user", "content": INSTRUCTION + self._question},
            ],
            tools=[SEARCH_TOOL],
        )

    async def step(self, completion_message: Message) -> MessageEnvStepOutput:
        # The renderer parses tool calls into ParsedToolCall objects (.name/.arguments).
        tool_calls: list[ParsedToolCall] = completion_message.get("tool_calls") or []
        if not tool_calls:
            # No tool call -> the assistant's reply is the final answer.
            return MessageEnvStepOutput(done=True)

        # Run every search call concurrently, then answer each with a tool message.
        passages = await asyncio.gather(
            *(
                _search(
                    _query_from_tool_call(tc),
                    url=self._search_url,
                    topk=self._topk,
                    timeout_s=self._timeout_s,
                )
                for tc in tool_calls
            )
        )
        env_messages: list[Message] = [{"role": "tool", "content": p} for p in passages]
        return MessageEnvStepOutput(env_messages=env_messages, done=False)
