# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
from dataclasses import dataclass

import aiohttp

from renderers import Message

from torchtitan.experiments.rl.environment import (
    MessageEnv,
    MessageEnvInitOutput,
    MessageEnvStepOutput,
)
from torchtitan.experiments.rl.examples.search_r1.data import SearchR1Sample


# Search-R1 instruction prompt. The model reasons in <think>, searches via
# <search> query </search> (results come back in <information>...</information>),
# and answers in <answer>...</answer>.
SEARCH_R1_INSTRUCTION = (
    "Answer the given question. You must conduct reasoning inside <think> and "
    "</think> first every time you get new information. After reasoning, if you "
    "find you lack some knowledge, you can call a search engine by <search> query "
    "</search> and it will return the top searched results between <information> "
    "and </information>. You can search as many times as your want. If you find no "
    "further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, <answer> "
    "Beijing </answer>. Question: "
)

_INVALID_ACTION_MESSAGE = (
    "\nMy previous action is invalid. If I want to search, I should put the query "
    "between <search> and </search>. If I want to give the final answer, I should "
    "put the answer between <answer> and </answer>. Let me try again.\n"
)

# First well-formed <search>...</search> or <answer>...</answer> block.
_ACTION_RE = re.compile(r"<(search|answer)>(.*?)</\1>", re.DOTALL)


# ---------------------------------------------------------------------------
# Local dense retrieval client (`retrieval_server.py`).
# `POST {url}` accepts {"queries": [...], "topk": k, "return_scores": false} and
# returns {"result": [[{"id", "contents"}, ...]]} (one list per query).
# ---------------------------------------------------------------------------
def _passages_to_string(docs: list[dict]) -> str:
    """Format retrieved docs into the ``Doc i(Title: ...) text`` block Search-R1
    feeds back to the model. Each doc's ``contents`` is ``"<title>\\n<text>"``."""
    out = ""
    for i, doc in enumerate(docs):
        contents = doc.get("contents", "") if isinstance(doc, dict) else ""
        lines = contents.split("\n")
        title = lines[0] if lines else ""
        text = "\n".join(lines[1:])
        out += f"Doc {i + 1}(Title: {title}) {text}\n"
    return out


async def _search(query: str, *, url: str, topk: int, timeout_s: float = 60.0) -> str:
    """Retrieve the top-``topk`` passages for ``query`` and format them as a string.

    Returns an empty string on any transport/server error so a single flaky
    retrieval degrades the rollout (empty ``<information>``) instead of crashing it.
    """
    payload = {"queries": [query], "topk": topk, "return_scores": False}
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
    except Exception:
        return ""
    results = data.get("result") or [[]]
    return _passages_to_string(results[0])


class SearchR1Env(MessageEnv):
    """Multi-turn open-domain QA env with a search tool (Search-R1).

    Each assistant turn either issues a ``<search>query</search>`` — which the env
    answers with a ``<information>...</information>`` user message — or gives a final
    ``<answer>...</answer>``, which ends the rollout. Malformed turns get a corrective
    nudge. The rollout also ends once ``max_assistant_turns`` is reached (the env's
    own turn budget, since the framework's ``TokenEnv`` only bounds total tokens).

    Uses the text-tag protocol (not function-calling tools); pair it with a
    renderer configured with ``enable_thinking=False`` so the model's ``<think>``
    tags stay in the completion text, and a generator ``SamplingConfig`` with
    ``stop=["</search>", "</answer>"]`` so each turn halts at its action tag.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MessageEnv.Config):
        search_url: str = "http://127.0.0.1:8000/retrieve"
        """URL of the local dense retrieval server."""

        topk: int = 3
        """Number of passages to retrieve per search query."""

        max_assistant_turns: int = 4
        """Per-rollout turn budget: end the rollout after this many assistant turns
        even without an ``<answer>`` (then it scores 0). Bounds runaway search loops."""

    def __init__(self, config: Config, *, env_input: SearchR1Sample) -> None:
        self._question = env_input.question
        self._search_url = config.search_url
        self._topk = config.topk
        self._max_assistant_turns = config.max_assistant_turns
        self._num_turns = 0

    async def init(self) -> MessageEnvInitOutput:
        return MessageEnvInitOutput(
            init_prompt_messages=[
                {"role": "user", "content": SEARCH_R1_INSTRUCTION + self._question},
            ]
        )

    async def step(self, completion_message: Message) -> MessageEnvStepOutput:
        self._num_turns += 1

        # The text-tag protocol only uses plain-text content.
        content = completion_message.get("content")
        content = content if isinstance(content, str) else ""
        match = _ACTION_RE.search(content)
        action = match.group(1) if match else None
        argument = match.group(2).strip() if match else ""

        if action == "answer":
            # Final answer: end the rollout.
            return MessageEnvStepOutput(done=True)

        # Out of turn budget: end the rollout (no further search/answer allowed).
        if self._num_turns >= self._max_assistant_turns:
            return MessageEnvStepOutput(done=True)

        if action == "search":
            passages = await _search(argument, url=self._search_url, topk=self._topk)
            observation = f"\n\n<information>{passages.strip()}</information>\n\n"
            return MessageEnvStepOutput(
                env_messages=[{"role": "user", "content": observation}],
                done=False,
            )

        # No valid action: nudge the model and let it try again.
        return MessageEnvStepOutput(
            env_messages=[{"role": "user", "content": _INVALID_ACTION_MESSAGE}],
            done=False,
        )
