# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the Search-R1 example: tool-call env + exact-match rubric.

No GPU, no retrieval server: the retrieval call is monkeypatched.
"""

import asyncio

import numpy as np
from renderers import CustomRendererConfig
from renderers.base import ParsedToolCall

from torchtitan.components.renderer import RenderersConfigAdapter
from torchtitan.config import ConfigLoader
from torchtitan.rl.controller.controller import Controller
from torchtitan.rl.examples.search_r1 import (
    env as sr1_env,
    RewardExactMatch,
    SearchR1Env,
    SearchR1Sample,
)
from torchtitan.rl.rollout import Rollout, RolloutStatus, RolloutTurn
from torchtitan.rl.rollout.types import RolloutTurnID


def test_muse_glimmer_search_recipe_uses_worker_loadable_renderer():
    config = ConfigLoader().load(
        [
            "--module",
            "torchtitan_recipes.rl.search_r1",
            "--config",
            "rl_grpo_muse_glimmer_30b_search_r1",
        ]
    )

    assert isinstance(config, Controller.Config)
    assert isinstance(config.renderer, RenderersConfigAdapter)
    renderer_config = config.renderer.get_renderers_config()
    assert isinstance(renderer_config, CustomRendererConfig)
    assert renderer_config.import_path == (
        "torchtitan.models.muse_glimmer.renderer.MuseGlimmerRenderer"
    )


def _build_env(monkeypatch, question="Who wrote Hamlet?"):
    async def _fake_search(query, *, url, topk, timeout_s=60.0):
        return f"Doc 1(Title: T) results for {query}"

    monkeypatch.setattr(sr1_env, "_search", _fake_search)
    return SearchR1Env(
        SearchR1Env.Config(),
        env_input=SearchR1Sample(question=question, golden_answers=["Shakespeare"]),
    )


def _search_call(query, call_id: str = "c1") -> ParsedToolCall:
    """A renderer-parsed search tool call (``arguments`` may be a dict or JSON str)."""
    return ParsedToolCall(raw="", name="search", arguments={"query": query}, id=call_id)


def test_env_init_exposes_search_tool(monkeypatch):
    env = _build_env(monkeypatch)
    out = asyncio.run(env.init())
    assert [t["name"] for t in out.tools] == ["search"]
    assert "Hamlet" in out.init_prompt_messages[0]["content"]


def test_env_tool_call_returns_tool_message(monkeypatch):
    env = _build_env(monkeypatch)
    out = asyncio.run(
        env.step({"role": "assistant", "tool_calls": [_search_call("Hamlet author")]})
    )
    assert out.done is False
    assert len(out.env_messages) == 1
    msg = out.env_messages[0]
    assert msg["role"] == "tool"
    assert "Hamlet author" in msg["content"]


def test_env_no_tool_call_terminates(monkeypatch):
    env = _build_env(monkeypatch)
    out = asyncio.run(env.step({"role": "assistant", "content": "Shakespeare"}))
    assert out.done is True
    assert out.env_messages == []


def test_env_handles_json_string_arguments(monkeypatch):
    env = _build_env(monkeypatch)
    call = ParsedToolCall(
        raw="", name="search", arguments='{"query": "stringified"}', id="c1"
    )
    out = asyncio.run(env.step({"role": "assistant", "tool_calls": [call]}))
    assert "stringified" in out.env_messages[0]["content"]


def _rollout(*, answer: str | None, tool_results: list[str] | None = None) -> Rollout:
    """Build a rollout: one search turn (with a tool result) then a final-answer turn.

    ``answer=None`` means the rollout ended on a tool call (truncated, no final answer).
    """
    turns = []
    for res in tool_results or []:
        turns.append(
            RolloutTurn(
                rollout_id=RolloutTurnID(group_id=0, rollout_id=0, turn_id=len(turns)),
                prompt_prefix_len=0,
                prompt_delta_token_ids=np.array([1], dtype=np.int32),
                completion_token_ids=np.array([2], dtype=np.int32),
                completion_logprobs=np.array([-0.1], dtype=np.float32),
                completion_message={
                    "role": "assistant",
                    "tool_calls": [_search_call("q")],
                },
                env_messages=[{"role": "tool", "content": res}],
            )
        )
    final = (
        {"role": "assistant", "tool_calls": [_search_call("q")]}
        if answer is None
        else {"role": "assistant", "content": answer}
    )
    turns.append(
        RolloutTurn(
            rollout_id=RolloutTurnID(group_id=0, rollout_id=0, turn_id=len(turns)),
            prompt_prefix_len=0,
            prompt_delta_token_ids=np.array([3], dtype=np.int32),
            completion_token_ids=np.array([4], dtype=np.int32),
            completion_logprobs=np.array([-0.1], dtype=np.float32),
            completion_message=final,
        )
    )
    return Rollout(
        group_id=0, rollout_id=0, status=RolloutStatus.COMPLETED, turns=turns
    )


EX = SearchR1Sample(question="q", golden_answers=["Paris"])


def test_reward_final_answer_exact_match():
    r = RewardExactMatch(RewardExactMatch.Config())
    # normalization lowercases + strips punctuation/articles.
    assert asyncio.run(r(_rollout(answer="The Paris."), EX)) == 1.0


def test_reward_final_answer_mismatch():
    r = RewardExactMatch(RewardExactMatch.Config())
    assert asyncio.run(r(_rollout(answer="London"), EX)) == 0.0


def test_reward_truncated_no_answer_is_zero():
    r = RewardExactMatch(RewardExactMatch.Config())
    assert (
        asyncio.run(r(_rollout(answer=None, tool_results=["capital is Paris"]), EX))
        == 0.0
    )


def test_reward_retrieval_credit_when_wrong_but_gold_retrieved():
    r = RewardExactMatch(RewardExactMatch.Config(retrieval_score=0.1))
    # wrong final answer, but a search surfaced the gold -> partial credit.
    got = asyncio.run(
        r(_rollout(answer="London", tool_results=["capital is Paris"]), EX)
    )
    assert abs(got - 0.1) < 1e-9


def test_reward_no_retrieval_credit_by_default():
    r = RewardExactMatch(RewardExactMatch.Config())  # retrieval_score=0
    got = asyncio.run(
        r(_rollout(answer="London", tool_results=["capital is Paris"]), EX)
    )
    assert got == 0.0


def test_reward_no_search_penalty_anti_closed_book():
    r = RewardExactMatch(RewardExactMatch.Config(no_search_penalty=0.2))
    # correct but never searched (closed-book) -> 1.0 - 0.2 = 0.8
    closed_book = asyncio.run(r(_rollout(answer="Paris"), EX))
    assert abs(closed_book - 0.8) < 1e-9
    # correct and searched -> full 1.0
    searched = asyncio.run(
        r(_rollout(answer="Paris", tool_results=["capital is Paris"]), EX)
    )
    assert searched == 1.0
