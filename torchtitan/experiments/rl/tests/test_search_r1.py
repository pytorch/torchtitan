# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the Search-R1 example: env action parsing + EM/format rubric.

No GPU, no retrieval server: the retrieval call is monkeypatched.
"""

import asyncio

from torchtitan.experiments.rl.examples.search_r1 import (
    env as sr1_env,
    RewardExactMatch,
    SearchR1Env,
    SearchR1Sample,
)
from torchtitan.experiments.rl.rollout import Rollout, RolloutStatus, RolloutTurn


def _build_env(monkeypatch, question="Who wrote Hamlet?"):
    async def _fake_search(query, *, url, topk, timeout_s=60.0):
        return f"Doc 1(Title: T) results for {query}"

    monkeypatch.setattr(sr1_env, "_search", _fake_search)
    return SearchR1Env(
        SearchR1Env.Config(),
        env_input=SearchR1Sample(question=question, golden_answers=["Shakespeare"]),
    )


def test_env_search_action_injects_information(monkeypatch):
    env = _build_env(monkeypatch)
    out = asyncio.run(
        env.step(
            {
                "role": "assistant",
                "content": "<think>hm</think><search>Hamlet author</search>",
            }
        )
    )
    assert out.done is False
    assert len(out.env_messages) == 1
    content = out.env_messages[0]["content"]
    assert "<information>" in content and "</information>" in content
    assert "Hamlet author" in content


def test_env_answer_action_terminates(monkeypatch):
    env = _build_env(monkeypatch)
    out = asyncio.run(
        env.step({"role": "assistant", "content": "<answer>Shakespeare</answer>"})
    )
    assert out.done is True
    assert out.env_messages == []


def test_env_invalid_action_nudges(monkeypatch):
    env = _build_env(monkeypatch)
    out = asyncio.run(env.step({"role": "assistant", "content": "no tags here"}))
    assert out.done is False
    assert "invalid" in out.env_messages[0]["content"].lower()


def _answer_rollout(text: str) -> Rollout:
    return Rollout(
        group_id="g",
        sample_id="s",
        status=RolloutStatus.COMPLETED,
        turns=[
            RolloutTurn(
                prompt_token_ids=[1],
                completion_token_ids=[2],
                completion_logprobs=[-0.1],
                completion_message={"role": "assistant", "content": text},
            )
        ],
    )


def test_reward_em_exact_match_normalized():
    em = RewardExactMatch(RewardExactMatch.Config())
    ex = SearchR1Sample(question="q", golden_answers=["Shakespeare"])
    # normalization lowercases + strips punctuation/articles.
    r = asyncio.run(em(_answer_rollout("<answer>The Shakespeare.</answer>"), ex))
    assert r == 1.0


def test_reward_em_mismatch():
    em = RewardExactMatch(RewardExactMatch.Config())
    ex = SearchR1Sample(question="q", golden_answers=["Shakespeare"])
    r = asyncio.run(em(_answer_rollout("<answer>Marlowe</answer>"), ex))
    assert r == 0.0


def test_reward_em_uses_last_answer():
    em = RewardExactMatch(RewardExactMatch.Config())
    ex = SearchR1Sample(question="q", golden_answers=["Paris"])
    r = asyncio.run(
        em(_answer_rollout("<answer>London</answer> ... <answer>Paris</answer>"), ex)
    )
    assert r == 1.0


def _search_rollout(
    answer: str, *, info: str = "passage", think: str = "ok"
) -> Rollout:
    """A valid two-turn trajectory: search turn (with injected <information>) then
    an answer turn — reconstructs to think->search->information->think->answer."""
    return Rollout(
        group_id="g",
        sample_id="s",
        status=RolloutStatus.COMPLETED,
        turns=[
            RolloutTurn(
                prompt_token_ids=[1],
                completion_token_ids=[2],
                completion_logprobs=[-0.1],
                completion_message={
                    "role": "assistant",
                    "content": f"<think>{think}</think><search>q</search>",
                },
                env_messages=[
                    {
                        "role": "user",
                        "content": f"\n\n<information>{info}</information>\n\n",
                    }
                ],
            ),
            RolloutTurn(
                prompt_token_ids=[3],
                completion_token_ids=[4],
                completion_logprobs=[-0.1],
                completion_message={
                    "role": "assistant",
                    "content": f"<think>{think}</think><answer>{answer}</answer>",
                },
            ),
        ],
    )


def _approx(a: float, b: float) -> bool:
    return abs(a - b) < 1e-9


# Fine-grained (graded) opt-in config: search/format/retrieval on the gradient.
_GRADED = dict(structure_format_score=0.2, retrieval_score=0.1, final_format_score=0.1)

EX = SearchR1Sample(question="q", golden_answers=["Paris"])


# --- default = pure-EM 0/1 (sub-scores 0): correct -> 1.0, else 0 ---
def test_searchr1_default_is_pure_em_correct():
    r = RewardExactMatch(RewardExactMatch.Config())
    # both a searched and a bare correct answer score 1.0 under the default.
    assert _approx(
        asyncio.run(r(_search_rollout("Paris", info="capital is Paris"), EX)), 1.0
    )
    assert _approx(asyncio.run(r(_answer_rollout("<answer>Paris</answer>"), EX)), 1.0)


def test_searchr1_default_is_pure_em_wrong():
    r = RewardExactMatch(RewardExactMatch.Config())
    assert _approx(
        asyncio.run(r(_search_rollout("London", info="capital is Paris"), EX)), 0.0
    )
    assert _approx(asyncio.run(r(_answer_rollout("<answer>London</answer>"), EX)), 0.0)


# --- fine-grained (graded) mode: bare correct < searched correct, partial credit ---
def test_searchr1_graded_searched_correct_full():
    r = RewardExactMatch(RewardExactMatch.Config(**_GRADED))
    assert _approx(
        asyncio.run(r(_search_rollout("Paris", info="capital is Paris"), EX)), 1.0
    )


def test_searchr1_graded_bare_correct_penalized():
    # The anti-collapse case: correct but no <think>/<search> -> 1.0 - 0.2 = 0.8.
    r = RewardExactMatch(RewardExactMatch.Config(**_GRADED))
    assert _approx(asyncio.run(r(_answer_rollout("<answer>Paris</answer>"), EX)), 0.8)


def test_searchr1_graded_wrong_valid_with_retrieval():
    r = RewardExactMatch(RewardExactMatch.Config(**_GRADED))
    assert _approx(
        asyncio.run(r(_search_rollout("London", info="capital is Paris"), EX)), 0.3
    )


def test_searchr1_graded_wrong_valid_no_retrieval():
    r = RewardExactMatch(RewardExactMatch.Config(**_GRADED))
    assert _approx(
        asyncio.run(r(_search_rollout("London", info="irrelevant"), EX)), 0.2
    )


def test_searchr1_graded_wrong_invalid_floor():
    r = RewardExactMatch(RewardExactMatch.Config(**_GRADED))
    assert _approx(asyncio.run(r(_answer_rollout("<answer>London</answer>"), EX)), 0.1)
