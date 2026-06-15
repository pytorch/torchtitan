# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Search-R1 exact-match reward over a tool-calling rollout.

By default this is a pure-EM 0/1 reward: the assistant's final answer matching a gold
answer -> 1.0, else 0. Set ``retrieval_score`` > 0 to also give partial credit when a
gold answer was actually surfaced by a search (rewards searching over answering from
parametric memory).
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass

from torchtitan.experiments.rl.examples.search_r1.data import SearchR1Sample
from torchtitan.experiments.rl.rollout import Rollout
from torchtitan.experiments.rl.rubrics import RewardFn


def _normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation/articles, and collapse whitespace for EM."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _is_exact_match(prediction: str, golden_answers: list[str]) -> bool:
    """True if the normalized prediction exactly equals any normalized gold answer."""
    normalized = _normalize_answer(prediction)
    return any(normalized == _normalize_answer(g) for g in golden_answers)


def _final_answer(rollout: Rollout) -> str | None:
    """The assistant's final answer: the last turn's content, if it isn't a tool call.

    A rollout that ends mid-search (e.g. truncated at the turn budget) has tool calls
    on its last turn and no final answer, so this returns None.
    """
    if not rollout.turns:
        return None
    last = rollout.turns[-1].completion_message or {}
    if last.get("tool_calls"):
        return None
    content = last.get("content")
    return content.strip() if isinstance(content, str) and content.strip() else None


def _retrieval_surfaced_gold(rollout: Rollout, golden_answers: list[str]) -> bool:
    """True if a gold answer (normalized) appears in any search (tool) result."""
    for turn in rollout.turns:
        for message in turn.env_messages or []:
            if message.get("role") != "tool":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            normalized = _normalize_answer(content)
            if any(_normalize_answer(g) in normalized for g in golden_answers):
                return True
    return False


class RewardExactMatch(RewardFn):
    """Exact-match reward over the tool-calling rollout's final answer.

    Default = pure-EM 0/1. Set ``retrieval_score`` > 0 to give partial credit to a
    wrong/unanswered rollout that nonetheless retrieved a gold answer, putting search
    on the gradient so the policy can't max reward by answering from parametric memory.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        score: float = 1.0
        """Reward for a correct final answer."""

        retrieval_score: float = 0.0
        """Credit when a gold answer was surfaced by a search but the final answer is
        wrong/missing. 0 (default) = pure EM."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._score = config.score
        self._retrieval_score = config.retrieval_score

    async def __call__(self, rollout: Rollout, env_input: SearchR1Sample) -> float:
        answer = _final_answer(rollout)
        if answer is not None and _is_exact_match(answer, env_input.golden_answers):
            return self._score
        if self._retrieval_score and _retrieval_surfaced_gold(
            rollout, env_input.golden_answers
        ):
            return self._retrieval_score
        return 0.0


__all__ = ["RewardExactMatch"]
