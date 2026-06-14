# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Search-R1 exact-match reward, with an optional graded format/retrieval mode.

By default this is a pure-EM 0/1 reward: the extracted ``<answer>`` matching a gold
answer -> 1.0, everything else -> 0 (the format/retrieval sub-scores default to 0).

Set the sub-scores > 0 (``structure_format_score`` / ``retrieval_score`` /
``final_format_score``) for the graded reward: it credits the whole
``<think>→<search>→<information>→<answer>`` trajectory and retrieval correctness, so a
bare correct answer scores less than a searched one and search is on the gradient — a
fix for the closed-book reward hacking that pure EM allows on a model that can answer
from parametric memory.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass

from torchtitan.experiments.rl.examples.search_r1.data import SearchR1Sample
from torchtitan.experiments.rl.rollout import Rollout
from torchtitan.experiments.rl.rubrics import RewardFn


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_INFORMATION_RE = re.compile(r"<information>(.*?)</information>", re.DOTALL)
_TAG_RE = re.compile(r"</?(?:think|search|information|answer)>")
_TAG_SPLIT_RE = re.compile(r"(</?(?:think|search|information|answer)>)")


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


def _all_assistant_text(rollout: Rollout) -> str:
    """Concatenate every assistant completion's text across the rollout's turns."""
    texts: list[str] = []
    for turn in rollout.turns:
        content = (turn.completion_message or {}).get("content")
        if isinstance(content, str):
            texts.append(content)
    return "\n".join(texts)


def _trajectory_text(rollout: Rollout) -> str:
    """Reconstruct the full Search-R1 trajectory text for format/retrieval scoring.

    Interleaves, per turn, the assistant completion (``<think>``/``<search>``/
    ``<answer>``) with the env-injected ``<information>...</information>`` messages.
    """
    parts: list[str] = []
    for turn in rollout.turns:
        content = (turn.completion_message or {}).get("content")
        if isinstance(content, str):
            parts.append(content)
        for msg in turn.env_messages or []:
            env_content = msg.get("content")
            if isinstance(env_content, str):
                parts.append(env_content)
    return "".join(parts)


def _extract_answer(text: str) -> str | None:
    """Return the last ``<answer>...</answer>`` content in the text, or None."""
    matches = _ANSWER_RE.findall(text)
    return matches[-1].strip() if matches else None


def _is_retrieval_correct(text: str, golden_answers: list[str]) -> bool:
    """True if any gold answer (normalized) appears inside a retrieved
    ``<information>`` block (i.e. the search actually surfaced the answer)."""
    for block in _INFORMATION_RE.findall(text):
        norm_block = _normalize_answer(block)
        if any(_normalize_answer(g) in norm_block for g in golden_answers):
            return True
    return False


def _is_valid_sequence(text: str) -> bool:
    """Validate the ``think → search → information → think → ... → answer`` structure.

    Requires balanced tags, the correct tag order, and only whitespace between tags.
    """
    # Balanced open/close tags.
    for tag in ("think", "search", "information", "answer"):
        if len(re.findall(f"<{tag}>", text)) != len(re.findall(f"</{tag}>", text)):
            return False

    state = "start"  # start -> in_think -> after_think -> in_search -> ... -> end
    for part in _TAG_SPLIT_RE.split(text):
        if not part.strip():
            continue
        if _TAG_RE.fullmatch(part):
            if part == "<think>" and state in ("start", "information"):
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<search>" and state == "after_think":
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False  # tag out of order
        else:
            # Non-tag text: only allowed inside a tag's body.
            if state not in ("in_think", "in_search", "in_information", "in_answer"):
                return False  # extraneous content between tags
    return state == "end"


def compute_score_em(
    trajectory: str,
    golden_answers: list[str],
    *,
    score: float = 1.0,
    structure_format_score: float = 0.0,
    retrieval_score: float = 0.0,
    final_format_score: float = 0.0,
) -> float:
    """Search-R1 exact-match reward.

    With all sub-scores 0 (default) this is pure EM: a correct answer -> ``score``
    (1.0), else 0. Set the sub-scores > 0 for the graded reward, which credits the
    search trajectory and retrieval::

        correct answer + valid format              -> score                     (1.0)
        correct answer + invalid format            -> score - structure_format  (0.8)
        wrong/no answer + valid format + retrieval -> structure + retrieval     (0.3)
        wrong/no answer + valid format             -> structure_format          (0.2)
        wrong answer    + invalid format           -> final_format_score        (0.1)
        no answer       + invalid format           -> 0.0

    Args:
        trajectory: reconstructed ``<think>/<search>/<information>/<answer>`` text.
        golden_answers: accepted gold answers for EM.
        score: reward for a correct answer with valid format.
        structure_format_score: credit for a valid multi-turn format.
        retrieval_score: extra credit when a gold answer was actually retrieved.
        final_format_score: floor credit for a wrong answer in an invalid format.
    """
    valid_format = _is_valid_sequence(trajectory)
    retrieval_correct = (
        _is_retrieval_correct(trajectory, golden_answers) if valid_format else False
    )
    answer = _extract_answer(trajectory)

    if answer is None:
        if valid_format:
            return structure_format_score + (
                retrieval_score if retrieval_correct else 0.0
            )
        return 0.0
    if _is_exact_match(answer, golden_answers):
        return score if valid_format else score - structure_format_score
    if valid_format:
        return structure_format_score + (retrieval_score if retrieval_correct else 0.0)
    return final_format_score


class RewardExactMatch(RewardFn):
    """Search-R1 reward: EM + (optional) format state-machine + retrieval credit.

    Default (all sub-scores 0) = pure-EM 0/1. Set the sub-scores > 0 for the graded
    reward, which puts search on the gradient (a bare correct answer scores
    ``score - structure_format_score`` < a searched one ``score``), so the policy
    can't max reward by answering from parametric memory without searching.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        score: float = 1.0
        """Reward for a correct answer with a valid search trajectory."""

        structure_format_score: float = 0.0
        """Credit for a valid ``think→search→…→answer`` format. 0 (default) = pure EM.
        Set > 0 (e.g. 0.2) to make a bare correct answer
        (``score - structure_format_score``) worth less than a searched one."""

        retrieval_score: float = 0.0
        """Extra credit when a gold answer actually appears in a retrieved
        ``<information>`` block (rewards searching *accurately*). 0 (default) = pure EM."""

        final_format_score: float = 0.0
        """Floor credit for a wrong answer produced in an invalid format. 0 = pure EM."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._score = config.score
        self._structure_format_score = config.structure_format_score
        self._retrieval_score = config.retrieval_score
        self._final_format_score = config.final_format_score

    async def __call__(self, rollout: Rollout, env_input: SearchR1Sample) -> float:
        return compute_score_em(
            _trajectory_text(rollout),
            env_input.golden_answers,
            score=self._score,
            structure_format_score=self._structure_format_score,
            retrieval_score=self._retrieval_score,
            final_format_score=self._final_format_score,
        )


class RewardAnswerEM(RewardFn):
    """1.0 if the final ``<answer>`` exactly matches any gold answer (normalized), else 0.

    Metric-only: configure with ``weight=0.0`` to log pure EM alongside the training
    reward without affecting the gradient.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    async def __call__(self, rollout: Rollout, env_input: SearchR1Sample) -> float:
        answer = _extract_answer(_all_assistant_text(rollout))
        if answer is None:
            return 0.0
        return 1.0 if _is_exact_match(answer, env_input.golden_answers) else 0.0


__all__ = ["RewardExactMatch", "RewardAnswerEM", "compute_score_em"]
