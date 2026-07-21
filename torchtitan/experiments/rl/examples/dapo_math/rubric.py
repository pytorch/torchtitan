# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from math_verify import LatexExtractionConfig, LatexNormalizationConfig, parse, verify
from math_verify.errors import TimeoutException

from torchtitan.experiments.rl.examples.dapo_math.data import DapoMathSample
from torchtitan.experiments.rl.rollout import Rollout
from torchtitan.experiments.rl.rubrics import RewardFn


_FINAL_ANSWER_EXTRACTION = [
    LatexExtractionConfig(
        normalization_config=LatexNormalizationConfig(units=True),
        boxed_match_priority=0,
        try_extract_without_anchor=False,
    )
]


def score_math_response(response: str, ground_truth: str) -> float:
    """Score an `Answer:` or `\\boxed{}` expression with Math-Verify.

    Args:
        response: Model response containing a marked final answer.
        ground_truth: Expected answer from the dataset.

    Example:
        score_math_response("work\nAnswer: $34$", "34")  # 1.0
    """
    try:
        gold = parse(ground_truth)
        prediction = parse(
            response,
            extraction_config=_FINAL_ANSWER_EXTRACTION,
            extraction_mode="first_match",
        )
        return float(bool(gold) and verify(gold, prediction))
    except (Exception, TimeoutException):
        # Model output is untrusted; malformed LaTeX is an incorrect answer, not a
        # training-loop failure. Math-Verify raises `TimeoutException` from BaseException.
        return 0.0


class RewardMathVerify(RewardFn):
    """Binary reward for a mathematically equivalent final answer."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    async def __call__(self, rollout: Rollout, env_input: DapoMathSample) -> float:
        """Return 1 when Math-Verify equates the response and ground truth."""
        if not rollout.turns:
            return 0.0
        completion_message = rollout.turns[-1].completion_message
        response = (
            (completion_message.get("content") or "") if completion_message else ""
        )
        return score_math_response(response, env_input.ground_truth)
