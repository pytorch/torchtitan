# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from dataclasses import dataclass

from math_verify import parse, verify

from torchtitan.experiments.rl.examples.dapo_math.data import DapoMathSample
from torchtitan.experiments.rl.examples.dapo_math.thread_timeout import (
    ThreadTimeout,
    ThreadTimeoutError,
)
from torchtitan.experiments.rl.rollout import Rollout
from torchtitan.experiments.rl.rubrics import RewardFn

logger = logging.getLogger(__name__)

_BOXED_START = r"\boxed{"
# Match the default timeout used by Math-Verify 0.9.0.
_MATH_VERIFY_TIMEOUT_SECONDS = 5


def _last_boxed_expression(text: str) -> str | None:
    """Return the last complete `\\boxed{...}` expression."""
    start = text.rfind(_BOXED_START)
    if start == -1:
        return None

    answer_start = start + len(_BOXED_START)
    depth = 1
    for index, char in enumerate(text[answer_start:], start=answer_start):
        depth += (char == "{") - (char == "}")
        if depth == 0:
            return text[start : index + 1]
    return None


def score_math_response(response: str, ground_truth: str) -> float:
    """Score the final `\\boxed{}` expression with Math-Verify.

    Args:
        response: Model response containing a boxed final answer.
        ground_truth: Expected answer from the dataset.

    Example:
        score_math_response(r"work\nAnswer: \boxed{34}", "34")  # 1.0
    """
    prediction = _last_boxed_expression(response)
    if prediction is None:
        return 0.0

    try:
        # Math-Verify uses SIGALRM for timeouts, which does not work in monarch
        # worker threads. Apply the same deadline with a thread-targeted timeout.
        with ThreadTimeout(_MATH_VERIFY_TIMEOUT_SECONDS):
            gold = parse(ground_truth, parsing_timeout=None)
            prediction = parse(prediction, parsing_timeout=None)
            return float(bool(gold) and verify(gold, prediction, timeout_seconds=None))
    except ThreadTimeoutError:
        logger.warning(
            "Math-Verify timed out after %s seconds; assigning zero reward",
            _MATH_VERIFY_TIMEOUT_SECONDS,
        )
        return 0.0
    except Exception:
        # Model output is untrusted; malformed LaTeX produces a zero reward
        # rather than failing the training loop.
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
