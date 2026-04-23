# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
import re

from torchtitan.experiments.rl.types import Step


def correctness_reward(completion: str, target: int) -> float:
    """+1.0 if the response contains a ``[ANSWER] <target>`` tag, 0.0 otherwise.

    Uses the last ``[ANSWER]`` tag in the response (the model may
    self-correct). Trailing text after the tag is allowed.
    """
    matches = re.findall(r"\[ANSWER\]\s*(-?\d+)", completion)
    return 1.0 if matches and int(matches[-1]) == target else 0.0


def format_reward(completion: str) -> float:
    """+0.3 if the response contains any ``[ANSWER] <number>`` tag.

    The 0.3 value (vs. 1.0 for correctness) is chosen so the ratio is
    13/3 — this avoids small-integer solutions to ``r₁ + (b/a)·r₂ = n``
    at the default group size of 8, so no non-uniform group has a
    sample sitting exactly at the mean (which would give zero advantage).
    """
    return 0.3 if re.search(r"\[ANSWER\]\s*-?\d+", completion) else 0.0


_SYSTEM_PROMPT = """\
You are a helpful assistant. Solve the problem step by step.
When you have your final answer, state it as [ANSWER] <number>.

Example:
User: What is the total digit sum of [12, 345, 67]?
Assistant: Break each number into digits:
12 → 1, 2
345 → 3, 4, 5
67 → 6, 7
Sum all digits: 1 + 2 + 3 + 4 + 5 + 6 + 7 = 28
[ANSWER] 28"""


class SumDigitsEnv:
    """Single-turn, single-use env for one sum-of-digits problem.

    The constructor pulls the problem (2-4 integers in [10, 99]) from
    the shared ``rng``. Each construction advances the generator, so
    successive envs get fresh problems; reproducibility lives in the
    caller's RNG state, not in per-env seeds. ``step(action)`` returns
    +1.0 for a correct ``[ANSWER] <target>`` tag plus a +0.3 format
    bonus for any ``[ANSWER] <number>`` tag (applied independently).
    """

    def __init__(self, rng: random.Random):
        n = rng.randint(2, 4)
        numbers = [rng.randint(10, 99) for _ in range(n)]
        self._target = sum(int(d) for num in numbers for d in str(num))
        question = f"What is the total digit sum of {numbers}?"
        self.prompt = f"{_SYSTEM_PROMPT}\n\n{question}"

    def step(self, completion: str) -> Step:
        reward = correctness_reward(completion, self._target) + format_reward(
            completion
        )
        return Step(reward=reward, done=True)
