# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
import re

from torchtitan.experiments.rl.types import Step


class SumDigitsEnv:
    """Single-turn, single-use env for one sum-of-digits problem.

    The constructor pulls the problem (2-4 integers in [10, 99]) from
    the shared ``rng``. Each construction advances the generator, so
    successive envs get fresh problems; reproducibility lives in the
    caller's RNG state, not in per-env seeds.

    ``step(completion)``
    returns +1.0 for a correct ``[ANSWER] <target>`` tag plus a +0.3
    format bonus for any ``[ANSWER] <number>`` tag (applied independently).
    """

    SYSTEM_PROMPT = """\
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

    CORRECT_REWARD = 1.0
    FORMAT_REWARD = 0.3

    def __init__(self, rng: random.Random):
        n = rng.randint(2, 4)
        numbers = [rng.randint(10, 99) for _ in range(n)]
        self._target = sum(int(d) for num in numbers for d in str(num))
        question = f"What is the total digit sum of {numbers}?"
        self.prompt = f"{self.SYSTEM_PROMPT}\n\n{question}"

    def step(self, completion: str) -> Step:
        return Step(
            rewards={
                "correctness": self._correctness_reward(completion),
                "format": self._format_reward(completion),
            },
            done=True,
        )

    def _correctness_reward(self, completion: str) -> float:
        """``CORRECT_REWARD`` if the last ``[ANSWER] <n>`` tag matches the target."""
        matches = re.findall(r"\[ANSWER\]\s*(-?\d+)", completion)
        correct = bool(matches) and int(matches[-1]) == self._target
        return self.CORRECT_REWARD if correct else 0.0

    def _format_reward(self, completion: str) -> float:
        """``FORMAT_REWARD`` if the response contains any ``[ANSWER] <n>`` tag."""
        if re.search(r"\[ANSWER\]\s*-?\d+", completion):
            return self.FORMAT_REWARD
        return 0.0
