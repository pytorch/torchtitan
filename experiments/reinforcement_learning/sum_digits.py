# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Sum digits - A synthetic benchmark for training LLMs on digit-level arithmetic.

Given a sequence of integers, the model must compute the total sum of the digits.
"""

import random

from task import Task


_SYSTEM_PROMPT = """You are a helpful assistant. Solve the problem step by step. When you have your final answer, state it as [ANSWER] <number>.

Example:
User: What is the total digit sum of [12, 345, 67]?
Assistant: Break each number into digits:
12 → 1, 2
345 → 3, 4, 5
67 → 6, 7
Sum all digits: 1 + 2 + 3 + 4 + 5 + 6 + 7 = 28
[ANSWER] 28"""


class SumDigitsSpec:
    """Sum of digits task: sum all digits across a sequence of integers.

    Task: "What is the total digit sum of [123, 45, 67]?"
    Answer: 1+2+3 + 4+5 + 6+7 = 28
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng = random.Random(seed)

    def generate_task(self) -> Task:
        n = self._rng.randint(2, 4)
        numbers = [self._rng.randint(10, 999) for _ in range(n)]
        answer = sum(int(d) for num in numbers for d in str(num))
        question = f"What is the total digit sum of {numbers}?"
        return Task(
            question=question,
            correct_answer=answer,
            metadata={"numbers": numbers},
        )

    def get_system_prompt(self) -> str:
        return _SYSTEM_PROMPT

