# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Reverse Digits - A synthetic benchmark for training LLMs on digit reversal.

Given an integer, the model must reverse its digits.
"""

import random

from task import Task


_SYSTEM_PROMPT = """You are a helpful assistant. Solve the problem step by step. When you have your final answer, state it as [ANSWER] <number>.

Example:
User: What is 1234 reversed?
Assistant: Reverse the digits of 1234:
1234 → 4321
[ANSWER] 4321"""


class ReverseDigitsSpec:
    """Reverse digits task: reverse the digits of an integer.

    Task: "What is 1234 reversed?"
    Answer: 4321
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng = random.Random(seed)

    def generate_task(self) -> Task:
        number = self._rng.randint(100000, 9999999)
        answer = int(str(number)[::-1])
        question = f"What is {number} reversed?"
        return Task(
            question=question,
            correct_answer=answer,
            metadata={"number": number},
        )

    def get_system_prompt(self) -> str:
        return _SYSTEM_PROMPT

