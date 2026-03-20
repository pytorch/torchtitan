# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
import re

import torch


def extract_answer(text: str) -> int | None:
    """Extract a numeric answer from model output.

    Tries in order:
    1. [ANSWER] <number> tag
    2. "the answer is" / "answer:" patterns
    3. Last number in text
    """
    # 1. [ANSWER] tag (last match, model may self-correct)
    answer_match = re.findall(r"\[ANSWER\]\s*(-?\d+)", text)
    if answer_match:
        return int(answer_match[-1])

    # 2. Natural language patterns
    patterns = [
        r"(?:the answer is|answer is|answer:)\s*(-?\d+)",
        r"=\s*(-?\d+)\.?\s*(?:The answer|$)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return int(matches[-1])

    # 3. Last number in text
    numbers = re.findall(r"-?\d+", text)
    return int(numbers[-1]) if numbers else None


_SYSTEM_PROMPT = """\
You are a helpful assistant. Solve the problem step by step.
When you have your final answer, state it as [ANSWER] <number>.
Do not write anything after the answer.

Example:
User: What is the total digit sum of [12, 345, 67]?
Assistant: Break each number into digits:
12 → 1, 2
345 → 3, 4, 5
67 → 6, 7
Sum all digits: 1 + 2 + 3 + 4 + 5 + 6 + 7 = 28
[ANSWER] 28"""


class SumDigitsTask:
    """Generates sum-of-digits tasks with sequences of 2-4 integers (10-99)."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng = random.Random(seed)

    def create_question(self) -> tuple[str, str]:
        """Return a (question, answer) pair."""
        n = self._rng.randint(2, 4)
        numbers = [self._rng.randint(10, 99) for _ in range(n)]
        answer = sum(int(d) for num in numbers for d in str(num))
        question = f"What is the total digit sum of {numbers}?"
        return question, str(answer)

    def get_system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    def reward_function(
        self,
        completions: list[str],
        expected_answer: str = "",
    ) -> torch.Tensor:
        """Compute rewards for sum digits task.

        Args:
            completions: List of completion strings for one prompt (len=group_size)
            expected_answer: Expected answer string for this prompt

        Returns:
            Tensor of rewards (+1.0 correct, -1.0 wrong, +0.2 format bonus)
        """
        expected = int(expected_answer) if expected_answer else 0
        rewards = []
        for completion in completions:
            extracted = extract_answer(completion)
            is_correct = extracted == expected
            reward = 1.0 if is_correct else -1.0
            # Format bonus: only if correct, exactly one [ANSWER] tag, and generation stops after it
            answer_tags = re.findall(r"\[ANSWER\]", completion)
            if (
                is_correct
                and len(answer_tags) == 1
                and re.search(r"\[ANSWER\]\s*-?\d+\s*$", completion)
            ):
                reward += 0.2
            rewards.append(reward)

        return torch.tensor(rewards, dtype=torch.float32)
