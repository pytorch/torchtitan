# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from renderers import Message

from torchtitan.experiments.rl.envs import MessageEnv, ResetOutput, StepOutput
from torchtitan.experiments.rl.tasks.sum_digits.data import SumDigitsInput


SYSTEM_PROMPT = """\
You are a helpful assistant. Solve the problem step by step.
When you have your final answer, state it as [ANSWER] <number>.

Example:
User: What is the total digit sum of [12, 345, 67]?
Assistant: Break each number into digits:
12 -> 1, 2
345 -> 3, 4, 5
67 -> 6, 7
Sum all digits: 1 + 2 + 3 + 4 + 5 + 6 + 7 = 28
[ANSWER] 28"""


class SumDigitsEnv(MessageEnv):
    def __init__(self, *, env_input: SumDigitsInput) -> None:
        self._numbers = env_input.numbers

    async def reset(self) -> ResetOutput:
        """Return the system prompt and one SumDigits user question."""
        question = f"What is the total digit sum of {self._numbers}?"
        return ResetOutput(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
        )

    async def step_message(self, msg: Message) -> StepOutput:
        # Terminate after the first assistant message, since it is a single-turn env.
        return StepOutput(done=True)
