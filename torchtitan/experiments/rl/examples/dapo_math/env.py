# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from renderers import Message

from torchtitan.experiments.rl.environment import (
    MessageEnv,
    MessageEnvInitOutput,
    MessageEnvStepOutput,
)
from torchtitan.experiments.rl.examples.dapo_math.data import DapoMathSample


class DapoMathEnv(MessageEnv):
    """Single-turn message environment for a verifiable math problem."""

    @dataclass(kw_only=True, slots=True)
    class Config(MessageEnv.Config):
        pass

    def __init__(self, config: Config, *, env_input: DapoMathSample) -> None:
        del config
        self._prompt = env_input.prompt

    async def init(self) -> MessageEnvInitOutput:
        """Return the problem as one user message."""
        return MessageEnvInitOutput(
            init_prompt_messages=[{"role": "user", "content": self._prompt}]
        )

    async def step(self, completion_message: Message) -> MessageEnvStepOutput:
        """End after the model's first response; the rubric scores its final answer."""
        del completion_message
        return MessageEnvStepOutput(done=True)
