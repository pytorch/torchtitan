# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from renderers import Message

from torchtitan.rl.examples.slake.data import SlakeSample
from torchtitan.rl.rollout.message_env import (
    MessageEnv,
    MessageEnvInitOutput,
    MessageEnvStepOutput,
)

# The instruction of the Muse Spark SLAKE LoRA SFT recipe (LLaVA 1.5's short-answer prompt):
# SLAKE labels are single words or short phrases, and without it the model answers in
# sentences.
ANSWER_INSTRUCTION = "Answer the question using a single word or phrase."


class SlakeEnv(MessageEnv):
    """Asks one SLAKE question about its image; the rollout ends after one answer.

    The user message holds the image followed by the question.

    Example:

        init:  [image] "Does the picture contain liver?\\nAnswer the question ..."
        step:  end the rollout
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MessageEnv.Config):
        pass

    def __init__(self, config: Config, *, env_input: SlakeSample) -> None:
        del config
        self._env_input = env_input

    async def init(self) -> MessageEnvInitOutput:
        """Ask the question, with the image first."""
        content = [
            {"type": "image", "path": self._env_input.image_path},
            {
                "type": "text",
                "text": f"{self._env_input.question}\n{ANSWER_INSTRUCTION}",
            },
        ]
        return MessageEnvInitOutput(
            init_prompt_messages=[{"role": "user", "content": content}]
        )

    async def step(self, completion_message: Message) -> MessageEnvStepOutput:
        """End the rollout: SLAKE questions are single-turn."""
        del completion_message
        return MessageEnvStepOutput(done=True)
