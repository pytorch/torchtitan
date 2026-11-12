# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from renderers import Message

from torchtitan.rl.examples.geometry3k.data import Geometry3kSample, IMAGE_MARKER
from torchtitan.rl.rollout.message_env import (
    MessageEnv,
    MessageEnvInitOutput,
    MessageEnvStepOutput,
)

ANSWER_FORMAT = r"Think step by step, then put the final answer in \boxed{}."


class Geometry3kEnv(MessageEnv):
    """Asks one Geometry3K problem; the rollout ends after one answer.

    The user message is the problem with each ``<image>`` marker replaced by its image,
    followed by the answer format.

    Example:

        problem "<image>Find x."  ->  init: [image] "Find x.\\nThink step by step, ..."
        step:  end the rollout
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MessageEnv.Config):
        pass

    def __init__(self, config: Config, *, env_input: Geometry3kSample) -> None:
        del config
        self._env_input = env_input

    async def init(self) -> MessageEnvInitOutput:
        """Ask the problem, its images in place of their markers."""
        first, *rest = f"{self._env_input.problem}\n{ANSWER_FORMAT}".split(IMAGE_MARKER)
        content: list[dict] = (
            [{"type": "text", "text": first.strip()}] if first.strip() else []
        )
        for image_path, text in zip(self._env_input.image_paths, rest, strict=True):
            content.append({"type": "image", "path": image_path})
            if text.strip():
                content.append({"type": "text", "text": text.strip()})
        return MessageEnvInitOutput(
            init_prompt_messages=[{"role": "user", "content": content}]
        )

    async def step(self, completion_message: Message) -> MessageEnvStepOutput:
        """End the rollout: Geometry3K problems are single-turn."""
        del completion_message
        return MessageEnvStepOutput(done=True)
