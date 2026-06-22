# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Placeholder MessageEnv for the Claude Code coding-agent example.

Unlike a native env-driven rollout (where the framework alternates env.step and
generator calls), the coding-agent harness runs the *whole* agent loop inside the
sandbox: Claude Code decides tool calls and edits files itself, talking to the
on-box Anthropic adapter for each model turn. The framework's env <-> generator
loop is therefore bypassed -- ``SWER2ERollouter`` overrides ``run_group_rollouts``
to drive Claude Code directly.

This class exists only to satisfy ``Rollouter.Config.message_env`` (a required
field) and to carry the per-rollout task spec; its ``init``/``step`` are never
called in the Claude Code flow.
"""

from __future__ import annotations

from dataclasses import dataclass

from renderers import Message

from torchtitan.experiments.rl.environment import (
    MessageEnv,
    MessageEnvInitOutput,
    MessageEnvStepOutput,
)
from torchtitan.experiments.rl.examples.swe_r2e.data import SWER2ESample


class SWER2EEnv(MessageEnv):
    """Carries one R2E task spec. The agent loop runs in-sandbox (see module doc)."""

    @dataclass(kw_only=True, slots=True)
    class Config(MessageEnv.Config):
        pass

    def __init__(self, config: Config, *, env_input: SWER2ESample) -> None:
        self.sample = env_input

    async def init(self) -> MessageEnvInitOutput:
        raise NotImplementedError(
            "SWER2EEnv is not driven via the env loop; SWER2ERollouter runs Claude "
            "Code in the sandbox. See examples/swe_r2e/rollouter.py."
        )

    async def step(self, completion_message: Message) -> MessageEnvStepOutput:
        raise NotImplementedError(
            "SWER2EEnv is not driven via the env loop; SWER2ERollouter runs Claude "
            "Code in the sandbox. See examples/swe_r2e/rollouter.py."
        )
