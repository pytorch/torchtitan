# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
from dataclasses import dataclass, field

from renderers import Message, ToolSpec


@dataclass(kw_only=True, slots=True)
class ResetOutput:
    """Initial messages + tool specs from `MessageEnv.reset`."""

    messages: list[Message]  # [M_initial]
    """The set of messages that form an initial prompt for the model."""

    tools: list[ToolSpec] = field(default_factory=list)  # [K_tools]
    """Tool schemas exposed to the model. Empty for tool-less envs."""


@dataclass(kw_only=True, slots=True)
class StepOutput:
    """Env response to one parsed assistant message.

    `done` ends the conversation; `RendererEnv` maps it to a `RolloutStatus`
    and detects truncation/errors the env never sees. The env may attach
    `reward_components` (e.g. tool-call success); the rubric decides how to use them.
    """

    messages: list[Message] = field(default_factory=list)  # [M_env]
    """Env-appended messages (tool / user replies). Empty when the rollout
    terminates with no follow-up."""

    done: bool = False
    """`True` ends the rollout."""

    reward_components: dict[str, float] = field(default_factory=dict)
    """Optional reward signal the env provides for this step; the rubric decides
    whether and how to use it. Empty if the env scores nothing."""

    def __post_init__(self) -> None:
        # env replies are tool/user turns; the assistant turn comes from the model
        if any(m.get("role") == "assistant" for m in self.messages):
            raise ValueError("StepOutput.messages may not contain assistant messages")


class MessageEnv(abc.ABC):
    """User-written env in message space. Subclass and implement `reset` +
    `step_message`; `RendererEnv` wraps it with a `Renderer` for tokens.

    Example:

        class SumDigitsEnv(MessageEnv):
            async def reset(self) -> ResetOutput:
                return ResetOutput(messages=[{"role": "user", "content": "sum [1, 2]"}])

            async def step_message(self, msg: Message) -> StepOutput:
                return StepOutput(done=True)        # single-turn; rubric scores later
    """

    @abc.abstractmethod
    async def reset(self) -> ResetOutput:
        """Return the initial conversation + tools. The wrapper takes ownership
        of the returned message list (it may mutate it across turns)."""

    @abc.abstractmethod
    async def step_message(self, msg: Message) -> StepOutput:
        """Apply one parsed assistant message; return the env's reply.

        Subclasses MUST NOT inspect `finish_reason` / token counts / parse
        failures; `RendererEnv` handles those before calling this.

        Args:
            msg: Parsed assistant message (`content`, optional
                `reasoning_content`, optional `tool_calls`).

        Returns:
            `StepOutput` with env reply messages, `done`, and optional
            `reward_components`. Final rewards are computed by the rubric.
        """

    async def close(self) -> None:
        """Release env-owned resources. Default no-op; idempotent."""
