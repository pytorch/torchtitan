# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
from dataclasses import dataclass, field

from renderers import Message, ToolSpec

from torchtitan.config import Configurable


@dataclass(kw_only=True, slots=True)
class MessageInitOutput:
    """Initial messages + tool specs from `MessageEnv.init`."""

    init_prompt_messages: list[Message]  # [M_prompt]
    """The opening messages (e.g. [system, user])."""

    tools: list[ToolSpec] = field(default_factory=list)  # [K_tools]
    """Tool schemas exposed to the assistant. Empty for tool-less envs."""


@dataclass(kw_only=True, slots=True)
class MessageStepOutput:
    """The env's reply to the assistant's turn."""

    env_messages: list[Message] = field(default_factory=list)  # [M_env]
    """The env's reply messages (tool / user). Empty when the rollout terminates
    with no follow-up."""

    done: bool = False
    """`True` ends the rollout."""

    env_rewards: dict[str, float] = field(default_factory=dict)
    """Optional reward signal the env provides for this step; the rubric decides
    whether and how to use it. Empty if the env scores nothing."""

    def __post_init__(self) -> None:
        # env replies are tool/user turns; the assistant turn comes from the generator
        if any(m.get("role") == "assistant" for m in self.env_messages):
            raise ValueError(
                "MessageStepOutput.env_messages may not contain assistant messages"
            )


class MessageEnv(Configurable, abc.ABC):
    """User-written env in message space. Implement `init` + `step`.

    Tip: `MessageEnv` works in messages and never sees token ids; You can have `TokenEnv`
    wrap it and use a `Renderer` to convert messages <-> token ids for the generator.
    Example:
        # a one-tool calculator env. It is multi-turn — the env answers the
        # assistant's tool call, then ends once the assistant replies without a tool.

        class CalculatorEnv(MessageEnv):
            @dataclass(kw_only=True, slots=True)
            class Config(MessageEnv.Config):
                pass

            def __init__(self, config: Config, *, env_input: CalculatorExample) -> None:
                self._expression = env_input.expression

            async def init(self) -> MessageInitOutput:
                return MessageInitOutput(
                    init_prompt_messages=[{"role": "user", "content": f"What is {self._expression}?"}],
                    tools=[CALCULATOR_TOOL],
                )

            async def step(self, parsed_completion_message: Message) -> MessageStepOutput:
                tool_calls = parsed_completion_message.get("tool_calls")
                if not tool_calls:
                    return MessageStepOutput(done=True)  # assistant gave its final answer
                result = run_calculator(tool_calls[0])
                return MessageStepOutput(
                    env_messages=[{"role": "tool", "content": result}]
                )
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Static env params; the per-rollout example is passed to `build(env_input=...)`."""

    @abc.abstractmethod
    async def init(self) -> MessageInitOutput:
        """Return the initial conversation + tools for prompt rendering."""

    @abc.abstractmethod
    async def step(self, parsed_completion_message: Message) -> MessageStepOutput:
        """Advance the env one turn given the completion message.

        `TokenEnv` parses the completion and handles
        finish_reason / length / parse / timeout failures before calling this,
        so the env only sees a well-formed completion message.

        Args:
            parsed_completion_message: the completion as a parsed message.

        Returns:
            `MessageStepOutput` with the env's reply messages.
        """

    async def close(self) -> None:
        """Release env-owned resources. Default no-op; idempotent."""
