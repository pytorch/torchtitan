# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
from dataclasses import dataclass, field

from renderers import Message, ToolSpec

from torchtitan.experiments.rl.rollouts.types import RolloutStatus


@dataclass(kw_only=True, slots=True)
class MsgResponseReset:
    """Initial messages + tool specs from ``MessageEnv.reset``."""

    messages: list[Message]
    """Initial conversation (system + user + few-shot, typically)."""

    tools: list[ToolSpec] = field(default_factory=list)
    """Tool schemas exposed to the model. Empty for tool-less envs."""


@dataclass(kw_only=True, slots=True)
class MsgResponseStep:
    """Env response to one parsed assistant message."""

    messages: list[Message] = field(default_factory=list)
    """Env-appended messages (tool / user replies). Empty when the
    rollout terminates with no follow-up."""

    done: bool = False
    """``True`` ends the rollout."""

    status: RolloutStatus | None = None
    """Terminal status; ``None`` on non-terminal steps."""

    def __post_init__(self) -> None:
        for msg in self.messages:
            if msg.get("role") == "assistant":
                raise ValueError(
                    "MsgResponseStep.messages must not contain assistant-role "
                    "messages; the env cannot forge assistant turns."
                )


class MessageEnv(abc.ABC):
    """User's message-level env. Subclass + implement 3 methods.

    User envs are renderer-free: ``reset`` / ``step_message`` deal only
    in messages. The framework's ``RendererEnv`` composes one of these
    with a ``Renderer`` to expose a token-level interface to the driver.
    """

    @abc.abstractmethod
    async def reset(self) -> MsgResponseReset:
        """Return the initial conversation + tool specs for this rollout."""

    @abc.abstractmethod
    async def step_message(self, msg: Message) -> MsgResponseStep:
        """Apply one parsed assistant message; return env-side response.

        Subclasses MUST NOT inspect ``finish_reason`` / token counts /
        parse failures; ``RendererEnv`` handles those.

        Args:
            msg: Parsed assistant message from ``Renderer.parse_response``.
                Fields include ``content``, optional ``reasoning_content``,
                and optional ``tool_calls``.

        Returns:
            ``MsgResponseStep`` carrying env-appended messages, terminal flag,
            and terminal status. Reward assignment lives entirely on the
            rubric (doc 37 Option B).
        """

    async def close(self) -> None:
        """Release env-owned resources. Default no-op; idempotent."""
