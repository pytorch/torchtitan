# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from renderers import Message, Renderer, ToolSpec

from torchtitan.experiments.rl.env_types.message_env import (
    MessageEnv,
    MessageStepOutput,
)
from torchtitan.experiments.rl.rollouts.types import RolloutStatus

if TYPE_CHECKING:
    from torchtitan.experiments.rl.actors.generator import Completion

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class TurnMessages:
    """Container for all message data collected during one rollut turn."""

    prompt_messages: list[Message]  # [M_prompt]
    """Full conversation rendered into the prompt (the assistant's input this turn)."""

    assistant_message: Message | None = None
    """The received assistant's message (generator output, parsed), if any. `None` on reset and on
    parse/overflow failures."""

    env_step: MessageStepOutput | None = None
    """The env's reply this turn. `None` on reset, truncation, abort, and timeout (the env was not stepped)."""


@dataclass(kw_only=True, slots=True)
class TokenizedStepOutput:
    """Output of `RendererWrapperEnv.reset` and `.step`. Holds the next
    prompt token ids, if turn is not terminal, and all messages collected so far."""

    next_prompt_token_ids: list[int] | None  # [L_prompt] or None
    """Input tokens for the NEXT generator cal; `None` on a terminal turn."""

    status: RolloutStatus
    """`ONGOING` while the rollout runs; a terminal `RolloutStatus` otherwise."""

    turn: TurnMessages
    """Message-space view of this turn (full conversation + assistant message + env reply)."""


class RendererWrapperEnv:
    """Token-space wrapper around a `MessageEnv`.

    In a rollout, the input to a generator is a tokenized prompt, but the input to MessageEnv.step an
    its output is a message. A step that translates message <-> token is necessary.
    This wrapper fills this role, using a renderer to convert between the two and facilitating
    the communication between the generator and the MessageEnv.

    In this process, several checks are necessary, e.g. prompt is too long,
    total number of tokens is too long, number of turns exceeded the limit, etc. This wrapper
    also takes care of that, so we can keep the rollout loop clean and simple.

    If users have extra or different logic, they can wrap their MessageEnv with another class instead.

    Args:
        message_env: the user's `MessageEnv` subclass instance.
        renderer: a `renderers.Renderer` that converts messages <-> token ids.
        config: `RendererWrapperEnv.Config`.

    Example:

        env = RendererWrapperEnv(message_env=SumDigitsEnv(...), renderer=renderer)
        step = await env.reset()
        while not step.status.is_terminal():
            completion = await generator.generate([step.next_prompt_token_ids])
            step = await env.step(completion)
    """

    @dataclass(kw_only=True, slots=True)
    class Config:
        """Limits enforced by the wrapper"""

        max_rollout_tokens: int | None = None
        """Hard cap on prompt length for the next turn. If the number of tokens meets/exceeds
        it, the turn is terminal; `None` disables the check."""

        # TODO: its unclear if timeout should be on this layer or handled by the messageEnv
        step_timeout_s: float | None = 1800.0
        """Wall-clock timeout for one `MessageEnv.step` call."""

        # TODO: add max_num_turns

    def __init__(
        self,
        *,
        message_env: MessageEnv,
        renderer: Renderer,
        config: "RendererWrapperEnv.Config | None" = None,
    ) -> None:
        self._message_env = message_env
        self._renderer = renderer
        self._config = config or RendererWrapperEnv.Config()
        self._tools: list[ToolSpec] | None = None
        self._messages: list[Message] = []
        self._last_prompt_ids: list[int] = []

    async def reset(self) -> TokenizedStepOutput:
        """Render the initial conversation into the first generator prompt."""
        env_reset = await self._message_env.reset()
        self._messages = list(env_reset.prompt_messages)

        # Render messages into tokens
        self._tools = list(env_reset.tools) if env_reset.tools else None
        token_ids = await asyncio.to_thread(
            self._renderer.render_ids,
            messages=self._messages,
            tools=self._tools,
            add_generation_prompt=True,
        )

        # Terminal if the prompt is already over budget
        if self._is_prompt_overflow(prompt_len=len(token_ids)):
            return _terminal(
                prompt_messages=self._messages,
                status=RolloutStatus.TRUNCATED_PROMPT_TOO_LONG,
            )

        self._last_prompt_ids = list(token_ids)
        return TokenizedStepOutput(
            next_prompt_token_ids=list(token_ids),
            status=RolloutStatus.ONGOING,
            turn=TurnMessages(prompt_messages=list(self._messages)),
        )

    async def step(self, completion: "Completion") -> TokenizedStepOutput:
        """Advance the env by one sampled completion from the generator.

        Args:
            completion: Generator output for the current prompt.

        Returns:
            `TokenizedStepOutput` for the next generator call, or a terminal turn
            when the rollout completes, truncates, or errors.
        """
        # Parse first, so a truncated / aborted response still carries its message
        try:
            parsed = await asyncio.to_thread(
                self._renderer.parse_response,
                token_ids=list(completion.token_ids),
            )
        except Exception:
            logger.exception(
                "parse_response failed (finish_reason=%s, %d tokens); -> ERROR_PARSE",
                completion.finish_reason,
                len(completion.token_ids),
            )
            return _terminal(
                prompt_messages=self._messages,
                status=RolloutStatus.ERROR_PARSE,
            )

        assistant: Message = {"role": "assistant", "content": parsed.content}
        if parsed.reasoning_content:
            assistant["reasoning_content"] = parsed.reasoning_content
        if parsed.tool_calls:
            assistant["tool_calls"] = parsed.tool_calls

        # Truncated / aborted: the response is final and partial. Keep it for
        # partial-reward grading and debugging; don't step the env on it.
        # TODO: check if we should step the env on an incomplete message
        if completion.finish_reason == "length":
            return _terminal(
                prompt_messages=self._messages,
                status=RolloutStatus.TRUNCATED_LENGTH,
                assistant_message=assistant,
            )
        if completion.finish_reason == "abort":
            return _terminal(
                prompt_messages=self._messages,
                status=RolloutStatus.ERROR_ABORT,
                assistant_message=assistant,
            )

        # Apply the user's env step under a timeout
        timeout = self._config.step_timeout_s
        try:
            if timeout is None:
                env_step = await self._message_env.step(assistant)
            else:
                env_step = await asyncio.wait_for(
                    self._message_env.step(assistant), timeout=timeout
                )
        except TimeoutError:
            logger.warning("step timed out after %ss; -> ERROR_TIMEOUT", timeout)
            return _terminal(
                prompt_messages=self._messages,
                status=RolloutStatus.ERROR_TIMEOUT,
                assistant_message=assistant,
            )

        self._messages.append(assistant)
        self._messages.extend(env_step.env_messages)

        if env_step.done:
            return _terminal(
                prompt_messages=self._messages,
                status=RolloutStatus.COMPLETED,
                assistant_message=assistant,
                env_step=env_step,
            )

        # Prepare the next prompt; full re-render if the renderer can't bridge.
        # `tools` is passed because tool schemas are part of the chat template, so
        # the bridged tokens must match what a full re-render (also tools-aware) produces.
        bridged = await asyncio.to_thread(
            self._renderer.bridge_to_next_turn,
            previous_prompt_ids=self._last_prompt_ids,
            previous_completion_ids=list(completion.token_ids),
            new_messages=env_step.env_messages,
            tools=self._tools,
        )
        if bridged is None:
            next_prompt_token_ids = await asyncio.to_thread(
                self._renderer.render_ids,
                messages=self._messages,
                tools=self._tools,
                add_generation_prompt=True,
            )
        else:
            next_prompt_token_ids = bridged.token_ids

        # Terminal if the next prompt is over budget
        if self._is_prompt_overflow(prompt_len=len(next_prompt_token_ids)):
            return _terminal(
                prompt_messages=self._messages,
                status=RolloutStatus.TRUNCATED_PROMPT_TOO_LONG,
                assistant_message=assistant,
                env_step=env_step,
            )

        self._last_prompt_ids = list(next_prompt_token_ids)
        return TokenizedStepOutput(
            next_prompt_token_ids=list(next_prompt_token_ids),
            status=RolloutStatus.ONGOING,
            turn=TurnMessages(
                prompt_messages=list(self._messages),
                assistant_message=assistant,
                env_step=env_step,
            ),
        )

    async def close(self) -> None:
        await self._message_env.close()

    def _is_prompt_overflow(self, *, prompt_len: int) -> bool:
        cap = self._config.max_rollout_tokens
        return cap is not None and prompt_len >= cap


def _terminal(
    *,
    prompt_messages: list[Message],
    status: RolloutStatus,
    assistant_message: Message | None = None,
    env_step: MessageStepOutput | None = None,
) -> TokenizedStepOutput:
    """Build a terminal `TokenizedStepOutput` with the given status."""
    return TokenizedStepOutput(
        next_prompt_token_ids=None,
        status=status,
        turn=TurnMessages(
            prompt_messages=list(prompt_messages),
            assistant_message=assistant_message,
            env_step=env_step,
        ),
    )
