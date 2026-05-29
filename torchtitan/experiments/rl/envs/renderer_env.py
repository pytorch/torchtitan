# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from renderers import Message, Renderer, ToolSpec

from torchtitan.experiments.rl.envs.message_env import MessageEnv
from torchtitan.experiments.rl.rollouts.types import RolloutStatus

if TYPE_CHECKING:
    from torchtitan.experiments.rl.actors.generator import Completion


# TODO: revisit the `EnvLimits` name once tool/browser envs share this policy.
@dataclass(kw_only=True, slots=True)
class EnvLimits:
    """Boundaries to be enforced by `RendererEnv`"""

    max_rollout_tokens: int | None = None
    """Hard cap on number of tokens a Rollout can hold.
    If the next prompt exceeds it, the rollout turn becomes terminal;
    `None` disables the check."""

    step_timeout_s: float | None = 1800.0
    """Wall-clock timeout for one `MessageEnv.step_message` call."""


@dataclass(kw_only=True, slots=True)
class TokenizedResponseStep:
    """One turn boundary in the rollout.

    Returned by both `RendererEnv.initial_prompt` and `RendererEnv.step_completion`.
    """

    next_token_ids: list[int] | None  # [L_prompt] or None
    """Prompt for the NEXT generator call. `None` on a terminal step. On
    initial: the prompt the model sees on turn 0. After a non-terminal
    `step_completion`: prompt for turn N+1."""

    next_messages: list[Message]  # [M_next]
    """Full conversation, from all turns, at the next-prompt point."""

    last_response_messages: list[Message] = field(default_factory=list)  # [M_response]
    """Messages appended this turn (parsed assistant message + env's tool /
    user replies). Empty on the initial prompt."""

    status: RolloutStatus
    """`ONGOING` while the rollout is still running (initial and non-terminal
    turns); a terminal `RolloutStatus` when this turn ends the rollout."""


class RendererEnv:
    """Framework wrapper around one `MessageEnv` + `Renderer`.

    From the rollout driver's view this IS the env; it does NOT inherit
    `MessageEnv`. Owns the renderer plumbing, length-stop / parse-error /
    timeout classification, and context-overflow checks. Rewards are
    assigned by the task rubric after the rollout is built.

    Args:
        message_env: The user's `MessageEnv` subclass instance.
        renderer: Renderer from `RendererConfig.build`.
        limits: Framework-level wrapper policy (timeouts, overflow caps).

    Example:

        env = RendererEnv(
            message_env=SumDigitsEnv(env_input=example.env_input),
            renderer=renderer,
        )
        turn = await env.initial_prompt()
        while turn.next_token_ids is not None:
            completion = await generator.generate([turn.next_token_ids])
            turn = await env.step_completion(completion[0])
    """

    def __init__(
        self,
        *,
        message_env: MessageEnv,
        renderer: Renderer,
        limits: EnvLimits | None = None,
    ) -> None:
        self._message_env = message_env
        self._renderer = renderer
        self._limits = limits or EnvLimits()
        self._tools: list[ToolSpec] | None = None
        self._messages: list[Message] = []
        self._last_prompt_ids: list[int] = []

    async def initial_prompt(self) -> TokenizedResponseStep:
        """Render the initial conversation into the first generator prompt.

        Returns:
            `TokenizedResponseStep` with `next_token_ids` set for runnable prompts,
            or a terminal `TRUNCATED_PROMPT_OVERFLOW` turn when the initial prompt
            exceeds `EnvLimits.max_rollout_tokens`.
        """
        init = await self._message_env.reset()
        self._messages = list(init.messages)
        # Renderer chat templates often emit a different tools-section
        # depending on None vs empty list; store None when no tools.
        self._tools = list(init.tools) if init.tools else None
        token_ids = await asyncio.to_thread(
            self._renderer.render_ids,
            messages=self._messages,
            tools=self._tools,
            add_generation_prompt=True,
        )
        if self._is_prompt_overflow(prompt_len=len(token_ids)):
            return _terminal(
                next_messages=list(self._messages),
                status=RolloutStatus.TRUNCATED_PROMPT_OVERFLOW,
            )
        self._last_prompt_ids = list(token_ids)
        return TokenizedResponseStep(
            next_token_ids=list(token_ids),
            next_messages=list(self._messages),
            last_response_messages=[],
            status=RolloutStatus.ONGOING,
        )

    async def step_completion(self, completion: Completion) -> TokenizedResponseStep:
        """Advance the env by one sampled completion.

        Args:
            completion: Generator output for the current prompt.

        Returns:
            `TokenizedResponseStep` for the next generator call, or a terminal turn
            when the rollout completes, truncates, or errors.
        """
        # Parse first, so a truncated / aborted response still carries its message.
        try:
            parsed = await asyncio.to_thread(
                self._renderer.parse_response,
                token_ids=list(completion.token_ids),
            )
        except Exception:
            return _terminal(
                next_messages=list(self._messages),
                status=RolloutStatus.ERROR_PARSE,
            )

        assistant: Message = {
            "role": "assistant",
            "content": parsed.content,
        }
        if parsed.reasoning_content:
            assistant["reasoning_content"] = parsed.reasoning_content
        if parsed.tool_calls:
            assistant["tool_calls"] = parsed.tool_calls

        # Truncated / aborted: the response is final and partial. Keep it (for
        # partial-reward grading and debugging); don't step the env on an
        # incomplete message.
        if completion.finish_reason == "length":
            return _terminal(
                next_messages=list(self._messages),
                status=RolloutStatus.TRUNCATED_LENGTH,
                last_response_messages=[assistant],
            )
        if completion.finish_reason == "abort":
            return _terminal(
                next_messages=list(self._messages),
                status=RolloutStatus.ERROR_ABORT,
                last_response_messages=[assistant],
            )

        # Apply the user's env step under a timeout
        timeout = self._limits.step_timeout_s
        try:
            if timeout is None:
                env_step = await self._message_env.step_message(assistant)
            else:
                env_step = await asyncio.wait_for(
                    self._message_env.step_message(assistant), timeout=timeout
                )
        except TimeoutError:
            return _terminal(
                next_messages=list(self._messages),
                status=RolloutStatus.ERROR_TIMEOUT,
                last_response_messages=[assistant],
            )

        new_messages: list[Message] = [assistant, *env_step.messages]
        self._messages.extend(new_messages)

        if env_step.done:
            return TokenizedResponseStep(
                next_token_ids=None,
                next_messages=list(self._messages),
                status=env_step.status or RolloutStatus.COMPLETED,
                last_response_messages=new_messages,
            )

        # Prepare the next prompt; full re-render if the renderer can't bridge
        bridged = await asyncio.to_thread(
            self._renderer.bridge_to_next_turn,
            previous_prompt_ids=self._last_prompt_ids,
            previous_completion_ids=list(completion.token_ids),
            new_messages=env_step.messages,
            tools=self._tools,
        )
        if bridged is None:
            next_token_ids = await asyncio.to_thread(
                self._renderer.render_ids,
                messages=self._messages,
                tools=self._tools,
                add_generation_prompt=True,
            )
        else:
            next_token_ids = bridged.token_ids

        # Context-overflow check before returning
        if self._is_prompt_overflow(prompt_len=len(next_token_ids)):
            return _terminal(
                next_messages=list(self._messages),
                status=RolloutStatus.TRUNCATED_PROMPT_OVERFLOW,
                last_response_messages=new_messages,
            )

        self._last_prompt_ids = list(next_token_ids)
        return TokenizedResponseStep(
            next_token_ids=list(next_token_ids),
            next_messages=list(self._messages),
            last_response_messages=new_messages,
            status=RolloutStatus.ONGOING,
        )

    async def close(self) -> None:
        await self._message_env.close()

    def _is_prompt_overflow(self, *, prompt_len: int) -> bool:
        cap = self._limits.max_rollout_tokens
        return cap is not None and prompt_len >= cap


def _terminal(
    *,
    next_messages: list[Message],
    status: RolloutStatus,
    last_response_messages: list[Message] | None = None,
) -> TokenizedResponseStep:
    """Build a terminal `TokenizedResponseStep` with the given terminal status."""
    return TokenizedResponseStep(
        next_token_ids=None,
        next_messages=next_messages,
        status=status,
        last_response_messages=last_response_messages or [],
    )
