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

from torchtitan.experiments.rl.envs.message_env import MessageEnv, MsgResponseStep
from torchtitan.experiments.rl.rollouts.types import RolloutStatus

if TYPE_CHECKING:
    from torchtitan.experiments.rl.actors.generator import Completion


# TODO: the name "EnvLimits" is not settled — fields are framework-level
# wrapper policy (timeouts, context overflow). Reconsider once we have
# more envs (tool, browser) and a clearer sense of what knobs cluster.
@dataclass(kw_only=True, slots=True)
class EnvLimits:
    """Operational policy applied at the ``RendererEnv`` boundary."""

    max_trajectory_tokens: int | None = None
    """Hard cap on ``prompt_tokens + generation_tokens`` per turn; ``None`` disables."""

    max_generation_tokens: int | None = None
    """Reserved tokens for the upcoming generation when checking overflow."""

    step_timeout_s: float | None = 1800.0
    """Wall-clock timeout for one ``MessageEnv.step_message`` call."""


@dataclass(kw_only=True, slots=True)
class TokenizedTurn:
    """One turn boundary in the rollout.

    Returned by both ``RendererEnv.initial_prompt()`` and
    ``RendererEnv.step_completion(completion)``.
    """

    next_token_ids: list[int] | None  # [L_prompt] or None
    """Prompt for the NEXT generator call. ``None`` on terminal step.
    On initial: the initial prompt the model sees on turn 0.
    After non-terminal step_completion: prompt for turn N+1."""

    next_messages: list[Message]
    """Full conversation snapshot at the next-prompt point."""

    last_env_step: MsgResponseStep | None = None
    """The env's response to the previous turn. ``None`` on initial."""

    last_response_messages: list[Message] = field(default_factory=list)
    """Messages appended this turn (parsed assistant message + env's
    tool / user replies). Empty on initial."""


class RendererEnv:
    """Framework wrapper around one ``MessageEnv`` + ``Renderer``.

    IS an env from the driver's view but does NOT inherit ``MessageEnv``.
    Owns the renderer plumbing, length-stop / parse-error / timeout
    classification, and context-overflow checks. Rewards live on the
    rubric; ``MsgResponseStep.reward`` is always ``None`` here.

    Args:
        message_env: The user's ``MessageEnv`` subclass instance.
        renderer: Renderer from ``RendererConfig.build``.
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

    async def initial_prompt(self) -> TokenizedTurn:
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
        if self._is_overflow(prompt_len=len(token_ids)):
            return _terminal(
                next_messages=list(self._messages),
                status=RolloutStatus.TRUNCATED_OVERFLOW,
            )
        self._last_prompt_ids = list(token_ids)
        return TokenizedTurn(
            next_token_ids=list(token_ids),
            next_messages=list(self._messages),
            last_env_step=None,
            last_response_messages=[],
        )

    async def step_completion(self, completion: Completion) -> TokenizedTurn:
        # Reject terminal-state completions
        if completion.finish_reason == "length":
            return _terminal(
                next_messages=list(self._messages),
                status=RolloutStatus.TRUNCATED_LENGTH,
            )
        if completion.finish_reason == "abort":
            return _terminal(
                next_messages=list(self._messages),
                status=RolloutStatus.ERROR_ABORT,
            )

        # Parse response tokens into an assistant message
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
            )

        new_messages: list[Message] = [assistant, *env_step.messages]
        self._messages.extend(new_messages)

        if env_step.done:
            if env_step.status is None:
                env_step.status = RolloutStatus.COMPLETED
            return TokenizedTurn(
                next_token_ids=None,
                next_messages=list(self._messages),
                last_env_step=env_step,
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
        if self._is_overflow(prompt_len=len(next_token_ids)):
            terminal = _terminal(
                next_messages=list(self._messages),
                status=RolloutStatus.TRUNCATED_OVERFLOW,
            )
            terminal.last_response_messages = new_messages
            return terminal

        self._last_prompt_ids = list(next_token_ids)
        return TokenizedTurn(
            next_token_ids=list(next_token_ids),
            next_messages=list(self._messages),
            last_env_step=env_step,
            last_response_messages=new_messages,
        )

    async def close(self) -> None:
        await self._message_env.close()

    def _is_overflow(self, *, prompt_len: int) -> bool:
        cap = self._limits.max_trajectory_tokens
        if cap is None:
            return False
        reserve = self._limits.max_generation_tokens or 0
        return prompt_len + reserve > cap


def _terminal(
    *,
    next_messages: list[Message],
    status: RolloutStatus,
) -> TokenizedTurn:
    """Build a terminal TokenizedTurn carrying a status-only MsgResponseStep.

    Reward stays ``None``; the rubric assigns reward off-env per doc 37
    Option B (rubric owns all reward semantics).
    """
    return TokenizedTurn(
        next_token_ids=None,
        next_messages=next_messages,
        last_env_step=MsgResponseStep(done=True, status=status),
        last_response_messages=[],
    )
