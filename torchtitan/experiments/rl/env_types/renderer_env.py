# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from renderers import Message, Renderer, ToolSpec

from torchtitan.experiments.rl.env_types.message_env import MessageEnv
from torchtitan.experiments.rl.rollouts.types import RolloutStatus

if TYPE_CHECKING:
    from torchtitan.experiments.rl.actors.generator import Completion

logger = logging.getLogger(__name__)


# TODO: revisit the `RendererEnvConfig` name once tool/browser envs share these limits.
@dataclass(kw_only=True, slots=True)
class RendererEnvConfig:
    """Limits enforced by `RendererEnv` (prompt cap + step timeout)."""

    max_rollout_tokens: int | None = None
    """Hard cap on prompt length (tokens). If the next prompt meets/exceeds
    it, the turn is terminal; `None` disables the check."""

    step_timeout_s: float | None = 1800.0
    """Wall-clock timeout for one `MessageEnv.step_message` call."""


@dataclass(kw_only=True, slots=True)
class TokenizedStepOutput:
    """One turn boundary in token space.

    Returned by both `RendererEnv.initial_prompt` and `RendererEnv.step_completion`.
    """

    next_prompt_token_ids: list[int] | None  # [L_prompt] or None
    """Prompt for the NEXT generator call; `None` on a terminal turn."""

    next_prompt_messages: list[Message]  # [M_prompt]
    """Full conversation at the next-prompt point."""

    status: RolloutStatus
    """`ONGOING` while the rollout runs; a terminal `RolloutStatus` otherwise."""

    assistant_message: Message | None = None
    """The model's parsed turn (generator output as a message). `None` on the
    initial prompt and on parse/overflow failures."""

    env_messages: list[Message] = field(default_factory=list)  # [M_env]
    """The env's reply messages this turn (tool / user). Empty on the initial
    prompt and on truncation/abort/timeout."""

    env_reward_components: dict[str, float] = field(default_factory=dict)
    """Optional per-step reward components the env attached, forwarded to
    `RolloutTurn.reward_components` for the rubric."""


class RendererEnv:
    """Wraps a user `MessageEnv` and handles the messages <-> tokens plumbing.

    Owns the renderer, the length-stop / parse-error / timeout classification,
    and the prompt-overflow check.

    Args:
        message_env: The user's `MessageEnv` subclass instance.
        renderer: Renderer from `RendererConfig.build`.
        config: `RendererEnvConfig` (prompt cap, step timeout).

    Example:

        env = RendererEnv(message_env=SumDigitsEnv(...), renderer=renderer)
        step = await env.initial_prompt()
        while not step.status.is_terminal():
            completion = (await generator.generate([step.next_prompt_token_ids]))[0]
            step = await env.step_completion(completion)
    """

    def __init__(
        self,
        *,
        message_env: MessageEnv,
        renderer: Renderer,
        config: RendererEnvConfig | None = None,
    ) -> None:
        self._message_env = message_env
        self._renderer = renderer
        self._config = config or RendererEnvConfig()
        self._tools: list[ToolSpec] | None = None
        self._messages: list[Message] = []
        self._last_prompt_ids: list[int] = []

    async def initial_prompt(self) -> TokenizedStepOutput:
        """Render the initial conversation into the first generator prompt."""
        env_reset = await self._message_env.reset()
        self._messages = env_reset.messages

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
                next_prompt_messages=list(self._messages),
                status=RolloutStatus.TRUNCATED_PROMPT_OVERFLOW,
            )

        self._last_prompt_ids = list(token_ids)
        return TokenizedStepOutput(
            next_prompt_token_ids=list(token_ids),
            next_prompt_messages=list(self._messages),
            status=RolloutStatus.ONGOING,
        )

    async def step_completion(self, completion: Completion) -> TokenizedStepOutput:
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
                next_prompt_messages=list(self._messages),
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
                next_prompt_messages=list(self._messages),
                status=RolloutStatus.TRUNCATED_LENGTH,
                assistant_message=assistant,
            )
        if completion.finish_reason == "abort":
            return _terminal(
                next_prompt_messages=list(self._messages),
                status=RolloutStatus.ERROR_ABORT,
                assistant_message=assistant,
            )

        # Apply the user's env step under a timeout
        timeout = self._config.step_timeout_s
        try:
            if timeout is None:
                env_step = await self._message_env.step_message(assistant)
            else:
                env_step = await asyncio.wait_for(
                    self._message_env.step_message(assistant), timeout=timeout
                )
        except TimeoutError:
            logger.warning(
                "step_message timed out after %ss; -> ERROR_TIMEOUT", timeout
            )
            return _terminal(
                next_prompt_messages=list(self._messages),
                status=RolloutStatus.ERROR_TIMEOUT,
                assistant_message=assistant,
            )

        self._messages.append(assistant)
        self._messages.extend(env_step.messages)

        if env_step.done:
            return _terminal(
                next_prompt_messages=list(self._messages),
                status=RolloutStatus.COMPLETED,
                assistant_message=assistant,
                env_messages=env_step.messages,
                env_reward_components=env_step.reward_components,
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
                next_prompt_messages=list(self._messages),
                status=RolloutStatus.TRUNCATED_PROMPT_OVERFLOW,
                assistant_message=assistant,
                env_messages=env_step.messages,
                env_reward_components=env_step.reward_components,
            )

        self._last_prompt_ids = list(next_prompt_token_ids)
        return TokenizedStepOutput(
            next_prompt_token_ids=list(next_prompt_token_ids),
            next_prompt_messages=list(self._messages),
            status=RolloutStatus.ONGOING,
            assistant_message=assistant,
            env_messages=env_step.messages,
            env_reward_components=env_step.reward_components,
        )

    async def close(self) -> None:
        await self._message_env.close()

    def _is_prompt_overflow(self, *, prompt_len: int) -> bool:
        cap = self._config.max_rollout_tokens
        return cap is not None and prompt_len >= cap


def _terminal(
    *,
    next_prompt_messages: list[Message],
    status: RolloutStatus,
    assistant_message: Message | None = None,
    env_messages: list[Message] | None = None,
    env_reward_components: dict[str, float] | None = None,
) -> TokenizedStepOutput:
    """Build a terminal `TokenizedStepOutput` with the given status."""
    return TokenizedStepOutput(
        next_prompt_token_ids=None,
        next_prompt_messages=next_prompt_messages,
        status=status,
        assistant_message=assistant_message,
        env_messages=list(env_messages or []),
        env_reward_components=dict(env_reward_components or {}),
    )
