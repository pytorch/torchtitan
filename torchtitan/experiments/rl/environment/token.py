# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from renderers import Message, Renderer, ToolSpec

from torchtitan.config import Configurable
from torchtitan.experiments.rl.environment.message import MessageEnv
from torchtitan.experiments.rl.rollout.types import RolloutStatus
from torchtitan.experiments.rl.types import Completion

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class TokenEnvOutput:
    """What `TokenEnv` produced this turn. Output only — it does not carry the prompt that was fed
    in or generator metadata (logprobs, policy_version). The rollout loop folds it together
    with those inputs into a durable `RolloutTurn`."""

    next_prompt_token_ids: list[int] | None  # [num_prompt_tokens] or None
    """Tokens for the next `generate` call. `None` once no further prompt is
    rendered (COMPLETED, or a completion-level error)."""

    next_prompt_messages: list[Message] | None = None  # [num_prompt_messages] or None
    """Message form of `next_prompt_token_ids`; `None` whenever that is `None`."""

    status: RolloutStatus
    """`ONGOING` while the rollout runs; a terminal `RolloutStatus` otherwise."""

    completion_message: Message | None = None
    """This turn's completion decoded from the generator's tokens into a message by the renderer
    (this is the TokenEnv's parse, not the raw generator output). `None` on init and on parse failure."""

    env_messages: list[Message] = field(default_factory=list)  # [num_env_messages]
    """The env's reply on this step (tool / user). Empty on init and on terminals
    where the env was not stepped."""

    env_rewards: dict[str, float] = field(default_factory=dict)
    """Per-step reward signals from the env; empty when the env was not stepped. This
    can be used by the rubric to compute the final rollout reward."""


class TokenEnv(Configurable):
    """Token-space env used to wrap a `MessageEnv` and drive it in tokens.

    In a rollout, the input and output of a generator are in tokens (Tokens-In-Tokens-Out).
    However, the MessageEnv is in messages (Messages-In-Messages-Out).

    Some process is necessary to:
    a. Decode the generator's completion tokens into messages;
    b. Call the MessageEnv;
    c. Encode the env response back into tokens for the next turn;

    This wrapper fills this role, using a renderer to convert between the generator and MessageEnv.

    Beyond encoding/decoding, several checks are necessary, e.g. prompt too long, too many
    turns, parse errors, timeouts. This wrapper also takes care of that, so we can keep the
    rollout loop clean and simple.

    If users have extra or different logic, they can wrap their MessageEnv with another class instead.

    Args:
        config: `TokenEnv.Config`.
        message_env: the user's `MessageEnv` subclass instance.
        renderer: a `renderers.Renderer` that converts messages <-> token ids.

    Example:

        env = TokenEnv.Config().build(message_env=MyMessageEnv(...), renderer=renderer)
        env_output = await env.init()
        while not env_output.status.is_terminal():
            completion = await generator.generate([env_output.next_prompt_token_ids])
            env_output = await env.step(completion)
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
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
        config: Config,
        *,
        message_env: MessageEnv,
        renderer: Renderer,
    ) -> None:
        self._message_env = message_env
        self._renderer = renderer
        self._config = config
        self._tools: list[ToolSpec] | None = None
        self._messages: list[Message] = []
        self._last_prompt_token_ids: list[int] = []

    async def init(self) -> TokenEnvOutput:
        """Render the initial conversation into the first generator prompt."""
        env_init_output = await self._message_env.init()

        # Copy our running conversation (messages) so we avoid mutating previous states
        self._messages = list(env_init_output.init_prompt_messages)

        # Render messages into tokens
        self._tools = env_init_output.tools if env_init_output.tools else None
        token_ids = await asyncio.to_thread(
            self._renderer.render_ids,
            messages=self._messages,
            tools=self._tools,
            add_generation_prompt=True,
        )

        # Carry the prompt either way: ONGOING if it fits, else
        # TRUNCATED_PROMPT_TOO_LONG so the over-budget prompt stays debuggable.
        prompt_too_long = self._is_prompt_too_long(prompt_len=len(token_ids))
        if not prompt_too_long:
            self._last_prompt_token_ids = token_ids
        return TokenEnvOutput(
            next_prompt_token_ids=token_ids,
            next_prompt_messages=self._messages,
            status=(
                RolloutStatus.TRUNCATED_PROMPT_TOO_LONG
                if prompt_too_long
                else RolloutStatus.ONGOING
            ),
        )

    async def step(self, completion: Completion) -> TokenEnvOutput:
        """Advance the env by one sampled completion from the generator.

        Args:
            completion: Generator output for the current prompt.

        Returns:
            `TokenEnvOutput` for the next generator call, or a terminal turn
            when the rollout completes, truncates, or errors.
        """
        # Parse first, so a truncated / aborted response still carries its message
        try:
            parsed = await asyncio.to_thread(
                self._renderer.parse_response,
                token_ids=completion.token_ids,
            )
        except Exception:
            logger.exception(
                "parse_response failed (finish_reason=%s, %d tokens); -> ERROR_PARSE",
                completion.finish_reason,
                len(completion.token_ids),
            )
            return TokenEnvOutput(
                next_prompt_token_ids=None,
                next_prompt_messages=None,
                status=RolloutStatus.ERROR_PARSE,
            )

        completion_message: Message = {
            "role": "assistant",
            "content": parsed.content,
        }
        if parsed.reasoning_content:
            completion_message["reasoning_content"] = parsed.reasoning_content
        if parsed.tool_calls:
            completion_message["tool_calls"] = parsed.tool_calls
        # Truncated / aborted: the response is final and partial. Keep it for
        # partial-reward grading and debugging; don't step the env on it.
        # TODO: check if we should step the env on an incomplete message
        if completion.finish_reason == "length":
            return TokenEnvOutput(
                next_prompt_token_ids=None,
                next_prompt_messages=None,
                status=RolloutStatus.TRUNCATED_LENGTH,
                completion_message=completion_message,
            )
        if completion.finish_reason == "abort":
            return TokenEnvOutput(
                next_prompt_token_ids=None,
                next_prompt_messages=None,
                status=RolloutStatus.ERROR_ABORT,
                completion_message=completion_message,
            )

        # Apply the user's env step under a timeout
        timeout = self._config.step_timeout_s
        try:
            if timeout is None:
                step_output = await self._message_env.step(completion_message)
            else:
                step_output = await asyncio.wait_for(
                    self._message_env.step(completion_message), timeout=timeout
                )
        except TimeoutError:
            logger.warning("step timed out after %ss; -> ERROR_TIMEOUT", timeout)
            return TokenEnvOutput(
                next_prompt_token_ids=None,
                next_prompt_messages=None,
                status=RolloutStatus.ERROR_TIMEOUT,
                completion_message=completion_message,
            )

        # TODO(history-edit): We hard-code the logic to only append new messages.
        # This may not satisfy all uses cases, such as compacting the history.
        # Update this when such cases arise.

        # Create a new list to avoid mutating previous states
        self._messages = (
            self._messages + [completion_message] + step_output.env_messages
        )

        if step_output.done:
            return TokenEnvOutput(
                next_prompt_token_ids=None,
                next_prompt_messages=None,
                status=RolloutStatus.COMPLETED,
                completion_message=completion_message,
                env_messages=step_output.env_messages,
                env_rewards=step_output.env_rewards,
            )

        # Prepare the next prompt; full re-render if the renderer can't bridge.
        # `tools` is passed because tool schemas are part of the chat template, so
        # the bridged tokens must match what a full re-render (also tools-aware) produces.
        bridged = await asyncio.to_thread(
            self._renderer.bridge_to_next_turn,
            previous_prompt_ids=self._last_prompt_token_ids,
            previous_completion_ids=completion.token_ids,
            new_messages=step_output.env_messages,
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
        if self._is_prompt_too_long(prompt_len=len(next_prompt_token_ids)):
            return TokenEnvOutput(
                next_prompt_token_ids=None,
                next_prompt_messages=None,
                status=RolloutStatus.TRUNCATED_PROMPT_TOO_LONG,
                completion_message=completion_message,
                env_messages=step_output.env_messages,
                env_rewards=step_output.env_rewards,
            )

        self._last_prompt_token_ids = next_prompt_token_ids
        return TokenEnvOutput(
            next_prompt_token_ids=next_prompt_token_ids,
            next_prompt_messages=self._messages,
            status=RolloutStatus.ONGOING,
            completion_message=completion_message,
            env_messages=step_output.env_messages,
            env_rewards=step_output.env_rewards,
        )

    async def close(self) -> None:
        await self._message_env.close()

    def _is_prompt_too_long(self, *, prompt_len: int) -> bool:
        cap = self._config.max_rollout_tokens
        return cap is not None and prompt_len >= cap
