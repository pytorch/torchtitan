# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Token-level adapter around a :class:`MessageEnv`.

The rollout driver wants a token-in / token-out interface: render the
current conversation to tokens, call the sampler, parse the sampled
tokens back to a :class:`Message`, hand it to the env, render the next
conversation, repeat. :class:`TokenEnv` owns that boilerplate plus four
edge cases the env author shouldn't have to handle:

1. **parse failure** — ``renderer.parse_response`` raised. Terminal
   :class:`EnvStep` with ``status=ERROR``.
2. **length-stop** — the sampler returned ``finish_reason="length"``.
   Terminal :class:`EnvStep` with ``status=TRUNCATED``.
3. **context overflow** — the next prompt + generation reserve would
   exceed ``max_trajectory_tokens``. Terminal :class:`EnvStep` with
   ``status=TRUNCATED``.
4. **step timeout** — ``env.step`` didn't return inside
   ``step_timeout_s`` seconds. Terminal :class:`EnvStep` with
   ``status=ERROR``.

Modelled on tinker-cookbook's ``EnvFromMessageEnv``
(``tinker_cookbook/rl/message_env.py:49-164``, Apache-2.0) with the
length / parse / context / timeout knobs preserved verbatim.
Tokenization runs in :func:`asyncio.to_thread` because HuggingFace
fast tokenizers release the GIL during Rust encoding; with N
concurrent rollouts on one event loop, threading gives real CPU
parallelism rather than just event-loop responsiveness.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from renderers import Message, ParsedResponse, Renderer, ToolSpec

from torchtitan.experiments.rl.actors.generator import GenerateOutput
from torchtitan.experiments.rl.envs.types import EnvStep, MessageEnv
from torchtitan.experiments.rl.types import RolloutStatus

logger = logging.getLogger(__name__)

__all__ = ["TokenEnv", "TokenEnvConfig"]


@dataclass(kw_only=True, slots=True)
class TokenEnvConfig:
    """Termination knobs for :class:`TokenEnv`.

    Defaults match tinker's ``EnvFromMessageEnv``: zero reward on parse
    failure / length stop / context overflow. Envs that want a negative
    shaping reward (e.g. `-1.0` on parse failure) override per-instance.
    """

    failed_parse_reward: float = 0.0
    context_overflow_reward: float = 0.0
    max_trajectory_tokens: int | None = None
    max_generation_tokens: int | None = None
    step_timeout_s: float | None = None


class TokenEnv:
    """Adapt a :class:`MessageEnv` to the token-level loop the driver expects.

    Example::

        token_env = TokenEnv(SumDigitsEnv(payload), renderer_pool)
        prompt_tokens, prompt_messages = await token_env.initial_observation()
        completion = await completion_fn(prompt_tokens, sampling)
        env_step, response_messages = await token_env.step(completion)
        if env_step.done:
            ...
    """

    def __init__(
        self,
        env: MessageEnv,
        renderer: Renderer,
        *,
        config: TokenEnvConfig | None = None,
    ) -> None:
        self._env = env
        self._renderer = renderer
        self._config = config or TokenEnvConfig()
        self._messages: list[Message] = []
        self._tools: list[ToolSpec] = []

    async def initial_observation(self) -> tuple[list[int], list[Message]]:
        """Render the env's initial messages to tokens.

        Returns ``(prompt_tokens, prompt_messages)`` so the rollout
        driver can snapshot both on the per-turn :class:`RolloutTurn`.
        """
        reset = await self._env.reset()
        self._messages = list(reset.messages)
        self._tools = list(reset.tools)
        return await self._render_prompt(), list(self._messages)

    async def step(self, completion: GenerateOutput) -> tuple[EnvStep, list[Message]]:
        """Translate sampler output into the env's :class:`EnvStep`.

        Returns ``(env_step, response_messages)`` where
        ``response_messages`` are the new messages this turn produced
        (assistant + any env follow-ups) — the driver stashes them on
        :class:`RolloutTurn`.

        Edge cases handled here, not by the env:

        - ``finish_reason == "length"`` → terminal ``EnvStep`` (TRUNCATED).
        - ``renderer.parse_response`` raised → terminal ``EnvStep`` (ERROR).
        - ``env.step`` timed out → terminal ``EnvStep`` (ERROR).
        - next prompt would exceed ``max_trajectory_tokens`` →
          terminal ``EnvStep`` (TRUNCATED).
        """
        if completion.finish_reason == "length":
            return (
                EnvStep(
                    reward=self._config.context_overflow_reward,
                    done=True,
                    status=RolloutStatus.TRUNCATED,
                    metadata={"reason": "length"},
                ),
                [],
            )

        try:
            parsed: ParsedResponse = self._renderer.parse_response(
                completion.response_token_ids
            )
        except Exception as exc:
            logger.warning("renderer.parse_response raised: %s", exc, exc_info=False)
            return (
                EnvStep(
                    reward=self._config.failed_parse_reward,
                    done=True,
                    status=RolloutStatus.ERROR,
                    metadata={"reason": "parse_error", "exc": type(exc).__name__},
                ),
                [],
            )

        assistant_msg = _assistant_message(parsed)
        env_step = await self._call_step(assistant_msg)
        response_messages = [assistant_msg, *env_step.messages]
        self._messages.append(assistant_msg)
        self._messages.extend(env_step.messages)

        if not env_step.done and await self._exceeds_context():
            return (
                EnvStep(
                    reward=self._config.context_overflow_reward,
                    done=True,
                    status=RolloutStatus.TRUNCATED,
                    metadata={**env_step.metadata, "reason": "context_overflow"},
                ),
                response_messages,
            )
        return env_step, response_messages

    async def next_observation(self) -> tuple[list[int], list[Message]]:
        """Render the next-turn prompt after a non-terminal step.

        Returns ``(prompt_tokens, prompt_messages)`` for the driver's
        next :class:`RolloutTurn`.
        """
        return await self._render_prompt(), list(self._messages)

    async def close(self) -> None:
        """Best-effort env teardown. Errors are logged and swallowed so
        a failing close doesn't mask a real rollout error from
        :func:`asyncio.gather`."""
        try:
            await self._env.close()
        except Exception:
            logger.exception("env.close() raised; swallowing")

    # ------------------------------------------------------------------ internals

    async def _call_step(self, msg: Message) -> EnvStep:
        coro = self._env.step(msg)
        if self._config.step_timeout_s is None:
            return await coro
        try:
            return await asyncio.wait_for(coro, timeout=self._config.step_timeout_s)
        except asyncio.TimeoutError:
            return EnvStep(
                reward=self._config.failed_parse_reward,
                done=True,
                status=RolloutStatus.ERROR,
                metadata={"reason": "step_timeout"},
            )

    async def _render_prompt(self) -> list[int]:
        # ``asyncio.to_thread`` keeps the controller's event loop
        # responsive across N concurrent rollouts. HF fast tokenizers
        # release the GIL during Rust encoding so threading also
        # buys real CPU parallelism.
        return await asyncio.to_thread(
            lambda: list(
                self._renderer.render_ids(
                    self._messages,
                    tools=self._tools or None,
                    add_generation_prompt=True,
                )
            )
        )

    async def _exceeds_context(self) -> bool:
        budget = self._config.max_trajectory_tokens
        if budget is None:
            return False
        prompt_len = len(await self._render_prompt())
        reserve = self._config.max_generation_tokens or 0
        return prompt_len + reserve > budget


def _assistant_message(parsed: ParsedResponse) -> Message:
    """Build an assistant :class:`Message` from a renderer parse result."""
    msg: Message = {"role": "assistant", "content": parsed.content}
    if parsed.tool_calls:
        msg["tool_calls"] = parsed.tool_calls
    if parsed.reasoning_content:
        msg["reasoning_content"] = parsed.reasoning_content
    return msg
