# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The rollout driver + the group fanout helper.

:func:`do_single_rollout` drives one :class:`TokenEnv` to completion;
termination policy lives in :class:`TokenEnv`, so this body is just
render → sample → step → append.

:func:`do_rollout_group` wraps N sibling envs in ``asyncio.gather``
with ``try/finally`` cleanup. The producer task calls this once per
group.

The completion function is injected so the driver has no Monarch or
vLLM dependency — its contract is pure
``(prompt_token_ids, sampling) -> GenerateOutput``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence

from renderers import Renderer

from torchtitan.experiments.rl.actors.generator import GenerateOutput, SamplingConfig
from torchtitan.experiments.rl.envs.token_env import TokenEnv, TokenEnvConfig
from torchtitan.experiments.rl.envs.types import MessageEnv
from torchtitan.experiments.rl.types import (
    RolloutOutput,
    RolloutStatus,
    RolloutTurn,
    validate_rollout_output,
)

__all__ = ["CompletionFn", "do_rollout_group", "do_single_rollout"]


CompletionFn = Callable[[list[int], SamplingConfig], Awaitable[GenerateOutput]]
"""Async ``(prompt_token_ids, sampling) -> GenerateOutput``.

Built by the controller as a closure over ``self.generator`` so the
driver itself has no Monarch dependency."""


async def do_single_rollout(
    *,
    token_env: TokenEnv,
    completion_fn: CompletionFn,
    sampling: SamplingConfig,
    group_id: str,
    sample_idx: int,
    max_turns: int,
) -> RolloutOutput:
    """Drive one env to terminal, return its :class:`RolloutOutput`.

    Loop per turn: snapshot prompt → sample tokens → step env →
    append a :class:`RolloutTurn`. Hits ``max_turns`` ⇒ ``status=TRUNCATED``.

    Args:
        token_env: token-level adapter around the user env.
        completion_fn: closure that hits the generator.
        sampling: ``SamplingConfig`` forwarded to ``completion_fn``.
        group_id: GRPO group identifier (shared by siblings).
        sample_idx: position of this rollout within its group.
        max_turns: hard cap.

    Returns:
        A validated :class:`RolloutOutput`.
    """
    prompt_token_ids, prompt_messages = await token_env.initial_observation()

    turns: list[RolloutTurn] = []
    final_reward: float | None = None
    final_components: dict[str, float] = {}
    overall_status = RolloutStatus.TRUNCATED  # default if loop exits via max_turns

    for _ in range(max_turns):
        completion = await completion_fn(prompt_token_ids, sampling)
        env_step, response_messages = await token_env.step(completion)

        if env_step.reward is not None:
            final_reward = float(env_step.reward)
            final_components = dict(env_step.reward_components)

        turns.append(
            RolloutTurn(
                prompt_token_ids=prompt_token_ids,
                response_token_ids=list(completion.response_token_ids),
                response_logprobs=list(completion.response_logprobs),
                prompt_messages=prompt_messages,
                response_messages=response_messages,
                policy_version=completion.policy_version,
            )
        )

        if env_step.done:
            overall_status = env_step.status or RolloutStatus.COMPLETED
            break

        prompt_token_ids, prompt_messages = await token_env.next_observation()

    output = RolloutOutput(
        group_id=group_id,
        sample_idx=sample_idx,
        turns=turns,
        status=overall_status,
        reward=final_reward,
        reward_components=final_components,
    )
    validate_rollout_output(output)
    return output


async def do_rollout_group(
    *,
    envs: Sequence[MessageEnv],
    renderer: Renderer,
    completion_fn: CompletionFn,
    sampling: SamplingConfig,
    group_id: str,
    max_turns: int,
    token_env_config: TokenEnvConfig | None = None,
) -> list[RolloutOutput]:
    """Run N sibling envs concurrently; close all in ``finally``.

    Constructs one :class:`TokenEnv` per env, fans out via
    ``asyncio.gather``, and closes all envs in a ``finally`` block so
    a single rollout failure doesn't leak resources from the others.
    """
    token_envs = [TokenEnv(env, renderer, config=token_env_config) for env in envs]
    try:
        return await asyncio.gather(
            *(
                do_single_rollout(
                    token_env=t,
                    completion_fn=completion_fn,
                    sampling=sampling,
                    group_id=group_id,
                    sample_idx=i,
                    max_turns=max_turns,
                )
                for i, t in enumerate(token_envs)
            )
        )
    finally:
        await asyncio.gather(*(t.close() for t in token_envs), return_exceptions=True)
