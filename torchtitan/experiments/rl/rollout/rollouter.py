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

from renderers import Renderer

from torchtitan.config import Configurable
from torchtitan.experiments.rl.environment import MessageEnv, TokenEnv
from torchtitan.experiments.rl.rollout.types import (
    Rollout,
    RolloutGroup,
    RolloutStatus,
    RolloutTurn,
)
from torchtitan.experiments.rl.rubrics import Rubric, RubricOutput

if TYPE_CHECKING:
    # Type-only: importing the generator module here would pull in vLLM at import time.
    from torchtitan.experiments.rl.actors.generator import GenerateFn, SamplingConfig

logger = logging.getLogger(__name__)

# Runaway guard: a healthy multi-turn env terminates well within this many turns.
_MAX_TURNS = 100


def _sample_id(group_id: str, sample_idx: int) -> str:
    return f"{group_id}/sample={sample_idx}"


class Rollouter(Configurable):
    """Turns a problem (train/val datasets, the `MessageEnv` to build per example, and a
    `Rubric`) into scored rollouts — the RL training data.

    Like a `Dataloader` turns a `Dataset` into training batches, a `Rollouter`
    turns a problem into rollouts: it builds the envs, the controller drives them against
    the inference engine, and `score_group` scores the results.

    Subclass only to override specific methods, such as `score_group` for cross-sibling scoring,
    or `make_env_group` for custom logic, such as using a pool of envs instead of creating a new one.

    The flow for one prompt group (the controller passes a `generate` callable bound to
    one generator; each rollout drives its own `generate` calls, so a whole group's calls
    coalesce into one continuous batch):

        sample = rollouter.get_training_sample()        # one sample from the dataset
        group = await rollouter.run_group_rollouts(     # build envs, drive turns, score
            generate=generate, example=sample, group_id="step=3/group=0",
            group_size=N, sampling=sampling, renderer=renderer)

    `MessageEnv` works in messages; `TokenEnv` (what `make_env_group` returns)
    adds the message <-> token plumbing.

    Example:
        rollouter = Rollouter.Config(
            train_dataset=MyDataset.Config(seed=42),
            validation_dataset=MyDataset.Config(seed=99),
            rubric=Rubric.Config(
                reward_fns=[RewardCorrect.Config(), RewardFormat.Config(weight=0.3)]
            ),
            message_env=MyEnv.Config(),
        ).build()
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        train_dataset: Configurable.Config
        """Dataset iterator for training (`next()` yields one env input)."""

        validation_dataset: Configurable.Config
        """Dataset iterator for validation."""

        rubric: Rubric.Config
        """Reward functions + weights used by `score_group`."""

        message_env: MessageEnv.Config
        """The env to build per sample; `make_env_group` calls `build(env_input=sample)`."""

        token_env: TokenEnv.Config = field(default_factory=TokenEnv.Config)
        """`TokenEnv` limits (e.g. `max_rollout_tokens`) passed to `make_env_group`."""

    def __init__(self, config: Config) -> None:
        self._train_dataset = config.train_dataset.build()
        self._validation_dataset = config.validation_dataset.build()
        self.rubric: Rubric = config.rubric.build()
        self._message_env_config = config.message_env
        self._token_env_config = config.token_env

    # TODO: revisit this abstraction: should it return a sample or a dataset or an iterator?
    def get_training_sample(self) -> object:
        """Get one training sample (the env input) from the training dataset."""
        return next(self._train_dataset)

    def get_validation_sample(self) -> object:
        """Get one validation sample (the env input) from the validation dataset."""
        return next(self._validation_dataset)

    # TODO: revisit the Renderer being injected into `make_env_group` once we
    # know whether Rollouter should own a Renderer (per-rollouter chat templates).
    def make_env_group(
        self,
        *,
        sample: object,
        group_size: int,
        renderer: Renderer,
    ) -> list[TokenEnv]:
        """Construct `group_size` single-use envs from one dataset sample.

        Args:
            sample: the dataset sample (the env input) from `get_training_sample` / `get_validation_sample`.
            group_size: number of sibling envs for this prompt group.
            renderer: Renderer shared by the rollout controller.

        Returns:
            `TokenEnv` * `group_size` instances, each ready for one rollout.
        """
        return [
            self._token_env_config.build(
                message_env=self._message_env_config.build(env_input=sample),
                renderer=renderer,
            )
            for _ in range(group_size)
        ]

    async def score_group(
        self,
        rollouts: list[Rollout],
        env_input: object,
    ) -> list[RubricOutput]:
        """Score one group's rollouts; the controller applies the rewards.

        Default impl delegates to `self.rubric.score_group`. Override for
        cross-sibling scoring (judge, pairwise, diversity) or partial-credit
        reward shaping.

        Args:
            rollouts: Sibling rollouts in one prompt group, already stepped.
            env_input: the env input shared by the group.

        Returns:
            One `RubricOutput` per rollout, in input order.
        """
        return await self.rubric.score_group(rollouts, env_input)

    async def run_group_rollouts(
        self,
        *,
        generate: GenerateFn,
        example: object,
        group_id: str,
        group_size: int,
        sampling: SamplingConfig,
        renderer: Renderer,
    ) -> RolloutGroup:
        """Roll out and score one prompt group.

        Builds `group_size` sibling envs from one example and drives them concurrently;
        each sibling drives its own `generate` calls, so the generator coalesces a whole
        group's calls into one continuous batch. Then `score_group` fills each reward.

        Args:
            generate: Async callable bound to one generator (Monarch hidden); a rollout
                awaits it once per turn.
            example: Dataset example shared by the group.
            group_id: Stable group id; siblings share it for advantage centering.
            group_size: Number of sibling rollouts.
            sampling: Sampling config for every generate call in the group.
            renderer: Renderer shared by the group's envs.

        Returns:
            One scored `RolloutGroup`.
        """
        envs = self.make_env_group(
            sample=example, group_size=group_size, renderer=renderer
        )
        # The group owns the envs' lifecycle: close them once all siblings finish (or one
        # raises / the group is cancelled), so a single rollout never closes its own env.
        try:
            rollouts = await asyncio.gather(
                *(
                    self.run_single_rollout(
                        generate=generate,
                        env=env,
                        sampling=sampling,
                        group_id=group_id,
                        sample_idx=sample_idx,
                    )
                    for sample_idx, env in enumerate(envs)
                )
            )
        finally:
            await asyncio.gather(*(env.close() for env in envs), return_exceptions=True)
        outputs = await self.score_group(rollouts, example)
        for rollout, output in zip(rollouts, outputs, strict=True):
            rollout.reward = output.reward
            rollout.reward_breakdown = output.reward_breakdown
        return RolloutGroup(group_id=group_id, rollouts=rollouts)

    async def run_single_rollout(
        self,
        *,
        generate: GenerateFn,
        env: TokenEnv,
        sampling: SamplingConfig,
        group_id: str,
        sample_idx: int,
    ) -> Rollout:
        """Drive one env to a terminal state via its own `generate` calls.

        Loops `generate -> env.step` until the env is terminal (env `done`, length /
        parse / timeout, or prompt overflow). Single-turn envs end after one step; a
        runaway env is cut off at `_MAX_TURNS`. On any error the rollout keeps the turns
        gathered so far, marked ERROR; the controller scores it afterward via `score_group`.

        Args:
            generate: Async callable bound to one generator (Monarch hidden).
            env: The env for this rollout; `run_group_rollouts` closes it.
            sampling: Sampling config for every generate call.
            group_id: Group id, prefixed onto each turn's `request_id` so all of a group's
                turns route to the same generator (prefix-cache reuse).
            sample_idx: Sample index within the group (0..group_size-1).

        Returns:
            One unscored `Rollout` (reward filled later by the controller).
        """
        turns: list[RolloutTurn] = []
        status = RolloutStatus.ERROR
        try:
            step = await env.init()
            while not step.status.is_terminal() and len(turns) < _MAX_TURNS:
                completion = await generate(
                    step.next_prompt_token_ids,
                    request_id=f"{group_id}/sample={sample_idx}/turn={len(turns)}",
                    sampling_config=sampling,
                )
                next_step = await env.step(completion)
                turns.append(
                    RolloutTurn(
                        prompt_token_ids=step.next_prompt_token_ids or [],
                        prompt_messages=step.next_prompt_messages or [],
                        completion_token_ids=completion.token_ids,
                        completion_logprobs=completion.token_logprobs,
                        policy_version=completion.policy_version,
                        completion_message=next_step.completion_message,
                        env_messages=next_step.env_messages,
                        env_rewards=next_step.env_rewards,
                        metrics=completion.metrics,
                    )
                )
                step = next_step
            if step.status.is_terminal():
                status = step.status
            else:
                logger.warning(
                    "rollout %s/sample=%d hit _MAX_TURNS=%d; truncating",
                    group_id,
                    sample_idx,
                    _MAX_TURNS,
                )
                status = RolloutStatus.TRUNCATED_LENGTH
        except Exception:
            logger.exception(
                "rollout %s/sample=%d failed after %d turn(s); marking ERROR",
                group_id,
                sample_idx,
                len(turns),
            )
            status = RolloutStatus.ERROR
        return Rollout(
            group_id=group_id,
            sample_id=_sample_id(group_id, sample_idx),
            status=status,
            turns=turns,
        )
