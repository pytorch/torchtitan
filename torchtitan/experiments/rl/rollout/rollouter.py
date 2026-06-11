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
    GenerateFn,
    Rollout,
    RolloutGroup,
    RolloutStatus,
    RolloutTurn,
)
from torchtitan.experiments.rl.rubrics import Rubric, RubricOutput

if TYPE_CHECKING:
    # Type-only: importing the generator module here would pull in vLLM at import time.
    from torchtitan.experiments.rl.actors.generator import SamplingConfig

logger = logging.getLogger(__name__)


class Rollouter(Configurable):
    """Turns a problem (train/val datasets, the `MessageEnv` to build per sample, and a
    `Rubric`) into scored rollouts — the RL training data.

    Like a `Dataloader` turns a `Dataset` into training batches, a `Rollouter`
    turns a problem into rollouts: it builds the envs, drives them against the inference engine
    (via a `generate_fn` the controller provides), and scores the results with `score_group`.

    Subclass only to override specific methods, such as `score_group` for cross-sibling scoring,
    or `make_env_group` for custom logic, such as using a pool of envs instead of creating a new one.

    The flow for one prompt group: the controller passes a `generate_fn` callable; each rollout
    drives its own calls, so the generator runs a whole group's calls together in one continuous
    batch.

        sample = rollouter.get_training_sample()        # one sample from the dataset
        group = await rollouter.run_group_rollouts(     # build envs, drive turns, score
            generate_fn=generate_fn, sample=sample, group_id="step=3/group=0",
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
            env_input: the env initial input shared by the group.

        Returns:
            One `RubricOutput` per rollout, in input order.
        """
        return await self.rubric.score_group(rollouts, env_input)

    async def run_group_rollouts(
        self,
        *,
        generate_fn: GenerateFn,
        sample: object,
        group_id: str,
        group_size: int,
        sampling: SamplingConfig,
        renderer: Renderer,
    ) -> RolloutGroup:
        """Roll out and score one prompt group.

        Builds `group_size` sibling envs from one sample and drives them concurrently;
        each sibling drives its own `generate_fn` calls, so the generator runs a whole
        group's calls together in one continuous batch. Then `score_group` fills each reward.

        For custom logic, users can override this method.

        Args:
            generate_fn: Async callable that returns a Completion given a prompt.
            sample: Dataset sample shared by the group.
            group_id: Stable group id; siblings share it for advantage centering.
            group_size: Number of sibling rollouts.
            sampling: Sampling config for every generate call in the group.
            renderer: Renderer shared by the group's envs.

        Returns:
            One scored `RolloutGroup`.
        """
        # One prompt becomes [env] * group_size.
        envs = self.make_env_group(
            sample=sample, group_size=group_size, renderer=renderer
        )

        try:
            # produce the rollouts
            rollouts = await asyncio.gather(
                *(
                    self._run_single_rollout(
                        generate_fn=generate_fn,
                        env=env,
                        sampling=sampling,
                        group_id=group_id,
                        rollout_id=f"{group_id}/sample={sample_idx}",
                    )
                    for sample_idx, env in enumerate(envs)
                )
            )
        finally:
            # close the envs
            await asyncio.gather(*(env.close() for env in envs), return_exceptions=True)

        # score
        outputs = await self.score_group(rollouts, sample)
        for rollout, output in zip(rollouts, outputs, strict=True):
            rollout.reward = output.reward
            rollout.reward_breakdown = output.reward_breakdown

        # TODO: move advantage calculation to here

        return RolloutGroup(group_id=group_id, rollouts=rollouts)

    async def _run_single_rollout(
        self,
        *,
        generate_fn: GenerateFn,
        env: TokenEnv,
        sampling: SamplingConfig,
        group_id: str,
        rollout_id: str,
    ) -> Rollout:
        """Produce a single rollout, alternating between env and generator calls,
        until the env is terminal (env `done`, truncation, errors).

        For custom logic, users can override this method.


        Args:
            generate_fn: Async callable that runs one generation; keeps the rollouter
                decoupled from the generator actor.
            env: The env for this rollout; `run_group_rollouts` closes it.
            sampling: Sampling config for every generate call.
            group_id: The GRPO group id.
            rollout_id: Stable id for this rollout, unique within the group; the per-turn
                `request_id` prefix, and stored as `Rollout.sample_id`.

        Returns:
            One unscored `Rollout` (reward filled later by the controller).
        """
        turns: list[RolloutTurn] = []
        status = RolloutStatus.ERROR
        try:
            step = await env.init()
            while not step.status.is_terminal():

                # generator call
                completion = await generate_fn(
                    prompt_token_ids=step.next_prompt_token_ids,
                    request_id=f"{rollout_id}/turn={len(turns)}",
                    sampling_config=sampling,
                )

                # env call
                next_step = await env.step(completion)

                # full snapshot of this turn from a token and message perspective
                turns.append(
                    RolloutTurn(
                        prompt_token_ids=step.next_prompt_token_ids or [],
                        prompt_messages=step.next_prompt_messages or [],
                        completion_token_ids=completion.token_ids,
                        completion_logprobs=completion.token_logprobs,
                        completion_message=next_step.completion_message,
                        env_messages=next_step.env_messages,
                        env_rewards=next_step.env_rewards,
                        policy_version=completion.policy_version,
                        metrics=completion.metrics,
                    )
                )

                # holds the input for next generation call
                step = next_step

            status = step.status
        except Exception:
            logger.exception(
                "rollout %s failed after %d turn(s); marking ERROR",
                rollout_id,
                len(turns),
            )
            status = RolloutStatus.ERROR

        return Rollout(
            group_id=group_id,
            sample_id=rollout_id,
            status=status,
            turns=turns,
        )
