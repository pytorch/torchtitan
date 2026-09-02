# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from monarch.actor import ProcMesh, this_host

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import Configurable
from torchtitan.experiments.rl.environment import MessageEnv, TokenEnv
from torchtitan.experiments.rl.renderer import build_renderer
from torchtitan.experiments.rl.rollout.advantage import AdvantageEstimator
from torchtitan.experiments.rl.rollout.types import (
    GenerateFn,
    Rollout,
    RolloutGroup,
    RolloutStatus,
    RolloutTurn,
)
from torchtitan.experiments.rl.rubrics import Rubric, RubricOutput
from torchtitan.experiments.rl.types import RolloutTurnID

if TYPE_CHECKING:
    from renderers import Renderer

    from renderers.configs import BaseRendererConfig

    # Type-only: importing the generator module here would pull in vLLM at import time.
    from torchtitan.experiments.rl.actors.generator import SamplingConfig

    from torchtitan.experiments.rl.actors.rollout_worker import RolloutWorkerActor


logger = logging.getLogger(__name__)


class Rollouter(Configurable):
    """Turns a problem (train/val datasets, the `MessageEnv` to build per sample, and a
    `Rubric`) into scored rollouts — the RL training data.

    Like a `Dataloader` turns a `Dataset` into training batches, a `Rollouter`
    turns a problem into rollouts: its worker builds the envs, drives them against the inference engine
    (via a `generate_fn` the controller provides), and scores the results with `score_group`.

    The flow for one prompt group: the controller passes a `generate_fn` callable; each rollout
    drives its own calls, so the generator runs a whole group's calls together in one continuous
    batch.

        sample = rollouter.get_training_sample()        # one sample from the dataset
        group = await rollouter.run_group_rollouts(     # build envs, drive turns, score
            generate_fn=generate_fn, sample=sample,
            group_id=group_index,  # assigned by the data input loop (a monotonic int)
            group_size=N, sampling=sampling)

    `MessageEnv` works in messages; `TokenEnv` (what `RolloutWorker.make_env_group` returns)
    adds the message <-> token plumbing.

    Example:
        rollouter = Rollouter.Config(
            train_dataset=MyDataset.Config(seed=42),
            validation_dataset=MyDataset.Config(seed=99),
            worker=RolloutWorker.Config(
                rubric=Rubric.Config(
                    reward_fns=[RewardCorrect.Config(), RewardFormat.Config(weight=0.3)]
                ),
                message_env=MyEnv.Config(),
            ),
        ).build()

    Customization:
        Rollouter supports customization at several levels:
          - Sample source: override `Config`'s dataset fields, and/or the
            `get_training_sample` / `get_validation_sample` methods.
          - Group execution, coarse: override `run_group_rollouts` for your own
            orchestration. `RolloutWorker` then becomes optional -- but override
            `setup_async` too, or the worker pool is still spawned unused.
          - Group execution, fine: keep the stock orchestration and point `worker`
            at a `RolloutWorker.Config` subclass, overriding only what you need
            (`make_env_group`, `score_group`, `run_group`).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        train_dataset: Configurable.Config
        """Dataset iterator for training (`next()` yields one env input)."""

        validation_dataset: Configurable.Config
        """Dataset iterator for validation."""

        worker: RolloutWorker.Config
        """How a rollout group is built, driven, scored and advantaged. Selects the
        `RolloutWorker` subclass by config type; one worker is built per pool process."""

        worker_pool_size: int = 4
        """CPU rollout worker processes to spawn on the controller host."""

        num_threads_per_worker: int = 4
        """Size of each worker process's default thread pool executor, i.e. the pool
        behind every `asyncio.to_thread` call in that process."""

        def __post_init__(self) -> None:
            if self.worker_pool_size < 1:
                raise ValueError(
                    "worker_pool_size must be at least 1, got "
                    f"{self.worker_pool_size}"
                )
            if self.num_threads_per_worker < 1:
                raise ValueError(
                    "num_threads_per_worker must be at least 1, got "
                    f"{self.num_threads_per_worker}"
                )

    def __init__(self, config: Config) -> None:
        self._config = config
        self._train_dataset = config.train_dataset.build()
        self._validation_dataset = config.validation_dataset.build()

        self._worker_actors: RolloutWorkerActor | None = None
        self._worker_mesh: ProcMesh | None = None

    # TODO: revisit this abstraction: should it return a sample or a dataset or an iterator?
    def get_training_sample(self) -> object:
        """Get one training sample (the env input) from the training dataset."""
        return next(self._train_dataset)

    def get_validation_sample(self) -> object:
        """Get one validation sample (the env input) from the validation dataset."""
        return next(self._validation_dataset)

    async def setup_async(
        self,
        *,
        renderer_config: BaseRendererConfig,
        hf_assets_path: str,
    ) -> None:
        """Spawn and initialize the owned worker proc mesh and actor pool."""
        # Import lazily to avoid a circular dependency through Rollouter.Config.
        from torchtitan.experiments.rl.actors.rollout_worker import RolloutWorkerActor

        if self._worker_mesh is not None or self._worker_actors is not None:
            raise RuntimeError("rollout worker pool is already initialized")

        self._worker_mesh = this_host().spawn_procs(
            per_host={"cpus": self._config.worker_pool_size},
        )
        self._worker_actors = self._worker_mesh.spawn(
            "rollout_worker",
            RolloutWorkerActor,
            worker_config=self._config.worker,
            num_threads=self._config.num_threads_per_worker,
        )
        await self._worker_actors.setup_async.call(
            renderer_config=renderer_config,
            hf_assets_path=hf_assets_path,
        )

    async def close(self) -> None:
        """Stop the owned rollout worker proc mesh."""
        worker_mesh = self._worker_mesh
        self._worker_actors = None
        self._worker_mesh = None
        if worker_mesh is not None:
            await worker_mesh.stop()

    async def sync_log_step(self, step: int) -> None:
        """Propagate the controller log step to every rollout worker."""
        if self._worker_actors is not None:
            await self._worker_actors.sync_log_step.call(step)

    async def run_group_rollouts(
        self,
        *,
        generate_fn: GenerateFn,
        sample: object,
        group_id: int,
        group_size: int,
        sampling: SamplingConfig,
    ) -> RolloutGroup:
        """Roll out and score one prompt group.

        Builds `group_size` sibling envs from one sample and drives them concurrently;
        each sibling drives its own `generate_fn` calls, so the generator runs a whole
        group's calls together in one continuous batch. Then `score_group` fills each reward.

        Args:
            generate_fn: Async callable that returns a Completion given a prompt.
            sample: Dataset sample shared by the group.
            group_id: Stable group id; siblings share it for advantage centering.
            group_size: Number of sibling rollouts.
            sampling: Sampling config for every generate call in the group.

        Returns:
            One scored `RolloutGroup`.
        """
        if self._worker_actors is None:
            raise RuntimeError("rollout worker pool is not initialized")

        # Use Monarch `choose` API to randomly select an actor in the mesh, and
        # send the message to its `run_group` endpoint.
        return await self._worker_actors.run_group.choose(
            generate_fn=generate_fn,
            sample=sample,
            group_id=group_id,
            group_size=group_size,
            sampling=sampling,
        )


class RolloutWorker(Configurable):
    """Builds, executes, scores, and advantages one rollout group."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        rubric: Rubric.Config
        """Reward functions + weights used by `score_group`."""

        message_env: MessageEnv.Config
        """The env to build per sample; `make_env_group` calls `build(env_input=sample)`."""

        token_env: TokenEnv.Config = field(default_factory=TokenEnv.Config)
        """`TokenEnv` wraps the `MessageEnv` in `make_env_group`."""

        advantage: Configurable.Config = field(
            default_factory=AdvantageEstimator.Config
        )
        """Post-scoring advantage estimator. Default = Dr.GRPO (mean-baseline only);
        set `AdvantageEstimator.Config(should_std_normalize=True)` for standard GRPO."""

    def __init__(self, config: Config) -> None:
        self.rubric: Rubric = config.rubric.build()
        self._message_env_config = config.message_env
        self._token_env_config = config.token_env
        self.advantage_estimator: AdvantageEstimator = config.advantage.build()
        self._renderer: Renderer

    async def setup_async(
        self,
        *,
        renderer_config: BaseRendererConfig,
        hf_assets_path: str,
    ) -> None:
        """Build runtime dependencies after the worker actor is spawned."""
        tokenizer = HuggingFaceTokenizer(tokenizer_path=hf_assets_path)
        self._renderer = build_renderer(tokenizer=tokenizer, config=renderer_config)

    def make_env_group(
        self,
        *,
        sample: object,
        group_size: int,
    ) -> list[TokenEnv]:
        """Construct `group_size` single-use envs from one dataset sample.

        Args:
            sample: the dataset sample (the env input) from `Rollouter.get_training_sample` / `Rollouter.get_validation_sample`.
            group_size: number of sibling envs for this prompt group.

        Returns:
            `TokenEnv` * `group_size` instances, each ready for one rollout.
        """
        return [
            self._token_env_config.build(
                message_env=self._message_env_config.build(env_input=sample),
                renderer=self._renderer,
            )
            for _ in range(group_size)
        ]

    async def score_group(
        self,
        rollouts: list[Rollout],
        env_input: object,
    ) -> list[RubricOutput]:
        """Score one group's rollouts; `run_group` applies the rewards.

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

    async def run_group(
        self,
        *,
        generate_fn: GenerateFn,
        sample: object,
        group_id: int,
        group_size: int,
        sampling: SamplingConfig,
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

        Returns:
            One scored `RolloutGroup`.
        """
        # One prompt becomes [env] * group_size.
        envs = self.make_env_group(
            sample=sample,
            group_size=group_size,
        )

        # TODO(perf): siblings in a group share the first-turn prompt; tokenize it once per group and
        # reuse across the group_size rollouts (truest spot is the worker's first-turn render).
        try:
            # produce the rollouts
            rollouts = await asyncio.gather(
                *(
                    self._run_single_rollout(
                        generate_fn=generate_fn,
                        env=env,
                        # Offset the base seed per sample so a group's n=1
                        # requests are diverse yet reproducible run-to-run.
                        sampling=(
                            sampling
                            if sampling.seed is None
                            else replace(sampling, seed=sampling.seed + sample_idx)
                        ),
                        group_id=group_id,
                        rollout_id=sample_idx,
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

        # Post-scoring: turn group rewards into per-rollout advantages.
        group = RolloutGroup(group_id=group_id, rollouts=rollouts)
        advantages = self.advantage_estimator(group)
        for rollout, advantage in zip(group.rollouts, advantages, strict=True):
            rollout.advantage = advantage
        return group

    async def _run_single_rollout(
        self,
        *,
        generate_fn: GenerateFn,
        env: TokenEnv,
        sampling: SamplingConfig,
        group_id: int,
        rollout_id: int,
    ) -> Rollout:
        """Produce a single rollout, alternating between env and generator calls,
        until the env is terminal (env `done`, truncation, errors).

        For custom logic, users can override this method.

        Args:
            generate_fn: Async callable that runs one generation; keeps the worker
                decoupled from the generator actor.
            env: The env for this rollout; `run_group` closes it.
            sampling: Sampling config for every generate call.
            group_id: The GRPO group id.
            rollout_id: Sibling index within the group; combined with the turn index into the
                per-turn `RolloutTurnID`, and stored as `Rollout.rollout_id`.

        Returns:
            One unscored `Rollout`; `run_group` fills its reward later.
        """
        turns: list[RolloutTurn] = []
        status = RolloutStatus.ERROR
        try:
            env_step = await env.init()
            while not env_step.status.is_terminal():
                turn_rollout_id = RolloutTurnID(
                    group_id=group_id,
                    rollout_id=rollout_id,
                    turn_id=len(turns),
                )

                # generator call
                completion = await generate_fn(
                    prompt_token_ids=env_step.next_prompt_token_ids,
                    request_id=turn_rollout_id.to_string(),
                    # Per-sample sticky key: a sample's turns reuse one generator's prefix cache.
                    routing_session_id=turn_rollout_id.to_string(include_turn=False),
                    sampling_config=sampling,
                )

                # env call
                next_env_step = await env.step(completion)

                # full snapshot of this turn from a token and message perspective
                turns.append(
                    RolloutTurn(
                        rollout_id=turn_rollout_id,
                        prompt_token_ids=env_step.next_prompt_token_ids or [],
                        prompt_messages=env_step.next_prompt_messages or [],
                        completion_token_ids=completion.token_ids,
                        completion_logprobs=completion.token_logprobs,
                        completion_message=next_env_step.completion_message,
                        env_messages=next_env_step.env_messages,
                        env_rewards=next_env_step.env_rewards,
                        min_policy_version=completion.min_policy_version,
                        max_policy_version=completion.max_policy_version,
                        metrics=completion.metrics,
                    )
                )

                # holds the input for next generation call
                env_step = next_env_step

            status = env_step.status
        except Exception:
            logger.exception(
                "rollout %s/rollout=%d failed after %d turn(s); marking ERROR",
                group_id,
                rollout_id,
                len(turns),
            )
            status = RolloutStatus.ERROR

        return Rollout(
            group_id=group_id,
            rollout_id=rollout_id,
            status=status,
            turns=turns,
        )
