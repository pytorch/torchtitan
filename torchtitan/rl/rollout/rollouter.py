# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import itertools
import logging
import statistics
import threading
import time
from collections.abc import Awaitable, Iterable
from dataclasses import dataclass, field, replace
from typing import Any, TYPE_CHECKING

import numpy as np

from torchtitan.components.renderer import RendererConfig

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import Configurable
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.advantage import AdvantageEstimator
from torchtitan.rl.rollout.data import GroupOutcome, RLDataLoader, RLDataset
from torchtitan.rl.rollout.message_env import MessageEnv
from torchtitan.rl.rollout.rubric import Rubric, RubricOutput
from torchtitan.rl.rollout.token_env import TokenEnv
from torchtitan.rl.rollout.types import (
    GenerateFn,
    ReleaseSessionFn,
    Rollout,
    RolloutGroup,
    RolloutStatus,
    RolloutTurn,
    RolloutTurnID,
)

if TYPE_CHECKING:
    from renderers import Renderer

    from torchtitan.rl.actors.system import ActorSystem

    # Type-only: importing the generator module here would pull in vLLM at import time.
    from torchtitan.rl.generator.types import SamplingConfig


logger = logging.getLogger(__name__)

# How long shutdown waits for in-flight groups: `Rollouter.close` for its `run_group` calls, and a
# cancelled dispatch for `cancel_group`. Drains took under 0.3 s once the generators closed; 30 s only
# covers a group stuck in a slow env step.
_RUN_GROUP_DRAIN_TIMEOUT_S = 30.0


class Rollouter(Configurable):
    """Turns a problem (train/val datasets, the `MessageEnv` to build per sample, and a
    `Rubric`) into scored rollouts — the RL training data.

    Like a `Dataloader` turns a `Dataset` into training batches, a `Rollouter`
    turns a problem into rollouts: its worker builds the envs, drives them against the inference engine
    (via a `generate_fn` the controller provides), and scores the results with `score_group`.

    The flow for one prompt group: the controller passes a `generate_fn` callable; each rollout
    drives its own calls, so the generator runs a whole group's calls together in one continuous
    batch.

        index, sample = rollouter.get_training_sample()
        group = await rollouter.run_group_rollouts(     # build envs, drive turns, score
            generate_fn=generate_fn, release_session_fn=release_session_fn,
            sample=sample, group_id=index,
            group_size=N, sampling=sampling)

    `MessageEnv` works in messages; `TokenEnv` (what `RolloutWorker.make_env_group` returns)
    adds the message <-> token plumbing.

    Example:
        rollouter = Rollouter.Config(
            training_dataloader=IterableRLDataLoader.Config(
                dataset=MyDataset.Config()
            ),
            validation_dataset=MyDataset.Config(),
            worker=RolloutWorker.Config(
                rubric=Rubric.Config(
                    reward_fns=[RewardCorrect.Config(), RewardFormat.Config(weight=0.3)]
                ),
                message_env=MyEnv.Config(),
            ),
        ).build()

    Customization:
        Rollouter supports customization at several levels:
          - Sample source: provide source configs for training and validation.
          - Group execution, coarse: override `run_group_rollouts` for your own
            orchestration. `RolloutWorker` then becomes optional -- but override
            `setup_async` too, or the worker pool is still spawned unused.
          - Group execution, fine: keep the stock orchestration and point `worker`
            at a `RolloutWorker.Config` subclass, overriding only what you need
            (`make_env_group`, `score_group`, `run_group`).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        training_dataloader: RLDataLoader.Config
        """Builds a checkpointable iterator of identified training inputs."""

        validation_dataset: RLDataset.Config
        """Builds an iterable validation dataset."""

        extra_validation_datasets: dict[str, RLDataset.Config] = field(
            default_factory=dict
        )
        """Named validation sets beside `validation_dataset`, run by
        `AsyncLoopConfig.extra_validations` entries of the same name."""

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
        training_dataloader = config.training_dataloader.build()
        if not isinstance(training_dataloader, RLDataLoader):
            raise ValueError("training_dataloader must build an RLDataLoader")
        self._training_dataloader = iter(training_dataloader)
        # One read, ack or save at a time: the controller runs them in different threads, and a
        # checkpoint saved in the middle of a read would skip that sample on resume.
        self._training_dataloader_lock = threading.Lock()
        # Group id -> its outcome, until acknowledged. Written on the event loop, popped by
        # acknowledge on a worker thread: they touch different keys (a group finishes before it
        # is acknowledged), and single dict operations are atomic, so it needs no lock.
        self._group_outcomes: dict[int, GroupOutcome] = {}

        # The worker pool's actor handle and the actor system that spawned it.
        self._worker_actors: Any | None = None
        self._actor_system: ActorSystem | None = None
        # `run_group` calls still running on a worker, including those whose caller was cancelled.
        self._inflight_run_group_calls: set[asyncio.Future[RolloutGroup]] = set()

    def get_training_sample(self) -> tuple[int, object]:
        """Return the next globally identified input from the training loader."""
        with self._training_dataloader_lock:
            return next(self._training_dataloader)

    def record_group_outcome(self, group: RolloutGroup) -> None:
        """Remember a finished training group's outcome, for the loader at acknowledgement.

        Zero spread is the training sample builder's drop rule. Truncated rollouts count with
        their rubric reward: their group is as wasted as any other zero-spread group. A group
        gets no outcome if a rollout errored (an infrastructure failure scores a fixed reward that
        says nothing about the task), produced no tokens, or the group has fewer than 2 rollouts.
        A group later dropped as too old still delivers its outcome: it was scored on a real policy.

        Example:
            rewards [1, 1, 1, 1]                 -> GroupOutcome(mean_reward=1.0, zero_spread=True)
            rewards [0, 0, 0, 0], 2 truncated    -> GroupOutcome(mean_reward=0.0, zero_spread=True)
            rewards [0, 1, 0, 0]                 -> GroupOutcome(mean_reward=0.25, zero_spread=False)
            any rollout errored                  -> nothing recorded
        """
        rollouts = group.rollouts
        if len(rollouts) < 2 or any(
            rollout.status.is_error()
            or rollout.reward is None
            or not any(len(turn.completion_token_ids) for turn in rollout.turns)
            for rollout in rollouts
        ):
            return
        rewards = [rollout.reward for rollout in rollouts]
        self._group_outcomes[group.group_id] = GroupOutcome(
            mean_reward=statistics.fmean(rewards),
            zero_spread=statistics.pstdev(rewards) == 0.0,
        )

    def acknowledge_training_sample_ids(self, sample_ids: Iterable[int]) -> None:
        """Mark training samples as safe to omit from the next checkpoint, after telling the loader
        how their groups went."""
        sample_ids = list(sample_ids)
        outcomes = {
            sample_id: self._group_outcomes.pop(sample_id)
            for sample_id in sample_ids
            if sample_id in self._group_outcomes
        }
        with self._training_dataloader_lock:
            self._training_dataloader.record_group_outcomes(outcomes)
            self._training_dataloader.acknowledge(sample_ids)

    def training_data_metrics(self) -> list[m.Metric]:
        """The training loader's metrics since the last call."""
        with self._training_dataloader_lock:
            return self._training_dataloader.metrics()

    def get_validation_samples(
        self, steps: int, validation_set: str | None = None
    ) -> list[object]:
        """Materialize one fresh validation pass, optionally bounded by steps.

        `validation_set` names an `extra_validation_datasets` entry; None is `validation_dataset`.
        """
        validation_data = self._validation_dataset_config(validation_set).build()
        try:
            iterator = iter(validation_data)
        except TypeError as error:
            raise ValueError("validation_dataset must build an iterable") from error
        if steps == -1:
            return list(iterator)
        return list(itertools.islice(iterator, steps))

    def _validation_dataset_config(
        self, validation_set: str | None
    ) -> RLDataset.Config:
        if validation_set is None:
            return self._config.validation_dataset
        try:
            return self._config.extra_validation_datasets[validation_set]
        except KeyError:
            raise ValueError(
                f"unknown validation set {validation_set!r}; extra_validation_datasets has "
                f"{sorted(self._config.extra_validation_datasets)}"
            ) from None

    async def setup_async(
        self,
        *,
        actor_system: ActorSystem,
        tokenizer_config: HuggingFaceTokenizer.Config,
        renderer_config: RendererConfig,
        hf_assets_path: str,
        worker_config: RolloutWorker.Config | None = None,
    ) -> None:
        """Spawn the rollout worker pool through `actor_system` and initialize it.
        `worker_config` replaces `Config.worker`, for a worker config known only at setup."""
        if self._worker_actors is not None:
            raise RuntimeError("rollout worker pool is already initialized")

        self._actor_system = actor_system
        self._worker_actors = await actor_system.spawn_rollout_workers(
            worker_config if worker_config is not None else self._config.worker,
            num_workers=self._config.worker_pool_size,
            num_threads=self._config.num_threads_per_worker,
        )
        await self._worker_actors.setup_async.call(
            tokenizer_config=tokenizer_config,
            renderer_config=renderer_config,
            hf_assets_path=hf_assets_path,
        )

    async def close(self) -> None:
        """Wait for in-flight `run_group` calls, then stop the rollout worker pool.

        Close the generators first, so pending `generate_fn` calls fail fast:

            generators closed -> generate_fn raises -> run_group returns -> stop_rollout_workers()
            run_group still running after _RUN_GROUP_DRAIN_TIMEOUT_S       -> stop_rollout_workers()

        Why wait: if a worker stops while the generator router still owes it a reply, Monarch fails the
        router and the finished job exits 130.
        """
        actor_system = self._actor_system
        self._worker_actors = None
        self._actor_system = None
        calls = tuple(self._inflight_run_group_calls)
        if calls:
            _, pending = await asyncio.wait(calls, timeout=_RUN_GROUP_DRAIN_TIMEOUT_S)
            if pending:
                logger.warning(
                    "Stopping rollout workers with %d run_group calls still running",
                    len(pending),
                )
                for call in pending:
                    call.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
        if actor_system is not None:
            await actor_system.stop_rollout_workers()

    async def sync_log_step(self, step: int) -> None:
        """Propagate the controller log step to every rollout worker."""
        if self._worker_actors is not None:
            await self._worker_actors.sync_log_step.call(step)

    async def run_group_rollouts(
        self,
        *,
        generate_fn: GenerateFn,
        release_session_fn: ReleaseSessionFn,
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
            release_session_fn: Async callable each rollout calls after its last turn, to
                release its routing session.
            sample: Dataset sample shared by the group.
            group_id: Stable group id; siblings share it for advantage centering.
            group_size: Number of sibling rollouts.
            sampling: Sampling config for every generate call in the group.

        Returns:
            One scored `RolloutGroup`.
        """
        if self._worker_actors is None:
            raise RuntimeError("rollout worker pool is not initialized")

        # `choose` (see ActorSystem's call convention) sends the call to the `run_group`
        # endpoint of one worker in the pool.
        call = asyncio.ensure_future(
            self._worker_actors.run_group.choose(
                generate_fn=generate_fn,
                release_session_fn=release_session_fn,
                sample=sample,
                group_id=group_id,
                group_size=group_size,
                sampling=sampling,
            )
        )
        self._inflight_run_group_calls.add(call)
        call.add_done_callback(self._on_run_group_call_done)
        try:
            # Shield `call` from the controller's end-of-training cancel, so `close` can still wait for it.
            return await asyncio.shield(call)
        except asyncio.CancelledError:
            # Monarch runs a call to completion even when its caller is cancelled, so stop the
            # group on its worker (see `RolloutWorkerActor.cancel_group`).
            try:
                async with asyncio.timeout(_RUN_GROUP_DRAIN_TIMEOUT_S):
                    await self._worker_actors.cancel_group.call(group_id)
            except Exception:
                logger.exception(
                    "cancel_group(%d) failed; the group may keep running", group_id
                )
            raise

    def _on_run_group_call_done(self, call: asyncio.Future[RolloutGroup]) -> None:
        self._inflight_run_group_calls.discard(call)
        # A cancelled caller never reads the error; read it so asyncio does not log it.
        if not call.cancelled():
            call.exception()

    def state_dict(self) -> dict[str, object]:
        """Return the checkpoint state of the training input iterator."""
        with self._training_dataloader_lock:
            return self._training_dataloader.state_dict()

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore the training iterator, including unacknowledged inputs."""
        with self._training_dataloader_lock:
            self._training_dataloader.load_state_dict(state_dict)


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
        set `AdvantageEstimator.Config(should_std_normalize=True)` for standard GRPO. It gets the
        scored rollouts only, without the `ERROR_SYSTEM` ones (so possibly none)."""

        rollout_timeout_s: float | None = None
        """Wall-clock limit of one rollout (env init and every turn, not `score_group`): a
        liveness guard against a hung env or generator. Past it the rollout ends as ERROR_TIMEOUT
        and its in-flight generate call is cancelled. Set it well above the p99 rollout time: a
        cut biases toward short rollouts. None means no limit."""

    def __init__(self, config: Config) -> None:
        self.rubric: Rubric = config.rubric.build()
        self._message_env_config = config.message_env
        self._token_env_config = config.token_env
        self.advantage_estimator: AdvantageEstimator = config.advantage.build()
        self._rollout_timeout_s = config.rollout_timeout_s
        self._renderer: Renderer

    async def setup_async(
        self,
        *,
        tokenizer_config: HuggingFaceTokenizer.Config,
        renderer_config: RendererConfig,
        hf_assets_path: str,
    ) -> None:
        """Build runtime dependencies after the worker actor is spawned."""
        tokenizer = tokenizer_config.build(tokenizer_path=hf_assets_path)
        self._renderer = renderer_config.build(tokenizer=tokenizer)

    def make_env_group(
        self,
        *,
        sample: object,
        group_size: int,
    ) -> list[TokenEnv]:
        """Construct `group_size` single-use envs from one dataset sample.

        Args:
            sample: The source sample used to build the environment.
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
            rollouts: Sibling rollouts in one prompt group, already stepped, without the
                `ERROR_SYSTEM` ones (so possibly empty).
            env_input: the env initial input shared by the group.

        Returns:
            One `RubricOutput` per rollout, in input order.
        """
        return await self.rubric.score_group(rollouts, env_input)

    async def run_group(
        self,
        *,
        generate_fn: GenerateFn,
        release_session_fn: ReleaseSessionFn,
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
            release_session_fn: Async callable each rollout calls after its last turn, to
                release its routing session.
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
            # produce the rollouts. A cancel cancels every sibling, and with return_exceptions the
            # gather still waits for each one's `finally` (session release) before it raises.
            results, group_metrics = await gather_rollouts(
                self._run_single_rollout(
                    generate_fn=generate_fn,
                    release_session_fn=release_session_fn,
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
        finally:
            # close the envs
            await asyncio.gather(*(env.close() for env in envs), return_exceptions=True)
        rollouts = rollouts_from_gather(results, group_id=group_id)

        # score, all but the ERROR_SYSTEM rollouts
        scored = [
            rollout
            for rollout in rollouts
            if rollout.status is not RolloutStatus.ERROR_SYSTEM
        ]
        outputs = await self.score_group(scored, sample)
        for rollout, output in zip(scored, outputs, strict=True):
            rollout.reward = output.reward
            rollout.reward_breakdown = output.reward_breakdown

        # Post-scoring: turn group rewards into per-rollout advantages.
        advantages = self.advantage_estimator(
            RolloutGroup(group_id=group_id, rollouts=scored)
        )
        for rollout, advantage in zip(scored, advantages, strict=True):
            rollout.advantage = advantage
        return RolloutGroup(group_id=group_id, rollouts=rollouts, metrics=group_metrics)

    async def _run_single_rollout(
        self,
        *,
        generate_fn: GenerateFn,
        release_session_fn: ReleaseSessionFn,
        env: TokenEnv,
        sampling: SamplingConfig,
        group_id: int,
        rollout_id: int,
    ) -> Rollout:
        """Produce a single rollout, alternating between env and generator calls,
        until the env is terminal (env `done`, truncation, errors). An exception ends it as
        `ERROR_SYSTEM`, which keeps it out of the advantage and training; past
        `rollout_timeout_s` it ends as `ERROR_TIMEOUT` and trains.

        For custom logic, users can override this method.

        Args:
            generate_fn: Async callable that runs one generation; keeps the worker
                decoupled from the generator actor.
            release_session_fn: Async callable this rollout calls once after its last turn,
                with its `routing_session_id`.
            env: The env for this rollout; `run_group` closes it.
            sampling: Sampling config for every generate call.
            group_id: The GRPO group id.
            rollout_id: Sibling index within the group; combined with the turn index into the
                per-turn `RolloutTurnID`, and stored as `Rollout.rollout_id`.

        Returns:
            One unscored `Rollout`; `run_group` fills its reward later.
        """
        # Per-sample sticky key: a sample's turns reuse one generator's prefix cache.
        routing_session_id = RolloutTurnID(
            group_id=group_id, rollout_id=rollout_id, turn_id=0
        ).to_string(include_turn=False)
        turns: list[RolloutTurn] = []
        status = RolloutStatus.ERROR
        logs: dict[str, Any] = {}
        timeout = asyncio.timeout(self._rollout_timeout_s)
        try:
            async with timeout:
                env_step = await env.init()
                while not env_step.status.is_terminal():
                    turn_rollout_id = RolloutTurnID(
                        group_id=group_id,
                        rollout_id=rollout_id,
                        turn_id=len(turns),
                    )

                    # generator call; with router replay, only the expert rows this turn adds
                    # (see `RolloutTurn.routed_expert_ids`)
                    completion = await generate_fn(
                        prompt_token_ids=env_step.next_prompt_token_ids,
                        request_id=turn_rollout_id.to_string(),
                        group_id=group_id,
                        routing_session_id=routing_session_id,
                        sampling_config=replace(
                            sampling,
                            routed_experts_prompt_start=max(
                                env_step.next_prompt_prefix_len - 1, 0
                            ),
                        ),
                    )

                    # env call
                    next_env_step = await env.step(completion)

                    # this turn from a token and message perspective
                    turns.append(
                        RolloutTurn(
                            rollout_id=turn_rollout_id,
                            prompt_prefix_len=env_step.next_prompt_prefix_len,
                            prompt_delta_token_ids=env_step.next_prompt_delta_token_ids,
                            prompt_messages=env_step.next_prompt_messages or [],
                            completion_token_ids=np.asarray(
                                completion.token_ids, dtype=np.int32
                            ),
                            completion_logprobs=np.asarray(
                                completion.token_logprobs, dtype=np.float32
                            ),
                            routed_expert_ids=completion.routed_expert_ids,
                            completion_topk_token_ids=completion.topk_token_ids,
                            completion_topk_logprobs=completion.topk_logprobs,
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
        except Exception as error:
            if timeout.expired():
                logger.warning(
                    "rollout %s/rollout=%d timed out after %d turn(s); marking ERROR_TIMEOUT",
                    group_id,
                    rollout_id,
                    len(turns),
                )
                status = RolloutStatus.ERROR_TIMEOUT
            else:
                logger.exception(
                    "rollout %s/rollout=%d failed after %d turn(s); marking ERROR_SYSTEM",
                    group_id,
                    rollout_id,
                    len(turns),
                )
                status = RolloutStatus.ERROR_SYSTEM
                logs["error"] = repr(error)
        finally:
            # Also on a cancel, so a cancelled rollout releases its session like a failed one.
            try:
                await release_session_fn(routing_session_id=routing_session_id)
            except Exception:
                # Not fatal: `max_sessions` evicts an unreleased session.
                logger.exception(
                    "releasing routing session %s failed", routing_session_id
                )
        return Rollout(
            group_id=group_id,
            rollout_id=rollout_id,
            status=status,
            turns=turns,
            logs=logs,
        )


async def gather_rollouts(
    rollouts: Iterable[Awaitable[Rollout]],
) -> tuple[list[Rollout | BaseException], list[m.Metric]]:
    """`asyncio.gather(*rollouts, return_exceptions=True)`, plus how long the group took and how
    long it waited for its slowest rollout: a group trains only once every sibling is done.

    Example (16 siblings; the first ends 120 s after the start, the last at 900 s):

        results, metrics = await gather_rollouts(coroutines)
        # metrics: rollout/group_wall_s 900, rollout/group_straggler_wait_s 780 (Mean and Max each)
    """
    start = time.perf_counter()
    finish_times: list[float] = []

    async def timed(rollout: Awaitable[Rollout]) -> Rollout:
        try:
            return await rollout
        finally:
            finish_times.append(time.perf_counter())

    results = await asyncio.gather(
        *(timed(rollout) for rollout in rollouts), return_exceptions=True
    )
    durations = {
        "rollout/group_wall_s": max(finish_times) - start,
        "rollout/group_straggler_wait_s": max(finish_times) - min(finish_times),
    }
    return results, [
        m.Metric(key, metric_type(value))
        for key, value in durations.items()
        for metric_type in (m.Mean, m.Max)
    ]


def rollouts_from_gather(
    results: list[Rollout | BaseException], *, group_id: int
) -> list[Rollout]:
    """Return a group's rollouts from `gather(..., return_exceptions=True)`. A rollout that raised
    becomes an `ERROR_SYSTEM` rollout: not the policy's doing, so it's kept for logs but not trained.
    Anything else (e.g. KeyboardInterrupt) is raised."""
    rollouts: list[Rollout] = []
    for rollout_id, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(
                "rollout group=%d/rollout=%d raised; marking ERROR_SYSTEM",
                group_id,
                rollout_id,
                exc_info=result,
            )
            result = Rollout(
                group_id=group_id,
                rollout_id=rollout_id,
                status=RolloutStatus.ERROR_SYSTEM,
                logs={"error": repr(result)},
            )
        elif isinstance(result, BaseException):
            raise result
        rollouts.append(result)
    return rollouts
