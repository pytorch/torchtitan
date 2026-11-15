# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import itertools
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any, TYPE_CHECKING

import numpy as np

from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.components.renderer import RendererConfig

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import Configurable
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.advantage import AdvantageEstimator
from torchtitan.rl.rollout.data import RLDataset, RolloutDataLoader
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

# How long `Rollouter.close` waits for in-flight `run_group` calls before stopping their workers anyway.
# Drains took under 0.3 s once the generators closed; 30 s only covers a group stuck in a slow env step.
_RUN_GROUP_DRAIN_TIMEOUT_S = 30.0

# How long `Rollouter.close` waits for the workers' own `close` calls, e.g. stopping their
# sandboxes, before it stops the worker processes anyway. Docker removes a container in seconds;
# a teardown stuck longer must not hold up the job's exit.
_WORKER_CLOSE_TIMEOUT_S = 60.0


class Rollouter(Configurable):
    """Turns a problem (train/val datasets and a rollout worker config) into scored rollouts, the
    RL training data.

    Like a `Dataloader` turns a `Dataset` into training batches, a `Rollouter`
    turns a problem into rollouts: its worker builds the envs, drives them against the inference engine
    (via a `generate_fn` the controller provides), and scores the results with `score_group`.

    The flow for one prompt group: each rollout drives its own calls to `generate_fn`, so the
    generator runs a whole group's calls together in one continuous batch.

        await rollouter.setup_async(generate_fn=generate_fn, ...)   # once
        index, sample = rollouter.get_training_sample()
        group = await rollouter.run_group_rollouts(     # one worker runs and scores the group
            sample=sample, group_id=index, group_size=N, sampling=sampling)

    Example:
        rollouter = Rollouter.Config(
            training_dataloader=IterableRolloutDataLoader.Config(
                dataset=MyDataset.Config()
            ),
            validation_dataset=MyDataset.Config(),
            worker=MessageEnvRolloutWorker.Config(
                rubric=Rubric.Config(
                    reward_fns=[RewardCorrect.Config(), RewardFormat.Config(weight=0.3)]
                ),
                message_env=MyEnv.Config(),
                token_env=TokenEnv.Config(max_rollout_tokens=32768),
            ),
        ).build()

    Customization:
        Rollouter supports customization at several levels:
          - Sample source: provide source configs for training and validation.
          - Group execution, coarse: override `run_group_rollouts` for your own
            orchestration. The worker then becomes optional -- but override
            `setup_async` too, or the worker pool is still spawned unused.
          - Group execution, fine: keep the stock orchestration and point `worker`
            at a `MessageEnvRolloutWorker.Config` subclass, overriding only what you need
            (`make_env_group`, `score_group`, `run_group`). A worker whose env isn't a
            `MessageEnv` subclasses `BaseRolloutWorker` and implements `run_group`.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        training_dataloader: RolloutDataLoader.Config
        """Builds a checkpointable iterator of identified training inputs."""

        validation_dataset: RLDataset.Config
        """Builds an iterable validation dataset."""

        worker: BaseRolloutWorker.Config
        """How a rollout group is built, driven, scored and advantaged. Selects the
        `BaseRolloutWorker` subclass by config type; one worker is built per pool process."""

        # All of a worker's rollouts share one event loop: a higher value starts fewer processes,
        # but each model call waits longer on that loop. At 64, 32 groups of 8 run on 4 workers.
        rollouts_per_worker: int = 64
        """Training rollouts in flight per rollout worker; sets the worker count
        (`Controller.Config.num_rollout_workers`). Whole groups are placed, so a worker can hold up
        to one group more."""

        # TODO: default to False once a multi-node run with spread workers matches one with the
        #   workers on the controller's host.
        workers_on_controller_host: bool = True
        """Where the rollout workers run:

            True    every worker on the controller's host
            False   the actor system spreads them over the job's other hosts; a one-node run
                    keeps them on this host

        Keep True when an env needs something only that host has, e.g. a retriever on its
        127.0.0.1, its Docker daemon, or image files on its disk (`TokenEnv` reads each image by
        path in the worker)."""

        num_threads_per_worker: int | None = 4
        """Size of each worker process's default thread pool executor, i.e. the pool
        behind every `asyncio.to_thread` call in that process. None: Python's default,
        min(32, CPUs + 4)."""

        def __post_init__(self) -> None:
            if self.rollouts_per_worker < 1:
                raise ValueError(
                    f"rollouts_per_worker must be at least 1, got {self.rollouts_per_worker}"
                )
            if (
                self.num_threads_per_worker is not None
                and self.num_threads_per_worker < 1
            ):
                raise ValueError(
                    "num_threads_per_worker must be at least 1, got "
                    f"{self.num_threads_per_worker}"
                )

    def __init__(self, config: Config) -> None:
        self._config = config
        training_dataloader = config.training_dataloader.build()
        if not isinstance(training_dataloader, RolloutDataLoader):
            raise ValueError("training_dataloader must build a RolloutDataLoader")
        self._training_dataloader = iter(training_dataloader)

        # One single-actor handle per worker process, its rollouts in flight, and the actor system
        # that spawned them.
        self._workers: list[Any] = []
        self._num_rollouts_in_flight: list[int] = []
        self._next_worker_index = 0
        self._actor_system: ActorSystem | None = None
        # `run_group` calls still running on a worker, including those whose caller was cancelled.
        self._inflight_run_group_calls: set[asyncio.Future[RolloutGroup]] = set()

    def get_training_sample(self) -> tuple[int, object]:
        """Return the next globally identified input from the training loader."""
        return next(self._training_dataloader)

    def commit_training_sample_ids(self, sample_ids: Iterable[int]) -> None:
        """Mark training samples as safe to omit from the next checkpoint."""
        self._training_dataloader.commit(sample_ids)

    def get_validation_samples(self, steps: int) -> list[object]:
        """Materialize one fresh validation pass, optionally bounded by steps."""
        validation_data = self._config.validation_dataset.build()
        try:
            iterator = iter(validation_data)
        except TypeError as error:
            raise ValueError("validation_dataset must build an iterable") from error
        if steps == -1:
            return list(iterator)
        return list(itertools.islice(iterator, steps))

    async def setup_async(
        self,
        *,
        actor_system: ActorSystem,
        num_workers: int,
        generate_fn: GenerateFn,
        release_session_fn: ReleaseSessionFn,
        tokenizer_config: HuggingFaceTokenizer.Config,
        renderer_config: RendererConfig,
        hf_assets_path: str,
        vision_processor_config: VisionProcessor.Config | None = None,
    ) -> None:
        """Spawn `num_workers` rollout workers through `actor_system` and set each one up.
        `BaseRolloutWorker.setup_async` describes the other arguments."""
        if self._workers:
            raise RuntimeError("rollout worker pool is already initialized")

        self._actor_system = actor_system
        self._workers = await actor_system.spawn_rollout_workers(
            self._config.worker,
            num_workers=num_workers,
            num_threads=self._config.num_threads_per_worker,
            on_controller_host=self._config.workers_on_controller_host,
        )
        self._num_rollouts_in_flight = [0] * num_workers
        await asyncio.gather(
            *(
                worker.setup_async.call_one(
                    generate_fn=generate_fn,
                    release_session_fn=release_session_fn,
                    tokenizer_config=tokenizer_config,
                    renderer_config=renderer_config,
                    hf_assets_path=hf_assets_path,
                    vision_processor_config=vision_processor_config,
                )
                for worker in self._workers
            )
        )

    async def close(self) -> None:
        """Wait for in-flight `run_group` calls, close each worker, then stop the worker processes.

        Close the generators first, so pending `generate_fn` calls fail fast. Then:

            1. wait for the run_group calls; cancel those still running after
               _RUN_GROUP_DRAIN_TIMEOUT_S
            2. call each worker's close(), e.g. it stops its sandboxes. A close that raises, or
               that still runs after _WORKER_CLOSE_TIMEOUT_S, is logged
            3. stop_rollout_workers(), even after a failed close

        Why wait: if a worker stops while a generator (or the generator router) still owes it a
        reply, Monarch fails that actor and the finished job exits 130. Why close first: Monarch
        cancels an actor's `__cleanup__` after 3 s (its cleanup timeout) and fails the job.
        """
        workers, actor_system = self._workers, self._actor_system
        self._workers = []
        self._actor_system = None
        calls = tuple(self._inflight_run_group_calls)
        if calls:
            _, pending = await asyncio.wait(calls, timeout=_RUN_GROUP_DRAIN_TIMEOUT_S)
            if pending:
                logger.warning(
                    "Stopping rollout workers with %d run_group calls still running; "
                    "the job may exit 130",
                    len(pending),
                )
                for call in pending:
                    call.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
        try:
            # Like the drain above: a close that hangs must not hide another worker's failed close.
            close_calls = [
                asyncio.ensure_future(worker.close.call_one()) for worker in workers
            ]
            if close_calls:
                await asyncio.wait(close_calls, timeout=_WORKER_CLOSE_TIMEOUT_S)
            for worker_index, call in enumerate(close_calls):
                if not call.done():
                    logger.error(
                        "Rollout worker %d close still running after %.0f s; stopping it",
                        worker_index,
                        _WORKER_CLOSE_TIMEOUT_S,
                    )
                    call.cancel()
                elif call.exception() is not None:
                    logger.error(
                        "Rollout worker %d close failed",
                        worker_index,
                        exc_info=call.exception(),
                    )
        finally:
            if actor_system is not None:
                await actor_system.stop_rollout_workers()

    async def sync_log_step(self, step: int) -> None:
        """Propagate the controller log step to every rollout worker."""
        await asyncio.gather(
            *(worker.sync_log_step.call_one(step) for worker in self._workers)
        )

    async def process_metrics(self) -> list[m.Metric]:
        """Every rollout worker's CPU since the previous call and its memory, agent subprocesses
        included; the metrics processor reduces them over workers."""
        # TODO: also each host's sum (and its max over hosts), to check a host's CPU and memory
        #   budget; these sums are job-wide.
        replies = await asyncio.gather(
            *(worker.process_metrics.call_one() for worker in self._workers)
        )
        return [metric for reply in replies for metric in reply]

    async def run_group_rollouts(
        self,
        *,
        sample: object,
        group_id: int,
        group_size: int,
        sampling: SamplingConfig,
    ) -> RolloutGroup:
        """Send one prompt group to the rollout worker with the fewest rollouts in flight, and
        return its scored group.

        Args:
            sample: Dataset sample shared by the group.
            group_id: Stable group id; siblings share it for advantage centering.
            group_size: Number of sibling rollouts.
            sampling: Sampling config for every generate call in the group.

        Returns:
            One scored `RolloutGroup`.
        """
        if not self._workers:
            raise RuntimeError("rollout worker pool is not initialized")

        # Fewest rollouts in flight. A tie goes to the next worker after the last one picked, so
        # tied workers take turns.
        # TODO: lower a worker's count as each rollout ends. Today the group's whole count drops
        #   when the group returns, so a group waiting for one slow rollout still counts all of
        #   them. Placing single rollouts (the TODO in `Controller.Config`) fixes this.
        # TODO: before groups of tens of GB (1M-token rollouts, 128 per group), reserve each group's
        #   bytes per worker host at admission, or pick by bytes with a fail-fast guard.
        num_workers = len(self._workers)
        worker_index = min(
            range(num_workers),
            key=lambda index: (
                self._num_rollouts_in_flight[index],
                (index - self._next_worker_index) % num_workers,
            ),
        )
        self._next_worker_index = (worker_index + 1) % num_workers
        call = asyncio.ensure_future(
            self._workers[worker_index].run_group.call_one(
                sample=sample,
                group_id=group_id,
                group_size=group_size,
                sampling=sampling,
            )
        )

        def on_done(call: asyncio.Future[RolloutGroup]) -> None:
            self._num_rollouts_in_flight[worker_index] -= group_size
            self._inflight_run_group_calls.discard(call)
            # A cancelled caller never reads the error; read it so asyncio does not log it.
            if not call.cancelled():
                call.exception()

        self._inflight_run_group_calls.add(call)
        call.add_done_callback(on_done)
        # Counted once the call is sent: `call_one` raises before it returns a Future when it can't
        # send, e.g. an argument that doesn't pickle. asyncio schedules done callbacks, so `on_done`
        # can't run before this line.
        self._num_rollouts_in_flight[worker_index] += group_size
        # Shield `call` from the controller's end-of-training cancel, so `close` can still wait for it.
        return await asyncio.shield(call)

    def state_dict(self) -> dict[str, object]:
        """Return the checkpoint state of the training input iterator."""
        return self._training_dataloader.state_dict()

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore the training iterator, including uncommitted inputs."""
        self._training_dataloader.load_state_dict(state_dict)


class BaseRolloutWorker(Configurable, ABC):
    """Runs one rollout group in a rollout worker process, then scores it and computes its
    advantages. Subclasses implement `run_group`."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        rubric: Rubric.Config
        """Reward functions + weights used by `score_group`."""

        advantage: Configurable.Config = field(
            default_factory=AdvantageEstimator.Config
        )
        """Post-scoring advantage estimator. Default = Dr.GRPO (mean-baseline only);
        set `AdvantageEstimator.Config(should_std_normalize=True)` for standard GRPO."""

    def __init__(self, config: Config) -> None:
        self.rubric: Rubric = config.rubric.build()
        self.advantage_estimator: AdvantageEstimator = config.advantage.build()

    @abstractmethod
    async def setup_async(
        self,
        *,
        generate_fn: GenerateFn,
        release_session_fn: ReleaseSessionFn,
        tokenizer_config: HuggingFaceTokenizer.Config,
        renderer_config: RendererConfig,
        hf_assets_path: str,
        vision_processor_config: VisionProcessor.Config | None = None,
    ) -> None:
        """Called once in the worker process after it is spawned.

        Args:
            generate_fn: Async callable that returns a Completion given a prompt.
            release_session_fn: Async callable each rollout calls after its last turn, to
                release resources associated to that session. E.g. generator's kv-cache.
            tokenizer_config: The model's tokenizer.
            renderer_config: Turns messages into prompt tokens and completions into messages.
            hf_assets_path: The model's Hugging Face folder, for the tokenizer files.
            vision_processor_config: `Controller.Config.vision_processor`; None for text prompts.
        """

    async def close(self) -> None:
        """Free what `setup_async` started, e.g. servers and sandboxes. Free only what is still
        up, and return at once when nothing is, because a close can also come:

            twice                   the Rollouter's close, then the actor's `__cleanup__`
            before setup ends       `setup_async` never ran, raised, or is still running
            during another close    `__cleanup__` after the Rollouter's close timed out
        """

    async def score_group(
        self,
        rollouts: list[Rollout],
        env_input: object,
    ) -> list[RubricOutput]:
        """Score one group's rollouts; `make_scored_group` applies the rewards.
        Default impl delegates to `self.rubric.score_group`.

        Args:
            rollouts: Sibling rollouts in one prompt group, already stepped. Only `is_trainable()`
                rollouts are passed (see `make_scored_group`).
            env_input: the env initial input shared by the group.

        Returns:
            One `RubricOutput` per rollout, in input order.
        """
        return await self.rubric.score_group(rollouts, env_input)

    @abstractmethod
    async def run_group(
        self,
        *,
        sample: object,
        group_id: int,
        group_size: int,
        sampling: SamplingConfig,
    ) -> RolloutGroup:
        """Roll out and score one prompt group. An implementation gathers its rollouts with
        `return_exceptions=True` and passes them to `make_scored_group`.

        Args:
            sample: Dataset sample shared by the group.
            group_id: Stable group id; siblings share it for advantage centering.
            group_size: Number of sibling rollouts.
            sampling: Sampling config for every generate call in the group.

        Returns:
            One scored `RolloutGroup`.
        """

    async def make_scored_group(
        self,
        rollouts_or_errors: list[Rollout | BaseException],
        sample: object,
        *,
        group_id: int,
    ) -> RolloutGroup:
        """Score one group and fill its advantages. A sibling that raised becomes ERROR_INFRA.

        Example, group 7, rewards from `score_group`:

            rollouts_or_errors  [Rollout(0, COMPLETED), ConnectionError(), Rollout(2, ERROR)]
            rollouts            [Rollout(0),            Rollout(1, ERROR_INFRA), Rollout(2)]
            rewards             [1.0,                   None,                    0.0]
            advantages          [0.5,                   None,                    -0.5]

        A rollout that isn't `is_trainable()`, or that `score_group` gave reward None, gets no
        advantage and isn't trained. A `CancelledError` in `rollouts_or_errors` is raised.
        """
        rollouts: list[Rollout] = []
        for rollout_id, rollout_or_error in enumerate(rollouts_or_errors):
            if isinstance(rollout_or_error, Exception):
                logger.error(
                    "rollout group=%d/rollout=%d raised; marking ERROR_INFRA",
                    group_id,
                    rollout_id,
                    exc_info=rollout_or_error,
                )
                rollouts.append(
                    Rollout(
                        group_id=group_id,
                        rollout_id=rollout_id,
                        status=RolloutStatus.ERROR_INFRA,
                        logs={"error": repr(rollout_or_error)},
                    )
                )
            elif isinstance(rollout_or_error, BaseException):
                raise rollout_or_error
            else:
                rollouts.append(rollout_or_error)

        # A group with no trainable rollout has nothing to score or center.
        rollouts_to_score = [
            rollout for rollout in rollouts if rollout.status.is_trainable()
        ]
        if rollouts_to_score:
            rubric_outputs = await self.score_group(rollouts_to_score, sample)
            for rollout, rubric_output in zip(
                rollouts_to_score, rubric_outputs, strict=True
            ):
                rollout.reward = rubric_output.reward
                rollout.reward_breakdown = rubric_output.reward_breakdown
        # TODO: run an ordered list of group postprocessors here, e.g. drop a group whose rewards all
        # match, then compute advantages. The advantage estimator becomes one of them, and so can the
        # group filters in `TrainingSampleBuilder.build_from_group`.
        rollouts_to_center = [
            rollout for rollout in rollouts_to_score if rollout.reward is not None
        ]
        if rollouts_to_center:
            advantages = self.advantage_estimator(
                RolloutGroup(group_id=group_id, rollouts=rollouts_to_center)
            )
            for rollout, advantage in zip(rollouts_to_center, advantages, strict=True):
                rollout.advantage = advantage
        return RolloutGroup(group_id=group_id, rollouts=rollouts)


class MessageEnvRolloutWorker(BaseRolloutWorker):
    """Builds each rollout's env from `message_env` and `token_env`, and drives it against the
    generator."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseRolloutWorker.Config):
        message_env: MessageEnv.Config
        """The env to build per sample; `make_env_group` calls `build(env_input=sample)`."""

        token_env: TokenEnv.Config
        """`TokenEnv` wraps the `MessageEnv` in `make_env_group`."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._message_env_config = config.message_env
        self._token_env_config = config.token_env
        self._renderer: Renderer
        self._generate_fn: GenerateFn
        self._release_session_fn: ReleaseSessionFn

    async def setup_async(
        self,
        *,
        generate_fn: GenerateFn,
        release_session_fn: ReleaseSessionFn,
        tokenizer_config: HuggingFaceTokenizer.Config,
        renderer_config: RendererConfig,
        hf_assets_path: str,
        vision_processor_config: VisionProcessor.Config | None = None,
    ) -> None:
        """Store `generate_fn` and `release_session_fn`, and build the renderer that each
        rollout's `TokenEnv` uses."""
        self._generate_fn = generate_fn
        self._release_session_fn = release_session_fn
        tokenizer = tokenizer_config.build(tokenizer_path=hf_assets_path)
        vision_processor = (
            vision_processor_config.build()
            if vision_processor_config is not None
            else None
        )
        self._renderer = renderer_config.build(
            tokenizer=tokenizer, vision_processor=vision_processor
        )

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

    async def run_group(
        self,
        *,
        sample: object,
        group_id: int,
        group_size: int,
        sampling: SamplingConfig,
    ) -> RolloutGroup:
        """Build `group_size` envs from `sample`, run them concurrently, then score them and fill their advantages."""
        # One prompt becomes [env] * group_size.
        envs = self.make_env_group(
            sample=sample,
            group_size=group_size,
        )

        # TODO(perf): siblings in a group share the first-turn prompt; tokenize it once per group and
        # reuse across the group_size rollouts (truest spot is the worker's first-turn render).
        try:
            # return_exceptions: every sibling ends before the controller frees the group's KV
            # cache. A sibling still generating would hold it again, and nothing would free it.
            rollouts_or_errors = await asyncio.gather(
                *(
                    self._run_single_rollout(
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
                ),
                return_exceptions=True,
            )
        finally:
            # close the envs
            await asyncio.gather(*(env.close() for env in envs), return_exceptions=True)
        return await self.make_scored_group(
            rollouts_or_errors, sample, group_id=group_id
        )

    async def _run_single_rollout(
        self,
        *,
        env: TokenEnv,
        sampling: SamplingConfig,
        group_id: int,
        rollout_id: int,
    ) -> Rollout:
        """Produce a single rollout, alternating between env and generator calls,
        until the env is terminal (env `done`, truncation, errors). An exception ends it as
        `ERROR_INFRA`.

        For custom logic, users can override this method.

        Args:
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
        logs: dict[str, Any] = {}
        try:
            env_step = await env.init()
            while not env_step.status.is_terminal():
                turn_rollout_id = RolloutTurnID(
                    group_id=group_id,
                    rollout_id=rollout_id,
                    turn_id=len(turns),
                )

                # generator call
                completion = await self._generate_fn(
                    prompt=env_step.next_prompt,
                    request_id=turn_rollout_id.to_string(),
                    group_id=group_id,
                    routing_session_id=routing_session_id,
                    sampling_config=sampling,
                )

                # env call
                next_env_step = await env.step(completion)

                # this turn from a token and message perspective
                turns.append(
                    RolloutTurn(
                        rollout_id=turn_rollout_id,
                        prompt_prefix_len=env_step.next_prompt_prefix_len,
                        prompt_delta_token_ids=env_step.next_prompt_delta_token_ids,
                        prompt_delta_images=env_step.next_prompt_delta_images,
                        prompt_delta_image_placeholders=(
                            env_step.next_prompt_delta_image_placeholders
                        ),
                        prompt_messages=env_step.next_prompt_messages or [],
                        completion_token_ids=np.asarray(
                            completion.token_ids, dtype=np.int32
                        ),
                        completion_logprobs=np.asarray(
                            completion.token_logprobs, dtype=np.float32
                        ),
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
            # An exception is read as a failure outside the policy: `TokenEnv` turns the policy's
            # failures into statuses (e.g. ERROR_PARSE), and an env answers bad output with a reply
            # (`MessageEnv.step`).
            # TODO: at shutdown, every rollout still generating lands here with a ~40-line remote
            # traceback. Raise a typed error on generator close and log it at info, without the traceback.
            logger.exception(
                "rollout %s/rollout=%d failed after %d turn(s); marking ERROR_INFRA",
                group_id,
                rollout_id,
                len(turns),
            )
            status = RolloutStatus.ERROR_INFRA
            logs["error"] = repr(error)

        await self._release_session_fn(routing_session_id=routing_session_id)
        return Rollout(
            group_id=group_id,
            rollout_id=rollout_id,
            status=status,
            turns=turns,
            logs=logs,
        )
