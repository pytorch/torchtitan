# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from torchtitan.config import Configurable
from torchtitan.observability import structured_logger as sl
from torchtitan.rl.controller.metrics import compute_rollout_metrics
from torchtitan.rl.observability import metrics as m

if TYPE_CHECKING:
    from torchtitan.rl.generator.types import SamplingConfig
    from torchtitan.rl.rollout.recorder import RolloutSampleRecorder
    from torchtitan.rl.rollout.rollouter import Rollouter
    from torchtitan.rl.rollout.types import GenerateFn, ReleaseSessionFn
    from torchtitan.rl.routing.inter_generator import InterGeneratorRouter

logger = logging.getLogger(__name__)

# `next(prompts, _END_OF_PASS)` returns it once the validation pass has no prompts left.
_END_OF_PASS = object()


class Evaluator(Configurable, ABC):
    """Scores the policy while training runs; the controller calls `on_step` every step and `close` once.

    To replace the default, subclass this and put its `Config` in `Controller.Config.evaluator`.
    The controller builds it with keyword arguments (see `RolloutEvaluator.__init__`): take the
    ones you use and `**kwargs` for the rest. A `Config` field must not have the name of one of
    them: `build` raises when a keyword argument and a field share a name.

        evaluator.on_step(0)                               start step (the resumed step on a resume)
        evaluator.on_step(1), ..., evaluator.on_step(N)    after each optimizer step; never waits
        await evaluator.close(training_completed=True)     awaits or cancels the work it still owes
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    @abstractmethod
    def on_step(self, step: int) -> None:
        """Policy `step` is trained: called with the start step, then after every optimizer step. Never waits.

        Its push and pull run in the background, so generators and TorchStore may still hold `step - 1`.
        """

    @abstractmethod
    async def close(self, *, training_completed: bool) -> None:
        """Called once, at the end of `Controller.run`.

        training_completed=False (a crash): cancel running work only.
        training_completed=True: may run final work, e.g. a pass on the final policy.
        """


class RolloutEvaluator(Evaluator):
    """Rolls out the rollouter's validation prompts on the training generators, beside training.

    One pass runs at a time. A pass that comes due meanwhile waits; a newer due pass replaces it.
    A pass logs at the last step `on_step` saw when it ended. So a pass that ends between the last
    `on_step` and `close()` also logs at N, next to the final pass; the final one has
    `validation/policy_version/min == N`.

    Example: freq=10, 40 steps, each pass takes ~13 steps.
        due 0:   starts at step 0, logs at 12
        due 10:  waits, starts at 13, logs at 25
        due 20:  waits, starts at 26, logs at 38
        due 30:  waits, starts at 39, cancelled by close()
        final:   runs on policy 40, logs at 40
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Evaluator.Config):
        # TODO: per-pass sampling settings, e.g. a longer max_tokens. A pass reuses training's
        # sampling at temperature 0.

        freq: int | None = None
        """A pass is due at each multiple of `freq` between 0 and the last step, both excluded. It waits
        while another pass runs, and a newer due step replaces it. None: no periodic passes."""

        validate_at_start: bool = True
        """Start a pass at step 0. It runs beside training, so later prompts may sample policy 1, 2, ...;
        `validation/policy_version/max` says which. Turn it off when you already know that score.
        A resumed run skips it and follows the `freq` grid."""

        validate_after_training: bool = True
        """After a completed run, one pass on the final policy. The job waits for it; the trainer GPUs idle."""

        max_prompts_in_flight: int = 64
        """Prompts rolled out at once. They run before training requests (vLLM priority -1), so this
        caps the generator slots a pass takes: 64 is 1/8 of one engine's default `max_num_seqs` (512).
        Lower it if training slows down while a pass runs."""

        def __post_init__(self) -> None:
            if self.freq is not None and self.freq < 1:
                raise ValueError(f"freq must be None or >= 1, got {self.freq}")
            if self.max_prompts_in_flight < 1:
                raise ValueError(
                    f"max_prompts_in_flight must be >= 1, got {self.max_prompts_in_flight}"
                )

    def __init__(
        self,
        config: Config,
        *,
        num_training_steps: int,
        rollouter: Rollouter,
        generate_fn: GenerateFn,
        release_session_fn: ReleaseSessionFn,
        generator_router: InterGeneratorRouter,
        sampling: SamplingConfig,
        rollout_recorder: RolloutSampleRecorder,
        metrics_processor: m.MetricsProcessor,
    ) -> None:
        self.config = config
        self._num_training_steps = num_training_steps
        self._rollouter = rollouter
        self._generate_fn = generate_fn
        self._release_session_fn = release_session_fn
        self._generator_router = generator_router
        # Training's max_tokens, stop tokens and seed, sampled greedily.
        self._sampling = replace(sampling, temperature=0.0)
        self._rollout_recorder = rollout_recorder
        self._metrics_processor = metrics_processor
        # The last step `on_step` saw; a pass logs its metrics at it.
        self._step = 0
        self._pass_task: asyncio.Task[None] | None = None
        # The newest due step whose pass waits for the running one.
        self._waiting_due_step: int | None = None
        # Validation group ids count down for the whole run and are never reused, so the late
        # calls of a cancelled pass never share an id with a later pass.
        self._group_ids = itertools.count(-1, -1)

    def on_step(self, step: int) -> None:
        self._step = step
        config = self.config
        is_due = (step == 0 and config.validate_at_start) or (
            config.freq is not None
            and 0 < step < self._num_training_steps
            and step % config.freq == 0
        )
        if is_due:
            if self._waiting_due_step is not None:
                logger.info(
                    "Validation pass due at step %d replaced by the one due at step %d",
                    self._waiting_due_step,
                    step,
                )
            self._waiting_due_step = step

        # The last step never starts a pass: only the final pass, if asked, starts on policy N.
        is_idle = self._pass_task is None or self._pass_task.done()
        if (
            self._waiting_due_step is not None
            and is_idle
            and step < self._num_training_steps
        ):
            self._pass_task = asyncio.create_task(
                self._run_pass(due_step=self._waiting_due_step),
                name=f"validation pass due at step {self._waiting_due_step}",
            )
            self._pass_task.add_done_callback(self._log_failed_pass)
            self._waiting_due_step = None

    async def close(self, *, training_completed: bool) -> None:
        if self._waiting_due_step is not None:
            logger.warning(
                "Dropping the validation pass due at step %d: training ended first",
                self._waiting_due_step,
            )
            self._waiting_due_step = None
        if self._pass_task is not None:
            if self._pass_task.cancel():
                logger.warning(
                    "Cancelled the %s: training ended", self._pass_task.get_name()
                )
            # A bare `await` re-raises the cancelled pass's CancelledError: it would replace
            # the training error being raised, or end a healthy run before the final pass.
            await asyncio.gather(self._pass_task, return_exceptions=True)

        if training_completed and self.config.validate_after_training:
            # TODO: abort the training requests still in the generators before this pass. They
            # keep decoding after training ends and slow this pass down several times. The
            # generators have no abort path yet.
            await self._run_pass(due_step=self._step)

    @sl.log_trace_span("validate")
    async def _run_pass(self, *, due_step: int) -> None:
        """Roll out one validation pass, `max_prompts_in_flight` prompts at a time, then log its metrics."""
        logger.info(
            "Validation pass due at step %d starts at step %d", due_step, self._step
        )
        pass_start_s = time.perf_counter()
        # Building the dataset can take seconds; keep it off the event loop.
        prompts = await asyncio.to_thread(self._rollouter.iter_validation_samples)
        free_slots = asyncio.Semaphore(self.config.max_prompts_in_flight)
        pass_metrics: list[m.Metric] = []

        async def run_group_and_free_slot(sample: object) -> None:
            try:
                group_metrics = await self._run_group(sample)
                # No await between this read and write, so no other group's metrics get lost.
                pass_metrics[:] = m.fold([*pass_metrics, *group_metrics])
            finally:
                free_slots.release()

        try:
            async with asyncio.TaskGroup() as group_tasks:
                while True:
                    # Read the next prompt only once a slot frees: at most max_prompts_in_flight in memory.
                    await free_slots.acquire()
                    # TODO: a dataset that decodes images does it here, in the controller (training too).
                    sample = await asyncio.to_thread(next, prompts, _END_OF_PASS)
                    if sample is _END_OF_PASS:
                        break
                    group_tasks.create_task(run_group_and_free_slot(sample))
        except ExceptionGroup as error_group:
            # Raise the error itself, e.g. the dataset's "ended after 3 prompts", not the group.
            raise error_group.exceptions[0] from error_group

        pass_metrics.append(
            m.Metric("timing/validate", m.NoReduce(time.perf_counter() - pass_start_s))
        )
        self._metrics_processor.log(
            step=self._step, metrics=pass_metrics, is_validation=True
        )

    async def _run_group(self, sample: object) -> list[m.Metric]:
        """Roll out one prompt greedily, record its group and return its metrics; a failure counts in `group_failures`."""
        # TODO: a group that never returns stalls this pass and every later one; add a timeout.
        group_id = next(self._group_ids)
        # TODO: n samples per prompt (group_size=n), with mean@n and pass@k.
        # TODO(naming): reserve "sample" for TrainingSample; rename the rollouter's raw-prompt
        # "sample" -> "prompt"/"data_input".
        try:
            group = await self._rollouter.run_group_rollouts(
                generate_fn=self._generate_fn,
                release_session_fn=self._release_session_fn,
                sample=sample,
                group_id=group_id,
                group_size=1,
                sampling=self._sampling,
            )
        except Exception:
            logger.exception("Validation group %d failed; dropping it", group_id)
            group = None
        # Not in a `finally`: a cancelled caller's worker call is shielded and may still be
        # generating; releasing now would let its later turns pin the group again for good.
        # TODO: release a cancelled group once its worker call ends. A cancelled pass, or one whose
        # dataset ends early, leaves up to max_prompts_in_flight groups pinned in the generators.
        await self._generator_router.release_groups.call_one([group_id])
        if group is None:
            return [m.Metric("validation/group_failures", m.Sum(1.0))]

        self._rollout_recorder.record(is_validation=True, rollout_groups=[group])
        rollout_turns = [
            rollout_turn for rollout in group.rollouts for rollout_turn in rollout.turns
        ]
        min_policy_versions = [
            rollout_turn.min_policy_version
            for rollout_turn in rollout_turns
            if rollout_turn.min_policy_version is not None
        ]
        max_policy_versions = [
            rollout_turn.max_policy_version
            for rollout_turn in rollout_turns
            if rollout_turn.max_policy_version is not None
        ]
        # Prefix a rubric's turn metrics (e.g. math_verify/timeouts): training logs the same keys.
        group_metrics = [
            (
                metric
                if metric.key.startswith("validation")
                else m.Metric(f"validation/{metric.key}", metric.value)
            )
            for metric in compute_rollout_metrics(
                prefix="validation", rollouts=group.rollouts
            )
        ]
        group_metrics.extend(
            [
                m.Metric(
                    "validation/policy_version", m.Min.from_list(min_policy_versions)
                ),
                m.Metric(
                    "validation/policy_version", m.Max.from_list(max_policy_versions)
                ),
                m.Metric("validation/group_failures", m.Sum(0.0)),
            ]
        )
        return group_metrics

    def _log_failed_pass(self, pass_task: asyncio.Task[None]) -> None:
        """Log a failed periodic pass at the last step seen; training goes on and the next due pass retries."""
        if pass_task.cancelled() or pass_task.exception() is None:
            return
        logger.error(
            "The %s failed; training goes on",
            pass_task.get_name(),
            exc_info=pass_task.exception(),
        )
        self._metrics_processor.log(
            step=self._step,
            metrics=[m.Metric("validation/failed_passes", m.Sum(1.0))],
            is_validation=True,
        )
