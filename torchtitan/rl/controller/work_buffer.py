# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Active work buffer shared between the data-input, rollout, and batcher loops.
NOTE: The buffer holds work slots, and not the finalized RolloutGroups necessarily.

Two buffers share one interface:

    RolloutGroupWorkBuffer          fixed `(target_offpolicy_steps + 1) * P` slots; optional FIFO window; never drops
    AdaptiveRolloutGroupWorkBuffer  slots from a quantile of the groups not ready at recent step starts, capped so
                                    the mean age stays under its `target_offpolicy_steps` (or `max_offpolicy_steps`);
                                    takes the oldest finalized group; drops groups past `max_offpolicy_steps`,
                                    cancelling the ones still generating
"""

import asyncio
import collections
import enum
import logging
from dataclasses import dataclass, field

from torchtitan.config import Configurable
from torchtitan.observability import structured_logger as sl
from torchtitan.rl.controller.adaptive_demand import StallDrivenDemand
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout import RolloutGroup

logger = logging.getLogger(__name__)

# Sequences one generator DP shard runs at once when `generation_capacity` falls back to filling vLLM;
# matches the controller's `max_num_seqs` cap.
DEFAULT_SEQS_PER_GENERATOR_DP_SHARD = 512


class _RolloutGroupWorkState(enum.Enum):
    """Where a RolloutGroupWork is in the WAITING -> INFLIGHT -> FINALIZED lifecycle."""

    WAITING = "waiting"
    INFLIGHT = "inflight"
    FINALIZED = "finalized"


@dataclass(slots=True)
class RolloutGroupWork:
    """One prompt group's work, tracked through _RolloutGroupWorkState.

    The input loop sets `group_id` + `sample`; the buffer owns every other field
    (`init=False`, so the input loop can't set them).
    """

    group_id: int
    sample: object
    """Indexed data input produced by the rollouter;
    passed unchanged to the env in `rollouter.run_group_rollouts`."""
    state: _RolloutGroupWorkState = field(
        default=_RolloutGroupWorkState.WAITING, init=False
    )
    rollout_group: RolloutGroup | None = field(
        default=None, init=False
    )  # set once FINALIZED
    policy_version_at_claim: int | None = field(default=None, init=False)
    """Generator policy version when generation started; only the adaptive buffer sets it, for its age drops."""
    generation_task: asyncio.Task | None = field(default=None, init=False)
    """Task generating this group, set by `track_generation`, so a buffer can cancel it."""
    cancelled: bool = field(default=False, init=False)
    """Set when the buffer cancelled this group's generation (the adaptive buffer, once the group is too old to
    ever train); its slot stays charged until the generation task ends, then the rollout loop releases it."""
    # TODO(async-rl): emit JSON lifecycle logging per RolloutGroupWork keyed by group_id:
    # admitted/claimed/finalized/batched/trained/dropped timestamps + policy version at admission and
    # at trainer consumption, for faithful end-to-end visibility.


class RolloutGroupWorkBuffer(Configurable):
    """Buffer of `RolloutGroupWork` shared between the data-input, rollout, and batcher loops.

    Each entry is a RolloutGroupWork moving WAITING -> INFLIGHT -> FINALIZED. An active-slot budget caps
    the pipeline at `max_active_rollout_groups` active slots. With `window_size=None`, the batcher takes
    the oldest finalized group anywhere in the buffer. A finite `window_size` restricts it to a look-ahead
    range anchored at the oldest entry.

    For details on the buffer's callers, check the diagram in the controller.py file.

    NOTE: a work slot is **NOT** released when marked as FINALIZED or taken by the batcher.
    Instead, it is only released on `release_active_groups` calls by the trainer
    or data filtering. This is done this way so that we can guarantee we never have more
    than `max_active_rollout_groups` in the entire pipeline (buffer+queue+training).
    Otherwise, we would produce born-stale examples.

    Entry lifecycle vs active slot:
        entry:        WAITING -> INFLIGHT -> FINALIZED -> removed by take_finalized()
        active slot:  charged by add_work() ............ freed by release_active_groups()

    Example:
        # target_offpolicy_steps=1, num_prompts_per_train_step=2 -> capacity=4
        await buffer.add_work(g0); await buffer.add_work(g1)   # 2/4 active
        await buffer.add_work(g2); await buffer.add_work(g3)   # 4/4 active (cap)
        g0 = await buffer.take_finalized()                     # g0 leaves the dict; still 4/4 active
        g1 = await buffer.take_finalized()                     # g1 leaves the dict; still 4/4 active
        slot_task = asyncio.create_task(buffer.wait_for_slot())  # waits: take_finalized did not free a slot
        assert not slot_task.done()
        await buffer.release_active_groups(2, reason="trained")  # trainer pulled -> a slot frees
        assert await slot_task                                   # wait_for_slot now returns
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """No tunables: capacity and window size are derived by the controller."""

    def __init__(
        self, config: Config, *, max_active_rollout_groups: int, window_size: int | None
    ) -> None:
        self._max_active_rollout_groups = max_active_rollout_groups
        self._window_size = window_size
        self._active_rollout_groups = 0
        # metric: Per-flush peak active slots; reset on `.metrics()` call.
        self._active_rollout_groups_peak_since_flush = 0
        self._work_by_group_id: collections.OrderedDict[
            int, RolloutGroupWork
        ] = collections.OrderedDict()
        # TODO(async-rl): Current we use a condition that alerts ALL rollout workers. There is no need to
        # alert all of them. Consider changing it to an async queue + event.

        # One Condition guards all three waits (slot-free / claimable-WAITING / takeable-FINALIZED):
        # every mutation notify_all()s and waiters re-check their predicate.
        self._condition = asyncio.Condition()
        self._closed = False
        # TODO(async-rl): warm start — admit a small number of groups at first and grow the effective cap as the
        # batcher consumes, so a cold start doesn't fill the whole off-policy window at policy version 0.

    def _has_active_slot_available(self) -> bool:
        return self._active_rollout_groups < self._max_active_rollout_groups

    async def wait_for_slot(self) -> bool:
        """Wait until one more rollout group may enter the active buffer.

        Example:
            # False means the buffer was closed, so the data input loop exits.
            group_index = 0
            while await buffer.wait_for_slot():
                await buffer.add_work(RolloutGroupWork(group_id=group_index, sample=sample))
                group_index += 1
        """
        async with self._condition:
            await self._condition.wait_for(
                lambda: self._closed or self._has_active_slot_available()
            )
            return not self._closed

    async def add_work(self, work: RolloutGroupWork) -> None:
        """Admit one rollout group as WAITING and charge one active slot."""
        async with self._condition:
            if not self._has_active_slot_available():
                raise RuntimeError(
                    "RolloutGroupWorkBuffer.add_work called without an active slot"
                )
            self._active_rollout_groups += 1
            self._active_rollout_groups_peak_since_flush = max(
                self._active_rollout_groups_peak_since_flush,
                self._active_rollout_groups,
            )
            self._work_by_group_id[work.group_id] = work
            self._condition.notify_all()

    async def claim_next(self) -> RolloutGroupWork | None:
        """Rollout loop: claim the oldest WAITING group (WAITING -> INFLIGHT). None once closed."""
        async with self._condition:
            while True:
                if self._closed:
                    return None
                for work in self._work_by_group_id.values():
                    if work.state is _RolloutGroupWorkState.WAITING:
                        work.state = _RolloutGroupWorkState.INFLIGHT
                        return work
                await self._condition.wait()

    def track_generation(self, work: RolloutGroupWork, task: asyncio.Task) -> None:
        """Rollout loop: record the task generating a claimed group, so the buffer can cancel it.

        This buffer never cancels; the adaptive buffer cancels groups that became too old to train while generating.
        """
        work.generation_task = task

    async def finalize_work(self, rollout_group: RolloutGroup) -> bool:
        """Rollout loop: store the produced RolloutGroup on its work entry (INFLIGHT -> FINALIZED) and wake the batcher.

        Returns False when the result is not stored: the buffer was closed, or it cancelled the group (whose slot
        is released here instead).
        """
        async with self._condition:
            work = self._work_by_group_id.get(rollout_group.group_id)
            if work is None:
                # run()'s shutdown called close(); drop the result.
                return False
            if work.cancelled:
                # Cancelled after its last rollout ended (cancel was a no-op): drop the result.
                self._release_cancelled(work)
                return False
            work.rollout_group = rollout_group
            work.state = _RolloutGroupWorkState.FINALIZED
            self._condition.notify_all()
            return True

    async def release_cancelled(self, group_id: int) -> None:
        """Rollout loop: the buffer cancelled this group and its generation task has ended; free its slot.

        Example:
            # the adaptive buffer cancelled g0 (too old); g0's rollout loop sees CancelledError
            await buffer.release_cancelled(0)   # g0 leaves the buffer; its slot is free
        """
        async with self._condition:
            work = self._work_by_group_id.get(group_id)
            if work is None:
                # run()'s shutdown called close().
                return
            assert work.cancelled, f"group {group_id} was not cancelled by the buffer"
            self._release_cancelled(work)

    def _release_cancelled(self, work: RolloutGroupWork) -> None:
        """Remove a cancelled group and free its slot. Caller holds the condition."""
        del self._work_by_group_id[work.group_id]
        self._active_rollout_groups -= 1
        sl.log_trace_scalar({"rollout_buffer/released/cancelled_too_old": 1.0})
        self._condition.notify_all()

    @sl.log_trace_span("take_finalized")
    async def take_finalized(
        self, *, consuming_policy_version: int | None = None
    ) -> RolloutGroup | None:
        """Batcher loop: return the oldest FINALIZED group the window allows.

        `consuming_policy_version` is for the adaptive buffer's age drops; this buffer drops nothing.

        Cases:
            window_size is None: every finalized group is eligible; the oldest is returned.
            window_size is W:    only group ids ``[head, head + W - 1]`` are eligible, where head is
                                 the oldest group still in the buffer. Finalized groups past the
                                 window stay blocked, and taking a non-head group does not move the
                                 window past the head.
            window_size is 1:    only the head is eligible: strict FIFO.

        This anchored window is inspired by MiniMax's rollout scheduling; see
        Section 6.2.4 of https://arxiv.org/pdf/2605.26494.

        Example:
            # window_size=3: g0 INFLIGHT, g1 WAITING, g2 and g3 FINALIZED
            group = await buffer.take_finalized()
            assert group.group_id == 2  # g2 is inside [g0, g2].
            # g3 remains blocked. With window_size=None, g3 would be taken next.
        """
        async with self._condition:
            while True:
                if self._closed:
                    return None
                if self._work_by_group_id:
                    head_group_id = next(iter(self._work_by_group_id))
                    for group_id, work in self._work_by_group_id.items():
                        if (
                            self._window_size is not None
                            and group_id >= head_group_id + self._window_size
                        ):
                            break
                        if work.state is not _RolloutGroupWorkState.FINALIZED:
                            continue
                        del self._work_by_group_id[group_id]
                        self._condition.notify_all()
                        return work.rollout_group
                await self._condition.wait()  # nothing finalized inside the window -> stall

    async def record_step_start(self, *, trainer_policy_version: int) -> None:
        """Trainer loop: called right before it waits for the next batch. This buffer ignores it;
        the adaptive buffer updates its demand here.

        Args:
            trainer_policy_version: Version of the weights that will train the batch being waited for.
        """

    def pop_dropped_group_ids(self) -> list[int]:
        """Trainer loop, at the optimizer step: groups dropped since the last call, so the dataloader
        stops replaying them. This buffer drops nothing; the adaptive buffer drops groups too old to train.
        """
        return []

    async def release_active_groups(self, count: int, *, reason: str) -> None:
        """Free active slots: the trainer releases trained slots after its model state dict pull; the batcher
        releases untrainable/filtered slots immediately.

        Args:
            count:  Number of rollout groups leaving the active slots.
            reason: Metric suffix such as `"trained"` or `"untrainable_group"`.

        Example:
            # trainer pulled weights after a step over 8 groups -> free their 8 slots
            await buffer.release_active_groups(8, reason="trained")
            # batcher dropped one zero-std group -> free its single slot
            await buffer.release_active_groups(1, reason="untrainable_group")
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        if count == 0:
            return
        async with self._condition:
            if count > self._active_rollout_groups:
                raise RuntimeError(
                    f"release_active_groups({count}) exceeds active count {self._active_rollout_groups}"
                )
            self._active_rollout_groups -= count
            sl.log_trace_scalar({f"rollout_buffer/released/{reason}": float(count)})
            self._condition.notify_all()

    async def close(self) -> None:
        """run() shutdown calls this once. Sets `_closed`, drops buffered work, and wakes every waiter.

        After this, all four waiters unblock and exit their loops: wait_for_slot() returns False, and
        claim_next()/take_finalized() return None.
        """
        async with self._condition:
            self._closed = True
            self._work_by_group_id.clear()
            self._condition.notify_all()

    def metrics(self) -> list[m.Metric]:
        """Trainer loop: point-in-time buffer gauges for this step; resets the per-flush peak.

        Every slot is in exactly one of these five, so they sum to `max_active_rollout_groups`:

            num_groups_waiting      admitted, no rollout started
            num_groups_inflight     rollouts running
            num_groups_finalized    rollouts done, not yet taken by the batcher
            num_groups_taken        taken by the batcher: in the batcher, the batch queue or the trainer
            available_active_slots  free
        """
        states = [work.state for work in self._work_by_group_id.values()]
        state_enum = _RolloutGroupWorkState
        # Taken groups left `_work_by_group_id` but keep their slot until `release_active_groups`.
        num_groups_taken = self._active_rollout_groups - len(self._work_by_group_id)
        out = [
            m.Metric(
                "rollout_buffer/num_groups_waiting",
                m.NoReduce(float(states.count(state_enum.WAITING))),
            ),
            m.Metric(
                "rollout_buffer/num_groups_inflight",
                m.NoReduce(float(states.count(state_enum.INFLIGHT))),
            ),
            m.Metric(
                "rollout_buffer/num_groups_finalized",
                m.NoReduce(float(states.count(state_enum.FINALIZED))),
            ),
            m.Metric(
                "rollout_buffer/num_groups_taken",
                m.NoReduce(float(num_groups_taken)),
            ),
            m.Metric(
                "rollout_buffer/active_slots_in_use_peak",
                m.NoReduce(float(self._active_rollout_groups_peak_since_flush)),
            ),
            m.Metric(
                "rollout_buffer/available_active_slots",
                m.NoReduce(
                    float(self._max_active_rollout_groups - self._active_rollout_groups)
                ),
            ),
        ]
        # Next interval starts from the current gauge, not 0: slots stay occupied across a flush.
        self._active_rollout_groups_peak_since_flush = self._active_rollout_groups
        return out


class AdaptiveRolloutGroupWorkBuffer(RolloutGroupWorkBuffer):
    """Oldest-ready buffer with exact age eviction and a demand learned from the run.

    Same callers and lifecycle as `RolloutGroupWorkBuffer`; three rules differ:

    1. Demand. At every step start the buffer counts the groups that are not ready (generating, or the batch
       being trained) and hands the count to `StallDrivenDemand`: demand = one batch + the value that count stays
       under on 95% of steps + one spare group, moved up half the gap or down one group per step, never above the
       mean-age ceiling (see `adaptive_demand.py`). `generation_capacity` stays a separate limit that admission
       enforces, so demand cannot enlarge vLLM concurrency.
    2. Selection. `take_finalized` returns the oldest FINALIZED group wherever it sits; a slow group never
       blocks younger finished ones and keeps its slot until it finishes.
    3. Age. A finalized group that would be consumed more than `max_offpolicy_steps` versions after it was
       claimed is dropped (slot released at once, prompt not retried); with `max_offpolicy_steps=None` nothing is
       dropped. A group still generating is cancelled as soon as the batch being assembled makes it that old, since
       it can never train; it counts as a dropped group at once, and its slot is released when its generation task
       ends, so a replacement group never overlaps it. The mean age is held under `target_offpolicy_steps` if set,
       else under `max_offpolicy_steps` if set, by the demand ceiling, at the price of stalls when the workload needs
       more groups than the ceiling allows. With neither, the demand follows the workload alone.

    Example:
        buffer = AdaptiveRolloutGroupWorkBuffer.Config(
            max_offpolicy_steps=10, generation_capacity=60
        ).build(num_prompts_per_train_step=8)
        buffer.metrics()   # rollout_buffer/demand_target_groups 24: three batches to start
        await buffer.record_step_start(trainer_policy_version=0)   # nothing ready: unavailable 24 -> demand 29
        buffer.metrics()   # rollout_buffer/demand_target_groups 29, rollout_buffer/ready_at_step_start 0

        # g0 claimed at version 0; the batch being assembled trains at version 11 -> age 11 > 10 -> dropped
        await buffer.finalize_work(RolloutGroup(group_id=0, rollouts=[...]))
        await buffer.take_finalized(consuming_policy_version=11)
        # -> slot released with reason "too_old"; selection skips to the next finalized group
        # g1, also claimed at version 0, is still generating -> its generation task is cancelled; the slot is
        # released with reason "cancelled_too_old" once the task ends

    Args:
        num_prompts_per_train_step: Prompt groups per train step (`P`).
        policy_version: Version the generators and the trainer hold at start (the resumed step).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        max_offpolicy_steps: int | None = 4
        """Bounds the age of EVERY trained group: a finalized group older than this at consumption is dropped
        (prompt not retried), and a group still generating is cancelled once it is that old. Without
        `target_offpolicy_steps`, also the mean age the demand ceiling holds under. None: nothing is dropped for age."""

        target_offpolicy_steps: int | None = None
        """Bounds the MEAN age at this value: demand is capped at the mean-age ceiling
        `target * P + P + generating * untrainable_share`. Costs stalls when generation cannot fill a batch within
        the cap, never waste. None: the same ceiling is applied at `max_offpolicy_steps`, if set."""

        lookback_steps: int = 10
        """How many of the most recent step starts the demand rule looks back over; older values are forgotten."""

        stall_probability: float = 0.05
        """Share of steps allowed to stall; the demand covers the unavailable count on 1 - stall_probability of steps."""

        start_batches: int = 3
        """Demand at the first step, in batches of P; the rule learns the rest from the run itself."""

        damping_factor: float = 0.5
        """Share of the gap to the computed need closed per step going up, or coming down to the mean-age ceiling;
        otherwise demand comes down one group per step."""

        generation_capacity: int | None = None
        """Most prompt groups generating at once; sizes rollout workers and vLLM `max_num_seqs`.

        None, with `target_offpolicy_steps`: `2 * (target_offpolicy_steps + 1) * num_prompts_per_train_step`.
        The demand stays under its mean-age ceiling near `(target + 1) * P`; the factor 2 leaves room for
        up to half the groups being untrainable (zero-std), which a step consumes on top of `P`.
        None, without `target_offpolicy_steps`: fill vLLM, `DEFAULT_SEQS_PER_GENERATOR_DP_SHARD` (512)
        sequences per generator DP shard, `512 * num_generator_dp_shards // num_samples_per_prompt`.
        Set it lower only for a hard limit outside the GPUs, e.g. `sandboxes // num_samples_per_prompt`.

        Example:
            # target_offpolicy_steps=6, 48 prompts per step -> 2 * 7 * 48 = 672 groups at most
            AdaptiveRolloutGroupWorkBuffer.Config(max_offpolicy_steps=10, target_offpolicy_steps=6)
            # no target, 24 generator DP shards, 16 samples per prompt -> 512 * 24 // 16 = 768 groups
            AdaptiveRolloutGroupWorkBuffer.Config(max_offpolicy_steps=10)
        """

        def __post_init__(self) -> None:
            if self.max_offpolicy_steps is not None and self.max_offpolicy_steps < 1:
                raise ValueError(
                    f"max_offpolicy_steps must be None or >= 1, got {self.max_offpolicy_steps}"
                )
            if self.target_offpolicy_steps is not None and (
                self.target_offpolicy_steps < 1
                or (
                    self.max_offpolicy_steps is not None
                    and self.target_offpolicy_steps > self.max_offpolicy_steps
                )
            ):
                raise ValueError(
                    "target_offpolicy_steps must be None or in [1, max_offpolicy_steps], "
                    f"got {self.target_offpolicy_steps} with max_offpolicy_steps={self.max_offpolicy_steps}"
                )
            if self.lookback_steps < 2:
                raise ValueError(
                    f"lookback_steps must be >= 2, got {self.lookback_steps}"
                )
            if not (0 < self.stall_probability < 0.5):
                raise ValueError(
                    f"stall_probability must be in (0, 0.5), got {self.stall_probability}"
                )
            if self.start_batches < 1:
                raise ValueError(
                    f"start_batches must be >= 1, got {self.start_batches}"
                )
            if not (0 < self.damping_factor <= 1):
                raise ValueError(
                    f"damping_factor must be in (0, 1], got {self.damping_factor}"
                )
            if self.generation_capacity is not None and self.generation_capacity < 1:
                raise ValueError(
                    f"generation_capacity must be >= 1, got {self.generation_capacity}"
                )

        def resolve_generation_capacity(
            self,
            num_prompts_per_train_step: int,
            *,
            num_generator_dp_shards: int | None = None,
            num_samples_per_prompt: int | None = None,
        ) -> int:
            """`generation_capacity`, or its default when None (see the field).

            Without `target_offpolicy_steps` the default needs the generator layout, known once the
            generators are spawned.

            Example:
                config = AdaptiveRolloutGroupWorkBuffer.Config(max_offpolicy_steps=10)
                config.resolve_generation_capacity(
                    48, num_generator_dp_shards=24, num_samples_per_prompt=16
                )  # 768
            """
            if self.generation_capacity is not None:
                return self.generation_capacity
            if self.target_offpolicy_steps is not None:
                return (
                    2 * (self.target_offpolicy_steps + 1) * num_prompts_per_train_step
                )
            if num_generator_dp_shards is None or num_samples_per_prompt is None:
                raise ValueError(
                    "generation_capacity defaults to filling vLLM without target_offpolicy_steps, "
                    "which needs num_generator_dp_shards and num_samples_per_prompt"
                )
            return max(
                1,
                DEFAULT_SEQS_PER_GENERATOR_DP_SHARD
                * num_generator_dp_shards
                // num_samples_per_prompt,
            )

    def __init__(
        self,
        config: Config,
        *,
        num_prompts_per_train_step: int,
        policy_version: int = 0,
    ) -> None:
        self._demand = StallDrivenDemand(
            num_prompts_per_train_step=num_prompts_per_train_step,
            max_offpolicy_steps=config.max_offpolicy_steps,
            target_offpolicy_steps=config.target_offpolicy_steps,
            lookback_steps=config.lookback_steps,
            stall_probability=config.stall_probability,
            start_batches=config.start_batches,
            damping_factor=config.damping_factor,
        )
        # The demand is the active-slot cap; only `record_step_start` moves it. No window: oldest finalized first.
        super().__init__(
            RolloutGroupWorkBuffer.Config(),
            max_active_rollout_groups=self._demand.demand,
            window_size=None,
        )
        self._num_prompts_per_train_step = num_prompts_per_train_step
        self._max_offpolicy_steps = config.max_offpolicy_steps
        self._generation_capacity = config.resolve_generation_capacity(
            num_prompts_per_train_step
        )
        self._warned_capacity_binds = False
        self._generator_policy_version = policy_version
        self._trainer_policy_version = policy_version
        # metrics: the shelf and the generating count at the last step start; flow since the last step start
        self._ready_at_step_start = 0
        self._generating_at_step_start = 0
        self._unavailable_at_step_start = 0
        self._completed_since_step_start = 0
        self._untrainable_since_step_start = 0
        self._dropped_since_step_start = 0
        self._dropped_too_old_since_flush = 0
        self._cancelled_too_old_since_flush = 0
        self._dropped_group_ids: list[int] = []
        # Highest version the batcher assembled a batch for; in-flight groups are checked when it advances.
        self._consuming_policy_version = policy_version

    def _has_active_slot_available(self) -> bool:
        generating = sum(
            work.state
            in (_RolloutGroupWorkState.WAITING, _RolloutGroupWorkState.INFLIGHT)
            for work in self._work_by_group_id.values()
        )
        return (
            super()._has_active_slot_available()
            and generating < self._generation_capacity
        )

    async def add_work(self, work: RolloutGroupWork) -> None:
        """Admit one rollout group as WAITING and charge one active slot, without re-checking the demand.

        `record_step_start` can lower the demand while the data input loop reads the sample a
        `wait_for_slot` let in; that group is still admitted, and the lower demand applies to the next one.
        """
        async with self._condition:
            self._active_rollout_groups += 1
            self._active_rollout_groups_peak_since_flush = max(
                self._active_rollout_groups_peak_since_flush,
                self._active_rollout_groups,
            )
            self._work_by_group_id[work.group_id] = work
            self._condition.notify_all()

    async def claim_next(self) -> RolloutGroupWork | None:
        """Claim like the base buffer, and record the generator version the group starts under."""
        work = await super().claim_next()
        if work is not None:
            work.policy_version_at_claim = self._generator_policy_version
        return work

    def _is_too_old(
        self, work: RolloutGroupWork, *, consuming_policy_version: int
    ) -> bool:
        assert work.policy_version_at_claim is not None
        return (
            self._max_offpolicy_steps is not None
            and consuming_policy_version - work.policy_version_at_claim
            > self._max_offpolicy_steps
        )

    def _drop_too_old(self, work: RolloutGroupWork) -> None:
        """Remove a finalized group past `max_offpolicy_steps` and free its slot at once. Caller holds the condition."""
        del self._work_by_group_id[work.group_id]
        self._dropped_group_ids.append(work.group_id)
        self._active_rollout_groups -= 1
        self._dropped_too_old_since_flush += 1
        self._dropped_since_step_start += 1
        sl.log_trace_scalar({"rollout_buffer/released/too_old": 1.0})
        self._condition.notify_all()

    def _cancel_too_old_inflight(self, *, consuming_policy_version: int) -> None:
        """Cancel every in-flight group already past `max_offpolicy_steps` for the batch being assembled.

        Batches train in version order and age only grows, so such a group would be dropped when it finalized:
        cancel its generation task now (which stops its rollouts) instead of waiting for its slowest rollout. It
        counts as a completed, dropped group, as a finalized too-old drop does; its slot stays charged until the
        task ends (`release_cancelled`). Caller holds the condition.
        """
        for work in self._work_by_group_id.values():
            if (
                work.state is not _RolloutGroupWorkState.INFLIGHT
                or work.cancelled
                or not self._is_too_old(
                    work, consuming_policy_version=consuming_policy_version
                )
            ):
                continue
            work.cancelled = True
            if work.generation_task is not None:
                work.generation_task.cancel()
            self._dropped_group_ids.append(work.group_id)
            self._cancelled_too_old_since_flush += 1
            self._completed_since_step_start += 1
            self._dropped_since_step_start += 1

    def pop_dropped_group_ids(self) -> list[int]:
        dropped_group_ids, self._dropped_group_ids = self._dropped_group_ids, []
        return dropped_group_ids

    async def finalize_work(self, rollout_group: RolloutGroup) -> bool:
        stored = await super().finalize_work(rollout_group)
        if stored:
            self._completed_since_step_start += 1
        return stored

    async def release_active_groups(self, count: int, *, reason: str) -> None:
        await super().release_active_groups(count, reason=reason)
        if reason == "untrainable_group":
            self._untrainable_since_step_start += count
        elif reason == "trained":
            # The weight sync releases one batch of trained slots per pulled version.
            self._generator_policy_version += 1

    @sl.log_trace_span("take_finalized")
    async def take_finalized(
        self, *, consuming_policy_version: int | None = None
    ) -> RolloutGroup | None:
        """Batcher loop: return the oldest FINALIZED group, dropping any that became too old while waiting.

        Example:
            # g0 is INFLIGHT (slow), g1 and g2 are FINALIZED -> g1 is returned; g0 keeps its slot
            group = await buffer.take_finalized()
            assert group.group_id == 1
        """
        if consuming_policy_version is None:
            consuming_policy_version = self._trainer_policy_version
        async with self._condition:
            if consuming_policy_version > self._consuming_policy_version:
                self._consuming_policy_version = consuming_policy_version
                self._cancel_too_old_inflight(
                    consuming_policy_version=consuming_policy_version
                )
            while True:
                if self._closed:
                    return None
                for work in list(self._work_by_group_id.values()):
                    if work.state is not _RolloutGroupWorkState.FINALIZED:
                        continue
                    if self._is_too_old(
                        work, consuming_policy_version=consuming_policy_version
                    ):
                        self._drop_too_old(work)
                        continue
                    del self._work_by_group_id[work.group_id]
                    self._condition.notify_all()
                    return work.rollout_group
                await self._condition.wait()  # nothing finalized -> stall

    async def record_step_start(self, *, trainer_policy_version: int) -> None:
        """Trainer loop, right before it waits for a batch: measure the shelf and update the demand.

        The shelf is everything admitted that is neither still generating nor held by the trainer:
        `active - (waiting + inflight) - trained awaiting release`, i.e. finalized, selected, and queued groups.
        """
        async with self._condition:
            self._trainer_policy_version = trainer_policy_version
            states = [work.state for work in self._work_by_group_id.values()]
            generating = states.count(_RolloutGroupWorkState.INFLIGHT) + states.count(
                _RolloutGroupWorkState.WAITING
            )
            # The trainer holds one batch of slots per version the generators have not pulled yet.
            trained_awaiting_release = self._num_prompts_per_train_step * (
                trainer_policy_version - self._generator_policy_version
            )
            ready = self._active_rollout_groups - generating - trained_awaiting_release
            self._generating_at_step_start = generating
            self._ready_at_step_start = ready
            self._unavailable_at_step_start = max(0, self._demand.demand - ready)
            completed = self._completed_since_step_start
            self._demand.observe(
                step=trainer_policy_version + 1,
                ready=ready,
                generating=generating,
                completed=completed,
                trainable=max(
                    0,
                    completed
                    - self._untrainable_since_step_start
                    - self._dropped_since_step_start,
                ),
            )
            self._completed_since_step_start = 0
            self._untrainable_since_step_start = 0
            self._dropped_since_step_start = 0
            # Warn once: past the capacity, extra demand cannot run and the trainer waits instead.
            if (
                self._demand.demand > self._generation_capacity
                and not self._warned_capacity_binds
            ):
                self._warned_capacity_binds = True
                logger.warning(
                    f"Rollout buffer: demand {self._demand.demand} groups > generation_capacity "
                    f"{self._generation_capacity}, so the trainer can wait for batches. If "
                    "generator/num_preemptions stays 0, the generators have room: raise "
                    "generation_capacity or leave it unset."
                )
            grew = self._demand.demand > self._max_active_rollout_groups
            self._max_active_rollout_groups = self._demand.demand
            if grew:
                self._condition.notify_all()

    def metrics(self) -> list[m.Metric]:
        generating = sum(
            work.state
            in (_RolloutGroupWorkState.WAITING, _RolloutGroupWorkState.INFLIGHT)
            for work in self._work_by_group_id.values()
        )
        out = [
            *super().metrics(),
            m.Metric(
                "rollout_buffer/generation_capacity_groups",
                m.NoReduce(float(self._generation_capacity)),
            ),
            m.Metric(
                "rollout_buffer/available_generation_permits",
                m.NoReduce(float(self._generation_capacity - generating)),
            ),
            m.Metric(
                "rollout_buffer/demand_target_groups",
                m.NoReduce(float(self._demand.demand)),
            ),
            m.Metric(
                "rollout_buffer/ready_at_step_start",
                m.NoReduce(float(self._ready_at_step_start)),
            ),
            m.Metric(
                "rollout_buffer/generating_at_step_start",
                m.NoReduce(float(self._generating_at_step_start)),
            ),
            # the rule's observable: slots that held no ready group at the step start
            m.Metric(
                "rollout_buffer/unavailable_at_step_start",
                m.NoReduce(float(self._unavailable_at_step_start)),
            ),
            m.Metric(
                "rollout_buffer/dropped_too_old",
                m.NoReduce(float(self._dropped_too_old_since_flush)),
            ),
            # groups cancelled while generating, already too old to train
            m.Metric(
                "rollout_buffer/cancelled_too_old",
                m.NoReduce(float(self._cancelled_too_old_since_flush)),
            ),
            # 1 while the mean-age ceiling caps the demand, else 0
            m.Metric(
                "rollout_buffer/demand_age_limited",
                m.NoReduce(float(self._demand.state == "age-limited")),
            ),
        ]
        self._dropped_too_old_since_flush = 0
        self._cancelled_too_old_since_flush = 0
        return out
