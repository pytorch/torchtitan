# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Buffers rollout groups between data input, rollout workers, and the batcher.

data input -> WAITING -> rollout worker -> FINALIZED -> batcher

The buffer is a strict-FIFO run-ahead queue with an active-slot budget: it caps how far generation may
run ahead of the trainer ((S+1)*B active slots) and hands the batcher the OLDEST finalized group,
stalling if that group is still in flight. The trainer releases active slots after each weight pull, so
nothing reaches the trainer with consume-time age > S.
"""

import asyncio
import collections
import enum
from dataclasses import dataclass, field

from torchtitan.config import Configurable
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.rollout import RolloutGroup
from torchtitan.observability import structured_logger as sl


class _WorkState(enum.Enum):
    """Where a rollout group is in the input -> generation -> batching lifecycle."""

    WAITING = "waiting"
    INFLIGHT = "inflight"
    FINALIZED = "finalized"


@dataclass(slots=True)
class RolloutGroupWork:
    """One prompt group's lifecycle inside the buffer: input -> generation -> ready for batching.

    The input loop sets `group_id` + `sample`; the buffer owns `state` and `rollout_group`
    (`init=False`, so the input loop can't set them).

    Example:
        RolloutGroupWork(group_id=5, sample=sample)
    """

    group_id: int
    sample: object
    state: _WorkState = field(default=_WorkState.WAITING, init=False)
    rollout_group: RolloutGroup | None = field(
        default=None, init=False
    )  # set once FINALIZED
    # TODO(async-rl): emit JSON lifecycle logging per RolloutGroupWork keyed by group_id:
    # admitted/claimed/finalized/batched/trained/dropped timestamps + policy version at admission and
    # at trainer consumption, for faithful end-to-end visibility.


class RolloutGroupWorkBuffer(Configurable):
    """Strict-FIFO rollout-group buffer with whole-pipeline active-slot backpressure.

    State:
        WAITING -> INFLIGHT -> FINALIZED -> removed by take_finalized()

    Active slot:
        add_work() ------------------------------------------------------ release_active_groups()
             | claim_next() | finalize_work() | take_finalized() | batcher | trainer |

    Example:
        # max_offpolicy_steps=1, num_groups_per_train_step=2 -> capacity=4
        # g0..g3 admitted and still active; g0 was already take_finalized()'d into the batcher.
        await buffer.wait_for_slot()  # waits: take_finalized does not free active slots
        await buffer.release_active_groups(2, reason="trained")
        await buffer.wait_for_slot()  # returns: the trainer released one train step
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """No tunables: capacity is passed in by the controller from the run-ahead sizing."""

    def __init__(self, config: Config, *, max_active_rollout_groups: int) -> None:
        self._max_active_rollout_groups = max_active_rollout_groups
        self._active_rollout_groups = 0
        self._active_rollout_groups_peak = (
            0  # high-water mark of active slots since start
        )
        # TODO(async-rl): stall-only backpressure today; strict FIFO + (S+1)*B active budget bounds consume age
        #   <= S, so nothing is ever stale enough to drop. When we move past stall-only, extract this active-slot
        #   counter into a StalenessBudget/manager and add there: (a) drop/recycle a stale group instead of
        #   stalling on it, (b) a drop_rollout_group_if_any_stale all-or-nothing mode.
        self._work_by_group_id: collections.OrderedDict[
            int, RolloutGroupWork
        ] = collections.OrderedDict()
        # One Condition guards all three waits (slot-free / claimable-WAITING / takeable-FINALIZED):
        # every mutation notify_all()s and waiters re-check their predicate. N workers wait to claim,
        # so a single Condition is lost-wakeup-proof here.
        # TODO(async-rl): if the claim herd ever shows up in a profile, move the WAITING->INFLIGHT
        # handoff to an asyncio.Queue (one get() wakes one worker); keep the strict-FIFO FINALIZED
        # consumption on the Condition.
        self._condition = asyncio.Condition()
        self._closed = False
        # TODO(async-rl): warm start — admit a small number of groups at first and grow the effective cap as the
        # batcher consumes, so a cold start doesn't fill the whole off-policy window at policy version 0.

    def _has_active_slot_unlocked(self) -> bool:
        return self._active_rollout_groups < self._max_active_rollout_groups

    async def wait_for_slot(self) -> bool:
        """Wait until one more rollout group may enter the active off-policy window.

        Example:
            # False means the buffer was closed, so the data input loop exits.
            if await buffer.wait_for_slot():
                await buffer.add_work(RolloutGroupWork(group_id=0, sample=sample))
        """
        async with self._condition:
            await self._condition.wait_for(
                lambda: self._closed or self._has_active_slot_unlocked()
            )
            return not self._closed

    async def add_work(self, work: RolloutGroupWork) -> None:
        """Admit one rollout group as WAITING and charge one active slot."""
        async with self._condition:
            if not self._has_active_slot_unlocked():
                raise RuntimeError(
                    "RolloutGroupWorkBuffer.add_work called without an active slot"
                )
            self._active_rollout_groups += 1
            self._active_rollout_groups_peak = max(
                self._active_rollout_groups_peak, self._active_rollout_groups
            )
            self._work_by_group_id[work.group_id] = work
            self._condition.notify_all()

    async def claim_next(self) -> RolloutGroupWork | None:
        """Rollout loop: take the oldest WAITING group to generate. None once closed."""
        async with self._condition:
            while True:
                if self._closed:
                    return None
                for work in self._work_by_group_id.values():
                    if work.state is _WorkState.WAITING:
                        work.state = _WorkState.INFLIGHT
                        return work
                await self._condition.wait()

    async def finalize_work(self, rollout_group: RolloutGroup) -> None:
        """Rollout loop: write the generated result into the existing work entry and wake the batcher."""
        async with self._condition:
            work = self._work_by_group_id.get(rollout_group.group_id)
            if work is None:
                # The buffer was closed and cleared while this group was in flight.
                return
            work.state = _WorkState.FINALIZED
            work.rollout_group = rollout_group
            self._condition.notify_all()

    @sl.log_trace_span("take_finalized")
    async def take_finalized(self) -> RolloutGroup | None:
        """Batcher loop: strict FIFO — return the OLDEST group once it is FINALIZED, else stall.

        Example:
            # head g0 still INFLIGHT, g1 FINALIZED -> WAITS for g0 (no skipping)
            await buffer.take_finalized()
        """
        async with self._condition:
            while True:
                if self._closed:
                    return None
                if self._work_by_group_id:
                    oldest_group_id, oldest_work = next(
                        iter(self._work_by_group_id.items())
                    )
                    if oldest_work.state is _WorkState.FINALIZED:
                        del self._work_by_group_id[oldest_group_id]
                        self._condition.notify_all()
                        if oldest_work.rollout_group is None:
                            raise RuntimeError(
                                f"finalized rollout group {oldest_group_id} has no payload"
                            )
                        return oldest_work.rollout_group
                await self._condition.wait()  # head still INFLIGHT -> STALL

    async def release_active_groups(self, count: int, *, reason: str) -> None:
        """Free active slots for groups that can no longer become stale training data.

        Args:
            count: Number of rollout groups leaving the active window.
            reason: Metric suffix such as `"trained"` or `"untrainable_group"`.

        Example:
            await buffer.release_active_groups(8, reason="trained")
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
            sl.log_trace_scalar(
                {f"rollout_active_budget/released/{reason}": float(count)}
            )
            self._condition.notify_all()

    async def close(self) -> None:
        """`run()` shutdown: abandon buffered work and wake every waiter."""
        async with self._condition:
            self._closed = True
            self._work_by_group_id.clear()
            self._condition.notify_all()

    def metrics(self) -> list[m.Metric]:
        """Trainer loop: point-in-time buffer gauges for this step."""
        states = [work.state for work in self._work_by_group_id.values()]
        S = _WorkState
        return [
            m.Metric(
                "rollout_buffer/num_groups_waiting",
                m.NoReduce(float(states.count(S.WAITING))),
            ),
            m.Metric(
                "rollout_buffer/num_groups_inflight",
                m.NoReduce(float(states.count(S.INFLIGHT))),
            ),
            m.Metric(
                "rollout_buffer/num_groups_finalized",
                m.NoReduce(float(states.count(S.FINALIZED))),
            ),
            m.Metric(
                "rollout_active_budget/capacity",
                m.NoReduce(float(self._max_active_rollout_groups)),
            ),
            m.Metric(
                "rollout_active_budget/in_use",
                m.NoReduce(float(self._active_rollout_groups)),
            ),
            m.Metric(
                "rollout_active_budget/in_use_peak",
                m.NoReduce(float(self._active_rollout_groups_peak)),
            ),
            m.Metric(
                "rollout_active_budget/available",
                m.NoReduce(
                    float(self._max_active_rollout_groups - self._active_rollout_groups)
                ),
            ),
        ]
