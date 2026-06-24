# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Buffers rollout groups between data input, rollout workers, and the batcher.

data input -> WAITING -> rollout worker -> FINALIZED -> batcher

The buffer is a bounded, strict-FIFO run-ahead queue: it caps how far generation may run ahead of the
trainer (count backpressure) and hands the batcher the OLDEST finalized group, stalling if that group
is still in flight. Policy-version staleness is enforced once, later, by `Batcher` at flush.
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
    """Bounded strict-FIFO buffer of rollout groups moving input -> generation -> batching.

    Count backpressure: admission blocks once `len(buffer) >= max_buffered_rollout_groups`. The batcher
    takes the OLDEST group and stalls if it is still in flight (no skipping, no eviction).

    Example:
        buffer = config.rollout_buffer.build(max_buffered_rollout_groups=32)

        # input loop:
        if await buffer.wait_for_slot():
            await buffer.add_work(RolloutGroupWork(group_id=0, sample=sample))

        # rollout worker:
        work = await buffer.claim_next()
        await buffer.finalize_work(RolloutGroup(group_id=work.group_id, rollouts=rollouts))

        # batcher:
        rollout_group = await buffer.take_finalized()
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """No tunables: capacity is passed in by the controller from the run-ahead sizing."""

    def __init__(self, config: Config, *, max_buffered_rollout_groups: int) -> None:
        self._max_buffered_rollout_groups = max_buffered_rollout_groups
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

    async def wait_for_slot(self) -> bool:
        """Data input loop: block until one more group can be admitted. Returns False once closed."""
        async with self._condition:
            await self._condition.wait_for(
                lambda: self._closed
                or len(self._work_by_group_id) < self._max_buffered_rollout_groups
            )
            return not self._closed

    async def add_work(self, work: RolloutGroupWork) -> None:
        """Data input loop: admit one group as WAITING."""
        async with self._condition:
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
                    head_id, head = next(iter(self._work_by_group_id.items()))
                    if head.state is _WorkState.FINALIZED:
                        del self._work_by_group_id[head_id]
                        self._condition.notify_all()
                        assert head.rollout_group is not None
                        return head.rollout_group
                await self._condition.wait()  # head still INFLIGHT -> STALL

    async def close(self) -> None:
        """`run()` shutdown: abandon buffered work and wake every waiter."""
        async with self._condition:
            self._closed = True
            self._work_by_group_id.clear()
            self._condition.notify_all()

    def metrics(self) -> list[m.Metric]:
        """Trainer loop: point-in-time buffer gauges for this step."""
        states = [work.state for work in self._work_by_group_id.values()]
        capacity_used = len(self._work_by_group_id) / max(
            self._max_buffered_rollout_groups, 1
        )
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
            m.Metric("rollout_buffer/capacity_used_frac", m.NoReduce(capacity_used)),
        ]
