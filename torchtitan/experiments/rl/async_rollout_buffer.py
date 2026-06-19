# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Buffers rollout groups between data input, rollout workers, and the batcher.

data input -> WAITING -> rollout worker -> GENERATED -> episode batcher

The buffer enforces count backpressure and evicts groups doomed to be stale. `EpisodeBatcher`
enforces exact policy staleness when it flushes a training batch.
"""

import asyncio
import collections
import enum
from dataclasses import dataclass

from torchtitan.config import Configurable
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.rollout import RolloutGroup
from torchtitan.observability import structured_logger as sl


class _RolloutGroupBufferState(enum.Enum):
    """Where a rollout group is in the input -> generation -> batching lifecycle."""

    WAITING = "waiting"
    GENERATING = "generating"
    GENERATED = "generated"


@dataclass(frozen=True, slots=True)
class RolloutGroupWork:
    """A prompt group waiting to be generated."""

    group_id: str
    sample: object


@dataclass(slots=True)
class _BufferedGroup:
    """One rollout group's mutable buffer state."""

    work: RolloutGroupWork
    state: _RolloutGroupBufferState = _RolloutGroupBufferState.WAITING
    rollout_group: RolloutGroup | None = None  # set once GENERATED
    groups_fed_ahead: int = 0
    """groups fed past this one since it started generating.
    Used to evict to-be-born-stale groups."""


class AsyncRolloutBuffer(Configurable):
    """Queues rollout-group work and returns generated groups in feed order.

    Two count-based controls:

        len(buffer)            >= max_buffered_rollout_groups -> block new admission (run-ahead cap)
        group.groups_fed_ahead > max_buffered_rollout_groups  -> evict a doomed-to-be-stale group

    For details on eviction, see `_evict_doomed_rollout_groups`.

    Example:
        buffer = config.buffer.build(max_buffered_rollout_groups=24)

        # input loop:
        if await buffer.wait_for_rollout_group_slot():
            await buffer.add_rollout_group_work(
                RolloutGroupWork(group_id="g0", sample=sample)
            )

        # rollout worker:
        work = await buffer.claim_next_rollout_group()
        await buffer.record_rollout_group_result(
            RolloutGroup(group_id=work.group_id, rollouts=rollouts)
        )

        # batcher:
        rollout_group = await buffer.take_generated_rollout_group()
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Buffer feed-order tuning."""

        buffer_to_batcher_feedable_window: int = 4
        """How far the batcher may look past a still-generating group to take a finished one. If 1, then we have FIFO,
        and a straggler group will block the batch. larger lets newer finished groups go first (dodges head-of-line stalls),
        but will prioritize shorter and newer group rollouts, potentially wasting older groups that might become stale"""

    def __init__(self, config: Config, *, max_buffered_rollout_groups: int) -> None:
        self._feedable_window = config.buffer_to_batcher_feedable_window
        self._max_buffered_rollout_groups = max_buffered_rollout_groups
        self._buffered_groups_by_id: collections.OrderedDict[
            str, _BufferedGroup
        ] = collections.OrderedDict()
        # One condition protects admission, rollout-worker claims, and batcher takes.
        # Any state change can satisfy a different waiter, so every mutation uses notify_all.
        self._condition = asyncio.Condition()
        self._closed = False
        self._num_groups_evicted_total = 0
        # TODO(async-rl): warm start — admit a small number of groups at first and grow the effective cap as the
        # batcher consumes, so a cold start doesn't fill the whole off-policy window at policy version 0.

    async def wait_for_rollout_group_slot(self) -> bool:
        """Data input loop: block until one more group can be admitted. Returns False once closed."""
        async with self._condition:
            await self._condition.wait_for(
                lambda: self._closed
                or len(self._buffered_groups_by_id) < self._max_buffered_rollout_groups
            )
            return not self._closed

    async def add_rollout_group_work(self, work: RolloutGroupWork) -> None:
        """Data input loop: admit one group as WAITING."""
        async with self._condition:
            self._buffered_groups_by_id[work.group_id] = _BufferedGroup(work)
            self._condition.notify_all()

    async def claim_next_rollout_group(self) -> RolloutGroupWork | None:
        """Rollout loop: take the oldest WAITING group to generate. None once closed."""
        async with self._condition:
            while True:
                for group in self._buffered_groups_by_id.values():
                    if group.state is _RolloutGroupBufferState.WAITING:
                        group.state = _RolloutGroupBufferState.GENERATING
                        return group.work
                if self._closed:
                    return None
                await self._condition.wait()

    async def record_rollout_group_result(self, rollout_group: RolloutGroup) -> None:
        """Rollout loop: store one generated group and wake the batcher."""
        async with self._condition:
            buffered = self._buffered_groups_by_id.get(rollout_group.group_id)
            if buffered is None:
                # A liveness eviction already freed this slot.
                return
            buffered.state = _RolloutGroupBufferState.GENERATED
            buffered.rollout_group = rollout_group
            self._condition.notify_all()

    @sl.log_trace_span("take_generated_rollout_group")
    async def take_generated_rollout_group(self) -> RolloutGroup | None:
        """Episode batcher loop: return the oldest generated group in the feedable window.

        Example:
            # window covers [g0 generating, g1 generated, g2 generated]
            await buffer.take_generated_rollout_group()
            # -> g1; g0 stays; other non-WAITING groups get groups_fed_ahead += 1
        """
        async with self._condition:
            while True:
                self._evict_doomed_rollout_groups()
                for index, (group_id, group) in enumerate(
                    self._buffered_groups_by_id.items()
                ):
                    if index >= self._feedable_window:
                        break
                    if group.state is _RolloutGroupBufferState.GENERATED:
                        del self._buffered_groups_by_id[group_id]
                        for other in self._buffered_groups_by_id.values():
                            if other.state is not _RolloutGroupBufferState.WAITING:
                                # WAITING groups have not sampled yet.
                                other.groups_fed_ahead += 1
                        self._condition.notify_all()
                        return group.rollout_group
                if self._closed:
                    return None
                await self._condition.wait()

    async def close(self) -> None:
        """`run()` shutdown: abandon buffered work and wake every waiter."""
        async with self._condition:
            self._closed = True
            self._buffered_groups_by_id.clear()
            self._condition.notify_all()

    def _evict_doomed_rollout_groups(self) -> None:
        """Episode batcher loop: evict non-WAITING groups fed past a full off-policy window.

        `max_buffered_rollout_groups == max_offpolicy_steps * groups_per_train_step`, so a group with
        more feeds ahead than that would exceed `max_offpolicy_steps` by the time it trains -- it is
        doomed to be stale-dropped, so evict it now to free the slot.
        """
        for group_id, group in list(self._buffered_groups_by_id.items()):
            if (
                group.state is not _RolloutGroupBufferState.WAITING
                and group.groups_fed_ahead > self._max_buffered_rollout_groups
            ):
                del self._buffered_groups_by_id[group_id]
                self._num_groups_evicted_total += 1

    def metrics(self) -> list[m.Metric]:
        """Trainer loop: return point-in-time buffer gauges for this step."""
        states = [group.state for group in self._buffered_groups_by_id.values()]
        fed_ahead = [
            group.groups_fed_ahead
            for group in self._buffered_groups_by_id.values()
            if group.state is not _RolloutGroupBufferState.WAITING
        ]
        capacity_used = len(self._buffered_groups_by_id) / max(
            self._max_buffered_rollout_groups, 1
        )
        S = _RolloutGroupBufferState
        return [
            m.Metric(
                "rollout_buffer/num_groups_waiting",
                m.NoReduce(float(states.count(S.WAITING))),
            ),
            m.Metric(
                "rollout_buffer/num_groups_generating",
                m.NoReduce(float(states.count(S.GENERATING))),
            ),
            m.Metric(
                "rollout_buffer/num_groups_generated",
                m.NoReduce(float(states.count(S.GENERATED))),
            ),
            m.Metric("rollout_buffer/capacity_used_frac", m.NoReduce(capacity_used)),
            m.Metric("rollout_buffer/groups_fed_ahead", m.Mean.from_list(fed_ahead)),
            m.Metric("rollout_buffer/groups_fed_ahead", m.Max.from_list(fed_ahead)),
            m.Metric(
                "rollout_buffer/num_groups_evicted_total",
                m.NoReduce(float(self._num_groups_evicted_total)),
            ),
        ]
