# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Active work buffer shared between the data-input, rollout, and batcher loops.
NOTE: The buffer holds work slots, and not the finalized RolloutGroups necessarily.

"""

import asyncio
import collections
import enum
import math
from dataclasses import dataclass, field

from torchtitan.config import Configurable
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.rollout import RolloutGroup
from torchtitan.observability import structured_logger as sl


def derive_window_size(
    *,
    num_prompts_per_train_step: int,
    target_offpolicy_steps: int,
    window_fifo_fraction: float,
) -> int:
    """Derive the fixed FIFO look-ahead window from the configured fraction.

    Symbols:
        ``P``: prompts per train step (``num_prompts_per_train_step``).
        ``S``: target mean offpolicy steps (``target_offpolicy_steps``).
        ``f``: fraction of the active buffer used as the FIFO window
            (``window_fifo_fraction``).
        ``B``: active buffer size in prompt groups, ``B = (S + 1) * P``.
        ``W``: FIFO look-ahead window size, ``W = floor(f * B)``.

    This window implies a worst case consume-time offpolicy bound of
    ``(B + W - 2) // P``.
    """
    max_active_rollout_groups = (
        target_offpolicy_steps + 1
    ) * num_prompts_per_train_step
    return math.floor(window_fifo_fraction * max_active_rollout_groups)


class _RolloutGroupWorkState(enum.Enum):
    """Where a RolloutGroupWork is in the WAITING -> INFLIGHT -> FINALIZED lifecycle."""

    WAITING = "waiting"
    INFLIGHT = "inflight"
    FINALIZED = "finalized"


@dataclass(slots=True)
class RolloutGroupWork:
    """One prompt group's work, tracked through _RolloutGroupWorkState.

    The input loop sets `group_id` + `sample`; the buffer owns `state` and `rollout_group`
    (`init=False`, so the input loop can't set them).
    """

    group_id: int
    sample: object
    """Data input produced by `rollouter.get_training_sample()`;
    passed unchanged to the env in `rollouter.run_group_rollouts`."""
    state: _RolloutGroupWorkState = field(
        default=_RolloutGroupWorkState.WAITING, init=False
    )
    rollout_group: RolloutGroup | None = field(
        default=None, init=False
    )  # set once FINALIZED
    # TODO(async-rl): emit JSON lifecycle logging per RolloutGroupWork keyed by group_id:
    # admitted/claimed/finalized/batched/trained/dropped timestamps + policy version at admission and
    # at trainer consumption, for faithful end-to-end visibility.


class RolloutGroupWorkBuffer(Configurable):
    """Run-ahead buffer of RolloutGroupWork shared by the data-input, rollout, and batcher loops.

    Each entry is a RolloutGroupWork moving WAITING -> INFLIGHT -> FINALIZED. An active-slot budget caps
    the pipeline at `max_active_rollout_groups` active slots; the batcher takes finalized groups within
    a fixed look-ahead window anchored at the oldest entry. The controller defaults
    to strict FIFO (`window_size=1`) unless `window_fifo_fraction` is set.

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
        self, config: Config, *, max_active_rollout_groups: int, window_size: int = 1
    ) -> None:
        if max_active_rollout_groups < 1:
            raise ValueError(
                f"max_active_rollout_groups must be >= 1, got {max_active_rollout_groups}"
            )
        if not 1 <= window_size <= max_active_rollout_groups:
            raise ValueError(
                "window_size must be between 1 and max_active_rollout_groups, got "
                f"window_size={window_size}, "
                f"max_active_rollout_groups={max_active_rollout_groups}"
            )
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
        # batcher consumes, so a cold start doesn't fill the whole FIFO window at policy version 0.

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

    async def finalize_work(self, rollout_group: RolloutGroup) -> None:
        """Rollout loop: store the produced RolloutGroup on its work entry (INFLIGHT -> FINALIZED) and wake the batcher."""
        async with self._condition:
            work = self._work_by_group_id.get(rollout_group.group_id)
            if work is None:
                # run()'s shutdown called close(); drop the result.
                return
            work.rollout_group = rollout_group
            work.state = _RolloutGroupWorkState.FINALIZED
            self._condition.notify_all()

    @sl.log_trace_span("take_finalized")
    async def take_finalized(self) -> RolloutGroup | None:
        """Batcher loop: return the oldest FINALIZED group inside the anchored window.

        The window covers group ids ``[head, head + window_size - 1]``. Entries outside the
        window stay blocked even if they are finalized, so taking non-head groups does not slide
        the window. The controller defaults to strict FIFO (`window_size=1`)
        unless `window_fifo_fraction` is set.

        Example:
            # window_size=1: head g0 still INFLIGHT, g1 FINALIZED -> waits for g0
            await buffer.take_finalized()
        """
        async with self._condition:
            while True:
                if self._closed:
                    return None
                if self._work_by_group_id:
                    head_group_id = next(iter(self._work_by_group_id))
                    window_end = head_group_id + self._window_size - 1
                    for group_id, work in self._work_by_group_id.items():
                        if group_id > window_end:
                            break
                        if work.state is not _RolloutGroupWorkState.FINALIZED:
                            continue
                        del self._work_by_group_id[group_id]
                        self._condition.notify_all()
                        return work.rollout_group
                await self._condition.wait()  # nothing finalized inside the window -> stall

    async def release_active_groups(self, count: int, *, reason: str) -> None:
        """Free active slots: the trainer releases trained slots after its weight pull; the batcher
        releases untrainable/filtered slots immediately.

        Args:
            count:  Number of rollout groups leaving the active window.
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
        """Trainer loop: point-in-time buffer gauges for this step; resets the per-flush peak."""
        states = [work.state for work in self._work_by_group_id.values()]
        state_enum = _RolloutGroupWorkState
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
