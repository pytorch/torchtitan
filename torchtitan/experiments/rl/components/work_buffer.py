# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run-ahead buffer of RolloutGroupWork shared between the data-input, rollout, and batcher loops.
NOTE: The buffer holds work slots, and not the finalized RolloutGroups necessarily.

"""

import asyncio
import collections
import enum
from dataclasses import dataclass, field

from torchtitan.config import Configurable
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.rollout import RolloutGroup
from torchtitan.observability import structured_logger as sl


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
    run-ahead at `max_active_rollout_groups` active slots; the batcher takes finalized groups by
    windowed FIFO -- greedily within a look-ahead window anchored at the oldest entry, stalling if
    nothing in the window is finalized. The window is sized from `window_lookahead_steps` (extra
    off-policy train-steps a straggler may incur), `num_prompts_per_train_step`, and the head's phase
    (`window_lookahead_steps == 0`, the default, is strict FIFO).

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
        # max_offpolicy_steps=1, num_prompts_per_train_step=2 -> capacity=4
        await buffer.add_work(g0); await buffer.add_work(g1)   # 2/4 active
        await buffer.add_work(g2); await buffer.add_work(g3)   # 4/4 active (cap)
        g0 = await buffer.take_finalized(pending_trainable_count=0)   # g0 leaves the dict; still 4/4 active
        g1 = await buffer.take_finalized(pending_trainable_count=0)   # g1 leaves the dict; still 4/4 active
        slot_task = asyncio.create_task(buffer.wait_for_slot())  # waits: take_finalized did not free a slot
        assert not slot_task.done()
        await buffer.release_active_groups(2, reason="trained")  # trainer pulled -> a slot frees
        assert await slot_task                                   # wait_for_slot now returns
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Buffer scheduling tunables; capacity is passed in by the controller via `build`."""

        window_lookahead_steps: int = 0
        """Windowed-FIFO look-ahead as extra off-policy train-steps (same units as `max_offpolicy_steps`).

        A stuck head may be bypassed by newer FINALIZED groups, but only far enough that the trainer
        consumes at most `s = window_lookahead_steps` extra optimizer steps ahead of that head -- so the
        max policy age is exactly `max_offpolicy_steps + s`, regardless of the head's phase. `0` (the
        default) is strict FIFO: the window is the head only (size 1 entry), so no batch is consumed
        ahead of the head and max age == `max_offpolicy_steps`. Larger values trade a longer
        straggler-staleness tail for less head-of-line stall; they do NOT raise the mean age or the
        capacity."""

    def __init__(
        self,
        config: Config,
        *,
        max_active_rollout_groups: int,
        num_prompts_per_train_step: int,
    ) -> None:
        self._max_active_rollout_groups = max_active_rollout_groups
        if config.window_lookahead_steps < 0:
            raise ValueError(
                f"window_lookahead_steps must be >= 0, got {config.window_lookahead_steps}"
            )
        if num_prompts_per_train_step < 1:
            raise ValueError(
                f"num_prompts_per_train_step must be >= 1, got {num_prompts_per_train_step}"
            )
        # Windowed FIFO look-ahead, sized dynamically per head so the extra staleness is exact. Let
        #   P = num_prompts_per_train_step (groups trained per optimizer step),
        #   K = max_offpolicy_steps        (=> capacity max_active_rollout_groups = (K + 1) * P),
        #   s = self._window_lookahead_steps (extra off-policy train-steps a straggler may incur),
        #   r0 = the head's phase snapshot (trainable groups already accumulated toward the in-progress
        #        batch when this head became the head; 0 <= r0 < P). Batches float -- the batcher packs
        #        once it has P *trainable* groups, so batch boundaries are NOT group_id-aligned, and the
        #        extra steps a stuck head incurs depends on r0.
        # For a head at group_id h with phase r0, the window covers group_ids [h, window_end] where
        #   s == 0 -> window_end = h                             (strict FIFO: head only)
        #   s >= 1 -> window_end = h + (s + 1) * P - r0 - 1      (window size W = (s + 1) * P - r0)
        # Extra steps a bypassing trainer incurs = floor((r0 + (W - 1)) / P) = floor(((s+1)*P - 1)/P) = s,
        # independent of r0. `(s + 1) * P - r0` is the LARGEST W still yielding exactly s (the valid band
        # is [s*P - r0 + 1, (s+1)*P - r0]); picking the max end maximizes scheduling flexibility at no
        # extra staleness. Effect: mean age stays ~= K (typical group consumed as fresh as strict FIFO),
        # max age = K + s exactly. r0 is snapshotted at head-change (not recomputed live) so the window
        # does not shrink from the right as groups are consumed. The consume-time invariant in
        # controller.py checks against exactly K + s.
        self._prompts_per_step = num_prompts_per_train_step
        self._window_lookahead_steps = config.window_lookahead_steps
        # Head-phase snapshot: the window_end is held fixed for a given head, recomputed on head-change.
        self._windowed_head_id: int | None = None
        self._head_pending_snapshot: int = 0
        self._active_rollout_groups = 0
        # metric: Per-flush peak active slots; reset on `.metrics()` call.
        self._active_rollout_groups_peak_since_flush = 0
        self._work_by_group_id: collections.OrderedDict[int, RolloutGroupWork] = (
            collections.OrderedDict()
        )
        # TODO(async-rl): Current we use a condition that alerts ALL rollout workers. There is no need to
        # alert all of them. Consider changing it to an async queue + event.

        # One Condition guards all three waits (slot-free / claimable-WAITING / takeable-FINALIZED):
        # every mutation notify_all()s and waiters re-check their predicate.
        self._condition = asyncio.Condition()
        self._closed = False
        # TODO(async-rl): warm start — admit a small number of groups at first and grow the effective cap as the
        # batcher consumes, so a cold start doesn't fill the whole off-policy window at policy version 0.

    @property
    def window_lookahead_steps(self) -> int:
        """Windowed-FIFO look-ahead in extra off-policy train-steps (>= 0); 0 = strict FIFO."""
        return self._window_lookahead_steps

    def _has_active_slot_available(self) -> bool:
        return self._active_rollout_groups < self._max_active_rollout_groups

    async def wait_for_slot(self) -> bool:
        """Wait until one more rollout group may enter the active off-policy window.

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
    async def take_finalized(self, *, pending_trainable_count: int) -> RolloutGroup | None:
        """Batcher loop: windowed FIFO -- greedily return any FINALIZED group within the head's
        look-ahead window, else stall. `window_lookahead_steps == 0` is strict FIFO (head only).

        The window is anchored at the head (oldest entry) and measured in `group_id` space, which
        `_data_input_loop` assigns as a contiguous, monotonically increasing position. Its right edge
        is sized so that a straggler head is bypassed by at most `s = window_lookahead_steps` extra
        train-steps, exactly, regardless of the head's phase. Let P = num_prompts_per_train_step and
        `r0` = the head-phase snapshot (`pending_trainable_count` captured when this head became the
        head). For head group_id `h`:
            s == 0 -> window_end = h                            (strict FIFO)
            s >= 1 -> window_end = h + (s + 1) * P - r0 - 1     (max window of the exact-s band)
        `r0` is snapshotted at head-change and held until the head is consumed, so the window does not
        shrink from the right as non-head groups are consumed. Consuming a non-head group leaves a gap
        but does NOT move the head, so the window slides right only when the head itself is consumed --
        entries beyond the window stay blocked regardless of completion.

        Args:
            pending_trainable_count: The batcher's in-progress fill `r` (trainable groups accumulated
                toward the next batch, `0 <= r < P`). Snapshotted as `r0` on each head-change.

        Example:
            # s=0: head g0 INFLIGHT, g1 FINALIZED -> WAITS for g0 (strict FIFO)
            # s=1, P=2, r0=0: window_end = g0 + 2*2 - 0 - 1 = g3 -> g1, g2 fetched ahead of stuck g0
            await buffer.take_finalized(pending_trainable_count=0)
        """
        s = self._window_lookahead_steps
        p = self._prompts_per_step
        async with self._condition:
            while True:
                if self._closed:
                    return None
                if self._work_by_group_id:
                    # Head anchors the window; group_id is a contiguous position, so entries beyond
                    # window_end are outside the window even across consumed gaps.
                    head_group_id = next(iter(self._work_by_group_id))
                    # Snapshot the head's phase r0 on head-change; hold window_end fixed until consumed.
                    if head_group_id != self._windowed_head_id:
                        self._windowed_head_id = head_group_id
                        self._head_pending_snapshot = pending_trainable_count
                    if s == 0:
                        window_end = head_group_id  # strict FIFO: head only
                    else:
                        window_end = (
                            head_group_id + (s + 1) * p - self._head_pending_snapshot - 1
                        )
                    for group_id, work in self._work_by_group_id.items():
                        if group_id > window_end:
                            break  # beyond the window -> blocked regardless of state
                        if work.state is _RolloutGroupWorkState.FINALIZED:
                            del self._work_by_group_id[group_id]
                            self._condition.notify_all()
                            return work.rollout_group
                await self._condition.wait()  # nothing FINALIZED in window -> STALL

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
