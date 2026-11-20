# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Active work buffer shared between the data-input, rollout, and batcher loops.
NOTE: The buffer holds work slots, and not the finished rollout groups necessarily.

"""

from __future__ import annotations

import asyncio
import collections
import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from torchtitan.config import Configurable
from torchtitan.observability import structured_logger as sl
from torchtitan.rl.observability import metrics as m

if TYPE_CHECKING:
    from torchtitan.rl.rollout.finished_group import FinishedRolloutGroup


class _RolloutGroupWorkState(enum.Enum):
    """Where a RolloutGroupWork is in the WAITING -> CLAIMED -> FINALIZED lifecycle."""

    WAITING = "waiting"
    CLAIMED = "claimed"
    FINALIZED = "finalized"


@dataclass(slots=True)
class RolloutGroupWork:
    """One prompt group's work, tracked through _RolloutGroupWorkState.

    The input loop sets `group_id` + `sample`; the buffer owns `state` and `finished_group`
    (`init=False`, so the input loop can't set them).
    """

    group_id: int
    sample: object
    """Indexed data input produced by the rollouter;
    passed unchanged to the env in `rollouter.run_group_rollouts`."""
    state: _RolloutGroupWorkState = field(
        default=_RolloutGroupWorkState.WAITING, init=False
    )
    finished_group: FinishedRolloutGroup | None = field(
        default=None, init=False
    )  # set once FINALIZED
    # TODO(async-rl): emit JSON lifecycle logging per RolloutGroupWork keyed by group_id:
    # admitted/claimed/finalized/batched/trained/dropped timestamps + policy version at admission and
    # at trainer consumption, for faithful end-to-end visibility.


class RolloutGroupWorkBuffer(Configurable):
    """Buffer of `RolloutGroupWork` shared between the data-input, rollout, and batcher loops.

    Each entry is a RolloutGroupWork moving WAITING -> CLAIMED -> FINALIZED. Active slots, keyed by
    group id, cap the pipeline at `max_active_rollout_groups` groups. With `window_size=None`, the
    batcher takes the oldest finalized group anywhere in the buffer. A finite `window_size` restricts it
    to a look-ahead range anchored at the oldest entry.

    For details on the buffer's callers, check the diagram in the controller.py file.

    NOTE: a group keeps its active slot after it is FINALIZED and taken by the batcher. Only
    `release_slots(group_ids)` frees it: the trainer releases the groups it trained, and the batcher
    loop releases a group with no trainable samples. This way the entire pipeline (buffer+queue+training)
    never holds more than `max_active_rollout_groups` groups. Otherwise, we would produce born-stale
    examples.

    Entry lifecycle vs active slot:
        entry:        WAITING -> CLAIMED -> FINALIZED -> removed by take_finalized()
        active slot:  taken by add_work() .............. freed by release_slots([group_id])

    Each wait has its own condition, and each event wakes only the waiters it can unblock, so a new
    group wakes one of the (S+1) x P rollout loops instead of all of them:
        add_work(work)     -> one claim_next() waiter; claims pop a FIFO of WAITING group ids
        finalize_work()    -> one take_finalized() waiter
        release_slots(ids) -> up to len(ids) wait_for_slot() waiters
        close()            -> every waiter

    Example:
        # target_offpolicy_steps=1, num_prompts_per_train_step=2 -> capacity=4
        await buffer.add_work(g0); await buffer.add_work(g1)   # 2/4 active
        await buffer.add_work(g2); await buffer.add_work(g3)   # 4/4 active (cap)
        g0 = await buffer.take_finalized()                     # g0 leaves the dict; still 4/4 active
        g1 = await buffer.take_finalized()                     # g1 leaves the dict; still 4/4 active
        slot_task = asyncio.create_task(buffer.wait_for_slot())  # waits: take_finalized did not free a slot
        assert not slot_task.done()
        await buffer.release_slots([0, 1], reason="trained")   # trainer pulled -> two slots free
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
        # Group ids holding an active slot, from add_work() until release_slots().
        self._active_group_ids: set[int] = set()
        # metric: Per-flush peak active slots; reset on `.metrics()` call.
        self._active_rollout_groups_peak_since_flush = 0
        self._work_by_group_id: collections.OrderedDict[
            int, RolloutGroupWork
        ] = collections.OrderedDict()
        # The WAITING entries' group ids, oldest first.
        self._waiting_group_ids: collections.deque[int] = collections.deque()

        # One lock guards the state above; the three waits each have a condition on it.
        self._lock = asyncio.Lock()
        self._slot_freed = asyncio.Condition(self._lock)
        self._work_added = asyncio.Condition(self._lock)
        self._work_finalized = asyncio.Condition(self._lock)
        self._closed = False
        # TODO(async-rl): warm start — admit a small number of groups at first and grow the effective cap as the
        # batcher consumes, so a cold start doesn't fill the whole off-policy window at policy version 0.

    def _has_active_slot_available(self) -> bool:
        return len(self._active_group_ids) < self._max_active_rollout_groups

    async def wait_for_slot(self) -> bool:
        """Wait until one more rollout group may enter the active buffer.

        Example:
            # False means the buffer was closed, so the data input loop exits.
            group_index = 0
            while await buffer.wait_for_slot():
                await buffer.add_work(RolloutGroupWork(group_id=group_index, sample=sample))
                group_index += 1
        """
        async with self._lock:
            await self._slot_freed.wait_for(
                lambda: self._closed or self._has_active_slot_available()
            )
            return not self._closed

    async def add_work(self, work: RolloutGroupWork) -> None:
        """Admit one rollout group as WAITING under an active slot keyed by its group id."""
        async with self._lock:
            if not self._has_active_slot_available():
                raise RuntimeError(
                    "RolloutGroupWorkBuffer.add_work called without an active slot"
                )
            # A repeated id would share one slot, and the buffer would admit one group too many.
            if work.group_id in self._active_group_ids:
                raise RuntimeError(
                    f"Group {work.group_id} already holds an active slot"
                )
            self._active_group_ids.add(work.group_id)
            self._active_rollout_groups_peak_since_flush = max(
                self._active_rollout_groups_peak_since_flush,
                len(self._active_group_ids),
            )
            self._work_by_group_id[work.group_id] = work
            self._waiting_group_ids.append(work.group_id)
            self._work_added.notify()

    async def claim_next(self) -> RolloutGroupWork | None:
        """Rollout loop: claim the oldest WAITING group (WAITING -> CLAIMED). None once closed."""
        async with self._lock:
            while True:
                if self._closed:
                    return None
                if self._waiting_group_ids:
                    work = self._work_by_group_id[self._waiting_group_ids.popleft()]
                    work.state = _RolloutGroupWorkState.CLAIMED
                    return work
                await self._work_added.wait()

    async def finalize_work(self, finished_group: FinishedRolloutGroup) -> None:
        """Rollout loop: store the finished group on its work entry (CLAIMED -> FINALIZED) and wake the batcher."""
        async with self._lock:
            work = self._work_by_group_id.get(finished_group.group_id)
            if work is None:
                # run()'s shutdown called close(); drop the result.
                return
            work.finished_group = finished_group
            work.state = _RolloutGroupWorkState.FINALIZED
            self._work_finalized.notify()

    @sl.log_trace_span("take_finalized")
    async def take_finalized(self) -> FinishedRolloutGroup | None:
        """Batcher loop: return the oldest FINALIZED group the window allows.

        Cases:
            window_size is None: every finalized group is eligible; the oldest is returned.
            window_size is W:    only group ids ``[head, head + W - 1]`` are eligible, where head is
                                 the oldest group still in the buffer. Finalized groups past the
                                 window stay blocked, and taking a non-head group does not move the
                                 window past the head.
            window_size is 1:    only the head is eligible: strict FIFO.

        This anchored window is inspired by MiniMax's rollout scheduling; see
        Section 6.2.4 of https://arxiv.org/pdf/2605.26494.

        The batcher loop is the only caller: `finalize_work` wakes one waiter, and taking a group
        wakes none.

        Example:
            # window_size=3: g0 CLAIMED, g1 WAITING, g2 and g3 FINALIZED
            group = await buffer.take_finalized()
            assert group.group_id == 2  # g2 is inside [g0, g2].
            # g3 remains blocked. With window_size=None, g3 would be taken next.
        """
        async with self._lock:
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
                        return work.finished_group
                await self._work_finalized.wait()  # nothing finalized inside the window -> stall

    async def release_slots(self, group_ids: list[int], *, reason: str) -> None:
        """Free the active slots of `group_ids`: the trainer releases the groups it trained after
        its model state dict pull; the batcher loop releases a group with no trainable samples
        immediately.

        Args:
            group_ids: Groups leaving the active slots. Each must hold one; naming a group that
                does not is a caller bug and raises.
            reason: Metric suffix such as `"trained"` or `"untrainable_group"`.

        Example:
            # trainer pulled weights after a step over groups 8..15 -> free their 8 slots
            await buffer.release_slots(list(range(8, 16)), reason="trained")
            # batcher loop got no training samples from group 17 -> free its slot
            await buffer.release_slots([17], reason="untrainable_group")
        """
        async with self._lock:
            released_group_ids = set(group_ids)
            not_active = released_group_ids - self._active_group_ids
            if not_active or len(released_group_ids) < len(group_ids):
                repeated = sorted(
                    group_id
                    for group_id, count in collections.Counter(group_ids).items()
                    if count > 1
                )
                raise RuntimeError(
                    f"release_slots({group_ids}) must name each group once, and only groups that "
                    f"hold an active slot; repeated: {repeated}, without a slot: {sorted(not_active)}"
                )
            self._active_group_ids -= released_group_ids
            sl.log_trace_scalar(
                {f"rollout_buffer/released/{reason}": float(len(group_ids))}
            )
            self._slot_freed.notify(len(group_ids))

    async def close(self) -> None:
        """run() shutdown calls this once. Sets `_closed`, drops buffered work, and wakes every waiter.

        After this, every waiter unblocks and exits its loop: wait_for_slot() returns False, and
        claim_next()/take_finalized() return None. Active slots stay, so a model state dict pull
        that ends after close can still release its groups.
        """
        async with self._lock:
            self._closed = True
            self._work_by_group_id.clear()
            self._waiting_group_ids.clear()
            for condition in (self._slot_freed, self._work_added, self._work_finalized):
                condition.notify_all()

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
                "rollout_buffer/num_groups_claimed",
                m.NoReduce(float(states.count(state_enum.CLAIMED))),
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
                    float(self._max_active_rollout_groups - len(self._active_group_ids))
                ),
            ),
        ]
        # Next interval starts from the current gauge, not 0: slots stay occupied across a flush.
        self._active_rollout_groups_peak_since_flush = len(self._active_group_ids)
        return out
