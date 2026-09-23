# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which idle interval of a rank's action list each vision encode runs in.

The plan is a pure function of the rank's action list, the micro-batch count and
the cost ratio, so every rank derives the same placements and no rank reaches a
vision collective the others do not.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Placement:
    """Encode ``microbatch`` right after the action named by ``anchor``."""

    # The anchor names an action rather than a slot index because the index does
    # not survive lowering: the runtime walks pipeline_order_with_comms, which
    # inserts sends and receives and holds no idle entries at all.
    slot: int
    microbatch: int
    anchor: tuple[str, int, int]


@dataclass(frozen=True)
class BubblePlan:
    """One rank's placements, and what fit nowhere."""

    rank: int
    upfront: tuple[int, ...]
    placed: tuple[Placement, ...]
    synchronous: tuple[int, ...]
    idle_slots: int
    cost_ratio: float
    # Why an idle slot placed nothing: starved means the bubbles are too short for
    # this cost ratio, exhausted means they come after every pending micro-batch
    # has already been consumed. The two call for opposite fixes.
    slots_starved: int = 0
    slots_exhausted: int = 0

    @property
    def hidden_share(self) -> float:
        total = len(self.upfront) + len(self.placed) + len(self.synchronous)
        return len(self.placed) / total if total else 0.0


def plan_for_rank(
    actions,
    *,
    rank: int,
    vision_microbatches: int,
    cost_ratio: float,
    upfront: int,
    vision_stage: int = 0,
) -> BubblePlan:
    """Walk one rank's action list, whose None entries are its idle slots, and place the encodes.

    ``cost_ratio`` is the encode's cost in units of one text-stage action. An
    encode is placed at the last idle slot whose accumulated budget first covers
    it, so it sits as close to its consumer as the budget allows and its features
    stay resident for as short a time.
    """
    if cost_ratio <= 0:
        raise ValueError(f"cost_ratio must be positive, got {cost_ratio}")
    # An idle slot after a micro-batch's features are consumed cannot pay for
    # encoding them, however much budget has accumulated by then.
    consume_slot: dict[int, int] = {}
    for slot, action in enumerate(actions):
        if action is None:
            continue
        mb = getattr(action, "microbatch_index", None)
        if mb is None or "FORWARD" not in str(getattr(action, "computation_type", "")):
            continue
        if int(getattr(action, "stage_index", -1)) != vision_stage:
            continue
        consume_slot.setdefault(int(mb), slot)

    pending = [m for m in range(vision_microbatches) if m >= upfront]
    placed: list[Placement] = []
    budget = 0.0
    idle = 0
    slots_starved = 0
    slots_exhausted = 0
    prev: tuple[str, int, int] | None = None
    for slot, action in enumerate(actions):
        if action is None:
            budget += 1.0
            idle += 1
            # One encode per idle slot would bound the placements by the slot
            # count however small the cost ratio, and dynamic CP makes it small
            # by dividing the per-rank encoder cost before this sees it.
            if prev is not None:
                while budget >= cost_ratio:
                    k = next(
                        (
                            i
                            for i, mb in enumerate(pending)
                            if consume_slot.get(mb, 1 << 30) > slot
                        ),
                        None,
                    )
                    if k is None:
                        slots_exhausted += 1
                        break
                    budget -= cost_ratio
                    placed.append(
                        Placement(slot=slot, microbatch=pending.pop(k), anchor=prev)
                    )
                else:
                    if not placed or placed[-1].slot != slot:
                        slots_starved += 1
            continue
        # The placement anchors on the action the rank has just finished, which
        # is where the idle interval starts and is reachable without a receive.
        prev = (
            str(getattr(action, "computation_type", "?")),
            int(getattr(action, "stage_index", -1)),
            int(
                action.microbatch_index
                if getattr(action, "microbatch_index", None) is not None
                else -1
            ),
        )
        budget = 0.0
    return BubblePlan(
        rank=rank,
        upfront=tuple(range(min(upfront, vision_microbatches))),
        placed=tuple(placed),
        synchronous=tuple(pending),
        idle_slots=idle,
        cost_ratio=cost_ratio,
        slots_starved=slots_starved,
        slots_exhausted=slots_exhausted,
    )
