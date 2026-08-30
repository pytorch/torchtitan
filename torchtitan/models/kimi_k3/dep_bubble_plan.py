# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Where to run each vision encode so it lands in a pipeline bubble.

Report sec 5.2.3: "The ViT forward passes of the first PP micro-batches are executed
synchronously upfront, the remaining forward passes are scheduled into pipeline
bubbles." This module answers only the scheduling question -- which slot each encode
goes in -- and knows nothing about the model.

Two properties it is built for.

**Every rank derives the same plan.** The plan is a pure function of (pp, vp,
micro-batch count, schedule name, cost ratio). All of those are known to every rank
before the step, so no rank can reach a vision collective the others do not. That is
what makes this safe where issuing collectives off a side stream has to be argued
about: consistency is derived, not assumed.

**A bubble is only usable before its consumer.** Idle time after a micro-batch's
features are needed cannot pay for encoding them, so the budget accumulates along the
rank's action list and is spent in order -- the same walk ``dep_hiding_theory.py``
uses to estimate the hideable share, reused here to decide placement.

The cost ratio ``r`` is in units of one text-stage forward. It is a hand-set
parameter rather than something inferred at runtime: a
plan that depended on a measurement each rank took locally would stop being identical
across ranks.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Placement:
    """Encode ``microbatch`` immediately after the action named by ``anchor``.

    Anchored on the following action's IDENTITY rather than on a slot index,
    because the index does not survive lowering: the runtime iterates
    ``pipeline_order_with_comms``, which inserts SEND/RECV actions and contains no
    idle entries at all, so "slot 37" means nothing there. The relative order of the
    compute actions is the same in both representations, so the first real action
    after the bubble run is a stable name for the position.

    ``anchor`` names the action the runtime fires AFTER, as
    ``(computation_type_name, stage_index, microbatch_index)``.
    ``slot`` is kept for reporting and for the occupancy check against the simulator.
    """

    slot: int
    microbatch: int
    anchor: tuple[str, int, int]


@dataclass(frozen=True)
class BubblePlan:
    """One rank's plan.

    ``upfront`` are the micro-batches whose encodes run synchronously before the
    action loop, which the report prescribes rather than concedes. ``placed`` are the
    ones that fit in a bubble. ``synchronous`` are the ones that fit nowhere and stay
    inline at their consumption point -- they lengthen the step, and counting them is
    how "most of the ViT computation is hidden" gets a number instead of an adjective.
    """

    rank: int
    upfront: tuple[int, ...]
    placed: tuple[Placement, ...]
    synchronous: tuple[int, ...]
    idle_slots: int
    cost_ratio: float
    # Why idle slots placed nothing -- the two reasons call for opposite fixes:
    # starved means the bubbles are too short for this cost ratio, exhausted
    # means they come after every remaining micro-batch was already consumed.
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
    """Walk one rank's action list and place the encodes.

    ``actions`` is the per-rank list the schedule produces, with ``None`` for a slot
    the rank cannot fill because a dependency is unmet. ``upfront`` micro-batches are
    taken out of the walk entirely: the report runs the first ones eagerly.

    An encode is placed at the LAST idle slot whose accumulated budget first covers
    ``cost_ratio``, so it sits as close to its consumer as the budget allows. Placing
    it earlier would work equally well for occupancy and worse for memory, since the
    features stay resident from the moment they are produced.
    """
    if cost_ratio <= 0:
        raise ValueError(f"cost_ratio must be positive, got {cost_ratio}")
    # Where each micro-batch's features are CONSUMED: the vision-owning stage's
    # forward of that micro-batch. A bubble after that point cannot pay for the
    # encode however much budget accumulated -- placing there is a wrong answer.
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
    # The action most recently completed. Placements anchor on THIS: the runtime
    # fires after it returns, at the start of the idle interval. The action
    # AFTER the bubble has no hook on stage 0, whose forward receives nothing.
    prev: tuple[str, int, int] | None = None
    for slot, action in enumerate(actions):
        if action is None:
            budget += 1.0
            idle += 1
            # Keep placing while this bubble's accumulated budget can pay. One
            # encode per slot would bound `placed` by the idle-slot count however
            # small the cost ratio -- and dynamic CP makes the ratio small.
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
                        # Bubbles this late serve nobody: every micro-batch still pending
                        # has already passed its consumption point.
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
        prev = (
            str(getattr(action, "computation_type", "?")),
            int(getattr(action, "stage_index", -1)),
            int(
                action.microbatch_index
                if getattr(action, "microbatch_index", None) is not None
                else -1
            ),
        )
        budget = 0.0  # an executed action ends the idle run
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


def build_plans(
    *,
    pp_size: int,
    vp: int,
    n_microbatches: int,
    cost_ratio: float,
    upfront: int | None = None,
    vision_stage: int = 0,
) -> dict[int, BubblePlan]:
    """Plans for every rank of an Interleaved1F1B schedule.

    ``upfront`` defaults to ``pp_size``: the report's "first PP micro-batches", which
    is also exactly the set that cannot be prefetched, since nothing precedes them.
    """
    from torch.distributed.pipelining.schedules import ScheduleInterleaved1F1B

    if upfront is None:
        upfront = pp_size
    num_stages = pp_size * vp
    sched = ScheduleInterleaved1F1B.__new__(ScheduleInterleaved1F1B)
    # Bypassing __init__ deliberately: it validates and wires real stages, and this is
    # a planning question with no model in it. Everything the action generation reads
    # is set explicitly below so nothing is left implicit.
    sched._num_stages = num_stages
    sched.pp_group_size = pp_size
    sched._n_microbatches = n_microbatches
    sched.n_microbatches = n_microbatches
    sched.stage_index_to_group_rank = {s: s % pp_size for s in range(num_stages)}
    sched.number_of_rounds = max(1, n_microbatches // pp_size)
    sched.microbatches_per_round = n_microbatches // sched.number_of_rounds
    if n_microbatches % sched.number_of_rounds != 0:
        raise ValueError(
            f"Interleaved1F1B needs n_microbatches ({n_microbatches}) to be a "
            f"multiple of the round count ({sched.number_of_rounds})"
        )

    class _FakeStage:
        def __init__(self, index: int) -> None:
            self.stage_index = index
            self.num_stages = num_stages
            self.group_rank = index % pp_size
            self.is_first = index == 0
            self.is_last = index == num_stages - 1

    plans = {}
    for rank in range(pp_size):
        stages = [_FakeStage(s) for s in range(rank, num_stages, pp_size)]
        sched._stages = stages
        sched.n_local_stages = len(stages)
        sched.rank = rank
        actions = sched._calculate_single_rank_operations(rank)
        plans[rank] = plan_for_rank(
            actions,
            rank=rank,
            vision_microbatches=n_microbatches,
            cost_ratio=cost_ratio,
            upfront=upfront,
            vision_stage=vision_stage,
        )
    return plans
