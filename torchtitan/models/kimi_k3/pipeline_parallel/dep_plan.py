# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which pipeline rank runs each micro-batch's vision work in a step, when, and
where the features and gradients change ranks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

Anchor = tuple[str, int, int]
Hook = tuple[str, Anchor]
Transfer = tuple[str, int, int]

START: Anchor = ("START", -1, -1)
END: Anchor = ("END", -1, -1)
STEP_START: Hook = ("after", START)
STEP_END: Hook = ("before", END)

# The tower's recompute plus its backward, in units of its forward.
_BACKWARD_COST = 3.0
# A transfer between two ranks, in units of one text-stage forward.
_TRANSFER = 1.0
# A text action's duration in units of its stage's forward.
_ACTION_COST = {"F": 1.0, "B": 2.0, "I": 1.0, "W": 1.0}

_KINDS = {
    "FORWARD": "F",
    "FULL_BACKWARD": "B",
    "BACKWARD_INPUT": "I",
    "BACKWARD_WEIGHT": "W",
}


def anchor_of(action: Any) -> Anchor:
    """The key a stage reports when it starts or finishes ``action``."""
    kind = getattr(action.computation_type, "name", str(action.computation_type))
    if kind not in _KINDS:
        raise ValueError(f"vision_dep cannot place work around a {kind} action.")
    return (_KINDS[kind], int(action.stage_index), int(action.microbatch_index))


@dataclass(frozen=True)
class DepPlan:
    """The rank and the point at which each micro-batch is encoded and backpropagated,
    and the points at which its features and gradient change ranks.

    ``prologue`` encodes run before the schedule, ``epilogue`` backwards after it, and
    ``anchored`` work after the named action (``START``: before the rank's first
    one). ``posts`` holds the sends and receives a rank issues at each hook: both
    ends of a transfer post it at the same slot boundary of the schedule, which each
    reaches without the other's later work.
    """

    encode_rank: dict[int, int]
    backward_rank: dict[int, int]
    prologue: dict[int, tuple[int, ...]]
    epilogue: dict[int, tuple[int, ...]]
    anchored: dict[int, dict[Anchor, tuple[tuple[str, int], ...]]]
    posts: dict[int, dict[Hook, tuple[Transfer, ...]]]
    placed: dict[tuple[str, int], tuple[int, float, float]] = field(
        default_factory=dict
    )


@dataclass
class _Run:
    start: int
    end: int
    anchor: Anchor
    begin: float
    stop: float
    cursor: float = field(init=False)

    def __post_init__(self) -> None:
        self.cursor = self.begin


def _slot_times(
    pipeline_order: Mapping[int, Sequence[Any]], step_end: int
) -> list[float]:
    """When each slot of the action order starts, a slot lasting as long as its longest action."""
    times = [0.0]
    for slot in range(step_end):
        costs = [
            _ACTION_COST[anchor_of(actions[slot])[0]]
            for actions in pipeline_order.values()
            if slot < len(actions) and actions[slot] is not None
        ]
        times.append(times[-1] + max(costs, default=0.0))
    return times


def _idle_runs(
    pipeline_order: Mapping[int, Sequence[Any]],
) -> tuple[dict[int, list[_Run]], int, list[float]]:
    last = {
        rank: max((i for i, a in enumerate(actions) if a is not None), default=-1)
        for rank, actions in pipeline_order.items()
    }
    step_end = max(last.values()) + 1
    times = _slot_times(pipeline_order, step_end)
    runs: dict[int, list[_Run]] = {}
    for rank, actions in pipeline_order.items():
        rank_runs: list[_Run] = []
        anchor, idle_from = START, None
        for slot, action in enumerate(actions[: last[rank] + 1]):
            if action is None:
                if idle_from is None:
                    idle_from = slot
                continue
            if idle_from is not None:
                rank_runs.append(
                    _Run(idle_from, slot, anchor, times[idle_from], times[slot])
                )
                idle_from = None
            anchor = anchor_of(action)
        if last[rank] + 1 < step_end:
            rank_runs.append(
                _Run(
                    last[rank] + 1,
                    step_end,
                    anchor,
                    times[last[rank] + 1],
                    times[step_end],
                )
            )
        runs[rank] = rank_runs
    return runs, step_end, times


def _stage0_slots(actions: Sequence[Any]) -> tuple[dict[int, int], dict[int, int]]:
    consume: dict[int, int] = {}
    ready: dict[int, int] = {}
    for slot, action in enumerate(actions):
        if action is None or int(action.stage_index) != 0:
            continue
        kind, _, mb = anchor_of(action)
        if kind == "F":
            consume.setdefault(mb, slot)
        elif kind in ("B", "I"):
            ready.setdefault(mb, slot + 1)
    return consume, ready


def _hook_at(actions: Sequence[Any] | None, boundary: int, step_end: int) -> Hook:
    """The hook at which a rank running ``actions`` stands on slot ``boundary``."""
    if boundary >= step_end:
        return STEP_END
    if actions is None or boundary <= 0:
        return STEP_START
    if boundary < len(actions) and actions[boundary] is not None:
        return ("before", anchor_of(actions[boundary]))
    for slot in range(min(boundary, len(actions)) - 1, -1, -1):
        if actions[slot] is not None:
            return ("after", anchor_of(actions[slot]))
    return STEP_START


def _encode_spot(
    runs: dict[int, list[_Run]],
    duration: float,
    load: list[float],
    *,
    due: float,
    stage0_rank: int,
) -> tuple[int, _Run, float, float] | None:
    best = None
    for rank, rank_runs in runs.items():
        for run in rank_runs:
            start, end = run.cursor, run.cursor + duration
            # A remote encode's features leave the rank when its run ends.
            arrival = end if rank == stage0_rank else run.stop + _TRANSFER
            if arrival > due:
                break
            if end <= run.stop:
                key = (load[rank], end, rank)
                if best is None or key < best[0]:
                    best = (key, rank, run, start, end)
                break
    return None if best is None else best[1:]


def _backward_spot(
    runs: dict[int, list[_Run]],
    duration: float,
    load: list[float],
    *,
    ready: float,
    stage0_rank: int,
) -> tuple[int, _Run, float, float] | None:
    best = None
    for rank, rank_runs in runs.items():
        transfer = 0.0 if rank == stage0_rank else _TRANSFER
        for run in rank_runs:
            # The gradient changes ranks at the later of its readiness and the run's start.
            start = max(run.cursor, max(run.begin, ready) + transfer)
            end = start + duration
            if end <= run.stop:
                key = (load[rank], end, rank)
                if best is None or key < best[0]:
                    best = (key, rank, run, start, end)
                break
    return None if best is None else best[1:]


def plan_dep(
    loads: Mapping[int, int],
    *,
    num_microbatches: int,
    num_ranks: int,
    stage0_rank: int,
    trainable: bool,
    pipeline_order: Mapping[int, Sequence[Any]] | None = None,
    cost_ratio: float = 1.0,
) -> DepPlan:
    """Place the encode and the backward of every micro-batch in ``loads``.

    ``loads`` maps each micro-batch that carries images to its patch count. Without
    ``pipeline_order`` every encode runs before the schedule and every backward after
    it, balanced across the ranks by patch count, and the features and gradients
    change ranks at those two points. With it, the first ``num_ranks`` micro-batches
    stage 0 consumes are encoded before the schedule and the others in idle slots
    ahead of the forward that reads them; a backward runs in an idle slot after its
    gradient reaches the rank. What fits no idle slot joins the balanced prologue or
    epilogue. ``cost_ratio`` is an average micro-batch's encode in units of one
    text-stage forward.
    """
    if cost_ratio <= 0:
        raise ValueError(f"bubble_cost_ratio must be positive, got {cost_ratio}.")
    ranks = range(num_ranks)
    mbs = sorted(loads)
    if pipeline_order is None:
        consume = {m: m for m in range(num_microbatches)}
        ready = {m: m for m in range(num_microbatches)}
        runs: dict[int, list[_Run]] = {r: [] for r in ranks}
        step_end = 1
        times = [float(t) for t in range(num_microbatches + 1)]
    else:
        if sorted(pipeline_order) != list(ranks):
            raise ValueError(
                f"pipeline_order covers ranks {sorted(pipeline_order)}, "
                f"expected 0..{num_ranks - 1}."
            )
        consume, ready = _stage0_slots(pipeline_order[stage0_rank])
        runs, step_end, times = _idle_runs(pipeline_order)
    missing = [m for m in mbs if m not in consume or (trainable and m not in ready)]
    if missing:
        raise ValueError(
            f"stage 0 runs no forward or backward of micro-batches {missing}."
        )

    mean = sum(loads[m] for m in mbs) / len(mbs) if mbs else 1.0
    cost = {m: cost_ratio * loads[m] / mean for m in mbs}
    load = [0.0] * num_ranks
    placed: dict[tuple[str, int], tuple[int, float, float]] = {}
    runs_of: dict[tuple[str, int], _Run] = {}
    encode_rank: dict[int, int] = {}
    backward_rank: dict[int, int] = {}

    by_consume = sorted(mbs, key=lambda m: consume[m])
    prologue: dict[int, list[int]] = {r: [] for r in ranks}
    prologue_busy = [0.0] * num_ranks

    def to_prologue(batch: list[int]) -> None:
        for m in sorted(batch, key=lambda mb: (-cost[mb], consume[mb])):
            rank = min(ranks, key=lambda r: (prologue_busy[r], r))
            prologue_busy[rank] += cost[m]
            load[rank] += cost[m]
            prologue[rank].append(m)
            encode_rank[m] = rank

    upfront = by_consume if pipeline_order is None else by_consume[:num_ranks]
    to_prologue(upfront)
    unplaced: list[int] = []
    for m in by_consume[len(upfront) :]:
        spot = _encode_spot(
            runs, cost[m], load, due=times[consume[m]], stage0_rank=stage0_rank
        )
        if spot is None:
            unplaced.append(m)
            continue
        rank, run, start, end = spot
        run.cursor = end
        load[rank] += cost[m]
        encode_rank[m] = rank
        placed[("encode", m)] = (rank, start, end)
        runs_of[("encode", m)] = run
    to_prologue(unplaced)
    for rank in ranks:
        prologue[rank].sort(key=lambda m: consume[m])

    epilogue: dict[int, list[int]] = {r: [] for r in ranks}
    by_ready = sorted(mbs, key=lambda m: ready[m]) if trainable else []
    epilogue_busy = [0.0] * num_ranks
    leftover: list[int] = []
    for m in by_ready:
        spot = _backward_spot(
            runs,
            _BACKWARD_COST * cost[m],
            load,
            ready=times[ready[m]],
            stage0_rank=stage0_rank,
        )
        if spot is None:
            leftover.append(m)
            continue
        rank, run, start, end = spot
        run.cursor = end
        load[rank] += _BACKWARD_COST * cost[m]
        backward_rank[m] = rank
        placed[("backward", m)] = (rank, start, end)
        runs_of[("backward", m)] = run
    for m in sorted(leftover, key=lambda mb: (-cost[mb], ready[mb])):
        rank = min(ranks, key=lambda r: (epilogue_busy[r], r))
        epilogue_busy[rank] += _BACKWARD_COST * cost[m]
        epilogue[rank].append(m)
        backward_rank[m] = rank
    for rank in ranks:
        epilogue[rank].sort(key=lambda m: ready[m])

    anchored: dict[int, dict[Anchor, list[tuple[str, int]]]] = {r: {} for r in ranks}
    for work, (rank, _, _) in sorted(
        placed.items(), key=lambda item: (item[1][0], item[1][1])
    ):
        anchored[rank].setdefault(runs_of[work].anchor, []).append(work)

    transfers: list[tuple[int, int, int, int, int]] = []
    for m in mbs:
        if encode_rank[m] != stage0_rank:
            run = runs_of.get(("encode", m))
            boundary = 0 if run is None else run.end
            transfers.append((boundary, 0, m, encode_rank[m], stage0_rank))
    for m in by_ready:
        if backward_rank[m] != stage0_rank:
            run = runs_of.get(("backward", m))
            boundary = step_end if run is None else max(run.start, ready[m])
            transfers.append((boundary, 1, m, stage0_rank, backward_rank[m]))

    posts: dict[int, dict[Hook, list[Transfer]]] = {r: {} for r in ranks}
    for boundary, what, m, sender, receiver in sorted(transfers):
        name = ("feature", "gradient")[what]
        for rank, kind, peer in (
            (sender, f"send_{name}", receiver),
            (receiver, f"recv_{name}", sender),
        ):
            actions = None if pipeline_order is None else pipeline_order[rank]
            hook = _hook_at(actions, boundary, step_end)
            posts[rank].setdefault(hook, []).append((kind, m, peer))

    return DepPlan(
        encode_rank=encode_rank,
        backward_rank=backward_rank,
        prologue={r: tuple(prologue[r]) for r in ranks},
        epilogue={r: tuple(epilogue[r]) for r in ranks},
        anchored={
            r: {a: tuple(items) for a, items in anchored[r].items()} for r in ranks
        },
        posts={r: {h: tuple(ts) for h, ts in posts[r].items()} for r in ranks},
        placed=placed,
    )
