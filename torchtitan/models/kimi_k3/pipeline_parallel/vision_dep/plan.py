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
        anchor = START
        idle_from: int | None = None
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
    best: tuple[tuple[float, float, int], int, _Run, float, float] | None = None
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
    best: tuple[tuple[float, float, int], int, _Run, float, float] | None = None
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


class VisionDepPlan:
    """Where and when each micro-batch is encoded and backpropagated, and where its
    features and gradient change ranks.

    ``loads`` maps each micro-batch that carries images to its patch count, and
    ``cost_ratio`` is an average micro-batch's encode in units of one text-stage
    forward. Both ends of a transfer post it at the same slot boundary of the
    schedule, which each reaches without the other's later work.
    """

    def __init__(
        self,
        loads: Mapping[int, int],
        *,
        num_microbatches: int,
        num_ranks: int,
        stage0_rank: int,
        trainable: bool,
        pipeline_order: Mapping[int, Sequence[Any]] | None = None,
        cost_ratio: float = 1.0,
    ) -> None:
        if cost_ratio <= 0:
            raise ValueError(f"bubble_cost_ratio must be positive, got {cost_ratio}.")
        self._ranks = range(num_ranks)
        self._stage0_rank = stage0_rank
        self._pipeline_order = pipeline_order
        if pipeline_order is None:
            self._consume = {m: m for m in range(num_microbatches)}
            self._ready = {m: m for m in range(num_microbatches)}
            self._runs: dict[int, list[_Run]] = {r: [] for r in self._ranks}
            self._step_end = 1
            self._times = [float(t) for t in range(num_microbatches + 1)]
        else:
            if sorted(pipeline_order) != list(self._ranks):
                raise ValueError(
                    f"pipeline_order covers ranks {sorted(pipeline_order)}, "
                    f"expected 0..{num_ranks - 1}."
                )
            self._consume, self._ready = _stage0_slots(pipeline_order[stage0_rank])
            self._runs, self._step_end, self._times = _idle_runs(pipeline_order)
        self._mbs = sorted(loads)
        missing = [
            m
            for m in self._mbs
            if m not in self._consume or (trainable and m not in self._ready)
        ]
        if missing:
            raise ValueError(
                f"stage 0 runs no forward or backward of micro-batches {missing}."
            )
        mean = sum(loads[m] for m in self._mbs) / len(self._mbs) if self._mbs else 1.0
        self._cost = {m: cost_ratio * loads[m] / mean for m in self._mbs}
        self._backward_mbs = (
            sorted(self._mbs, key=lambda m: self._ready[m]) if trainable else []
        )
        self._load = [0.0] * num_ranks
        self._run_of: dict[tuple[str, int], _Run] = {}
        self.encode_rank: dict[int, int] = {}
        self.backward_rank: dict[int, int] = {}
        self.prologue: dict[int, tuple[int, ...]] = {}
        self.epilogue: dict[int, tuple[int, ...]] = {}
        self.anchored: dict[int, dict[Anchor, tuple[tuple[str, int], ...]]] = {}
        self.posts: dict[int, dict[Hook, tuple[Transfer, ...]]] = {}
        self.placed: dict[tuple[str, int], tuple[int, float, float]] = {}
        self._build()

    def _build(self) -> None:
        self._place_encodes()
        self._place_backwards()
        self._anchor_work()
        self._hook_transfers()

    def _place(
        self,
        work: tuple[str, int],
        spot: tuple[int, _Run, float, float],
        duration: float,
    ) -> int:
        rank, run, start, end = spot
        run.cursor = end
        self._load[rank] += duration
        self.placed[work] = (rank, start, end)
        self._run_of[work] = run
        return rank

    def _place_encodes(self) -> None:
        consume, cost = self._consume, self._cost
        prologue: dict[int, list[int]] = {r: [] for r in self._ranks}
        busy = [0.0] * len(self._ranks)

        def to_prologue(batch: list[int]) -> None:
            for m in sorted(batch, key=lambda mb: (-cost[mb], consume[mb])):
                rank = min(self._ranks, key=lambda r: (busy[r], r))
                busy[rank] += cost[m]
                self._load[rank] += cost[m]
                prologue[rank].append(m)
                self.encode_rank[m] = rank

        by_consume = sorted(self._mbs, key=lambda m: consume[m])
        if self._pipeline_order is None:
            upfront = by_consume
        else:
            upfront = by_consume[: len(self._ranks)]
        to_prologue(upfront)
        unplaced: list[int] = []
        for m in by_consume[len(upfront) :]:
            spot = _encode_spot(
                self._runs,
                cost[m],
                self._load,
                due=self._times[consume[m]],
                stage0_rank=self._stage0_rank,
            )
            if spot is None:
                unplaced.append(m)
            else:
                self.encode_rank[m] = self._place(("encode", m), spot, cost[m])
        to_prologue(unplaced)
        self.prologue = {
            r: tuple(sorted(prologue[r], key=lambda m: consume[m])) for r in self._ranks
        }

    def _place_backwards(self) -> None:
        ready, cost = self._ready, self._cost
        epilogue: dict[int, list[int]] = {r: [] for r in self._ranks}
        busy = [0.0] * len(self._ranks)
        leftover: list[int] = []
        for m in self._backward_mbs:
            duration = _BACKWARD_COST * cost[m]
            spot = _backward_spot(
                self._runs,
                duration,
                self._load,
                ready=self._times[ready[m]],
                stage0_rank=self._stage0_rank,
            )
            if spot is None:
                leftover.append(m)
            else:
                self.backward_rank[m] = self._place(("backward", m), spot, duration)
        for m in sorted(leftover, key=lambda mb: (-cost[mb], ready[mb])):
            rank = min(self._ranks, key=lambda r: (busy[r], r))
            busy[rank] += _BACKWARD_COST * cost[m]
            epilogue[rank].append(m)
            self.backward_rank[m] = rank
        self.epilogue = {
            r: tuple(sorted(epilogue[r], key=lambda m: ready[m])) for r in self._ranks
        }

    def _anchor_work(self) -> None:
        anchored: dict[int, dict[Anchor, list[tuple[str, int]]]] = {
            r: {} for r in self._ranks
        }
        for work, (rank, _, _) in sorted(
            self.placed.items(), key=lambda item: (item[1][0], item[1][1])
        ):
            anchored[rank].setdefault(self._run_of[work].anchor, []).append(work)
        self.anchored = {
            r: {a: tuple(works) for a, works in anchored[r].items()}
            for r in self._ranks
        }

    def _hook_transfers(self) -> None:
        stage0_rank = self._stage0_rank
        transfers: list[tuple[int, int, int, int, int]] = []
        for m in self._mbs:
            if self.encode_rank[m] != stage0_rank:
                run = self._run_of.get(("encode", m))
                boundary = 0 if run is None else run.end
                transfers.append((boundary, 0, m, self.encode_rank[m], stage0_rank))
        for m in self._backward_mbs:
            if self.backward_rank[m] != stage0_rank:
                run = self._run_of.get(("backward", m))
                if run is None:
                    boundary = self._step_end
                else:
                    boundary = max(run.start, self._ready[m])
                transfers.append((boundary, 1, m, stage0_rank, self.backward_rank[m]))
        posts: dict[int, dict[Hook, list[Transfer]]] = {r: {} for r in self._ranks}
        for boundary, what, m, sender, receiver in sorted(transfers):
            name = ("feature", "gradient")[what]
            for rank, kind, peer in (
                (sender, f"send_{name}", receiver),
                (receiver, f"recv_{name}", sender),
            ):
                order = self._pipeline_order
                actions = None if order is None else order[rank]
                hook = _hook_at(actions, boundary, self._step_end)
                posts[rank].setdefault(hook, []).append((kind, m, peer))
        self.posts = {
            r: {h: tuple(ts) for h, ts in posts[r].items()} for r in self._ranks
        }
