# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Balance microbatch attention cost across data-parallel ranks by exchanging rows.

Heaviest-first (``balancing.py``) reorders each rank's own window and so cannot fix a
window in which one rank holds several heavy rows. This stage plans the window across
all DP ranks and moves rows between them:

1. each rank buffers its next ``window_size`` microbatches and computes their costs;
2. the costs are all-gathered over a CPU (gloo) group of the DP peers;
3. every rank computes the same plan: sort all ``N * W`` rows by cost, step ``k`` of the
   window gets the ``k``-th group of ``N`` rows, and within a step each row stays on the
   rank that produced it when possible (at most one kept row per rank per step);
4. moved rows are sent point to point (gloo isend / irecv) to their new rank.

CP peers (same DP rank, different CP rank) read identical rows and compute the same
plan, so each CP rank exchanges with the ranks of the same CP index in other DP groups.
Everything runs on the data loader's prefetch thread, overlapped with training.

The set of rows trained per window is unchanged; only which rank trains which row at
which step changes. A row moves at most ``window_size - 1`` steps from packed order.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import grain.python as grain
import numpy as np
import torch
import torch.distributed as dist

from torchtitan.components.data.balancing import document_attention_cost
from torchtitan.components.data.types import TokenizedTrainingMicrobatch

logger = logging.getLogger(__name__)

# Microbatch tensors sent for a moved row, with the dtype used on the wire.
_WIRE_FIELDS = (
    ("input", torch.int32),
    ("labels", torch.int32),
    ("positions", torch.int32),
    ("padding_mask", torch.uint8),
    ("loss_token_counts", None),
    ("routing_token_counts", None),
)


@dataclass(frozen=True, kw_only=True, slots=True)
class GlobalPlanConfig:
    """Plan each window of ``window_size`` steps across all DP ranks."""

    window_size: int = 8
    cost_fn: Callable[[Any], float] = document_attention_cost
    timeout_s: float = 3600.0
    """Timeout of the gloo exchange group (the first window waits for loader startup)."""
    on_window: Callable[[dict[str, Any]], None] | None = None
    """Called on the loader thread with each window's timings and traffic (fill /
    all-gather / plan / exchange ms, bytes sent and received, rows moved). None logs
    them at DEBUG."""

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError("window_size must be positive")


def plan_window(costs: np.ndarray) -> list[list[tuple[int, int]]]:
    """Plan one window. ``costs[s, i]`` is the cost of rank ``s``'s ``i``-th row.

    Returns ``assign[k][d] = (s, i)``: at step ``k``, rank ``d`` trains rank ``s``'s
    ``i``-th row. Deterministic, so every rank computes the same plan.
    """
    n, w = costs.shape
    items = sorted(
        ((float(costs[s, i]), s, i) for s in range(n) for i in range(w)),
        key=lambda x: (-x[0], x[1], x[2]),
    )
    assign: list[list[tuple[int, int]]] = []
    for k in range(w):
        group = items[k * n : (k + 1) * n]
        step: list[tuple[int, int] | None] = [None] * n
        rest = []
        for _, s, i in group:
            if step[s] is None:
                step[s] = (s, i)  # keep it where it is
            else:
                rest.append((s, i))
        free = [d for d in range(n) if step[d] is None]
        for d, src in zip(free, rest):
            step[d] = src
        assign.append(step)  # type: ignore[arg-type]
    return assign


_GROUPS: dict[tuple[int, int, float], Any] = {}


def _exchange_group(dp_rank: int, dp_world_size: int, timeout_s: float):
    """Gloo group over the DP peers that share this rank's CP index.

    Creates the groups for every CP index (``new_group`` is collective over all ranks).
    Assumes the global rank is ``dp_rank * cp + cp_rank`` (mesh order pp, dp_replicate,
    dp_shard, cp, tp with pp = tp = 1); checked against the caller's ``dp_rank``.
    """
    world, rank = dist.get_world_size(), dist.get_rank()
    if world % dp_world_size:
        raise ValueError(f"world {world} not divisible by dp {dp_world_size}")
    cp = world // dp_world_size
    if rank // cp != dp_rank:
        raise ValueError(
            f"global plan assumes rank = dp_rank * cp + cp_rank; got rank {rank}, "
            f"dp_rank {dp_rank}, cp {cp}"
        )
    key = (dp_world_size, cp, timeout_s)
    if key not in _GROUPS:
        from datetime import timedelta

        groups = [
            dist.new_group(
                [d * cp + c for d in range(dp_world_size)],
                backend="gloo",
                timeout=timedelta(seconds=timeout_s),
            )
            for c in range(cp)
        ]
        _GROUPS[key] = groups
    return _GROUPS[key][rank % cp], cp


def _to_wire(mb: TokenizedTrainingMicrobatch) -> list[torch.Tensor]:
    if mb.model_kwargs:
        raise ValueError("global plan does not move microbatches with model_kwargs")
    out = []
    for name, dtype in _WIRE_FIELDS:
        t = getattr(mb, name)
        out.append((t.to(dtype) if dtype is not None else t).contiguous())
    return out


def _from_wire(wire: list[torch.Tensor], template: TokenizedTrainingMicrobatch):
    fields = {}
    for (name, _), t in zip(_WIRE_FIELDS, wire):
        ref = getattr(template, name)
        fields[name] = t.to(ref.dtype)
    return TokenizedTrainingMicrobatch(**fields, model_kwargs={})


class GlobalPlanIterDataset(grain.IterDataset):
    """Plan each window of ``window_size`` microbatches across DP ranks and exchange rows."""

    def __init__(
        self,
        parent: grain.IterDataset,
        *,
        config: GlobalPlanConfig,
        dp_rank: int,
        dp_world_size: int,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._dp_rank = dp_rank
        self._dp_world_size = dp_world_size
        # Collective group creation happens here, on the main thread of every rank.
        self._group, self._cp = _exchange_group(
            dp_rank, dp_world_size, config.timeout_s
        )

    def __iter__(self) -> grain.DatasetIterator:
        return _GlobalPlanIterator(
            iter(self._parent),
            config=self._config,
            dp_rank=self._dp_rank,
            dp_world_size=self._dp_world_size,
            group=self._group,
        )


class _GlobalPlanIterator(grain.DatasetIterator):
    def __init__(self, parent, *, config, dp_rank, dp_world_size, group) -> None:
        super().__init__(parent)
        self._w = config.window_size
        self._cost_fn = config.cost_fn
        self._on_window = config.on_window
        self._d = dp_rank
        self._n = dp_world_size
        self._group = group
        self._window: list[Any] = []
        self._window_parent_state: dict[str, Any] | None = None
        self._next_index = 0
        self._window_count = 0

    def __next__(self) -> Any:
        self._assert_not_closed()
        if self._next_index == len(self._window):
            self._fill_window()
        element = self._window[self._next_index]
        self._window[self._next_index] = None
        self._next_index += 1
        return element

    def _fill_window(self) -> None:
        t0 = time.perf_counter()
        self._window_parent_state = self._parent.get_state()
        local = [next(self._parent) for _ in range(self._w)]  # repeat=True: never short
        costs = torch.tensor([self._cost_fn(e) for e in local], dtype=torch.float64)
        t1 = time.perf_counter()

        gathered = [torch.empty_like(costs) for _ in range(self._n)]
        dist.all_gather(gathered, costs, group=self._group)
        all_costs = torch.stack(gathered).numpy()
        t2 = time.perf_counter()

        assign = plan_window(all_costs)
        t3 = time.perf_counter()

        # Exchange: post all receives, then all sends, then wait. Peers are group ranks.
        template = local[0]
        recv_bufs: dict[int, list[torch.Tensor]] = {}
        works = []
        sent = recv = moved_in = moved_out = 0
        wire_local = {}
        for k, step in enumerate(assign):
            s, i = step[self._d]
            if s != self._d:
                bufs = [torch.empty_like(t) for t in _to_wire(template)]
                for f, b in enumerate(bufs):
                    works.append(
                        dist.irecv(b, group_src=s, group=self._group, tag=k * 16 + f)
                    )
                    recv += b.numel() * b.element_size()
                recv_bufs[k] = bufs
                moved_in += 1
            for dst, (s2, i2) in enumerate(step):
                if s2 == self._d and dst != self._d:
                    if i2 not in wire_local:
                        wire_local[i2] = _to_wire(local[i2])
                    for f, t in enumerate(wire_local[i2]):
                        works.append(
                            dist.isend(
                                t, group_dst=dst, group=self._group, tag=k * 16 + f
                            )
                        )
                        sent += t.numel() * t.element_size()
                    moved_out += 1
        for wk in works:
            if wk is not None:
                wk.wait()
        t4 = time.perf_counter()

        window = []
        for k, step in enumerate(assign):
            s, i = step[self._d]
            window.append(
                local[i] if s == self._d else _from_wire(recv_bufs[k], template)
            )
        self._window = window
        self._next_index = 0
        stats = {
            "window": self._window_count,
            "fill_ms": (t1 - t0) * 1e3,
            "allgather_ms": (t2 - t1) * 1e3,
            "plan_ms": (t3 - t2) * 1e3,
            "exchange_ms": (t4 - t3) * 1e3,
            "bytes_sent": sent,
            "bytes_recv": recv,
            "rows_in": moved_in,
            "rows_out": moved_out,
        }
        if self._on_window is not None:
            self._on_window(stats)
        else:
            logger.debug("global plan window %s", stats)
        self._window_count += 1

    def get_state(self) -> dict[str, Any]:
        if self._next_index == len(self._window):
            return {"parent": self._parent.get_state(), "window_index": 0}
        assert self._window_parent_state is not None
        return {"parent": self._window_parent_state, "window_index": self._next_index}

    def set_state(self, state: dict[str, Any]) -> None:
        # Every rank resumes at the same window, so re-filling repeats the same exchange.
        self._parent.set_state(state["parent"])
        self._window = []
        self._window_parent_state = None
        self._next_index = 0
        if state["window_index"]:
            self._fill_window()
            for i in range(state["window_index"]):
                self._window[i] = None
            self._next_index = state["window_index"]
