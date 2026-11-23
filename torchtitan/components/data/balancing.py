# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Balance microbatch cost across data-parallel ranks, one window at a time.

Data-parallel ranks synchronize every microbatch (FSDP all-gathers and
reduce-scatters, the gradient all-reduce, CP collectives), so a step runs at
the speed of its slowest rank. With document masking, attention cost grows
with the sum of squared document lengths, so equal-length packed rows can
differ several times in cost. Packers emit rows in arbitrary order, so a
rank's heavy row often lands in the same step as other ranks' light rows:

    cost per step, 2 ranks, window of 4 microbatches
    (# = one unit of compute, step time = slowest rank)

    packer order                         heaviest first
    step   rank 0     rank 1     time    rank 0     rank 1     time
      0    ####       ########   8       ########   ########   8
      1    ########   ##         8       ####       ######     6
      2    #          ######     6       ##         ##         2
      3    ##         ##         2       #          ##         2
                          total 24                        total 18

Each rank buffers ``window_size`` consecutive microbatches. A planner then
decides, for every step of the window, which microbatch each rank trains:

    plan[k][d] = (s, i): at step k, rank d trains rank s's i-th microbatch

``plan_window`` sorts the costs of a group of ``G`` ranks' windows together
and gives step ``k`` the ``k``-th group of ``G`` microbatches, keeping each
microbatch on its own rank when possible. Every rank in a group computes the
same plan, so they agree without a coordinator.

``group_size=1`` (the default) plans each rank's window alone: each rank
yields its own microbatches heaviest first. Every rank opens a window at the
same step, so step ``k`` runs every rank's ``k``-th heaviest microbatch. For
sorted sequences this pairing minimizes the sum of per-step maxima, with no
communication. It cannot fix a window in which one rank holds several heavy
microbatches.

Larger groups move microbatches between ranks to fix that. Each rank buffers
its window, the group all-gathers the costs over gloo, every rank computes the
same plan, and moved microbatches are sent point to point. This runs on the
loader's prefetch thread, so it overlaps training. ``group_size=None`` plans
across all data-parallel ranks; ``group_size=G`` plans within contiguous
blocks of ``G`` data-parallel ranks, which bounds the gather and the plan for
very large jobs. Ranks that share a data-parallel rank (CP, TP) read the same
microbatches and compute the same plan, so each exchanges only with the ranks
in the same position of the other data-parallel ranks.

The set of microbatches per window is unchanged; each moves at most
``window_size - 1`` steps (like the packer's own bin shuffling). With
``group_size=1`` nothing depends on the mesh, so it applies unchanged to
dp_replicate, dp_shard, CP (CP peers read the same data and plan it the same
way), gradient accumulation, and pipeline-parallel microbatches. No plan can
split a single document: a row that is one full-length document stays as
heavy as it is.
"""

import io
import logging
import pickle
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import grain.python as grain
import torch
import torch.distributed as dist

from torchtitan.components.data.types import (
    TokenizedTrainingMicrobatch,
    TrainingMicrobatch,
)

logger = logging.getLogger(__name__)

# Exchange timeouts for group_size > 1. The first window also waits for every
# rank's loader to start; later windows only for loader skew between ranks,
# which is a small fraction of a window of training steps. There is no
# fallback: a timed-out gloo pair is closed, and a rank that kept its own
# microbatches would desynchronize from its CP/TP peers, so the loader raises.
FIRST_WINDOW_TIMEOUT = timedelta(minutes=30)
WINDOW_TIMEOUT = timedelta(minutes=10)

Plan = list[list[tuple[int, int]]]
"""``plan[k][d] = (s, i)``: at step ``k``, group rank ``d`` trains group rank
``s``'s ``i``-th microbatch of the window."""


def document_attention_cost(
    microbatch: TrainingMicrobatch, *, token_weight: float = 0.0
) -> float:
    """Relative compute cost of a token microbatch with document masking.

    ``cost = sum(document_length ** 2) + token_weight * num_non_padding_tokens``

    Documents are maximal runs of consecutive positions over non-padding
    tokens, so a position reset starts a new document::

        positions     0 1 2 3 0 1 0 1 2 . .      (. = padding)
        documents     [ 4   ] [2] [ 3 ]
        cost          16 + 4 + 9 = 29

    The squared term is proportional to document-masked attention FLOPs. The
    token term covers linear layers; it only changes the order when rows hold
    different numbers of non-padding tokens. Set ``token_weight`` to the
    ratio of per-token linear cost to per-token-pair attention cost, about
    the non-embedding parameters per layer divided by the hidden size. A fit
    of Llama-3-8B step times on GB300 gives 57,220.
    """
    if not isinstance(microbatch, TokenizedTrainingMicrobatch):
        raise ValueError(
            "document_attention_cost requires TokenizedTrainingMicrobatch, got "
            f"{type(microbatch).__name__}"
        )
    positions = microbatch.positions.reshape(-1)
    valid = ~microbatch.padding_mask.reshape(-1)
    if positions.numel() == 0:
        return 0.0
    # A token starts a document if it is valid and does not continue the
    # previous valid token's position run.
    prev = torch.cat([positions.new_full((1,), -2), positions[:-1]])
    prev_valid = torch.cat([valid.new_zeros(1), valid[:-1]])
    starts = valid & ~(prev_valid & (positions == prev + 1))
    doc_ids = torch.cumsum(starts.to(torch.int64), dim=0)[valid]
    if doc_ids.numel() == 0:
        return 0.0
    lengths = torch.bincount(doc_ids).to(torch.float64)
    return float(torch.dot(lengths, lengths)) + token_weight * doc_ids.numel()


def plan_window(costs: Sequence[Sequence[float]]) -> Plan:
    """Plan one window for a group of ranks.

    ``costs[s][i]`` is the cost of group rank ``s``'s ``i``-th microbatch; every
    rank holds the same number of microbatches. All microbatches are sorted by
    cost (ties by rank, then index), step ``k`` gets the ``k``-th group of ``G``,
    and within a step each microbatch stays on its own rank when possible (at
    most one per rank), so as few as possible move. With one rank this is the
    rank's own window heaviest first.

    Args:
        costs: Per-rank microbatch costs, ``G`` lists of equal length ``W``.

    Returns:
        ``plan[k][d] = (s, i)`` for ``k < W`` and ``d < G``.
    """
    num_ranks = len(costs)
    window = len(costs[0]) if num_ranks else 0
    if any(len(c) != window for c in costs):
        raise ValueError("every rank must hold the same number of microbatches")
    order = sorted(
        ((float(costs[s][i]), s, i) for s in range(num_ranks) for i in range(window)),
        key=lambda x: (-x[0], x[1], x[2]),
    )
    plan: Plan = []
    for k in range(window):
        step: list[tuple[int, int] | None] = [None] * num_ranks
        moved = []
        for _, s, i in order[k * num_ranks : (k + 1) * num_ranks]:
            if step[s] is None:
                step[s] = (s, i)
            else:
                moved.append((s, i))
        free = [d for d in range(num_ranks) if step[d] is None]
        for d, src in zip(free, moved):
            step[d] = src
        plan.append([src for src in step if src is not None])
    return plan


@dataclass(frozen=True, kw_only=True, slots=True)
class MicrobatchBalancingConfig:
    """Plan each window of ``window_size`` microbatches across ``group_size`` ranks."""

    window_size: int = 8
    """Consecutive microbatches planned together. Larger windows balance better
    but hold more collated microbatches in host memory and move microbatches
    further from their packed order."""
    group_size: int | None = 1
    """Data-parallel ranks that plan a window together. 1 orders each rank's own
    window heaviest first, without communication. ``G > 1`` plans within
    contiguous blocks of ``G`` data-parallel ranks and moves microbatches
    between them over gloo; it must divide the data-parallel degree. None
    plans across all data-parallel ranks."""
    cost_fn: Callable[[TrainingMicrobatch], float] = document_attention_cost
    """Relative cost of one microbatch; use ``functools.partial`` to set the
    weights of a cost function."""
    planner: Callable[[Sequence[Sequence[float]]], Plan] = plan_window
    """Maps a group's per-rank costs to a plan."""

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError("window_size must be positive")
        if self.group_size is not None and self.group_size < 1:
            raise ValueError("group_size must be positive or None")


class _LocalPeers:
    """A group of one rank: nothing to gather or exchange."""

    def gather_costs(self, costs: list[float]) -> tuple[list[list[float]], int]:
        return [costs], 0

    def route(self, plan: Plan, local: list[Any], me: int) -> list[Any]:
        return [local[step[me][1]] for step in plan]


def _split(obj: Any) -> tuple[bytes, list[torch.Tensor]]:
    """Pickle ``obj`` with its tensors left out; returns the header and tensors.

    Works for any picklable microbatch (dataclasses, ``model_kwargs`` dicts of
    image tensors, nested containers): tensors become persistent ids and are
    sent as raw bytes, everything else travels in the header.
    """
    tensors: list[torch.Tensor] = []

    class _Pickler(pickle.Pickler):
        def persistent_id(self, o: Any) -> Any:
            if isinstance(o, torch.Tensor):
                if o.layout != torch.strided or o.device.type != "cpu":
                    raise ValueError("can only move dense CPU tensors")
                tensors.append(o.detach().contiguous())
                return len(tensors) - 1
            return None

    body = io.BytesIO()
    _Pickler(body, protocol=pickle.HIGHEST_PROTOCOL).dump(obj)
    meta = [(t.dtype, tuple(t.shape)) for t in tensors]
    return pickle.dumps((meta, body.getvalue())), tensors


def _read_header(
    header: bytes,
) -> tuple[list[tuple[torch.dtype, tuple[int, ...]]], bytes]:
    return pickle.loads(header)


def _join(body: bytes, tensors: list[torch.Tensor]) -> Any:
    class _Unpickler(pickle.Unpickler):
        def persistent_load(self, pid: Any) -> Any:
            return tensors[pid]

    return _Unpickler(io.BytesIO(body)).load()


def _as_bytes(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1).view(torch.uint8)


_GROUPS: dict[tuple[int, int, int], Any] = {}


def _exchange_group(dp_rank: int, dp_world_size: int, group_size: int) -> Any:
    """The gloo group of this rank's planning group, at this rank's position.

    Ranks that share a data-parallel rank (CP, TP) get separate groups, one per
    position, so each exchanges its own copy. Assumes the data-parallel mesh
    dimensions come before CP and TP and there is no pipeline parallelism, so
    global rank = ``dp_rank * (world // dp_world_size) + position``; checked.
    All groups are created on every rank (``new_group`` is collective).
    """
    if not dist.is_initialized():
        raise RuntimeError("group_size > 1 requires torch.distributed")
    world, rank = dist.get_world_size(), dist.get_rank()
    if world % dp_world_size:
        raise ValueError(f"world size {world} not divisible by DP {dp_world_size}")
    inner = world // dp_world_size
    if rank // inner != dp_rank:
        raise ValueError(
            f"microbatch balancing assumes global rank = dp_rank * {inner} + "
            f"position (no pipeline parallelism); got rank {rank}, dp_rank {dp_rank}"
        )
    key = (dp_world_size, group_size, inner)
    if key not in _GROUPS:
        ranks = [
            [(g * group_size + j) * inner + c for j in range(group_size)]
            for g in range(dp_world_size // group_size)
            for c in range(inner)
        ]
        _GROUPS[key], _ = dist.new_subgroups_by_enumeration(
            ranks, backend="gloo", timeout=FIRST_WINDOW_TIMEOUT
        )
    return _GROUPS[key]


class _GroupPeers:
    """A group of data-parallel ranks that gathers costs and moves microbatches."""

    def __init__(self, group: Any, rank: int, size: int) -> None:
        self._group = group
        self._rank = rank
        self._size = size
        self._timeout = FIRST_WINDOW_TIMEOUT
        self._windows = 0
        self._t0 = self._t1 = 0.0

    def _wait(self, works: list[Any], what: str) -> None:
        try:
            for work in works:
                work.wait(self._timeout)
        except RuntimeError as e:
            raise RuntimeError(
                f"microbatch balancing: {what} of window {self._windows} timed out "
                f"after {self._timeout} or failed; a data-parallel peer stopped "
                "producing microbatches"
            ) from e

    def gather_costs(self, costs: list[float]) -> tuple[list[list[float]], int]:
        self._t0 = time.perf_counter()
        local = torch.tensor(costs, dtype=torch.float64)
        gathered = [torch.empty_like(local) for _ in range(self._size)]
        work = dist.all_gather(gathered, local, group=self._group, async_op=True)
        self._wait([work], "cost all-gather")
        self._t1 = time.perf_counter()
        return [g.tolist() for g in gathered], self._rank

    def route(self, plan: Plan, local: list[Any], me: int) -> list[Any]:
        t2 = time.perf_counter()
        # Moves in step order; both sides enumerate them identically, so the
        # step index is a unique tag per (sender, receiver) pair.
        sends = [
            (k, d, i)
            for k, step in enumerate(plan)
            for d, (s, i) in enumerate(step)
            if s == me and d != me
        ]
        recvs = [(k, step[me][0]) for k, step in enumerate(plan) if step[me][0] != me]
        g, works = self._group, []

        # Round 1: header sizes. Round 2: headers (tensor dtypes and shapes plus
        # everything that is not a tensor). Round 3: tensor bytes.
        split = {i: _split(local[i]) for i in {i for _, _, i in sends}}
        sizes = {k: torch.empty(1, dtype=torch.int64) for k, _ in recvs}
        for k, s in recvs:
            works.append(dist.irecv(sizes[k], group_src=s, group=g, tag=k))
        for k, d, i in sends:
            size = torch.tensor([len(split[i][0])], dtype=torch.int64)
            works.append(dist.isend(size, group_dst=d, group=g, tag=k))
        self._wait(works, "size exchange")

        works = []
        headers = {k: torch.empty(int(sizes[k]), dtype=torch.uint8) for k, _ in recvs}
        for k, s in recvs:
            works.append(dist.irecv(headers[k], group_src=s, group=g, tag=k))
        for k, d, i in sends:
            header = torch.frombuffer(bytearray(split[i][0]), dtype=torch.uint8)
            works.append(dist.isend(header, group_dst=d, group=g, tag=k))
        self._wait(works, "header exchange")

        works, received, sent_bytes, recv_bytes = [], {}, 0, 0
        for k, s in recvs:
            meta, body = _read_header(headers[k].numpy().tobytes())
            tensors = [torch.empty(shape, dtype=dtype) for dtype, shape in meta]
            for j, t in enumerate(tensors):
                if t.numel():
                    buf = _as_bytes(t)
                    works.append(
                        dist.irecv(buf, group_src=s, group=g, tag=k * _MAX_TENSORS + j)
                    )
                    recv_bytes += buf.numel()
            received[k] = (body, tensors)
        for k, d, i in sends:
            tensors = split[i][1]
            if len(tensors) > _MAX_TENSORS:
                raise ValueError(f"a microbatch holds over {_MAX_TENSORS} tensors")
            for j, t in enumerate(tensors):
                if t.numel():
                    buf = _as_bytes(t)
                    works.append(
                        dist.isend(buf, group_dst=d, group=g, tag=k * _MAX_TENSORS + j)
                    )
                    sent_bytes += buf.numel()
        self._wait(works, "microbatch exchange")

        window = [
            _join(*received[k]) if s != me else local[i]
            for k, (s, i) in enumerate(step[me] for step in plan)
        ]
        t3 = time.perf_counter()
        logger.debug(
            "microbatch balancing window %d: gather %.1f ms, plan %.1f ms, "
            "exchange %.1f ms, %d in / %d out, %d / %d bytes received / sent",
            self._windows,
            (self._t1 - self._t0) * 1e3,
            (t2 - self._t1) * 1e3,
            (t3 - t2) * 1e3,
            len(recvs),
            len(sends),
            recv_bytes,
            sent_bytes,
        )
        self._windows += 1
        self._timeout = WINDOW_TIMEOUT
        return window


_MAX_TENSORS = 1 << 16


class MicrobatchBalancingIterDataset(grain.IterDataset):
    """Plan each window of ``window_size`` parent elements across a group of ranks."""

    def __init__(
        self,
        parent: grain.IterDataset,
        *,
        config: MicrobatchBalancingConfig,
        dp_rank: int = 0,
        dp_world_size: int = 1,
    ) -> None:
        super().__init__(parent)
        self._config = config
        group_size = config.group_size or dp_world_size
        if dp_world_size % group_size:
            raise ValueError(
                f"group_size {group_size} must divide the DP degree {dp_world_size}"
            )
        # Group creation is collective, so it happens here on every rank's main
        # thread, not on the prefetch thread.
        self._peers: _LocalPeers | _GroupPeers = (
            _LocalPeers()
            if group_size == 1
            else _GroupPeers(
                _exchange_group(dp_rank, dp_world_size, group_size),
                dp_rank % group_size,
                group_size,
            )
        )

    def __iter__(self) -> grain.DatasetIterator:
        return _MicrobatchBalancingIterator(
            iter(self._parent), config=self._config, peers=self._peers
        )


class _MicrobatchBalancingIterator(grain.DatasetIterator):
    """Buffer one window, plan it, and yield this rank's share in step order.

    The checkpoint state is the parent state at the window start plus the
    number of elements already yielded from the window; restoring re-reads
    the window from the parent (and re-plans it) and skips them.
    """

    def __init__(
        self,
        parent: grain.DatasetIterator,
        *,
        config: MicrobatchBalancingConfig,
        peers: Any,
    ) -> None:
        super().__init__(parent)
        self._window_size = config.window_size
        self._cost_fn = config.cost_fn
        self._planner = config.planner
        self._peers = peers
        self._window: list[Any] = []
        self._window_parent_state: dict[str, Any] | None = None
        self._next_index = 0

    def __next__(self) -> Any:
        self._assert_not_closed()
        if self._next_index == len(self._window):
            self._fill_window()
        if not self._window:
            raise StopIteration
        element = self._window[self._next_index]
        self._window[self._next_index] = None  # release yielded elements
        self._next_index += 1
        return element

    def _fill_window(self) -> None:
        self._window_parent_state = self._parent.get_state()
        local = []
        for _ in range(self._window_size):
            try:
                local.append(next(self._parent))
            except StopIteration:
                break
        if len(local) < self._window_size and isinstance(self._peers, _GroupPeers):
            raise RuntimeError(
                "group_size > 1 needs full windows on every rank (repeat=True)"
            )
        costs, me = self._peers.gather_costs([self._cost_fn(e) for e in local])
        self._window = self._peers.route(self._planner(costs), local, me)
        self._next_index = 0

    def get_state(self) -> dict[str, Any]:
        if self._next_index == len(self._window):
            return {"parent": self._parent.get_state(), "window_index": 0}
        assert self._window_parent_state is not None
        return {
            "parent": self._window_parent_state,
            "window_index": self._next_index,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._parent.set_state(state["parent"])
        self._window = []
        self._window_parent_state = None
        self._next_index = 0
        if state["window_index"]:
            self._fill_window()
            for i in range(state["window_index"]):
                self._window[i] = None
            self._next_index = state["window_index"]
