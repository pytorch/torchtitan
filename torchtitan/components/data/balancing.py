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

Each rank buffers ``window_size`` consecutive microbatches and a
``MicrobatchBalancer`` decides which of them it trains at each step of the
window. ``LocalBalancer`` yields each rank's own window heaviest first. Every
rank opens a window at the same step, so step ``k`` runs every rank's ``k``-th
heaviest microbatch. For sorted sequences this pairing minimizes the sum of
per-step maxima, with no communication. It cannot fix a window in which one
rank holds several heavy microbatches.

``GlobalBalancer`` fixes that by planning each window across all
data-parallel ranks: the ranks all-gather their costs over gloo, every rank
computes the same plan (``plan_window``), and moved microbatches are sent
point to point. It runs on the loader's prefetch thread, so it overlaps
training.

The set of microbatches per window is unchanged; each moves at most
``window_size - 1`` steps (like the packer's own bin shuffling). Local
balancing does not depend on the mesh, so it applies unchanged to
dp_replicate, dp_shard, CP (CP peers read the same data and order it the same
way), and gradient accumulation when FSDP synchronizes every microbatch (the
default). It does not help when ranks synchronize once per several
microbatches: with pipeline parallelism, or with
``defer_fsdp_gradient_reduction``, what matters is each rank's sum over the
step's microbatches, which reordering within a window does not balance. No
order can
split a single document: a row that is one full-length document stays as
heavy as it is.
"""

# TODO: group-local balancing, which plans within blocks of data-parallel ranks
# instead of all of them, to bound the gather and the plan on very large jobs.

import logging
import pickle
import time
from abc import ABC, abstractmethod
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
from torchtitan.config import Configurable

logger = logging.getLogger(__name__)


def attention_cost(segment_lengths: Sequence[int]) -> int:
    """Document-masked attention cost: ``sum(L ** 2)`` over attention segments.

    Each token attends only to earlier tokens of its own segment, so this is
    proportional to attention FLOPs. Linear layers cost the same for every
    microbatch of a fixed size, padding included, so they do not change the
    order.
    """
    return sum(length * length for length in segment_lengths)


def padding_segment_lengths(num_padding: int, max_context_length: int) -> list[int]:
    """Segments of a padding tail whose positions reset every ``max_context_length``.

    Collators and packers give padding positions ``arange(n) % max_context_length``,
    so attention computes 10 padding tokens with ``max_context_length`` 4 as
    segments ``[4, 4, 2]``.
    """
    num_full, rest = divmod(num_padding, max_context_length)
    return [max_context_length] * num_full + ([rest] if rest else [])


def positions_segment_lengths(positions: torch.Tensor) -> list[int]:
    """Segments as the varlen attention metadata sees them.

    ``create_varlen_metadata_for_document`` starts a segment at every
    ``positions == 0``, padding included; the first token always starts one
    (a row may begin mid-document)::

        positions     5 6 7 0 1 0 1 2 | 0 1 2 3 4 0 1
        segments      [ 3 ] [2] [ 3 ]   [   5   ] [2]

    (``|``: padding starts; ``max_context_length`` = 5.)
    """
    positions = positions.reshape(-1)
    if positions.numel() == 0:
        return []
    starts = positions == 0
    starts[0] = True
    return torch.bincount(torch.cumsum(starts, dim=0))[1:].tolist()


def packed_attention_cost(
    document_lengths: Sequence[int], *, num_tokens: int, max_context_length: int
) -> int:
    """``attention_cost`` of documents packed into ``num_tokens``, then padding.

    For packing decisions made from lengths, before a microbatch exists (RL's
    batcher): the segments are the documents plus the padding tail.
    """
    tail = padding_segment_lengths(
        num_tokens - sum(document_lengths), max_context_length
    )
    return attention_cost([*document_lengths, *tail])


def document_attention_cost(microbatch: TrainingMicrobatch) -> float:
    """``attention_cost`` of a collated token microbatch, from its ``positions``.

    Reading segments from ``positions`` matches the kernel exactly, including
    per-document padding (``per_sample_pad_multiple``) and padding added by
    both the packer and the collator.
    """
    if not isinstance(microbatch, TokenizedTrainingMicrobatch):
        raise ValueError(
            "document_attention_cost requires TokenizedTrainingMicrobatch, got "
            f"{type(microbatch).__name__}"
        )
    return float(attention_cost(positions_segment_lengths(microbatch.positions)))


class MicrobatchBalancer(Configurable, ABC):
    """Decides which microbatch of a window this rank trains at each step."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        window_size: int = 8
        """Consecutive microbatches balanced together. Larger windows balance
        better but hold more collated microbatches in host memory and move
        microbatches further from their packed order."""
        cost_fn: Callable[[TrainingMicrobatch], float] = document_attention_cost
        """Relative cost of one microbatch."""

        def __post_init__(self) -> None:
            if self.window_size < 1:
                raise ValueError("window_size must be positive")

    def __init__(self, config: Config, **kwargs: Any) -> None:
        del kwargs  # runtime context (dp_rank, dp_world_size) some balancers use
        self.window_size = config.window_size
        self._cost_fn = config.cost_fn

    @abstractmethod
    def balance(self, rows: list[Any]) -> list[Any]:
        """Return the microbatches this rank trains in the next ``len(rows)``
        steps, in step order. Every rank calls it at the same step."""


class LocalBalancer(MicrobatchBalancer):
    """Yield this rank's own window heaviest first, without communication."""

    @dataclass(kw_only=True, slots=True)
    class Config(MicrobatchBalancer.Config):
        pass

    def balance(self, rows: list[Any]) -> list[Any]:
        # Stable: equal costs keep their packed order.
        costs = [self._cost_fn(row) for row in rows]
        return [rows[i] for i in sorted(range(len(rows)), key=lambda i: -costs[i])]


def assign_sorted_bins_keep_local(
    owners: Sequence[int], *, dp_degree: int
) -> list[list[int]]:
    """Assign D-wide rows of sorted bins to DP ranks, moving as few as possible.

    The bins are sorted from highest to lowest cost; each run of D consecutive
    bins is a row, one microbatch on every rank. ``owners[j]`` is the rank that
    already holds bin ``j`` (data-parallel loaders split data across ranks
    before balancing). Each bin stays on its owner when possible (at most one
    per rank per row) and the rest go to the ranks left free. Returns, per
    row, the bin index each rank trains.

    One microbatch per rank per step only: then a row's cost is its heaviest
    bin whatever the ranks, so keeping bins local is free. RL's batcher, which
    gathers the batch in one place, uses the same rows with a different rank
    choice (``torchtitan.rl.trainer.batcher.assign_sorted_bins_to_dp_ranks``).

    TODO: a PP version (M microbatches per step) must also balance each rank's
    step total, trading moves against balance (RL's planner balances step
    totals but ignores where bins are).

    Example:
        # Sorted bins 9, 8 | 4, 3 held by ranks 0, 0 | 1, 1.
        assign_sorted_bins_keep_local([0, 0, 1, 1], dp_degree=2)
        # -> [[0, 1], [3, 2]]: rank 0 keeps 9 and sends 8 to rank 1;
        #    rank 1 keeps 4 and sends 3 to rank 0.
    """
    assert dp_degree > 0 and len(owners) % dp_degree == 0
    assignments = []
    for row_start in range(0, len(owners), dp_degree):
        slots: list[int | None] = [None] * dp_degree
        moved = []
        for bin_index in range(row_start, row_start + dp_degree):
            if slots[owners[bin_index]] is None:
                slots[owners[bin_index]] = bin_index
            else:
                moved.append(bin_index)
        free = [rank for rank in range(dp_degree) if slots[rank] is None]
        for rank, bin_index in zip(free, moved):
            slots[rank] = bin_index
        assignments.append([i for i in slots if i is not None])
    return assignments


def plan_window(costs: Sequence[Sequence[float]]) -> list[list[tuple[int, int]]]:
    """Plan one window across ``N`` ranks.

    ``costs[s][i]`` is the cost of rank ``s``'s ``i``-th microbatch; every rank
    holds the same number, ``W``. All ``N * W`` microbatches are sorted by cost
    (ties by rank, then index) and cut into ``W`` groups of ``N``: step ``k``
    runs the ``k``-th group, so each step's microbatches cost about the same.
    Within a step, each microbatch stays on the rank that holds it when it can
    (``assign_sorted_bins_keep_local``). Deterministic, so every rank computes
    the same plan from the same costs.

    Returns:
        ``plan[k][d] = (s, i)``: at step ``k``, rank ``d`` trains rank ``s``'s
        ``i``-th microbatch.
    """
    order = sorted(
        ((cost, s, i) for s, row in enumerate(costs) for i, cost in enumerate(row)),
        key=lambda x: (-x[0], x[1], x[2]),
    )
    rows = assign_sorted_bins_keep_local([s for _, s, _ in order], dp_degree=len(costs))
    return [[order[j][1:] for j in row] for row in rows]


class GlobalBalancer(MicrobatchBalancer):
    """Plan each window across all data-parallel ranks and move microbatches.

    ``LocalBalancer`` can only reorder a rank's own window, so a rank that
    holds several heavy microbatches stays the slowest for several steps.
    ``GlobalBalancer`` also moves microbatches between ranks. Two ranks, a
    window of 4, step time = slowest rank::

        packed order         LocalBalancer        GlobalBalancer
        step  r0  r1  time   step  r0  r1  time   step  r0  r1  time
          0    9   3    9      0    9   4    9      0    9   8*   9
          1    1   2    2      1    8   3    8      1    3*  4    4
          2    8   1    8      2    2   2    2      2    2   2    2
          3    2   4    4      3    1   1    1      3    1   1    1
                 total 23             total 20             total 16

        * moved between ranks

    Every microbatch is still trained once per window. Ranks that share a
    data-parallel rank (CP, TP) read the same microbatches and compute the same
    plan, so each exchanges only with the ranks in the same position of the
    other data-parallel ranks, over its own gloo group. Assumes the
    data-parallel mesh dimensions come before CP and TP and there is no pipeline
    parallelism (checked). Every rank must fill full windows, which the loader
    guarantees by requiring ``repeat`` with DP > 1.

    Any picklable microbatch can move, including ``model_kwargs`` with image
    tensors. The gloo groups time out after 30 minutes, which also covers loader
    startup on the first window; on timeout the loader raises rather than
    falling back, since a rank that kept its own microbatches would train
    different data than its CP peers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MicrobatchBalancer.Config):
        pass

    def __init__(self, config: Config, *, dp_rank: int, dp_world_size: int) -> None:
        super().__init__(config)
        if not dist.is_initialized():
            raise RuntimeError("GlobalBalancer requires torch.distributed")
        world, rank = dist.get_world_size(), dist.get_rank()
        inner = world // dp_world_size  # ranks per data-parallel rank
        if world % dp_world_size or rank // inner != dp_rank:
            raise ValueError(
                "GlobalBalancer assumes global rank = dp_rank * (world // dp) + "
                f"position (no PP); got rank {rank}, dp_rank {dp_rank}, "
                f"dp {dp_world_size}, world {world}"
            )
        # One group per position, created on every rank (new_group is collective).
        group, _ = dist.new_subgroups_by_enumeration(
            [[d * inner + c for d in range(dp_world_size)] for c in range(inner)],
            backend="gloo",
            timeout=timedelta(minutes=30),
        )
        assert isinstance(group, dist.ProcessGroup)
        self._dp_gloo_group = group

    def balance(self, rows: list[Any]) -> list[Any]:
        """Return the microbatches this rank trains in the next W steps.

        All-gathers the costs, computes the shared plan, keeps the rows the plan
        leaves here, and exchanges the rest point to point (pickled; one round
        of sizes, one of payloads).
        """
        group = self._dp_gloo_group
        me = group.rank()
        start = time.perf_counter()
        try:
            # 1. Gather every rank's costs and compute the shared plan.
            costs = torch.tensor([self._cost_fn(r) for r in rows], dtype=torch.float64)
            gathered = [torch.empty_like(costs) for _ in range(group.size())]
            dist.all_gather(gathered, costs, group=group)
            plan = plan_window([g.tolist() for g in gathered])

            # 2. List my moves: (step k, receiver d, my row i) for each row I
            # send, and (step k, sender s) for each step I train another rank's
            # row. Both sides enumerate moves in step order, and a rank trains
            # one microbatch per step, so k is a unique tag per (sender,
            # receiver) pair.
            sends = [
                (k, d, i)
                for k, step in enumerate(plan)
                for d, (s, i) in enumerate(step)
                if s == me and d != me
            ]
            recvs = [
                (k, step[me][0]) for k, step in enumerate(plan) if step[me][0] != me
            ]

            def exchange(recv: dict[int, torch.Tensor], send: list[torch.Tensor]):
                ops = [
                    dist.P2POp(dist.irecv, recv[k], group=group, group_peer=s, tag=k)
                    for k, s in recvs
                ] + [
                    dist.P2POp(dist.isend, t, group=group, group_peer=d, tag=k)
                    for (k, d, _), t in zip(sends, send)
                ]
                # P2POp.__new__ is annotated to return object.
                # pyrefly: ignore [bad-argument-type]
                for work in dist.batch_isend_irecv(ops) if ops else []:
                    work.wait()

            # 3. Pickle the rows I send, then exchange sizes (so receivers can
            # allocate buffers) and payloads.
            payloads = [
                torch.frombuffer(bytearray(pickle.dumps(rows[i])), dtype=torch.uint8)
                for _, _, i in sends
            ]
            sizes = {k: torch.empty(1, dtype=torch.int64) for k, _ in recvs}
            exchange(sizes, [torch.tensor([p.numel()]) for p in payloads])
            received = {
                k: torch.empty(int(sizes[k]), dtype=torch.uint8) for k, _ in recvs
            }
            exchange(received, payloads)
        except RuntimeError as e:
            raise RuntimeError(
                "GlobalBalancer exchange failed or timed out; a data-parallel peer "
                "stopped producing microbatches"
            ) from e

        logger.debug(
            "GlobalBalancer: %d in, %d out, %d bytes received, %.1f ms",
            len(recvs),
            len(sends),
            sum(t.numel() for t in received.values()),
            (time.perf_counter() - start) * 1e3,
        )
        # 4. Assemble my window in step order: kept rows plus received ones.
        return [
            pickle.loads(received[k].numpy().tobytes()) if s != me else rows[i]
            for k, (s, i) in enumerate(step[me] for step in plan)
        ]


class MicrobatchBalancingIterDataset(grain.IterDataset):
    """Balance each window of ``balancer.window_size`` parent elements."""

    def __init__(self, parent: grain.IterDataset, *, balancer: MicrobatchBalancer):
        super().__init__(parent)
        self._balancer = balancer

    def __iter__(self) -> grain.DatasetIterator:
        return _MicrobatchBalancingIterator(iter(self._parent), self._balancer)


class _MicrobatchBalancingIterator(grain.DatasetIterator):
    """Buffer one window, balance it, and yield this rank's share in step order.

    The checkpoint state is the parent state at the window start plus the
    number of elements already yielded from the window; restoring re-reads
    the window from the parent, balances it again, and skips them.
    """

    def __init__(
        self, parent: grain.DatasetIterator, balancer: MicrobatchBalancer
    ) -> None:
        super().__init__(parent)
        self._balancer = balancer
        self._window: list[Any] = []  # this rank's microbatches, in step order
        self._window_start: dict[str, Any] | None = None  # parent state
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
        self._window_start = self._parent.get_state()
        rows = []
        for _ in range(self._balancer.window_size):
            try:
                rows.append(next(self._parent))
            except StopIteration:
                break
        self._window = self._balancer.balance(rows) if rows else []
        self._next_index = 0

    def get_state(self) -> dict[str, Any]:
        if self._next_index == len(self._window):
            return {"parent": self._parent.get_state(), "window_index": 0}
        assert self._window_start is not None
        return {"parent": self._window_start, "window_index": self._next_index}

    def set_state(self, state: dict[str, Any]) -> None:
        self._parent.set_state(state["parent"])
        self._window = []
        self._window_start = None
        self._next_index = 0
        if state["window_index"]:
            self._fill_window()
            for i in range(state["window_index"]):
                self._window[i] = None
            self._next_index = state["window_index"]
