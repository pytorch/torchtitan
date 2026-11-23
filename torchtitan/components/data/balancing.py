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
microbatches; larger groups move microbatches between ranks to do that.

The set of microbatches per window is unchanged; each moves at most
``window_size - 1`` steps (like the packer's own bin shuffling). With
``group_size=1`` nothing depends on the mesh, so it applies unchanged to
dp_replicate, dp_shard, CP (CP peers read the same data and plan it the same
way), gradient accumulation, and pipeline-parallel microbatches. No plan can
split a single document: a row that is one full-length document stays as
heavy as it is.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import grain.python as grain
import torch

from torchtitan.components.data.types import (
    TokenizedTrainingMicrobatch,
    TrainingMicrobatch,
)

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
    window heaviest first, without communication."""
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
        # TODO: group_size > 1 (and None = all DP ranks) exchanges microbatches
        # between ranks; implemented in the next PR of this stack.
        if self.group_size != 1:
            raise NotImplementedError("group_size other than 1 is not supported yet")


class _LocalPeers:
    """A group of one rank: nothing to gather or exchange."""

    def gather_costs(self, costs: list[float]) -> tuple[list[list[float]], int]:
        return [costs], 0

    def route(self, plan: Plan, local: list[Any], me: int) -> list[Any]:
        return [local[step[me][1]] for step in plan]


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
        del dp_rank, dp_world_size  # used by groups larger than one rank
        self._config = config
        self._peers = _LocalPeers()

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
