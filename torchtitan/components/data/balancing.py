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

The set of microbatches per window is unchanged; each moves at most
``window_size - 1`` steps (like the packer's own bin shuffling). Local
balancing does not depend on the mesh, so it applies unchanged to
dp_replicate, dp_shard, CP (CP peers read the same data and order it the same
way), gradient accumulation, and pipeline-parallel microbatches. No order can
split a single document: a row that is one full-length document stays as
heavy as it is.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import grain.python as grain
import torch

from torchtitan.components.data.types import (
    TokenizedTrainingMicrobatch,
    TrainingMicrobatch,
)
from torchtitan.config import Configurable


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


class MicrobatchBalancer(Configurable, ABC):
    """Decides which microbatch of a window this rank trains at each step."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        window_size: int = 8
        """Consecutive microbatches balanced together. Larger windows balance
        better but hold more collated microbatches in host memory and move
        microbatches further from their packed order."""
        cost_fn: Callable[[TrainingMicrobatch], float] = document_attention_cost
        """Relative cost of one microbatch; use ``functools.partial`` to set the
        weights of a cost function."""

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
