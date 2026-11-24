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
way), and gradient accumulation when FSDP synchronizes every microbatch (the
default). It does not help when ranks synchronize once per several
microbatches: with pipeline parallelism, or with
``defer_fsdp_gradient_reduction``, what matters is each rank's sum over the
step's microbatches, which reordering within a window does not balance. No
order can
split a single document: a row that is one full-length document stays as
heavy as it is.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import grain.python as grain
import torch

from torchtitan.components.data.types import (
    TokenizedTrainingMicrobatch,
    TrainingMicrobatch,
)
from torchtitan.config import Configurable


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
