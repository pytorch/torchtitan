# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Lightweight weighted interleaving of multiple IterableDatasets.

Weights are sampling probabilities (normalised internally): the relative
likelihood of drawing from each source at each step.

Iteration stops when the first source is exhausted, defining an epoch
boundary.  Re-looping, infinite behaviour, and within-source shuffling
are each source's responsibility.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any, Literal

from torch.utils.data import IterableDataset


class InterleavedDataset(IterableDataset):
    """Randomly interleaves multiple IterableDatasets weighted by sampling probability.

    Re-looping, infinite behaviour, and within-source shuffling are each
    source's responsibility.

    Args:
        datasets: Source datasets. Each must implement ``state_dict()``
            and ``load_state_dict()``.
        weights: Sampling weights (normalised internally). Controls the
            relative likelihood of drawing from each source at each step.
        seed: Seed for the interleaver RNG.
        stopping_strategy: Controls when iteration ends.
            ``"on_first_exhausted"`` (default) stops as soon as any source
            raises ``StopIteration``, defining a natural epoch boundary.
            ``"all_exhausted"`` restarts an exhausted source and continues
            until every source has been exhausted at least once.
    """

    def __init__(
        self,
        datasets: list,
        weights: list[float],
        seed: int | None = None,
        stopping_strategy: Literal[
            "on_first_exhausted", "all_exhausted"
        ] = "on_first_exhausted",
    ) -> None:
        if not datasets:
            raise ValueError("At least one dataset must be provided.")
        if len(datasets) != len(weights):
            raise ValueError(
                f"len(datasets)={len(datasets)} must equal len(weights)={len(weights)}"
            )
        if any(w < 0 for w in weights):
            raise ValueError("All weights must be non-negative.")

        total = sum(weights)
        if total == 0:
            raise ValueError("Sum of weights must be positive.")

        if stopping_strategy not in ("on_first_exhausted", "all_exhausted"):
            raise ValueError(
                "stopping_strategy must be 'on_first_exhausted' or 'all_exhausted', "
                f"got {stopping_strategy!r}"
            )

        missing = [
            type(ds).__name__
            for ds in datasets
            if not (
                callable(getattr(ds, "state_dict", None))
                and callable(getattr(ds, "load_state_dict", None))
            )
        ]
        if missing:
            raise TypeError(
                "All datasets must implement state_dict() and load_state_dict(). "
                f"Missing on: {missing}"
            )

        self._datasets = list(datasets)
        self._probs = [w / total for w in weights]
        self._rng = random.Random(seed)
        self._stopping_strategy = stopping_strategy

    @staticmethod
    def _add_source_idx(sample: Any, source_idx: int) -> Any:
        """Attach source_idx (int) to a sample, based on its type:

        * (input_dict, label) → ({**input_dict, "source_idx": i}, label)
        * plain dict          → {**d, "source_idx": i}
        * anything else           → returned unchanged (no source_idx injected)
        """
        if (
            isinstance(sample, tuple)
            and len(sample) == 2
            and isinstance(sample[0], dict)
        ):
            return ({**sample[0], "source_idx": source_idx}, sample[1])
        if isinstance(sample, dict):
            return {**sample, "source_idx": source_idx}
        return sample

    def __iter__(self) -> Iterator:
        """Yield weighted-interleaved samples across all sources."""
        indices = list(range(len(self._datasets)))
        iterators = [iter(ds) for ds in self._datasets]
        exhausted = [False] * len(self._datasets)

        while True:
            idx = self._rng.choices(indices, weights=self._probs, k=1)[0]
            try:
                yield self._add_source_idx(next(iterators[idx]), idx)
            except StopIteration:
                if self._stopping_strategy == "on_first_exhausted":
                    return
                exhausted[idx] = True
                if all(exhausted):
                    return
                iterators[idx] = iter(self._datasets[idx])

    def state_dict(self) -> dict[str, Any]:
        """Snapshot of interleaver RNG and all source states."""
        return {
            "rng_state": self._rng.getstate(),
            "sources": {i: ds.state_dict() for i, ds in enumerate(self._datasets)},
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        """Restore from *sd*."""
        self._rng.setstate(sd["rng_state"])
        for i, ds in enumerate(self._datasets):
            ds.load_state_dict(sd["sources"][i])
