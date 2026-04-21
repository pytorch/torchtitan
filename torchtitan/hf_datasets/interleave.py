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
from typing import Any

from torch.utils.data import IterableDataset


class InterleavedDataset(IterableDataset):
    """Randomly interleaves multiple IterableDatasets weighted by sampling probability.

    At each step a source is drawn proportionally to its weight.  Iteration
    stops as soon as any source is exhausted, defining a natural epoch
    boundary consistent with HuggingFaceTextDataset / ChatDataset.

    All sources must implement ``state_dict`` / ``load_state_dict``.
    Re-looping, infinite behaviour, and within-source shuffling are each
    source's responsibility.
    """

    def __init__(
        self,
        datasets: list,
        weights: list[float],
        seed: int | None = None,
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

    def __iter__(self) -> Iterator:
        """Yield weighted-interleaved samples across all sources.

        A source is selected at each step proportionally to its weight.
        Iteration stops as soon as any source raises StopIteration.
        """
        indices = list(range(len(self._datasets)))
        iterators = [iter(ds) for ds in self._datasets]

        while True:
            idx = self._rng.choices(indices, weights=self._probs, k=1)[0]
            try:
                yield next(iterators[idx])
            except StopIteration:
                return

    def state_dict(self) -> dict[str, Any]:
        """Snapshot of interleaver RNG and all source states."""
        return {
            "rng_state": self._rng.getstate(),
            "sources": [ds.state_dict() for ds in self._datasets],
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        """Restore from *sd*."""
        self._rng.setstate(sd["rng_state"])
        for ds, src_sd in zip(self._datasets, sd["sources"]):
            ds.load_state_dict(src_sd)
