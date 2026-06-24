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
            and ``load_state_dict()``. ``"all_exhausted"`` additionally
            requires ``reloop()``.
        weights: Sampling weights (normalised internally). Controls the
            relative likelihood of drawing from each source at each step.
        seed: Seed for the interleaver RNG.
        stopping_strategy: Controls when iteration ends.
            ``"on_first_exhausted"`` (default) stops as soon as any source
            raises ``StopIteration``, defining a natural epoch boundary.
            ``"all_exhausted"`` re-loops an exhausted source via its
            ``reloop()`` (reset + reshuffle) so its sampling weight is
            preserved, and continues until every source has been exhausted at
            least once.
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

        if stopping_strategy == "all_exhausted":
            no_reloop = [
                type(ds).__name__
                for ds in datasets
                if not callable(getattr(ds, "reloop", None))
            ]
            if no_reloop:
                raise TypeError(
                    "stopping_strategy='all_exhausted' re-loops exhausted sources, "
                    "so each dataset must implement reloop(). Missing on: "
                    f"{no_reloop}"
                )

        self._datasets = list(datasets)
        self._probs = [w / total for w in weights]
        self._rng = random.Random(seed)
        self._stopping_strategy = stopping_strategy

    def __iter__(self) -> Iterator:
        """Yield weighted-interleaved samples across all sources."""
        indices = list(range(len(self._datasets)))
        iterators = [iter(ds) for ds in self._datasets]
        exhausted = [False] * len(self._datasets)

        while True:
            idx = self._rng.choices(indices, weights=self._probs, k=1)[0]
            # TODO: `idx` identifies the source drawn here. To observe the
            # per-source sample mix, accumulate per-source counts and all-reduce
            # them across DP ranks into a metric (e.g. {"dataset_a": n, ...}) for
            # MetricsProcessor. Keep this in the dataloader; defer until
            # distributed metrics land.
            try:
                yield next(iterators[idx])
            except StopIteration:
                if self._stopping_strategy == "on_first_exhausted":
                    break
                exhausted[idx] = True
                if all(exhausted):
                    break
                # Re-loop the exhausted source (reset position + reshuffle) so it
                # keeps contributing at its sampling weight until every source has
                # been exhausted at least once.
                self._datasets[idx].reloop()
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
