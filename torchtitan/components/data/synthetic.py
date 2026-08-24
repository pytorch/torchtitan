# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""On-the-fly synthetic token sequences with a configurable length distribution.

Intended for DP load-balancing and throughput experiments where token *content*
is irrelevant and only the sequence-length distribution matters.

To inspect a length distribution (stats, histogram, DP-balance) before training,
run ``scripts/preview_synthetic_lengths.py`` against a JSON spec, e.g.::

    python -m scripts.preview_synthetic_lengths --spec buckets.json \\
        --seed 0 --samples 100000 --dp 8 --per-rank-batch 4
"""

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import grain.python as grain
import numpy as np
from torchtitan.components.data.dataset import SingleDatasetConfig, TextSequence
from torchtitan.components.data.loader import GrainDataLoader
from torchtitan.components.data.packing import FirstFitPackingConfig
from torchtitan.components.data.types import DatasetIterationPolicy
from torchtitan.config import Configurable


@runtime_checkable
class LengthSpec(Protocol):
    """Draws integer sequence lengths."""

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Return `size` sequence lengths as an int64 array."""
        ...


@dataclass(frozen=True, kw_only=True, slots=True)
class LengthBucket:
    """One length range and its relative selection weight."""

    min_len: int
    max_len: int
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.min_len < 1:
            raise ValueError("min_len must be >= 1")
        if self.max_len < self.min_len:
            raise ValueError("max_len must be >= min_len")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("weight must be finite and positive")


@dataclass(frozen=True, kw_only=True, slots=True)
class BucketLengthSpec:
    """Pick a bucket by weight, then a length uniformly within it (inclusive)."""

    buckets: tuple[LengthBucket, ...]

    def __post_init__(self) -> None:
        if not self.buckets:
            raise ValueError("BucketLengthSpec requires at least one bucket")

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        weights = np.array([b.weight for b in self.buckets], dtype=np.float64)
        probs = weights / weights.sum()
        choices = rng.choice(len(self.buckets), size=size, p=probs)
        lows = np.array([b.min_len for b in self.buckets], dtype=np.int64)
        highs = np.array([b.max_len for b in self.buckets], dtype=np.int64)
        lengths = rng.integers(lows[choices], highs[choices] + 1)
        return lengths.astype(np.int64)


class SyntheticSource(Configurable, grain.IterDataset):
    """Infinite stream of random-token ``TextSequence`` records.

    Lengths are drawn from ``length_spec`` and filled with random token ids, so
    only the length distribution is meaningful. Being a source, it has no access
    to the tokenizer or context, so ``vocab_size`` is explicit and oversize
    sequences are not dropped here -- rely on ``FirstFitPackingConfig`` (which
    drops ``len(input_ids) > max_context_length``) and on sizing the length spec
    to the context window.

    Each DP rank derives an independent, reproducible, checkpoint-resumable
    stream from ``config.seed + policy.seed + policy.dp_rank``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        length_spec: LengthSpec
        vocab_size: int
        seed: int = 0

        def __post_init__(self) -> None:
            if self.vocab_size < 1:
                raise ValueError("vocab_size must be >= 1")

    def __init__(
        self,
        config: "SyntheticSource.Config",
        *,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> None:
        super().__init__()
        self._length_spec = config.length_spec
        self._vocab_size = config.vocab_size
        self._seed = (
            config.seed
            + dataset_iteration_policy.seed
            + dataset_iteration_policy.dp_rank
        )

    def __iter__(self) -> grain.DatasetIterator:
        return _SyntheticIterator(self._length_spec, self._seed, self._vocab_size)


class _SyntheticIterator(grain.DatasetIterator):
    """Exposes the sampling RNG to Grain checkpoint recursion.

    Lengths are drawn in chunks so per-record NumPy dispatch does not distort
    throughput/DP-balance measurements. One RNG stream feeds both length and
    token draws, so the bit generator state plus the undrawn length buffer fully
    describe the stream position.
    """

    _buffer_size = 1024

    def __init__(self, length_spec: LengthSpec, seed: int, vocab_size: int) -> None:
        super().__init__()
        self._length_spec = length_spec
        self._vocab_size = vocab_size
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._buffer = np.empty(0, dtype=np.int64)
        self._offset = 0

    def __next__(self) -> TextSequence:
        if self._offset >= self._buffer.size:
            self._buffer = self._length_spec.sample(self._rng, self._buffer_size)
            self._offset = 0
        length = int(self._buffer[self._offset])
        self._offset += 1
        ids = self._rng.integers(0, self._vocab_size, size=length + 1, dtype=np.int64)
        return TextSequence(input_ids=ids[:-1], labels=ids[1:])

    def get_state(self) -> dict:
        return {
            "bit_generator": self._rng.bit_generator.state,
            "remaining": self._buffer[self._offset :].tolist(),
        }

    def set_state(self, state: dict) -> None:
        self._rng.bit_generator.state = state["bit_generator"]
        self._buffer = np.array(state["remaining"], dtype=np.int64)
        self._offset = 0


def synthetic_dataloader(
    *,
    length_spec: LengthSpec,
    vocab_size: int,
    seed: int = 0,
    num_packing_bins: int = 8,
) -> GrainDataLoader.Config:
    """Wire a synthetic source into a FirstFit-packed dataloader config.

    Returns a ready ``GrainDataLoader.Config`` for the common experiment case:
    ``SyntheticSource`` -> ``SingleDatasetConfig`` -> ``FirstFitPackingConfig``.

    Args:
        length_spec: Distribution to draw sequence lengths from.
        vocab_size: Vocab size for the random token ids; required and
            independent of the tokenizer. Set it to your model's vocab size.
        seed: Base seed. Each DP rank derives an independent, reproducible
            stream from ``seed`` + loader policy seed + ``dp_rank``, and the
            source is checkpoint-resumable.
        num_packing_bins: Number of FirstFit packing bins.

    Notes:
        - Content is random and irrelevant for perf/load-balancing; only the
          length distribution matters. Because tokens are random, MoE routing is
          effectively random and expert load is not balanced; for deterministic
          per-expert load in MoE perf/DP-balance runs, set the training config's
          ``moe_force_load_balance=True`` (round-robin token assignment,
          debug-only).
        - Lengths are exact (token ids are generated directly, so there is no
          retokenization drift).
        - The source does not drop oversize sequences; ``FirstFitPackingConfig``
          drops ``len(input_ids) > max_context_length``, so size ``length_spec``
          to your ``max_context_length`` -- otherwise long draws are silently
          dropped and the realized distribution is biased.
        - Inspect a spec without training via
          ``python -m scripts.preview_synthetic_lengths --spec spec.json --dp 8``.
    """
    synthetic_ds = SingleDatasetConfig(
        source=SyntheticSource.Config(
            length_spec=length_spec,
            vocab_size=vocab_size,
            seed=seed,
        ),
    )
    synthetic_packed_ds = FirstFitPackingConfig(
        dataset=synthetic_ds,
        num_packing_bins=num_packing_bins,
    )
    return GrainDataLoader.Config(dataset=synthetic_packed_ds, shuffle=False)
