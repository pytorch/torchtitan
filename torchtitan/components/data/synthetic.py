# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""On-the-fly synthetic token sequences with a configurable length distribution.

Intended for DP load-balancing and throughput experiments where token *content*
is irrelevant and only the sequence-length distribution matters.
"""

import math
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import grain.python as grain
import numpy as np
from torchtitan.components.data.dataset import SampleProcessor, TextSequence
from torchtitan.components.data.types import DatasetBuildContext, DatasetIterationPolicy
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


@dataclass(frozen=True, kw_only=True, slots=True)
class ParametricLengthSpec:
    """Named distribution, rounded to int and clamped to [min_len, max_len].

    - uniform: integer uniform on [min_len, max_len]
    - normal: Gaussian(mean, std)
    - lognormal: exp(Normal(mean, std)); mean/std are the underlying normal's
    - zipf: Zipf(alpha) with alpha > 1
    """

    kind: Literal["uniform", "normal", "lognormal", "zipf"]
    min_len: int
    max_len: int
    mean: float | None = None
    std: float | None = None
    alpha: float | None = None

    def __post_init__(self) -> None:
        if self.min_len < 1 or self.max_len < self.min_len:
            raise ValueError("require 1 <= min_len <= max_len")
        if self.kind in ("normal", "lognormal"):
            if self.mean is None or self.std is None:
                raise ValueError(f"{self.kind} requires mean and std")
            if self.std <= 0:
                raise ValueError("std must be positive")
        elif self.kind == "zipf":
            if self.alpha is None or self.alpha <= 1.0:
                raise ValueError("zipf requires alpha > 1.0")
        elif self.kind != "uniform":
            raise ValueError(f"unknown kind {self.kind!r}")

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        if self.kind == "uniform":
            raw = rng.integers(self.min_len, self.max_len + 1, size=size)
        elif self.kind == "normal":
            raw = rng.normal(self.mean, self.std, size=size)
        elif self.kind == "lognormal":
            raw = rng.lognormal(self.mean, self.std, size=size)
        else:  # zipf
            raw = rng.zipf(self.alpha, size=size)
        lengths = np.rint(np.asarray(raw, dtype=np.float64)).astype(np.int64)
        return np.clip(lengths, self.min_len, self.max_len)


class SyntheticLengthSource(Configurable, grain.IterDataset):
    """Infinite stream of ``{"length": L}`` records drawn from a length spec.

    Each DP rank derives an independent, reproducible stream from
    ``config.seed + policy.seed + policy.dp_rank``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        length_spec: LengthSpec
        seed: int = 0

    def __init__(
        self,
        config: "SyntheticLengthSource.Config",
        *,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> None:
        super().__init__()
        self._length_spec = config.length_spec
        self._seed = (
            config.seed
            + dataset_iteration_policy.seed
            + dataset_iteration_policy.dp_rank
        )

    def __iter__(self) -> grain.DatasetIterator:
        return _SyntheticLengthIterator(self._length_spec, self._seed)


class _SyntheticLengthIterator(grain.DatasetIterator):
    """Exposes the length-sampling RNG to Grain checkpoint recursion.

    Lengths are drawn in chunks so per-record NumPy dispatch does not distort
    throughput/DP-balance measurements.
    """

    _CHUNK = 1024

    def __init__(self, length_spec: LengthSpec, seed: int) -> None:
        super().__init__()
        self._length_spec = length_spec
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._buffer = np.empty(0, dtype=np.int64)
        self._offset = 0

    def __next__(self) -> dict[str, int]:
        if self._offset >= self._buffer.size:
            self._buffer = self._length_spec.sample(self._rng, self._CHUNK)
            self._offset = 0
        length = int(self._buffer[self._offset])
        self._offset += 1
        return {"length": length}

    def get_state(self) -> dict:
        return {
            "bit_generator": self._rng.bit_generator.state,
            "remaining": self._buffer[self._offset :].tolist(),
        }

    def set_state(self, state: dict) -> None:
        self._rng.bit_generator.state = state["bit_generator"]
        self._buffer = np.array(state["remaining"], dtype=np.int64)
        self._offset = 0


def _should_drop(length: int, max_context_length: int) -> bool:
    return length < 1 or length + 1 > max_context_length


def _split_next_token(ids: np.ndarray) -> TextSequence:
    return TextSequence(input_ids=ids[:-1], labels=ids[1:])


class RandomTokenProcessor(SampleProcessor):
    """Realizes ``{"length": L}`` into random-token next-token pairs."""

    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        vocab_size: int | None = None
        """Explicit vocab; falls back to context.tokenizer.get_vocab_size()."""

    def __init__(
        self, config: "RandomTokenProcessor.Config", *, context: DatasetBuildContext
    ) -> None:
        self._max_context_length = context.max_context_length
        self._vocab_size = (
            config.vocab_size
            if config.vocab_size is not None
            else context.tokenizer.get_vocab_size()
        )
        if self._vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")

    def __call__(self, sample: dict, rng: np.random.Generator) -> TextSequence | None:
        length = int(sample["length"])
        if _should_drop(length, self._max_context_length):
            return None
        ids = rng.integers(0, self._vocab_size, size=length + 1, dtype=np.int64)
        return _split_next_token(ids)


class ConstantTokenProcessor(SampleProcessor):
    """Fills each sequence with a constant token id (no RNG). Dense-only.

    Constant tokens make MoE routing degenerate (all tokens hit one expert);
    use RandomTokenProcessor for MoE. Pick a non-special id (avoid pad/eos) so
    downstream masking doesn't blank the batch.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        constant_token_id: int = 0

    def __init__(
        self,
        config: "ConstantTokenProcessor.Config",
        *,
        context: DatasetBuildContext,
    ) -> None:
        if config.constant_token_id < 0:
            raise ValueError("constant_token_id must be >= 0")
        self._max_context_length = context.max_context_length
        self._token_id = config.constant_token_id

    def __call__(self, sample: dict, rng: np.random.Generator) -> TextSequence | None:
        length = int(sample["length"])
        if _should_drop(length, self._max_context_length):
            return None
        ids = np.full(length + 1, self._token_id, dtype=np.int64)
        return _split_next_token(ids)
