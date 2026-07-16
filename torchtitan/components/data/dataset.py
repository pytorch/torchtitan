# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dataset composition for the grain dataloader.

Mental model: common text configs lower to a Grain `MapDataset`; custom configs may
return an `IterDataset`.

    leaf:    source -> process -> filters          (SingleDatasetConfig)
    compose: combine_fn(leaves)                    (MultiDatasetConfig)
    global:  seed shuffle -> DP stride shard -> repeat   (finish_map_dataset, applied once)

Example:

    dataset_config = weighted_interleave([
        (SingleDatasetConfig(source=JsonlSourceConfig(patterns=("math_*.jsonl",)),
                             sample_processor=text_row_to_token_sample), 2.0),
        (SingleDatasetConfig(source=JsonlSourceConfig(patterns=("code_*.jsonl",)),
                             sample_processor=text_row_to_token_sample), 1.0),
    ])
    # -> deterministic 2:1 document interleave, globally shuffled, sharded per DP rank
"""

import hashlib
import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import grain.python as grain
import numpy as np

from torchtitan.components.data.sources import SourceConfig
from torchtitan.components.tokenizer import BaseTokenizer


@dataclass(frozen=True, kw_only=True, slots=True)
class DataRuntime:
    """Training objects and sizes available while a dataset recipe is built."""

    seq_len: int
    local_batch_size: int
    tokenizer: BaseTokenizer | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class BuildOptions:
    """Deterministic build parameters applied globally, once, by `finish_map_dataset`."""

    seed: int = 0
    shuffle: bool = True
    infinite: bool = True
    dp_rank: int = 0
    dp_world_size: int = 1


class DatasetConfig(Protocol):
    """Anything that lowers to a grain dataset; the escape hatch for custom pipelines."""

    def build(
        self, *, runtime: DataRuntime, options: BuildOptions
    ) -> grain.MapDataset | grain.IterDataset:
        ...

    def fingerprint(self) -> str:
        ...


@dataclass(frozen=True, kw_only=True, slots=True)
class TokenSample:
    """One document after processing: token ids plus which positions train the loss.

    Example:

        # SFT: prompt tokens masked out, response tokens trained
        TokenSample(
            token_ids=np.array([1, 15, 27, 99, 42, 2]),
            loss_mask=np.array([False, False, False, True, True, True]),
        )
    """

    token_ids: np.ndarray  # [doc_tokens]
    loss_mask: np.ndarray  # [doc_tokens] bool; True = train on this position's label


# A processor maps a raw source row to the sample expected by downstream stages.
# Built-in processors and filters must be deterministic functions of their arguments.
SampleProcessor = Callable[..., Any]
FilterFn = Callable[..., bool]


def fingerprint_parts(*parts: str) -> str:
    """sha256 over NUL-separated parts (separators prevent ("ab","c")/("a","bc") collisions)."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def callable_fingerprint(fn: Callable[..., Any] | None) -> str:
    """Stable identity of a configured callable for resume fingerprints.

    Configured callable objects (mixers, packers with values) must implement
    `fingerprint()` naming their configuration; plain functions use module:qualname;
    lambdas add their line number so two same-file lambdas differ. Renaming or moving
    a function invalidates checkpoints on purpose: a conservative false rejection is
    safer than resuming grain state into a stream with different semantics.
    """
    if fn is None:
        return "none"
    fingerprint = getattr(fn, "fingerprint", None)
    if fingerprint is not None:
        return fingerprint_parts(
            f"{type(fn).__module__}:{type(fn).__qualname__}", fingerprint()
        )
    if inspect.isfunction(fn) or inspect.ismethod(fn):
        name = f"{fn.__module__}:{fn.__qualname__}"
        if fn.__name__ == "<lambda>":
            return f"{name}:{fn.__code__.co_firstlineno}"
        return name
    raise TypeError(
        f"configured callable {type(fn).__qualname__} must implement fingerprint()"
    )


def _bind_runtime(fn: Callable[..., Any], runtime: DataRuntime) -> Callable[[Any], Any]:
    """Bind `(row)` or `(row, runtime)` by REQUIRED positional arity.

    Example:

        def a(row): ...                    # called as a(row)
        def b(row, runtime): ...           # called as b(row, runtime)
        def c(row, suffix="!"): ...        # called as c(row) — optionals are not runtime
    """
    positional = [
        parameter
        for parameter in inspect.signature(fn).parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    required = [p for p in positional if p.default is inspect.Parameter.empty]
    if len(required) == 1:
        return fn
    if len(required) == 2 and len(positional) == 2:
        return lambda row: fn(row, runtime)
    raise TypeError(f"{getattr(fn, '__name__', fn)} must take (row) or (row, runtime)")


def finish_map_dataset(
    dataset: grain.MapDataset, *, options: BuildOptions
) -> grain.MapDataset:
    """Apply the global stages exactly once: seed shuffle -> DP stride shard -> repeat.

    Shuffle reseeds per epoch by construction; sharding after shuffle gives each rank a
    disjoint 1/N of a global permutation (verified: research/opus/probes/firstfit_shard_probe.py).
    """
    if options.shuffle:
        dataset = dataset.shuffle(seed=options.seed)
    if options.dp_world_size > 1:
        # a rank with zero rows stops producing batches and hangs SPMD collectives
        if len(dataset) < options.dp_world_size:
            raise ValueError(
                f"dataset has {len(dataset)} rows, fewer than "
                f"dp_world_size={options.dp_world_size}"
            )
        dataset = dataset[options.dp_rank :: options.dp_world_size]
    if options.infinite:
        dataset = dataset.repeat()
    return dataset


@dataclass(frozen=True, kw_only=True, slots=True)
class SingleDatasetConfig:
    """One source with its row-level processing.

    Example:

        def c4_row_to_token_sample(row: dict, runtime: DataRuntime) -> TokenSample:
            ids = np.asarray(runtime.tokenizer.encode(row["text"], add_bos=True, add_eos=True))
            return TokenSample(token_ids=ids, loss_mask=np.ones(ids.shape, dtype=np.bool_))

        SingleDatasetConfig(
            source=JsonlSourceConfig(patterns=("data.json",)),
            sample_processor=c4_row_to_token_sample,
        )
    """

    source: SourceConfig
    sample_processor: SampleProcessor | None = None
    sample_filters: tuple[FilterFn, ...] = ()

    def build_processed_dataset(self, *, runtime: DataRuntime) -> grain.MapDataset:
        """Build source -> sample processor -> sample filters, without global stages."""
        dataset = grain.MapDataset.source(self.source.build())
        if self.sample_processor is not None:
            dataset = dataset.map(_bind_runtime(self.sample_processor, runtime))
        for filter_fn in self.sample_filters:
            dataset = dataset.filter(_bind_runtime(filter_fn, runtime))
        return dataset

    def build(self, *, runtime: DataRuntime, options: BuildOptions) -> grain.MapDataset:
        return finish_map_dataset(
            self.build_processed_dataset(runtime=runtime),
            options=options,
        )

    def fingerprint(self) -> str:
        return fingerprint_parts(
            type(self).__qualname__,
            self.source.fingerprint(),
            callable_fingerprint(self.sample_processor),
            *(callable_fingerprint(filter_fn) for filter_fn in self.sample_filters),
        )


# A combiner receives child configs so it can inspect source metadata and build finite
# selected views. It must build each child it uses via
# `build_processed_dataset` and must not apply global shuffle/shard/repeat.
DatasetCombineFn = Callable[
    [tuple[SingleDatasetConfig, ...], DataRuntime, BuildOptions], grain.MapDataset
]


@dataclass(frozen=True, kw_only=True, slots=True)
class MultiDatasetConfig:
    """Several sources combined before global shuffle, sharding, and repetition."""

    datasets: tuple[SingleDatasetConfig, ...]
    combine_fn: DatasetCombineFn

    def build(self, *, runtime: DataRuntime, options: BuildOptions) -> grain.MapDataset:
        combined = self.combine_fn(self.datasets, runtime, options)
        return finish_map_dataset(combined, options=options)

    def fingerprint(self) -> str:
        return fingerprint_parts(
            type(self).__qualname__,
            callable_fingerprint(self.combine_fn),
            *(dataset.fingerprint() for dataset in self.datasets),
        )


@dataclass(frozen=True, slots=True)
class WeightedDatasetInterleave:
    """Deterministic document-proportion interleave (grain index arithmetic, not RNG).

    Example:

        WeightedDatasetInterleave(weights=(2.0, 1.0))
        # children [a0,a1,a2,...], [b0,b1,...] -> [a0, a1, b0, a2, a3, b1, ...]
    """

    weights: tuple[float, ...]

    def __call__(
        self,
        datasets: tuple[SingleDatasetConfig, ...],
        runtime: DataRuntime,
        options: BuildOptions,
    ) -> grain.MapDataset:
        del options
        if len(datasets) != len(self.weights) or any(w <= 0 for w in self.weights):
            raise ValueError("one positive weight is required per dataset")
        children = [
            dataset.build_processed_dataset(runtime=runtime) for dataset in datasets
        ]
        total = sum(self.weights)
        return grain.MapDataset.mix(
            children, weights=[weight / total for weight in self.weights]
        )

    def fingerprint(self) -> str:
        return fingerprint_parts(
            type(self).__qualname__, *(repr(weight) for weight in self.weights)
        )


def concatenate_datasets(
    datasets: tuple[SingleDatasetConfig, ...],
    runtime: DataRuntime,
    options: BuildOptions,
) -> grain.MapDataset:
    """All of the first dataset, then the second, and so on."""
    del options
    return grain.MapDataset.concatenate(
        [dataset.build_processed_dataset(runtime=runtime) for dataset in datasets]
    )


def weighted_interleave(
    datasets_and_weights: Sequence[tuple[SingleDatasetConfig, float]],
) -> MultiDatasetConfig:
    """Weighted deterministic interleave of several datasets.

    Example:

        weighted_interleave([(math_ds, 2.0), (code_ds, 1.0)])
        # 2 math docs : 1 code doc, exactly
    """
    return MultiDatasetConfig(
        datasets=tuple(dataset for dataset, _ in datasets_and_weights),
        combine_fn=WeightedDatasetInterleave(
            weights=tuple(float(weight) for _, weight in datasets_and_weights)
        ),
    )


def concat(datasets: Sequence[SingleDatasetConfig]) -> MultiDatasetConfig:
    """Consume finite datasets one after another, in order."""
    return MultiDatasetConfig(
        datasets=tuple(datasets),
        combine_fn=concatenate_datasets,
    )
