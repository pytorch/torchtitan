# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Composable dataset recipes backed by Grain."""

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, cast, Protocol, TypeAlias

import grain.python as grain
import numpy as np

from torchtitan.components.data.sources import RandomAccessDataSource, SourceConfig
from torchtitan.components.data.types import DatasetBuildContext, DatasetIterationPolicy
from torchtitan.config import Configurable


GrainDataset: TypeAlias = grain.MapDataset | grain.IterDataset


class DatasetConfig(Protocol):
    """Builds one node of a Grain dataset graph."""

    def build(
        self,
        *,
        context: DatasetBuildContext,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> GrainDataset:
        ...


@dataclass(frozen=True, kw_only=True, slots=True)
class TextSequence:
    """Next-token-aligned text preserved through composition and packing.

    NOTE: It is the dataset's processor responsibility to shift tokens into
    aligned input and label pairs. The trainer does not do it.
    """

    input_ids: np.ndarray
    """Tokens provided to the model."""
    labels: np.ndarray
    """Target for each input token, with `IGNORE_INDEX` where loss is disabled."""
    positions: np.ndarray | None = None
    """Per-token positions; `None` until packing or collation materializes them."""


class SampleProcessor(Configurable, ABC):
    """Configured row processor using Grain-provided deterministic randomness."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    @abstractmethod
    def __call__(
        self,
        sample: Any,
        rng: np.random.Generator,
    ) -> Any:
        ...


@dataclass(frozen=True, kw_only=True, slots=True)
class SingleDatasetConfig:
    """Build one dataset from a source and row-level transforms.

    `pre_filters` run before `processor`; `post_filters` run afterward.
    This node owns leaf shuffle, repeat, and effective-DP sharding.
    """

    source: SourceConfig
    pre_filters: tuple[Callable[[Any], bool], ...] = ()
    processor: SampleProcessor.Config | None = None
    post_filters: tuple[Callable[[Any], bool], ...] = ()

    def build(
        self,
        *,
        context: DatasetBuildContext,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> GrainDataset:
        source = self.source.build(
            dataset_iteration_policy=dataset_iteration_policy,
        )
        if isinstance(source, RandomAccessDataSource):
            dataset: GrainDataset = grain.MapDataset.source(source)
        elif isinstance(source, grain.IterDataset):
            dataset = source
        else:
            raise TypeError(
                "source must be a RandomAccessDataSource or grain.IterDataset"
            )

        if isinstance(dataset, grain.MapDataset):
            return self._build_map_dataset(
                dataset,
                context=context,
                dataset_iteration_policy=dataset_iteration_policy,
            )
        return self._build_iter_dataset(
            dataset,
            context=context,
            dataset_iteration_policy=dataset_iteration_policy,
        )

    def _build_map_dataset(
        self,
        dataset: grain.MapDataset,
        *,
        context: DatasetBuildContext,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> grain.MapDataset:
        """Process globally indexed rows before shuffle, sharding, and repeat."""
        # Filter raw rows.
        for filter_fn in self.pre_filters:
            dataset = dataset.filter(filter_fn)

        # Process rows into training samples.
        if self.processor is not None:
            dataset = dataset.random_map(
                self.processor.build(context=context),
                seed=dataset_iteration_policy.seed,
            )

        # Filter processed samples.
        for filter_fn in self.post_filters:
            dataset = dataset.filter(filter_fn)

        # Shuffle globally, then give each DP rank a disjoint slice.
        if dataset_iteration_policy.shuffle:
            dataset = dataset.shuffle(seed=dataset_iteration_policy.seed)
        if len(dataset) < dataset_iteration_policy.dp_world_size:
            raise ValueError(
                f"dataset has {len(dataset)} rows, fewer than "
                f"dp_world_size={dataset_iteration_policy.dp_world_size}"
            )
        shard_size, remainder = divmod(
            len(dataset), dataset_iteration_policy.dp_world_size
        )
        shard_start = dataset_iteration_policy.dp_rank * shard_size + min(
            dataset_iteration_policy.dp_rank, remainder
        )
        shard_stop = shard_start + shard_size
        if dataset_iteration_policy.dp_rank < remainder:
            shard_stop += 1
        dataset = dataset[shard_start:shard_stop]
        if dataset_iteration_policy.repeat:
            # Grain preserves the epoch through sliced map indices, so the
            # upstream shuffle uses seed + epoch on each repeat.
            dataset = dataset.repeat()
        return dataset

    def _build_iter_dataset(
        self,
        dataset: grain.IterDataset,
        *,
        context: DatasetBuildContext,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> grain.IterDataset:
        """Shuffle streaming rows before row processing."""
        # Shuffle raw stream rows.
        if dataset_iteration_policy.shuffle:
            dataset = grain.experimental.WindowShuffleIterDataset(
                dataset,
                window_size=dataset_iteration_policy.streaming_shuffle_buffer_size,
                seed=dataset_iteration_policy.seed,
            )

        # Filter and process rows in stream order.
        for filter_fn in self.pre_filters:
            dataset = dataset.filter(filter_fn)
        if self.processor is not None:
            dataset = dataset.random_map(
                self.processor.build(context=context),
                seed=dataset_iteration_policy.seed + dataset_iteration_policy.dp_rank,
            )
        for filter_fn in self.post_filters:
            dataset = dataset.filter(filter_fn)
        return dataset


@dataclass(frozen=True, kw_only=True, slots=True)
class WeightedDataset:
    """A dataset and its relative selection weight."""

    dataset: DatasetConfig
    weight: float = 1.0


@dataclass(frozen=True, kw_only=True, slots=True)
class DatasetMixConfig:
    """Interleave weighted children.

    `MapDataset.filter` leaves rejected indices as `None`, so the all-map path
    weights attempted indices rather than accepted samples. Otherwise, weights
    select elements emitted by each iterable child. The child defines the
    element: mixing `TextSequence` children weights documents; mixing packed
    fixed-length children weights physical tokens.

    With `repeat=True` the mix is infinite. With `repeat=False` the mix stops
    at the first exhausted child, so larger children are not fully covered.
    """

    datasets: tuple[WeightedDataset, ...]

    def build(
        self,
        *,
        context: DatasetBuildContext,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> GrainDataset:
        if not self.datasets or any(
            not math.isfinite(item.weight) or item.weight <= 0 for item in self.datasets
        ):
            raise ValueError(
                "DatasetMixConfig requires finite, positive-weight datasets"
            )
        children = [
            item.dataset.build(
                context=context,
                # Inserting or reordering a child reseeds every later child.
                dataset_iteration_policy=replace(
                    dataset_iteration_policy,
                    seed=dataset_iteration_policy.seed + index,
                ),
            )
            for index, item in enumerate(self.datasets)
        ]
        weights = [item.weight for item in self.datasets]
        # TODO(data-token-weighted-mix): Support source weights by token count,
        # not only emitted examples or packed rows. Checkpoint running estimates.
        if all(isinstance(child, grain.MapDataset) for child in children):
            return grain.MapDataset.mix(
                cast(list[grain.MapDataset], children),
                weights=weights,
            )
        children = [
            child.to_iter_dataset(read_options=context.read_options)
            if isinstance(child, grain.MapDataset)
            else child
            for child in children
        ]
        return grain.IterDataset.mix(children, weights=weights)


@dataclass(frozen=True, kw_only=True, slots=True)
class DatasetConcatConfig:
    """Concatenates finite children before global shuffle and DP sharding."""

    datasets: tuple[DatasetConfig, ...]

    def build(
        self,
        *,
        context: DatasetBuildContext,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> grain.MapDataset:
        child_iteration_policy = replace(
            dataset_iteration_policy,
            shuffle=False,
            repeat=False,
            dp_rank=0,
            dp_world_size=1,
        )
        children = [
            dataset.build(
                context=context,
                dataset_iteration_policy=child_iteration_policy,
            )
            for dataset in self.datasets
        ]

        if not children or not all(
            isinstance(child, grain.MapDataset) for child in children
        ):
            raise TypeError("DatasetConcatConfig requires map-style children")

        dataset = grain.MapDataset.concatenate(cast(list[grain.MapDataset], children))

        if dataset_iteration_policy.shuffle:
            dataset = dataset.shuffle(seed=dataset_iteration_policy.seed)

        if len(dataset) < dataset_iteration_policy.dp_world_size:
            raise ValueError(
                f"dataset has {len(dataset)} rows, fewer than "
                f"dp_world_size={dataset_iteration_policy.dp_world_size}"
            )

        shard_size, remainder = divmod(
            len(dataset), dataset_iteration_policy.dp_world_size
        )
        shard_start = dataset_iteration_policy.dp_rank * shard_size + min(
            dataset_iteration_policy.dp_rank, remainder
        )
        shard_stop = shard_start + shard_size
        if dataset_iteration_policy.dp_rank < remainder:
            shard_stop += 1
        dataset = dataset[shard_start:shard_stop]

        if dataset_iteration_policy.repeat:
            dataset = dataset.repeat()
        return dataset
