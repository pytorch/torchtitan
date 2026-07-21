# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Composable dataset recipes backed by Grain."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, cast, Protocol, TypeAlias

import grain.python as grain
import numpy as np

from torchtitan.components.data.sources import RandomAccessSource, SourceConfig
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import Configurable


GrainDataset: TypeAlias = grain.MapDataset | grain.IterDataset


@dataclass(frozen=True, kw_only=True, slots=True)
class DatasetBuildContext:
    """Values needed while constructing a dataset graph."""

    tokenizer: BaseTokenizer
    seq_len: int
    local_batch_size: int
    read_options: grain.ReadOptions


@dataclass(frozen=True, kw_only=True, slots=True)
class DatasetIterationPolicy:
    """Controls dataset order, repetition, and data-parallel ownership."""

    seed: int
    shuffle: bool
    repeat: bool
    dp_rank: int
    dp_world_size: int
    streaming_shuffle_window_size: int


class DatasetConfig(Protocol):
    """Builds one node of a Grain dataset graph."""

    def build(
        self, *, context: DatasetBuildContext, iteration: DatasetIterationPolicy
    ) -> GrainDataset:
        ...


@dataclass(frozen=True, kw_only=True, slots=True)
class TokenSequence:
    """One tokenized document and the positions that train the loss."""

    token_ids: np.ndarray
    loss_mask: np.ndarray


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
    """One source with row processing and filtering."""

    source: SourceConfig
    processor: SampleProcessor.Config | None = None
    filters: tuple[Callable[[Any], bool], ...] = ()

    def build(
        self, *, context: DatasetBuildContext, iteration: DatasetIterationPolicy
    ) -> GrainDataset:
        source = self.source.build(
            dp_rank=iteration.dp_rank, dp_world_size=iteration.dp_world_size
        )
        if isinstance(source, grain.IterDataset):
            # Shuffle raw stream rows before processing them into larger training samples.
            dataset = source
            if iteration.repeat:
                dataset = grain.experimental.RepeatIterDataset(dataset)
                # TODO(data-hf-set-epoch): Reshuffle HF shards across repeats.
            if iteration.shuffle:
                dataset = grain.experimental.WindowShuffleIterDataset(
                    dataset,
                    window_size=iteration.streaming_shuffle_window_size,
                    seed=iteration.seed,
                )
            if self.processor is not None:
                dataset = dataset.random_map(
                    self.processor.build(context=context),
                    seed=iteration.seed + iteration.dp_rank,
                )
            for filter_fn in self.filters:
                dataset = dataset.filter(filter_fn)
            return dataset

        if not isinstance(source, RandomAccessSource):
            raise TypeError("source must support len/getitem or be a Grain IterDataset")
        dataset = grain.MapDataset.source(source)
        if self.processor is not None:
            dataset = dataset.random_map(
                self.processor.build(context=context), seed=iteration.seed
            )
        for filter_fn in self.filters:
            dataset = dataset.filter(filter_fn)
        if iteration.shuffle:
            dataset = dataset.shuffle(seed=iteration.seed)
        if len(dataset) < iteration.dp_world_size:
            raise ValueError(
                f"dataset has {len(dataset)} rows, fewer than "
                f"dp_world_size={iteration.dp_world_size}"
            )
        dataset = dataset[iteration.dp_rank :: iteration.dp_world_size]
        if iteration.repeat:
            dataset = dataset.repeat()
        return dataset


@dataclass(frozen=True, kw_only=True, slots=True)
class WeightedDataset:
    """A dataset and its relative selection weight."""

    dataset: DatasetConfig
    weight: float = 1.0


@dataclass(frozen=True, kw_only=True, slots=True)
class DatasetMixConfig:
    """Deterministically interleaves weighted child datasets."""

    datasets: tuple[WeightedDataset, ...]

    def build(
        self, *, context: DatasetBuildContext, iteration: DatasetIterationPolicy
    ) -> GrainDataset:
        if not self.datasets or any(item.weight <= 0 for item in self.datasets):
            raise ValueError("DatasetMixConfig requires positive-weight datasets")
        children = [
            item.dataset.build(
                context=context,
                iteration=replace(iteration, seed=iteration.seed + index),
            )
            for index, item in enumerate(self.datasets)
        ]
        weights = [item.weight for item in self.datasets]
        if all(isinstance(child, grain.MapDataset) for child in children):
            return grain.MapDataset.mix(
                cast(list[grain.MapDataset], children), weights=weights
            )
        if all(isinstance(child, grain.IterDataset) for child in children):
            return grain.IterDataset.mix(
                cast(list[grain.IterDataset], children), weights=weights
            )
        raise TypeError("a mix cannot combine map and iterable child datasets")


@dataclass(frozen=True, kw_only=True, slots=True)
class DatasetConcatConfig:
    """Concatenates finite children before global shuffle and DP sharding."""

    datasets: tuple[DatasetConfig, ...]

    def build(
        self,
        *,
        context: DatasetBuildContext,
        iteration: DatasetIterationPolicy,
    ) -> grain.MapDataset:
        finite = replace(
            iteration,
            shuffle=False,
            repeat=False,
            dp_rank=0,
            dp_world_size=1,
        )
        children = [
            dataset.build(context=context, iteration=finite)
            for dataset in self.datasets
        ]
        if not children or not all(
            isinstance(child, grain.MapDataset) for child in children
        ):
            raise TypeError("DatasetConcatConfig requires map-style children")

        dataset = grain.MapDataset.concatenate(cast(list[grain.MapDataset], children))
        if iteration.shuffle:
            dataset = dataset.shuffle(seed=iteration.seed)
        if len(dataset) < iteration.dp_world_size:
            raise ValueError(
                f"dataset has {len(dataset)} rows, fewer than "
                f"dp_world_size={iteration.dp_world_size}"
            )
        dataset = dataset[iteration.dp_rank :: iteration.dp_world_size]
        if iteration.repeat:
            dataset = dataset.repeat()
        return dataset
