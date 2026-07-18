# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stateful packing recipes for tokenized documents."""

from dataclasses import dataclass
from functools import partial
from typing import Any, TypeAlias

import grain.python as grain
import numpy as np
import torch
from grain import experimental as grain_experimental

from torchtitan.components.data.dataset import (
    BuildOptions,
    DataRuntime,
    DatasetConfig,
    TokenSequence,
)
from torchtitan.components.loss import IGNORE_INDEX


TextTrainingRow: TypeAlias = tuple[dict[str, torch.Tensor], torch.Tensor]


def token_sequence_to_shifted_features(
    sample: TokenSequence,
) -> dict[str, np.ndarray] | None:
    """Shift one document before packing so labels never cross documents.

    Example:

        token_ids=[5, 6, 7, 8], loss_mask=[1, 1, 1, 1]
        # -> input_ids=[5, 6, 7], labels=[6, 7, 8]
    """
    token_ids = np.asarray(sample.token_ids, dtype=np.int64)
    loss_mask = np.asarray(sample.loss_mask, dtype=np.bool_)
    if token_ids.ndim != 1 or loss_mask.shape != token_ids.shape:
        raise ValueError("token_ids and loss_mask must be aligned 1-D arrays")
    if len(token_ids) < 2:
        return None
    return {
        "input_ids": token_ids[:-1],
        "labels": np.where(
            loss_mask[1:],
            token_ids[1:],
            np.int64(IGNORE_INDEX),
        ),
    }


def _is_not_none(value: Any) -> bool:
    return value is not None


def token_rows_to_iter_dataset(
    dataset: grain.MapDataset | grain.IterDataset,
    *,
    runtime: DataRuntime,
) -> grain.IterDataset:
    dataset = dataset.map(token_sequence_to_shifted_features).filter(_is_not_none)
    if isinstance(dataset, grain.MapDataset):
        dataset = dataset.to_iter_dataset(read_options=runtime.read_options)
    return dataset


def packed_features_to_training_row(
    features: dict[str, np.ndarray],
) -> TextTrainingRow:
    """Convert one packed feature row to TorchTitan's trainer-row contract."""
    padding = np.asarray(features["input_ids_segment_ids"]) == 0
    labels = np.where(
        padding,
        np.int64(IGNORE_INDEX),
        np.asarray(features["labels"]),
    )
    inputs = {
        "input": torch.as_tensor(
            np.asarray(features["input_ids"]),
            dtype=torch.long,
        ),
        "positions": torch.as_tensor(
            np.asarray(features["input_ids_positions"]),
            dtype=torch.long,
        ),
    }
    return inputs, torch.as_tensor(labels, dtype=torch.long)


@dataclass(frozen=True, kw_only=True, slots=True)
class ConcatThenSplitPackingConfig:
    """Concatenates tokenized documents and splits fixed-length rows."""

    dataset: DatasetConfig

    def build(
        self,
        *,
        runtime: DataRuntime,
        options: BuildOptions,
    ) -> grain.IterDataset:
        _validate_packing_options(options)
        dataset = self.dataset.build(runtime=runtime, options=options)
        # TODO(data-global-pack-plan): Plan packed rows before effective-DP sharding
        # when measurements justify shared length metadata and a cached plan.
        dataset = token_rows_to_iter_dataset(dataset, runtime=runtime)
        dataset = grain_experimental.ConcatThenSplitIterDataset(
            dataset,
            length_struct={
                "input_ids": runtime.seq_len,
                "labels": runtime.seq_len,
            },
            meta_features=(),
        )
        return dataset.map(packed_features_to_training_row)


@dataclass(frozen=True, kw_only=True, slots=True)
class FirstFitPackingConfig:
    """Packs whole tokenized documents into fixed-length rows."""

    dataset: DatasetConfig
    num_packing_bins: int | None = None

    def __post_init__(self) -> None:
        if self.num_packing_bins is not None and self.num_packing_bins <= 0:
            raise ValueError("num_packing_bins must be positive")

    def build(
        self,
        *,
        runtime: DataRuntime,
        options: BuildOptions,
    ) -> grain.IterDataset:
        _validate_packing_options(options)
        dataset = self.dataset.build(runtime=runtime, options=options)
        # TODO(data-global-pack-plan): Plan packed rows before effective-DP sharding
        # when measurements justify shared length metadata and a cached plan.
        dataset = token_rows_to_iter_dataset(dataset, runtime=runtime)
        dataset = dataset.filter(
            partial(_fits_packed_row, sequence_length=runtime.seq_len)
        )
        dataset = grain_experimental.FirstFitPackIterDataset(
            dataset,
            length_struct={
                "input_ids": runtime.seq_len,
                "labels": runtime.seq_len,
            },
            num_packing_bins=(self.num_packing_bins or runtime.local_batch_size),
            meta_features=("labels",),
            seed=options.seed,
            shuffle_bins=options.shuffle,
        )
        return dataset.map(packed_features_to_training_row)


def _fits_packed_row(
    features: dict[str, np.ndarray],
    *,
    sequence_length: int,
) -> bool:
    return len(features["input_ids"]) <= sequence_length


def _validate_packing_options(options: BuildOptions) -> None:
    if not options.repeat and options.dp_world_size > 1:
        raise ValueError(
            "finite packed datasets are not supported with data parallelism"
        )
