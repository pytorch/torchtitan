# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stateful packing recipes for tokenized documents."""

from dataclasses import dataclass

import grain.python as grain
import numpy as np
import torch

from torchtitan.components.data.dataset import (
    DatasetBuildContext,
    DatasetConfig,
    DatasetIterationPolicy,
    TokenSequence,
)
from torchtitan.components.loss import IGNORE_INDEX


def _token_sequence_to_input_ids_and_labels(
    sample: TokenSequence,
) -> dict[str, np.ndarray]:
    """Shift one document before packing so labels never cross documents.

    Example:

        token_ids=[5, 6, 7, 8], loss_mask=[1, 1, 1, 1]
        # -> input_ids=[5, 6, 7], labels=[6, 7, 8]
    """
    token_ids = np.asarray(sample.token_ids, dtype=np.int64)
    loss_mask = np.asarray(sample.loss_mask, dtype=np.bool_)
    if token_ids.ndim != 1 or loss_mask.shape != token_ids.shape:
        raise ValueError("token_ids and loss_mask must be aligned 1-D arrays")
    return {
        "input_ids": token_ids[:-1],
        "labels": np.where(loss_mask[1:], token_ids[1:], np.int64(IGNORE_INDEX)),
    }


def _packed_to_input_and_labels(
    packed: dict[str, np.ndarray],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Convert one packed row to the trainer contract; padding is loss-masked."""
    padding = np.asarray(packed["input_ids_segment_ids"]) == 0
    labels = np.where(padding, np.int64(IGNORE_INDEX), np.asarray(packed["labels"]))
    inputs = {
        "input": torch.as_tensor(np.asarray(packed["input_ids"]), dtype=torch.long),
        "positions": torch.as_tensor(
            np.asarray(packed["input_ids_positions"]), dtype=torch.long
        ),
    }
    return inputs, torch.as_tensor(labels, dtype=torch.long)


@dataclass(frozen=True, kw_only=True, slots=True)
class ConcatThenSplitPackingConfig:
    """Concatenates tokenized documents and splits fixed-length rows."""

    dataset: DatasetConfig

    def build(
        self, *, context: DatasetBuildContext, iteration: DatasetIterationPolicy
    ) -> grain.IterDataset:
        dataset = self.dataset.build(context=context, iteration=iteration)
        dataset = dataset.filter(lambda sample: len(sample.token_ids) >= 2)
        dataset = dataset.map(_token_sequence_to_input_ids_and_labels)
        if isinstance(dataset, grain.MapDataset):
            dataset = dataset.to_iter_dataset(read_options=context.read_options)
        # TODO(data-global-pack-plan): Plan packed rows before effective-DP sharding
        # when measurements justify shared length metadata and a cached plan.
        # TODO(data-overflow-policy): Make long-document handling configurable
        # (chunk, truncate, or drop); concat-then-split always chunks today.
        dataset = grain.experimental.ConcatThenSplitIterDataset(
            dataset,
            length_struct={"input_ids": context.seq_len, "labels": context.seq_len},
            meta_features=(),
        )
        return dataset.map(_packed_to_input_and_labels)


@dataclass(frozen=True, kw_only=True, slots=True)
class FirstFitPackingConfig:
    """Packs whole tokenized documents into fixed-length rows."""

    dataset: DatasetConfig
    num_packing_bins: int | None = None

    def __post_init__(self) -> None:
        if self.num_packing_bins is not None and self.num_packing_bins <= 0:
            raise ValueError("num_packing_bins must be positive")

    def build(
        self, *, context: DatasetBuildContext, iteration: DatasetIterationPolicy
    ) -> grain.IterDataset:
        dataset = self.dataset.build(context=context, iteration=iteration)
        dataset = dataset.filter(lambda sample: len(sample.token_ids) >= 2)
        dataset = dataset.map(_token_sequence_to_input_ids_and_labels)
        if isinstance(dataset, grain.MapDataset):
            dataset = dataset.to_iter_dataset(read_options=context.read_options)
        dataset = dataset.filter(
            lambda packed: len(packed["input_ids"]) <= context.seq_len
        )
        # TODO(data-global-pack-plan): Plan packed rows before effective-DP sharding
        # when measurements justify shared length metadata and a cached plan.
        dataset = grain.experimental.FirstFitPackIterDataset(
            dataset,
            length_struct={"input_ids": context.seq_len, "labels": context.seq_len},
            num_packing_bins=(self.num_packing_bins or context.local_batch_size),
            meta_features=("labels",),
            seed=iteration.seed,
            shuffle_bins=iteration.shuffle,
        )
        return dataset.map(_packed_to_input_and_labels)
