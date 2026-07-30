# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stateful packing recipes for tokenized documents."""

from dataclasses import dataclass

import grain.python as grain
import numpy as np

from torchtitan.components.data.dataset import DatasetConfig, TextSequence
from torchtitan.components.data.types import DatasetBuildContext, DatasetIterationPolicy
from torchtitan.components.loss import IGNORE_INDEX


def _text_sequence_to_packing_dict(
    sample: TextSequence,
) -> dict[str, np.ndarray]:
    """Expose token-aligned arrays consumed by the packing operators."""
    positions = sample.positions
    if positions is None:
        positions = np.arange(len(sample.input_ids), dtype=np.int64)
    return {
        "input_ids": np.asarray(sample.input_ids),
        "labels": np.asarray(sample.labels),
        "positions": np.asarray(positions),
    }


def _packed_dict_to_text_sequence(
    packed: dict[str, np.ndarray],
) -> TextSequence:
    """Finalize packed text by masking padding and canonicalizing positions."""
    labels = np.asarray(packed["labels"]).copy()
    labels[np.asarray(packed["input_ids_segment_ids"]) == 0] = IGNORE_INDEX

    boundaries = np.asarray(packed["positions"]) == 0
    token_indices = np.arange(len(boundaries), dtype=np.int64)
    segment_starts = np.maximum.accumulate(np.where(boundaries, token_indices, 0))

    return TextSequence(
        input_ids=np.asarray(packed["input_ids"]),
        labels=labels,
        positions=token_indices - segment_starts,
    )


@dataclass(frozen=True, kw_only=True, slots=True)
class ConcatThenSplitPackingConfig:
    """Concatenates tokenized documents and splits fixed-length rows."""

    dataset: DatasetConfig

    def build(
        self,
        *,
        context: DatasetBuildContext,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> grain.IterDataset:
        dataset = self.dataset.build(
            context=context,
            dataset_iteration_policy=dataset_iteration_policy,
        )
        dataset = dataset.map(_text_sequence_to_packing_dict)
        if isinstance(dataset, grain.MapDataset):
            dataset = dataset.to_iter_dataset(read_options=context.read_options)
        # TODO(data-global-pack-plan): Each DP rank packs its own documents, so
        # changing the DP size changes packed rows. Plan rows before DP sharding.
        # TODO(data-overflow-policy): Concat-then-split chunks long documents, while
        # first-fit drops them. Expose a shared split, truncate, or drop policy.
        dataset = grain.experimental.ConcatThenSplitIterDataset(
            dataset,
            length_struct={
                "input_ids": context.seq_len,
                "labels": context.seq_len,
                "positions": context.seq_len,
            },
        )
        return dataset.map(_packed_dict_to_text_sequence)


@dataclass(frozen=True, kw_only=True, slots=True)
class FirstFitPackingConfig:
    """Packs whole tokenized documents into fixed-length rows."""

    dataset: DatasetConfig
    num_packing_bins: int = 8
    """Candidate rows kept open; more bins can reduce padding but buffer more samples."""

    def __post_init__(self) -> None:
        if self.num_packing_bins <= 0:
            raise ValueError("num_packing_bins must be positive")

    def build(
        self,
        *,
        context: DatasetBuildContext,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> grain.IterDataset:
        dataset = self.dataset.build(
            context=context,
            dataset_iteration_policy=dataset_iteration_policy,
        )
        dataset = dataset.filter(
            lambda sample: len(sample.input_ids) <= context.seq_len
        )
        dataset = dataset.map(_text_sequence_to_packing_dict)
        if isinstance(dataset, grain.MapDataset):
            dataset = dataset.to_iter_dataset(read_options=context.read_options)
        # TODO(data-global-pack-plan): Each DP rank packs its own documents, so
        # changing the DP size changes packed rows. Plan rows before DP sharding.
        dataset = grain.experimental.FirstFitPackIterDataset(
            dataset,
            length_struct={
                "input_ids": context.seq_len,
                "labels": context.seq_len,
                "positions": context.seq_len,
            },
            padding_struct={
                "input_ids": 0,
                "labels": IGNORE_INDEX,
                "positions": 0,
            },
            num_packing_bins=self.num_packing_bins,
            meta_features=("labels", "positions"),
            seed=dataset_iteration_policy.seed,
            shuffle_bins=dataset_iteration_policy.shuffle,
        )
        return dataset.map(_packed_dict_to_text_sequence)
