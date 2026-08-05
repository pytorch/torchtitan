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
        dataset = dataset.map(_text_sequence_to_packing_input)
        if isinstance(dataset, grain.MapDataset):
            dataset = dataset.to_iter_dataset(read_options=context.read_options)
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
        return dataset.map(_packing_output_to_text_sequence)


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
        dataset = dataset.map(_text_sequence_to_packing_input)
        if isinstance(dataset, grain.MapDataset):
            dataset = dataset.to_iter_dataset(read_options=context.read_options)
        # TODO(data-global-pack-plan): Consider packing before DP sharding so
        # ranks receive similarly filled rows.
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
        return dataset.map(_packing_output_to_text_sequence)


def _text_sequence_to_packing_input(
    text_sequence: TextSequence,
) -> dict[str, np.ndarray]:
    """Convert a `TextSequence` to the array dictionary expected by text packing.

    Missing positions become `0..num_tokens-1`.
    """
    positions = text_sequence.positions
    if positions is None:
        positions = np.arange(len(text_sequence.input_ids), dtype=np.int64)
    return {
        "input_ids": np.asarray(text_sequence.input_ids),
        "labels": np.asarray(text_sequence.labels),
        "positions": np.asarray(positions),
    }


def _packing_output_to_text_sequence(
    packing_output: dict[str, np.ndarray],
) -> TextSequence:
    """Finalize packed text by masking padding and canonicalizing positions."""
    labels = np.asarray(packing_output["labels"]).copy()
    labels[np.asarray(packing_output["input_ids_segment_ids"]) == 0] = IGNORE_INDEX

    # A zero starts a document. For [0, 1, 2, 0, 1], segment_starts is
    # [0, 0, 0, 3, 3], so subtracting it restores [0, 1, 2, 0, 1].
    boundaries = np.asarray(packing_output["positions"]) == 0
    token_indices = np.arange(len(boundaries), dtype=np.int64)
    segment_starts = np.maximum.accumulate(np.where(boundaries, token_indices, 0))

    return TextSequence(
        input_ids=np.asarray(packing_output["input_ids"]),
        labels=labels,
        positions=token_indices - segment_starts,
    )
