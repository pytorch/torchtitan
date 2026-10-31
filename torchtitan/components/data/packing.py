# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stateful packing recipes for tokenized documents."""

from dataclasses import dataclass
from functools import partial
from typing import Any

import grain.python as grain
import numpy as np

from torchtitan.components.data.dataset import DatasetConfig, TextSequence
from torchtitan.components.data.types import DatasetBuildContext, DatasetIterationPolicy
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.tools.utils import round_up


@dataclass(frozen=True, kw_only=True, slots=True)
class ConcatThenSplitPackingConfig:
    """Concatenates documents, chunking them into fixed-length rows."""

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
        if context.max_num_documents is not None:
            if isinstance(dataset, grain.MapDataset):
                dataset = dataset.to_iter_dataset(read_options=context.read_options)
            return _DocumentAwareConcatThenSplitIterDataset(
                dataset,
                max_num_documents_per_row=context.max_num_documents,
                max_context_length=context.max_context_length,
                num_tokens_per_row=context.num_tokens_per_microbatch,
                document_alignment=context.document_alignment,
            )
        if context.document_alignment is not None:
            raise ValueError(
                "document_alignment requires max_num_documents so documents are "
                "emitted as separate segments."
            )
        dataset = dataset.map(
            partial(
                _text_sequence_to_packing_input,
                max_context_length=context.max_context_length,
            )
        )
        if isinstance(dataset, grain.MapDataset):
            dataset = dataset.to_iter_dataset(read_options=context.read_options)
        dataset = grain.experimental.ConcatThenSplitIterDataset(
            dataset,
            length_struct={
                "input_ids": context.num_tokens_per_microbatch,
                "labels": context.num_tokens_per_microbatch,
                "positions": context.num_tokens_per_microbatch,
                "padding_mask": context.num_tokens_per_microbatch,
            },
        )
        dataset = dataset.filter(_packing_output_is_full)
        return dataset.map(
            partial(
                _packing_output_to_text_sequence,
                max_context_length=context.max_context_length,
            )
        )


class _DocumentAwareConcatThenSplitIterDataset(grain.IterDataset):
    """Concat-then-split packing with a document-segment capacity."""

    def __init__(
        self,
        parent: grain.IterDataset,
        *,
        max_num_documents_per_row: int,
        max_context_length: int,
        num_tokens_per_row: int,
        document_alignment: int | None = None,
    ) -> None:
        super().__init__(parent)
        self._max_num_documents_per_row = max_num_documents_per_row
        self._max_context_length = max_context_length
        self._num_tokens_per_row = num_tokens_per_row
        self._document_alignment = document_alignment

    def __iter__(self) -> grain.DatasetIterator:
        return _DocumentAwareConcatThenSplitIterator(
            iter(self._parent),
            max_num_documents_per_row=self._max_num_documents_per_row,
            max_context_length=self._max_context_length,
            num_tokens_per_row=self._num_tokens_per_row,
            document_alignment=self._document_alignment,
        )


class _DocumentAwareConcatThenSplitIterator(grain.DatasetIterator):
    """Build fixed-size rows while preserving document order and remainders."""

    def __init__(
        self,
        parent: grain.DatasetIterator,
        *,
        max_num_documents_per_row: int,
        max_context_length: int,
        num_tokens_per_row: int,
        document_alignment: int | None = None,
    ) -> None:
        super().__init__(parent)
        self._max_num_documents_per_row = max_num_documents_per_row
        self._max_context_length = max_context_length
        self._num_tokens_per_row = num_tokens_per_row
        self._document_alignment = document_alignment
        self._remainder: TextSequence | None = None
        self._remainder_parent_state: dict[str, Any] | None = None
        self._remainder_offset = 0
        self._finished = False

    def __next__(self) -> TextSequence:
        self._assert_not_closed()
        if self._finished:
            raise StopIteration

        input_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        position_parts: list[np.ndarray] = []
        mask_parts: list[np.ndarray] = []
        num_tokens = 0

        while (
            num_tokens < self._num_tokens_per_row
            and len(position_parts) < self._max_num_documents_per_row
        ):
            if self._remainder is None:
                parent_state = self._parent.get_state()
                try:
                    sequence = next(self._parent)
                except StopIteration:
                    self._finished = True
                    break
                if len(sequence.input_ids) == 0:
                    continue
                self._remainder = sequence
                self._remainder_parent_state = parent_state
                self._remainder_offset = 0

            sequence = self._remainder
            assert sequence is not None
            source_positions = (
                None if sequence.positions is None else np.asarray(sequence.positions)
            )
            segment_end = _next_document_chunk_end(
                num_tokens=len(sequence.input_ids),
                positions=source_positions,
                start=self._remainder_offset,
                max_context_length=self._max_context_length,
            )
            available_tokens = self._num_tokens_per_row - num_tokens
            real_tokens = min(
                segment_end - self._remainder_offset,
                available_tokens,
            )
            pad_tokens = 0
            if self._document_alignment is not None:
                aligned = round_up(real_tokens, self._document_alignment)
                if aligned > available_tokens:
                    # Reserve the padding inside this row instead of spilling.
                    aligned = (
                        available_tokens // self._document_alignment
                    ) * self._document_alignment
                    if aligned == 0:
                        break
                    real_tokens = aligned
                pad_tokens = aligned - real_tokens
                if real_tokens == 0 and pad_tokens == 0:
                    break

            token_slice = slice(
                self._remainder_offset,
                self._remainder_offset + real_tokens,
            )
            segment_tokens = real_tokens + pad_tokens
            input_parts.append(
                np.concatenate(
                    [
                        np.asarray(sequence.input_ids[token_slice]),
                        np.zeros(pad_tokens, dtype=np.int64),
                    ]
                )
            )
            label_parts.append(
                np.concatenate(
                    [
                        np.asarray(sequence.labels[token_slice]),
                        np.full(pad_tokens, IGNORE_INDEX, dtype=np.int64),
                    ]
                )
            )
            # Positions stay continuous across the padding, so the padded tail
            # belongs to the same document segment (and its compression groups).
            position_parts.append(np.arange(segment_tokens, dtype=np.int64))
            # Keep padding from an upstream packer marked as padding.
            source_mask = (
                np.zeros(real_tokens, dtype=np.bool_)
                if sequence.padding_mask is None
                else np.asarray(sequence.padding_mask[token_slice], dtype=np.bool_)
            )
            mask_parts.append(
                np.concatenate(
                    [source_mask, np.ones(pad_tokens, dtype=np.bool_)]
                )
            )
            num_tokens += segment_tokens
            self._remainder_offset += real_tokens
            if self._remainder_offset == len(sequence.input_ids):
                self._remainder = None
                self._remainder_parent_state = None
                self._remainder_offset = 0

        if not input_parts:
            raise StopIteration

        input_ids = np.concatenate(input_parts)
        labels = np.concatenate(label_parts)
        positions = np.concatenate(position_parts)
        padding_mask = np.concatenate(mask_parts)
        pad_len = self._num_tokens_per_row - num_tokens
        if pad_len:
            padding_positions = (
                np.arange(pad_len, dtype=positions.dtype) % self._max_context_length
            )
            input_ids = np.pad(input_ids, (0, pad_len))
            labels = np.pad(labels, (0, pad_len), constant_values=IGNORE_INDEX)
            positions = np.concatenate((positions, padding_positions))
            padding_mask = np.pad(padding_mask, (0, pad_len), constant_values=True)

        return TextSequence(
            input_ids=input_ids,
            labels=labels,
            positions=positions,
            padding_mask=padding_mask,
        )

    def get_state(self) -> dict[str, Any]:
        if self._remainder is None:
            parent_state = self._parent.get_state()
        else:
            assert self._remainder_parent_state is not None
            parent_state = self._remainder_parent_state
        return {
            "parent": parent_state,
            "has_remainder": self._remainder is not None,
            "remainder_offset": self._remainder_offset,
            "finished": self._finished,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._parent.set_state(state["parent"])
        self._remainder = None
        self._remainder_parent_state = None
        self._remainder_offset = 0
        self._finished = state["finished"]
        if state["has_remainder"]:
            self._remainder_parent_state = state["parent"]
            self._remainder = next(self._parent)
            self._remainder_offset = state["remainder_offset"]


@dataclass(frozen=True, kw_only=True, slots=True)
class FirstFitPackingConfig:
    """Packs document chunks no longer than the context window."""

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
        if isinstance(dataset, grain.MapDataset):
            dataset = dataset.to_iter_dataset(read_options=context.read_options)
        dataset = grain.experimental.FlatMapIterDataset(
            dataset,
            _SplitTextSequenceDocuments(
                max_context_length=context.max_context_length,
            ),
        )
        if context.document_alignment is not None:
            raise ValueError(
                "document_alignment requires max_num_documents so documents are "
                "emitted as separate segments."
            )
        dataset = dataset.map(
            partial(
                _text_sequence_to_packing_input,
                max_context_length=context.max_context_length,
            )
        )
        # TODO(data-global-pack-plan): Consider packing before DP sharding so
        # ranks receive similarly filled rows.
        dataset = grain.experimental.FirstFitPackIterDataset(
            dataset,
            length_struct={
                "input_ids": context.num_tokens_per_microbatch,
                "labels": context.num_tokens_per_microbatch,
                "positions": context.num_tokens_per_microbatch,
                "padding_mask": context.num_tokens_per_microbatch,
            },
            padding_struct={
                "input_ids": 0,
                "labels": IGNORE_INDEX,
                "positions": 0,
                "padding_mask": True,
            },
            num_packing_bins=self.num_packing_bins,
            meta_features=("labels", "positions"),
            seed=dataset_iteration_policy.seed,
            shuffle_bins=dataset_iteration_policy.shuffle,
            max_sequences_per_bin=(
                context.max_num_documents
                if context.max_num_documents is not None
                else None
            ),
        )
        return dataset.map(
            partial(
                _packing_output_to_text_sequence,
                max_context_length=context.max_context_length,
            )
        )


class _SplitTextSequenceDocuments(grain.experimental.FlatMapTransform):
    """Expose context-sized document chunks to Grain's native packing limit."""

    def __init__(
        self,
        *,
        max_context_length: int,
    ) -> None:
        self._max_context_length = max_context_length
        self.max_fan_out = max_context_length

    def flat_map(self, element: TextSequence) -> list[TextSequence]:
        if len(element.input_ids) == 0:
            return []

        positions = None if element.positions is None else np.asarray(element.positions)
        chunks = []
        chunk_start = 0
        while chunk_start < len(element.input_ids):
            chunk_end = _next_document_chunk_end(
                num_tokens=len(element.input_ids),
                positions=positions,
                start=chunk_start,
                max_context_length=self._max_context_length,
            )
            chunks.append(
                TextSequence(
                    input_ids=np.asarray(element.input_ids[chunk_start:chunk_end]),
                    labels=np.asarray(element.labels[chunk_start:chunk_end]),
                    positions=np.arange(chunk_end - chunk_start, dtype=np.int64),
                    padding_mask=(
                        None
                        if element.padding_mask is None
                        else np.asarray(element.padding_mask[chunk_start:chunk_end])
                    ),
                )
            )
            chunk_start = chunk_end
        return chunks


def _next_document_chunk_end(
    *,
    num_tokens: int,
    positions: np.ndarray | None,
    start: int,
    max_context_length: int,
) -> int:
    """Return the next document boundary or context-sized chunk boundary."""
    end = min(start + max_context_length, num_tokens)
    if positions is not None:
        next_starts = np.flatnonzero(positions[start + 1 : end] == 0)
        if next_starts.size:
            end = start + 1 + int(next_starts[0])
    return end


def _packing_output_is_full(packing_output: dict[str, np.ndarray]) -> bool:
    """Return whether concat-then-split filled the entire token microbatch."""
    return bool(np.all(np.asarray(packing_output["input_ids_segment_ids"]) != 0))


def _text_sequence_to_packing_input(
    text_sequence: TextSequence,
    *,
    max_context_length: int,
) -> dict[str, np.ndarray]:
    """Convert a `TextSequence` to the array dictionary expected by text packing.

    Positions restart at each document start (a zero position) and every
    `max_context_length` tokens after it, matching how the other packers split
    long documents. Missing positions mean the sample is one document.
    """
    token_indices = np.arange(len(text_sequence.input_ids), dtype=np.int64)
    document_starts = 0
    if text_sequence.positions is not None:
        document_starts = np.maximum.accumulate(
            np.where(np.asarray(text_sequence.positions) == 0, token_indices, 0)
        )
    positions = (token_indices - document_starts) % max_context_length
    padding_mask = text_sequence.padding_mask
    if padding_mask is None:
        padding_mask = np.zeros(len(text_sequence.input_ids), dtype=np.bool_)
    return {
        "input_ids": np.asarray(text_sequence.input_ids),
        "labels": np.asarray(text_sequence.labels),
        "positions": np.asarray(positions),
        "padding_mask": np.asarray(padding_mask),
    }


def _packing_output_to_text_sequence(
    packing_output: dict[str, np.ndarray],
    *,
    max_context_length: int,
) -> TextSequence:
    """Finalize packed text by masking padding and canonicalizing positions."""
    segment_ids = np.asarray(packing_output["input_ids_segment_ids"])
    padding_mask = np.asarray(packing_output["padding_mask"], dtype=np.bool_).copy()
    padding_mask[segment_ids == 0] = True
    labels = np.asarray(packing_output["labels"]).copy()
    labels[padding_mask] = IGNORE_INDEX

    # A zero starts a document. For [0, 1, 2, 0, 1], segment_starts is
    # [0, 0, 0, 3, 3], so subtracting it restores [0, 1, 2, 0, 1].
    boundaries = np.asarray(packing_output["positions"]) == 0
    token_indices = np.arange(len(boundaries), dtype=np.int64)
    segment_starts = np.maximum.accumulate(np.where(boundaries, token_indices, 0))
    positions = token_indices - segment_starts

    packing_padding = segment_ids == 0
    if np.any(packing_padding):
        first_padding_token = int(np.flatnonzero(packing_padding)[0])
        positions[first_padding_token:] = (
            np.arange(len(positions) - first_padding_token) % max_context_length
        )

    return TextSequence(
        input_ids=np.asarray(packing_output["input_ids"]),
        labels=labels,
        positions=positions,
        padding_mask=padding_mask,
    )
