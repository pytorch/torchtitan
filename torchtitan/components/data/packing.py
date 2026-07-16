# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Document packing for the grain dataloader.

Two grain-native algorithms behind one protocol; a custom `PackingConfig` is the seam
for anything else (e.g. offline OBFD).

    ConcatThenSplitPackingConfig  concat docs, split at seq_len; zero padding; docs fragment
    FirstFitPackingConfig         whole docs into bins; padding; docs never fragment
"""

from dataclasses import dataclass, field
from typing import Protocol

import grain.python as grain
import numpy as np
import torch
from grain import experimental as grain_experimental

from torchtitan.components.data.dataset import (
    BuildOptions,
    DataRuntime,
    DatasetConfig,
    fingerprint_parts,
    TokenSample,
)
from torchtitan.components.loss import IGNORE_INDEX


class PackingConfig(Protocol):
    """Builds the packing stage; `fingerprint` names the algorithm AND its configured values."""

    def build(
        self,
        parent: grain.IterDataset,
        *,
        runtime: DataRuntime,
        options: BuildOptions,
    ) -> grain.IterDataset:
        ...

    def fingerprint(self) -> str:
        ...


def token_sample_to_shifted_features(
    sample: TokenSample,
) -> dict[str, np.ndarray] | None:
    """Per-document next-token shift, done BEFORE packing so labels never cross documents.

    Example:

        TokenSample(token_ids=[5, 6, 7, 8], loss_mask=[True, True, True, True])
        # -> {"input_ids": [5, 6, 7], "labels": [6, 7, 8]}
        # a False at loss_mask[i+1] turns labels[i] into IGNORE_INDEX
    """
    token_ids = np.asarray(sample.token_ids, dtype=np.int64)
    loss_mask = np.asarray(sample.loss_mask, dtype=np.bool_)
    if token_ids.ndim != 1 or loss_mask.shape != token_ids.shape:
        raise ValueError("token_ids and loss_mask must be aligned 1-D arrays")
    if len(token_ids) < 2:
        return None
    return {
        "input_ids": token_ids[:-1],
        "labels": np.where(loss_mask[1:], token_ids[1:], np.int64(IGNORE_INDEX)),
    }


def packed_features_to_training_batch(
    features: dict[str, np.ndarray],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Packed features -> trainer batch; packer padding (segment 0) is loss-masked.

    Example:

        features = {"input_ids": [[5, 6, 9, 0]], "labels": [[6, 7, 10, 0]],
                    "input_ids_positions": [[0, 1, 0, 0]], "input_ids_segment_ids": [[1, 1, 2, 0]]}
        inputs, labels = packed_features_to_training_batch(features)
        # inputs = {"input": [[5, 6, 9, 0]], "positions": [[0, 1, 0, 0]]}
        # labels = [[6, 7, 10, IGNORE_INDEX]]   # segment 0 masked
    """
    padding = np.asarray(features["input_ids_segment_ids"]) == 0
    labels = np.where(padding, np.int64(IGNORE_INDEX), np.asarray(features["labels"]))
    inputs = {
        "input": torch.as_tensor(np.asarray(features["input_ids"]), dtype=torch.long),
        "positions": torch.as_tensor(
            np.asarray(features["input_ids_positions"]), dtype=torch.long
        ),
    }
    return inputs, torch.as_tensor(labels, dtype=torch.long)


@dataclass(frozen=True, kw_only=True, slots=True)
class ConcatThenSplitPackingConfig:
    """Concatenate documents and split at `seq_len`: zero padding, documents may fragment.

    Matches current torchtitan greedy-buffer behavior (docs split across chunks, positions
    restart per doc). `labels` must be an ordinary aligned feature, NOT a meta feature —
    meta features are never split, which silently drops documents from packed rows.
    """

    def build(
        self,
        parent: grain.IterDataset,
        *,
        runtime: DataRuntime,
        options: BuildOptions,
    ) -> grain.IterDataset:
        del options
        return grain_experimental.ConcatThenSplitIterDataset(
            parent,
            length_struct={"input_ids": runtime.seq_len, "labels": runtime.seq_len},
            meta_features=(),
        )

    def fingerprint(self) -> str:
        return type(self).__qualname__


@dataclass(frozen=True, kw_only=True, slots=True)
class FirstFitPackingConfig:
    """First-fit bin packing: whole documents only, remainder is padding.

    Choose this when examples must not fragment (e.g. SFT).
    """

    def build(
        self,
        parent: grain.IterDataset,
        *,
        runtime: DataRuntime,
        options: BuildOptions,
    ) -> grain.IterDataset:
        return grain_experimental.FirstFitPackIterDataset(
            parent,
            length_struct={"input_ids": runtime.seq_len, "labels": runtime.seq_len},
            num_packing_bins=runtime.local_batch_size,
            # FirstFit never splits an element, so labels follow the input's bin
            # assignment without redundant positions/segment arrays.
            meta_features=("labels",),
            seed=options.seed,
            shuffle_bins=options.shuffle,
        )

    def fingerprint(self) -> str:
        return type(self).__qualname__


@dataclass(frozen=True, kw_only=True, slots=True)
class PackedTokenDatasetConfig:
    """A `TokenSample` dataset shifted, packed to `seq_len`, batched, trainer-ready.

    Example:

        PackedTokenDatasetConfig(
            dataset=weighted_interleave([(math_ds, 2.0), (code_ds, 1.0)])
        )
        # iterates ({"input": [B, L], "positions": [B, L]}, labels [B, L]) long CPU tensors
    """

    dataset: DatasetConfig
    packing: PackingConfig = field(default_factory=ConcatThenSplitPackingConfig)

    def build(
        self, *, runtime: DataRuntime, options: BuildOptions
    ) -> grain.IterDataset:
        dataset = self.dataset.build(runtime=runtime, options=options)
        dataset = dataset.map(token_sample_to_shifted_features).filter(
            lambda features: features is not None
        )
        if isinstance(dataset, grain.MapDataset):
            dataset = dataset.to_iter_dataset(
                read_options=grain.ReadOptions(
                    num_threads=4,
                    prefetch_buffer_size=64,
                )
            )
        packed = self.packing.build(
            dataset,
            runtime=runtime,
            options=options,
        )
        return packed.batch(
            batch_size=runtime.local_batch_size, drop_remainder=True
        ).map(packed_features_to_training_batch)

    def fingerprint(self) -> str:
        return fingerprint_parts(
            type(self).__qualname__,
            self.dataset.fingerprint(),
            self.packing.fingerprint(),
        )
