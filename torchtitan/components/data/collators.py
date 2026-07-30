# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configured conversion from dataset rows to trainer batches."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import torch
from torch.utils.data import default_collate

from torchtitan.components.data.dataset import TextSequence
from torchtitan.components.data.types import DatasetBuildContext
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.config import Configurable


TrainerBatch: TypeAlias = tuple[dict[str, Any], torch.Tensor]


class Collator(Configurable, ABC):
    """Configured row-to-batch conversion."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    @abstractmethod
    def __call__(self, rows: Sequence[Any]) -> TrainerBatch:
        ...


class DefaultCollator(Collator):
    """Stacks rows with PyTorch default collation."""

    @dataclass(kw_only=True, slots=True)
    class Config(Collator.Config):
        pass

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        del config, context

    def __call__(self, rows: Sequence[Any]) -> TrainerBatch:
        batch = default_collate(list(rows))
        if not isinstance(batch, (list, tuple)) or len(batch) != 2:
            raise TypeError(
                "DefaultCollator rows must contain (model_inputs, labels) pairs"
            )
        model_inputs, labels = batch
        return model_inputs, labels


class TextCollator(Collator):
    """Pads token-aligned text and creates next-token trainer batches."""

    @dataclass(kw_only=True, slots=True)
    class Config(Collator.Config):
        pass

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        del config
        self._seq_len = context.seq_len

    def __call__(self, rows: Sequence[TextSequence]) -> TrainerBatch:
        if any(len(row.input_ids) > self._seq_len for row in rows):
            raise ValueError("unpacked text exceeds seq_len")

        input_ids = torch.zeros(
            len(rows),
            self._seq_len,
            dtype=torch.long,
        )
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        positions = torch.zeros_like(input_ids)

        for row_index, row in enumerate(rows):
            length = len(row.input_ids)
            input_ids[row_index, :length] = torch.as_tensor(row.input_ids)
            labels[row_index, :length] = torch.as_tensor(row.labels)
            row_positions = (
                np.arange(length, dtype=np.int64)
                if row.positions is None
                else row.positions
            )
            positions[row_index, :length] = torch.as_tensor(row_positions)

        input_ids, labels, positions = shift_causal_labels(
            input_ids,
            labels,
            positions,
        )
        return {
            "input": input_ids,
            "positions": positions,
        }, labels


def shift_causal_labels(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shift causal labels once and mask targets across position resets."""
    shifted_labels = torch.nn.functional.pad(
        labels[..., 1:],
        (0, 1),
        value=IGNORE_INDEX,
    )
    shifted_labels[..., :-1].masked_fill_(positions[..., 1:] == 0, IGNORE_INDEX)
    return input_ids, shifted_labels, positions
