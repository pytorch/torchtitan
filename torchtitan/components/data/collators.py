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

import torch

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


class TextCollator(Collator):
    """Pads next-token-aligned text rows into trainer batches."""

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
                torch.arange(length)
                if row.positions is None
                else torch.as_tensor(row.positions)
            )
            positions[row_index, :length] = row_positions

        return {
            "input": input_ids,
            "positions": positions,
        }, labels
