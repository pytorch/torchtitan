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


# The input dict holds the model's forward kwargs plus ``num_valid_tokens``, the
# number of labels that contribute to the loss. Collators over token labels
# count them here so the trainer does not rescan every batch on the critical
# path; the trainer pops the field before the batch reaches the model.
TrainerBatch: TypeAlias = tuple[dict[str, Any], torch.Tensor]

# Page-locked batches let the trainer issue an async host-to-device copy; a copy
# out of pageable memory is synchronous whatever ``non_blocking`` says. There has
# to be an accelerator to pin for -- allocating with ``pin_memory=True`` raises
# without one -- so CPU-only runs fall back to ordinary pageable memory.
HAS_PIN_MEMORY = torch.accelerator.is_available()


class Collator(Configurable, ABC):
    """Configured row-to-batch conversion."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    @abstractmethod
    def __call__(self, rows: Sequence[Any]) -> TrainerBatch:
        ...

    def num_rows_per_batch(self) -> int:
        """Return the number of dataset rows consumed by one trainer batch."""
        return 1


class TextCollator(Collator):
    """Packs text rows into one page-locked, pre-padded token batch."""

    @dataclass(kw_only=True, slots=True)
    class Config(Collator.Config):
        pass

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        del config
        self._num_tokens_per_batch = context.num_tokens_per_batch

    def __call__(self, rows: Sequence[TextSequence]) -> TrainerBatch:
        num_tokens = sum(len(row.input_ids) for row in rows)
        if num_tokens > self._num_tokens_per_batch:
            raise ValueError("text rows exceed the configured token batch")

        size = self._num_tokens_per_batch
        input_ids = torch.zeros(size, dtype=torch.int64, pin_memory=HAS_PIN_MEMORY)
        positions = torch.zeros(size, dtype=torch.int64, pin_memory=HAS_PIN_MEMORY)
        labels = torch.full(
            (size,), IGNORE_INDEX, dtype=torch.int64, pin_memory=HAS_PIN_MEMORY
        )

        torch.cat(
            [torch.as_tensor(row.input_ids) for row in rows],
            out=input_ids[:num_tokens],
        )
        torch.cat(
            [torch.as_tensor(row.labels) for row in rows],
            out=labels[:num_tokens],
        )
        torch.cat(
            [
                torch.arange(len(row.input_ids))
                if row.positions is None
                else torch.as_tensor(row.positions)
                for row in rows
            ],
            out=positions[:num_tokens],
        )

        return {
            "input": input_ids,
            "positions": positions,
            "num_valid_tokens": int((labels != IGNORE_INDEX).sum()),
        }, labels
