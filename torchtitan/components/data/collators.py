# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configured conversion from dataset rows to trainer microbatches."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from torchtitan.components.data.dataset import TextSequence
from torchtitan.components.data.types import (
    DatasetBuildContext,
    TokenizedTrainingMicrobatch,
    TrainingMicrobatch,
)
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.config import Configurable


# Page-locked microbatches let the trainer issue an async host-to-device copy; a copy
# out of pageable memory is synchronous whatever ``non_blocking`` says. There has
# to be an accelerator to pin for -- allocating with ``pin_memory=True`` raises
# without one -- so CPU-only runs fall back to ordinary pageable memory.
HAS_PIN_MEMORY = torch.accelerator.is_available()


class Collator(Configurable, ABC):
    """Configured row-to-microbatch conversion."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    @abstractmethod
    def __call__(self, rows: Sequence[Any]) -> TrainingMicrobatch:
        ...

    def num_rows_per_microbatch(self) -> int:
        """Return the number of dataset rows consumed by one trainer microbatch."""
        return 1


class TextCollator(Collator):
    """Packs text rows into one page-locked, pre-padded token microbatch."""

    @dataclass(kw_only=True, slots=True)
    class Config(Collator.Config):
        pass

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        del config
        self._num_tokens_per_microbatch = context.num_tokens_per_microbatch
        self._max_context_length = context.max_context_length

    def __call__(self, rows: Sequence[TextSequence]) -> TokenizedTrainingMicrobatch:
        num_tokens = sum(len(row.input_ids) for row in rows)
        if num_tokens > self._num_tokens_per_microbatch:
            raise ValueError("text rows exceed the configured token microbatch")

        size = self._num_tokens_per_microbatch
        input_ids = torch.zeros(size, dtype=torch.int64, pin_memory=HAS_PIN_MEMORY)
        positions = torch.zeros(size, dtype=torch.int64, pin_memory=HAS_PIN_MEMORY)
        labels = torch.full(
            (size,), IGNORE_INDEX, dtype=torch.int64, pin_memory=HAS_PIN_MEMORY
        )
        padding_mask = torch.ones(size, dtype=torch.bool, pin_memory=HAS_PIN_MEMORY)

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
                (
                    torch.arange(len(row.input_ids))
                    if row.positions is None
                    else torch.as_tensor(row.positions)
                )
                for row in rows
            ],
            out=positions[:num_tokens],
        )
        torch.cat(
            [
                (
                    torch.zeros(len(row.input_ids), dtype=torch.bool)
                    if row.padding_mask is None
                    else torch.as_tensor(row.padding_mask, dtype=torch.bool)
                )
                for row in rows
            ],
            out=padding_mask[:num_tokens],
        )

        pad_len = self._num_tokens_per_microbatch - num_tokens
        if pad_len:
            torch.arange(pad_len, out=positions[num_tokens:])
            positions[num_tokens:].remainder_(self._max_context_length)

        return TokenizedTrainingMicrobatch(
            input=input_ids,
            labels=labels,
            positions=positions,
            padding_mask=padding_mask,
            num_valid_tokens=int((labels != IGNORE_INDEX).sum()),
        )
