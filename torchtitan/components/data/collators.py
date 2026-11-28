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

import numpy as np
import torch

from torchtitan.components.data.dataset import TextSequence
from torchtitan.components.data.padding import (
    get_microbatch_token_counts,
    pack_microbatch_fields,
)
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
        self._num_mtp_layers = context.num_mtp_layers

    def __call__(self, rows: Sequence[TextSequence]) -> TokenizedTrainingMicrobatch:
        packed = pack_microbatch_fields(
            {
                "input_ids": [row.input_ids for row in rows],
                "labels": [row.labels for row in rows],
                "positions": [
                    (
                        np.arange(len(row.input_ids))
                        if row.positions is None
                        else row.positions
                    )
                    for row in rows
                ],
                "padding_mask": [
                    (
                        np.zeros(len(row.input_ids), dtype=np.bool_)
                        if row.padding_mask is None
                        else row.padding_mask
                    )
                    for row in rows
                ],
            },
            num_tokens=self._num_tokens_per_microbatch,
            max_context_length=self._max_context_length,
            fill_values={"input_ids": 0, "labels": IGNORE_INDEX},
            dtypes={"input_ids": torch.int64, "labels": torch.int64},
            pin_memory=HAS_PIN_MEMORY,
        )
        loss_token_counts, routing_token_counts = get_microbatch_token_counts(
            target_mask=packed["labels"] != IGNORE_INDEX,
            positions=packed["positions"],
            padding_mask=packed["padding_mask"],
            num_mtp_layers=self._num_mtp_layers,
        )
        return TokenizedTrainingMicrobatch(
            input=packed["input_ids"],
            labels=packed["labels"],
            positions=packed["positions"],
            padding_mask=packed["padding_mask"],
            loss_token_counts=loss_token_counts,
            routing_token_counts=routing_token_counts,
        )
