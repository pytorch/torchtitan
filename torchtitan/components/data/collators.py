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


class BatchInputsWithMetadata(dict[str, Any]):
    """Model inputs plus batch metadata computed by the data pipeline."""

    num_valid_tokens: int | None

    def __init__(
        self,
        inputs: dict[str, Any],
        *,
        num_valid_tokens: int | None = None,
    ) -> None:
        super().__init__(inputs)
        self.num_valid_tokens = num_valid_tokens


def batch_with_valid_token_count(
    inputs: dict[str, Any],
    labels: torch.Tensor,
) -> TrainerBatch:
    """Pair collated inputs with labels, recording the loss-bearing label count.

    Only collators whose labels are token targets should call this; the count is
    meaningless for other label types.
    """
    return (
        BatchInputsWithMetadata(
            inputs,
            num_valid_tokens=int((labels != IGNORE_INDEX).sum()),
        ),
        labels,
    )


def get_batch_num_valid_tokens(
    input_dict: dict[str, Any],
    labels: torch.Tensor,
    *,
    ignore_index: int,
) -> int:
    """Use a pipeline-provided token count, falling back to a local label scan."""
    num_valid_tokens = getattr(input_dict, "num_valid_tokens", None)
    if num_valid_tokens is not None:
        return num_valid_tokens
    return int((labels != ignore_index).sum())


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
    """Concatenates text rows and pads only the final token-batch tail."""

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

        input_ids = torch.cat([torch.as_tensor(row.input_ids) for row in rows])
        labels = torch.cat([torch.as_tensor(row.labels) for row in rows])
        positions = torch.cat(
            [
                torch.arange(len(row.input_ids))
                if row.positions is None
                else torch.as_tensor(row.positions)
                for row in rows
            ]
        )

        pad_len = self._num_tokens_per_batch - num_tokens
        if pad_len:
            input_ids = torch.nn.functional.pad(input_ids, (0, pad_len))
            labels = torch.nn.functional.pad(labels, (0, pad_len), value=IGNORE_INDEX)
            positions = torch.cat(
                [positions, torch.zeros(pad_len, dtype=positions.dtype)]
            )

        return batch_with_valid_token_count(
            {
                "input": input_ids,
                "positions": positions,
            },
            labels,
        )
