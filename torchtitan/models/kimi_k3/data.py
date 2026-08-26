# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from torchtitan.components.data.collators import TrainerBatch
from torchtitan.components.data.types import DatasetBuildContext
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator


class KimiK3MultiModalCollator(MultiModalCollator):
    """Batch equal-length K3 rows without joining their recurrent states."""

    @dataclass(kw_only=True, slots=True)
    class Config(MultiModalCollator.Config):
        pass

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        super().__init__(config, context=context)
        self._num_rows_per_batch, remainder = divmod(
            context.num_tokens_per_batch,
            context.max_context_length,
        )
        if remainder or self._num_rows_per_batch == 0:
            raise ValueError(
                "Kimi K3 token batches must be a positive multiple of "
                "max_context_length"
            )

    def num_rows_per_batch(self) -> int:
        return self._num_rows_per_batch

    def collate_text(
        self,
        batch: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_rows = []
        label_rows = []
        position_rows = []
        for sample in batch:
            input_ids = sample["input_ids"]
            labels = sample["labels"]
            positions = sample["positions"]
            pad_len = self._max_context_length - input_ids.shape[0]
            if pad_len < 0:
                raise ValueError("Kimi K3 rows exceed max_context_length")
            if pad_len:
                input_ids = torch.nn.functional.pad(
                    input_ids,
                    (0, pad_len),
                    value=self.tokenizer.pad_id,
                )
                labels = torch.nn.functional.pad(
                    labels,
                    (0, pad_len),
                    value=IGNORE_INDEX,
                )
                positions = torch.cat(
                    [positions, torch.arange(pad_len, dtype=positions.dtype)]
                )
            input_rows.append(input_ids)
            label_rows.append(labels)
            position_rows.append(positions)

        return (
            torch.cat(input_rows),
            torch.cat(label_rows),
            torch.cat(position_rows),
        )

    def __call__(self, batch: Sequence[dict[str, Any]]) -> TrainerBatch:
        input_dict, labels = super().__call__(batch)
        input_dict["sequence_offsets"] = torch.arange(
            0,
            self._num_tokens_per_batch + 1,
            self._max_context_length,
            dtype=torch.int64,
        )
        return input_dict, labels
