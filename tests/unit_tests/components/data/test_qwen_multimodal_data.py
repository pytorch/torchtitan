# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import grain.python as grain
import torch

from torchtitan.components.data.dataset import (
    BuildOptions,
    DataRuntime,
    SingleDatasetConfig,
)
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.hf_datasets.multimodal.mm_datasets import MMSamplePackingConfig


class _Tokenizer:
    pass


@dataclass(frozen=True)
class _RowsSource:
    rows: tuple[dict, ...]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


RUNTIME = DataRuntime(
    tokenizer=_Tokenizer(),
    seq_len=8,
    local_batch_size=1,
    read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
)


def _row(
    value: int,
    image: torch.Tensor | None = None,
    length: int = 2,
) -> dict:
    tokens = torch.arange(value, value + length)
    return {
        "input_ids": tokens,
        "labels": tokens.clone(),
        "positions": torch.arange(length),
        "pixel_values": [] if image is None else [image],
        "pixel_values_videos": [],
    }


def _dataset(rows):
    return MMSamplePackingConfig(
        dataset=SingleDatasetConfig(source=_RowsSourceConfig(rows=tuple(rows))),
        buffer_size=2,
    ).build(
        runtime=RUNTIME,
        options=BuildOptions(
            seed=0,
            shuffle=False,
            repeat=False,
            dp_rank=0,
            dp_world_size=1,
        ),
    )


@dataclass(frozen=True)
class _RowsSourceConfig:
    rows: tuple[dict, ...]

    def build(self, **_):
        return _RowsSource(self.rows)


def test_packing_preserves_ordered_images_when_merging_rows():
    first_image = torch.zeros(1, 16, 16, 3)
    second_image = torch.ones(1, 16, 16, 3)
    row = next(
        iter(
            _dataset(
                [
                    _row(1, first_image),
                    _row(3, second_image),
                ]
            )
        )
    )

    assert torch.equal(row["input_ids"], torch.tensor([1, 2, 3, 4]))
    assert torch.equal(
        row["labels"],
        torch.tensor([1, 2, IGNORE_INDEX, 4]),
    )
    assert len(row["pixel_values"]) == 2
    assert torch.equal(row["pixel_values"][0], first_image)
    assert torch.equal(row["pixel_values"][1], second_image)


def test_buffered_packing_is_checkpointed_exactly():
    dataset = _dataset(
        [
            _row(1, length=6),
            _row(10, length=4),
            _row(20, length=2),
        ]
    )
    iterator = iter(dataset)
    first = next(iterator)
    state = iterator.get_state()
    expected = next(iterator)

    restored = iter(dataset)
    restored.set_state(state)
    actual = next(restored)

    assert torch.equal(first["input_ids"], torch.tensor([1, 2, 3, 4, 5, 6, 20, 21]))
    assert torch.equal(expected["input_ids"], actual["input_ids"])
