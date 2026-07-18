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
from torchtitan.hf_datasets.multimodal.mm_datasets import QwenMultimodalPackingConfig


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


def _row(value: int, image: torch.Tensor | None = None) -> dict:
    return {
        "input_ids": torch.tensor([value, value + 1]),
        "labels": torch.tensor([value, value + 1]),
        "positions": torch.tensor([0, 1]),
        "pixel_values": [] if image is None else [image],
        "pixel_values_videos": [],
    }


def _dataset(rows, *, max_images_per_batch=8):
    return QwenMultimodalPackingConfig(
        dataset=SingleDatasetConfig(source=_RowsSourceConfig(rows=tuple(rows))),
        max_images_per_batch=max_images_per_batch,
        max_patches_per_batch=100,
        patch_size=16,
        temporal_patch_size=2,
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


def test_admission_lookahead_is_checkpointed_exactly():
    first_image = torch.zeros(1, 16, 16, 3)
    second_image = torch.ones(1, 16, 16, 3)
    dataset = _dataset(
        [
            _row(1, first_image),
            _row(3, second_image),
            _row(5),
        ],
        max_images_per_batch=1,
    )
    iterator = iter(dataset)
    first = next(iterator)
    state = iterator.get_state()
    expected = next(iterator)

    restored = iter(dataset)
    restored.set_state(state)
    actual = next(restored)

    assert len(first["pixel_values"]) == 1
    assert torch.equal(first["pixel_values"][0], first_image)
    assert len(actual["pixel_values"]) == 1
    assert torch.equal(actual["pixel_values"][0], second_image)
    assert torch.equal(expected["input_ids"], actual["input_ids"])
