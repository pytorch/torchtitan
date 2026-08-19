# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import grain.python as grain
import numpy as np
import pytest
import torch

from torchtitan.components.data.dataset import SingleDatasetConfig
from torchtitan.components.data.types import DatasetBuildContext, DatasetIterationPolicy
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator
from torchtitan.hf_datasets.multimodal.mm_datasets import (
    MM_DATASETS,
    MMSamplePackingConfig,
    MultiModalProcessor,
)
from torchtitan.hf_datasets.multimodal.utils.image import resize_to_patch_budget
from torchtitan.models.kimi_k2_7 import config_registry as kimi_configs
from torchtitan.models.qwen3_5 import config_registry as qwen35_configs


class _Tokenizer:
    pad_id = 0
    TOKEN_FIELDS = ()


@dataclass(frozen=True)
class _RowsSource:
    rows: tuple[dict, ...]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


CONTEXT = DatasetBuildContext(
    tokenizer=_Tokenizer(),
    seq_len=9,
    local_batch_size=1,
    read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
)


def _row(
    value: int,
    image: torch.Tensor | None = None,
    video: torch.Tensor | None = None,
    length: int = 2,
) -> dict:
    tokens = torch.arange(value, value + length)
    return {
        "input_ids": tokens,
        "labels": tokens.clone(),
        "positions": torch.arange(length),
        "pixel_values": [] if image is None else [image],
        "pixel_values_videos": [] if video is None else [video],
    }


def _dataset(rows, *, repeat=False):
    return MMSamplePackingConfig(
        dataset=SingleDatasetConfig(source=_RowsSourceConfig(rows=tuple(rows))),
        num_packing_bins=2,
    ).build(
        context=CONTEXT,
        dataset_iteration_policy=DatasetIterationPolicy(
            seed=0,
            shuffle=False,
            repeat=repeat,
            dp_rank=0,
            dp_world_size=1,
            streaming_shuffle_buffer_size=128,
        ),
    )


@dataclass(frozen=True)
class _RowsSourceConfig:
    rows: tuple[dict, ...]

    def build(
        self,
        *,
        dataset_iteration_policy: DatasetIterationPolicy,
    ):
        del dataset_iteration_policy
        return _RowsSource(self.rows)


def test_multimodal_registry_does_not_enable_packing():
    assert isinstance(MM_DATASETS["cc12m"], SingleDatasetConfig)


def test_multimodal_processor_forwards_resize_config():
    captured = {}

    def process_sample(**kwargs):
        captured.update(kwargs)
        return None

    processor = MultiModalProcessor.Config(
        sample_processor=process_sample,
        resize_fn=resize_to_patch_budget,
        max_patches=123,
        max_patches_per_side=45,
    ).build(context=CONTEXT)

    assert processor({}, np.random.default_rng(0)) is None
    assert captured["resize_fn"] is resize_to_patch_budget
    assert captured["max_patches"] == 123
    assert captured["max_patches_per_side"] == 45


@pytest.mark.parametrize(
    "recipe_name",
    [
        "kimi_k2_5_debugmodel",
        "kimi_vl_a3b",
    ],
)
def test_kimi_multimodal_recipe_copies_unpacked_dataset(recipe_name):
    base_dataset = MM_DATASETS[
        "cc12m-test" if recipe_name == "kimi_k2_5_debugmodel" else "cc12m"
    ]
    base_processor = base_dataset.processor

    config = getattr(kimi_configs, recipe_name)()
    dataset = config.dataloader.dataset
    collator = config.dataloader.collator

    assert isinstance(dataset, SingleDatasetConfig)
    assert dataset is not base_dataset
    assert isinstance(dataset.processor, MultiModalProcessor.Config)
    assert dataset.processor is not base_processor
    assert dataset.processor.resize_fn is resize_to_patch_budget
    assert dataset.processor.max_patches == 16_384
    assert isinstance(collator, MultiModalCollator.Config)
    assert collator.patch_order == "raster"
    assert (
        MM_DATASETS["cc12m-test" if recipe_name == "kimi_k2_5_debugmodel" else "cc12m"]
        is base_dataset
    )


@pytest.mark.parametrize(
    "recipe_name",
    [
        "qwen35_debugmodel",
        "qwen35_debugmodel_moe",
        "qwen35_0_8b",
        "qwen35_2b",
        "qwen35_4b",
        "qwen35_9b",
        "qwen35_27b",
        "qwen35_35b_a3b",
        "qwen35_122b_a10b",
        "qwen35_397b_a17b",
    ],
)
def test_qwen35_recipe_geometry_matches_dataset_processor(recipe_name):
    registry_state = {
        name: (
            id(dataset),
            id(dataset.processor),
            dataset.processor.patch_size,
            dataset.processor.temporal_patch_size,
            dataset.processor.spatial_merge_size,
        )
        for name, dataset in MM_DATASETS.items()
        if isinstance(dataset.processor, MultiModalProcessor.Config)
    }

    config = getattr(qwen35_configs, recipe_name)()
    dataset = config.dataloader.dataset
    collator = config.dataloader.collator

    assert isinstance(dataset, SingleDatasetConfig)
    assert isinstance(dataset.processor, MultiModalProcessor.Config)
    assert isinstance(collator, MultiModalCollator.Config)
    assert collator.patch_size == dataset.processor.patch_size
    assert collator.temporal_patch_size == dataset.processor.temporal_patch_size
    assert collator.spatial_merge_size == dataset.processor.spatial_merge_size
    assert {
        name: (
            id(dataset),
            id(dataset.processor),
            dataset.processor.patch_size,
            dataset.processor.temporal_patch_size,
            dataset.processor.spatial_merge_size,
        )
        for name, dataset in MM_DATASETS.items()
        if isinstance(dataset.processor, MultiModalProcessor.Config)
    } == registry_state


def test_packing_preserves_ordered_media_when_merging_rows():
    first_image = torch.zeros(1, 16, 16, 3)
    second_image = torch.ones(1, 16, 32, 3)
    first_video = torch.full((2, 16, 16, 3), 2.0)
    second_video = torch.full((4, 32, 16, 3), 3.0)
    row = next(
        iter(
            _dataset(
                [
                    _row(1, first_image, first_video),
                    _row(3, second_image, second_video),
                ]
            )
        )
    )

    assert row["input_ids"].tolist() == [1, 2, 3, 4, 0, 0, 0, 0, 0]
    assert torch.equal(
        row["labels"],
        torch.tensor([1, 2, 3, 4] + [IGNORE_INDEX] * (CONTEXT.seq_len - 4)),
    )
    assert row["positions"].tolist() == [0, 1, 0, 1, 0, 0, 0, 0, 0]
    assert len(row["pixel_values"]) == 2
    assert torch.equal(row["pixel_values"][0], first_image)
    assert torch.equal(row["pixel_values"][1], second_image)
    assert len(row["pixel_values_videos"]) == 2
    assert torch.equal(row["pixel_values_videos"][0], first_video)
    assert torch.equal(row["pixel_values_videos"][1], second_video)


def test_buffered_packing_is_checkpointed_exactly():
    first_image = torch.zeros(1, 16, 16, 3)
    second_image = torch.ones(1, 16, 16, 3)
    dataset = _dataset(
        [
            _row(1, image=first_image, length=6),
            _row(10, image=second_image, length=4),
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

    assert first["input_ids"].tolist() == [1, 2, 3, 4, 5, 6, 20, 21, 0]
    for key in ("input_ids", "labels", "positions"):
        assert torch.equal(expected[key], actual[key])
    assert len(expected["pixel_values"]) == len(actual["pixel_values"]) == 1
    assert torch.equal(expected["pixel_values"][0], actual["pixel_values"][0])
    assert expected["pixel_values_videos"] == actual["pixel_values_videos"] == []


def test_repeated_underfilled_rows_emit_at_bin_pressure():
    dataset = _dataset(
        [
            _row(1, length=6),
            _row(10, length=6),
        ],
        repeat=True,
    )
    iterator = iter(dataset)

    assert torch.equal(
        next(iterator)["input_ids"],
        torch.tensor([1, 2, 3, 4, 5, 6, 0, 0, 0]),
    )
    assert torch.equal(
        next(iterator)["input_ids"],
        torch.tensor([10, 11, 12, 13, 14, 15, 0, 0, 0]),
    )


def test_oversized_row_does_not_clear_valid_buffered_rows():
    dataset = _dataset(
        [
            _row(1, length=6),
            _row(20, length=10),
            _row(10, length=3),
        ]
    )

    row = next(iter(dataset))

    assert torch.equal(
        row["input_ids"],
        torch.tensor([1, 2, 3, 4, 5, 6, 10, 11, 12]),
    )


def test_multimodal_packing_uses_seq_len():
    row = next(iter(_dataset([_row(1, length=9)])))

    assert len(row["input_ids"]) == CONTEXT.seq_len


def test_multimodal_collator_preserves_aligned_labels():
    collator = MultiModalCollator.Config().build(context=CONTEXT)
    packed = {
        "input_ids": torch.tensor([1, 2, 3, 4]),
        "labels": torch.tensor([2, 9, 4, 10]),
        "positions": torch.tensor([0, 1, 0, 1]),
        "pixel_values": [],
        "pixel_values_videos": [],
    }

    _, labels = collator([packed])

    assert labels[0, :4].tolist() == [2, 9, 4, 10]


def test_mm_finite_underfilled_tail_flushes():
    rows = list(
        _dataset(
            [
                _row(1, length=3),
                _row(10, length=2),
            ]
        )
    )

    assert len(rows) == 1
    assert rows[0]["input_ids"].tolist() == [1, 2, 3, 10, 11, 0, 0, 0, 0]
    assert rows[0]["labels"].tolist() == [
        1,
        2,
        3,
        10,
        11,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
    ]


def test_multimodal_packing_rejects_nonpositive_num_bins():
    with pytest.raises(ValueError, match="num_packing_bins must be positive"):
        MMSamplePackingConfig(
            dataset=SingleDatasetConfig(source=_RowsSourceConfig(rows=())),
            num_packing_bins=0,
        )
