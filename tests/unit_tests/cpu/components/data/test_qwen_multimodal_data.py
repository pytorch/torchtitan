# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, replace

import grain.python as grain
import numpy as np
import pytest
import torch

from torchtitan.components.data import GrainDataLoader
from torchtitan.components.data.dataset import SingleDatasetConfig
from torchtitan.components.data.multimodal.image_processor import ImageProcessor
from torchtitan.components.data.multimodal.mm_collator import MultiModalCollator
from torchtitan.components.data.multimodal.mm_datasets import (
    MMSamplePackingConfig,
    VisionProcessor,
)
from torchtitan.components.data.multimodal.utils.resize import (
    resize_to_navit_patch_grid,
    resize_to_pixel_budget,
)
from torchtitan.components.data.multimodal.video_processor import VideoProcessor
from torchtitan.components.data.types import DatasetBuildContext, DatasetIterationPolicy
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan_recipes.tests.datasets.synthetic_multimodal import SYNTHETIC_MM_DATASETS
from torchtitan_recipes.tests.models import (
    kimi_k2_7 as kimi_configs,
    kimi_k3 as kimi_k3_configs,
    muse_glimmer as muse_glimmer_configs,
    muse_spark as muse_spark_configs,
    qwen3_5 as qwen35_configs,
    qwen3_6 as qwen36_configs,
    qwen3_8 as qwen38_configs,
)
from torchtitan_recipes.tests.models.multimodal import DPRankImagePresenceDatasetConfig

from tests.integration_tests import b200 as b200_recipes, models as integration_recipes


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
    max_context_length=9,
    num_tokens_per_microbatch=9,
    read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
)

_PROCESSOR_FIELDS = dict(
    patch_size=16,
    temporal_patch_size=2,
    spatial_merge_size=2,
    image_mean=(0.5, 0.5, 0.5),
    image_std=(0.5, 0.5, 0.5),
    resize_fn=resize_to_pixel_budget,
    patch_order="block",
)
VISION_CONTEXT = replace(
    CONTEXT,
    image_processor=ImageProcessor.Config(**_PROCESSOR_FIELDS).build(),
    video_processor=VideoProcessor.Config(**_PROCESSOR_FIELDS).build(),
)


def _row(
    value: int,
    image: torch.Tensor | None = None,
    video: torch.Tensor | None = None,
    length: int = 2,
    waveforms: list[torch.Tensor] | None = None,
) -> dict:
    tokens = torch.arange(value, value + length)
    return {
        "input_ids": tokens,
        "labels": tokens.clone(),
        "positions": torch.arange(length),
        "pixel_values": [] if image is None else [image],
        "pixel_values_videos": [] if video is None else [video],
        "waveforms": [] if waveforms is None else waveforms,
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

    def build(self):
        return _RowsSource(self.rows)


def test_vision_processor_passes_trainer_processors():
    captured = {}

    def process_sample(**kwargs):
        captured.update(kwargs)
        return None

    processor = VisionProcessor.Config(sample_processor=process_sample).build(
        context=VISION_CONTEXT
    )

    assert processor({}, np.random.default_rng(0)) is None
    assert captured["image_processor"] is VISION_CONTEXT.image_processor
    assert captured["video_processor"] is VISION_CONTEXT.video_processor


def test_vision_processor_requires_a_processor():
    with pytest.raises(ValueError, match="image_processor"):
        VisionProcessor.Config(sample_processor=lambda **kwargs: None).build(
            context=CONTEXT
        )


def test_dp_rank_image_presence_dataset_drops_images_on_even_ranks():
    def process_sample(sample, **kwargs):
        del kwargs
        return {
            "input_ids": torch.tensor([1]),
            "has_image": sample.get("jpg") is not None,
        }

    base = SingleDatasetConfig(
        source=_RowsSourceConfig(rows=({"jpg": b"image"}, {"jpg": b"image"})),
        processor=VisionProcessor.Config(sample_processor=process_sample),
    )
    has_image = []
    for dp_rank in range(2):
        dataset = DPRankImagePresenceDatasetConfig(dataset=base).build(
            context=VISION_CONTEXT,
            dataset_iteration_policy=DatasetIterationPolicy(
                seed=0,
                shuffle=False,
                repeat=False,
                dp_rank=dp_rank,
                dp_world_size=2,
                streaming_shuffle_buffer_size=1,
            ),
        )
        has_image.append(dataset[0]["has_image"])

    assert has_image == [False, True]


@pytest.mark.parametrize(
    "recipe",
    [
        integration_recipes.qwen35_debugmodel_moe_fsdp2_tp2_pp2_ep4,
        integration_recipes.qwen35_debugmodel_moe_fsdp4_tp2_ep4,
        integration_recipes.qwen35_debugmodel_varlen_attn_fsdp2_tp2_sac,
        integration_recipes.kimi_k2_5_debugmodel_muon_fsdp2_pp2_ep2,
        integration_recipes.kimi_k2_5_debugmodel_muon_fsdp8_ep8,
        integration_recipes.muse_glimmer_debugmodel_mm_fsdp2_tp2,
        b200_recipes.kimi_k3_debugmodel_mm,
    ],
)
def test_multimodal_dp_integration_recipes_use_rank_conditional_images(recipe):
    config = recipe()

    assert isinstance(config.dataloader, GrainDataLoader.Config)
    assert isinstance(config.dataloader.dataset, DPRankImagePresenceDatasetConfig)


@pytest.mark.parametrize(
    ("recipe_module", "recipe_name"),
    [
        (kimi_configs, "kimi_k2_5_debugmodel"),
        (kimi_configs, "kimi_vl_a3b"),
        (kimi_k3_configs, "kimi_k3_debugmodel"),
    ],
)
def test_kimi_recipe_image_processor_matches_vision_encoder(recipe_module, recipe_name):
    config = getattr(recipe_module, recipe_name)()
    image_processor = config.image_processor
    vision_encoder = config.model.vision_encoder

    assert isinstance(image_processor, ImageProcessor.Config)
    assert image_processor.patch_size == vision_encoder.patch_size
    # MoonViT3d patches one frame at a time and merges a square kernel.
    assert image_processor.temporal_patch_size == 1
    assert (image_processor.spatial_merge_size,) * 2 == tuple(
        vision_encoder.merge_kernel_size
    )
    assert image_processor.resize_fn is resize_to_navit_patch_grid
    assert image_processor.patch_order == "raster"


@pytest.mark.parametrize(
    ("recipe_module", "recipe_name"),
    [
        (muse_glimmer_configs, "muse_glimmer_debugmodel_mm"),
        (muse_glimmer_configs, "muse_glimmer_30b_mm"),
        (muse_spark_configs, "muse_spark_debugmodel_mm"),
    ],
)
def test_muse_recipe_processors_match_vision_encoder(recipe_module, recipe_name):
    config = getattr(recipe_module, recipe_name)()
    vision_encoder = config.model.vision_encoder

    processors = [config.image_processor, config.video_processor]
    for processor in (p for p in processors if p is not None):
        assert processor.patch_size == vision_encoder.patch_size
        assert processor.temporal_patch_size == vision_encoder.patch_temporal
        assert processor.spatial_merge_size == vision_encoder.downsample_factor
        assert processor.patch_order == "raster"


def _assert_matches_vision_encoder(processor_config, vision_encoder):
    assert processor_config.patch_size == vision_encoder.patch_size
    assert processor_config.temporal_patch_size == vision_encoder.temporal_patch_size
    assert processor_config.spatial_merge_size == vision_encoder.spatial_merge_size


@pytest.mark.parametrize(
    ("recipe_module", "recipe_name"),
    [
        (qwen35_configs, "qwen35_debugmodel"),
        (qwen35_configs, "qwen35_debugmodel_moe"),
        (qwen36_configs, "qwen36_debugmodel"),
        (qwen36_configs, "qwen36_debugmodel_moe"),
        (qwen36_configs, "qwen36_27b"),
        (qwen36_configs, "qwen36_35b_a3b"),
        (qwen38_configs, "qwen38_debugmodel"),
        (qwen38_configs, "qwen38_debugmodel_moe"),
        (qwen38_configs, "qwen38_27b"),
    ],
)
def test_qwen_recipe_image_processor_matches_vision_encoder(recipe_module, recipe_name):
    config = getattr(recipe_module, recipe_name)()

    assert isinstance(config.dataloader.dataset, SingleDatasetConfig)
    assert isinstance(config.dataloader.dataset.processor, VisionProcessor.Config)
    assert isinstance(config.image_processor, ImageProcessor.Config)
    assert config.image_processor.patch_order == "block"
    _assert_matches_vision_encoder(config.image_processor, config.model.vision_encoder)


def test_qwen35_video_recipe_uses_synthetic_video_dataset():
    config = qwen35_configs.qwen35_debugmodel_video()
    assert isinstance(config.dataloader, GrainDataLoader.Config)
    dataset = config.dataloader.dataset

    assert dataset is SYNTHETIC_MM_DATASETS["synthetic-video-text"]
    assert isinstance(dataset.processor, VisionProcessor.Config)
    assert isinstance(config.video_processor, VideoProcessor.Config)
    _assert_matches_vision_encoder(config.video_processor, config.model.vision_encoder)
    assert config.training.steps == 10
    assert config.training.disable_accelerator_graphs


def test_packing_preserves_ordered_media_when_merging_rows():
    first_image = torch.zeros(1, 16, 16, 3)
    second_image = torch.ones(1, 16, 32, 3)
    first_video = torch.full((2, 16, 16, 3), 2.0)
    second_video = torch.full((4, 32, 16, 3), 3.0)
    first_waveform = torch.full((3, 1), 4.0)
    second_waveform = torch.full((5, 1), 5.0)
    row = next(
        iter(
            _dataset(
                [
                    _row(
                        1,
                        first_image,
                        first_video,
                        waveforms=[first_waveform],
                    ),
                    _row(
                        3,
                        second_image,
                        second_video,
                        waveforms=[second_waveform],
                    ),
                ]
            )
        )
    )

    assert row["input_ids"].tolist() == [1, 2, 3, 4, 0, 0, 0, 0, 0]
    assert torch.equal(
        row["labels"],
        torch.tensor([1, 2, 3, 4] + [IGNORE_INDEX] * (CONTEXT.max_context_length - 4)),
    )
    # Padding is numbered like the collator's tail padding, so it forms one
    # segment instead of one document start per padded token.
    assert row["positions"].tolist() == [0, 1, 0, 1, 0, 1, 2, 3, 4]
    assert row["padding_mask"].tolist() == [False] * 4 + [True] * 5
    assert len(row["pixel_values"]) == 2
    assert torch.equal(row["pixel_values"][0], first_image)
    assert torch.equal(row["pixel_values"][1], second_image)
    assert len(row["pixel_values_videos"]) == 2
    assert torch.equal(row["pixel_values_videos"][0], first_video)
    assert torch.equal(row["pixel_values_videos"][1], second_video)
    assert len(row["waveforms"]) == 2
    assert torch.equal(row["waveforms"][0], first_waveform)
    assert torch.equal(row["waveforms"][1], second_waveform)


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
    assert expected["waveforms"] == actual["waveforms"] == []


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


def test_multimodal_packing_uses_max_context_length():
    row = next(iter(_dataset([_row(1, length=9)])))

    assert len(row["input_ids"]) == CONTEXT.max_context_length


def test_multimodal_collator_preserves_aligned_labels():
    collator = MultiModalCollator.Config().build(context=CONTEXT)
    packed = {
        "input_ids": torch.tensor([1, 2, 3, 4]),
        "labels": torch.tensor([2, 9, 4, 10]),
        "positions": torch.tensor([0, 1, 0, 1]),
        "pixel_values": [],
        "pixel_values_videos": [],
    }

    microbatch = collator([packed])
    labels = microbatch.labels

    assert labels[:4].tolist() == [2, 9, 4, 10]
    torch.testing.assert_close(microbatch.loss_token_counts, torch.tensor(4))
    torch.testing.assert_close(microbatch.loss_token_counts, torch.tensor(4))
    torch.testing.assert_close(microbatch.routing_token_counts, torch.tensor([4]))


def test_mrope_collator_rejects_raster_patch_order():
    context = replace(
        CONTEXT,
        image_processor=ImageProcessor.Config(
            **{**_PROCESSOR_FIELDS, "patch_order": "raster"}
        ).build(),
    )

    with pytest.raises(ValueError, match="MRoPE requires patch_order='block'"):
        MultiModalCollator.Config(build_mrope_positions=True).build(context=context)


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
