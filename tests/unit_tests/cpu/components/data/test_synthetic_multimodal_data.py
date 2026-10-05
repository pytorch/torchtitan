# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from dataclasses import dataclass, replace
from pathlib import Path

import grain.python as grain
import pytest
import torch
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from torchtitan.components.data import GrainDataLoader
from torchtitan.components.data.dataset import SingleDatasetConfig
from torchtitan.components.data.sources import IndexedJsonlSource
from torchtitan.components.data.types import DatasetBuildContext
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.hf_datasets.multimodal import mm_datasets
from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator
from torchtitan.hf_datasets.multimodal.utils.audio import load_pcm_waveform
from torchtitan.hf_datasets.multimodal.utils.video import load_npy_video_frames
from torchtitan_recipes.tests.datasets import synthetic_multimodal


_ASSET_ROOT = Path(__file__).resolve().parents[4] / "assets/synthetic_multimodal"
_TOKEN_STRINGS = {
    "image": "<image>",
    "video": "<video>",
    "vision_start": "<vision_start>",
    "vision_end": "<vision_end>",
    "pad": "<pad>",
    "audio": "<audio>",
    "audio_start": "<audio_start>",
    "audio_end": "<audio_end>",
}


class _AudioMultiModalTokenizer(MultiModalTokenizer):
    @dataclass(kw_only=True, slots=True)
    class Config(MultiModalTokenizer.Config):
        audio_token: str
        audio_start_token: str
        audio_end_token: str

    TOKEN_FIELDS = (
        *MultiModalTokenizer.TOKEN_FIELDS,
        "audio",
        "audio_start",
        "audio_end",
    )
    LOSS_MASK_TOKEN_FIELDS = (
        *MultiModalTokenizer.LOSS_MASK_TOKEN_FIELDS,
        "audio",
        "audio_start",
        "audio_end",
    )


@pytest.fixture
def audio_tokenizer(tmp_path):
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "000": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.add_special_tokens(
        [AddedToken(token, special=True) for token in _TOKEN_STRINGS.values()]
    )
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<eos>"}))
    tokenizer.add_special_tokens([AddedToken("<eos>", special=True)])
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    config = _AudioMultiModalTokenizer.Config(
        **{f"{name}_token": token for name, token in _TOKEN_STRINGS.items()}
    )
    return _AudioMultiModalTokenizer(config, tokenizer_path=str(tmp_path))


def _first_manifest_row(modality: str) -> dict:
    manifest = _ASSET_ROOT / f"{modality}_text/data.jsonl"
    return json.loads(manifest.read_text().splitlines()[0])


def _assert_media_targets_masked(result, tokenizer, fields):
    target_ids = result["input_ids"][1:]
    labels = result["labels"][:-1]
    media_token_ids = torch.tensor(
        [getattr(tokenizer, f"{field}_id") for field in fields]
    )
    media_targets = torch.isin(target_ids, media_token_ids)
    assert torch.any(media_targets)
    assert torch.all(labels[media_targets] == IGNORE_INDEX)


def test_audio_sample_processor_resamples_truncates_and_masks(audio_tokenizer):
    row = _first_manifest_row("audio")
    waveform, sample_rate = load_pcm_waveform(
        _ASSET_ROOT / "audio_text" / row["audio_path"],
        expected_num_channels=1,
    )

    result = mm_datasets._process_audio_text_sample(
        text=row["text"],
        waveform=waveform,
        source_sample_rate=sample_rate,
        tokenizer=audio_tokenizer,
        target_sample_rate=16_000,
        max_audio_samples=400,
        audio_token_count_fn=lambda num_samples, sample_rate: 5,
    )

    assert result["waveforms"][0].shape == (400, 1)
    assert result["waveforms"][0].dtype == torch.float32
    assert int((result["input_ids"] == audio_tokenizer.audio_id).sum()) == 5
    _assert_media_targets_masked(
        result,
        audio_tokenizer,
        ("audio", "audio_start", "audio_end"),
    )


def test_video_sample_processor_resizes_and_masks(audio_tokenizer):
    row = _first_manifest_row("video")
    frames = load_npy_video_frames(_ASSET_ROOT / "video_text" / row["video_path"])

    result = mm_datasets._process_video_text_sample(
        text=row["text"],
        frames=frames,
        tokenizer=audio_tokenizer,
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=1,
        min_pixels=1,
        max_pixels=1_000_000,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        resize_fn=lambda height, width, **kwargs: (height, width, 0, 6),
        max_patches=128,
        max_patches_per_side=32,
    )

    assert result["pixel_values_videos"][0].shape == (2, 28, 42, 3)
    assert torch.any(result["input_ids"] == audio_tokenizer.video_id)
    _assert_media_targets_masked(
        result,
        audio_tokenizer,
        ("video", "vision_start", "vision_end"),
    )


def test_synthetic_datasets_register_absolute_sources_and_processors():
    expected = {
        "synthetic-audio-text": ("audio_text", mm_datasets.AudioProcessor.Config),
        "synthetic-video-text": ("video_text", mm_datasets.VisionProcessor.Config),
    }
    for name, (directory, processor_type) in expected.items():
        dataset = synthetic_multimodal.SYNTHETIC_MM_DATASETS[name]
        assert isinstance(dataset, SingleDatasetConfig)
        assert isinstance(dataset.source, IndexedJsonlSource.Config)
        assert dataset.source.patterns == (
            str((_ASSET_ROOT / directory / "data.jsonl").resolve()),
        )
        assert isinstance(dataset.processor, processor_type)


def _synthetic_loader(audio_tokenizer, dataset_name: str):
    dataset = synthetic_multimodal.SYNTHETIC_MM_DATASETS[dataset_name]
    assert dataset.processor is not None
    if dataset_name == "synthetic-video-text":
        dataset = replace(
            dataset,
            processor=replace(
                dataset.processor,
                min_pixels=784,
                max_pixels=200_000,
            ),
        )
    collator = MultiModalCollator.Config()
    if dataset_name == "synthetic-audio-text":
        collator = replace(
            collator,
            waveform_pad_multiple=4,
            audio_input_channels=1,
        )
    return GrainDataLoader.Config(
        dataset=dataset,
        collator=collator,
        shuffle=True,
        repeat=True,
        read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
        num_prefetch_microbatches=1,
    ).build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=audio_tokenizer,
        max_context_length=512,
        num_tokens_per_microbatch=512,
    )


@pytest.mark.parametrize(
    "dataset_name",
    ["synthetic-audio-text", "synthetic-video-text"],
)
def test_synthetic_loader_restores_exact_next_batch(audio_tokenizer, dataset_name):
    loader = _synthetic_loader(audio_tokenizer, dataset_name)
    iterator = iter(loader)
    try:
        next(iterator)
        state = loader.state_dict()
        expected = next(iterator).as_input_dict()

        resumed = _synthetic_loader(audio_tokenizer, dataset_name)
        try:
            resumed.load_state_dict(state)
            actual = next(iter(resumed)).as_input_dict()
        finally:
            resumed.close()
    finally:
        loader.close()

    assert actual.keys() == expected.keys()
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, torch.Tensor):
            assert torch.equal(actual_value, expected_value), key
        else:
            assert actual_value == expected_value, key
    media_key = (
        "waveforms" if dataset_name == "synthetic-audio-text" else "pixel_values_videos"
    )
    assert actual[media_key].numel() > 0


def _collator(audio_tokenizer):
    context = DatasetBuildContext(
        tokenizer=audio_tokenizer,
        max_context_length=16,
        num_tokens_per_microbatch=16,
        read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
    )
    return MultiModalCollator.Config(
        waveform_pad_multiple=4,
        audio_input_channels=1,
    ).build(context=context)


def _collator_row(audio_tokenizer, waveforms):
    tokens = torch.tensor(
        [
            audio_tokenizer.audio_start_id,
            audio_tokenizer.audio_id,
            audio_tokenizer.audio_end_id,
        ]
    )
    return {
        "input_ids": tokens,
        "labels": tokens.clone(),
        "positions": torch.arange(len(tokens)),
        "pixel_values": [],
        "pixel_values_videos": [],
        "waveforms": waveforms,
    }


def test_audio_collator_pads_waveforms_in_clip_order(audio_tokenizer):
    first = torch.tensor([[0.1], [0.2], [0.3]])
    second = torch.tensor([[-0.1], [-0.2], [-0.3], [-0.4], [-0.5]])

    batch = _collator(audio_tokenizer)(
        [_collator_row(audio_tokenizer, [first, second])]
    )

    assert batch.model_kwargs["waveform_lengths"].tolist() == [3, 5]
    assert batch.model_kwargs["waveform_padded_lengths"].tolist() == [4, 8]
    packed = batch.model_kwargs["waveforms"]
    assert torch.equal(packed[:3], first)
    assert torch.equal(packed[4:9], second)


def test_audio_collator_omits_audio_kwargs_without_waveforms(audio_tokenizer):
    batch = _collator(audio_tokenizer)([_collator_row(audio_tokenizer, [])])

    assert (
        not {
            "waveforms",
            "waveform_lengths",
            "waveform_padded_lengths",
        }
        & batch.model_kwargs.keys()
    )
