# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Repository-local synthetic datasets for multimodal test recipes."""

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import torchvision.transforms.v2.functional as TVF

from torchtitan.components.data.dataset import SingleDatasetConfig
from torchtitan.components.data.sources import IndexedJsonlSource
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.hf_datasets.multimodal.mm_datasets import (
    _process_audio_text_sample,
    _process_video_text_sample,
    AudioProcessor,
    VisionProcessor,
)
from torchtitan.hf_datasets.multimodal.utils.audio import load_pcm_waveform
from torchtitan.hf_datasets.multimodal.utils.video import load_npy_video_frames


_ASSET_ROOT = (
    Path(__file__).resolve().parents[3] / "tests/assets/synthetic_multimodal"
).resolve()
_AUDIO_ROOT = (_ASSET_ROOT / "audio_text").resolve()
_VIDEO_ROOT = (_ASSET_ROOT / "video_text").resolve()


def _process_synthetic_audio_text_sample(
    sample: dict[str, Any],
    tokenizer: MultiModalTokenizer,
    *,
    audio_root: Path,
    expected_num_channels: int,
    target_sample_rate: int,
    max_audio_samples: int | None,
    audio_token_count_fn: Callable[[int, int], int],
) -> dict[str, Any] | None:
    waveform, source_sample_rate = load_pcm_waveform(
        audio_root / sample["audio_path"],
        expected_num_channels=expected_num_channels,
    )
    return _process_audio_text_sample(
        text=sample["text"],
        waveform=waveform,
        source_sample_rate=source_sample_rate,
        tokenizer=tokenizer,
        target_sample_rate=target_sample_rate,
        max_audio_samples=max_audio_samples,
        audio_token_count_fn=audio_token_count_fn,
    )


def _process_synthetic_video_text_sample(
    sample: dict[str, Any],
    tokenizer: MultiModalTokenizer,
    patch_size: int,
    temporal_patch_size: int,
    spatial_merge_size: int,
    min_pixels: int,
    max_pixels: int,
    image_mean: tuple[float, ...],
    image_std: tuple[float, ...],
    resize_fn: Callable[..., tuple[int, int, int, int]],
    max_patches: int,
    max_patches_per_side: int,
    *,
    video_root: Path,
    image_interpolation_mode: TVF.InterpolationMode = TVF.InterpolationMode.BICUBIC,
) -> dict[str, Any] | None:
    frames = load_npy_video_frames(video_root / sample["video_path"])
    return _process_video_text_sample(
        text=sample["text"],
        frames=frames,
        tokenizer=tokenizer,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        spatial_merge_size=spatial_merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        image_mean=image_mean,
        image_std=image_std,
        resize_fn=resize_fn,
        image_interpolation_mode=image_interpolation_mode,
        max_patches=max_patches,
        max_patches_per_side=max_patches_per_side,
    )


def _audio_token_count(
    num_samples: int,
    sample_rate: int,
    *,
    samples_per_audio_token: int,
) -> int:
    del sample_rate
    return (num_samples + samples_per_audio_token - 1) // samples_per_audio_token


SYNTHETIC_MM_DATASETS: dict[str, SingleDatasetConfig] = {
    "synthetic-audio-text": SingleDatasetConfig(
        source=IndexedJsonlSource.Config(
            patterns=(str(_AUDIO_ROOT / "data.jsonl"),),
        ),
        processor=AudioProcessor.Config(
            sample_processor=partial(
                _process_synthetic_audio_text_sample,
                audio_root=_AUDIO_ROOT,
            ),
            expected_num_channels=1,
            target_sample_rate=16_000,
            max_audio_samples=None,
            audio_token_count_fn=partial(
                _audio_token_count,
                samples_per_audio_token=320,
            ),
        ),
        post_filters=(lambda sample: sample is not None,),
    ),
    "synthetic-video-text": SingleDatasetConfig(
        source=IndexedJsonlSource.Config(
            patterns=(str(_VIDEO_ROOT / "data.jsonl"),),
        ),
        processor=VisionProcessor.Config(
            sample_processor=partial(
                _process_synthetic_video_text_sample,
                video_root=_VIDEO_ROOT,
            ),
        ),
        post_filters=(lambda sample: sample is not None,),
    ),
}
