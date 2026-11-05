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

import numpy as np
import torch

from torchtitan.components.data.dataset import SingleDatasetConfig
from torchtitan.components.data.multimodal.mm_datasets import (
    _process_audio_text_sample,
    _process_video_text_sample,
    AudioProcessor,
    VisionProcessor,
)
from torchtitan.components.data.multimodal.utils.audio import load_pcm_waveform
from torchtitan.components.data.multimodal.video_processor import VideoProcessor
from torchtitan.components.data.sources import IndexedJsonlSource
from torchtitan.components.tokenizer import MultiModalTokenizer


_ASSET_ROOT = (
    Path(__file__).resolve().parents[3] / "tests/assets/synthetic_multimodal"
).resolve()
_AUDIO_ROOT = (_ASSET_ROOT / "audio_text").resolve()
_VIDEO_ROOT = (_ASSET_ROOT / "video_text").resolve()


def load_npy_video_frames(path: str | Path) -> torch.Tensor:
    """Load a deterministic uint8 frame array from a NumPy file.

    Args:
        path: Path to a ``.npy`` array shaped ``(T, H, W, 3)``.

    Returns:
        A C-contiguous uint8 tensor with the same shape and values.

    Raises:
        ValueError: If the array is not a non-empty, C-contiguous uint8 THWC
            array with three channels. NumPy also raises ``ValueError`` for
            object arrays because pickle loading is disabled.
    """
    frames_THWC = np.load(path, allow_pickle=False)
    if frames_THWC.dtype != np.uint8:
        raise ValueError(
            f"Video frame array must have dtype uint8, got {frames_THWC.dtype}"
        )
    if frames_THWC.ndim != 4:
        raise ValueError(
            f"Video frame array must have 4 dimensions (T, H, W, C), "
            f"got shape {frames_THWC.shape}"
        )
    if any(size == 0 for size in frames_THWC.shape):
        raise ValueError(
            f"Video frame array must have non-empty dimensions, "
            f"got shape {frames_THWC.shape}"
        )
    if frames_THWC.shape[-1] != 3:
        raise ValueError(
            f"Video frame array must have 3 channels, got {frames_THWC.shape[-1]}"
        )
    if not frames_THWC.flags.c_contiguous:
        raise ValueError("Video frame array must be C-contiguous")
    return torch.from_numpy(frames_THWC)


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
    video_processor: VideoProcessor | None,
    *,
    video_root: Path,
    **kwargs,
) -> dict[str, Any] | None:
    frames = load_npy_video_frames(video_root / sample["video_path"])
    return _process_video_text_sample(
        text=sample["text"],
        frames=frames,
        tokenizer=tokenizer,
        video_processor=video_processor,
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
