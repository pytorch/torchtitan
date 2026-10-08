# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Video processing utilities for multimodal datasets."""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

import torchvision.transforms.v2.functional as TVF

from .image import smart_resize


logger = logging.getLogger(__name__)


def _resize_video_to_pixel_budget(
    height: int,
    width: int,
    *,
    patch_size: int,
    merge_size: int,
    min_pixels: int,
    max_pixels: int,
    **_: object,
) -> tuple[int, int, int, int]:
    """Adapt the default video resize geometry to the shared callback API."""
    resize_h, resize_w = smart_resize(
        height,
        width,
        factor=patch_size * merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return resize_h, resize_w, 0, 0


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


def load_video(
    path: str,
    fps: float = 2.0,
    min_frames: int = 4,
    max_frames: int = 768,
) -> torch.Tensor | None:
    """Load and sample frames from a video file using PyAV.

    Iterates through the video stream and only converts the sampled frames
    to numpy, avoiding full decode of every frame.

    Args:
        path: Path to the video file.
        fps: Target frames per second for sampling.
        min_frames: Minimum number of frames to sample.
        max_frames: Maximum number of frames to sample.

    Returns:
        Tensor of shape (T, H, W, C) in uint8, channel-last format,
        or None if loading fails.
    """
    try:
        import av  # pyrefly: ignore [missing-import]

        with av.open(path) as container:
            stream = container.streams.video[0]

            video_fps = float(stream.average_rate or stream.guessed_rate or 24)
            total_frames = stream.frames
            if total_frames == 0 and stream.duration:
                total_frames = int(
                    float(stream.duration * stream.time_base) * video_fps
                )
            if total_frames == 0:
                # No metadata available — decode all frames to count them
                container.seek(0)
                all_frames = [
                    frame.to_ndarray(format="rgb24")
                    for frame in container.decode(video=0)
                ]
                if not all_frames:
                    return None
                total_frames = len(all_frames)
                duration = total_frames / video_fps
                nframes = max(min_frames, min(int(duration * fps), max_frames))
                nframes = min(nframes, total_frames)
                indices = np.linspace(0, total_frames - 1, nframes).astype(int).tolist()
                selected = [all_frames[i] for i in indices]
                return torch.from_numpy(np.stack(selected))

            duration = total_frames / video_fps
            nframes = int(duration * fps)
            nframes = max(min_frames, min(nframes, max_frames))
            nframes = min(nframes, total_frames)

            # Compute which frame indices to keep
            indices = set(
                np.linspace(0, total_frames - 1, nframes).astype(int).tolist()
            )

            # Must decode sequentially — inter-frame codecs (H.264/H.265) have
            # dependencies between frames. Only selected frames are converted to
            # numpy to avoid the RGB conversion cost.
            frames = []
            container.seek(0)
            for i, frame in enumerate(container.decode(video=0)):
                if i in indices:
                    frames.append(frame.to_ndarray(format="rgb24"))

        if not frames:
            return None

        return torch.from_numpy(np.stack(frames))  # (T, H, W, C)

    except Exception as e:
        logger.warning(f"Error loading video {path}: {e}")
        return None


def process_video(
    video: torch.Tensor,
    patch_size: int,
    merge_size: int,
    max_pixels: int,
    min_pixels: int,
    image_mean: tuple[float, ...] = (0.5, 0.5, 0.5),
    image_std: tuple[float, ...] = (0.5, 0.5, 0.5),
    resize_fn: Callable[..., tuple[int, int, int, int]] = _resize_video_to_pixel_budget,
    image_interpolation_mode: TVF.InterpolationMode = TVF.InterpolationMode.BICUBIC,
    max_patches: int = 4096,
    max_patches_per_side: int = 512,
) -> torch.Tensor:
    """Resize and normalize video frames for VLM training.

    Uses torchvision v2 APIs with uint8 resize for faster SIMD paths.

    Args:
        video: Raw video tensor of shape (T, H, W, C) in uint8.
        patch_size: Spatial patch size.
        merge_size: Spatial merge size.
        max_pixels: Maximum spatial pixels per frame (H * W budget).
        min_pixels: Minimum spatial pixels per frame (H * W budget).
        image_mean: Per-channel mean for normalization.
        image_std: Per-channel std for normalization.
        resize_fn: Resize-strategy callable shared with ``process_image``.
        image_interpolation_mode: Torchvision interpolation mode used to resize
            the video frames.
        max_patches: Pre-padding raw-patch budget.
        max_patches_per_side: Pre-padding per-side raw-patch limit.

    Returns:
        Normalized tensor of shape (T, H', W', C) in float32.
    """
    _, height, width, _ = video.shape
    resize_h, resize_w, pad_h, pad_w = resize_fn(
        height,
        width,
        patch_size=patch_size,
        merge_size=merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        max_patches=max_patches,
        max_patches_per_side=max_patches_per_side,
    )

    # Resize on uint8 for faster SIMD paths (AVX2/NEON)
    # (T, H, W, C) → (T, C, H, W) for torchvision
    video = video.permute(0, 3, 1, 2)  # (T, C, H, W) uint8
    if (resize_h, resize_w) != (height, width):
        video = TVF.resize(
            video,
            [resize_h, resize_w],
            interpolation=image_interpolation_mode,
            antialias=True,
        )
    if pad_h or pad_w:
        video = TVF.pad(video, [0, 0, pad_w, pad_h])

    # uint8 → float32 [0, 1] → normalize → channel-last
    video = TVF.to_dtype(video, torch.float32, scale=True)
    video = TVF.normalize(video, list(image_mean), list(image_std), inplace=True)
    # (T, C, H', W') → (T, H', W', C)
    return video.permute(0, 2, 3, 1)
