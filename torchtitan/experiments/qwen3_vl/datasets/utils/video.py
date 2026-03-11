# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Video processing utilities for Qwen3-VL datasets."""

import math

import numpy as np
import torch

from torchtitan.tools.logging import logger


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
        import av

        container = av.open(path)
        stream = container.streams.video[0]

        video_fps = float(stream.average_rate or stream.guessed_rate or 24)
        total_frames = stream.frames
        if total_frames == 0 and stream.duration:
            total_frames = int(float(stream.duration * stream.time_base) * video_fps)
        if total_frames == 0:
            # No metadata available — decode all frames to count them
            container.seek(0)
            all_frames = [
                frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)
            ]
            container.close()
            if not all_frames:
                return None
            total_frames = len(all_frames)
            duration = total_frames / video_fps
            nframes = max(min_frames, min(int(duration * fps), max_frames))
            nframes = min(nframes, total_frames)
            indices = set(
                np.linspace(0, total_frames - 1, nframes).astype(int).tolist()
            )
            selected = [all_frames[i] for i in sorted(indices)]
            return torch.from_numpy(np.stack(selected))

        duration = total_frames / video_fps
        nframes = int(duration * fps)
        nframes = max(min_frames, min(nframes, max_frames))
        nframes = min(nframes, total_frames)

        # Compute which frame indices to keep
        indices = set(np.linspace(0, total_frames - 1, nframes).astype(int).tolist())

        # Iterate stream, only convert selected frames to numpy
        frames = []
        container.seek(0)
        for i, frame in enumerate(container.decode(video=0)):
            if i in indices:
                frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) == nframes:
                    break

        container.close()

        if not frames:
            return None

        return torch.from_numpy(np.stack(frames))  # (T, H, W, C)

    except Exception as e:
        logger.warning(f"Error loading video {path}: {e}")
        return None


def smart_resize_video(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Compute target spatial dimensions for video frames.

    Rounds spatial dims to multiples of ``factor`` (patch_size * merge_size)
    and scales down if the total pixel budget (T * H * W) is exceeded.

    Args:
        num_frames: Number of sampled frames (T).
        height: Original frame height.
        width: Original frame width.
        temporal_factor: Temporal patch size for rounding T.
        factor: Spatial factor (patch_size * merge_size).
        min_pixels: Minimum spatial pixels per frame.
        max_pixels: Maximum total pixels (T * H * W budget).

    Returns:
        (resized_height, resized_width)
    """
    # Round temporal dim
    t = max(1, round(num_frames / temporal_factor)) * temporal_factor

    # Round spatial dims to nearest factor
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    # Ensure minimum spatial size
    h_bar = max(h_bar, factor)
    w_bar = max(w_bar, factor)

    # Scale up if below minimum spatial pixels (before max budget check)
    spatial_pixels = h_bar * w_bar
    if spatial_pixels < min_pixels:
        beta = math.sqrt(min_pixels / spatial_pixels)
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    # Scale down if total pixels exceed budget (this takes priority)
    total_pixels = t * h_bar * w_bar
    if total_pixels > max_pixels:
        max_spatial = max_pixels / t
        beta = math.sqrt((h_bar * w_bar) / max_spatial)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
        h_bar = max(h_bar, factor)
        w_bar = max(w_bar, factor)

    return h_bar, w_bar


def process_video(
    video: torch.Tensor,
    patch_size: int,
    merge_size: int,
    temporal_patch_size: int,
    max_pixels: int,
    min_pixels: int,
    image_mean: tuple[float, ...] = (0.5, 0.5, 0.5),
    image_std: tuple[float, ...] = (0.5, 0.5, 0.5),
) -> torch.Tensor | None:
    """Resize and normalize video frames for Qwen3-VL.

    Args:
        video: Raw video tensor of shape (T, H, W, C) in uint8.
        patch_size: Spatial patch size.
        merge_size: Spatial merge size.
        temporal_patch_size: Temporal patch size.
        max_pixels: Maximum total pixels (T * H * W budget).
        min_pixels: Minimum spatial pixels per frame.
        image_mean: Per-channel mean for normalization.
        image_std: Per-channel std for normalization.

    Returns:
        Normalized tensor of shape (T, H', W', C) in float32,
        or None on failure.
    """
    try:
        import torchvision.transforms.functional as F

        T, H, W, C = video.shape
        factor = patch_size * merge_size

        target_h, target_w = smart_resize_video(
            num_frames=T,
            height=H,
            width=W,
            temporal_factor=temporal_patch_size,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        # Resize frames: torchvision F.resize expects (..., H, W) format
        # video is (T, H, W, C) -> permute to (T, C, H, W)
        video = video.permute(0, 3, 1, 2).float()  # (T, C, H, W)
        video = F.resize(
            video, [target_h, target_w], interpolation=F.InterpolationMode.BICUBIC
        )
        # Back to channel-last: (T, H', W', C)
        video = video.permute(0, 2, 3, 1)

        # Normalize
        video = video / 255.0
        mean = torch.tensor(image_mean, dtype=video.dtype)
        std = torch.tensor(image_std, dtype=video.dtype)
        video = (video - mean) / std

        return video

    except Exception as e:
        logger.warning(f"Error processing video: {e}")
        return None


def calculate_video_tokens(
    num_frames: int,
    height: int,
    width: int,
    patch_size: int,
    spatial_merge_size: int,
    temporal_patch_size: int,
) -> tuple[int, int, int]:
    """Calculate number of tokens needed for a video.

    Args:
        num_frames: Number of frames (T).
        height: Frame height (H).
        width: Frame width (W).
        patch_size: Spatial patch size.
        spatial_merge_size: Spatial merge factor.
        temporal_patch_size: Temporal patch size.

    Returns:
        (total_tokens, tokens_per_row, num_rows) where total_tokens
        includes the temporal dimension.
    """
    t_patches = math.ceil(num_frames / temporal_patch_size)
    h_merged = height // (patch_size * spatial_merge_size)
    w_merged = width // (patch_size * spatial_merge_size)
    total_tokens = t_patches * h_merged * w_merged
    return total_tokens, w_merged, h_merged
