# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Video processing utilities for Qwen3-VL datasets."""

import numpy as np
import torch

from torchtitan.hf_datasets.utils.image import smart_resize
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

        target_h, target_w = smart_resize(
            H,
            W,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            num_frames=T,
            temporal_factor=temporal_patch_size,
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


