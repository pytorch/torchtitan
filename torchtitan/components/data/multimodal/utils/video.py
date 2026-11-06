# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Video processing utilities for multimodal datasets."""

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


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
