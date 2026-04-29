# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Video processing utilities for multimodal datasets."""

import numpy as np
import torch

# pyrefly: ignore [missing-import]
import torchvision.transforms.v2.functional as TVF

from torchtitan.tools.logging import logger

from .image import smart_resize


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

    Returns:
        Normalized tensor of shape (T, H', W', C) in float32.
    """
    T, H, W, C = video.shape
    factor = patch_size * merge_size

    target_h, target_w = smart_resize(
        H,
        W,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    # Resize on uint8 for faster SIMD paths (AVX2/NEON)
    # (T, H, W, C) → (T, C, H, W) for torchvision
    video = video.permute(0, 3, 1, 2)  # (T, C, H, W) uint8
    video = TVF.resize(
        video,
        [target_h, target_w],
        interpolation=TVF.InterpolationMode.BICUBIC,
        antialias=True,
    )

    # uint8 → float32 [0, 1] → normalize → channel-last
    video = TVF.to_dtype(video, torch.float32, scale=True)
    video = TVF.normalize(video, list(image_mean), list(image_std), inplace=True)
    # (T, C, H', W') → (T, H', W', C)
    return video.permute(0, 2, 3, 1)
