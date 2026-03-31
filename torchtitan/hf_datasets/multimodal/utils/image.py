# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Image and video preprocessing utilities for Qwen3-VL multimodal datasets.

Handles resizing, normalization, and patch extraction for the vision encoder.
The preprocessing matches HuggingFace's Qwen2VLImageProcessor so that
converted checkpoints produce numerically equivalent outputs.
"""

import math
from io import BytesIO

import einops as E
import requests
import torch
import torchvision.transforms.v2.functional as TVF

from PIL import Image

from torchtitan.tools.logging import logger


def process_image(
    image: str | bytes | Image.Image,
    patch_size: int = 16,
    merge_size: int = 2,
    max_pixels: int = 16777216,
    min_pixels: int = 65536,
    image_mean: tuple[float, ...] = (0.5, 0.5, 0.5),
    image_std: tuple[float, ...] = (0.5, 0.5, 0.5),
) -> torch.Tensor | None:
    """Load and preprocess a single image for Qwen3-VL.

    Resizes to a pixel budget while keeping both dimensions multiples of
    ``patch_size * merge_size``, then normalizes with the given mean/std.

    Args:
        image: PIL Image, raw bytes, file path, or HTTP(S) URL.
        patch_size: Spatial patch size used by the vision encoder.
        merge_size: Spatial merge factor (patches merged per dimension).
        max_pixels: Upper pixel budget for resizing.
        min_pixels: Lower pixel budget for resizing.
        image_mean: Per-channel mean for normalization.
        image_std: Per-channel std for normalization.

    Returns:
        Tensor of shape (1, H, W, C) with a dummy temporal dim, or None on failure.
    """
    try:
        if isinstance(image, str) and image.startswith("http"):
            response = requests.get(image, timeout=10)
            image = Image.open(BytesIO(response.content))
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        elif isinstance(image, str):
            image = Image.open(image)

        if image.mode != "RGB":
            image = image.convert("RGB")

        factor = patch_size * merge_size
        original_width, original_height = image.size

        # Ensure both dimensions are at least ``factor`` so that
        # smart_resize always has a valid starting point.
        if original_height < factor or original_width < factor:
            scale = max(factor / original_width, factor / original_height)
            original_width = int(original_width * scale)
            original_height = int(original_height * scale)

        resized_height, resized_width = smart_resize(
            original_height,
            original_width,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        # Bicubic resize with antialias to match HF's processor.
        img_tensor = TVF.pil_to_tensor(image)  # (C, H, W) uint8
        img_tensor = TVF.resize(
            img_tensor,
            [resized_height, resized_width],
            interpolation=TVF.InterpolationMode.BICUBIC,
            antialias=True,
        )

        # [0, 255] → [0, 1] → normalize
        img_tensor = img_tensor.float() / 255.0
        mean = torch.tensor(image_mean).view(3, 1, 1)
        std = torch.tensor(image_std).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        # (C, H, W) → (1, H, W, C)
        return img_tensor.permute(1, 2, 0).unsqueeze(0)

    except Exception as e:
        logger.warning(f"Error processing image: {e}")
        return None


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
    num_frames: int = 1,
    temporal_factor: int = 1,
) -> tuple[int, int]:
    """Compute target (height, width) that satisfy pixel budget constraints.

    Both output dimensions are rounded to multiples of ``factor``.  The total
    pixel count ``t * h * w`` is kept within [min_pixels, max_pixels].

    Works for both images (``num_frames=1``) and videos (accounts for the
    temporal dimension in the total pixel budget).

    Args:
        height: Original height.
        width: Original width.
        factor: Spatial rounding factor (``patch_size * merge_size``).
        min_pixels: Minimum spatial pixels per frame.
        max_pixels: Maximum total pixels (T * H * W budget).
        num_frames: Number of frames. Use 1 for images.
        temporal_factor: Temporal patch size for rounding T. Use 1 for images.

    Returns:
        (resized_height, resized_width)
    """
    if max(height, width) / min(height, width) > 200:
        logger.warning(
            f"Aspect ratio {max(height, width) / min(height, width):.1f} exceeds 200"
        )

    t = max(1, round(num_frames / temporal_factor)) * temporal_factor

    # Round spatial dims to nearest multiple of factor
    h_bar = max(round(height / factor) * factor, factor)
    w_bar = max(round(width / factor) * factor, factor)

    # Scale up if below minimum spatial pixels
    if h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h_bar * w_bar))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    # Scale down if total pixels exceed budget (takes priority over min_pixels)
    if t * h_bar * w_bar > max_pixels:
        beta = math.sqrt((h_bar * w_bar) / (max_pixels / t))
        h_bar = max(math.floor(height / beta / factor) * factor, factor)
        w_bar = max(math.floor(width / beta / factor) * factor, factor)

    return h_bar, w_bar


def calculate_vision_tokens(
    num_frames: int,
    height: int,
    width: int,
    patch_size: int,
    spatial_merge_size: int,
    temporal_patch_size: int,
) -> tuple[int, int, int]:
    """Calculate the number of visual tokens after patching and merging.

    Args:
        num_frames: Number of frames (T). Use 1 for images.
        height: Frame height in pixels.
        width: Frame width in pixels.
        patch_size: Spatial patch size.
        spatial_merge_size: Spatial merge factor.
        temporal_patch_size: Temporal patch size.

    Returns:
        (total_tokens, tokens_per_row, num_rows)
    """
    t_patches = math.ceil(num_frames / temporal_patch_size)
    tokens_per_row = width // (patch_size * spatial_merge_size)
    num_rows = height // (patch_size * spatial_merge_size)
    total_tokens = t_patches * tokens_per_row * num_rows
    return total_tokens, tokens_per_row, num_rows


def vision_to_patches(
    img: torch.Tensor,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert an image/video tensor to flattened patches in block order.

    Patches are ordered so that each ``merge_size × merge_size`` spatial group
    is contiguous, matching the layout expected by the vision encoder's
    spatial merge layer.

    Args:
        img: (T, H, W, C) image or video tensor.
        patch_size: Spatial patch size (pixels per patch side).
        temporal_patch_size: Temporal patch size (frames per temporal patch).
        merge_size: Spatial merge size (patches merged per dimension).

    Returns:
        patches: (num_patches, patch_dim) flattened patch vectors in
            channel-first ``(C, pt, ph, pw)`` layout matching Conv3d kernels.
        grid_thw: (3,) tensor ``[T_patches, H_patches, W_patches]`` counting
            raw patches (before spatial merging).
    """
    T, H, W, C = img.shape
    ps = patch_size
    ts = temporal_patch_size

    # Pad temporal dim by repeating the last frame to reach a multiple of
    # temporal_patch_size, ref: HF's Qwen2VLImageProcessor._preprocess.
    if T % ts != 0:
        pad_t = ts - (T % ts)
        img = torch.cat([img, img[-1:].expand(pad_t, -1, -1, -1)], dim=0)
        T = img.shape[0]

    T_patches = T // ts
    H_patches = H // ps
    W_patches = W // ps

    # Reshape (T, H, W, C) → (num_patches, patch_dim) in block order:
    #   T = t × pt        temporal patches × frames per temporal patch
    #   H = bh × m × ph   block rows × merge patches × pixels per patch
    #   W = bw × n × pw   block cols × merge patches × pixels per patch
    # Sequence order:  (t, bh, bw, m, n) — merge group is contiguous
    # Patch vector:    (c, pt, ph, pw)   — channel-first
    patches = E.rearrange(
        img,
        "(t pt) (bh m ph) (bw n pw) c -> (t bh bw m n) (c pt ph pw)",
        pt=ts,
        ph=ps,
        pw=ps,
        m=merge_size,
        n=merge_size,
    )

    grid_thw = torch.tensor([T_patches, H_patches, W_patches])
    return patches, grid_thw
