# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for image processing in Qwen3-VL datasets."""

import math
from io import BytesIO

import einops as E
import numpy as np
import requests
import torch

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
    """Process a single image into normalized tensor format for Qwen3-VL.

    Args:
        image: PIL Image, bytes, or URL string
        patch_size: Size of each spatial patch
        merge_size: Spatial merge size factor
        max_pixels: Maximum number of pixels
        min_pixels: Minimum number of pixels
        image_mean: Per-channel mean for normalization
        image_std: Per-channel std for normalization

    Returns:
        Tensor of shape (1, H, W, 3) or None if processing fails
    """
    try:
        # Convert various input formats to PIL Image
        if isinstance(image, str) and image.startswith("http"):
            response = requests.get(image, timeout=10)
            image = Image.open(BytesIO(response.content))
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        elif isinstance(image, str):
            image = Image.open(image)

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize maintaining aspect ratio within pixel budget
        factor = patch_size * merge_size
        original_width, original_height = image.size

        # Ensure both dimensions are at least factor before smart_resize
        if original_height < factor or original_width < factor:
            scale_factor = max(factor / original_width, factor / original_height)
            original_width = int(original_width * scale_factor)
            original_height = int(original_height * scale_factor)

        resized_height, resized_width = smart_resize(
            original_height,
            original_width,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        image = image.resize((resized_width, resized_height))

        # Convert to numpy and normalize
        img_array = np.array(image)
        img_array = img_array / 255.0

        mean = np.array(image_mean)
        std = np.array(image_std)
        img_array = (img_array - mean) / std

        # Convert to tensor (1, H, W, 3) with dummy temporal dim
        return torch.from_numpy(img_array).float().unsqueeze(0)

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
    """Compute target spatial dimensions that satisfy pixel budget constraints.

    Rounds spatial dims to multiples of ``factor`` and scales down/up to keep
    ``num_frames * h * w`` within [min_pixels, max_pixels].

    Works for both images (num_frames=1, temporal_factor=1) and videos
    (accounts for temporal dim in the total pixel budget).

    Args:
        height: Original height.
        width: Original width.
        factor: Spatial factor (patch_size * merge_size).
        min_pixels: Minimum spatial pixels per frame.
        max_pixels: Maximum total pixels (T * H * W budget).
        num_frames: Number of frames (T). Use 1 for images.
        temporal_factor: Temporal patch size for rounding T. Use 1 for images.

    Returns:
        (resized_height, resized_width)
    """
    if max(height, width) / min(height, width) > 200:
        logger.warning(
            f"Aspect ratio {max(height, width) / min(height, width):.1f} exceeds 200"
        )

    # Round temporal dim
    t = max(1, round(num_frames / temporal_factor)) * temporal_factor

    # Round spatial dims to nearest factor, ensuring minimum size
    h_bar = max(round(height / factor) * factor, factor)
    w_bar = max(round(width / factor) * factor, factor)

    # Scale up if below minimum spatial pixels
    spatial_pixels = h_bar * w_bar
    if spatial_pixels < min_pixels:
        beta = math.sqrt(min_pixels / spatial_pixels)
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    # Scale down if total pixels exceed budget (takes priority)
    total_pixels = t * h_bar * w_bar
    if total_pixels > max_pixels:
        max_spatial = max_pixels / t
        beta = math.sqrt((h_bar * w_bar) / max_spatial)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
        h_bar = max(h_bar, factor)
        w_bar = max(w_bar, factor)

    return h_bar, w_bar


def calculate_vision_tokens(
    num_frames: int,
    height: int,
    width: int,
    patch_size: int,
    spatial_merge_size: int,
    temporal_patch_size: int,
) -> tuple[int, int, int]:
    """Calculate number of tokens needed for an image or video.

    Args:
        num_frames: Number of frames (T). Use 1 for images.
        height: Frame height (H).
        width: Frame width (W).
        patch_size: Spatial patch size.
        spatial_merge_size: Spatial merge factor.
        temporal_patch_size: Temporal patch size. Use 1 for images.

    Returns:
        (total_tokens, tokens_per_row, num_rows) where total_tokens
        includes the temporal dimension.
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
    """Convert a single image/video tensor to patches in block-order.

    Args:
        img: Image tensor of shape (T, H, W, C)
        patch_size: Spatial patch size
        temporal_patch_size: Temporal patch size
        merge_size: Spatial merge size

    Returns:
        patches: (num_patches, patch_dim) flattened patch vectors
        grid_thw: (3,) tensor of [T_patches, H_patches, W_patches]
    """
    T, H, W, C = img.shape
    ps = patch_size
    ts = temporal_patch_size

    # Pad temporal dimension if needed
    if T % ts != 0:
        pad_t = ts - (T % ts)
        img = torch.nn.functional.pad(img, (0, 0, 0, 0, 0, 0, 0, pad_t))
        T = img.shape[0]

    # Calculate grid dimensions (in raw patches, before merging)
    T_patches = T // ts
    H_patches = H // ps
    W_patches = W // ps

    # Convert (T, H, W, C) to (num_patches, patch_dim) in block order
    #   T = t  × pt          t  = temporal patches  pt = frames per temporal patch
    #   H = bh × m × ph      bh = block rows        m  = merge patches per block
    #   W = bw × n × pw      bw = block cols        n  = merge patches per block
    #   ph/pw = pixels per patch (height/width)
    # Output sequence: (t bh bw m n) — m×n patches in each merge group are
    #   contiguous so the spatial merge layer can reshape to combine them.
    # Patch vector:    (pt ph pw c)  — flattened pixel values per patch.
    patches = E.rearrange(
        img,
        "(t pt) (bh m ph) (bw n pw) c -> (t bh bw m n) (pt ph pw c)",
        pt=ts,
        ph=ps,
        pw=ps,
        m=merge_size,
        n=merge_size,
    )

    grid_thw = torch.tensor([T_patches, H_patches, W_patches])
    return patches, grid_thw
