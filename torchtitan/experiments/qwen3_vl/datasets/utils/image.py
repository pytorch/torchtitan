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
    patch_size: int = 14,
    merge_size: int = 2,
    temporal_patch_size: int = 2,
    max_patch_per_image: int = 256,
    min_patch_per_image: int = 4,
) -> torch.Tensor | None:
    """Process a single image into normalized tensor format for Qwen3-VL.

    Args:
        image: PIL Image, bytes, or URL string
        patch_size: Size of each spatial patch
        merge_size: Spatial merge size factor
        temporal_patch_size: Temporal patch size
        max_patch_per_image: Maximum patches allowed per image
        min_patch_per_image: Minimum patches per image

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

        # Resize maintaining aspect ratio
        image = _resize_image_by_patch_count(
            image,
            max_patch_per_image=max_patch_per_image,
            patch_size=patch_size,
            merge_size=merge_size,
            min_patch_per_image=min_patch_per_image,
        )

        # Convert to numpy and normalize using Qwen3-VL normalization
        img_array = np.array(image)
        img_array = img_array / 255.0

        # Qwen3-VL uses OpenCLIP normalization
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        img_array = (img_array - mean) / std

        # Convert to tensor (1, H, W, 3) with dummy temporal dim
        return torch.from_numpy(img_array).float().unsqueeze(0)

    except Exception as e:
        logger.warning(f"Error processing image: {e}")
        return None


def _smart_resize(
    height: int,
    width: int,
    factor: int,
    max_patch_per_image: int,
    min_patch_per_image: int = 1,
):
    """Calculate dimensions that maintain aspect ratio and satisfy constraints."""
    if height < factor or width < factor:
        raise ValueError(
            f"height:{height} and width:{width} must be larger than factor:{factor}"
        )
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    # Calculate patch count from adjusted dimensions
    current_patches = (h_bar * w_bar) // (factor * factor)

    if current_patches > max_patch_per_image:
        max_area = max_patch_per_image * (factor * factor)
        beta = math.sqrt((h_bar * w_bar) / max_area)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif current_patches < min_patch_per_image:
        beta = math.sqrt(min_patch_per_image / current_patches)
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


def _resize_image_by_patch_count(
    image: Image.Image,
    max_patch_per_image: int,
    patch_size: int,
    merge_size: int,
    min_patch_per_image: int = 1,
) -> Image.Image:
    """Resize image while maintaining aspect ratio."""
    original_width, original_height = image.size
    factor = patch_size * merge_size

    current_patches = (original_height * original_width) // (factor * factor)

    if current_patches < min_patch_per_image:
        if current_patches == 0:
            scale_factor = max(factor / original_width, factor / original_height)
        else:
            scale_factor = math.sqrt(min_patch_per_image / current_patches)

        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        resized_height, resized_width = _smart_resize(
            new_height,
            new_width,
            factor,
            max_patch_per_image,
        )
        return image.resize((resized_width, resized_height))

    elif current_patches <= max_patch_per_image:
        resized_height, resized_width = _smart_resize(
            original_height, original_width, factor, max_patch_per_image
        )
        return image.resize((resized_width, resized_height))

    else:
        scale_factor = math.sqrt(max_patch_per_image / current_patches)
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        resized_height, resized_width = _smart_resize(
            new_height, new_width, factor, max_patch_per_image
        )
        return image.resize((resized_width, resized_height))


def calculate_image_tokens(
    image: Image.Image | torch.Tensor,
    patch_size: int,
    spatial_merge_size: int,
) -> tuple[int, int, int]:
    """Calculate number of tokens needed for an image."""
    if isinstance(image, torch.Tensor):
        height, width = image.shape[1:3]
    else:
        width, height = image.size

    tokens_per_row = width // (patch_size * spatial_merge_size)
    num_rows = height // (patch_size * spatial_merge_size)
    total_tokens = tokens_per_row * num_rows

    return total_tokens, tokens_per_row, num_rows


def convert_to_patches(
    pixel_values: torch.Tensor,
    patch_size: int,
    temporal_patch_size: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert single image/video tensor to patches and generate coordinate grids.

    Args:
        pixel_values: Tensor of shape (T, H, W, C)
        patch_size: Spatial patch size (height and width)
        temporal_patch_size: Temporal patch size

    Returns:
        patches: Tensor of shape (L, D)
        grid: Tensor of shape (L, 3) containing (t, h, w) coordinates
    """
    T, H, W, C = pixel_values.shape
    ps = patch_size
    ts = temporal_patch_size
    device = pixel_values.device

    # Pad temporal dimension if needed
    if T % ts != 0:
        pad_t = ts - (T % ts)
        pixel_values = torch.nn.functional.pad(pixel_values, (0, 0, 0, 0, 0, 0, 0, pad_t))
        T = pixel_values.shape[0]

    if H % ps != 0 or W % ps != 0:
        raise ValueError(
            f"Spatial dimensions {H},{W} must be divisible by patch_size {ps}"
        )

    patches = E.rearrange(
        pixel_values,
        "(t pt) (h ph) (w pw) c -> (t h w) (pt ph pw c)",
        pt=ts,
        ph=ps,
        pw=ps,
    )

    # Generate coordinate grid
    coords = torch.meshgrid(
        torch.arange(T // ts, device=device),
        torch.arange(H // ps, device=device),
        torch.arange(W // ps, device=device),
        indexing="ij",
    )
    grid = E.rearrange(torch.stack(coords), "coords t h w -> (t h w) coords")

    return patches, grid


def pad_patches(
    patches: torch.Tensor,
    grids: torch.Tensor,
    max_patches: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Pad or truncate patches and grids to max_patches length."""
    L, D = patches.shape

    if L == max_patches:
        return patches, grids
    elif L < max_patches:
        pad_len = max_patches - L
        zero_patches = torch.zeros(pad_len, D, device=patches.device)
        invalid_grids = torch.full((pad_len, 3), -1, device=grids.device)
        return (
            torch.cat([patches, zero_patches], 0),
            torch.cat([grids, invalid_grids], 0),
        )
    else:
        logger.error(
            f"Truncating Image Patches from {L} to {max_patches} should not happen."
        )
        return None, None


def pad_empty_images_to_target_batch_size(
    patches: torch.Tensor,
    grids: torch.Tensor,
    max_images: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad vision encoder batch with blank images if needed."""
    N, L, D = patches.shape
    if N >= max_images:
        return patches, grids

    blank_count = max_images - N
    blank_patches = torch.zeros(blank_count, L, D, device=patches.device)
    blank_grids = torch.full((blank_count, L, 3), -1, device=grids.device)
    return (
        torch.cat([patches, blank_patches], dim=0),
        torch.cat([grids, blank_grids], dim=0),
    )
