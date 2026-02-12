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
        temporal_patch_size: Temporal patch size for videos
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

    # Ensure both dimensions are at least factor (required by _smart_resize)
    if original_height < factor or original_width < factor:
        scale_factor = max(factor / original_width, factor / original_height)
        original_width = int(original_width * scale_factor)
        original_height = int(original_height * scale_factor)

    resized_height, resized_width = _smart_resize(
        original_height, original_width, factor, max_patch_per_image, min_patch_per_image
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


def image_to_patches(
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

    # Convert to patches in block-order (matching position embedding order)
    # From 4d to 2d
    # On the left side:
    #   T = t × pt
    #   H = bh × m x ph
    #   W = bw x n x pw
    #   c = c, channels
    # On the right side:
    #   (t bh bw m n), sequence of tokens before merging
    #   (pt ph pw c), patch dimensions
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
