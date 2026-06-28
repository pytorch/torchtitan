# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Image preprocessing and vision utilities for multimodal datasets.

Handles image decoding, resizing, normalization, and patch extraction for the
vision encoder.
"""

import math
from collections.abc import Callable

import einops as E
import requests
import torch

import torchvision.io

import torchvision.transforms.v2.functional as TVF

from PIL import Image

from torchtitan.tools.logging import logger


def _decode_image(image: str | bytes | Image.Image) -> torch.Tensor:
    """Decode an image to a (C, H, W) uint8 RGB tensor.

    Uses torchvision.io.decode_image for bytes/paths (faster SIMD decode),
    falls back to TVF.pil_to_tensor for PIL Image inputs.
    """
    if isinstance(image, str) and image.startswith("http"):
        response = requests.get(image, timeout=10)
        image = response.content
    if isinstance(image, bytes):
        raw = torch.frombuffer(bytearray(image), dtype=torch.uint8)
        return torchvision.io.decode_image(raw, mode=torchvision.io.ImageReadMode.RGB)
    elif isinstance(image, str):
        return torchvision.io.decode_image(image, mode=torchvision.io.ImageReadMode.RGB)
    else:
        # PIL Image fallback
        if image.mode != "RGB":
            image = image.convert("RGB")
        return TVF.pil_to_tensor(image)


def resize_to_pixel_budget(
    height: int,
    width: int,
    *,
    patch_size: int,
    merge_size: int,
    min_pixels: int,
    max_pixels: int,
    **_: object,
) -> tuple[int, int, int, int]:
    """Resize so the spatial pixel count lands in ``[min_pixels, max_pixels]``,
    with both dims rounded to a ``patch_size * merge_size`` multiple (Qwen-VL
    convention). Content is rescaled to the grid, so no padding is needed.

    A resize strategy (``resize_fn``) for ``process_image`` -- extra budget
    kwargs (e.g. ``max_patches``) are accepted and ignored via ``**_``.

    Args:
        height: Original height in pixels.
        width: Original width in pixels.
        patch_size: Spatial patch size.
        merge_size: Spatial merge factor.
        min_pixels: Lower spatial-pixel budget.
        max_pixels: Upper spatial-pixel budget.

    Returns:
        ``(resize_h, resize_w, 0, 0)`` -- trailing zeros are padding (always 0
        here), kept for a uniform interface with ``resize_to_patch_budget``.
    """
    factor = patch_size * merge_size
    # Ensure both dims >= factor so the rounding has a valid starting point.
    if height < factor or width < factor:
        scale = max(factor / width, factor / height)
        width, height = int(width * scale), int(height * scale)

    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"Absolute aspect ratio must be smaller than 200, "
            f"got {max(height, width) / min(height, width):.1f}"
        )

    h_bar = max(round(height / factor) * factor, factor)
    w_bar = max(round(width / factor) * factor, factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(math.floor(height / beta / factor) * factor, factor)
        w_bar = max(math.floor(width / beta / factor) * factor, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar, 0, 0


def resize_to_patch_budget(
    height: int,
    width: int,
    *,
    patch_size: int,
    merge_size: int,
    max_patches: int,
    max_patches_per_side: int,
    **_: object,
) -> tuple[int, int, int, int]:
    """Cap the raw-patch count at ``max_patches`` (scale down,
    aspect-preserving) then pad right/bottom to a ``patch_size * merge_size``
    multiple. Small images are not upscaled.

    A resize strategy (``resize_fn``) for ``process_image`` -- extra budget
    kwargs (e.g. ``min_pixels`` / ``max_pixels``) are accepted and ignored via
    ``**_``.

    Args:
        height: Original height in pixels.
        width: Original width in pixels.
        patch_size: Spatial patch size.
        merge_size: Spatial merge factor.
        max_patches: Max raw patches per image.
        max_patches_per_side: Per-side patch cap (vision position-embedding limit).

    Returns:
        ``(resize_h, resize_w, pad_h, pad_w)`` -- resize to the first two, then
        pad right/bottom by the last two. Raises if the padded grid exceeds
        ``max_patches_per_side``.
    """
    h, w = height, width
    num_patches = (h // patch_size) * (w // patch_size)
    if num_patches > max_patches:
        scale = math.sqrt(max_patches / num_patches)
        h, w = int(h * scale), int(w * scale)

    factor = patch_size * merge_size
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor

    side = max((h + pad_h) // patch_size, (w + pad_w) // patch_size)
    if side >= max_patches_per_side:
        raise ValueError(
            f"image grid {(h + pad_h) // patch_size}x{(w + pad_w) // patch_size} "
            f"patches exceeds the vision position-embedding limit "
            f"{max_patches_per_side} per side"
        )
    return h, w, pad_h, pad_w


def process_image(
    image: str | bytes | Image.Image,
    patch_size: int = 16,
    merge_size: int = 2,
    max_pixels: int = 16777216,
    min_pixels: int = 65536,
    image_mean: tuple[float, ...] = (0.5, 0.5, 0.5),
    image_std: tuple[float, ...] = (0.5, 0.5, 0.5),
    resize_fn: Callable[..., tuple[int, int, int, int]] = resize_to_pixel_budget,
    max_patches: int = 4096,
    max_patches_per_side: int = 512,
) -> torch.Tensor | None:
    """Load and preprocess a single image for VLM training.

    Uses torchvision APIs for decoding and resizing (faster uint8 SIMD paths),
    then normalizes with the given mean/std. ``resize_fn`` is the resize strategy
    -- ``resize_to_pixel_budget`` (Qwen-VL, default) or
    ``resize_to_patch_budget`` (Kimi-VL) -- with the uniform signature
    ``(height, width, *, patch_size, merge_size, **budget) -> (resize_h,
    resize_w, pad_h, pad_w)``. All budget kwargs are passed through; each
    strategy uses the subset it needs.

    Args:
        image: Raw bytes, file path, HTTP(S) URL, or PIL Image.
        patch_size: Spatial patch size used by the vision encoder.
        merge_size: Spatial merge factor (patches merged per dimension).
        max_pixels: Upper pixel budget (``resize_to_pixel_budget``).
        min_pixels: Lower pixel budget (``resize_to_pixel_budget``).
        image_mean: Per-channel mean for normalization.
        image_std: Per-channel std for normalization.
        resize_fn: Resize-strategy callable (see above).
        max_patches: Max raw patches per image (``resize_to_patch_budget``).
        max_patches_per_side: Per-side patch cap (``resize_to_patch_budget``).

    Returns:
        Tensor of shape (1, H, W, C) with a dummy temporal dim, or None on failure.
    """
    try:
        # Decode to (C, H, W) uint8 tensor
        img_tensor = _decode_image(image)
        _, original_height, original_width = img_tensor.shape

        resize_h, resize_w, pad_h, pad_w = resize_fn(
            original_height,
            original_width,
            patch_size=patch_size,
            merge_size=merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_patches=max_patches,
            max_patches_per_side=max_patches_per_side,
        )
        if (resize_h, resize_w) != (original_height, original_width):
            # Bicubic resize on uint8 (leverages AVX2/NEON SIMD fast paths)
            img_tensor = TVF.resize(
                img_tensor,
                [resize_h, resize_w],
                interpolation=TVF.InterpolationMode.BICUBIC,
                antialias=True,
            )
        if pad_h or pad_w:
            # Pad right/bottom with zeros (black); after normalization these are
            # constant padding patches the encoder sees as part of the grid.
            img_tensor = TVF.pad(img_tensor, [0, 0, pad_w, pad_h])

        # uint8 → float32 [0, 1] → normalize
        img_tensor = TVF.to_dtype(img_tensor, torch.float32, scale=True)
        img_tensor = TVF.normalize(
            img_tensor, list(image_mean), list(image_std), inplace=True
        )

        # (C, H, W) → (1, H, W, C)
        return img_tensor.permute(1, 2, 0).unsqueeze(0)

    except Exception as e:
        logger.warning(f"Error processing image: {e}")
        return None


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
    patch_order: str = "block",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert an image/video tensor to flattened patches.

    ``patch_order`` selects the sequence layout:

    - ``"block"`` (default): each ``merge_size x merge_size`` spatial group is
      contiguous (sequence ``(t, bh, bw, m, n)``), matching Qwen-style mergers
      that fuse consecutive ``merge_size**2`` patches.
    - ``"raster"``: plain row-major ``(t, h, w)`` order, matching encoders whose
      RoPE / position embeddings index ``row = p // w, col = p % w`` (MoonViT3d).

    Example -- a 2x4 patch grid (h=2, w=4), ``merge_size=2``, patches labeled by
    their raster index ``row * w + col``::

        raster grid:   0 1 2 3
                       4 5 6 7

        raster order:  0 1 2 3 4 5 6 7
        block order:   0 1 4 5  2 3 6 7
                       \\_____/  \\_____/
                       2x2 block  2x2 block

    In block order the 4 patches that merge into one token are adjacent; in
    raster order they are not.

    Args:
        img: (T, H, W, C) image or video tensor.
        patch_size: Spatial patch size (pixels per patch side).
        temporal_patch_size: Temporal patch size (frames per temporal patch).
        merge_size: Spatial merge size (patches merged per dimension).
        patch_order: ``"block"`` or ``"raster"`` sequence layout.

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

    # Reshape (T, H, W, C) -> (num_patches, patch_dim). The patch vector is
    # always channel-first (c, pt, ph, pw); only the sequence ordering differs.
    if patch_order == "block":
        # (t, bh, bw, m, n) -- the merge group is contiguous.
        patches = E.rearrange(
            img,
            "(t pt) (bh m ph) (bw n pw) c -> (t bh bw m n) (c pt ph pw)",
            pt=ts,
            ph=ps,
            pw=ps,
            m=merge_size,
            n=merge_size,
        )
    elif patch_order == "raster":
        # (t, h, w) -- plain row-major (merge_size unused in the layout).
        patches = E.rearrange(
            img,
            "(t pt) (h ph) (w pw) c -> (t h w) (c pt ph pw)",
            pt=ts,
            ph=ps,
            pw=ps,
        )
    else:
        raise ValueError(
            f"patch_order must be 'block' or 'raster', got {patch_order!r}."
        )

    grid_thw = torch.tensor([T_patches, H_patches, W_patches])
    return patches, grid_thw
