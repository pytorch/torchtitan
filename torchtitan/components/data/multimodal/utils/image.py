# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Image preprocessing and vision utilities for multimodal datasets.

Handles image decoding, vision token counts, and patch extraction for the
vision encoder; ``ImageProcessor`` resizes and normalizes.
"""

import math

import einops as E
import requests
import torch

import torchvision.io

import torchvision.transforms.v2.functional as TVF

from PIL import Image


def decode_image(image: str | bytes | Image.Image) -> torch.Tensor:
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
