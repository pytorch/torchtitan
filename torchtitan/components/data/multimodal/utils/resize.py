# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Resize rules for ``ImageProcessor.Config.resize_fn``.

A rule maps an image's ``(height, width)`` to ``(resize_h, resize_w, pad_h,
pad_w)`` on the vision encoder's ``patch_size * merge_size`` grid. Rules are
plain math with no torchvision import, so model ``flavors.py`` files can name
them in their image-processor settings without the optional dependency.
"""

import math


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

    A resize rule for ``ImageProcessor.Config.resize_fn`` -- extra budget
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
        here), kept for a uniform interface with ``resize_to_navit_patch_grid``.
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


def resize_to_navit_patch_grid(
    height: int,
    width: int,
    *,
    patch_size: int,
    merge_size: int,
    max_patches: int,
    max_patches_per_side: int,
    **_: object,
) -> tuple[int, int, int, int]:
    """Apply pre-padding total and per-side raw-patch budgets while preserving
    aspect ratio, then pad right/bottom to a ``patch_size * merge_size`` grid.
    Small images within both limits are not upscaled. Grid padding can make the
    final raw-patch count exceed ``max_patches``.

    A resize rule for ``ImageProcessor.Config.resize_fn`` -- extra budget
    kwargs (e.g. ``min_pixels`` / ``max_pixels``) are accepted and ignored via
    ``**_``.

    Args:
        height: Original height in pixels.
        width: Original width in pixels.
        patch_size: Spatial patch size.
        merge_size: Spatial merge factor.
        max_patches: Pre-padding raw-patch budget used to calculate resize scale.
        max_patches_per_side: Pre-padding per-side raw-patch limit.

    Returns:
        ``(resize_h, resize_w, pad_h, pad_w)`` -- resize to the first two, then
        pad right/bottom by the last two.
    """
    num_patches_h = max(1.0, height // patch_size)
    num_patches_w = max(1.0, width // patch_size)
    num_patches = num_patches_h * num_patches_w

    scale = min(
        1.0,
        math.sqrt(max_patches / num_patches),
        max_patches_per_side * patch_size / height,
        max_patches_per_side * patch_size / width,
    )
    h = max(1, int(height * scale))
    w = max(1, int(width * scale))
    h = min(h, max_patches_per_side * patch_size)
    w = min(w, max_patches_per_side * patch_size)

    factor = patch_size * merge_size
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    return h, w, pad_h, pad_w
