# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Image preprocessing for one model, shared by every component that turns
images into model inputs.

Like the tokenizer, an ``ImageProcessor`` is fixed by the model: the patch and
merge sizes match the vision encoder's patch embedding and merger, and the
resize rule and normalization match how the encoder was trained. Each
multimodal model keeps its settings next to its special tokens (e.g.
``QWEN3_5_IMAGE_PROCESSOR``), and recipes build the config from them:
``ImageProcessor.Config(**QWEN3_5_IMAGE_PROCESSOR)``. The trainer builds one from
``Trainer.Config.image_processor`` and hands it to the dataloader, as it does
the tokenizer. ``VideoProcessor`` extends it for video frames.

Preprocessing runs in two stages, split by where they run:

1. ``decode_and_resize``: per image, on the CPU. Decode and resize to the
   model's resolution: uint8 ``(1, H, W, C)``.
2. ``normalize_and_patchify``: a list of resized items, on their device.
   Normalize and cut into patches: float32 ``(num_patches, patch_dim)`` plus
   each item's ``grid_thw``.

The multimodal dataloader runs stage 1 in its sample processor and stage 2 in
the collator. A model whose preprocessing does not fit these stages (e.g.
tiling or cropping) overrides one and reuses the helpers in ``utils/image.py``
and the resize rules in ``utils/resize.py``.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import torchvision.transforms.v2.functional as TVF
from PIL import Image

from torchtitan.config import Configurable

from .utils.image import calculate_vision_tokens, decode_image, vision_to_patches


class ImageProcessor(Configurable):
    """Turns images into a model's vision-encoder inputs."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        patch_size: int
        """Spatial patch size of the vision encoder's patch embedding."""

        temporal_patch_size: int
        """Frames per temporal patch."""

        spatial_merge_size: int
        """Patches merged per spatial dimension into one placeholder token."""

        image_mean: tuple[float, ...]
        image_std: tuple[float, ...]

        resize_fn: Callable[..., tuple[int, int, int, int]]
        """Resize rule ``(height, width, *, patch_size, merge_size, **budget)
        -> (resize_h, resize_w, pad_h, pad_w)``, e.g. ``resize_to_pixel_budget``
        or ``resize_to_navit_patch_grid``. Each rule reads the budget fields it
        needs."""

        patch_order: Literal["block", "raster"]
        """Patch sequence layout the encoder expects; see ``vision_to_patches``."""

        min_pixels: int = 65_536
        """Lower pixel budget of ``resize_to_pixel_budget``."""

        max_pixels: int = 16_777_216
        """Upper pixel budget of ``resize_to_pixel_budget``."""

        max_patches: int = 4096
        """Patch budget of ``resize_to_navit_patch_grid``."""

        max_patches_per_side: int = 512
        """Per-side patch limit of ``resize_to_navit_patch_grid``."""

        image_interpolation_mode: TVF.InterpolationMode = TVF.InterpolationMode.BICUBIC

    def __init__(self, config: Config) -> None:
        self.config = config

    def decode_and_resize(self, image: str | bytes | Image.Image) -> torch.Tensor:
        """Decode one image and resize it: uint8 ``(1, H, W, C)``.

        Accepts encoded bytes, a file path, an HTTP(S) URL, or a PIL image.
        """
        image_CHW = self._resize(decode_image(image))
        return image_CHW.permute(1, 2, 0).unsqueeze(0)

    def normalize_and_patchify(
        self, items: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize resized items and cut them into patches.

        Args:
            items: Non-empty list of uint8 ``(T, H, W, C)`` items from
                ``decode_and_resize`` (or ``VideoProcessor.resize_video``).

        Returns:
            ``pixel_values``: float32 ``(num_patches, patch_dim)``, the items'
            patches concatenated; ``grid_thw``: ``(num_items, 3)`` patch grids
            ``[T, H, W]`` before merging. ``grid_thw.prod(-1)`` gives each
            item's length in ``pixel_values``.
        """
        config = self.config
        patches, grids = [], []
        for item_THWC in items:
            # A float item would be normalized again, in place.
            if item_THWC.dtype != torch.uint8:
                raise ValueError(
                    "normalize_and_patchify expects resized uint8 items, got "
                    f"{item_THWC.dtype}"
                )
            pixels_TCHW = TVF.to_dtype(
                item_THWC.permute(0, 3, 1, 2), torch.float32, scale=True
            )
            pixels_TCHW = TVF.normalize(
                pixels_TCHW,
                list(config.image_mean),
                list(config.image_std),
                inplace=True,
            )
            item_patches, grid_thw = vision_to_patches(
                pixels_TCHW.permute(0, 2, 3, 1),
                config.patch_size,
                config.temporal_patch_size,
                config.spatial_merge_size,
                patch_order=config.patch_order,
            )
            patches.append(item_patches)
            grids.append(grid_thw)
        return torch.cat(patches), torch.stack(grids)

    def num_placeholder_tokens(self, num_frames: int, height: int, width: int) -> int:
        """Placeholder tokens of a resized item with ``num_frames`` frames of
        ``height x width`` pixels (``num_frames = 1`` for an image)."""
        config = self.config
        num_tokens, _, _ = calculate_vision_tokens(
            num_frames=num_frames,
            height=height,
            width=width,
            patch_size=config.patch_size,
            spatial_merge_size=config.spatial_merge_size,
            temporal_patch_size=config.temporal_patch_size,
        )
        return num_tokens

    def resized_size(self, height: int, width: int) -> tuple[int, int]:
        """``(H, W)`` that ``decode_and_resize`` gives a ``height x width`` image,
        padding included, without decoding it.

        For callers that know an image's size but not its pixels, e.g. a renderer
        that reads the size from the file header to count placeholder tokens with
        ``num_placeholder_tokens(1, *processor.resized_size(height, width))``.
        """
        resize_h, resize_w, pad_h, pad_w = self._resize_rule(height, width)
        return resize_h + pad_h, resize_w + pad_w

    def _resize_rule(self, height: int, width: int) -> tuple[int, int, int, int]:
        """``config.resize_fn``'s ``(resize_h, resize_w, pad_h, pad_w)`` for a
        ``height x width`` image."""
        config = self.config
        return config.resize_fn(
            height,
            width,
            patch_size=config.patch_size,
            merge_size=config.spatial_merge_size,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            max_patches=config.max_patches,
            max_patches_per_side=config.max_patches_per_side,
        )

    def _resize(self, image: torch.Tensor) -> torch.Tensor:
        """Resize uint8 ``(..., C, H, W)`` pixels to ``config.resize_fn``'s
        ``(resize_h, resize_w, pad_h, pad_w)``, then zero-pad right/bottom.

        Resizes on uint8, which uses torchvision's AVX2/NEON SIMD paths. After
        normalization, padded pixels are constant patches the encoder sees as
        part of the grid.
        """
        config = self.config
        height, width = image.shape[-2:]
        resize_h, resize_w, pad_h, pad_w = self._resize_rule(height, width)
        if (resize_h, resize_w) != (height, width):
            image = TVF.resize(
                image,
                [resize_h, resize_w],
                interpolation=config.image_interpolation_mode,
                antialias=True,
            )
        if pad_h or pad_w:
            image = TVF.pad(image, [0, 0, pad_w, pad_h])
        return image
