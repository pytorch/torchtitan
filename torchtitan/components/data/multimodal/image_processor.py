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
``QWEN3_5_IMAGE_PROCESSOR_KWARGS``), and recipes build the config from them:
``ImageProcessor.Config(**QWEN3_5_IMAGE_PROCESSOR_KWARGS)``. The trainer builds
one from ``Trainer.Config.image_processor`` and hands it to the dataloader, as
it does the tokenizer. ``VideoProcessor`` extends it for video frames.

An image becomes the encoder's input in two steps, on the CPU:

- ``resize_and_normalize(image)``: decode -> resize to the model's resolution
  -> normalize, returning float32 ``(1, H, W, C)`` pixels and the ``(3,)``
  patch grid ``patchify`` will cut them into;
  ``num_placeholder_tokens(grid_thw)`` gives the image's placeholder count in
  the prompt.
- ``patchify(pixels)``: cut the pixels into float32 ``(num_patches,
  patch_dim)`` patches and return them with the grid.

The multimodal dataloader calls ``resize_and_normalize`` in its sample
processor and ``patchify`` in its collator, which concatenates the
microbatch's patches. The resize is a config field (``resize_fn``, e.g.
``resize_to_pixel_budget`` or ``resize_to_navit_patch_grid`` below); a model
whose preprocessing does not fit (e.g. tiling or cropping) overrides
``resize_and_normalize``.

torchvision is an optional dependency (``requirements-vlm.txt``), so it is
imported only where images are decoded and resized: model flavors name the
resize rules, and the trainer types its processors, without it.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import einops as E
import requests
import torch
from PIL import Image

from torchtitan.config import Configurable


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
        """Per-channel mean subtracted from the pixels after scaling them to
        ``[0, 1]``, e.g. ``(0.5, 0.5, 0.5)``."""

        image_std: tuple[float, ...]
        """Per-channel standard deviation the pixels are divided by after
        subtracting ``image_mean``."""

        resize_fn: Callable[..., tuple[int, int, int, int]]
        """Resize rule ``(height, width, *, patch_size, merge_size, **budget)
        -> (resize_h, resize_w, pad_h, pad_w)``, e.g. ``resize_to_pixel_budget``
        or ``resize_to_navit_patch_grid``. Each rule reads the budget fields it
        needs."""

        patch_order: Literal["block", "raster"]
        """Patch sequence layout the encoder expects; see ``patchify``."""

        min_pixels: int = 65_536
        """Lower pixel budget of ``resize_to_pixel_budget``."""

        max_pixels: int = 16_777_216
        """Upper pixel budget of ``resize_to_pixel_budget``."""

        max_patches: int = 4096
        """Patch budget of ``resize_to_navit_patch_grid``."""

        max_patches_per_side: int = 512
        """Per-side patch limit of ``resize_to_navit_patch_grid``."""

        image_interpolation_mode: Literal[
            "nearest", "nearest-exact", "bilinear", "bicubic", "lanczos"
        ] = "bicubic"
        """Interpolation of the resize, a torchvision ``InterpolationMode``
        value."""

        def __post_init__(self) -> None:
            if self.patch_order not in ("block", "raster"):
                raise ValueError(
                    "patch_order must be 'block' or 'raster', got "
                    f"{self.patch_order!r}."
                )

    def __init__(self, config: Config) -> None:
        self.config = config

    def resize_and_normalize(
        self, image: str | bytes | Image.Image
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode, resize and normalize one image.

        Accepts encoded bytes, a file path, an HTTP(S) URL, or a PIL image.

        Returns:
            ``pixels``: float32 ``(1, H, W, C)``, resized and normalized;
            ``patchify`` cuts them into the encoder's patches.
            ``grid_thw``: the ``(3,)`` patch grid ``[T, H, W]`` ``patchify``
            gives them, before merging.
        """
        image_CHW = self._resize(self._decode_image(image))
        return self._normalize(image_CHW.unsqueeze(0))

    def num_placeholder_tokens(self, grid_thw: Sequence[int] | torch.Tensor) -> int:
        """Placeholder tokens of an item with patch grid ``grid_thw``
        ``[T, H, W]``: one per ``spatial_merge_size x spatial_merge_size``
        patches, e.g. 300 for grid ``(1, 30, 40)`` with a 2x2 merge."""
        t, h, w = (int(size) for size in grid_thw)
        return t * h * w // self.config.spatial_merge_size**2

    @staticmethod
    def _decode_image(image: str | bytes | Image.Image) -> torch.Tensor:
        """Decode one image to a uint8 ``(C, H, W)`` RGB tensor.

        Uses ``torchvision.io.decode_image`` for bytes and paths (faster SIMD
        decode), ``pil_to_tensor`` for PIL images.
        """
        import torchvision.io
        import torchvision.transforms.v2.functional as TVF

        if isinstance(image, str) and image.startswith("http"):
            response = requests.get(image, timeout=10)
            image = response.content
        if isinstance(image, bytes):
            raw = torch.frombuffer(bytearray(image), dtype=torch.uint8)
            return torchvision.io.decode_image(
                raw, mode=torchvision.io.ImageReadMode.RGB
            )
        if isinstance(image, str):
            return torchvision.io.decode_image(
                image, mode=torchvision.io.ImageReadMode.RGB
            )
        if image.mode != "RGB":
            image = image.convert("RGB")
        return TVF.pil_to_tensor(image)

    def _normalize(
        self, frames_TCHW: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize resized uint8 frames: float32 ``(T, H, W, C)`` pixels and
        the ``(3,)`` patch grid ``patchify`` gives them."""
        import torchvision.transforms.v2.functional as TVF

        config = self.config
        pixels_TCHW = TVF.to_dtype(frames_TCHW, torch.float32, scale=True)
        pixels_TCHW = TVF.normalize(
            pixels_TCHW,
            list(config.image_mean),
            list(config.image_std),
            inplace=True,
        )
        num_frames, _, height, width = pixels_TCHW.shape
        grid_thw = torch.tensor(
            [
                math.ceil(num_frames / config.temporal_patch_size),
                height // config.patch_size,
                width // config.patch_size,
            ]
        )
        return pixels_TCHW.permute(0, 2, 3, 1), grid_thw

    def patchify(self, frames_THWC: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cut ``resize_and_normalize``'s (or ``resize_and_normalize_video``'s) pixels into
        flattened patches with ``config.patch_size``, ``temporal_patch_size``
        and ``spatial_merge_size``.

        ``config.patch_order`` selects the sequence layout:

        - ``"block"``: each ``merge_size x merge_size`` spatial group is
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
            frames_THWC: Image or video frames.

        Returns:
            patches: (num_patches, patch_dim) flattened patch vectors in
                channel-first ``(C, pt, ph, pw)`` layout matching Conv3d kernels.
            grid_thw: (3,) tensor ``[T_patches, H_patches, W_patches]`` counting
                raw patches (before spatial merging).
        """
        config = self.config
        img = frames_THWC
        T, H, W, C = img.shape
        ps = config.patch_size
        ts = config.temporal_patch_size
        merge_size = config.spatial_merge_size

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
        if config.patch_order == "block":
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
        else:
            # (t, h, w) -- plain row-major (merge_size unused in the layout).
            patches = E.rearrange(
                img,
                "(t pt) (h ph) (w pw) c -> (t h w) (c pt ph pw)",
                pt=ts,
                ph=ps,
                pw=ps,
            )

        grid_thw = torch.tensor([T_patches, H_patches, W_patches])
        return patches, grid_thw

    def _resize(self, image: torch.Tensor) -> torch.Tensor:
        """Resize uint8 ``(..., C, H, W)`` pixels to ``config.resize_fn``'s
        ``(resize_h, resize_w, pad_h, pad_w)``, then zero-pad right/bottom.

        Resizes on uint8, which uses torchvision's AVX2/NEON SIMD paths. After
        normalization, padded pixels are constant patches the encoder sees as
        part of the grid.
        """
        import torchvision.transforms.v2.functional as TVF

        config = self.config
        height, width = image.shape[-2:]
        resize_h, resize_w, pad_h, pad_w = config.resize_fn(
            height,
            width,
            patch_size=config.patch_size,
            merge_size=config.spatial_merge_size,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            max_patches=config.max_patches,
            max_patches_per_side=config.max_patches_per_side,
        )
        if (resize_h, resize_w) != (height, width):
            image = TVF.resize(
                image,
                [resize_h, resize_w],
                interpolation=TVF.InterpolationMode(config.image_interpolation_mode),
                antialias=True,
            )
        if pad_h or pad_w:
            image = TVF.pad(image, [0, 0, pad_w, pad_h])
        return image
