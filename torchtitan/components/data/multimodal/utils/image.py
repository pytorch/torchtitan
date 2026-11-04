# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Image preprocessing and vision utilities for multimodal datasets.

Handles image decoding, resizing, normalization, and patch extraction for the
vision encoder.

RL image prompts carry each image's file bytes (JPEG, PNG, ...), not pixels, and use
the steps separately, all from the model's `ImageSpec`:

- `lookup_image` (the renderer): reads the file, hashes it into the content key, and
  computes the patch grid from the image header, without decoding the pixels.
- `decode_image_files` (the generator) and `decode_image_files_flat` (the trainer):
  decode the image files and resize the images to uint8.
- `patchify` / `patchify_flat`: normalize and cut into patches on the images' device;
  the generator and the trainer run them on the GPU (~20 ms per image on one CPU
  thread), bitwise equal to the CPU.

``patchify(decode_image_files(files, spec), spec)`` returns exactly the patches
``process_image`` + ``vision_to_patches`` produce for ``spec``. A file is usually much
smaller than its resized pixels (Geometry3K median: 14 KB vs 0.31 MB of uint8 pixels; a
large photo the resize scales down gains less), and the processes that decode the image
file need no access to the file system the dataset is on.
"""

import functools
import hashlib
import io
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import einops as E
import requests
import torch

import torchvision.io

import torchvision.transforms.v2.functional as TVF

from PIL import Image


logger = logging.getLogger(__name__)


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


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Compute target (height, width) that satisfy per-frame pixel budget.

    Both output dimensions are rounded to multiples of ``factor``.  The spatial
    pixel count ``h * w`` is kept within [min_pixels, max_pixels].

    Args:
        height: Original height.
        width: Original width.
        factor: Spatial rounding factor (``patch_size * merge_size``).
        min_pixels: Minimum spatial pixels per frame.
        max_pixels: Maximum spatial pixels per frame.

    Returns:
        (resized_height, resized_width)
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"Absolute aspect ratio must be smaller than 200, "
            f"got {max(height, width) / min(height, width):.1f}"
        )

    # Round spatial dims to nearest multiple of factor
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

    return h_bar, w_bar


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

    A resize strategy (``resize_fn``) for ``process_image`` -- extra budget
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


def decode_and_resize_image(
    image: str | bytes | Image.Image,
    patch_size: int = 16,
    merge_size: int = 2,
    max_pixels: int = 16777216,
    min_pixels: int = 65536,
    resize_fn: Callable[..., tuple[int, int, int, int]] = resize_to_pixel_budget,
    image_interpolation_mode: TVF.InterpolationMode = TVF.InterpolationMode.BICUBIC,
    max_patches: int = 4096,
    max_patches_per_side: int = 512,
) -> torch.Tensor:
    """Decode an image and resize (and pad) it to the vision encoder's patch grid.

    The uint8 half of ``process_image`` (see there for the arguments); the image
    cache stores its output. Raises on failure.

    Returns:
        Tensor of shape (C, H, W), uint8.
    """
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
        # Resize on uint8 (leverages AVX2/NEON SIMD fast paths)
        img_tensor = TVF.resize(
            img_tensor,
            [resize_h, resize_w],
            interpolation=image_interpolation_mode,
            antialias=True,
        )
    if pad_h or pad_w:
        # Pad right/bottom with zeros (black); after normalization these are
        # constant padding patches the encoder sees as part of the grid.
        img_tensor = TVF.pad(img_tensor, [0, 0, pad_w, pad_h])
    return img_tensor


def normalize_image(
    image: torch.Tensor,
    image_mean: tuple[float, ...] = (0.5, 0.5, 0.5),
    image_std: tuple[float, ...] = (0.5, 0.5, 0.5),
) -> torch.Tensor:
    """Normalize a (C, H, W) uint8 image to a (1, H, W, C) float32 tensor.

    The float half of ``process_image``; elementwise, so it gives the same values
    on any device.
    """
    # uint8 → float32 [0, 1] → normalize
    image = TVF.to_dtype(image, torch.float32, scale=True)
    image = TVF.normalize(image, list(image_mean), list(image_std), inplace=True)
    # (C, H, W) → (1, H, W, C)
    return image.permute(1, 2, 0).unsqueeze(0)


def process_image(
    image: str | bytes | Image.Image,
    patch_size: int = 16,
    merge_size: int = 2,
    max_pixels: int = 16777216,
    min_pixels: int = 65536,
    image_mean: tuple[float, ...] = (0.5, 0.5, 0.5),
    image_std: tuple[float, ...] = (0.5, 0.5, 0.5),
    resize_fn: Callable[..., tuple[int, int, int, int]] = resize_to_pixel_budget,
    image_interpolation_mode: TVF.InterpolationMode = TVF.InterpolationMode.BICUBIC,
    max_patches: int = 4096,
    max_patches_per_side: int = 512,
) -> torch.Tensor | None:
    """Load and preprocess a single image for VLM training.

    Uses torchvision APIs for decoding and resizing (faster uint8 SIMD paths),
    then normalizes with the given mean/std: ``decode_and_resize_image``, then
    ``normalize_image``. ``resize_fn`` is the resize strategy
    -- ``resize_to_pixel_budget`` (Qwen-VL, default) or
    ``resize_to_navit_patch_grid`` (Kimi-VL) -- with the uniform signature
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
        image_interpolation_mode: Torchvision interpolation mode used to resize
            the image tensor.
        max_patches: Pre-padding patch budget (``resize_to_navit_patch_grid``).
        max_patches_per_side: Pre-padding per-side patch limit
            (``resize_to_navit_patch_grid``).

    Returns:
        Tensor of shape (1, H, W, C) with a dummy temporal dim, or None on failure.
    """
    try:
        img_tensor = decode_and_resize_image(
            image,
            patch_size=patch_size,
            merge_size=merge_size,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            resize_fn=resize_fn,
            image_interpolation_mode=image_interpolation_mode,
            max_patches=max_patches,
            max_patches_per_side=max_patches_per_side,
        )
        return normalize_image(img_tensor, image_mean, image_std)

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


@dataclass(frozen=True, kw_only=True, slots=True)
class ImageSpec:
    """The model's image preprocessing; defaults match Qwen3-VL / Qwen3.5.

    The arguments of ``process_image`` + ``vision_to_patches`` for the model's
    vision encoder. It must match the model's vision encoder and the SFT data
    pipeline of the same model, so the generator and the trainer see the pixels
    the model was trained on.
    """

    patch_size: int = 16
    temporal_patch_size: int = 2
    merge_size: int = 2
    """Patches merged per side into one placeholder token (``spatial_merge_size``)."""
    min_pixels: int = 65536
    """Lower bound on H * W after ``resize_to_pixel_budget``."""
    max_pixels: int = 16777216
    """Upper bound on H * W after ``resize_to_pixel_budget``."""
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5)


@dataclass(frozen=True, kw_only=True, slots=True)
class ImageInfo:
    """One image of a prompt: content key, file, raw patch grid and file bytes."""

    key: str
    """Hash of the file bytes."""
    path: str
    """The file `lookup_image` read."""
    grid_thw: tuple[int, int, int]
    """Raw patch grid ``(T, H, W)`` before spatial merging."""
    data: bytes = field(repr=False, compare=False)
    """The file's bytes, which the generator and the trainer decode into the image. Not
    compared: ``key`` is their hash."""


def lookup_image(path: str, spec: ImageSpec) -> ImageInfo:
    """Return ``path``'s content key, its patch grid under ``spec`` and its bytes, without
    decoding the pixels. The file must be readable where the prompt is rendered.

    Example:

        image = lookup_image("./assets/datasets/geometry3k/images/train/00000_0.png", spec)
        num_image_tokens(image.grid_thw, spec)  # placeholder tokens after spatial merging
        pixel_values = patchify(decode_image_files([image.data], spec), spec)
    """
    with open(path, "rb") as f:
        data = f.read()
    # The header gives the stored size; decode_image applies no EXIF rotation either.
    width, height = Image.open(io.BytesIO(data)).size
    return ImageInfo(
        key=_content_key(data),
        path=path,
        grid_thw=_grid_thw(height, width, spec),
        data=data,
    )


def num_image_tokens(grid_thw: tuple[int, int, int], spec: ImageSpec) -> int:
    """Placeholder tokens of a ``grid_thw`` image after ``merge_size x merge_size`` merging."""
    t, h, w = grid_thw
    return t * h * w // spec.merge_size**2


def largest_image_grid(spec: ImageSpec, max_tokens: int) -> tuple[int, int, int]:
    """The largest square grid within ``spec``'s resize budget and ``max_tokens``
    placeholder tokens; vLLM profiles the vision encoder with it. The encoder's
    cost depends on the token count, not on the shape."""
    budget = spec.max_pixels // (spec.patch_size * spec.merge_size) ** 2
    side = math.isqrt(min(max_tokens, budget))
    if side == 0:
        raise ValueError(
            f"No image fits {max_tokens} tokens with patch size {spec.patch_size} "
            f"and merge size {spec.merge_size}."
        )
    return (1, side * spec.merge_size, side * spec.merge_size)


def decode_image_files(files: Sequence[bytes], spec: ImageSpec) -> list[torch.Tensor]:
    """Return the resized uint8 images ``(C, H, W)`` of image ``files``, unnormalized."""
    return [_decode_and_resize_image_file(data, spec) for data in files]


@functools.lru_cache(maxsize=64)
def read_image_file(path: str, key: str) -> bytes:
    """Return the bytes of the image file at ``path``, which must hash to ``key``.

    The renderer read them in `lookup_image`, but the renderers library keeps only
    ``pixel_values`` and the grid of what `_process_image` returns, so `TokenEnv` reads
    the file again for the prompt. Cached: the samples of a GRPO group render together in one
    process, so they share one bytes object, and a message that holds several of them
    pickles the bytes once.
    """
    with open(path, "rb") as f:
        data = f.read()
    if _content_key(data) != key:
        raise ValueError(
            f"{path!r} changed after it was rendered: its content key is "
            f"{_content_key(data)}, the prompt has {key}."
        )
    return data


def decode_image_files_flat(
    image_files: torch.Tensor,
    file_sizes: list[int],
    grid_thw: list[list[int]],
    spec: ImageSpec,
) -> torch.Tensor:
    """Decode and resize concatenated image files into `patchify_flat`'s input.

    Raises unless each image file decodes to the grid its prompt has: placeholders and
    M-RoPE positions follow the prompt's grid, the patches the decoded image.

    Args:
        image_files: uint8 ``(sum(file_sizes),)``, the files' bytes concatenated.
        file_sizes: Bytes per file.
        grid_thw: Each file's raw patch grid ``(T, H, W)``, from its prompt.
        spec: The model's image preprocessing.
    """
    images = []
    for index, (data, (t, h, w)) in enumerate(
        zip(image_files.split(file_sizes), grid_thw, strict=True)
    ):
        image = _decode_and_resize_image_file(data.numpy().tobytes(), spec)
        patch = spec.patch_size
        decoded_image_grid = (1, image.shape[1] // patch, image.shape[2] // patch)
        if decoded_image_grid != (t, h, w):
            raise ValueError(
                f"Image file {index} decodes to grid {decoded_image_grid}, but its prompt has "
                f"{(t, h, w)}; the renderer and the batcher must use the same "
                "ImageSpec."
            )
        images.append(image.flatten())
    return torch.cat(images)


def patchify(images: Sequence[torch.Tensor], spec: ImageSpec) -> torch.Tensor:
    """Normalize resized uint8 images and cut them into concatenated patches.

    These are ``normalize_image`` and ``vision_to_patches``. They run on the images'
    device and are elementwise float32 ops plus reshapes, so the CPU and the GPU
    give bitwise-equal patches.
    """
    patches = []
    for image in images:
        image_patches, _ = vision_to_patches(
            normalize_image(image, spec.image_mean, spec.image_std),
            spec.patch_size,
            spec.temporal_patch_size,
            spec.merge_size,
        )
        patches.append(image_patches)
    return torch.cat(patches)


def patchify_flat(
    flat_images: torch.Tensor, grid_thw: list[list[int]], spec: ImageSpec
) -> torch.Tensor:
    """`patchify` for `decode_image_files` outputs flattened and concatenated in
    ``grid_thw`` order.

    Decoded images are ``(3, h * patch_size, w * patch_size)`` with ``t == 1`` (see
    `_decode_and_resize_image_file`).
    """
    assert all(t == 1 for t, _, _ in grid_thw), grid_thw
    shapes = [(3, h * spec.patch_size, w * spec.patch_size) for _, h, w in grid_thw]
    flat = flat_images.split([c * h * w for c, h, w in shapes])
    return patchify(
        [image.view(shape) for image, shape in zip(flat, shapes, strict=True)], spec
    )


def _content_key(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:32]


def _decode_and_resize_image_file(data: bytes, spec: ImageSpec) -> torch.Tensor:
    return decode_and_resize_image(
        data,
        patch_size=spec.patch_size,
        merge_size=spec.merge_size,
        min_pixels=spec.min_pixels,
        max_pixels=spec.max_pixels,
    )


def _grid_thw(height: int, width: int, spec: ImageSpec) -> tuple[int, int, int]:
    """Patch grid of a ``height x width`` image after `_decode_and_resize_image_file`."""
    resize_h, resize_w, pad_h, pad_w = resize_to_pixel_budget(
        height,
        width,
        patch_size=spec.patch_size,
        merge_size=spec.merge_size,
        min_pixels=spec.min_pixels,
        max_pixels=spec.max_pixels,
    )
    return (
        1,
        (resize_h + pad_h) // spec.patch_size,
        (resize_w + pad_w) // spec.patch_size,
    )
