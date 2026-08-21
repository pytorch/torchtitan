# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""K3's NaViT image preprocessing, ported from the released processor.

Transcribed from ``media_utils.navit_resize_image`` and
``preprocessor_config.json`` in the HF model repo, so the token counts a training
run produces match what the official processor would produce for the same image:

    patch_size 14        merge_kernel_size 2
    image_mean/std 0.5   -> normalize to [-1, 1]
    in_patch_limit 65536             total patches per image
    patch_limit_on_one_side 512      patches along either side
    in_patch_limit_each_frame 16384  per video frame
    temporal_merge_kernel_size 4     frames grouped into one sample

Two details that are easy to get wrong and change the token count:

* the aspect-preserving downscale uses ``width // patch_size`` (integer
  division) inside the area limit, not ``width / patch_size``;
* dimensions are reached by ZERO-PADDING after the resize, not by resizing to a
  multiple. Padding to ``merge_kernel_size * patch_size = 28`` is what
  guarantees the patch grid tiles the 2x2 merge -- the condition
  ``tpool_patch_merger`` raises on.

``temporal_merge_kernel_size 4`` and ``init_pos_emb_time 4`` are the same number
for a reason: the processor groups up to 4 frames into one sample with ``t <= 4``,
and the merger then means over exactly those frames. A video longer than 4 frames
becomes several samples, which is why the position table never needs
interpolation along time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

PATCH_SIZE = 14
MERGE_KERNEL_SIZE = 2
IMAGE_MEAN = 0.5
IMAGE_STD = 0.5
IN_PATCH_LIMIT = 65536
PATCH_LIMIT_ON_ONE_SIDE = 512
IN_PATCH_LIMIT_EACH_FRAME = 16384
TEMPORAL_MERGE_KERNEL_SIZE = 4


@dataclass(frozen=True)
class ResizePlan:
    """What the official resize decides for one image."""

    new_width: int
    new_height: int
    pad_width: int
    pad_height: int
    num_tokens: int
    # The patch size this plan was computed with. Carried rather than assumed: the
    # property below used to divide by the module constant, so a plan built with a
    # non-default patch_size reported a grid that did not match its own dimensions and
    # prepare_image died in the view. A default keeps every existing constructor call
    # working.
    patch_size: int = PATCH_SIZE

    @property
    def padded_size(self) -> tuple[int, int]:
        return self.new_height + self.pad_height, self.new_width + self.pad_width

    @property
    def patch_grid(self) -> tuple[int, int]:
        h, w = self.padded_size
        return h // self.patch_size, w // self.patch_size


def navit_resize(
    width: int,
    height: int,
    *,
    patch_size: int = PATCH_SIZE,
    merge_kernel_size: int = MERGE_KERNEL_SIZE,
    in_patch_limit: int = IN_PATCH_LIMIT,
    patch_limit_on_one_side: int = PATCH_LIMIT_ON_ONE_SIDE,
    fixed_output_tokens: int | None = None,
) -> ResizePlan:
    """Aspect-preserving downscale plus zero-pad, as the release does it."""
    s1 = math.sqrt(
        in_patch_limit
        / (max(1.0, width // patch_size) * max(1.0, height // patch_size))
    )
    s2 = patch_limit_on_one_side * patch_size / width
    s3 = patch_limit_on_one_side * patch_size / height
    scale = min(1.0, s1, s2, s3)
    new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
    new_w = min(new_w, patch_limit_on_one_side * patch_size)
    new_h = min(new_h, patch_limit_on_one_side * patch_size)

    factor = merge_kernel_size * patch_size
    pad_h = (factor - new_h % factor) % factor
    pad_w = (factor - new_w % factor) % factor

    token_h = (new_h + pad_h) // factor
    token_w = (new_w + pad_w) // factor
    if token_h * merge_kernel_size > patch_limit_on_one_side:
        raise ValueError(
            f"token_height {token_h} * {merge_kernel_size} exceeds "
            f"patch_limit_on_one_side {patch_limit_on_one_side}"
        )
    if token_w * merge_kernel_size > patch_limit_on_one_side:
        raise ValueError(
            f"token_width {token_w} * {merge_kernel_size} exceeds "
            f"patch_limit_on_one_side {patch_limit_on_one_side}"
        )
    num_tokens = (
        fixed_output_tokens if fixed_output_tokens is not None else token_h * token_w
    )
    return ResizePlan(new_w, new_h, pad_w, pad_h, num_tokens, patch_size)


def normalize_pixels(pixels: torch.Tensor) -> torch.Tensor:
    """``[..., C, H, W]`` in [0, 1] -> normalized with mean/std 0.5, i.e. [-1, 1]."""
    return (pixels - IMAGE_MEAN) / IMAGE_STD


def prepare_image(
    pixels_CHW: torch.Tensor,
    *,
    patch_size: int = PATCH_SIZE,
    merge_kernel_size: int = MERGE_KERNEL_SIZE,
    already_normalized: bool = False,
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """One image -> ``([N, C, p, p]`` patches, ``(1, h, w))``.

    Resizes per :func:`navit_resize`, zero-pads, normalizes, then cuts patches in
    row-major order -- the order ``MoonViTPatchEmbed`` and the position tables
    assume.
    """
    if pixels_CHW.dim() != 3:
        raise ValueError(f"expected [C, H, W], got {tuple(pixels_CHW.shape)}")
    C, H, W = pixels_CHW.shape
    plan = navit_resize(
        W, H, patch_size=patch_size, merge_kernel_size=merge_kernel_size
    )
    x = pixels_CHW.unsqueeze(0)
    if (plan.new_height, plan.new_width) != (H, W):
        x = torch.nn.functional.interpolate(
            x.float(),
            size=(plan.new_height, plan.new_width),
            mode="bicubic",
            align_corners=False,
            # antialias=True to match the reference resampler. PIL/torchvision
            # antialias on DOWNSCALE; interpolate defaults to False, which skips the
            # prefilter and aliases high-frequency detail. Every downscaled image then
            # differs systematically from what the released preprocessing produces --
            # no error, just a parity gap that surfaces as worse finetune numbers.
            antialias=True,
        ).clamp(0.0, 1.0)
    if plan.pad_height or plan.pad_width:
        # Pad bottom/right with zeros, matching the release. Note this happens
        # BEFORE normalization, so padded pixels become -1 rather than 0 -- the
        # same as the official order (np.pad then normalize).
        x = torch.nn.functional.pad(x, (0, plan.pad_width, 0, plan.pad_height))
    if not already_normalized:
        x = normalize_pixels(x)

    h, w = plan.patch_grid
    x = x.view(1, C, h, patch_size, w, patch_size)
    patches = x.permute(0, 2, 4, 1, 3, 5).reshape(h * w, C, patch_size, patch_size)
    return patches.to(pixels_CHW.dtype), (1, h, w)


def pack_images(images: list[torch.Tensor], **kw) -> tuple[torch.Tensor, torch.Tensor]:
    """Several images of DIFFERENT sizes -> one packed batch + ``grid_thws``.

    This is the shape MoonViT's forward takes, and the reason it takes that
    shape: native-resolution training mixes resolutions inside one batch, so
    there is no rectangular tensor to hand it.
    """
    if not images:
        raise ValueError("pack_images needs at least one image")
    patch_list, grids = [], []
    for img in images:
        patches, grid = prepare_image(img, **kw)
        patch_list.append(patches)
        grids.append(grid)
    return torch.cat(patch_list, dim=0), torch.tensor(
        grids, dtype=torch.long, device=images[0].device
    )


def pack_video(
    frames_FCHW: torch.Tensor,
    *,
    temporal_merge_kernel_size: int = TEMPORAL_MERGE_KERNEL_SIZE,
    **kw,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A video -> packed patches + ``grid_thws``, grouped by the temporal kernel.

    Frames are grouped into samples of at most ``temporal_merge_kernel_size``,
    which is why ``t`` never exceeds ``init_pos_emb_time``. Each group records a
    single ``(t, h, w)``, which is only valid if every frame in it resizes to the
    same grid -- and it does, structurally: the parameter is one stacked
    ``[F, C, H, W]`` tensor, so all frames share H and W, and
    :func:`prepare_image` derives the grid from ``navit_resize(W, H, ...)``, a
    pure function of those. Ragged input goes to :func:`pack_images`, which
    records a grid per image. Nothing to enforce here; the type does it.
    """
    if frames_FCHW.dim() != 4:
        raise ValueError(f"expected [F, C, H, W], got {tuple(frames_FCHW.shape)}")
    total = frames_FCHW.shape[0]
    patch_list, grids = [], []
    for start in range(0, total, temporal_merge_kernel_size):
        group = frames_FCHW[start : start + temporal_merge_kernel_size]
        per_frame = [prepare_image(f, **kw) for f in group]
        _, (_, h, w) = per_frame[0]
        patch_list.extend(p for p, _ in per_frame)
        grids.append((len(group), h, w))
    return torch.cat(patch_list, dim=0), torch.tensor(
        grids, dtype=torch.long, device=frames_FCHW.device
    )


def from_titan_collator(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    patch_size: int = PATCH_SIZE,
    merge_kernel_size: int = MERGE_KERNEL_SIZE,
    channels: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapt torchtitan's multimodal collator output to MoonViT's input.

    Core's ``MultiModalCollator`` yields ``pixel_values`` as
    ``[num_images, max_num_patch, patch_dim]`` -- PADDED to the batch's longest
    image -- plus ``grid_thw`` as ``[num_images, 3]``. MoonViT takes a PACKED
    ``[L, C, p, p]`` stream with no padding. Two conversions are needed, and the
    second one is a correctness trap rather than a reshape:

    1. Drop the padding. ``grid_thw`` gives each image's true patch count, so the
       valid prefix of each row is ``t * h * w``.

    2. **Reorder from block order to row-major.** Core's ``vision_to_patches``
       emits patches so that each ``merge_size x merge_size`` spatial group is
       contiguous (the Qwen2-VL convention, matching a merger that consumes
       groups in sequence). MoonViT's position tables and ``tpool_patch_merger``
       both assume ROW-MAJOR order -- the merger reshapes
       ``[t, h/kh, kh, w/kw, kw, D]``, which only groups 2x2 neighbourhoods if
       the patches arrive row by row. Feeding block order straight through
       scrambles which patches get merged and which position each receives, and
       produces a perfectly plausible loss curve while doing it.
    """
    if pixel_values.dim() != 3:
        raise ValueError(
            f"expected [num_images, max_num_patch, patch_dim], got "
            f"{tuple(pixel_values.shape)}"
        )
    kh = kw = merge_kernel_size
    out = []
    for row, (t, h, w) in zip(pixel_values, grid_thw.tolist()):
        n = t * h * w
        if h % kh or w % kw:
            raise ValueError(
                f"patch grid {h}x{w} does not tile the merge kernel {kh}x{kw}"
            )
        flat = row[:n]
        # block order -> row-major: the block layout is
        # [t, h/kh, w/kw, kh, kw]; permute back to [t, h, w].
        blocks = flat.view(t, h // kh, w // kw, kh, kw, -1)
        rowmajor = blocks.permute(0, 1, 3, 2, 4, 5).reshape(n, -1)
        out.append(rowmajor)
    packed = torch.cat(out, dim=0)
    expected = channels * patch_size * patch_size
    if packed.shape[-1] != expected:
        raise ValueError(
            f"patch_dim {packed.shape[-1]} does not match C*p*p = {expected}"
        )
    return (
        packed.view(-1, channels, patch_size, patch_size),
        grid_thw.to(torch.long),
    )
