# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stage-boundary planning for the ViT/text pipeline hop (DEP, report 5.2.3).

Vision features cross a pipeline hop, and PP's point-to-point buffers are
sized ONCE: the exchange shape must be a CONFIG-level upper bound, never
derived from the current batch -- that fails inside the P2P on the first
larger batch. grid_thw is replicated, so every stage computes the same
layout with no communication (the property dynamic CP relies on).
"""

from __future__ import annotations

import torch


def stage_exchange_lengths(
    grids: list[tuple[int, int, int]], *, kh: int, kw: int
) -> list[int]:
    """Per-image projected token counts for this batch.

    ``merged_tokens`` per image -- no ``t``, because the projector's temporal mean
    collapses it. These are the lengths the receiving stage uses to unpack, and it
    can compute them itself from the replicated ``grid_thw``.
    """
    return [merged_tokens(h, w, kh, kw) for _, h, w in grids]


def stage_exchange_capacity(
    max_grid_h: int, max_grid_w: int, max_images: int, *, kh: int, kw: int
) -> int:
    """The FIXED row count the ViT stage always sends.

    Derived from configured maxima, not from a batch. Returns the padded token
    capacity; a batch using less pads and the receiver slices by the real lengths.
    """
    if max_images < 0 or max_grid_h < 0 or max_grid_w < 0:
        raise ValueError("capacity inputs must be non-negative")
    return max_images * merged_tokens(max_grid_h, max_grid_w, kh, kw)


def pack_stage_features(feats, capacity: int):
    """Concatenate per-image features and pad to ``capacity`` rows.

    Raises when the batch does not fit rather than truncating: a truncated vision
    feature is a silently wrong model, and the receiving stage cannot tell.
    """
    import torch

    if not feats:
        raise ValueError("no features to pack; the ViT stage has nothing to send")
    flat = torch.cat(list(feats), dim=0)
    used = flat.size(0)
    if used > capacity:
        raise ValueError(
            f"vision features need {used} rows but the stage exchange capacity is "
            f"{capacity}; raise the configured maxima rather than truncating -- a "
            "truncated feature is a silently wrong model"
        )
    if used == capacity:
        return flat
    pad = flat.new_zeros(capacity - used, flat.size(1))
    return torch.cat([flat, pad], dim=0)


def unpack_stage_features(packed, lengths: list[int]):
    """Split a padded exchange buffer back into per-image features."""
    total = sum(lengths)
    if total > packed.size(0):
        raise ValueError(
            f"unpack needs {total} rows but the buffer holds {packed.size(0)}; the "
            "sender and receiver disagree on the layout, which they compute "
            "independently from the replicated grid_thw"
        )
    out, off = [], 0
    for n in lengths:
        out.append(packed[off : off + n])
        off += n
    return out


def stage_patch_capacity(max_grid_h: int, max_grid_w: int, max_images: int) -> int:
    """The FIXED row count a MID-tower stage boundary always carries.

    Distinct from :func:`stage_exchange_capacity`, which sizes the buffer that leaves
    the tower: that one counts PROJECTED tokens (``merged_tokens``, time collapsed),
    while a boundary INSIDE the tower carries un-merged patch hidden states, so the
    count is ``t * h * w`` summed over images. Using the projected capacity for an
    inner boundary would under-size it by ``kh * kw``.

    ``t`` has no configured maximum, so a video whose frames push the real patch count
    past this capacity must raise at the sender rather than truncate -- the padding
    helpers already do. Treat ``max_images`` as a budget over FRAMES for that reason.
    """
    if max_images < 0 or max_grid_h < 0 or max_grid_w < 0:
        raise ValueError("capacity inputs must be non-negative")
    return max_images * max_grid_h * max_grid_w


def pack_stage_patches(x_LD, capacity: int):
    """Pad patch hidden states to ``capacity`` rows for a fixed-shape pipe payload.

    PP sizes its point-to-point buffers once, not per step, so a boundary inside the
    tower cannot carry a batch-dependent row count. Returns the padded tensor; the
    receiver slices back to the real length, which it computes from the replicated
    ``grid_thw`` rather than being told.
    """
    import torch

    used = x_LD.size(0)
    if used > capacity:
        raise ValueError(
            f"patch hidden states need {used} rows but the mid-tower stage capacity "
            f"is {capacity}; raise dep_max_images / dep_max_grid_h / dep_max_grid_w "
            "rather than truncating -- a truncated activation is a silently wrong "
            "model the receiving stage cannot detect"
        )
    if used == capacity:
        return x_LD
    pad = x_LD.new_zeros((capacity - used,) + tuple(x_LD.shape[1:]))
    return torch.cat([x_LD, pad], dim=0)


def unpack_stage_patches(padded, num_rows: int):
    """Slice a fixed-capacity mid-tower payload back to its real row count."""
    if num_rows > padded.size(0):
        raise ValueError(
            f"unpack needs {num_rows} rows but the buffer holds {padded.size(0)}; "
            "sender and receiver disagree on the layout, which they compute "
            "independently from the replicated grid_thw"
        )
    return padded[:num_rows]
