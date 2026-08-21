# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Planning for dynamic CP in the vision encoder (report 5.2.3).

Pure functions, no collectives and no torch tensors in the signatures, so the
scheduling decisions can be tested without spawning ranks. The distributed half
lives in ``multimodal_model`` and ``moonvit``.

Report 5.2.3 asks for two things:

1. "A single large image is partitioned along the patch dimension across multiple
   devices, and attention is computed by gathering key-value pairs (gather-KV)
   across CP ranks."
2. "we divide each CP group into several sub-CP groups and distribute multiple
   large images across them in a load-balanced manner, preventing the
   communication fraction from growing with scale."

The reason (2) exists is in its own clause: gather-KV over the WHOLE CP group
makes every rank exchange every large image's keys, so the communication fraction
grows with the group. Partitioning one image over a sub-group of 2 while another
image occupies a different sub-group keeps the exchange local and the ranks busy.

**The merge kernel constrains where a partition may cut.** The projector merges
each ``(kh, kw)`` block of patches into one output token, so a cut inside a block
would ask two ranks to merge halves of the same block. The safe unit is a
MERGE-ROW BLOCK -- ``kh`` consecutive grid rows, ``kh * w`` patches. Since patches
are laid out row-major over ``(t, h, w)``, such a block is contiguous in the
packed stream and consecutive blocks abut, including across a video's frame
boundary. Cutting on arbitrary patch counts is merge-unsafe; cutting "rows r0..r1
of every frame" is merge-safe but NOT contiguous once ``t > 1``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageShard:
    """One rank's slice of one partitioned image: a row band, all frames."""

    row_start: int
    row_end: int
    """Half-open band of patch-grid ROWS, a multiple of ``kh``. Empty when the
    image has fewer row blocks than the sub-group has ranks."""

    grid: tuple[int, int, int]
    """The shard's own (t, h, w) -- all frames, this rank's rows."""

    ranges: tuple[tuple[int, int], ...]
    """One flat ``[start, end)`` range per frame. A still gives one range; a video
    gives ``t`` of them, because the band is strided in the packed stream."""


def row_partition(
    t: int, h: int, w: int, *, kh: int, group_size: int
) -> list[ImageShard]:
    """Split one image across ``group_size`` ranks along the SPATIAL rows.

    Every rank keeps every frame and takes a band of rows. Two constraints force
    this shape, and both were learned by measuring:

    * **The merge kernel.** The projector merges each ``(kh, kw)`` block, so a cut
      inside ``kh`` rows would ask two ranks to merge halves of one block. Bands are
      therefore multiples of ``kh``.
    * **The temporal pool.** ``tpool_patch_merger`` is ``sd2_tpool``: it takes
      ``mean(dim=0)`` over ALL frames, collapsing time completely. Splitting a video
      by FRAMES therefore gives each rank the mean of its own frames and the
      concatenation is ``t`` times too many tokens, not the mean -- measured as a
      100% mismatch on t=2. Keeping all frames on every rank makes each rank's
      temporal mean the true one for its rows.

    So the output token count of a partitioned image is ``(h/kh) * (w/kw)``,
    independent of ``t``, and each rank contributes ``(band/kh) * (w/kw)`` of it.

    A band is strided in the packed stream once ``t > 1``, hence ``ranges`` rather
    than one offset pair. An image with fewer row blocks than ranks leaves the tail
    ranks empty; the caller pads for the fixed-shape collective. The ceiling split
    keeps any deficit on the TRAILING ranks, so padding lands at the end of the
    gathered stream rather than inside it.
    """
    if h % kh:
        raise ValueError(
            f"patch grid height {h} must divide the merge kernel height {kh}; "
            "the projector merges (kh, kw) blocks and a partition cannot cut "
            "inside one"
        )
    blocks = h // kh
    per = -(-blocks // group_size)
    frame = h * w
    shards: list[ImageShard] = []
    for r in range(group_size):
        b0 = min(r * per, blocks)
        b1 = min((r + 1) * per, blocks)
        r0, r1 = b0 * kh, b1 * kh
        ranges = tuple((f * frame + r0 * w, f * frame + r1 * w) for f in range(t))
        shards.append(
            ImageShard(
                row_start=r0,
                row_end=r1,
                grid=(t, r1 - r0, w),
                ranges=ranges,
            )
        )
    return shards


def merged_tokens(h: int, w: int, kh: int, kw: int) -> int:
    """Tokens the projector emits for one image -- time is collapsed, so ``t``
    does not appear. ``patch_count // (kh*kw)`` is only right when ``t == 1``."""
    return (h // kh) * (w // kw)


def subgroup_layout(num_large: int, cp_size: int) -> tuple[int, int]:
    """Choose (number of sub-CP groups, ranks per sub-group).

    One large image and a CP group of 8 gives (1, 8) -- the report's "a single
    large image is partitioned across multiple devices". Four large images and 8
    ranks gives (4, 2), so each image is exchanged inside a pair instead of across
    all eight, which is the communication-fraction argument.

    Sub-groups are equal in size because a process group is formed from a rank
    list and an uneven split would leave a sub-group whose gather is a different
    shape on different ranks. So the count is the largest divisor of ``cp_size``
    that does not exceed ``num_large``.
    """
    if num_large <= 0 or cp_size <= 1:
        return (1, cp_size)
    best = 1
    for n in range(1, cp_size + 1):
        if cp_size % n == 0 and n <= num_large:
            best = n
    return (best, cp_size // best)


def balance_images(sizes: list[int], num_groups: int) -> list[int]:
    """Assign each image to a sub-group, longest-processing-time-first.

    Returns ``group_of[i]`` for every entry of ``sizes``. LPT rather than
    round-robin: round-robin on sizes [100, 10, 10, 10] with two groups gives 110
    against 20, while LPT gives 100 against 30. The report asks for "a
    load-balanced manner" and the imbalance it is trying to remove is exactly this
    one.
    """
    if num_groups <= 1:
        return [0] * len(sizes)
    load = [0] * num_groups
    group_of = [0] * len(sizes)
    for i in sorted(range(len(sizes)), key=lambda j: -sizes[j]):
        g = min(range(num_groups), key=lambda x: load[x])
        group_of[i] = g
        load[g] += sizes[i]
    return group_of


def classify(counts: list[int], cp_size: int, *, min_patches: int) -> list[int]:
    """Indices of the images worth partitioning within a sub-group.

    An image is only worth splitting if the split leaves each rank real work: the
    threshold is on the image's own patch count, not on the batch. Below it the
    image-level round-robin already balances better, because splitting a small
    image buys one gather per layer for nothing.
    """
    if cp_size <= 1:
        return []
    return [i for i, c in enumerate(counts) if c >= min_patches]


# --- The ViT/text stage boundary (DEP, report 5.2.3) ----------------------- #
#
# DEP splits ViT and text into separate PP stages, so the vision features have to
# cross a pipeline hop. PP's point-to-point buffers are sized ONCE, not per step,
# which forces a property that is easy to get wrong: the exchange shape must be a
# CONFIG-level upper bound, never derived from the current batch. A batch-derived
# shape works until a later batch carries more image tokens than the first one did,
# and then it fails inside the P2P rather than anywhere near the cause.
#
# Sizing needs no communication either way: grid_thw is replicated, so every stage
# computes the same layout from it. That is the same property dynamic CP relies on.


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
