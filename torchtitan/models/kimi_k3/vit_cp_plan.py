# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Where the cuts go for dynamic vision context parallelism: how one large image
is split across the ranks of a CP sub-group, and how several images are spread
over the sub-groups.

Pure functions, so the scheduling decisions are testable without ranks. A cut may
only land on a merge-row block, the ``kh`` consecutive grid rows the projector
merges as one unit; splitting a sub-group per image rather than gathering over
the whole CP group keeps the key exchange from growing with the group.
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
    """Split one image across ``group_size`` ranks along the spatial rows.

    Every rank keeps every frame and takes a band of rows that is a multiple of
    ``kh``: the projector merges each ``(kh, kw)`` block, and its temporal pool
    means over all frames, so a split by frame would give each rank the mean of
    its own frames instead. A band is strided in the packed stream once ``t > 1``,
    hence ``ranges``, and the ceiling split keeps any deficit on the trailing
    ranks so the caller's padding lands at the end of the gathered stream.
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
    """Tokens the projector emits for one image; time is collapsed, so ``t`` does not appear."""
    return (h // kh) * (w // kw)


def subgroup_layout(num_large: int, cp_size: int) -> tuple[int, int]:
    """Choose (number of sub-CP groups, ranks per sub-group).

    Sub-groups are equal in size, since an uneven split would make one gather a
    different shape on different ranks, so the count is the largest divisor of
    ``cp_size`` that does not exceed ``num_large``.
    """
    if num_large <= 0 or cp_size <= 1:
        return (1, cp_size)
    best = 1
    for n in range(1, cp_size + 1):
        if cp_size % n == 0 and n <= num_large:
            best = n
    return (best, cp_size // best)


def balance_images(sizes: list[int], num_groups: int) -> list[int]:
    """Assign each image to a sub-group longest-first, as ``group_of[i]``."""
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
    """Indices of the images big enough to be worth a split, which costs one gather per layer."""
    if cp_size <= 1:
        return []
    return [i for i, c in enumerate(counts) if c >= min_patches]
