# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Where the vision encoder's dynamic context parallelism cuts a micro-batch's images."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageShard:
    """One rank's band of rows of one image, in every frame."""

    row_start: int
    row_end: int
    grid: tuple[int, int, int]
    ranges: tuple[tuple[int, int], ...]
    """One flat ``[start, end)`` patch range per frame."""


def row_partition(
    t: int, h: int, w: int, *, kh: int, group_size: int
) -> list[ImageShard]:
    """Split an image's rows over ``group_size`` ranks in whole merge blocks, trailing ranks short."""
    if h % kh:
        raise ValueError(
            f"patch grid height {h} must divide the merge kernel height {kh}"
        )
    blocks = h // kh
    per = -(-blocks // group_size)
    frame = h * w
    shards: list[ImageShard] = []
    for r in range(group_size):
        r0 = min(r * per, blocks) * kh
        r1 = min((r + 1) * per, blocks) * kh
        ranges = tuple((f * frame + r0 * w, f * frame + r1 * w) for f in range(t))
        shards.append(
            ImageShard(row_start=r0, row_end=r1, grid=(t, r1 - r0, w), ranges=ranges)
        )
    return shards


def merged_tokens(h: int, w: int, kh: int, kw: int) -> int:
    """Tokens the projector emits for one image; time is pooled away."""
    return (h // kh) * (w // kw)


def subgroup_layout(num_large: int, cp_size: int) -> tuple[int, int]:
    """Equal sub-CP groups: the largest divisor of ``cp_size`` not above ``num_large``, and its size."""
    if num_large <= 0 or cp_size <= 1:
        return (1, cp_size)
    best = 1
    for n in range(1, cp_size + 1):
        if cp_size % n == 0 and n <= num_large:
            best = n
    return (best, cp_size // best)


def balance_images(sizes: list[int], num_groups: int) -> list[int]:
    """Assign each image to the least loaded group, longest first."""
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
    """The images with at least ``min_patches`` patches."""
    if cp_size <= 1:
        return []
    return [i for i, c in enumerate(counts) if c >= min_patches]


@dataclass(frozen=True)
class DynamicCPPlan:
    """A micro-batch's split images, the sub-CP group of each, and the groups' size."""

    whole: tuple[int, ...]
    split: tuple[int, ...]
    subgroup_of: tuple[int, ...]
    num_subgroups: int
    subgroup_size: int

    def images_of(self, subgroup: int) -> list[int]:
        return [
            i
            for i, s in zip(self.split, self.subgroup_of, strict=True)
            if s == subgroup
        ]


def plan_dynamic_cp(
    grids: list[list[int]], *, cp_size: int, kh: int, min_patches: int
) -> DynamicCPPlan | None:
    """Split every image of at least ``min_patches`` patches over a sub-CP group; ``None`` when none qualifies."""
    counts = [t * h * w for t, h, w in grids]
    split = [
        i
        for i in classify(counts, cp_size, min_patches=min_patches)
        if grids[i][1] % kh == 0
    ]
    if not split:
        return None
    num_subgroups, subgroup_size = subgroup_layout(len(split), cp_size)
    subgroup_of = balance_images([counts[i] for i in split], num_subgroups)
    split_set = set(split)
    return DynamicCPPlan(
        whole=tuple(i for i in range(len(grids)) if i not in split_set),
        split=tuple(split),
        subgroup_of=tuple(subgroup_of),
        num_subgroups=num_subgroups,
        subgroup_size=subgroup_size,
    )


def key_runs(
    grids: list[list[int]],
    whole: list[int],
    images: list[int],
    *,
    kh: int,
    group_size: int,
) -> list[tuple[int, int]]:
    """``(image, length)`` runs of the keys after the gather; padding rows carry image ``-1``."""
    runs = [(i, grids[i][0] * grids[i][1] * grids[i][2]) for i in whole]
    shards = {i: row_partition(*grids[i], kh=kh, group_size=group_size) for i in images}
    for r in range(group_size):
        for i in images:
            t, _, w = grids[i]
            band = shards[i][0].row_end - shards[i][0].row_start
            real = shards[i][r].row_end - shards[i][r].row_start
            for _ in range(t):
                runs.append((i, real * w))
                runs.append((-1, (band - real) * w))
    return [run for run in runs if run[1]]
