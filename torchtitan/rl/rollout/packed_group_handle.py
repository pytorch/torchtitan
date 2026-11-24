# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Packed rollout groups that stay in their rollout worker
(`BaseRolloutWorker.Config.rdma_min_group_bytes`): the worker packs a large training group
(`pack_rollout_group`) straight into a buffer registered with torch transport
(`RolloutGroupBuffers`), and the messages carry a handle instead of its bytes
(`PackedRolloutGroupHandle`). The controller sends a trainer DP replica leader only the handles its
column names; the leader reads the regions it trains on (`read_packed_rollout_group_handles`) and
unpacks them like any packed group (`unpack_rollout_group`).

    registered buffer, laid out by pack_rollout_group(..., region_padding=16):
    | tag | rollout 0 | tag | tag | rollout 1 | tag | ... | tag | image a | tag |
      tag = (group_id, generation), written by the worker, checked by the reader
"""

from __future__ import annotations

import itertools
import struct
import time
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.packed_group import (
    pack_rollout_group,
    PackedRolloutGroup,
    selected_rollout_fields,
)

if TYPE_CHECKING:
    from torchtitan.rl.rollout.types import RolloutGroup, RolloutGroupSummary
    from torchtitan.rl.torch_transport_rdma import RDMAByteBuffer, RDMAByteBuffers

# What the worker writes before and after each region: (group_id, generation).
_REGION_TAG = struct.Struct("<qq")
# Regions closer than this (only alignment padding and tags between them) are read as one range.
_MERGE_GAP_BYTES = 64


@dataclass(frozen=True, slots=True)
class PackedRegion:
    """A region of a group's registered buffer with its tags: `(group_id, generation)`, the
    region, then the same tag again, so a reader can tell the range still holds this group.

    The tags catch a read made after the buffer was reused for another group. They cannot catch a
    read racing a reuse (old tags, new region): only the release order rules that out (the
    controller releases a group after every read of its step finished)."""

    offset: int
    """Bytes from the start of the buffer, at the first tag."""

    num_bytes: int
    """Both tags and the region."""


@dataclass(frozen=True, slots=True)
class PackedRolloutGroupHandle:
    """A packed rollout group that stays in its rollout worker's registered buffer: where each
    rollout's region and each image's region sit, so a DP replica leader reads exactly the
    rollouts it trains on and the images they show. A few hundred bytes whatever the group's
    size."""

    buffer: RDMAByteBuffer

    group_id: int

    generation: int
    """The worker's count of packed groups, so `(group_id, generation)` names one use of a
    pooled buffer."""

    rollout_ids: tuple[int, ...]
    """Each rollout region's `Rollout.rollout_id`, as `PackedRolloutGroup.rollout_ids`."""

    rollout_regions: tuple[PackedRegion, ...]

    rollout_image_hashes: tuple[tuple[str, ...], ...]
    """Per rollout region: the `PromptImage.hash_value` of each image its turns show."""

    image_regions: dict[str, PackedRegion]

    def regions(self) -> list[PackedRegion]:
        return [*self.rollout_regions, *self.image_regions.values()]

    @property
    def num_bytes(self) -> int:
        """The bytes a reader of these rollouts and images reads."""
        return sum(region.num_bytes for region in self.regions())

    def select(self, rollout_ids: set[int]) -> PackedRolloutGroupHandle:
        """Only the rollouts in `rollout_ids` and the images they show, as
        `PackedRolloutGroup.select`."""
        selected = selected_rollout_fields(self, rollout_ids)
        return self if selected is None else replace(self, **selected)


class RolloutGroupBuffers:
    """A rollout worker's training groups that stay here: each packed into one registered buffer
    from `buffers`, from `pack` until the controller releases the group (`release`).

    Example:
        group_buffers = RolloutGroupBuffers(min_bytes=1 << 20, buffers=RDMAByteBuffers())
        summary, handle = group_buffers.pack(group)  # >= 1 MiB packed: a handle
        group_buffers.release([group.group_id])  # after the pull of the step that trains it
    """

    def __init__(self, *, min_bytes: int, buffers: RDMAByteBuffers) -> None:
        self._min_bytes = min_bytes
        self._buffers = buffers
        self._generations = itertools.count(1)
        self._handles_by_group_id: dict[int, PackedRolloutGroupHandle] = {}
        self._num_registrations_logged = 0

    def pack(
        self, group: RolloutGroup
    ) -> tuple[RolloutGroupSummary, PackedRolloutGroup | PackedRolloutGroupHandle]:
        """`pack_rollout_group(group)`: into a registered buffer here, returning a handle, if the
        packed group is at least `min_bytes`; else by value, as `pack_rollout_group`. Adds the
        worker's buffer metrics to the summary's group metrics for groups kept here."""
        pack_start = time.perf_counter()
        registered: list[tuple[np.ndarray, RDMAByteBuffer]] = []

        def allocate(num_bytes: int) -> np.ndarray:
            if num_bytes < self._min_bytes:
                return np.zeros(num_bytes, dtype=np.uint8)
            tensor, buffer = self._buffers.allocate(num_bytes)
            registered.append((tensor.numpy(), buffer))
            return registered[-1][0]

        summary, packed = pack_rollout_group(
            group, allocate=allocate, region_padding=_REGION_TAG.size
        )
        if not registered:
            return summary, packed
        [(destination, buffer)] = registered
        generation = next(self._generations)
        tag = np.frombuffer(
            _REGION_TAG.pack(group.group_id, generation), dtype=np.uint8
        )

        def tagged(region: np.ndarray) -> PackedRegion:
            # A view into `destination` with `region_padding` free bytes on each side.
            start = region.ctypes.data - destination.ctypes.data
            end = start + region.nbytes
            destination[start - _REGION_TAG.size : start] = tag
            destination[end : end + _REGION_TAG.size] = tag
            return PackedRegion(
                offset=start - _REGION_TAG.size,
                num_bytes=region.nbytes + 2 * _REGION_TAG.size,
            )

        handle = PackedRolloutGroupHandle(
            buffer=buffer,
            group_id=group.group_id,
            generation=generation,
            rollout_ids=packed.rollout_ids,
            rollout_regions=tuple(tagged(region) for region in packed.rollout_regions),
            rollout_image_hashes=packed.rollout_image_hashes,
            image_regions={
                hash_value: tagged(region)
                for hash_value, region in packed.image_regions.items()
            },
        )
        self._handles_by_group_id[group.group_id] = handle
        pool = self._buffers.pool
        new_registrations = pool.num_registrations - self._num_registrations_logged
        self._num_registrations_logged = pool.num_registrations
        summary.group.metrics.extend(
            [
                m.Metric(
                    "timing/rollout_worker/pack_group",
                    m.Mean(time.perf_counter() - pack_start),
                ),
                # 0 once the pool has a free buffer for every group size it sees.
                m.Metric("rollout_worker/pool_registrations", m.Sum(new_registrations)),
                m.Metric("rollout_worker/pool_bytes", m.Max(pool.registered_bytes)),
                # The pool's buffers handed out to live groups, size-class rounded.
                m.Metric("rollout_worker/pool_bytes_in_use", m.Max(pool.bytes_in_use)),
            ]
        )
        return summary, handle

    def release(self, group_ids: Iterable[int]) -> None:
        """Return the buffers of the groups in `group_ids` that are here to the pool; ignore the
        others (the controller sends every worker the same ids)."""
        for group_id in group_ids:
            handle = self._handles_by_group_id.pop(group_id, None)
            if handle is not None:
                self._buffers.release(handle.buffer)


def read_packed_rollout_group_handles(
    handles: list[PackedRolloutGroupHandle], buffers: RDMAByteBuffers
) -> list[PackedRolloutGroup]:
    """The packed groups `handles` name, read from their rollout workers: every region of every
    group in one `buffers.read` call, adjacent regions merged into one range. Unpack each with
    `unpack_rollout_group`.

    Raises:
        RuntimeError: a region does not start and end with its group's `(group_id, generation)`,
            i.e. the buffer was released and reused before this read (a bug in the release order).
    """
    requests = [(handle.buffer, _merged_ranges(handle.regions())) for handle in handles]
    chunks_by_request = buffers.read(requests)
    packed_groups = []
    for handle, (_, ranges), chunks in zip(
        handles, requests, chunks_by_request, strict=True
    ):
        tag = _REGION_TAG.pack(handle.group_id, handle.generation)

        def region_bytes(region: PackedRegion) -> np.ndarray:
            # The merged range holding the region, then the region within it, without its tags.
            for (range_start, range_length), chunk in zip(ranges, chunks, strict=True):
                if range_start <= region.offset < range_start + range_length:
                    start = region.offset - range_start
                    data = chunk[start : start + region.num_bytes]
                    head = data[: _REGION_TAG.size].tobytes()
                    tail = data[-_REGION_TAG.size :].tobytes()
                    if head != tag or tail != tag:
                        raise RuntimeError(
                            f"Rollout group {handle.group_id} (generation "
                            f"{handle.generation}): bytes {region.offset}.."
                            f"{region.offset + region.num_bytes} of its buffer hold "
                            f"{_REGION_TAG.unpack(head)} ... {_REGION_TAG.unpack(tail)}, so "
                            "the rollout worker reused the buffer before this read."
                        )
                    return data[_REGION_TAG.size : -_REGION_TAG.size]
            raise AssertionError(f"region {region} is in no read range")

        packed_groups.append(
            PackedRolloutGroup(
                group_id=handle.group_id,
                rollout_ids=handle.rollout_ids,
                rollout_regions=tuple(
                    region_bytes(region) for region in handle.rollout_regions
                ),
                rollout_image_hashes=handle.rollout_image_hashes,
                image_regions={
                    hash_value: region_bytes(region)
                    for hash_value, region in handle.image_regions.items()
                },
            )
        )
    return packed_groups


def _merged_ranges(regions: Iterable[PackedRegion]) -> list[tuple[int, int]]:
    """`(offset, num_bytes)` ranges covering `regions`, merging regions less than
    `_MERGE_GAP_BYTES` apart, so a leader that needs most of a group issues few reads."""
    ranges: list[list[int]] = []
    for region in sorted(regions, key=lambda packed_region: packed_region.offset):
        end = region.offset + region.num_bytes
        if ranges and region.offset - ranges[-1][1] < _MERGE_GAP_BYTES:
            ranges[-1][1] = end
        else:
            ranges.append([region.offset, end])
    return [(start, end - start) for start, end in ranges]
