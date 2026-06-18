# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class SegmentReduceDescriptor:
    """Flat input segment that should be reduced to one owner rank."""

    src_offset: int
    numel: int
    dst_rank: int
    name: str | None = None


@dataclass(frozen=True)
class PlannedSegmentReduceDescriptor:
    """Segment descriptor after owner-partition offsets are assigned."""

    src_offset: int
    numel: int
    dst_rank: int
    dst_offset: int
    original_index: int
    name: str | None = None


@dataclass(frozen=True)
class CoalescedSegmentReduceDescriptor:
    """Contiguous run copied as one source-to-owner-partition segment."""

    src_offset: int
    numel: int
    dst_rank: int
    dst_offset: int
    original_indices: tuple[int, ...]


@dataclass(frozen=True)
class SegmentReducePlan:
    """Flat owner-partition layout for a segment reduce-to-owner operation."""

    segments: tuple[PlannedSegmentReduceDescriptor, ...]
    coalesced_segments: tuple[CoalescedSegmentReduceDescriptor, ...]
    rank_offsets: tuple[int, ...]
    rank_numels: tuple[int, ...]
    chunk_numel: int

    @property
    def world_size(self) -> int:
        return len(self.rank_numels)

    @property
    def send_numel(self) -> int:
        return self.world_size * self.chunk_numel


@dataclass(frozen=True)
class SegmentReduceResult:
    """Result of the packed reduce-scatter fast path."""

    output: torch.Tensor
    owned_segments: tuple[PlannedSegmentReduceDescriptor, ...]
    owned_views: tuple[torch.Tensor, ...]
    plan: SegmentReducePlan


def segment_descriptors_from_offsets(
    offsets: Sequence[int],
    dst_ranks: Sequence[int],
    *,
    names: Sequence[str | None] | None = None,
) -> tuple[SegmentReduceDescriptor, ...]:
    """Build flat descriptors from segment end offsets and owner ranks."""
    if len(offsets) != len(dst_ranks):
        raise ValueError(
            f"Expected {len(offsets)} offsets to match {len(dst_ranks)} dst_ranks."
        )
    if names is not None and len(names) != len(offsets):
        raise ValueError(
            f"Expected {len(names)} names to match {len(offsets)} offsets."
        )

    descriptors: list[SegmentReduceDescriptor] = []
    start = 0
    for index, (end, dst_rank) in enumerate(zip(offsets, dst_ranks, strict=True)):
        if end < start:
            raise ValueError(
                f"Segment offsets must be non-decreasing, but offset {index} "
                f"is {end} after previous offset {start}."
            )
        descriptors.append(
            SegmentReduceDescriptor(
                src_offset=start,
                numel=end - start,
                dst_rank=dst_rank,
                name=None if names is None else names[index],
            )
        )
        start = end
    return tuple(descriptors)


def build_segment_reduce_plan(
    descriptors: Sequence[SegmentReduceDescriptor],
    world_size: int,
) -> SegmentReducePlan:
    """Assign owner-partition offsets and coalesce adjacent copy descriptors."""
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, but got {world_size}.")

    rank_numels = [0] * world_size
    planned: list[PlannedSegmentReduceDescriptor] = []
    coalesced: list[CoalescedSegmentReduceDescriptor] = []
    for index, descriptor in enumerate(descriptors):
        _validate_descriptor(descriptor, index, world_size)
        dst_offset = rank_numels[descriptor.dst_rank]
        rank_numels[descriptor.dst_rank] += descriptor.numel
        planned_descriptor = PlannedSegmentReduceDescriptor(
            src_offset=descriptor.src_offset,
            numel=descriptor.numel,
            dst_rank=descriptor.dst_rank,
            dst_offset=dst_offset,
            original_index=index,
            name=descriptor.name,
        )
        planned.append(planned_descriptor)
        if descriptor.numel == 0:
            continue
        if coalesced and _can_coalesce(coalesced[-1], planned_descriptor):
            previous = coalesced[-1]
            coalesced[-1] = CoalescedSegmentReduceDescriptor(
                src_offset=previous.src_offset,
                numel=previous.numel + planned_descriptor.numel,
                dst_rank=previous.dst_rank,
                dst_offset=previous.dst_offset,
                original_indices=(
                    *previous.original_indices,
                    planned_descriptor.original_index,
                ),
            )
        else:
            coalesced.append(
                CoalescedSegmentReduceDescriptor(
                    src_offset=planned_descriptor.src_offset,
                    numel=planned_descriptor.numel,
                    dst_rank=planned_descriptor.dst_rank,
                    dst_offset=planned_descriptor.dst_offset,
                    original_indices=(planned_descriptor.original_index,),
                )
            )

    rank_offsets: list[int] = []
    total_numel = 0
    for numel in rank_numels:
        rank_offsets.append(total_numel)
        total_numel += numel

    return SegmentReducePlan(
        segments=tuple(planned),
        coalesced_segments=tuple(coalesced),
        rank_offsets=tuple(rank_offsets),
        rank_numels=tuple(rank_numels),
        chunk_numel=max(rank_numels, default=0),
    )


def pack_segment_reduce_scatter_input(
    flat_input: torch.Tensor,
    plan: SegmentReducePlan,
) -> torch.Tensor:
    """Pack flat input into equal-size owner chunks for reduce-scatter."""
    _validate_flat_input(flat_input, plan)
    send = flat_input.new_zeros(plan.send_numel)
    if plan.send_numel == 0:
        return send

    send_by_rank = send.view(plan.world_size, plan.chunk_numel)
    copy_srcs: list[torch.Tensor] = []
    copy_dsts: list[torch.Tensor] = []
    for segment in plan.coalesced_segments:
        copy_srcs.append(flat_input.narrow(0, segment.src_offset, segment.numel))
        copy_dsts.append(
            send_by_rank[segment.dst_rank].narrow(
                0,
                segment.dst_offset,
                segment.numel,
            )
        )
    _foreach_copy_(copy_dsts, copy_srcs)
    return send


def owned_segment_views(
    output: torch.Tensor,
    plan: SegmentReducePlan,
    rank: int,
) -> tuple[torch.Tensor, ...]:
    """Return owner-local views in original descriptor order for ``rank``."""
    if rank < 0 or rank >= plan.world_size:
        raise ValueError(
            f"rank must be in [0, {plan.world_size}), but got {rank}."
        )
    if output.dim() != 1:
        raise ValueError(f"Expected a flat output tensor, but got dim={output.dim()}.")
    if output.numel() < plan.rank_numels[rank]:
        raise ValueError(
            f"Output has {output.numel()} elements but rank {rank} owns "
            f"{plan.rank_numels[rank]} elements."
        )

    return tuple(
        output.narrow(0, segment.dst_offset, segment.numel)
        for segment in plan.segments
        if segment.dst_rank == rank
    )


def segment_reduce_to_owner(
    flat_input: torch.Tensor,
    descriptors: Sequence[SegmentReduceDescriptor],
    *,
    group: Any | None = None,
    op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM,
) -> SegmentReduceResult:
    """Reduce flat segments to owner ranks via packed ``reduce_scatter_tensor``.

    This is the Phase 1 fast path for owner-partitioned reductions. It pads
    owner partitions to ``max(rank_numels)`` and relies on the process group's
    native reduce-scatter implementation, which is NCCL for CUDA/NCCL groups.
    """
    world_size = dist.get_world_size(group)
    plan = build_segment_reduce_plan(descriptors, world_size)
    return segment_reduce_to_owner_with_plan(
        flat_input,
        plan,
        group=group,
        op=op,
    )


def segment_reduce_to_owner_with_plan(
    flat_input: torch.Tensor,
    plan: SegmentReducePlan,
    *,
    group: Any | None = None,
    op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM,
) -> SegmentReduceResult:
    """Run packed reduce-scatter using a precomputed segment reduce plan."""
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    if world_size != plan.world_size:
        raise ValueError(
            f"Plan world_size={plan.world_size} does not match process group "
            f"world_size={world_size}."
        )
    send = pack_segment_reduce_scatter_input(flat_input, plan)
    output = flat_input.new_empty(plan.chunk_numel)
    if plan.chunk_numel > 0:
        dist.reduce_scatter_tensor(
            output=output,
            input=send,
            op=op,
            group=group,
        )

    owned_segments = tuple(
        segment for segment in plan.segments if segment.dst_rank == rank
    )
    views = owned_segment_views(output, plan, rank)
    return SegmentReduceResult(
        output=output,
        owned_segments=owned_segments,
        owned_views=views,
        plan=plan,
    )


def _validate_descriptor(
    descriptor: SegmentReduceDescriptor,
    index: int,
    world_size: int,
) -> None:
    if descriptor.src_offset < 0:
        raise ValueError(
            f"Segment {index} has negative src_offset {descriptor.src_offset}."
        )
    if descriptor.numel < 0:
        raise ValueError(f"Segment {index} has negative numel {descriptor.numel}.")
    if descriptor.dst_rank < 0 or descriptor.dst_rank >= world_size:
        raise ValueError(
            f"Segment {index} dst_rank must be in [0, {world_size}), "
            f"but got {descriptor.dst_rank}."
        )


def _can_coalesce(
    previous: CoalescedSegmentReduceDescriptor,
    current: PlannedSegmentReduceDescriptor,
) -> bool:
    return (
        previous.dst_rank == current.dst_rank
        and previous.src_offset + previous.numel == current.src_offset
        and previous.dst_offset + previous.numel == current.dst_offset
    )


def _validate_flat_input(
    flat_input: torch.Tensor,
    plan: SegmentReducePlan,
) -> None:
    if flat_input.dim() != 1:
        raise ValueError(
            f"Expected a flat input tensor, but got dim={flat_input.dim()}."
        )
    for segment in plan.coalesced_segments:
        if segment.src_offset + segment.numel > flat_input.numel():
            raise ValueError(
                "Segment input range exceeds flat input numel: "
                f"offset={segment.src_offset}, numel={segment.numel}, "
                f"input_numel={flat_input.numel()}."
            )


def _foreach_copy_(
    dst_tensors: list[torch.Tensor],
    src_tensors: list[torch.Tensor],
) -> None:
    if len(dst_tensors) != len(src_tensors):
        raise AssertionError(
            f"Expected {len(dst_tensors)} destination tensors to match "
            f"{len(src_tensors)} source tensors."
        )
    if not dst_tensors:
        return
    if torch.compiler.is_compiling():
        for dst, src in zip(dst_tensors, src_tensors, strict=True):
            dst.copy_(src)
    else:
        torch._foreach_copy_(dst_tensors, src_tensors)


__all__ = [
    "CoalescedSegmentReduceDescriptor",
    "PlannedSegmentReduceDescriptor",
    "SegmentReduceDescriptor",
    "SegmentReducePlan",
    "SegmentReduceResult",
    "build_segment_reduce_plan",
    "owned_segment_views",
    "pack_segment_reduce_scatter_input",
    "segment_descriptors_from_offsets",
    "segment_reduce_to_owner",
    "segment_reduce_to_owner_with_plan",
]
