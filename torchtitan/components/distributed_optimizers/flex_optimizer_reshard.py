# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public bucket contracts and private resharding for distributed optimizers."""

from __future__ import annotations

import fnmatch
import hashlib
import heapq
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Generic, TypeVar

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard


__all__ = ["BucketConfig", "BucketSpec", "assign_balanced_owners"]


@dataclass(frozen=True, slots=True)
class BucketConfig:
    """Static bucket configuration resolved after runtime meshes exist.

    ``mesh_axes`` contains exactly one storage mesh axis name. When
    ``owner_rank_by_fqn`` is nonempty, its redistributed parameters determine
    the resolved mesh; otherwise all parameters matching ``patterns`` do. For
    DistributedMuon, owner-assigned parameters use ``Owned`` while compute-ready
    parameters in the same bucket may use a different storage mesh.
    """

    patterns: tuple[str, ...]
    owner_rank_by_fqn: Mapping[str, int]
    mesh_axes: tuple[str, ...]
    name: str = ""

    def __post_init__(self) -> None:
        mesh_axes = tuple(self.mesh_axes)
        if len(mesh_axes) != 1:
            raise ValueError(
                "BucketConfig mesh_axes must contain exactly one mesh axis"
            )
        object.__setattr__(self, "patterns", tuple(self.patterns))
        object.__setattr__(self, "owner_rank_by_fqn", dict(self.owner_rank_by_fqn))
        object.__setattr__(self, "mesh_axes", mesh_axes)

    def bind(self, mesh: DeviceMesh) -> BucketSpec:
        return BucketSpec(
            patterns=self.patterns,
            owner_rank_by_fqn=self.owner_rank_by_fqn,
            mesh=mesh,
            name=self.name,
        )


@dataclass(frozen=True, slots=True)
class BucketSpec:
    """One ordered optimizer-work bucket selected by canonical FQN.

    Patterns use case-sensitive ``fnmatch`` syntax. Every optimizer FQN must
    match exactly one bucket, and sequence order controls execution order.
    ``mesh`` is the bucket's exact one-dimensional communication mesh.
    ``owner_rank_by_fqn`` must exactly cover parameters requiring whole-tensor
    redistribution and uses mesh-local ranks. Compute-ready parameters have no
    owner entry. A rank-0 entry is also accepted for compute-ready parameters
    on a one-rank mesh, where sharded storage may normalize to replication.
    ``name`` is diagnostic metadata only.
    """

    patterns: tuple[str, ...]
    owner_rank_by_fqn: Mapping[str, int]
    mesh: DeviceMesh
    name: str = ""

    def __post_init__(self) -> None:
        if self.mesh.ndim != 1:
            raise ValueError("BucketSpec mesh must be one-dimensional")
        object.__setattr__(self, "patterns", tuple(self.patterns))
        object.__setattr__(self, "owner_rank_by_fqn", dict(self.owner_rank_by_fqn))


def assign_balanced_owners(
    bucket_fqns: Sequence[Sequence[str]],
    memory_estimate_by_fqn: Mapping[str, int],
    *,
    num_ranks: int,
    initial_memory_by_rank: Sequence[int] | None = None,
) -> tuple[dict[str, int], ...]:
    """Greedily balance selected FQNs across group-local rank indices.

    Only FQNs present in ``memory_estimate_by_fqn`` receive owners. One running
    load vector balances cumulatively across buckets; FQN and rank ordering
    make equal-load assignments deterministic.
    """
    initial_memory_by_rank = initial_memory_by_rank or (0,) * num_ranks
    rank_loads = list(zip(initial_memory_by_rank, range(num_ranks), strict=True))
    heapq.heapify(rank_loads)
    owners_by_bucket = []
    for bucket in bucket_fqns:
        bucket_owners = {}
        candidates = (fqn for fqn in bucket if fqn in memory_estimate_by_fqn)
        for fqn in sorted(
            candidates, key=lambda name: (-memory_estimate_by_fqn[name], name)
        ):
            load, rank = heapq.heappop(rank_loads)
            bucket_owners[fqn] = rank
            heapq.heappush(rank_loads, (load + memory_estimate_by_fqn[fqn], rank))
        owners_by_bucket.append(bucket_owners)
    return tuple(owners_by_bucket)


_ItemT = TypeVar("_ItemT")


def _bind_bucket_configs(
    configs: Sequence[BucketConfig],
    storage_by_fqn: Mapping[str, DTensor],
) -> tuple[BucketSpec, ...]:
    """Bind static configs to storage meshes after model parallelization."""
    specs = []
    for config in configs:
        candidates = tuple(config.owner_rank_by_fqn) or tuple(
            fqn
            for fqn in storage_by_fqn
            if any(fnmatch.fnmatchcase(fqn, pattern) for pattern in config.patterns)
        )
        if not candidates:
            continue

        meshes = []
        for fqn in candidates:
            storage_mesh = storage_by_fqn[fqn].device_mesh
            meshes.append(storage_mesh[config.mesh_axes])

        mesh = meshes[0]
        if any(not torch.equal(candidate.mesh, mesh.mesh) for candidate in meshes[1:]):
            raise ValueError(
                f"bucket {config.name!r} resolves to inconsistent communication meshes"
            )
        specs.append(config.bind(mesh))
    return tuple(specs)


def _resolve_buckets(
    items: Sequence[_ItemT],
    specs: Sequence[BucketSpec],
    *,
    fqn: Callable[[_ItemT], str],
) -> tuple[tuple[_ItemT, ...], ...]:
    resolved: list[list[_ItemT]] = [[] for _ in specs]
    for item in items:
        name = fqn(item)
        matches = [
            index
            for index, spec in enumerate(specs)
            if any(fnmatch.fnmatchcase(name, pattern) for pattern in spec.patterns)
        ]
        if len(matches) != 1:
            raise ValueError(f"optimizer parameter {name!r} must match one bucket")
        resolved[matches[0]].append(item)
    return tuple(tuple(bucket) for bucket in resolved)


@dataclass(frozen=True, slots=True)
class _MatrixBlock:
    """One rectangular tensor region."""

    offsets: tuple[int, ...]
    shape: tuple[int, ...]

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True, slots=True)
class _ParticipantPartition:
    """One participant's tensor shape and its global logical regions."""

    participant: int
    tensor_shape: tuple[int, ...]
    logical_blocks: tuple[_MatrixBlock, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tensor_shape", tuple(self.tensor_shape))
        object.__setattr__(self, "logical_blocks", tuple(self.logical_blocks))
        if any(size < 0 for size in self.tensor_shape):
            raise ValueError("participant tensor shape must be nonnegative")


@dataclass(frozen=True, slots=True)
class _MatrixBlockRoute:
    """Map one logical block between differently shaped endpoint tensors."""

    logical_block: _MatrixBlock
    source_block: _MatrixBlock
    destination_block: _MatrixBlock
    source_participants: tuple[int, ...]
    destination_participants: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_participants", tuple(self.source_participants))
        object.__setattr__(
            self, "destination_participants", tuple(self.destination_participants)
        )
        if not (
            self.logical_block.numel
            == self.source_block.numel
            == self.destination_block.numel
        ):
            raise ValueError("route endpoint blocks must have equal numel")
        if len(set(self.source_participants)) != len(self.source_participants) or len(
            set(self.destination_participants)
        ) != len(self.destination_participants):
            raise ValueError("route endpoint participants must be unique")


@dataclass(frozen=True, slots=True)
class _RedistributionPlan:
    """Transport-neutral exact block partitions in both directions."""

    participants: tuple[int, ...]
    logical_shape: tuple[int, ...]
    storage_partitions: tuple[_ParticipantPartition, ...]
    compute_partitions: tuple[_ParticipantPartition, ...]
    storage_to_compute_routes: tuple[_MatrixBlockRoute, ...]
    compute_to_storage_routes: tuple[_MatrixBlockRoute, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "participants", tuple(self.participants))
        object.__setattr__(self, "logical_shape", tuple(self.logical_shape))
        object.__setattr__(self, "storage_partitions", tuple(self.storage_partitions))
        object.__setattr__(self, "compute_partitions", tuple(self.compute_partitions))
        object.__setattr__(
            self, "storage_to_compute_routes", tuple(self.storage_to_compute_routes)
        )
        object.__setattr__(
            self, "compute_to_storage_routes", tuple(self.compute_to_storage_routes)
        )
        if len(set(self.participants)) != len(self.participants):
            raise ValueError("redistribution participants must be unique")
        for name, partitions in (
            ("storage", self.storage_partitions),
            ("compute", self.compute_partitions),
        ):
            if tuple(partition.participant for partition in partitions) != tuple(
                self.participants
            ):
                raise ValueError(
                    f"{name} partitions must follow the redistribution participants"
                )
            _validate_matrix_block_partition(
                tuple(
                    block
                    for partition in partitions
                    for block in partition.logical_blocks
                ),
                self.logical_shape,
                direction=f"{name} logical partition",
            )

        all_routes = self.storage_to_compute_routes + self.compute_to_storage_routes
        if any(
            not route.source_participants or not route.destination_participants
            for route in all_routes
        ):
            raise ValueError("redistribution routes require sources and destinations")
        participant_set = set(self.participants)
        if any(
            participant not in participant_set
            for route in all_routes
            for participant in (
                route.source_participants + route.destination_participants
            )
        ):
            raise ValueError("redistribution route references an unknown participant")

        def route_key(route: _MatrixBlockRoute) -> tuple[Any, ...]:
            return (
                route.logical_block.offsets,
                route.logical_block.shape,
                route.source_block.offsets,
                route.source_block.shape,
                route.destination_block.offsets,
                route.destination_block.shape,
                route.source_participants,
                route.destination_participants,
            )

        mirrored_forward_keys = sorted(
            (
                route.logical_block.offsets,
                route.logical_block.shape,
                route.destination_block.offsets,
                route.destination_block.shape,
                route.source_block.offsets,
                route.source_block.shape,
                route.destination_participants,
                route.source_participants,
            )
            for route in self.storage_to_compute_routes
        )
        reverse_keys = sorted(map(route_key, self.compute_to_storage_routes))
        if mirrored_forward_keys != reverse_keys:
            raise ValueError(
                "compute-to-storage routes must exactly invert storage-to-compute"
            )
        for direction, routes in (
            ("storage-to-compute", self.storage_to_compute_routes),
            ("compute-to-storage", self.compute_to_storage_routes),
        ):
            _validate_matrix_block_partition(
                tuple(route.logical_block for route in routes),
                self.logical_shape,
                direction=direction,
            )

        storage_by_participant = {
            partition.participant: partition for partition in self.storage_partitions
        }
        compute_by_participant = {
            partition.participant: partition for partition in self.compute_partitions
        }
        for participant in self.participants:
            storage = storage_by_participant[participant]
            compute = compute_by_participant[participant]
            endpoint_specs = (
                (
                    "storage-to-compute source",
                    self.storage_to_compute_routes,
                    "source_participants",
                    "source_block",
                    storage,
                ),
                (
                    "storage-to-compute destination",
                    self.storage_to_compute_routes,
                    "destination_participants",
                    "destination_block",
                    compute,
                ),
                (
                    "compute-to-storage source",
                    self.compute_to_storage_routes,
                    "source_participants",
                    "source_block",
                    compute,
                ),
                (
                    "compute-to-storage destination",
                    self.compute_to_storage_routes,
                    "destination_participants",
                    "destination_block",
                    storage,
                ),
            )
            for (
                direction,
                routes,
                participants_attr,
                block_attr,
                partition,
            ) in endpoint_specs:
                participant_routes = tuple(
                    route
                    for route in routes
                    if participant in getattr(route, participants_attr)
                )
                _validate_blocks_cover_partition(
                    tuple(route.logical_block for route in participant_routes),
                    partition.logical_blocks,
                    self.logical_shape,
                    direction=f"{direction} {participant}",
                )
                _validate_matrix_block_partition(
                    tuple(getattr(route, block_attr) for route in participant_routes),
                    partition.tensor_shape,
                    direction=f"{direction} tensor {participant}",
                )

    def storage_partition(self, participant: int) -> _ParticipantPartition:
        return next(
            partition
            for partition in self.storage_partitions
            if partition.participant == participant
        )

    def compute_partition(self, participant: int) -> _ParticipantPartition:
        return next(
            partition
            for partition in self.compute_partitions
            if partition.participant == participant
        )


def _validate_matrix_block_partition(
    blocks: tuple[_MatrixBlock, ...],
    logical_shape: tuple[int, ...],
    *,
    direction: str,
) -> None:
    if any(size < 0 for size in logical_shape) or any(
        len(block.offsets) != len(logical_shape)
        or len(block.shape) != len(logical_shape)
        or any(
            offset < 0 or size < 0 or offset + size > logical_size
            for offset, size, logical_size in zip(
                block.offsets, block.shape, logical_shape, strict=True
            )
        )
        for block in blocks
    ):
        raise ValueError(f"{direction} blocks must be in bounds")

    positive_blocks = tuple(block for block in blocks if block.numel)
    for index, first in enumerate(positive_blocks):
        for second in positive_blocks[index + 1 :]:
            if all(
                max(first_offset, second_offset)
                < min(
                    first_offset + first_size,
                    second_offset + second_size,
                )
                for first_offset, first_size, second_offset, second_size in zip(
                    first.offsets,
                    first.shape,
                    second.offsets,
                    second.shape,
                    strict=True,
                )
            ):
                raise NotImplementedError(
                    "overlapping logical matrix blocks are not supported"
                )

    if sum(block.numel for block in blocks) != math.prod(logical_shape):
        raise ValueError(f"{direction} blocks do not cover the logical tensor")


def _matrix_block_intersection_numel(first: _MatrixBlock, second: _MatrixBlock) -> int:
    if len(first.shape) != len(second.shape):
        return 0
    intersection_shape = tuple(
        max(
            0,
            min(first_offset + first_size, second_offset + second_size)
            - max(first_offset, second_offset),
        )
        for first_offset, first_size, second_offset, second_size in zip(
            first.offsets,
            first.shape,
            second.offsets,
            second.shape,
            strict=True,
        )
    )
    return math.prod(intersection_shape)


def _validate_blocks_cover_partition(
    blocks: tuple[_MatrixBlock, ...],
    expected: tuple[_MatrixBlock, ...],
    logical_shape: tuple[int, ...],
    *,
    direction: str,
) -> None:
    """Validate that nonoverlapping blocks exactly cover an expected region."""
    if any(
        len(block.offsets) != len(logical_shape)
        or len(block.shape) != len(logical_shape)
        or any(
            offset < 0 or size < 0 or offset + size > logical_size
            for offset, size, logical_size in zip(
                block.offsets, block.shape, logical_shape, strict=True
            )
        )
        for block in blocks
    ):
        raise ValueError(f"{direction} blocks must be in bounds")

    positive_blocks = tuple(block for block in blocks if block.numel)
    for index, first in enumerate(positive_blocks):
        for second in positive_blocks[index + 1 :]:
            if _matrix_block_intersection_numel(first, second):
                raise NotImplementedError(
                    "overlapping logical matrix blocks are not supported"
                )

    if sum(block.numel for block in blocks) != sum(block.numel for block in expected):
        raise ValueError(f"{direction} blocks do not cover the participant partition")
    for block in positive_blocks:
        if (
            sum(
                _matrix_block_intersection_numel(block, expected_block)
                for expected_block in expected
            )
            != block.numel
        ):
            raise ValueError(f"{direction} blocks leave the participant partition")


def _build_owned_redistribution_plan(
    storage_blocks: Sequence[tuple[tuple[int, ...], _MatrixBlock]],
    *,
    participants: tuple[int, ...],
    owner: int,
    logical_shape: tuple[int, ...],
) -> _RedistributionPlan:
    """Build mirrored routes from one canonical block-to-holders mapping."""
    storage_partitions = []
    storage_mapping_by_participant = {}
    for holders, logical_block in storage_blocks:
        if len(holders) != 1:
            raise NotImplementedError(
                "redistributed optimizer storage requires one holder per block"
            )
        participant = holders[0]
        tensor_block = _MatrixBlock(
            offsets=(0,) * len(logical_block.shape),
            shape=logical_block.shape,
        )
        storage_mapping_by_participant[participant] = (logical_block, tensor_block)
    for participant in participants:
        logical_block, tensor_block = storage_mapping_by_participant[participant]
        storage_partitions.append(
            _ParticipantPartition(
                participant=participant,
                tensor_shape=tensor_block.shape,
                logical_blocks=(logical_block,),
            )
        )

    full_block = _MatrixBlock(
        offsets=(0,) * len(logical_shape),
        shape=logical_shape,
    )
    compute_partitions = tuple(
        _ParticipantPartition(
            participant=participant,
            tensor_shape=logical_shape if participant == owner else (0,),
            logical_blocks=(full_block,) if participant == owner else (),
        )
        for participant in participants
    )
    return _RedistributionPlan(
        participants=participants,
        logical_shape=logical_shape,
        storage_partitions=tuple(storage_partitions),
        compute_partitions=compute_partitions,
        storage_to_compute_routes=tuple(
            _MatrixBlockRoute(
                logical_block=logical_block,
                source_block=storage_mapping_by_participant[holders[0]][1],
                destination_block=logical_block,
                source_participants=holders,
                destination_participants=(owner,),
            )
            for holders, logical_block in storage_blocks
        ),
        compute_to_storage_routes=tuple(
            _MatrixBlockRoute(
                logical_block=logical_block,
                source_block=logical_block,
                destination_block=storage_mapping_by_participant[holders[0]][1],
                source_participants=(owner,),
                destination_participants=holders,
            )
            for holders, logical_block in storage_blocks
        ),
    )


@dataclass(frozen=True, slots=True)
class _PackedSpan:
    """Physical packed-buffer location for a logical matrix block."""

    block: _MatrixBlock
    buffer_offset: int

    @property
    def numel(self) -> int:
        return self.block.numel


@dataclass(frozen=True, slots=True)
class _PackedAllToAllSchedule:
    process_group: dist.ProcessGroup
    participants: tuple[int, ...]
    local_participant: int
    input_split_sizes: tuple[int, ...]
    output_split_sizes: tuple[int, ...]
    input_spans_by_parameter: tuple[tuple[_PackedSpan, ...], ...]
    output_spans_by_parameter: tuple[tuple[_PackedSpan, ...], ...]

    @property
    def input_buffer_numel(self) -> int:
        return sum(self.input_split_sizes)

    @property
    def output_buffer_numel(self) -> int:
        return sum(self.output_split_sizes)

    def execute(self, output: Tensor, input: Tensor) -> None:
        dist.all_to_all_single(
            output[: self.output_buffer_numel],
            input[: self.input_buffer_numel],
            output_split_sizes=list(self.output_split_sizes),
            input_split_sizes=list(self.input_split_sizes),
            group=self.process_group,
        )


@dataclass(frozen=True, slots=True)
class _LocalSchedule:
    participants: tuple[int, ...] = ()
    local_participant: int = -1
    input_spans_by_parameter: tuple[tuple[_PackedSpan, ...], ...] = ()
    output_spans_by_parameter: tuple[tuple[_PackedSpan, ...], ...] = ()
    input_buffer_numel: int = 0
    output_buffer_numel: int = 0

    def execute(self, output: Tensor, input: Tensor) -> None:
        output[: self.output_buffer_numel].copy_(input[: self.input_buffer_numel])


_CommunicationSchedule = _PackedAllToAllSchedule | _LocalSchedule


@dataclass(frozen=True, slots=True)
class _RedistributionGroup:
    process_group: dist.ProcessGroup
    participants: tuple[int, ...]
    local_participant: int


@dataclass(slots=True)
class _BucketPlan(Generic[_ItemT]):
    local_items: tuple[_ItemT, ...]
    redistributed_items: tuple[_ItemT, ...]
    redistribution_plans: tuple[_RedistributionPlan, ...]
    group: _RedistributionGroup
    storage_to_compute_schedule: _CommunicationSchedule
    compute_to_storage_schedule: _CommunicationSchedule
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True, slots=True)
class _BucketPlanningResult(Generic[_ItemT]):
    plans: tuple[_BucketPlan[_ItemT], ...]
    ordered_items: tuple[_ItemT, ...]


@dataclass(slots=True)
class _BucketWork(Generic[_ItemT]):
    plan: _BucketPlan[_ItemT]
    slot: _BufferSlot
    storage_buffer: Tensor
    compute_fragment_buffer: Tensor
    forward_ready: torch.Event | None = None
    compute_done: torch.Event | None = None
    done: torch.Event | None = None


@dataclass(slots=True)
class _BufferSlot:
    storage_exchange_storage: dict[tuple[torch.device, torch.dtype], Tensor] = field(
        default_factory=dict
    )
    compute_exchange_storage: dict[tuple[torch.device, torch.dtype], Tensor] = field(
        default_factory=dict
    )
    compute_storage: dict[tuple[torch.device, torch.dtype], Tensor] = field(
        default_factory=dict
    )
    local_storage: dict[tuple[torch.device, torch.dtype], Tensor] = field(
        default_factory=dict
    )

    @staticmethod
    def _ensure_capacity(
        storage: dict[tuple[torch.device, torch.dtype], Tensor],
        *,
        numel: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        key = (device, dtype)
        buffer = storage.get(key)
        if buffer is None or buffer.numel() < numel:
            buffer = torch.empty(numel, dtype=dtype, device=device)
            storage[key] = buffer
        return buffer[:numel]

    def communication_buffers(self, plan: _BucketPlan[Any]) -> tuple[Tensor, Tensor]:
        to_compute = plan.storage_to_compute_schedule
        to_storage = plan.compute_to_storage_schedule
        return (
            self._ensure_capacity(
                self.storage_exchange_storage,
                numel=max(
                    to_compute.input_buffer_numel,
                    to_storage.output_buffer_numel,
                ),
                dtype=plan.dtype,
                device=plan.device,
            ),
            self._ensure_capacity(
                self.compute_exchange_storage,
                numel=max(
                    to_compute.output_buffer_numel,
                    to_storage.input_buffer_numel,
                ),
                dtype=plan.dtype,
                device=plan.device,
            ),
        )

    def compute_buffer(
        self,
        shape: torch.Size | tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        return self._ensure_capacity(
            self.compute_storage,
            numel=math.prod(shape),
            dtype=dtype,
            device=device,
        ).view(shape)

    def local_buffer(
        self,
        shape: torch.Size | tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        return self._ensure_capacity(
            self.local_storage,
            numel=math.prod(shape),
            dtype=dtype,
            device=device,
        ).view(shape)


@dataclass(slots=True)
class _CommunicationContext:
    device_handle: ModuleType
    transfer_stream: torch.Stream
    slots: tuple[_BufferSlot, _BufferSlot]

    @classmethod
    def create(cls, device: torch.device) -> _CommunicationContext:
        device_handle = torch.get_device_module(device)
        transfer_stream = device_handle.Stream(device=device, priority=0)
        return cls(
            device_handle=device_handle,
            transfer_stream=transfer_stream,
            slots=(_BufferSlot(), _BufferSlot()),
        )


class _BucketedRedistributionRuntime(Generic[_ItemT]):
    """Execute bucket plans with one-bucket-ahead communication prefetch.

    ``prepare`` writes a Muon input into runtime-owned scratch, ``compute``
    updates its runtime-owned input in place, and ``finalize`` consumes a
    runtime-owned result before reuse. Callbacks run under the stream selected
    by the runtime and must not retain tensors, synchronize, or call
    ``Tensor.record_stream()``.
    """

    def __init__(self, device: torch.device) -> None:
        # pyrefly: ignore [read-only]
        self._device = device
        self._context: _CommunicationContext | None = None

    def run(
        self,
        plans: Sequence[_BucketPlan[_ItemT]],
        *,
        local_tensor_spec: Callable[
            [_ItemT], tuple[torch.Size, torch.dtype, torch.device]
        ],
        prepare: Callable[[_ItemT, Tensor], None],
        compute: Callable[[_ItemT, Tensor], None],
        finalize: Callable[[_ItemT, Tensor], None],
    ) -> None:
        if self._context is None:
            self._context = _CommunicationContext.create(self._device)
        context = self._context
        handle = context.device_handle
        caller = handle.current_stream(self._device)
        context.transfer_stream.wait_stream(caller)

        previous: _BucketWork[_ItemT] | None = None
        redistributed_index = 0
        try:
            for plan in plans:
                slot = context.slots[redistributed_index % 2]
                if not plan.redistributed_items:
                    with handle.stream(caller):
                        self._compute_local(
                            plan,
                            slot,
                            local_tensor_spec=local_tensor_spec,
                            prepare=prepare,
                            compute=compute,
                            finalize=finalize,
                        )
                    continue

                work = self._begin_communication(
                    plan,
                    slot,
                    context,
                    prepare=prepare,
                )
                redistributed_index += 1
                # Keep collective launches ahead of owner-dependent work:
                # gather(current) -> return(previous) -> compute(current).
                if previous is not None:
                    self._complete(previous, context, finalize=finalize)

                self._compute_bucket(
                    work,
                    slot,
                    caller,
                    context,
                    local_tensor_spec=local_tensor_spec,
                    prepare=prepare,
                    compute=compute,
                    finalize=finalize,
                )
                if previous is not None:
                    self._release(previous, caller)
                previous = work

            if previous is not None:
                self._complete(previous, context, finalize=finalize)
                self._release(previous, caller)
        except Exception:
            # Preserve allocator lifetime ordering for work already enqueued on
            # either stream. This is an error-path drain, not synchronization.
            context.transfer_stream.wait_stream(caller)
            caller.wait_stream(context.transfer_stream)
            raise

    @staticmethod
    def _begin_communication(
        plan: _BucketPlan[_ItemT],
        slot: _BufferSlot,
        context: _CommunicationContext,
        *,
        prepare: Callable[[_ItemT, Tensor], None],
    ) -> _BucketWork[_ItemT]:
        handle = context.device_handle
        transfer = context.transfer_stream
        with handle.stream(transfer):
            storage_buffer, compute_fragment_buffer = slot.communication_buffers(plan)
            work = _BucketWork(plan, slot, storage_buffer, compute_fragment_buffer)
            _prepare_redistributed(
                plan,
                slot,
                storage_buffer,
                prepare=prepare,
            )
            plan.storage_to_compute_schedule.execute(
                output=compute_fragment_buffer,
                input=storage_buffer,
            )
            work.forward_ready = handle.Event()
            work.forward_ready.record(transfer)
        return work

    @staticmethod
    def _compute_bucket(
        work: _BucketWork[_ItemT],
        slot: _BufferSlot,
        caller_stream: torch.Stream,
        context: _CommunicationContext,
        *,
        local_tensor_spec: Callable[
            [_ItemT], tuple[torch.Size, torch.dtype, torch.device]
        ],
        prepare: Callable[[_ItemT, Tensor], None],
        compute: Callable[[_ItemT, Tensor], None],
        finalize: Callable[[_ItemT, Tensor], None],
    ) -> None:
        assert work.forward_ready is not None
        handle = context.device_handle
        with handle.stream(caller_stream):
            _BucketedRedistributionRuntime._compute_local(
                work.plan,
                slot,
                local_tensor_spec=local_tensor_spec,
                prepare=prepare,
                compute=compute,
                finalize=finalize,
            )
            caller_stream.wait_event(work.forward_ready)
            _compute_redistributed(
                work,
                slot,
                compute=compute,
            )
            work.compute_done = handle.Event()
            work.compute_done.record(caller_stream)

    @staticmethod
    def _complete(
        work: _BucketWork[_ItemT],
        context: _CommunicationContext,
        *,
        finalize: Callable[[_ItemT, Tensor], None],
    ) -> None:
        assert work.compute_done is not None
        handle = context.device_handle
        transfer = context.transfer_stream
        with handle.stream(transfer):
            transfer.wait_event(work.compute_done)
            work.plan.compute_to_storage_schedule.execute(
                output=work.storage_buffer,
                input=work.compute_fragment_buffer,
            )
            _finalize_redistributed(work, work.slot, finalize=finalize)
            work.done = handle.Event()
            work.done.record(transfer)

    @staticmethod
    def _release(work: _BucketWork[_ItemT], caller_stream: torch.Stream) -> None:
        assert work.done is not None
        caller_stream.wait_event(work.done)

    @staticmethod
    def _compute_local(
        plan: _BucketPlan[_ItemT],
        slot: _BufferSlot,
        *,
        local_tensor_spec: Callable[
            [_ItemT], tuple[torch.Size, torch.dtype, torch.device]
        ],
        prepare: Callable[[_ItemT, Tensor], None],
        compute: Callable[[_ItemT, Tensor], None],
        finalize: Callable[[_ItemT, Tensor], None],
    ) -> None:
        for item in plan.local_items:
            shape, dtype, device = local_tensor_spec(item)
            prepared = slot.compute_buffer(shape, dtype=dtype, device=device)
            prepare(item, prepared)
            compute(item, prepared)
            finalize(item, prepared)


def _prepare_redistributed(
    plan: _BucketPlan[_ItemT],
    slot: _BufferSlot,
    storage_buffer: Tensor,
    *,
    prepare: Callable[[_ItemT, Tensor], None],
) -> None:
    schedule = plan.storage_to_compute_schedule
    participant = plan.group.local_participant
    for index, (item, redistribution_plan) in enumerate(
        zip(
            plan.redistributed_items,
            plan.redistribution_plans,
            strict=True,
        )
    ):
        partition = redistribution_plan.storage_partition(participant)
        prepared = slot.local_buffer(
            partition.tensor_shape,
            dtype=plan.dtype,
            device=plan.device,
        )
        prepare(item, prepared)
        spans = schedule.input_spans_by_parameter[index]
        for span in spans:
            packed = storage_buffer[
                span.buffer_offset : span.buffer_offset + span.numel
            ]
            packed.copy_(_matrix_block_view(prepared, span.block).reshape(-1))


def _compute_redistributed(
    work: _BucketWork[_ItemT],
    slot: _BufferSlot,
    *,
    compute: Callable[[_ItemT, Tensor], None],
) -> None:
    plan = work.plan
    participant = plan.group.local_participant
    to_compute = plan.storage_to_compute_schedule
    to_storage = plan.compute_to_storage_schedule
    for index, (item, redistribution_plan) in enumerate(
        zip(
            plan.redistributed_items,
            plan.redistribution_plans,
            strict=True,
        )
    ):
        partition = redistribution_plan.compute_partition(participant)
        if not math.prod(partition.tensor_shape):
            continue
        received_spans = to_compute.output_spans_by_parameter[index]
        compute_tensor = slot.compute_buffer(
            partition.tensor_shape,
            dtype=plan.dtype,
            device=plan.device,
        )
        for span in received_spans:
            received = work.compute_fragment_buffer[
                span.buffer_offset : span.buffer_offset + span.numel
            ]
            _matrix_block_view(compute_tensor, span.block).copy_(
                received.view(span.block.shape)
            )

        compute(item, compute_tensor)

        for span in to_storage.input_spans_by_parameter[index]:
            packed = work.compute_fragment_buffer[
                span.buffer_offset : span.buffer_offset + span.numel
            ]
            packed.copy_(_matrix_block_view(compute_tensor, span.block).reshape(-1))


def _finalize_redistributed(
    work: _BucketWork[_ItemT],
    slot: _BufferSlot,
    *,
    finalize: Callable[[_ItemT, Tensor], None],
) -> None:
    plan = work.plan
    participant = plan.group.local_participant
    schedule = work.plan.compute_to_storage_schedule
    for index, (item, redistribution_plan) in enumerate(
        zip(
            plan.redistributed_items,
            plan.redistribution_plans,
            strict=True,
        )
    ):
        partition = redistribution_plan.storage_partition(participant)
        update = slot.local_buffer(
            partition.tensor_shape,
            dtype=plan.dtype,
            device=plan.device,
        )
        spans = schedule.output_spans_by_parameter[index]
        for span in spans:
            packed = work.storage_buffer[
                span.buffer_offset : span.buffer_offset + span.numel
            ]
            _matrix_block_view(update, span.block).copy_(packed.view(span.block.shape))
        finalize(item, update)


def _copy_transfers(
    routes: tuple[_MatrixBlockRoute, ...], participants: tuple[int, ...]
) -> tuple[tuple[int, int, _MatrixBlock, _MatrixBlock], ...]:
    participant_order = {
        participant: index for index, participant in enumerate(participants)
    }
    transfers = []
    for route in routes:
        sources = tuple(
            sorted(route.source_participants, key=participant_order.__getitem__)
        )
        for destination in route.destination_participants:
            source = destination if destination in sources else sources[0]
            transfers.append(
                (source, destination, route.source_block, route.destination_block)
            )
    return tuple(transfers)


def _packed_spans_by_parameter(
    indexed_spans: list[tuple[int, _PackedSpan]], parameter_count: int
) -> tuple[tuple[_PackedSpan, ...], ...]:
    return tuple(
        tuple(
            span
            for span_parameter_index, span in indexed_spans
            if span_parameter_index == parameter_index
        )
        for parameter_index in range(parameter_count)
    )


def _lower_packed_all_to_all(
    redistribution_plans: tuple[_RedistributionPlan, ...],
    *,
    storage_to_compute: bool,
    process_group: dist.ProcessGroup,
    local_participant: int,
) -> _PackedAllToAllSchedule:
    """Lower nonempty plans with one shared participant order to packed A2A."""
    participants = redistribution_plans[0].participants
    if storage_to_compute:
        routes_by_parameter = tuple(
            plan.storage_to_compute_routes for plan in redistribution_plans
        )
    else:
        routes_by_parameter = tuple(
            plan.compute_to_storage_routes for plan in redistribution_plans
        )
    transfers_by_parameter = tuple(
        _copy_transfers(routes, participants) for routes in routes_by_parameter
    )

    input_split_sizes = []
    input_spans = []
    input_cursor = 0
    for destination in participants:
        split_start = input_cursor
        for parameter_index, transfers in enumerate(transfers_by_parameter):
            for (
                source,
                transfer_destination,
                source_block,
                _destination_block,
            ) in transfers:
                if source != local_participant or transfer_destination != destination:
                    continue
                input_spans.append(
                    (parameter_index, _PackedSpan(source_block, input_cursor))
                )
                input_cursor += source_block.numel
        input_split_sizes.append(input_cursor - split_start)

    output_split_sizes = []
    output_spans = []
    output_cursor = 0
    for source in participants:
        split_start = output_cursor
        for parameter_index, transfers in enumerate(transfers_by_parameter):
            for (
                transfer_source,
                destination,
                _source_block,
                destination_block,
            ) in transfers:
                if transfer_source != source or destination != local_participant:
                    continue
                output_spans.append(
                    (parameter_index, _PackedSpan(destination_block, output_cursor))
                )
                output_cursor += destination_block.numel
        output_split_sizes.append(output_cursor - split_start)

    return _PackedAllToAllSchedule(
        process_group=process_group,
        participants=participants,
        local_participant=local_participant,
        input_split_sizes=tuple(input_split_sizes),
        output_split_sizes=tuple(output_split_sizes),
        input_spans_by_parameter=_packed_spans_by_parameter(
            input_spans, len(redistribution_plans)
        ),
        output_spans_by_parameter=_packed_spans_by_parameter(
            output_spans, len(redistribution_plans)
        ),
    )


def _device_mesh_ranks(mesh: DeviceMesh) -> tuple[int, ...]:
    # Process groups can canonicalize rank order, but DeviceMesh order defines
    # which logical shard each global rank holds.
    return tuple(mesh.mesh.flatten().tolist())


def _redistribution_group(mesh: DeviceMesh) -> _RedistributionGroup:
    process_group = mesh.get_group()
    participants = tuple(dist.get_process_group_ranks(process_group))
    return _RedistributionGroup(
        process_group=process_group,
        participants=participants,
        local_participant=participants[dist.get_rank(process_group)],
    )


def _dtensor_storage_block_for_participant(
    tensor: DTensor,
    participant: int,
) -> _MatrixBlock:
    mesh_shape = tuple(tensor.device_mesh.shape)
    mesh_rank = _device_mesh_ranks(tensor.device_mesh).index(participant)
    coordinate = [0] * len(mesh_shape)
    for mesh_dim in range(len(mesh_shape) - 1, -1, -1):
        mesh_rank, coordinate[mesh_dim] = divmod(mesh_rank, mesh_shape[mesh_dim])

    local_shape = list(tensor.shape)
    global_offsets = [0] * tensor.ndim
    for mesh_dim, placement in enumerate(tensor.placements):
        if type(placement) is not Shard:
            raise ValueError(
                "redistributed optimizer storage requires exact Shard placements"
            )
        tensor_dim = placement.dim % tensor.ndim
        local_size, global_offset = Shard.local_shard_size_and_offset(
            tensor.shape[tensor_dim],
            mesh_shape[mesh_dim],
            coordinate[mesh_dim],
        )
        local_shape[tensor_dim] = local_size
        global_offsets[tensor_dim] = global_offset
    return _MatrixBlock(
        offsets=tuple(global_offsets),
        shape=tuple(local_shape),
    )


def _dtensor_storage_blocks(
    tensor: DTensor,
    participants: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], _MatrixBlock], ...]:
    storage_participants = _device_mesh_ranks(tensor.device_mesh)
    if storage_participants != participants:
        raise ValueError(
            "bucket mesh participants must match redistributed DTensor storage"
        )
    return tuple(
        (
            (participant,),
            _dtensor_storage_block_for_participant(tensor, participant),
        )
        for participant in participants
    )


def _build_bucket_plans(
    items: Sequence[_ItemT],
    specs: Sequence[BucketSpec],
    *,
    fqn: Callable[[_ItemT], str],
    requires_owner: Callable[[_ItemT], bool],
    storage_dtensor: Callable[[_ItemT], DTensor],
    redistribution_plan: Callable[
        [_ItemT, _RedistributionGroup, int | None],
        _RedistributionPlan | None,
    ],
) -> _BucketPlanningResult[_ItemT]:
    """Build ordered local and redistributed optimizer bucket plans.

    ``redistribution_plan`` receives a mesh-local owner rank exactly when
    ``requires_owner`` is true. It returns ``None`` for compute-ready local
    storage or a transport-neutral plan for redistribution. This keeps bucket
    ordering, owner validation, dtype validation, and packed communication
    independent of a particular optimizer compute placement.
    """
    resolved = _resolve_buckets(items, specs, fqn=fqn)
    plans = []
    ordered_items = []
    for spec, bucket in zip(specs, resolved, strict=True):
        if not bucket:
            continue
        group = _redistribution_group(spec.mesh)
        sorted_bucket = tuple(sorted(bucket, key=fqn))
        expected_owners = {fqn(item) for item in sorted_bucket if requires_owner(item)}
        provided_owners = set(spec.owner_rank_by_fqn)
        missing_owners = expected_owners - provided_owners
        if missing_owners:
            raise ValueError(
                f"bucket {spec.name!r} owner assignment must exactly cover "
                "whole-tensor-owned parameters; "
                f"missing={sorted(missing_owners)}, "
                f"extra={sorted(provided_owners - expected_owners)}"
            )

        local_items_list = []
        redistributed_items_list = []
        redistribution_plans = []
        for item in sorted_bucket:
            needs_owner = requires_owner(item)
            owner_rank = spec.owner_rank_by_fqn[fqn(item)] if needs_owner else None
            if owner_rank is not None and owner_rank not in range(
                len(group.participants)
            ):
                raise ValueError(
                    f"bucket {spec.name!r} has owner outside its process group"
                )
            item_plan = redistribution_plan(item, group, owner_rank)
            if item_plan is None:
                local_items_list.append(item)
                continue
            if item_plan.participants != group.participants:
                raise ValueError(
                    f"bucket {spec.name!r} redistribution participants do not "
                    "match its process group"
                )
            local_tensor = storage_dtensor(item).to_local()
            storage_partition = item_plan.storage_partition(group.local_participant)
            if tuple(local_tensor.shape) != storage_partition.tensor_shape:
                raise ValueError(
                    f"bucket {spec.name!r} storage partition does not match its mesh"
                )
            redistributed_items_list.append(item)
            redistribution_plans.append(item_plan)

        local_items = tuple(local_items_list)
        redistributed_items = tuple(redistributed_items_list)
        # Size-one sharded storage may normalize to Replicate. In that case a
        # static rank-0 owner entry is equivalent to the resolved local compute.
        redundant_owners = {
            fqn(item)
            for item in local_items
            if len(group.participants) == 1
            and spec.owner_rank_by_fqn.get(fqn(item)) == 0
        }
        effective_provided_owners = provided_owners - redundant_owners
        if effective_provided_owners != expected_owners:
            raise ValueError(
                f"bucket {spec.name!r} owner assignment must exactly cover "
                "whole-tensor-owned parameters; "
                f"missing={sorted(expected_owners - effective_provided_owners)}, "
                f"extra={sorted(effective_provided_owners - expected_owners)}"
            )
        ordered_items.extend(local_items)
        ordered_items.extend(redistributed_items)

        if not redistributed_items:
            tensor = storage_dtensor(local_items[0]).to_local()
            plans.append(
                _BucketPlan(
                    local_items=local_items,
                    redistributed_items=(),
                    redistribution_plans=(),
                    group=group,
                    storage_to_compute_schedule=_LocalSchedule(),
                    compute_to_storage_schedule=_LocalSchedule(),
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
            )
            continue

        storage_dtensors = [storage_dtensor(item) for item in redistributed_items]
        local_tensors = [tensor.to_local() for tensor in storage_dtensors]
        dtype = local_tensors[0].dtype
        device = local_tensors[0].device
        if any(
            tensor.dtype != dtype or tensor.device != device for tensor in local_tensors
        ):
            raise ValueError(f"bucket {spec.name!r} mixes dtype or device")

        redistribution_plans_tuple = tuple(redistribution_plans)
        plans.append(
            _BucketPlan(
                local_items=local_items,
                redistributed_items=redistributed_items,
                redistribution_plans=redistribution_plans_tuple,
                group=group,
                storage_to_compute_schedule=_lower_packed_all_to_all(
                    redistribution_plans_tuple,
                    storage_to_compute=True,
                    process_group=group.process_group,
                    local_participant=group.local_participant,
                ),
                compute_to_storage_schedule=_lower_packed_all_to_all(
                    redistribution_plans_tuple,
                    storage_to_compute=False,
                    process_group=group.process_group,
                    local_participant=group.local_participant,
                ),
                dtype=dtype,
                device=device,
            )
        )

    return _BucketPlanningResult(
        plans=tuple(plans),
        ordered_items=tuple(ordered_items),
    )


def _build_owned_bucket_plans(
    items: Sequence[_ItemT],
    specs: Sequence[BucketSpec],
    *,
    fqn: Callable[[_ItemT], str],
    compute_locally: Callable[[_ItemT], bool],
    storage_dtensor: Callable[[_ItemT], DTensor],
) -> _BucketPlanningResult[_ItemT]:
    """Compatibility wrapper for local and whole-tensor-owned compute."""

    def build_plan(
        item: _ItemT,
        group: _RedistributionGroup,
        owner_rank: int | None,
    ) -> _RedistributionPlan | None:
        if compute_locally(item):
            return None
        assert owner_rank is not None
        tensor = storage_dtensor(item)
        return _build_owned_redistribution_plan(
            _dtensor_storage_blocks(tensor, group.participants),
            participants=group.participants,
            owner=group.participants[owner_rank],
            logical_shape=tuple(tensor.shape),
        )

    return _build_bucket_plans(
        items,
        specs,
        fqn=fqn,
        requires_owner=lambda item: not compute_locally(item),
        storage_dtensor=storage_dtensor,
        redistribution_plan=build_plan,
    )


def _validate_bucket_plans_across_ranks(
    plans: Sequence[_BucketPlan[_ItemT]],
    *,
    item_signature: Callable[[_ItemT], tuple[Any, ...]],
) -> None:
    """Collectively verify rank-stable plans before runtime communication.

    Every rank must provide the same plan count and process-group order so all
    workers enter these validation collectives in the same sequence.
    """
    for plan in plans:
        description = (
            str(plan.dtype),
            plan.device.type,
            tuple(
                _redistribution_plan_key(redistribution_plan)
                for redistribution_plan in plan.redistribution_plans
            ),
            [
                item_signature(item)
                for item in plan.local_items + plan.redistributed_items
            ],
        )
        digest = hashlib.sha256(repr(description).encode()).digest()
        plan_hash = int.from_bytes(digest[:7], "little")
        local_hash = torch.tensor(plan_hash, dtype=torch.int64, device=plan.device)
        process_group = plan.group.process_group
        gathered = [
            torch.empty_like(local_hash)
            for _ in range(dist.get_world_size(process_group))
        ]
        dist.all_gather(gathered, local_hash, group=process_group)
        if any(value.item() != plan_hash for value in gathered):
            raise RuntimeError("optimizer bucket plans differ across ranks")


def _matrix_block_view(tensor: Tensor, block: _MatrixBlock) -> Tensor:
    view = tensor[
        tuple(
            slice(offset, offset + size)
            for offset, size in zip(block.offsets, block.shape, strict=True)
        )
    ]
    return view


def _redistribution_plan_key(plan: _RedistributionPlan) -> tuple[Any, ...]:
    def partition_key(partition: _ParticipantPartition) -> tuple[Any, ...]:
        return (
            partition.participant,
            partition.tensor_shape,
            tuple((block.offsets, block.shape) for block in partition.logical_blocks),
        )

    def route_key(route: _MatrixBlockRoute) -> tuple[Any, ...]:
        return (
            route.logical_block.offsets,
            route.logical_block.shape,
            route.source_block.offsets,
            route.source_block.shape,
            route.destination_block.offsets,
            route.destination_block.shape,
            route.source_participants,
            route.destination_participants,
        )

    return (
        plan.participants,
        plan.logical_shape,
        tuple(map(partition_key, plan.storage_partitions)),
        tuple(map(partition_key, plan.compute_partitions)),
        tuple(map(route_key, plan.storage_to_compute_routes)),
        tuple(map(route_key, plan.compute_to_storage_routes)),
    )
