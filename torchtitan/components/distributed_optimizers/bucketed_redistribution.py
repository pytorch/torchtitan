# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Private bucketed storage-to-compute runtime for DistributedMuon."""

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

    ``mesh_axes`` selects an ordered storage submesh. Multiple axes are
    flattened into the one-dimensional communication mesh used by the bucket.
    """

    patterns: tuple[str, ...]
    owner_rank_by_fqn: Mapping[str, int]
    mesh_axes: tuple[str, ...]
    name: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.mesh_axes, str) or not self.mesh_axes:
            raise ValueError("mesh_axes must be a non-empty sequence of axis names")
        object.__setattr__(self, "patterns", tuple(self.patterns))
        object.__setattr__(self, "owner_rank_by_fqn", dict(self.owner_rank_by_fqn))
        object.__setattr__(self, "mesh_axes", tuple(self.mesh_axes))

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
    owner entry. ``name`` is diagnostic metadata only.
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


def _bind_bucket_configs(
    configs: Sequence[BucketConfig],
    storage_by_fqn: Mapping[str, DTensor],
) -> tuple[BucketSpec, ...]:
    specs = []
    for config in configs:
        candidates = tuple(config.owner_rank_by_fqn) or tuple(
            fqn
            for fqn in storage_by_fqn
            if any(fnmatch.fnmatchcase(fqn, pattern) for pattern in config.patterns)
        )
        if not candidates:
            raise ValueError(f"bucket {config.name!r} matched no storage tensor")

        meshes = []
        for fqn in candidates:
            if fqn not in storage_by_fqn:
                raise ValueError(f"bucket {config.name!r} references unknown {fqn!r}")
            storage_mesh = storage_by_fqn[fqn].device_mesh
            if storage_mesh.mesh_dim_names is None or any(
                axis not in storage_mesh.mesh_dim_names
                for axis in config.mesh_axes
            ):
                raise ValueError(
                    f"bucket {config.name!r} mesh axes {config.mesh_axes!r} "
                    f"are not present on storage for {fqn!r}"
                )
            storage_axis_order = tuple(
                axis
                for axis in storage_mesh.mesh_dim_names
                if axis in config.mesh_axes
            )
            if storage_axis_order != config.mesh_axes:
                raise ValueError(
                    f"bucket {config.name!r} mesh axes must follow storage "
                    f"order {storage_mesh.mesh_dim_names!r}"
                )
            selected_mesh = storage_mesh[config.mesh_axes]
            meshes.append(
                selected_mesh._flatten()
                if selected_mesh.ndim > 1
                else selected_mesh
            )

        mesh = meshes[0]
        if any(not torch.equal(candidate.mesh, mesh.mesh) for candidate in meshes[1:]):
            raise ValueError(
                f"bucket {config.name!r} resolves to inconsistent communication meshes"
            )
        specs.append(config.bind(mesh))
    return tuple(specs)


def assign_balanced_owners(
    bucket_fqns: Sequence[Sequence[str]],
    memory_estimate_by_fqn: Mapping[str, int],
    *,
    num_ranks: int,
    initial_memory_by_rank: Sequence[int] | None = None,
) -> tuple[dict[str, int], ...]:
    """Greedily balance selected parameters across group-local ranks."""
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
            heapq.heappush(
                rank_loads, (load + memory_estimate_by_fqn[fqn], rank)
            )
        owners_by_bucket.append(bucket_owners)
    return tuple(owners_by_bucket)


_ItemT = TypeVar("_ItemT")


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
            raise ValueError(
                f"optimizer parameter {name!r} must match one bucket"
            )
        resolved[matches[0]].append(item)
    return tuple(tuple(bucket) for bucket in resolved)


@dataclass(frozen=True, slots=True)
class _MatrixBlock:
    """A rectangular logical compute unit, independent of placement."""

    offsets: tuple[int, ...]
    shape: tuple[int, ...]

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True, slots=True)
class _MatrixBlockRoute:
    """Map one logical block from storage holders to compute holders."""

    block: _MatrixBlock
    source_participants: tuple[int, ...]
    destination_participants: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _RedistributionPlan:
    """Transport-neutral exact block partitions in both directions."""

    participants: tuple[int, ...]
    logical_shape: tuple[int, ...]
    storage_to_compute_routes: tuple[_MatrixBlockRoute, ...]
    compute_to_storage_routes: tuple[_MatrixBlockRoute, ...]

    def __post_init__(self) -> None:
        all_routes = self.storage_to_compute_routes + self.compute_to_storage_routes
        if any(
            not route.source_participants or not route.destination_participants
            for route in all_routes
        ):
            raise ValueError("redistribution routes require sources and destinations")
        for direction, routes in (
            ("storage-to-compute", self.storage_to_compute_routes),
            ("compute-to-storage", self.compute_to_storage_routes),
        ):
            _validate_matrix_block_partition(
                tuple(route.block for route in routes),
                self.logical_shape,
                direction=direction,
            )

        compute_destinations = {
            destination
            for route in self.storage_to_compute_routes
            for destination in route.destination_participants
        }
        for destination in compute_destinations:
            _validate_matrix_block_partition(
                tuple(
                    route.block
                    for route in self.storage_to_compute_routes
                    if destination in route.destination_participants
                ),
                self.logical_shape,
                direction=f"compute destination {destination}",
            )
        if any(
            source not in compute_destinations
            for route in self.compute_to_storage_routes
            for source in route.source_participants
        ):
            raise ValueError("compute-to-storage source has no complete compute tensor")


def _validate_matrix_block_partition(
    blocks: tuple[_MatrixBlock, ...],
    logical_shape: tuple[int, ...],
    *,
    direction: str,
) -> None:
    if any(size < 0 for size in logical_shape):
        raise ValueError("logical tensor shape must be nonnegative")
    for block in blocks:
        if len(block.offsets) != len(logical_shape) or len(block.shape) != len(
            logical_shape
        ):
            raise ValueError(f"{direction} block rank does not match logical tensor")
        if any(
            offset < 0 or size < 0 or offset + size > logical_size
            for offset, size, logical_size in zip(
                block.offsets, block.shape, logical_shape, strict=True
            )
        ):
            raise ValueError(f"{direction} block is outside the logical tensor")

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


def _build_owned_redistribution_plan(
    storage_blocks: Sequence[tuple[tuple[int, ...], _MatrixBlock]],
    *,
    participants: tuple[int, ...],
    owner: int,
    logical_shape: tuple[int, ...],
) -> _RedistributionPlan:
    """Build mirrored routes from one canonical block-to-holders mapping."""
    return _RedistributionPlan(
        participants=participants,
        logical_shape=logical_shape,
        storage_to_compute_routes=tuple(
            _MatrixBlockRoute(
                block=block,
                source_participants=holders,
                destination_participants=(owner,),
            )
            for holders, block in storage_blocks
        ),
        compute_to_storage_routes=tuple(
            _MatrixBlockRoute(
                block=block,
                source_participants=(owner,),
                destination_participants=holders,
            )
            for holders, block in storage_blocks
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


class _CommunicationSchedule:
    """Physical execution strategy produced from redistribution routes."""

    __slots__ = ()
    participants: tuple[int, ...]
    local_participant: int
    input_spans_by_parameter: tuple[tuple[_PackedSpan, ...], ...]
    output_spans_by_parameter: tuple[tuple[_PackedSpan, ...], ...]
    input_buffer_numel: int
    output_buffer_numel: int

    def execute(
        self, output: Tensor, input: Tensor
    ) -> None:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class _PackedAllToAllSchedule(_CommunicationSchedule):
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

    def execute(
        self, output: Tensor, input: Tensor
    ) -> None:
        dist.all_to_all_single(
            output[: self.output_buffer_numel],
            input[: self.input_buffer_numel],
            output_split_sizes=list(self.output_split_sizes),
            input_split_sizes=list(self.input_split_sizes),
            group=self.process_group,
        )


@dataclass(frozen=True, slots=True)
class _LocalSchedule(_CommunicationSchedule):
    participants: tuple[int, ...] = ()
    local_participant: int = -1
    input_spans_by_parameter: tuple[tuple[_PackedSpan, ...], ...] = ()
    output_spans_by_parameter: tuple[tuple[_PackedSpan, ...], ...] = ()
    input_buffer_numel: int = 0
    output_buffer_numel: int = 0

    def execute(
        self, output: Tensor, input: Tensor
    ) -> None:
        if self.input_buffer_numel != self.output_buffer_numel:
            raise ValueError("local schedules require equal buffer sizes")
        output[: self.output_buffer_numel].copy_(
            input[: self.input_buffer_numel]
        )


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
class _RedistributionGroup:
    process_group: dist.ProcessGroup
    participants: tuple[int, ...]
    local_participant: int


@dataclass(frozen=True, slots=True)
class _BucketPlanningResult(Generic[_ItemT]):
    plans: tuple[_BucketPlan[_ItemT], ...]
    ordered_items: tuple[_ItemT, ...]


@dataclass(slots=True)
class _BucketWork(Generic[_ItemT]):
    plan: _BucketPlan[_ItemT]
    storage_buffer: Tensor
    compute_fragment_buffer: Tensor
    forward_ready: torch.Event | None = None
    compute_done: torch.Event | None = None
    done: torch.Event | None = None


@dataclass(slots=True)
class _BufferSlot:
    storage_exchange_storage: dict[
        tuple[torch.device, torch.dtype], Tensor
    ] = field(default_factory=dict)
    compute_exchange_storage: dict[
        tuple[torch.device, torch.dtype], Tensor
    ] = field(default_factory=dict)
    compute_storage: dict[tuple[torch.device, torch.dtype], Tensor] = field(
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

    def communication_buffers(
        self, plan: _BucketPlan[Any]
    ) -> tuple[Tensor, Tensor]:
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

    Callbacks run under the stream selected by the runtime. They must enqueue
    work without synchronizing or calling ``Tensor.record_stream()``.
    """

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._context: _CommunicationContext | None = None

    def run(
        self,
        plans: Sequence[_BucketPlan[_ItemT]],
        *,
        local_tensor_spec: Callable[
            [_ItemT], tuple[torch.Size, torch.dtype, torch.device]
        ],
        compute_shape: Callable[[_ItemT], torch.Size],
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

        pending: list[_BucketWork[_ItemT]] = []
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
                work = self._begin(
                    plan,
                    slot,
                    caller,
                    context,
                    local_tensor_spec=local_tensor_spec,
                    compute_shape=compute_shape,
                    prepare=prepare,
                    compute=compute,
                    finalize=finalize,
                )
                redistributed_index += 1
                pending.append(work)
                if len(pending) == 2:
                    oldest = pending.pop(0)
                    self._complete(oldest, context, finalize=finalize)
                    self._release(oldest, caller)
            for work in pending:
                self._complete(work, context, finalize=finalize)
                self._release(work, caller)
        except Exception:
            # Preserve allocator lifetime ordering for work already enqueued on
            # either stream. This is an error-path drain, not synchronization.
            context.transfer_stream.wait_stream(caller)
            caller.wait_stream(context.transfer_stream)
            raise

    @staticmethod
    def _begin(
        plan: _BucketPlan[_ItemT],
        slot: _BufferSlot,
        caller_stream: torch.Stream,
        context: _CommunicationContext,
        *,
        local_tensor_spec: Callable[
            [_ItemT], tuple[torch.Size, torch.dtype, torch.device]
        ],
        compute_shape: Callable[[_ItemT], torch.Size],
        prepare: Callable[[_ItemT, Tensor], None],
        compute: Callable[[_ItemT, Tensor], None],
        finalize: Callable[[_ItemT, Tensor], None],
    ) -> _BucketWork[_ItemT]:
        handle = context.device_handle
        transfer = context.transfer_stream
        with handle.stream(transfer):
            storage_buffer, compute_fragment_buffer = slot.communication_buffers(
                plan
            )
            work = _BucketWork(plan, storage_buffer, compute_fragment_buffer)
            _prepare_redistributed(plan, storage_buffer, prepare=prepare)
            plan.storage_to_compute_schedule.execute(
                output=compute_fragment_buffer,
                input=storage_buffer,
            )
            work.forward_ready = handle.Event()
            work.forward_ready.record(transfer)

        with handle.stream(caller_stream):
            _BucketedRedistributionRuntime._compute_local(
                plan,
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
                compute_shape=compute_shape,
                compute=compute,
            )
            work.compute_done = handle.Event()
            work.compute_done.record(caller_stream)
        return work

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
            _finalize_redistributed(work, finalize=finalize)
            work.done = handle.Event()
            work.done.record(transfer)

    @staticmethod
    def _release(
        work: _BucketWork[_ItemT], caller_stream: torch.Stream
    ) -> None:
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
    storage_buffer: Tensor,
    *,
    prepare: Callable[[_ItemT, Tensor], None],
) -> None:
    schedule = plan.storage_to_compute_schedule
    for index, item in enumerate(plan.redistributed_items):
        spans = schedule.input_spans_by_parameter[index]
        assert len(spans) == 1
        span = spans[0]
        out = storage_buffer[
            span.buffer_offset : span.buffer_offset + span.numel
        ].view(span.block.shape)
        prepare(item, out)


def _compute_redistributed(
    work: _BucketWork[_ItemT],
    slot: _BufferSlot,
    *,
    compute_shape: Callable[[_ItemT], torch.Size],
    compute: Callable[[_ItemT, Tensor], None],
) -> None:
    plan = work.plan
    to_compute = plan.storage_to_compute_schedule
    to_storage = plan.compute_to_storage_schedule
    for index, item in enumerate(plan.redistributed_items):
        received_spans = to_compute.output_spans_by_parameter[index]
        if not received_spans:
            continue
        compute_tensor = slot.compute_buffer(
            compute_shape(item),
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
            packed.view(span.block.shape).copy_(
                _matrix_block_view(compute_tensor, span.block)
            )


def _finalize_redistributed(
    work: _BucketWork[_ItemT],
    *,
    finalize: Callable[[_ItemT, Tensor], None],
) -> None:
    schedule = work.plan.compute_to_storage_schedule
    for index, item in enumerate(work.plan.redistributed_items):
        spans = schedule.output_spans_by_parameter[index]
        assert len(spans) == 1
        span = spans[0]
        update = work.storage_buffer[
            span.buffer_offset : span.buffer_offset + span.numel
        ].view(span.block.shape)
        finalize(item, update)


def _copy_transfers(
    routes: tuple[_MatrixBlockRoute, ...], participants: tuple[int, ...]
) -> tuple[tuple[int, int, _MatrixBlock], ...]:
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
            transfers.append((source, destination, route.block))
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
    direction: str,
    process_group: dist.ProcessGroup,
    local_participant: int,
) -> _PackedAllToAllSchedule:
    participants = redistribution_plans[0].participants
    if any(plan.participants != participants for plan in redistribution_plans):
        raise ValueError("one all-to-all schedule requires one participant order")
    if tuple(dist.get_process_group_ranks(process_group)) != participants:
        raise ValueError(
            "redistribution participants must match process-group rank order"
        )
    if local_participant not in participants:
        raise ValueError("local rank is not a redistribution participant")
    if direction == "storage_to_compute":
        routes_by_parameter = tuple(
            plan.storage_to_compute_routes for plan in redistribution_plans
        )
    elif direction == "compute_to_storage":
        routes_by_parameter = tuple(
            plan.compute_to_storage_routes for plan in redistribution_plans
        )
    else:
        raise ValueError(f"unsupported redistribution direction {direction!r}")
    transfers_by_parameter = tuple(
        _copy_transfers(routes, participants) for routes in routes_by_parameter
    )

    input_split_sizes = []
    input_spans = []
    input_cursor = 0
    for destination in participants:
        split_start = input_cursor
        for parameter_index, transfers in enumerate(transfers_by_parameter):
            for source, transfer_destination, block in transfers:
                if source != local_participant or transfer_destination != destination:
                    continue
                input_spans.append(
                    (parameter_index, _PackedSpan(block, input_cursor))
                )
                input_cursor += block.numel
        input_split_sizes.append(input_cursor - split_start)

    output_split_sizes = []
    output_spans = []
    output_cursor = 0
    for source in participants:
        split_start = output_cursor
        for parameter_index, transfers in enumerate(transfers_by_parameter):
            for transfer_source, destination, block in transfers:
                if transfer_source != source or destination != local_participant:
                    continue
                output_spans.append(
                    (parameter_index, _PackedSpan(block, output_cursor))
                )
                output_cursor += block.numel
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
    if mesh.ndim == 1:
        return tuple(dist.get_process_group_ranks(mesh.get_group()))
    return tuple(mesh.mesh.flatten().tolist())


def _redistribution_group(mesh: DeviceMesh) -> _RedistributionGroup:
    if mesh.ndim != 1:
        raise ValueError("optimizer redistribution mesh must be one-dimensional")
    process_group = mesh.get_group()
    participants = tuple(dist.get_process_group_ranks(process_group))
    return _RedistributionGroup(
        process_group=process_group,
        participants=participants,
        local_participant=participants[dist.get_rank(process_group)],
    )


def _normalize_dim(dim: int, ndim: int) -> int:
    normalized = dim if dim >= 0 else dim + ndim
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"dimension {dim} is invalid for a rank-{ndim} tensor")
    return normalized


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
        tensor_dim = _normalize_dim(placement.dim, tensor.ndim)
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


def _build_owned_bucket_plans(
    items: Sequence[_ItemT],
    specs: Sequence[BucketSpec],
    *,
    fqn: Callable[[_ItemT], str],
    compute_locally: Callable[[_ItemT], bool],
    storage_dtensor: Callable[[_ItemT], DTensor],
) -> _BucketPlanningResult[_ItemT]:
    """Build the local and whole-matrix-owned DistributedMuon plans.

    The active planner supports Replicate -> Replicate and Shard(0) matrix
    batches as local compute, plus Shard(...) -> Owned through packed
    all-to-all and Owned -> Shard(...) through reverse packed all-to-all.
    Other placement transitions are intentionally unsupported.
    """
    resolved = _resolve_buckets(items, specs, fqn=fqn)
    plans = []
    ordered_items = []
    for spec, bucket in zip(specs, resolved, strict=True):
        if not bucket:
            continue
        group = _redistribution_group(spec.mesh)
        local_items = tuple(
            sorted(
                (item for item in bucket if compute_locally(item)),
                key=fqn,
            )
        )
        redistributed_items = tuple(
            sorted(
                (item for item in bucket if not compute_locally(item)),
                key=fqn,
            )
        )
        expected_owners = {fqn(item) for item in redistributed_items}
        provided_owners = set(spec.owner_rank_by_fqn)
        if provided_owners != expected_owners:
            raise ValueError(
                f"bucket {spec.name!r} owner assignment must exactly cover "
                "whole-tensor-owned parameters; "
                f"missing={sorted(expected_owners - provided_owners)}, "
                f"extra={sorted(provided_owners - expected_owners)}"
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

        owner_ranks = [
            spec.owner_rank_by_fqn[fqn(item)] for item in redistributed_items
        ]
        if any(rank not in range(len(group.participants)) for rank in owner_ranks):
            raise ValueError(
                f"bucket {spec.name!r} has owner outside its process group"
            )

        storage_dtensors = [storage_dtensor(item) for item in redistributed_items]
        local_tensors = [tensor.to_local() for tensor in storage_dtensors]
        dtype = local_tensors[0].dtype
        device = local_tensors[0].device
        if any(
            tensor.dtype != dtype or tensor.device != device
            for tensor in local_tensors
        ):
            raise ValueError(f"bucket {spec.name!r} mixes dtype or device")

        blocks_by_item = tuple(
            _dtensor_storage_blocks(tensor, group.participants)
            for tensor in storage_dtensors
        )
        for tensor, blocks in zip(local_tensors, blocks_by_item, strict=True):
            local_blocks = [
                block
                for holders, block in blocks
                if group.local_participant in holders
            ]
            if len(local_blocks) != 1 or tuple(tensor.shape) != local_blocks[0].shape:
                raise ValueError(
                    f"bucket {spec.name!r} storage block does not match its mesh"
                )

        redistribution_plans = tuple(
            _build_owned_redistribution_plan(
                blocks,
                participants=group.participants,
                owner=group.participants[owner_rank],
                logical_shape=tuple(tensor.shape),
            )
            for tensor, blocks, owner_rank in zip(
                storage_dtensors, blocks_by_item, owner_ranks, strict=True
            )
        )
        plans.append(
            _BucketPlan(
                local_items=local_items,
                redistributed_items=redistributed_items,
                redistribution_plans=redistribution_plans,
                group=group,
                storage_to_compute_schedule=_lower_packed_all_to_all(
                    redistribution_plans,
                    direction="storage_to_compute",
                    process_group=group.process_group,
                    local_participant=group.local_participant,
                ),
                compute_to_storage_schedule=_lower_packed_all_to_all(
                    redistribution_plans,
                    direction="compute_to_storage",
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


def _validate_bucket_plans_across_ranks(
    plans: Sequence[_BucketPlan[_ItemT]],
    *,
    item_signature: Callable[[_ItemT], tuple[Any, ...]],
) -> None:
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
    assert tuple(view.shape) == block.shape
    return view


def _redistribution_plan_key(plan: _RedistributionPlan) -> tuple[Any, ...]:
    def route_key(route: _MatrixBlockRoute) -> tuple[Any, ...]:
        return (
            route.block.offsets,
            route.block.shape,
            route.source_participants,
            route.destination_participants,
        )

    return (
        plan.participants,
        plan.logical_shape,
        tuple(map(route_key, plan.storage_to_compute_routes)),
        tuple(map(route_key, plan.compute_to_storage_routes)),
    )
