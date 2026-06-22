# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import heapq
import math
import os

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
from typing_extensions import override

from ..flex_shard.placement_contract import (
    BucketParamStorageLayout,
    BucketStorageLayout,
    Placement,
    PlacementPreparedReduceGrad,
    PlacementPreparedUnshard,
    PlacementReduceGradResult,
    PlacementUnshardResult,
)
from ..flex_shard.reduce_policy import (
    dist_reduce_op,
    gradient_reduce_op_from_infos,
    GradientReduceOp,
)
from ..flex_shard.utils import (
    _record_comm_if_eager,
    _record_copy_in_if_eager,
    _record_copy_out_if_eager,
    _record_function_if_eager,
    _record_view_out_if_eager,
)
from ._pack_utils import (
    foreach_copy_,
    pack_tensors_into_flat_buffer,
    pack_tensors_into_flat_buffer_with_scratch,
    try_pack_segments_into_flat_buffer_triton,
)

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from ..flex_shard.bucket_storage import ParamInfo, PlacementFn


@dataclass
class _ReduceScratchSlot:
    send: torch.Tensor
    in_use: bool = False
    ready_event: torch.Event | None = None


@dataclass
class _ReduceScratchLease:
    _pool: _ReduceScratchPool
    _slot: _ReduceScratchSlot
    send: torch.Tensor
    _released: bool = False

    def release(self, stream: torch.Stream | None = None) -> None:
        """Mark this lease reusable after queued work on ``stream`` is complete."""
        if self._released:
            return
        self._released = True
        self._pool._release(self._slot, stream)


class _ReduceScratchPool:
    """Bounded reusable scratch for GroupedOwned packed reduce-scatter sends.

    Phase 2a deliberately stops at bounding temporary send buffers with explicit
    stream lifetime. Defer direct owner-layout and optimizer aliasing until the
    current Triton packed RS path is accepted and profiler/memory evidence shows
    temporary memory is the limiting issue.
    """

    def __init__(self, max_slots: int = 1) -> None:
        if max_slots <= 0:
            raise ValueError(f"max_slots must be positive, but got {max_slots}.")
        self.max_slots = max_slots
        self._slots: list[_ReduceScratchSlot] = []

    def acquire(
        self,
        *,
        send_numel: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> _ReduceScratchLease:
        """Acquire a reusable send buffer sized for one reduce-scatter call."""
        if send_numel < 0:
            raise ValueError(f"Scratch size must be non-negative, got {send_numel}.")
        device = torch.device(device)
        slot = self._find_ready_slot(send_numel, dtype, device)
        if slot is None and len(self._slots) < self.max_slots:
            slot = self._make_slot(send_numel, dtype, device)
            self._slots.append(slot)
        if slot is None:
            slot = self._wait_for_released_slot(send_numel, dtype, device)

        slot.in_use = True
        slot.ready_event = None
        return _ReduceScratchLease(self, slot, slot.send.narrow(0, 0, send_numel))

    def _find_ready_slot(
        self,
        send_numel: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> _ReduceScratchSlot | None:
        fallback: _ReduceScratchSlot | None = None
        for slot in self._slots:
            if slot.in_use or not self._is_ready(slot):
                continue
            if self._slot_can_satisfy(slot, send_numel, dtype, device):
                return slot
            fallback = fallback or slot
        if fallback is None:
            return None
        self._resize_slot(fallback, send_numel, dtype, device)
        return fallback

    def _wait_for_released_slot(
        self,
        send_numel: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> _ReduceScratchSlot:
        for slot in self._slots:
            if not slot.in_use:
                if slot.ready_event is not None:
                    slot.ready_event.synchronize()
                    slot.ready_event = None
                self._resize_slot(slot, send_numel, dtype, device)
                return slot
        raise RuntimeError(
            "No released GroupedOwned reduce scratch slot is available. "
            "Call _ReduceScratchLease.release() before launching more than "
            "max_slots in-flight reduce-scatter operations."
        )

    @staticmethod
    def _make_slot(
        send_numel: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> _ReduceScratchSlot:
        return _ReduceScratchSlot(
            send=torch.empty(send_numel, dtype=dtype, device=device),
        )

    @staticmethod
    def _slot_can_satisfy(
        slot: _ReduceScratchSlot,
        send_numel: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> bool:
        return (
            slot.send.dtype == dtype
            and slot.send.device == device
            and slot.send.numel() >= send_numel
        )

    @staticmethod
    def _resize_slot(
        slot: _ReduceScratchSlot,
        send_numel: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if slot.send.dtype != dtype or slot.send.device != device:
            slot.send = torch.empty(send_numel, dtype=dtype, device=device)
        elif slot.send.numel() < send_numel:
            slot.send = torch.empty(send_numel, dtype=dtype, device=device)

    @staticmethod
    def _is_ready(slot: _ReduceScratchSlot) -> bool:
        if slot.ready_event is None:
            return True
        if slot.ready_event.query():
            slot.ready_event = None
            return True
        return False

    def _release(
        self,
        slot: _ReduceScratchSlot,
        stream: torch.Stream | None,
    ) -> None:
        slot.in_use = False
        if slot.send.device.type != "cuda":
            slot.ready_event = None
            return
        if stream is None:
            stream = torch.cuda.current_stream(slot.send.device)
        event = torch.cuda.Event()
        event.record(stream)
        slot.ready_event = event


def _grouped_owned_reduce_scratch_slots() -> int:
    raw = os.environ.get("FLEX_SHARD_GROUPED_OWNED_SCRATCH_SLOTS")
    if raw is not None:
        try:
            return max(1, int(raw))
        except ValueError:
            return 1

    raw = os.environ.get("FLEX_SHARD_MAX_PENDING_REDUCE_GRADS")
    if raw is None:
        return 1
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


_GROUPED_OWNED_REDUCE_SCRATCH_POOL: _ReduceScratchPool | None = None
_GROUPED_OWNED_REDUCE_SCRATCH_POOL_SLOTS: int | None = None


def _grouped_owned_reduce_scratch_pool() -> _ReduceScratchPool:
    global _GROUPED_OWNED_REDUCE_SCRATCH_POOL
    global _GROUPED_OWNED_REDUCE_SCRATCH_POOL_SLOTS

    slots = _grouped_owned_reduce_scratch_slots()
    if (
        _GROUPED_OWNED_REDUCE_SCRATCH_POOL is None
        or _GROUPED_OWNED_REDUCE_SCRATCH_POOL_SLOTS != slots
    ):
        _GROUPED_OWNED_REDUCE_SCRATCH_POOL = _ReduceScratchPool(max_slots=slots)
        _GROUPED_OWNED_REDUCE_SCRATCH_POOL_SLOTS = slots
    return _GROUPED_OWNED_REDUCE_SCRATCH_POOL


class Owned(Placement):
    """Placement where one rank owns the full parameter and other ranks hold empty tensors."""

    @dataclass(frozen=True)
    class _UnshardState:
        infos: list[ParamInfo]
        offsets: list[int]
        numels: list[int]
        pg: Any
        debug_fqn: str | None

    @dataclass(frozen=True)
    class _ReduceGradState:
        infos: list[ParamInfo]
        offsets: list[int]
        numels: list[int]
        rank: int
        world_size: int
        pg: Any
        debug_fqn: str | None
        gradient_reduce_op: GradientReduceOp

    def __init__(self, owner_rank: int):
        self.owner_rank = owner_rank

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Owned):
            return False
        return self.owner_rank == other.owner_rank

    def __hash__(self) -> int:
        return hash((type(self), self.owner_rank))

    def __repr__(self) -> str:
        return f"Owned({self.owner_rank})"

    def _validate_owner_rank(self, world_size: int) -> None:
        if self.owner_rank < 0 or self.owner_rank >= world_size:
            raise ValueError(
                f"Owned placement requires owner_rank in [0, {world_size}), "
                f"but got {self.owner_rank}."
            )

    def _try_get_contiguous_flat_bucket_view(
        self,
        tensors: list[torch.Tensor],
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Return a flat alias when bucket tensors are already contiguous."""
        if not tensors:
            return None

        device = tensors[0].device
        non_empty_tensors = [tensor for tensor in tensors if tensor.numel() > 0]
        if not non_empty_tensors:
            tensor = tensors[0]
            if tensor.dtype != dtype or tensor.device != device:
                return None
            return tensor.reshape(-1)

        first_tensor = non_empty_tensors[0]
        storage_data_ptr = first_tensor.untyped_storage().data_ptr()
        expected_storage_offset = first_tensor.storage_offset()
        total_numel = 0
        for tensor in tensors:
            numel = tensor.numel()
            if numel == 0:
                continue
            if tensor.dtype != dtype or tensor.device != device:
                return None
            if not tensor.is_contiguous():
                return None
            if tensor.untyped_storage().data_ptr() != storage_data_ptr:
                return None
            if tensor.storage_offset() != expected_storage_offset:
                return None
            expected_storage_offset += numel
            total_numel += numel

        return torch.as_strided(
            first_tensor,
            (total_numel,),
            (1,),
            storage_offset=first_tensor.storage_offset(),
        )

    def _require_uniform_dtype(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        attr: str,
        op_name: str,
    ) -> torch.dtype:
        if not infos or not tensors:
            raise AssertionError(f"Expected at least one tensor for {op_name}.")
        dtype = getattr(infos[0], attr, tensors[0].dtype)
        for tensor, info in zip(tensors[1:], infos[1:], strict=True):
            other = getattr(info, attr, tensor.dtype)
            if other != dtype:
                raise ValueError(
                    f"Owned {op_name} requires one communication dtype per bucket, "
                    f"but {infos[0].fqn!r} uses {dtype} and {info.fqn!r} uses "
                    f"{other}."
                )
        return dtype

    def _flat_layout(
        self,
        infos: list[ParamInfo],
    ) -> tuple[list[int], list[int], int]:
        numels = [math.prod(info.global_shape) for info in infos]
        offsets: list[int] = []
        total_numel = 0
        for numel in numels:
            offsets.append(total_numel)
            total_numel += numel
        return offsets, numels, total_numel

    def _views_from_flat(
        self,
        flat: torch.Tensor,
        infos: list[ParamInfo],
        offsets: list[int],
        numels: list[int],
    ) -> list[torch.Tensor]:
        if len(infos) != len(offsets) or len(infos) != len(numels):
            raise AssertionError("Expected flat bucket metadata to match infos.")
        return [
            flat.narrow(0, offset, numel).view(info.global_shape)
            for info, offset, numel in zip(infos, offsets, numels, strict=True)
        ]

    @override
    def compute_local_shape(
        self, global_shape: torch.Size, rank: int, world_size: int
    ) -> torch.Size:
        self._validate_owner_rank(world_size)
        if rank == self.owner_rank:
            return global_shape
        return torch.Size([0] * len(global_shape))

    @override
    def compute_local_numel(
        self, global_shape: torch.Size, rank: int, world_size: int
    ) -> int:
        self._validate_owner_rank(world_size)
        if rank == self.owner_rank:
            return math.prod(global_shape)
        return 0

    @override
    def extract_local_shard(
        self,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        self._validate_owner_rank(world_size)
        if rank == self.owner_rank:
            return param
        return param.new_empty(0)

    @override
    def prepare_unshard_bucket(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedUnshard:
        """Prepare one flat full-parameter buffer for owner-rank broadcast."""
        rank = mesh.get_local_rank()
        self._validate_owner_rank(mesh.size())
        if len(tensors) != len(infos):
            raise AssertionError(
                f"Expected {len(tensors)} tensors to match {len(infos)} infos."
            )
        dtype = self._require_uniform_dtype(
            tensors,
            infos,
            "unsharded_dtype",
            "unshard",
        )
        offsets, numels, total_numel = self._flat_layout(infos)

        with _record_copy_in_if_eager():
            copy_in_scratch: list[torch.Tensor] = []
            if rank == self.owner_rank:
                flat = None
                if not torch.compiler.is_compiling():
                    flat = self._try_get_contiguous_flat_bucket_view(tensors, dtype)
                if flat is None:
                    flat, copy_in_scratch = pack_tensors_into_flat_buffer_with_scratch(
                        tensors,
                        dtype,
                    )
            else:
                flat = torch.empty(total_numel, dtype=dtype, device=tensors[0].device)
            if flat.numel() != total_numel:
                raise AssertionError(
                    f"Owned unshard flat buffer has {flat.numel()} elements, "
                    f"expected {total_numel}."
                )
        return PlacementPreparedUnshard(
            placement=self,
            buffers=[flat, *copy_in_scratch],
            placement_state=Owned._UnshardState(
                infos=infos,
                offsets=offsets,
                numels=numels,
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
            ),
        )

    @override
    def run_prepared_unshard(self, prepared: PlacementPreparedUnshard) -> None:
        """Broadcast the prepared flat full-parameter buffer from the owner rank."""
        if not isinstance(prepared.placement_state, Owned._UnshardState):
            raise AssertionError(
                "Expected Owned._UnshardState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        if len(prepared.buffers) < 1:
            raise AssertionError(
                f"Expected at least one flat unshard buffer, got {len(prepared.buffers)}."
            )
        with _record_comm_if_eager(
            "FlexShard::broadcast",
            prepared.placement_state.debug_fqn,
        ):
            dist.broadcast(
                prepared.buffers[0],
                src=self.owner_rank,
                group=prepared.placement_state.pg,
            )

    @override
    def finish_prepared_unshard(
        self,
        prepared: PlacementPreparedUnshard,
    ) -> PlacementUnshardResult:
        """Return full-parameter views produced by the owner-rank broadcast."""
        if not isinstance(prepared.placement_state, Owned._UnshardState):
            raise AssertionError(
                "Expected Owned._UnshardState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        if len(prepared.buffers) < 1:
            raise AssertionError(
                f"Expected at least one flat unshard buffer, got {len(prepared.buffers)}."
            )
        flat = prepared.buffers[0]
        full_params = self._views_from_flat(
            flat,
            prepared.placement_state.infos,
            prepared.placement_state.offsets,
            prepared.placement_state.numels,
        )
        return PlacementUnshardResult(full_params, prepared.buffers)

    @override
    def prepare_reduce_grad(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedReduceGrad:
        """Pack full gradients into one flat buffer for reduce-to-owner."""
        self._validate_owner_rank(mesh.size())
        if len(tensors) != len(infos):
            raise AssertionError(
                f"Expected {len(tensors)} tensors to match {len(infos)} infos."
            )
        dtype = self._require_uniform_dtype(
            tensors,
            infos,
            "grad_reduce_dtype",
            "reduce-grad",
        )
        offsets, numels, total_numel = self._flat_layout(infos)
        with _record_function_if_eager("FlexShard::reduce_copy_in", debug_fqn):
            flat = pack_tensors_into_flat_buffer(tensors, dtype)
            if flat.numel() != total_numel:
                raise AssertionError(
                    f"Owned reduce-grad flat buffer has {flat.numel()} elements, "
                    f"expected {total_numel}."
                )
        return PlacementPreparedReduceGrad(
            placement=self,
            buffers=[flat],
            placement_state=Owned._ReduceGradState(
                infos=infos,
                offsets=offsets,
                numels=numels,
                rank=mesh.get_local_rank(),
                world_size=mesh.size(),
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
                gradient_reduce_op=gradient_reduce_op_from_infos(infos),
            ),
        )

    @override
    def reduce_prepared_grad(
        self,
        prepared: PlacementPreparedReduceGrad,
    ) -> PlacementReduceGradResult:
        """Reduce one flat full-gradient buffer to the owner rank."""
        if not isinstance(prepared.placement_state, Owned._ReduceGradState):
            raise AssertionError(
                "Expected Owned._ReduceGradState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        if len(prepared.buffers) != 1:
            raise AssertionError(
                f"Expected one flat reduce-grad buffer, got {len(prepared.buffers)}."
            )
        flat = prepared.buffers[0]
        with _record_comm_if_eager(
            "FlexShard::reduce",
            prepared.placement_state.debug_fqn,
        ):
            dist.reduce(
                flat,
                dst=self.owner_rank,
                op=dist.ReduceOp.SUM,
                group=prepared.placement_state.pg,
            )
        if prepared.placement_state.rank == self.owner_rank:
            if prepared.placement_state.gradient_reduce_op == "avg":
                flat.div_(prepared.placement_state.world_size)
            sharded_grads = self._views_from_flat(
                flat,
                prepared.placement_state.infos,
                prepared.placement_state.offsets,
                prepared.placement_state.numels,
            )
        else:
            sharded_grads = [
                flat.new_empty(
                    self.compute_local_shape(
                        info.global_shape,
                        prepared.placement_state.rank,
                        prepared.placement_state.world_size,
                    )
                )
                for info in prepared.placement_state.infos
            ]
        return PlacementReduceGradResult(sharded_grads)


@dataclass(frozen=True)
class GroupedOwnedSegmentSpec:
    """Flat slice of one parameter owned by one rank inside a GroupedOwned bucket."""

    name: str
    fqn: str
    param_offset: int
    numel: int
    owner_rank: int
    storage_order: int = 0


class GroupedOwned(Placement):
    """Grouped owner-partition placement for one physical bucket collective.

    Unlike :class:`Owned`, each parameter may be split into multiple owner-local
    segments, and different segments in one bucket may have different owners.
    Forward gathers the padded owner partitions with one all-gather. Backward
    packs full gradients in owner-partition order and uses one reduce-scatter.
    """

    @dataclass(frozen=True)
    class _Segment:
        name: str
        fqn: str
        owner_rank: int
        param_offset: int
        bucket_offset: int
        rank_offset: int
        param_rank_offset: int
        numel: int
        global_shape: torch.Size

    @dataclass(frozen=True)
    class _Layout:
        segments: list["GroupedOwned._Segment"]
        rank_offsets: list[int]
        rank_numels: list[int]
        total_numel: int
        padded_rank_numel: int

    @dataclass(frozen=True)
    class _UnshardState:
        infos: list[ParamInfo]
        layout: "GroupedOwned._Layout"
        rank: int
        world_size: int
        pg: Any
        debug_fqn: str | None

    @dataclass(frozen=True)
    class _ReduceGradState:
        infos: list[ParamInfo]
        layout: "GroupedOwned._Layout"
        rank: int
        world_size: int
        pg: Any
        debug_fqn: str | None
        scratch_lease: _ReduceScratchLease | None
        gradient_reduce_op: GradientReduceOp

    def __init__(
        self,
        segments_by_fqn: dict[str, list[GroupedOwnedSegmentSpec]],
    ) -> None:
        normalized: dict[str, tuple[GroupedOwnedSegmentSpec, ...]] = {}
        for fqn, segments in segments_by_fqn.items():
            if not segments:
                raise ValueError(f"GroupedOwned requires at least one segment for {fqn!r}.")
            normalized_segments = tuple(
                sorted(segments, key=lambda segment: segment.param_offset)
            )
            for segment in normalized_segments:
                if segment.fqn != fqn:
                    raise ValueError(
                        f"GroupedOwned segment {segment.name!r} is under key {fqn!r} "
                        f"but names FQN {segment.fqn!r}."
                    )
                if segment.numel <= 0:
                    raise ValueError(
                        f"GroupedOwned segment {segment.name!r} has invalid "
                        f"numel {segment.numel}."
                    )
            normalized[fqn] = normalized_segments
        self.segments_by_fqn = normalized
        self._hash_key = tuple(
            (fqn, segments) for fqn, segments in sorted(normalized.items())
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupedOwned):
            return False
        return self._hash_key == other._hash_key

    def __hash__(self) -> int:
        return hash((type(self), self._hash_key))

    def __repr__(self) -> str:
        return f"GroupedOwned(num_params={len(self.segments_by_fqn)})"

    @staticmethod
    def _empty_shape(global_shape: torch.Size) -> torch.Size:
        return torch.Size([0] * len(global_shape))

    def _require_uniform_dtype(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        attr: str,
        op_name: str,
    ) -> torch.dtype:
        if not infos or not tensors:
            raise AssertionError(f"Expected at least one tensor for {op_name}.")
        dtype = getattr(infos[0], attr, tensors[0].dtype)
        for tensor, info in zip(tensors[1:], infos[1:], strict=True):
            other = getattr(info, attr, tensor.dtype)
            if other != dtype:
                raise ValueError(
                    f"GroupedOwned {op_name} requires one communication dtype per "
                    f"bucket, but {infos[0].fqn!r} uses {dtype} and {info.fqn!r} "
                    f"uses {other}."
                )
        return dtype

    def _layout_from_infos(
        self,
        infos: list[ParamInfo],
        world_size: int,
    ) -> _Layout:
        param_rank_numels: dict[tuple[str, int], int] = {}
        segment_inputs: list[tuple[ParamInfo, GroupedOwnedSegmentSpec, int, int]] = []
        for info_order, info in enumerate(infos):
            if info.fqn not in self.segments_by_fqn:
                raise ValueError(f"GroupedOwned missing segments for {info.fqn!r}.")
            specs = sorted(
                self.segments_by_fqn[info.fqn],
                key=lambda segment: segment.param_offset,
            )
            covered = 0
            expected_offset = 0
            for spec in specs:
                owner = spec.owner_rank
                if owner < 0 or owner >= world_size:
                    raise ValueError(
                        f"GroupedOwned owner for segment {spec.name!r} must be in "
                        f"[0, {world_size}), but got {owner}."
                    )
                if spec.param_offset != expected_offset:
                    raise ValueError(
                        f"GroupedOwned segments for {info.fqn!r} must be contiguous "
                        f"and sorted; expected offset {expected_offset}, got "
                        f"{spec.param_offset} for {spec.name!r}."
                    )
                expected_offset += spec.numel
                covered += spec.numel
                param_rank_key = (info.fqn, owner)
                param_rank_offset = param_rank_numels.get(param_rank_key, 0)
                param_rank_numels[param_rank_key] = param_rank_offset + spec.numel
                segment_inputs.append((info, spec, param_rank_offset, info_order))
            if covered != info.global_numel:
                raise ValueError(
                    f"GroupedOwned segments for {info.fqn!r} cover {covered} "
                    f"elements, expected {info.global_numel}."
                )

        rank_numels = [0] * world_size
        rank_offsets_by_spec: dict[tuple[str, int, int], int] = {}
        for info, spec, _param_rank_offset, info_order in sorted(
            segment_inputs,
            key=lambda item: (
                item[1].owner_rank,
                item[1].storage_order,
                item[3],
                item[1].param_offset,
                item[1].name,
            ),
        ):
            rank_offsets_by_spec[(info.fqn, spec.param_offset, spec.numel)] = rank_numels[
                spec.owner_rank
            ]
            rank_numels[spec.owner_rank] += spec.numel

        rank_offsets: list[int] = []
        total_numel = 0
        for numel in rank_numels:
            rank_offsets.append(total_numel)
            total_numel += numel
        padded_rank_numel = max(rank_numels, default=0)

        segments = [
            GroupedOwned._Segment(
                name=spec.name,
                fqn=info.fqn,
                owner_rank=spec.owner_rank,
                param_offset=spec.param_offset,
                bucket_offset=rank_offsets[spec.owner_rank]
                + rank_offsets_by_spec[(info.fqn, spec.param_offset, spec.numel)],
                rank_offset=rank_offsets_by_spec[(info.fqn, spec.param_offset, spec.numel)],
                param_rank_offset=param_rank_offset,
                numel=spec.numel,
                global_shape=info.global_shape,
            )
            for info, spec, param_rank_offset, _info_order in segment_inputs
        ]
        return GroupedOwned._Layout(
            segments=segments,
            rank_offsets=rank_offsets,
            rank_numels=rank_numels,
            total_numel=total_numel,
            padded_rank_numel=padded_rank_numel,
        )

    def _local_param_shape(
        self,
        info: ParamInfo,
        segments: list[_Segment],
        local_numel: int,
    ) -> torch.Size:
        if (
            len(segments) == 1
            and segments[0].param_offset == 0
            and local_numel == info.global_numel
        ):
            return info.global_shape
        if local_numel == 0:
            return self._empty_shape(info.global_shape)
        if len(info.global_shape) == 3 and segments:
            expert_numel = math.prod(info.global_shape[1:])
            if expert_numel > 0 and all(
                segment.numel == expert_numel for segment in segments
            ):
                sorted_segments = sorted(
                    segments,
                    key=lambda segment: segment.param_offset,
                )
                base_expert = sorted_segments[0].param_offset // expert_numel
                if all(
                    segment.param_offset == (base_expert + idx) * expert_numel
                    for idx, segment in enumerate(sorted_segments)
                ):
                    return torch.Size([len(sorted_segments), *info.global_shape[1:]])
        return torch.Size([local_numel])

    def _try_get_contiguous_owner_partition_view(
        self,
        tensors: list[torch.Tensor],
        layout: _Layout,
        rank: int,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        expected_numel = layout.rank_numels[rank]
        if sum(tensor.numel() for tensor in tensors) != expected_numel:
            return None
        non_empty_tensors = [tensor for tensor in tensors if tensor.numel() > 0]
        if not non_empty_tensors:
            device = tensors[0].device if tensors else torch.device("cuda")
            return torch.empty(0, dtype=dtype, device=device)

        first = non_empty_tensors[0]
        if first.dtype != dtype or not first.is_contiguous():
            return None
        storage_data_ptr = first.untyped_storage().data_ptr()
        expected_storage_offset = first.storage_offset()
        for tensor in non_empty_tensors:
            if tensor.dtype != dtype or tensor.device != first.device:
                return None
            if not tensor.is_contiguous():
                return None
            if tensor.untyped_storage().data_ptr() != storage_data_ptr:
                return None
            if tensor.storage_offset() != expected_storage_offset:
                return None
            expected_storage_offset += tensor.numel()

        return first.as_strided(
            (expected_numel,),
            (1,),
            storage_offset=first.storage_offset(),
        )

    def _owner_partition_matches_tensor_order(
        self,
        infos: list[ParamInfo],
        layout: _Layout,
        rank: int,
    ) -> bool:
        segments_by_fqn: dict[str, list[GroupedOwned._Segment]] = {}
        for segment in layout.segments:
            if segment.owner_rank == rank:
                segments_by_fqn.setdefault(segment.fqn, []).append(segment)
        expected_rank_offset = 0
        for info in infos:
            for segment in sorted(
                segments_by_fqn.get(info.fqn, []),
                key=lambda segment: segment.param_rank_offset,
            ):
                if segment.rank_offset != expected_rank_offset:
                    return False
                expected_rank_offset += segment.numel
        return expected_rank_offset == layout.rank_numels[rank]

    def _try_view_full_param_from_padded_gathered(
        self,
        gathered: torch.Tensor,
        info: ParamInfo,
        segments: list[_Segment],
        layout: _Layout,
    ) -> torch.Tensor | None:
        if not segments:
            return None
        expected_param_offset = 0
        base_storage_offset = (
            segments[0].owner_rank * layout.padded_rank_numel
            + segments[0].rank_offset
            - segments[0].param_offset
        )
        for segment in segments:
            if segment.param_offset != expected_param_offset:
                return None
            storage_offset = (
                segment.owner_rank * layout.padded_rank_numel + segment.rank_offset
            )
            if storage_offset != base_storage_offset + segment.param_offset:
                return None
            expected_param_offset += segment.numel
        if expected_param_offset != info.global_numel:
            return None

        return gathered.narrow(
            0,
            base_storage_offset,
            info.global_numel,
        ).view(info.global_shape)

    def _try_strided_expert_block_view(
        self,
        flat: torch.Tensor,
        info: ParamInfo,
        segments: list[_Segment],
        *,
        global_storage: bool,
        layout: _Layout | None = None,
        local_rank: int | None = None,
    ) -> torch.Tensor | None:
        if len(info.global_shape) != 3 or not segments:
            return None
        num_experts = info.global_shape[0]
        if global_storage and len(segments) != num_experts:
            return None
        expert_numel = math.prod(info.global_shape[1:])
        if expert_numel <= 0:
            return None
        sorted_segments = sorted(segments, key=lambda segment: segment.param_offset)
        base_expert_idx = sorted_segments[0].param_offset // expert_numel
        if global_storage and base_expert_idx != 0:
            return None
        storage_offsets: list[int] = []
        for expert_idx, segment in enumerate(sorted_segments):
            if segment.param_offset != (base_expert_idx + expert_idx) * expert_numel:
                return None
            if segment.numel != expert_numel:
                return None
            if global_storage:
                if layout is None:
                    return None
                storage_offsets.append(
                    segment.owner_rank * layout.padded_rank_numel + segment.rank_offset
                )
            else:
                if local_rank is None or segment.owner_rank != local_rank:
                    return None
                storage_offsets.append(segment.rank_offset)
        if len(storage_offsets) == 1:
            return None
        expert_stride = storage_offsets[1] - storage_offsets[0]
        if expert_stride <= 0:
            return None
        for prev, cur in zip(storage_offsets, storage_offsets[1:]):
            if cur - prev != expert_stride:
                return None
        if storage_offsets[-1] + expert_numel > flat.numel():
            return None
        row_stride = info.global_shape[2]
        return flat.as_strided(
            (len(sorted_segments), *tuple(info.global_shape[1:])),
            (expert_stride, row_stride, 1),
            storage_offset=storage_offsets[0],
        )

    def _views_from_padded_gathered(
        self,
        gathered: torch.Tensor,
        infos: list[ParamInfo],
        layout: _Layout,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        segments_by_fqn: dict[str, list[GroupedOwned._Segment]] = {}
        for segment in layout.segments:
            segments_by_fqn.setdefault(segment.fqn, []).append(segment)

        direct_views: list[torch.Tensor] = []
        for info in infos:
            segments = sorted(
                segments_by_fqn[info.fqn],
                key=lambda segment: segment.param_offset,
            )
            direct_view = self._try_view_full_param_from_padded_gathered(
                gathered,
                info,
                segments,
                layout,
            )
            if direct_view is None:
                direct_view = self._try_strided_expert_block_view(
                    gathered,
                    info,
                    segments,
                    global_storage=True,
                    layout=layout,
                )
            if direct_view is None:
                direct_views = []
                break
            direct_views.append(direct_view)

        # Avoid keeping the whole gathered bucket alive for a subset of params.
        if len(direct_views) == len(infos):
            with _record_view_out_if_eager():
                return direct_views, []

        with _record_copy_out_if_eager():
            full_params: list[torch.Tensor] = []
            for info in infos:
                segments = sorted(
                    segments_by_fqn[info.fqn],
                    key=lambda segment: segment.param_offset,
                )
                full = torch.empty(
                    info.global_numel,
                    dtype=gathered.dtype,
                    device=gathered.device,
                )
                for segment in segments:
                    full.narrow(0, segment.param_offset, segment.numel).copy_(
                        gathered.narrow(
                            0,
                            segment.owner_rank * layout.padded_rank_numel
                            + segment.rank_offset,
                            segment.numel,
                        )
                    )
                full_params.append(full.view(info.global_shape))
        return full_params, []

    def _views_from_rank_flat(
        self,
        flat: torch.Tensor,
        infos: list[ParamInfo],
        layout: _Layout,
        rank: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[torch.Tensor]:
        segments_by_fqn: dict[str, list[GroupedOwned._Segment]] = {}
        for segment in layout.segments:
            if segment.owner_rank == rank:
                segments_by_fqn.setdefault(segment.fqn, []).append(segment)
        grads: list[torch.Tensor] = []
        for info in infos:
            segments = sorted(
                segments_by_fqn.get(info.fqn, []),
                key=lambda segment: segment.param_rank_offset,
            )
            local_numel = sum(segment.numel for segment in segments)
            if local_numel == 0:
                grads.append(
                    torch.empty(
                        self._empty_shape(info.global_shape),
                        dtype=dtype,
                        device=device,
                    )
                )
                continue
            strided_view = self._try_strided_expert_block_view(
                flat,
                info,
                segments,
                global_storage=False,
                local_rank=rank,
            )
            if strided_view is not None:
                grads.append(strided_view)
                continue
            start = segments[0].rank_offset
            grads.append(
                flat.narrow(0, start, local_numel).view(
                    self._local_param_shape(info, segments, local_numel)
                )
            )
        return grads

    def _acquire_reduce_send(
        self,
        send_numel: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, _ReduceScratchLease | None]:
        if torch.compiler.is_compiling():
            return torch.empty(send_numel, dtype=dtype, device=device), None
        lease = _grouped_owned_reduce_scratch_pool().acquire(
            send_numel=send_numel,
            dtype=dtype,
            device=device,
        )
        return lease.send, lease

    def _zero_reduce_send_padding(
        self,
        send_by_owner: torch.Tensor,
        layout: _Layout,
    ) -> None:
        for owner, numel in enumerate(layout.rank_numels):
            padding = layout.padded_rank_numel - numel
            if padding > 0:
                send_by_owner[owner].narrow(0, numel, padding).zero_()

    @staticmethod
    def _segment_source_offset(
        tensor: torch.Tensor,
        segment: _Segment,
    ) -> int | None:
        if tensor.is_contiguous():
            return segment.param_offset
        if tensor.dim() != 3:
            return None
        expert_numel = math.prod(tensor.shape[1:])
        if expert_numel <= 0:
            return None
        if segment.numel != expert_numel or segment.param_offset % expert_numel != 0:
            return None
        if tensor.stride(-1) != 1 or tensor.stride(-2) != tensor.shape[-1]:
            return None
        expert_idx = segment.param_offset // expert_numel
        if expert_idx < 0 or expert_idx >= tensor.shape[0]:
            return None
        return expert_idx * tensor.stride(0)

    @classmethod
    def _flat_source_for_segments(
        cls,
        tensor: torch.Tensor,
        segments: list[_Segment],
    ) -> tuple[torch.Tensor, list[int]] | None:
        if not segments:
            return tensor.reshape(-1), []
        offsets: list[int] = []
        max_end = 0
        for segment in segments:
            offset = cls._segment_source_offset(tensor, segment)
            if offset is None:
                return None
            offsets.append(offset)
            max_end = max(max_end, offset + segment.numel)
        if tensor.is_contiguous():
            return tensor.reshape(-1), offsets
        try:
            flat_storage_span = tensor.as_strided(
                (max_end,),
                (1,),
                storage_offset=tensor.storage_offset(),
            )
        except RuntimeError:
            return None
        return flat_storage_span, offsets

    def _try_pack_reduce_send_triton(
        self,
        send_by_owner: torch.Tensor,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        layout: _Layout,
    ) -> list[torch.Tensor] | None:
        if send_by_owner.device.type != "cuda":
            return None
        segments_by_fqn: dict[str, list[GroupedOwned._Segment]] = {}
        for segment in layout.segments:
            segments_by_fqn.setdefault(segment.fqn, []).append(segment)

        inputs: list[torch.Tensor] = []
        tensor_indices: list[int] = []
        src_offsets: list[int] = []
        numels: list[int] = []
        dst_offsets: list[int] = []
        input_dtype: torch.dtype | None = None
        for tensor, info in zip(tensors, infos, strict=True):
            segments = segments_by_fqn[info.fqn]
            source = self._flat_source_for_segments(tensor, segments)
            if source is None:
                return None
            flat_input, source_offsets = source
            if input_dtype is None:
                input_dtype = flat_input.dtype
            elif flat_input.dtype != input_dtype:
                return None
            input_index = len(inputs)
            inputs.append(flat_input)
            for segment, source_offset in zip(segments, source_offsets, strict=True):
                tensor_indices.append(input_index)
                src_offsets.append(source_offset)
                numels.append(segment.numel)
                dst_offsets.append(
                    segment.owner_rank * layout.padded_rank_numel
                    + segment.rank_offset
                )

        return try_pack_segments_into_flat_buffer_triton(
            inputs,
            tensor_indices,
            src_offsets,
            numels,
            dst_offsets,
            send_by_owner.reshape(-1),
        )

    def _pack_reduce_send(
        self,
        send_by_owner: torch.Tensor,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        layout: _Layout,
    ) -> list[torch.Tensor]:
        pack_scratch = self._try_pack_reduce_send_triton(
            send_by_owner,
            tensors,
            infos,
            layout,
        )
        if pack_scratch is not None:
            return pack_scratch

        segments_by_fqn: dict[str, list[GroupedOwned._Segment]] = {}
        for segment in layout.segments:
            segments_by_fqn.setdefault(segment.fqn, []).append(segment)

        dst_views: list[torch.Tensor] = []
        src_views: list[torch.Tensor] = []
        for tensor, info in zip(tensors, infos, strict=True):
            tensor_flat = tensor.reshape(-1)
            for segment in segments_by_fqn[info.fqn]:
                dst_views.append(
                    send_by_owner[segment.owner_rank].narrow(
                        0,
                        segment.rank_offset,
                        segment.numel,
                    )
                )
                src_views.append(
                    tensor_flat.narrow(0, segment.param_offset, segment.numel)
                )
        if dst_views:
            foreach_copy_(dst_views, src_views)
        return []

    @override
    def compute_local_shape(
        self, global_shape: torch.Size, rank: int, world_size: int
    ) -> torch.Size:
        raise NotImplementedError(
            "GroupedOwned local shape is FQN-dependent; use bucket_storage_layout()."
        )

    @override
    def extract_local_shard(
        self,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "GroupedOwned local shard is FQN-dependent; use copy_param_to_storage()."
        )

    @override
    def bucket_storage_layout(
        self,
        named_params: list[tuple[str, nn.Parameter]],
        param_placements: dict[str, tuple[Placement, ...]],
        mesh: DeviceMesh,
    ) -> BucketStorageLayout | None:
        from ..flex_shard.bucket_storage import ParamInfo

        rank = mesh.get_local_rank()
        world_size = mesh.size()
        expected = {fqn for fqn, _ in named_params}
        actual = set(self.segments_by_fqn)
        if expected != actual:
            raise ValueError(
                "GroupedOwned segment map must match the bucket FQNs exactly: "
                f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
            )
        for fqn, _ in named_params:
            placements = param_placements[fqn]
            if placements != (self,):
                raise ValueError(
                    "GroupedOwned requires the same placement instance for every "
                    f"parameter in a bucket; {fqn!r} uses {placements!r}."
                )

        infos = [
            ParamInfo(
                fqn=fqn,
                global_shape=param.shape,
                global_stride=tuple(param.stride()),
                dtype=param.dtype,
                requires_grad=param.requires_grad,
                placements=(self,),
                global_numel=param.numel(),
            )
            for fqn, param in named_params
        ]
        layout = self._layout_from_infos(infos, world_size)
        segments_by_fqn: dict[str, list[GroupedOwned._Segment]] = {}
        for segment in layout.segments:
            if segment.owner_rank == rank:
                segments_by_fqn.setdefault(segment.fqn, []).append(segment)

        param_layouts: dict[str, BucketParamStorageLayout] = {}
        byte_offset = 0
        info_by_fqn = {info.fqn: info for info in infos}
        for fqn, param in named_params:
            segments = sorted(
                segments_by_fqn.get(fqn, []),
                key=lambda segment: segment.param_rank_offset,
            )
            local_numel = sum(segment.numel for segment in segments)
            if local_numel > 0:
                local_shape = self._local_param_shape(
                    info_by_fqn[fqn],
                    segments,
                    local_numel,
                )
                storage_nbytes = local_numel * param.dtype.itemsize
                offset = byte_offset
                byte_offset += storage_nbytes
            else:
                local_shape = self._empty_shape(param.shape)
                storage_nbytes = 0
                offset = 0
            param_layouts[fqn] = BucketParamStorageLayout(
                local_shape=local_shape,
                local_numel=local_numel,
                byte_offset=offset,
                storage_nbytes=storage_nbytes,
            )
        return BucketStorageLayout(param_layouts=param_layouts, total_bytes=byte_offset)

    @override
    def copy_param_to_storage(
        self,
        byte_storage: torch.Tensor,
        info: ParamInfo,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> None:
        param_data = param.detach()
        if param_data.device.type == "meta":
            return
        if not param_data.is_contiguous():
            param_data = param_data.contiguous()
        layout = self._layout_from_infos([info], world_size)
        owned_segments = [
            segment for segment in layout.segments if segment.owner_rank == rank
        ]
        if not owned_segments:
            return
        local_flat = byte_storage[
            info.byte_offset : info.byte_offset + info.storage_nbytes
        ].view(param_data.dtype)
        param_flat = param_data.reshape(-1)
        for segment in owned_segments:
            local_flat.narrow(
                0,
                segment.param_rank_offset,
                segment.numel,
            ).copy_(
                param_flat.narrow(0, segment.param_offset, segment.numel)
            )

    @override
    def prepare_unshard_bucket(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedUnshard:
        rank = mesh.get_local_rank()
        world_size = mesh.size()
        dtype = self._require_uniform_dtype(tensors, infos, "unsharded_dtype", "unshard")
        layout = self._layout_from_infos(infos, world_size)
        device = tensors[0].device
        with _record_copy_in_if_eager():
            true_send_numel = layout.rank_numels[rank]
            local_owner_view = self._try_get_contiguous_owner_partition_view(
                tensors,
                layout,
                rank,
                dtype,
            )
            if local_owner_view is not None and not self._owner_partition_matches_tensor_order(
                infos,
                layout,
                rank,
            ):
                local_owner_view = None
            if (
                local_owner_view is not None
                and true_send_numel == layout.padded_rank_numel
            ):
                send = local_owner_view
            else:
                send = torch.empty(layout.padded_rank_numel, dtype=dtype, device=device)
                if layout.padded_rank_numel != true_send_numel:
                    send.zero_()
                if local_owner_view is not None:
                    send.narrow(0, 0, true_send_numel).copy_(local_owner_view)
                else:
                    segments_by_fqn: dict[str, list[GroupedOwned._Segment]] = {}
                    for segment in layout.segments:
                        if segment.owner_rank == rank:
                            segments_by_fqn.setdefault(segment.fqn, []).append(segment)
                    for tensor, info in zip(tensors, infos, strict=True):
                        tensor_flat = tensor.reshape(-1).to(dtype)
                        for segment in sorted(
                            segments_by_fqn.get(info.fqn, []),
                            key=lambda segment: segment.param_rank_offset,
                        ):
                            send.narrow(0, segment.rank_offset, segment.numel).copy_(
                                tensor_flat.narrow(
                                    0,
                                    segment.param_rank_offset,
                                    segment.numel,
                                )
                            )
            gathered = torch.empty(
                world_size * layout.padded_rank_numel,
                dtype=dtype,
                device=device,
            )
        return PlacementPreparedUnshard(
            placement=self,
            buffers=[send, gathered],
            placement_state=GroupedOwned._UnshardState(
                infos=infos,
                layout=layout,
                rank=rank,
                world_size=world_size,
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
            ),
        )

    @override
    def run_prepared_unshard(self, prepared: PlacementPreparedUnshard) -> None:
        if not isinstance(prepared.placement_state, GroupedOwned._UnshardState):
            raise AssertionError(
                "Expected GroupedOwned._UnshardState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        with _record_comm_if_eager(
            "FlexShard::grouped_owned_all_gather",
            prepared.placement_state.debug_fqn,
        ):
            dist.all_gather_into_tensor(
                output_tensor=prepared.buffers[1],
                input_tensor=prepared.buffers[0],
                group=prepared.placement_state.pg,
            )

    @override
    def finish_prepared_unshard(
        self,
        prepared: PlacementPreparedUnshard,
    ) -> PlacementUnshardResult:
        if not isinstance(prepared.placement_state, GroupedOwned._UnshardState):
            raise AssertionError(
                "Expected GroupedOwned._UnshardState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        layout = prepared.placement_state.layout
        full_params, copy_out_scratch = self._views_from_padded_gathered(
            prepared.buffers[1],
            prepared.placement_state.infos,
            layout,
        )
        return PlacementUnshardResult(
            full_params,
            [*prepared.buffers, *copy_out_scratch],
        )

    @override
    def prepare_reduce_grad(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedReduceGrad:
        rank = mesh.get_local_rank()
        world_size = mesh.size()
        dtype = self._require_uniform_dtype(
            tensors,
            infos,
            "grad_reduce_dtype",
            "reduce-grad",
        )
        layout = self._layout_from_infos(infos, world_size)
        device = tensors[0].device
        with _record_function_if_eager(
            "FlexShard::grouped_owned_reduce_scatter_copy_in",
            debug_fqn,
        ):
            send, scratch_lease = self._acquire_reduce_send(
                world_size * layout.padded_rank_numel,
                dtype,
                device,
            )
            send_by_owner = send.view(world_size, layout.padded_rank_numel)
            self._zero_reduce_send_padding(send_by_owner, layout)
            pack_scratch = self._pack_reduce_send(send_by_owner, tensors, infos, layout)
        return PlacementPreparedReduceGrad(
            placement=self,
            buffers=[send, *pack_scratch],
            placement_state=GroupedOwned._ReduceGradState(
                infos=infos,
                layout=layout,
                rank=rank,
                world_size=world_size,
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
                scratch_lease=scratch_lease,
                gradient_reduce_op=gradient_reduce_op_from_infos(infos),
            ),
        )

    @override
    def reduce_prepared_grad(
        self,
        prepared: PlacementPreparedReduceGrad,
    ) -> PlacementReduceGradResult:
        if not isinstance(prepared.placement_state, GroupedOwned._ReduceGradState):
            raise AssertionError(
                "Expected GroupedOwned._ReduceGradState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        state = prepared.placement_state
        send = prepared.buffers[0]
        recv = torch.empty(
            state.layout.padded_rank_numel,
            dtype=send.dtype,
            device=send.device,
        )
        with _record_comm_if_eager(
            "FlexShard::grouped_owned_reduce_scatter",
            state.debug_fqn,
        ):
            try:
                dist.reduce_scatter_tensor(
                    output=recv,
                    input=send,
                    op=dist_reduce_op(state.gradient_reduce_op),
                    group=state.pg,
                )
            finally:
                if state.scratch_lease is not None:
                    state.scratch_lease.release()
        with _record_function_if_eager(
            "FlexShard::grouped_owned_reduce_scatter_copy_out",
            state.debug_fqn,
        ):
            sharded_grads = self._views_from_rank_flat(
                recv,
                state.infos,
                state.layout,
                state.rank,
                send.dtype,
                send.device,
            )
        return PlacementReduceGradResult(sharded_grads, [recv])


def _assign_params_to_ranks(
    named_params: list[tuple[str, nn.Parameter]],
    world_size: int,
) -> dict[str, int]:
    """Greedily assign each full parameter to the currently least-loaded rank."""
    loads = [(0, rank) for rank in range(world_size)]
    heapq.heapify(loads)
    assignments: dict[str, int] = {}
    for fqn, param in sorted(
        named_params,
        key=lambda item: item[1].numel(),
        reverse=True,
    ):
        load, rank = heapq.heappop(loads)
        assignments[fqn] = rank
        heapq.heappush(loads, (load + param.numel(), rank))
    return assignments


def param_boundary_placements(
    named_params: list[tuple[str, nn.Parameter]],
    mesh: DeviceMesh,
) -> dict[str, tuple[Placement, ...]]:
    """Assign each full parameter to one rank using Owned placements."""
    assignments = _assign_params_to_ranks(named_params, mesh.size())
    return {fqn: (Owned(assignments[fqn]),) for fqn, _ in named_params}


def _expert_block_order_key(fqn: str, suffix_order: tuple[str, ...]) -> int:
    for index, suffix in enumerate(suffix_order):
        if fqn.endswith(suffix):
            return index
    raise ValueError(f"Unexpected grouped expert weight FQN: {fqn}")


def make_grouped_owned_expert_block_segments(
    named_params: list[tuple[str, nn.Parameter]],
    world_size: int,
    *,
    suffix_order: tuple[str, ...] = (".w1", ".w3", ".w2"),
) -> dict[str, list[GroupedOwnedSegmentSpec]]:
    """Build per-expert GroupedOwned segments for packed 3D expert weights.

    Each input parameter is expected to be a packed expert stack with shape
    ``(num_experts, ..., ...)``. Experts are assigned to contiguous owner ranges,
    while ``suffix_order`` controls how each expert's matrices are interleaved in
    owner storage. The default order keeps DeepSeek V3's ``w1/w3/w2`` block
    layout suitable for the GroupedOwned view-out path.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, but got {world_size}.")
    if not suffix_order:
        raise ValueError("suffix_order must contain at least one suffix.")
    if not named_params:
        return {}

    bad = [
        (fqn, tuple(param.shape))
        for fqn, param in named_params
        if param.dim() != 3 or not fqn.endswith(suffix_order)
    ]
    if bad:
        raise ValueError(f"GroupedOwned expert block expects packed 3D weights: {bad}")

    num_experts = named_params[0][1].shape[0]
    if any(param.shape[0] != num_experts for _, param in named_params):
        raise ValueError("GroupedOwned expert block requires matching expert counts.")

    ordered_params = sorted(
        named_params,
        key=lambda item: _expert_block_order_key(item[0], suffix_order),
    )
    segments_by_fqn: dict[str, list[GroupedOwnedSegmentSpec]] = {
        fqn: [] for fqn, _ in ordered_params
    }

    # Fill owner rows in contiguous equal-capacity expert ranges. Since the
    # all-gather input is padded to the max owner row length, this keeps the
    # gathered expert slices evenly strided and enables view-out.
    experts_per_owner = max(1, (num_experts + world_size - 1) // world_size)
    for expert_idx in range(num_experts):
        owner = min(expert_idx // experts_per_owner, world_size - 1)
        for param_order, (fqn, param) in enumerate(ordered_params):
            expert_numel = math.prod(param.shape[1:])
            segments_by_fqn[fqn].append(
                GroupedOwnedSegmentSpec(
                    name=f"{fqn}#expert{expert_idx}",
                    fqn=fqn,
                    param_offset=expert_idx * expert_numel,
                    numel=expert_numel,
                    owner_rank=owner,
                    storage_order=expert_idx * len(ordered_params) + param_order,
                )
            )
    return segments_by_fqn


def make_grouped_owned_expert_block_placement_fn(
    *,
    suffix_order: tuple[str, ...] = (".w1", ".w3", ".w2"),
) -> PlacementFn:
    """Return a placement function for one packed 3D expert-weight bucket."""

    def placement_fn(
        named_params: list[tuple[str, nn.Parameter]],
        mesh: DeviceMesh,
    ) -> dict[str, tuple[Placement, ...]]:
        segments_by_fqn = make_grouped_owned_expert_block_segments(
            named_params,
            mesh.size(),
            suffix_order=suffix_order,
        )
        placement = GroupedOwned(segments_by_fqn)
        return {fqn: (placement,) for fqn, _ in named_params}

    return placement_fn


def make_grouped_owned_placement_fn(
    segments_by_fqn: dict[str, list[GroupedOwnedSegmentSpec]],
) -> PlacementFn:
    """Return a placement function assigning one GroupedOwned placement per bucket."""

    def placement_fn(
        named_params: list[tuple[str, nn.Parameter]],
        mesh: DeviceMesh,
    ) -> dict[str, tuple[Placement, ...]]:
        del mesh
        expected = {fqn for fqn, _ in named_params}
        actual = set(segments_by_fqn)
        if expected != actual:
            raise ValueError(
                "GroupedOwned segment map must match the bucket FQNs exactly: "
                f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
            )
        placement = GroupedOwned(segments_by_fqn)
        return {fqn: (placement,) for fqn, _ in named_params}

    return placement_fn


__all__ = [
    "GroupedOwned",
    "GroupedOwnedSegmentSpec",
    "make_grouped_owned_expert_block_placement_fn",
    "make_grouped_owned_expert_block_segments",
    "make_grouped_owned_placement_fn",
    "Owned",
    "param_boundary_placements",
]
