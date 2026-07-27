# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bucketed storage-to-compute sharding for optimizer steps."""

from __future__ import annotations

import fnmatch
import hashlib
import heapq
import math
import re
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from types import MethodType, ModuleType
from typing import Any, cast, Protocol, TYPE_CHECKING, TypeVar

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import _get_device_handle, DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torch.optim import Optimizer


if TYPE_CHECKING:
    from torch import Tensor


__all__ = [
    "build_layer_bucket_specs",
    "BucketAssignment",
    "BucketSpec",
    "ComputeShardingRequirement",
    "FlexShardOptimizer",
    "Owned",
    "SameAsStorage",
    "flex_shard",
    "get_flex_shard_assignments",
]


_OptimizerT = TypeVar("_OptimizerT", bound=Optimizer)


class _HasFqn(Protocol):
    fqn: str


_BucketBindingT = TypeVar("_BucketBindingT", bound=_HasFqn)


@dataclass(frozen=True, slots=True)
class BucketSpec:
    """One ordered optimizer-compute communication bucket.

    ``patterns`` use ``fnmatch`` syntax and match canonical optimizer FQNs.
    Every parameter managed by the optimizer must match exactly one bucket.
    """

    patterns: tuple[str, ...]
    mesh: DeviceMesh
    name: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.patterns, str):
            raise TypeError("BucketSpec patterns must be a sequence of strings")
        object.__setattr__(self, "patterns", tuple(self.patterns))
        if not self.patterns or any(not pattern for pattern in self.patterns):
            raise ValueError("BucketSpec requires non-empty FQN patterns")


@dataclass(frozen=True, slots=True)
class BucketAssignment:
    """Resolved compute-bucket information for one optimizer parameter.

    ``owner_rank`` is a global distributed rank when compute is routed to one
    owner. It is ``None`` when compute preserves the storage sharding.
    """

    bucket_name: str
    fqn: str
    owner_rank: int | None


@dataclass(frozen=True, slots=True)
class SameAsStorage:
    """Compute directly in each parameter's persistent storage placement."""


@dataclass(frozen=True, slots=True)
class Owned:
    """Place each complete trailing compute block on exactly one mesh rank."""

    trailing_dims: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.trailing_dims, bool)
            or not isinstance(self.trailing_dims, int)
            or self.trailing_dims <= 0
        ):
            raise ValueError("Owned trailing_dims must be a positive integer")


ComputeShardingRequirement = SameAsStorage | Owned


class FlexShardOptimizer(Protocol):
    """Optimizer-side declaration and distribution-agnostic compute hooks.

    ``prepare``, ``compute``, and ``finalize`` receive only plain tensors.
    FlexShard chooses owners and supplies output buffers; the optimizer writes
    distribution-agnostic results into those buffers. ``prepare`` may update
    local optimizer state in place, and ``finalize`` must support ``out``
    aliasing ``param``.
    """

    param_groups: list[dict[str, Any]]
    state: dict[Any, Any]

    def flex_shard_compute_requirement(
        self,
        param: Tensor,
        group: MutableMapping[str, Any],
    ) -> ComputeShardingRequirement:
        ...

    def flex_shard_validate_group(
        self,
        group_index: int,
        group: MutableMapping[str, Any],
    ) -> None:
        ...

    def flex_shard_group_signature(
        self,
        group: MutableMapping[str, Any],
    ) -> object:
        ...

    def flex_shard_init_state(
        self,
        param: Tensor,
        grad: Tensor,
        group: MutableMapping[str, Any],
    ) -> MutableMapping[str, Any]:
        ...

    def flex_shard_prepare(
        self,
        param: Tensor,
        grad: Tensor,
        state: Mapping[str, Tensor],
        group: MutableMapping[str, Any],
        *,
        out: Tensor,
    ) -> None:
        ...

    def flex_shard_compute(
        self,
        compute_input: Tensor,
        group: MutableMapping[str, Any],
    ) -> Tensor:
        ...

    def flex_shard_finalize(
        self,
        param: Tensor,
        update: Tensor,
        group: MutableMapping[str, Any],
        *,
        out: Tensor,
    ) -> None:
        ...


class _FlexShardRuntime(Protocol):
    """Execution plan built and owned by :func:`flex_shard`."""

    @property
    def assignments(self) -> tuple[BucketAssignment, ...]:
        ...


def get_flex_shard_assignments(
    optimizer: Optimizer,
) -> tuple[BucketAssignment, ...]:
    """Return the resolved compute plan for a flex-sharded optimizer."""
    runtime = getattr(optimizer, "_flex_shard_runtime", None)
    return () if runtime is None else runtime.assignments


def build_layer_bucket_specs(
    optimizer: Optimizer,
    *,
    mesh: DeviceMesh | None = None,
) -> tuple[BucketSpec, ...]:
    """Build one exact-FQN compute bucket per canonical transformer layer."""
    layer_fqns: dict[str, list[str]] = {}
    inferred_mesh = mesh
    for group in optimizer.param_groups:
        params = group["params"]
        param_names = group.get("param_names")
        if param_names is None or len(param_names) != len(params):
            raise ValueError(
                "Layer compute buckets require param_names aligned with params"
            )
        for fqn, param in zip(param_names, params, strict=True):
            match = re.match(r"^(.*?layers\.\d+)\.", fqn)
            if match is None:
                raise ValueError(
                    f"Optimizer parameter {fqn!r} is not under a canonical "
                    "'<prefix>layers.<index>.' FQN"
                )
            layer_fqns.setdefault(match.group(1), []).append(fqn)
            if inferred_mesh is None:
                if not isinstance(param, DTensor):
                    raise ValueError(
                        f"Cannot infer a DeviceMesh from non-DTensor parameter {fqn!r}"
                    )
                inferred_mesh = param.device_mesh

    if inferred_mesh is None:
        raise ValueError("Layer compute buckets require at least one parameter")

    def layer_order(item: tuple[str, list[str]]) -> tuple[str, int]:
        layer_name = item[0]
        prefix, index = layer_name.rsplit(".", 1)
        return prefix, int(index)

    return tuple(
        BucketSpec(
            name=layer_name,
            patterns=tuple(fqns),
            mesh=inferred_mesh,
        )
        for layer_name, fqns in sorted(layer_fqns.items(), key=layer_order)
    )


@dataclass(frozen=True, slots=True)
class _FqnBinding:
    fqn: str


def _bind_optimizer_fqns(optimizer: Optimizer) -> tuple[_FqnBinding, ...]:
    bindings = []
    seen_fqns: set[str] = set()
    seen_param_ids: set[int] = set()
    for group in optimizer.param_groups:
        params = group["params"]
        param_names = group.get("param_names")
        if param_names is None or len(param_names) != len(params):
            raise ValueError(
                "flex_shard parameter groups require param_names aligned with params"
            )
        for fqn, param in zip(param_names, params, strict=True):
            if not isinstance(fqn, str) or not fqn:
                raise ValueError(f"Invalid optimizer parameter FQN {fqn!r}")
            if fqn in seen_fqns:
                raise ValueError(f"Duplicate optimizer parameter FQN {fqn!r}")
            if id(param) in seen_param_ids:
                raise ValueError(f"Optimizer parameter {fqn!r} appears more than once")
            seen_fqns.add(fqn)
            seen_param_ids.add(id(param))
            bindings.append(_FqnBinding(fqn))
    if not bindings:
        raise ValueError("flex_shard requires at least one optimizer parameter")
    return tuple(bindings)


@dataclass(frozen=True, slots=True)
class _UnassignedBinding:
    fqn: str
    param: DTensor
    group_index: int
    requirement: Owned
    global_shape: torch.Size
    global_stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    shard_numels: tuple[int, ...]
    shard_offsets: tuple[int, ...]

    @property
    def global_numel(self) -> int:
        return self.global_shape.numel()


@dataclass(frozen=True, slots=True)
class _Binding:
    fqn: str
    param: DTensor
    group_index: int
    requirement: Owned
    owner_group_rank: int
    global_shape: torch.Size
    global_stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    shard_numels: tuple[int, ...]
    shard_offsets: tuple[int, ...]

    @property
    def global_numel(self) -> int:
        return self.global_shape.numel()


@dataclass(slots=True)
class _BucketPlan:
    spec: BucketSpec
    bindings: tuple[_Binding, ...]
    process_group: dist.ProcessGroup
    group_rank: int
    world_size: int
    input_split_sizes: list[int]
    output_split_sizes: list[int]
    local_offsets: tuple[int, ...]
    owner_offsets: dict[tuple[int, int], int]
    dtype: torch.dtype
    device: torch.device
    local_buffer_numel: int
    owner_buffer_numel: int
    owner_buffer_is_compute_layout: bool


@dataclass(slots=True)
class _BucketWork:
    plan: _BucketPlan
    active_by_param: dict[int, bool]
    local_buffer: Tensor
    owner_buffer: Tensor
    reverse_buffer: Tensor | None = None
    forward_ready: torch.Event | None = None
    compute_done: torch.Event | None = None
    done: torch.Event | None = None
    compute_keepalives: list[Tensor] = field(default_factory=list)


@dataclass(slots=True)
class _BucketBufferSlot:
    local_storage: dict[tuple[torch.device, torch.dtype], Tensor] = field(
        default_factory=dict
    )
    owner_storage: dict[tuple[torch.device, torch.dtype], Tensor] = field(
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

    def buffers(self, plan: _BucketPlan) -> tuple[Tensor, Tensor]:
        local_buffer = self._ensure_capacity(
            self.local_storage,
            numel=plan.local_buffer_numel,
            dtype=plan.dtype,
            device=plan.device,
        )
        owner_buffer = self._ensure_capacity(
            self.owner_storage,
            numel=plan.owner_buffer_numel,
            dtype=plan.dtype,
            device=plan.device,
        )
        return local_buffer, owner_buffer


@dataclass(slots=True)
class _OwnedFlexShardCommContext:
    device_handle: ModuleType
    transfer_stream: torch.Stream
    slots: tuple[_BucketBufferSlot, _BucketBufferSlot]

    @classmethod
    def create(cls, device: torch.device) -> _OwnedFlexShardCommContext:
        device_handle = _get_device_handle(device.type)
        stream = (
            device_handle.Stream(priority=0)
            if device.type == "cpu"
            else device_handle.Stream(device=device, priority=0)
        )
        return cls(
            device_handle=device_handle,
            transfer_stream=stream,
            slots=(_BucketBufferSlot(), _BucketBufferSlot()),
        )


def _mesh_ranks(mesh: DeviceMesh) -> tuple[int, ...]:
    return tuple(dist.get_process_group_ranks(mesh.get_group()))


def _shard_metadata(
    shape: torch.Size,
    world_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    row_numel = math.prod(shape[1:])
    shard_numels = []
    shard_offsets = []
    for rank in range(world_size):
        rows, row_offset = Shard.local_shard_size_and_offset(shape[0], world_size, rank)
        shard_numels.append(int(rows) * row_numel)
        shard_offsets.append(int(row_offset) * row_numel)
    return tuple(shard_numels), tuple(shard_offsets)


def _compute_sharding_requirement(
    optimizer: Optimizer,
    param: Tensor,
    group: MutableMapping[str, Any],
) -> ComputeShardingRequirement:
    provider = getattr(optimizer, "flex_shard_compute_requirement", None)
    if provider is None:
        # torch.optim.AdamW is pointwise, so its compute placement is exactly
        # its persistent parameter and state placement.
        if type(optimizer) is torch.optim.AdamW:
            return SameAsStorage()
        raise TypeError(
            f"{type(optimizer).__name__} must implement "
            "flex_shard_compute_requirement()"
        )
    requirement = provider(param, group)
    if not isinstance(requirement, (SameAsStorage, Owned)):
        raise TypeError(
            "flex_shard_compute_requirement() must return SameAsStorage or Owned, "
            f"got {requirement!r}"
        )
    return requirement


def _bind_optimizer_params(
    optimizer: Optimizer,
) -> tuple[_UnassignedBinding, ...]:
    bindings = []
    seen_fqns: set[str] = set()
    seen_param_ids: set[int] = set()
    process_group_ranks: tuple[int, ...] | None = None
    tensor_device: torch.device | None = None

    for group_index, group in enumerate(optimizer.param_groups):
        params = group["params"]
        param_names = group.get("param_names")
        if param_names is None or len(param_names) != len(params):
            raise ValueError(
                "Owned FlexShard parameter groups require param_names aligned with "
                "params"
            )
        for fqn, param in zip(param_names, params, strict=True):
            if not isinstance(fqn, str) or not fqn:
                raise ValueError(f"Invalid FlexShard parameter FQN {fqn!r}")
            if fqn in seen_fqns:
                raise ValueError(f"Duplicate FlexShard parameter FQN {fqn!r}")
            seen_fqns.add(fqn)
            if id(param) in seen_param_ids:
                raise ValueError(f"FlexShard parameter {fqn!r} appears more than once")
            seen_param_ids.add(id(param))

            requirement = _compute_sharding_requirement(optimizer, param, group)
            if not isinstance(requirement, Owned):
                raise ValueError(
                    "Owned FlexShard runtime cannot mix SameAsStorage and Owned "
                    f"requirements; {fqn!r} declared {requirement!r}"
                )

            if not isinstance(param, DTensor):
                raise ValueError(f"Owned FlexShard parameter {fqn!r} must be a DTensor")
            if param.ndim < requirement.trailing_dims:
                raise ValueError(
                    f"Owned FlexShard parameter {fqn!r} has {param.ndim} dimensions, "
                    f"fewer than its {requirement.trailing_dims} complete trailing "
                    "dimensions"
                )
            if param.device_mesh.ndim != 1:
                raise ValueError(
                    f"Owned FlexShard parameter {fqn!r} must use a 1D mesh"
                )
            if len(param.placements) != 1 or type(param.placements[0]) is not Shard:
                raise ValueError(
                    f"Owned FlexShard parameter {fqn!r} must have exactly "
                    "one Shard placement"
                )
            if param.placements[0].dim % param.ndim != 0:
                raise ValueError(
                    f"Owned FlexShard parameter {fqn!r} must use Shard(0), "
                    f"got {param.placements[0]}"
                )

            ranks = _mesh_ranks(param.device_mesh)
            if process_group_ranks is None:
                process_group_ranks = ranks
            elif ranks != process_group_ranks:
                raise ValueError(
                    "Owned FlexShard parameters must use the same process group"
                )

            # FlexShard needs the plain storage shard, not Parameter preservation.
            local_param = param._local_tensor
            if tensor_device is None:
                tensor_device = local_param.device
            elif local_param.device != tensor_device:
                raise ValueError("Owned FlexShard parameters must use the same device")
            if not local_param.is_contiguous():
                raise ValueError(
                    f"Owned FlexShard parameter {fqn!r} must have contiguous "
                    "local Shard(0) storage"
                )
            if tuple(param.stride()) != tuple(
                torch.empty(param.shape, device="meta").stride()
            ):
                raise ValueError(
                    f"Owned FlexShard parameter {fqn!r} must have contiguous "
                    "global storage"
                )

            world_size = param.device_mesh.size()
            shard_numels, shard_offsets = _shard_metadata(
                torch.Size(param.shape), world_size
            )
            group_rank = param.device_mesh.get_local_rank()
            expected_shape = torch.Size(
                (
                    shard_numels[group_rank] // math.prod(param.shape[1:]),
                    *param.shape[1:],
                )
            )
            if local_param.shape != expected_shape:
                raise ValueError(
                    f"Owned FlexShard parameter {fqn!r} has local shape "
                    f"{tuple(local_param.shape)}, expected {tuple(expected_shape)}"
                )

            bindings.append(
                _UnassignedBinding(
                    fqn=fqn,
                    param=param,
                    group_index=group_index,
                    requirement=requirement,
                    global_shape=torch.Size(param.shape),
                    global_stride=tuple(param.stride()),
                    dtype=param.dtype,
                    device=local_param.device,
                    shard_numels=shard_numels,
                    shard_offsets=shard_offsets,
                )
            )

    if not bindings:
        raise ValueError("Owned FlexShard requires at least one parameter")
    return tuple(bindings)


def _resolve_buckets(
    bindings: Sequence[_BucketBindingT],
    specs: Sequence[BucketSpec],
) -> list[list[_BucketBindingT]]:
    if not specs:
        raise ValueError("flex_shard requires at least one BucketSpec")

    resolved = [[] for _ in specs]
    for binding in bindings:
        matches = [
            index
            for index, spec in enumerate(specs)
            if any(
                fnmatch.fnmatchcase(binding.fqn, pattern) for pattern in spec.patterns
            )
        ]
        if not matches:
            raise ValueError(
                f"Optimizer parameter {binding.fqn!r} is not covered by any "
                "BucketSpec"
            )
        if len(matches) != 1:
            names = [specs[index].name or str(index) for index in matches]
            raise ValueError(
                f"Optimizer parameter {binding.fqn!r} matched multiple compute "
                f"buckets: {names}"
            )
        resolved[matches[0]].append(binding)
    return resolved


@dataclass(frozen=True, slots=True)
class _IdentityFlexShardRuntime:
    """A validated no-op plan for pointwise optimizer compute."""

    assignments: tuple[BucketAssignment, ...]


def _build_identity_runtime(
    optimizer: Optimizer,
    specs: Sequence[BucketSpec],
) -> _FlexShardRuntime:
    resolved = _resolve_buckets(_bind_optimizer_fqns(optimizer), specs)
    assignments = tuple(
        BucketAssignment(
            bucket_name=spec.name or str(bucket_index),
            fqn=binding.fqn,
            owner_rank=None,
        )
        for bucket_index, (spec, bindings) in enumerate(
            zip(specs, resolved, strict=True)
        )
        for binding in bindings
    )
    return _IdentityFlexShardRuntime(assignments)


def _assign_balanced_owners(
    bucket_bindings: Sequence[Sequence[_UnassignedBinding]],
    world_size: int,
) -> list[dict[str, int]]:
    """Balance each bucket first, then rotate bucket slots across global ranks."""
    bucket_slots: list[list[tuple[int, list[str]]]] = []
    for bindings in bucket_bindings:
        slot_loads = [0] * world_size
        slot_fqns: list[list[str]] = [[] for _ in range(world_size)]
        heap = [(0, rank) for rank in range(world_size)]
        heapq.heapify(heap)
        for binding in sorted(
            bindings,
            key=lambda item: (-item.global_numel, item.fqn),
        ):
            load, slot = heapq.heappop(heap)
            slot_fqns[slot].append(binding.fqn)
            load += binding.global_numel
            slot_loads[slot] = load
            heapq.heappush(heap, (load, slot))
        bucket_slots.append(
            [(slot_loads[slot], slot_fqns[slot]) for slot in range(world_size)]
        )

    global_heap = [(0, rank) for rank in range(world_size)]
    heapq.heapify(global_heap)
    assignments = []
    for slots in bucket_slots:
        available_ranks = sorted(heapq.heappop(global_heap) for _ in range(world_size))
        bucket_assignment = {}
        for (slot_load, fqns), (rank_load, rank) in zip(
            sorted(slots, key=lambda item: (-item[0], item[1])),
            available_ranks,
            strict=True,
        ):
            for fqn in fqns:
                bucket_assignment[fqn] = rank
            heapq.heappush(global_heap, (rank_load + slot_load, rank))
        assignments.append(bucket_assignment)
    return assignments


def _routing_metadata(
    bindings: tuple[_Binding, ...],
    group_rank: int,
    world_size: int,
) -> tuple[list[int], list[int], tuple[int, ...], dict[tuple[int, int], int]]:
    bindings_by_owner = [
        [
            index
            for index, binding in enumerate(bindings)
            if binding.owner_group_rank == owner
        ]
        for owner in range(world_size)
    ]
    input_split_sizes = []
    local_offsets = [-1] * len(bindings)
    local_cursor = 0
    for owner_bindings in bindings_by_owner:
        split_size = 0
        for binding_index in owner_bindings:
            binding = bindings[binding_index]
            local_offsets[binding_index] = local_cursor
            local_numel = binding.shard_numels[group_rank]
            local_cursor += local_numel
            split_size += local_numel
        input_split_sizes.append(split_size)

    owned_bindings = bindings_by_owner[group_rank]
    output_split_sizes = []
    owner_offsets = {}
    owner_cursor = 0
    for source_rank in range(world_size):
        split_size = 0
        for binding_index in owned_bindings:
            binding = bindings[binding_index]
            owner_offsets[(binding_index, source_rank)] = owner_cursor
            source_numel = binding.shard_numels[source_rank]
            owner_cursor += source_numel
            split_size += source_numel
        output_split_sizes.append(split_size)

    return (
        input_split_sizes,
        output_split_sizes,
        tuple(local_offsets),
        owner_offsets,
    )


def _build_bucket_plan(
    spec: BucketSpec,
    bindings: tuple[_Binding, ...],
) -> _BucketPlan:
    process_group = spec.mesh.get_group()
    group_rank = spec.mesh.get_local_rank()
    world_size = spec.mesh.size()
    local_params = [binding.param._local_tensor for binding in bindings]
    dtype = local_params[0].dtype
    device = local_params[0].device
    if any(param.dtype != dtype for param in local_params):
        raise ValueError(
            f"Compute bucket {spec.name!r} must contain one communication dtype"
        )
    if any(param.device != device for param in local_params):
        raise ValueError(
            f"Compute bucket {spec.name!r} must contain tensors on one device"
        )

    (
        input_split_sizes,
        output_split_sizes,
        local_offsets,
        owner_offsets,
    ) = _routing_metadata(bindings, group_rank, world_size)
    # With one binding per owner, source-rank Shard(0) pieces are already
    # concatenated in global flat-tensor order by the all-to-all.
    owner_buffer_is_compute_layout = len(
        {binding.owner_group_rank for binding in bindings}
    ) == len(bindings)

    return _BucketPlan(
        spec=spec,
        bindings=bindings,
        process_group=process_group,
        group_rank=group_rank,
        world_size=world_size,
        input_split_sizes=input_split_sizes,
        output_split_sizes=output_split_sizes,
        local_offsets=local_offsets,
        owner_offsets=owner_offsets,
        dtype=dtype,
        device=device,
        local_buffer_numel=sum(input_split_sizes),
        owner_buffer_numel=sum(output_split_sizes),
        owner_buffer_is_compute_layout=owner_buffer_is_compute_layout,
    )


class _OwnedFlexShardRuntime:
    def __init__(
        self,
        optimizer: Optimizer,
        bucket_specs: Sequence[BucketSpec],
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Owned FlexShard requires distributed initialization")

        self._max_in_flight_buckets = 2
        self._step_started = False
        self._comm_context: _OwnedFlexShardCommContext | None = None
        self._specs = tuple(bucket_specs)
        if any(spec.mesh.ndim != 1 for spec in self._specs):
            raise ValueError("Owned FlexShard currently requires 1D bucket meshes")
        setup_error = None
        try:
            self._initialize_local(optimizer)
        except Exception as error:
            setup_error = error
        self._synchronize_setup_error(setup_error)
        self._validate_plan_across_ranks(optimizer)

    def _initialize_local(self, optimizer: Optimizer) -> None:
        unassigned = _bind_optimizer_params(optimizer)
        resolved = _resolve_buckets(unassigned, self._specs)

        nonempty_specs = [
            (spec, bindings)
            for spec, bindings in zip(self._specs, resolved, strict=True)
            if bindings
        ]
        expected_ranks = _mesh_ranks(unassigned[0].param.device_mesh)
        for spec, bindings in nonempty_specs:
            if _mesh_ranks(spec.mesh) != expected_ranks:
                raise ValueError(
                    f"Compute bucket {spec.name!r} must use the parameters' "
                    "Shard(0) process group"
                )
            for binding in bindings:
                if _mesh_ranks(binding.param.device_mesh) != _mesh_ranks(spec.mesh):
                    raise ValueError(
                        f"Parameter {binding.fqn!r} does not use compute bucket "
                        f"{spec.name!r}'s mesh"
                    )

        world_size = unassigned[0].param.device_mesh.size()
        owner_maps = _assign_balanced_owners(resolved, world_size)
        plans = []
        assignments = []
        for index, (spec, unassigned_bindings) in enumerate(
            zip(self._specs, resolved, strict=True)
        ):
            if not unassigned_bindings:
                continue
            owner_map = owner_maps[index]
            bindings = tuple(
                _Binding(
                    fqn=binding.fqn,
                    param=binding.param,
                    group_index=binding.group_index,
                    requirement=binding.requirement,
                    owner_group_rank=owner_map[binding.fqn],
                    global_shape=binding.global_shape,
                    global_stride=binding.global_stride,
                    dtype=binding.dtype,
                    device=binding.device,
                    shard_numels=binding.shard_numels,
                    shard_offsets=binding.shard_offsets,
                )
                for binding in sorted(unassigned_bindings, key=lambda item: item.fqn)
            )
            plans.append(_build_bucket_plan(spec, bindings))
            assignments.extend(
                BucketAssignment(
                    bucket_name=spec.name or str(index),
                    fqn=binding.fqn,
                    owner_rank=_mesh_ranks(spec.mesh)[binding.owner_group_rank],
                )
                for binding in bindings
            )

        self._plans = tuple(plans)
        self.assignments = tuple(assignments)
        self._bindings = tuple(
            binding for plan in self._plans for binding in plan.bindings
        )
        self._process_group = self._plans[0].process_group
        self._world_size = self._plans[0].world_size
        self._tensor_device = self._plans[0].device
        self._validate_groups(optimizer)

    def set_max_in_flight_buckets(self, value: int) -> None:
        value_is_valid = (
            not isinstance(value, bool)
            and isinstance(value, int)
            and value in (1, 2)
        )
        local_config = torch.tensor(
            [value if value_is_valid else 0, int(self._step_started)],
            dtype=torch.int64,
            device=self._tensor_device,
        )
        gathered_configs = [
            torch.empty_like(local_config) for _ in range(self._world_size)
        ]
        dist.all_gather(
            gathered_configs,
            local_config,
            group=self._process_group,
        )
        gathered_values = [int(config[0].item()) for config in gathered_configs]
        if any(gathered_value not in (1, 2) for gathered_value in gathered_values):
            raise ValueError("max_in_flight_buckets must be 1 or 2 on every rank")
        if any(bool(config[1].item()) for config in gathered_configs):
            raise RuntimeError(
                "set_max_in_flight_buckets() must be called before the first step"
            )
        if any(gathered_value != value for gathered_value in gathered_values):
            raise RuntimeError("max_in_flight_buckets differs across ranks")
        self._max_in_flight_buckets = value

    def _synchronize_setup_error(self, error: Exception | None) -> None:
        if not self._specs:
            if error is not None:
                raise error
            raise ValueError("flex_shard requires at least one BucketSpec")

        mesh = self._specs[0].mesh
        device = torch.device(mesh.device_type)
        if mesh.device_type == "cuda":
            device = torch.device("cuda", torch.cuda.current_device())
        status = torch.tensor(
            int(error is not None),
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(status, op=dist.ReduceOp.SUM, group=mesh.get_group())
        if status.item():
            if error is not None:
                raise error
            raise RuntimeError(
                "Owned FlexShard setup failed validation on another rank"
            )

    @staticmethod
    def _validate_groups(optimizer: Optimizer) -> None:
        errors = _OwnedFlexShardRuntime._group_validation_errors(optimizer)
        if errors:
            raise ValueError(f"Invalid FlexShard optimizer group: {errors[0]}")

    def _validate_plan_across_ranks(self, optimizer: Optimizer) -> None:
        plan_description = (
            [
                (
                    plan.spec.name,
                    plan.spec.patterns,
                    [
                        (
                            binding.fqn,
                            binding.group_index,
                            tuple(binding.global_shape),
                            binding.global_stride,
                            str(binding.dtype),
                            str(binding.device.type),
                            binding.requirement,
                            binding.owner_group_rank,
                            binding.shard_numels,
                        )
                        for binding in plan.bindings
                    ],
                    _mesh_ranks(plan.spec.mesh),
                    [
                        (
                            input_splits,
                            output_splits,
                            local_offsets,
                            sorted(owner_offsets.items()),
                        )
                        for group_rank in range(plan.world_size)
                        for (
                            input_splits,
                            output_splits,
                            local_offsets,
                            owner_offsets,
                        ) in [
                            _routing_metadata(
                                plan.bindings,
                                group_rank,
                                plan.world_size,
                            )
                        ]
                    ],
                    str(plan.dtype),
                    plan.device.type,
                )
                for plan in self._plans
            ],
            self._group_config_hash(optimizer),
        )
        digest = hashlib.sha256(repr(plan_description).encode("utf-8")).digest()
        plan_hash = int.from_bytes(digest[:7], byteorder="little")
        local_hash = torch.tensor(
            plan_hash, dtype=torch.int64, device=self._tensor_device
        )
        gathered_hashes = [
            torch.empty_like(local_hash) for _ in range(self._world_size)
        ]
        dist.all_gather(
            gathered_hashes,
            local_hash,
            group=self._process_group,
        )
        if any(value.item() != plan_hash for value in gathered_hashes):
            raise RuntimeError("FlexShard plans differ across ranks")

    @staticmethod
    def _group_validation_errors(optimizer: Optimizer) -> list[str]:
        errors = []
        required_methods = (
            "flex_shard_compute_requirement",
            "flex_shard_validate_group",
            "flex_shard_group_signature",
            "flex_shard_init_state",
            "flex_shard_prepare",
            "flex_shard_compute",
            "flex_shard_finalize",
        )
        for method_name in required_methods:
            if not callable(getattr(optimizer, method_name, None)):
                errors.append(
                    f"{type(optimizer).__name__} must implement {method_name}()"
                )
        if errors:
            return errors
        validator = getattr(optimizer, "flex_shard_validate_group", None)
        assert validator is not None
        for group_index, group in enumerate(optimizer.param_groups):
            try:
                validator(group_index, group)
            except Exception as error:
                errors.append(str(error))
        return errors

    @staticmethod
    def _group_config_hash(optimizer: Optimizer) -> int:
        signature_provider = getattr(optimizer, "flex_shard_group_signature", None)
        if signature_provider is None:
            raise TypeError(
                f"{type(optimizer).__name__} must implement "
                "flex_shard_group_signature()"
            )
        signature = [signature_provider(group) for group in optimizer.param_groups]
        digest = hashlib.sha256(repr(signature).encode("utf-8")).digest()
        return int.from_bytes(digest[:7], byteorder="little")

    @staticmethod
    def _compute_owned_update(
        optimizer: Optimizer,
        binding: _Binding,
        compute_input: Tensor,
    ) -> Tensor:
        group = optimizer.param_groups[binding.group_index]
        return cast(FlexShardOptimizer, optimizer).flex_shard_compute(
            compute_input, group
        )

    @staticmethod
    def _validate_local_value(
        value: Tensor,
        expected: Tensor,
        hook_name: str,
    ) -> None:
        if not isinstance(value, torch.Tensor) or isinstance(value, DTensor):
            raise TypeError(f"{hook_name}() must return a plain Tensor")
        if (
            value.shape != expected.shape
            or value.dtype != expected.dtype
            or value.device != expected.device
        ):
            raise ValueError(
                f"{hook_name}() output must match the expected local tensor layout"
            )

    @staticmethod
    def _prepare_local_compute(
        optimizer: Optimizer,
        binding: _Binding,
        grad: DTensor,
        out: Tensor,
    ) -> None:
        group = optimizer.param_groups[binding.group_index]
        compute_optimizer = cast(FlexShardOptimizer, optimizer)
        persistent_state = compute_optimizer.flex_shard_init_state(
            binding.param, grad, group
        )
        if not isinstance(persistent_state, MutableMapping):
            raise TypeError("flex_shard_init_state() must return a mutable mapping")

        local_state = {}
        tensor_state = {}
        for name, value in persistent_state.items():
            if not isinstance(value, torch.Tensor):
                continue
            if not isinstance(value, DTensor):
                raise TypeError(
                    f"Owned FlexShard state {name!r} must be stored as a DTensor"
                )
            local_value = value.to_local()
            local_state[name] = local_value
            tensor_state[name] = (value, local_value, local_value._version)

        result = compute_optimizer.flex_shard_prepare(
            binding.param._local_tensor,
            grad.to_local(),
            local_state,
            group,
            out=out,
        )
        if result is not None:
            raise TypeError("flex_shard_prepare() must write to out and return None")
        if local_state.keys() != tensor_state.keys() or any(
            local_state[name] is not local_value
            for name, (_storage, local_value, _version) in tensor_state.items()
        ):
            raise TypeError("flex_shard_prepare() must update state tensors in place")
        for storage, local_value, version in tensor_state.values():
            if local_value._version != version:
                torch.autograd.graph.increment_version(storage)

    def _active_params(self) -> dict[int, bool]:
        active_by_param = {}
        for binding in self._bindings:
            grad = binding.param.grad
            active_by_param[id(binding.param)] = grad is not None
            if grad is None:
                continue
            if not isinstance(grad, DTensor):
                raise RuntimeError(
                    f"FlexShard gradient for {binding.fqn!r} must be a DTensor"
                )
            if grad.is_sparse:
                raise RuntimeError(
                    f"FlexShard gradient for {binding.fqn!r} must be dense"
                )
            local_grad = grad.to_local()
            local_param = binding.param._local_tensor
            if (
                torch.Size(grad.shape) != binding.global_shape
                or _mesh_ranks(grad.device_mesh)
                != _mesh_ranks(binding.param.device_mesh)
                or grad.placements != binding.param.placements
                or local_grad.shape != local_param.shape
                or local_grad.dtype != local_param.dtype
                or local_grad.device != local_param.device
                or tuple(grad.stride()) != binding.global_stride
                or not local_grad.is_contiguous()
            ):
                raise RuntimeError(
                    f"FlexShard gradient layout for {binding.fqn!r} changed"
                )
        return active_by_param

    def _prepare_bucket(
        self,
        optimizer: Optimizer,
        plan: _BucketPlan,
        active_by_param: dict[int, bool],
        *,
        local_buffer: Tensor | None = None,
        owner_buffer: Tensor | None = None,
    ) -> _BucketWork:
        if local_buffer is None:
            local_buffer = torch.empty(
                plan.local_buffer_numel,
                dtype=plan.dtype,
                device=plan.device,
            )
        if owner_buffer is None:
            owner_buffer = torch.empty(
                plan.owner_buffer_numel,
                dtype=plan.dtype,
                device=plan.device,
            )
        if local_buffer.numel() != plan.local_buffer_numel:
            raise ValueError("FlexShard local buffer has incorrect capacity")
        if owner_buffer.numel() != plan.owner_buffer_numel:
            raise ValueError("FlexShard owner buffer has incorrect capacity")
        for binding_index, binding in enumerate(plan.bindings):
            local_offset = plan.local_offsets[binding_index]
            local_numel = binding.shard_numels[plan.group_rank]
            local_output = local_buffer[local_offset : local_offset + local_numel]
            if not active_by_param[id(binding.param)]:
                local_output.zero_()
                continue
            grad = binding.param.grad
            assert isinstance(grad, DTensor)
            local_param = binding.param._local_tensor
            self._prepare_local_compute(
                optimizer,
                binding,
                grad,
                local_output.view(local_param.shape),
            )

        return _BucketWork(
            plan=plan,
            active_by_param=active_by_param,
            local_buffer=local_buffer,
            owner_buffer=owner_buffer,
        )

    @staticmethod
    def _forward_bucket(work: _BucketWork) -> None:
        plan = work.plan

        dist.all_to_all_single(
            work.owner_buffer,
            work.local_buffer,
            output_split_sizes=plan.output_split_sizes,
            input_split_sizes=plan.input_split_sizes,
            group=plan.process_group,
        )

    def _compute_bucket(
        self,
        optimizer: Optimizer,
        work: _BucketWork,
    ) -> None:
        plan = work.plan
        owner_buffer = work.owner_buffer
        reverse_buffer = owner_buffer
        for binding_index, binding in enumerate(plan.bindings):
            if (
                binding.owner_group_rank != plan.group_rank
                or not work.active_by_param[id(binding.param)]
            ):
                continue
            if plan.owner_buffer_is_compute_layout:
                full_input = owner_buffer.view(binding.global_shape)
            else:
                full_input = torch.empty(
                    binding.global_shape,
                    dtype=owner_buffer.dtype,
                    device=owner_buffer.device,
                )
                flat_input = full_input.view(-1)
                for source_rank in range(plan.world_size):
                    source_numel = binding.shard_numels[source_rank]
                    source_offset = binding.shard_offsets[source_rank]
                    owner_offset = plan.owner_offsets[(binding_index, source_rank)]
                    flat_input[source_offset : source_offset + source_numel].copy_(
                        owner_buffer[owner_offset : owner_offset + source_numel]
                    )
            full_update = self._compute_owned_update(optimizer, binding, full_input)
            self._validate_local_value(
                full_update,
                full_input,
                "flex_shard_compute",
            )
            flat_update = full_update.view(-1)
            if plan.owner_buffer_is_compute_layout:
                reverse_buffer = flat_update
                work.compute_keepalives.append(full_update)
            else:
                for destination_rank in range(plan.world_size):
                    destination_numel = binding.shard_numels[destination_rank]
                    destination_offset = binding.shard_offsets[destination_rank]
                    owner_offset = plan.owner_offsets[(binding_index, destination_rank)]
                    owner_buffer[owner_offset : owner_offset + destination_numel].copy_(
                        flat_update[
                            destination_offset : destination_offset + destination_numel
                        ]
                    )
        work.reverse_buffer = reverse_buffer

    @staticmethod
    def _reverse_bucket(work: _BucketWork) -> None:
        plan = work.plan
        reverse_buffer = work.reverse_buffer
        if reverse_buffer is None:
            raise AssertionError("FlexShard owner compute must precede reverse")

        dist.all_to_all_single(
            work.local_buffer,
            reverse_buffer,
            output_split_sizes=plan.input_split_sizes,
            input_split_sizes=plan.output_split_sizes,
            group=plan.process_group,
        )

    @staticmethod
    def _finalize_bucket(
        optimizer: Optimizer,
        work: _BucketWork,
    ) -> None:
        plan = work.plan
        for binding_index, binding in enumerate(plan.bindings):
            if not work.active_by_param[id(binding.param)]:
                continue
            local_param = binding.param._local_tensor
            local_offset = plan.local_offsets[binding_index]
            local_update = work.local_buffer[
                local_offset : local_offset + local_param.numel()
            ].view(local_param.shape)
            group = optimizer.param_groups[binding.group_index]
            result = cast(FlexShardOptimizer, optimizer).flex_shard_finalize(
                local_param,
                local_update,
                group,
                out=local_param,
            )
            if result is not None:
                raise TypeError(
                    "flex_shard_finalize() must write to out and return None"
                )
            torch.autograd.graph.increment_version(binding.param)

    def _step_bucket(
        self,
        optimizer: Optimizer,
        plan: _BucketPlan,
        active_by_param: dict[int, bool],
    ) -> None:
        if not any(active_by_param[id(binding.param)] for binding in plan.bindings):
            return

        work = self._prepare_bucket(optimizer, plan, active_by_param)
        self._forward_bucket(work)
        self._compute_bucket(optimizer, work)
        self._reverse_bucket(work)
        self._finalize_bucket(optimizer, work)

    def _begin_pipelined_bucket(
        self,
        optimizer: Optimizer,
        plan: _BucketPlan,
        active_by_param: dict[int, bool],
        slot: _BucketBufferSlot,
        caller_stream: torch.Stream,
        context: _OwnedFlexShardCommContext,
    ) -> _BucketWork:
        device_handle = context.device_handle
        transfer_stream = context.transfer_stream
        with device_handle.stream(transfer_stream):
            local_buffer, owner_buffer = slot.buffers(plan)
            work = self._prepare_bucket(
                optimizer,
                plan,
                active_by_param,
                local_buffer=local_buffer,
                owner_buffer=owner_buffer,
            )
            self._forward_bucket(work)
            forward_ready = device_handle.Event()
            forward_ready.record(transfer_stream)
            work.forward_ready = forward_ready

        with device_handle.stream(caller_stream):
            caller_stream.wait_event(forward_ready)
            self._compute_bucket(optimizer, work)
            compute_done = device_handle.Event()
            compute_done.record(caller_stream)
            work.compute_done = compute_done
        return work

    def _complete_pipelined_bucket(
        self,
        optimizer: Optimizer,
        work: _BucketWork,
        context: _OwnedFlexShardCommContext,
    ) -> None:
        compute_done = work.compute_done
        if compute_done is None:
            raise AssertionError("FlexShard compute must precede reverse")

        device_handle = context.device_handle
        transfer_stream = context.transfer_stream
        with device_handle.stream(transfer_stream):
            # Queue the next forward redistribution before this wait. Delaying
            # the lifetime back edge preserves the one-bucket overlap window.
            transfer_stream.wait_event(compute_done)
            self._reverse_bucket(work)
            self._finalize_bucket(optimizer, work)
            done = device_handle.Event()
            done.record(transfer_stream)
            work.done = done

    @staticmethod
    def _release_pipelined_bucket(
        work: _BucketWork,
        caller_stream: torch.Stream,
    ) -> None:
        done = work.done
        if done is None:
            raise AssertionError("FlexShard reverse must precede buffer release")

        # The caller stream is the allocation stream for any distinct tensor
        # returned by flex_shard_compute(). Enqueue the allocator-lifetime back
        # edge before dropping its final Python reference. Do not use
        # Tensor.record_stream() here.
        caller_stream.wait_event(done)
        work.compute_keepalives.clear()
        work.reverse_buffer = None

    def _pipelined_step(
        self,
        optimizer: Optimizer,
        active_plans: list[_BucketPlan],
        active_by_param: dict[int, bool],
    ) -> None:
        if not active_plans:
            return
        if self._comm_context is None:
            self._comm_context = _OwnedFlexShardCommContext.create(
                self._tensor_device
            )
        context = self._comm_context
        device_handle = context.device_handle
        caller_stream = device_handle.current_stream(self._tensor_device)
        context.transfer_stream.wait_stream(caller_stream)

        pending: list[_BucketWork] = []
        try:
            for bucket_index, plan in enumerate(active_plans):
                work = self._begin_pipelined_bucket(
                    optimizer,
                    plan,
                    active_by_param,
                    context.slots[bucket_index % 2],
                    caller_stream,
                    context,
                )
                pending.append(work)
                if len(pending) == 2:
                    oldest = pending.pop(0)
                    self._complete_pipelined_bucket(optimizer, oldest, context)
                    self._release_pipelined_bucket(oldest, caller_stream)

            for work in pending:
                self._complete_pipelined_bucket(optimizer, work, context)
                self._release_pipelined_bucket(work, caller_stream)
        except Exception:
            # Order both streams before releasing compute-owned temporaries.
            # This is an error-path drain, not a host or device synchronization.
            context.transfer_stream.wait_stream(caller_stream)
            caller_stream.wait_stream(context.transfer_stream)
            for work in pending:
                work.compute_keepalives.clear()
                work.reverse_buffer = None
            raise

    def step(self, optimizer: Optimizer, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_started = True

        # TorchTitan executes one SPMD graph and optimizer configuration on all
        # ranks. Rank-identical gradient presence is therefore a runtime
        # precondition, not a per-step control-plane collective.
        active_by_param = self._active_params()
        active_plans = [
            plan
            for plan in self._plans
            if any(
                active_by_param[id(binding.param)] for binding in plan.bindings
            )
        ]
        if self._max_in_flight_buckets == 1:
            for plan in active_plans:
                self._step_bucket(optimizer, plan, active_by_param)
        else:
            self._pipelined_step(optimizer, active_plans, active_by_param)
        return loss


@torch.no_grad()
def _run_owned_flex_shard_step(optimizer: Optimizer, closure=None):
    runtime = optimizer.__dict__["_flex_shard_runtime"]
    assert isinstance(runtime, _OwnedFlexShardRuntime)
    return runtime.step(optimizer, closure)


def _reject_flex_shard_param_group(
    optimizer: Optimizer,
    param_group: dict[str, Any],
) -> None:
    raise RuntimeError(
        f"{type(optimizer).__name__} cannot add parameter groups after "
        "flex_shard plan construction"
    )


def _set_max_in_flight_buckets(
    optimizer: Optimizer,
    value: int,
) -> None:
    runtime = optimizer.__dict__["_flex_shard_runtime"]
    if not isinstance(runtime, _OwnedFlexShardRuntime):
        raise RuntimeError(
            "set_max_in_flight_buckets() requires an Owned FlexShard runtime"
        )
    runtime.set_max_in_flight_buckets(value)


def flex_shard(
    optimizer: _OptimizerT,
    bucket_spec: Sequence[BucketSpec],
) -> _OptimizerT:
    """Build bucketed transformations from storage and compute requirements.

    The optimizer declares its compute placement and implements plain-tensor
    math. FlexShard owns FQN resolution, owner assignment, redistribution,
    temporary buffers, and writeback into persistent storage placements.

    Parameter-group membership, order, names, and compute requirements are
    fixed after plan construction. Persistent parameters and optimizer state
    must retain their planned DTensor storage layouts. Mutable optimizer
    settings and per-parameter gradient presence must remain identical across
    ranks. The training hot path treats these as SPMD preconditions and does
    not launch control-plane collectives to diagnose rank divergence.
    """
    if getattr(optimizer, "_flex_shard_runtime", None) is not None:
        raise RuntimeError(f"{type(optimizer).__name__} is already flex-sharded")

    specs = tuple(bucket_spec)
    if not all(isinstance(spec, BucketSpec) for spec in specs):
        raise TypeError("bucket_spec must contain only BucketSpec objects")

    requirements = tuple(
        _compute_sharding_requirement(optimizer, param, group)
        for group in optimizer.param_groups
        for param in group["params"]
    )
    if not requirements:
        raise ValueError("flex_shard requires at least one optimizer parameter")
    if all(isinstance(requirement, SameAsStorage) for requirement in requirements):
        runtime = _build_identity_runtime(optimizer, specs)
    elif all(isinstance(requirement, Owned) for requirement in requirements):
        runtime = _OwnedFlexShardRuntime(optimizer, specs)
    else:
        raise ValueError(
            "One flex_shard optimizer cannot mix SameAsStorage and Owned "
            "compute requirements"
        )
    optimizer.__dict__["_flex_shard_runtime"] = runtime
    optimizer.add_param_group = MethodType(  # type: ignore[method-assign]
        _reject_flex_shard_param_group,
        optimizer,
    )
    if isinstance(runtime, _OwnedFlexShardRuntime):
        step = Optimizer.profile_hook_step(_run_owned_flex_shard_step)
        optimizer.step = MethodType(step, optimizer)  # type: ignore[method-assign]
        optimizer.set_max_in_flight_buckets = MethodType(  # type: ignore[attr-defined]
            _set_max_in_flight_buckets,
            optimizer,
        )
    return optimizer
