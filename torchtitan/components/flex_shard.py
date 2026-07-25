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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING, TypeVar

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torch.optim import Optimizer


if TYPE_CHECKING:
    from torch import Tensor

    from torchtitan.components.muon_adapter import MuonAdapter


__all__ = [
    "build_layer_bucket_specs",
    "BucketAssignment",
    "BucketSpec",
    "FlexShardBackend",
    "flex_shard",
    "get_flex_shard_assignments",
    "register_flex_shard_backend",
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


class FlexShardBackend(Protocol):
    """Optimizer-specific implementation selected by :func:`flex_shard`."""

    @property
    def assignments(self) -> tuple[BucketAssignment, ...]:
        ...


_BackendFactory = Callable[
    [Optimizer, Sequence[BucketSpec]],
    FlexShardBackend,
]
_BACKEND_FACTORIES: dict[type[Optimizer], _BackendFactory] = {}


def register_flex_shard_backend(
    optimizer_type: type[Optimizer],
    factory: _BackendFactory,
) -> None:
    """Register an optimizer's storage-to-compute sharding implementation."""
    _BACKEND_FACTORIES[optimizer_type] = factory


def get_flex_shard_assignments(
    optimizer: Optimizer,
) -> tuple[BucketAssignment, ...]:
    """Return the resolved compute plan for a flex-sharded optimizer."""
    backend = getattr(optimizer, "_flex_shard_backend", None)
    return () if backend is None else backend.assignments


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
    matrix_shape: tuple[int, int] | None
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
    matrix_shape: tuple[int, int] | None
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


def _bind_optimizer_params(
    optimizer: MuonAdapter,
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
                "Bucketed MuonAdapter parameter groups require param_names aligned "
                "with params"
            )
        matrix_shape = group.get("matrix_shape")
        for fqn, param in zip(param_names, params, strict=True):
            if not isinstance(fqn, str) or not fqn:
                raise ValueError(f"Invalid MuonAdapter parameter FQN {fqn!r}")
            if fqn in seen_fqns:
                raise ValueError(f"Duplicate MuonAdapter parameter FQN {fqn!r}")
            seen_fqns.add(fqn)
            if id(param) in seen_param_ids:
                raise ValueError(
                    f"MuonAdapter parameter {fqn!r} appears more than once"
                )
            seen_param_ids.add(id(param))

            if not isinstance(param, DTensor):
                raise ValueError(
                    f"Bucketed MuonAdapter parameter {fqn!r} must be a DTensor"
                )
            if param.ndim < 2:
                raise ValueError(
                    f"Bucketed MuonAdapter parameter {fqn!r} must be at least 2D"
                )
            if param.device_mesh.ndim != 1:
                raise ValueError(
                    f"Bucketed MuonAdapter parameter {fqn!r} must use a 1D mesh"
                )
            if len(param.placements) != 1 or type(param.placements[0]) is not Shard:
                raise ValueError(
                    f"Bucketed MuonAdapter parameter {fqn!r} must have exactly "
                    "one Shard placement"
                )
            if param.placements[0].dim % param.ndim != 0:
                raise ValueError(
                    f"Bucketed MuonAdapter parameter {fqn!r} must use Shard(0), "
                    f"got {param.placements[0]}"
                )
            if torch.is_complex(param):
                raise RuntimeError("Muon does not support complex parameters")
            optimizer._validate_matrix_shape(param, matrix_shape)

            ranks = _mesh_ranks(param.device_mesh)
            if process_group_ranks is None:
                process_group_ranks = ranks
            elif ranks != process_group_ranks:
                raise ValueError(
                    "Bucketed MuonAdapter parameters must use the same process group"
                )

            local_param = param.to_local()
            if tensor_device is None:
                tensor_device = local_param.device
            elif local_param.device != tensor_device:
                raise ValueError(
                    "Bucketed MuonAdapter parameters must use the same device"
                )
            if not local_param.is_contiguous():
                raise ValueError(
                    f"Bucketed MuonAdapter parameter {fqn!r} must have contiguous "
                    "local Shard(0) storage"
                )
            if tuple(param.stride()) != tuple(
                torch.empty(param.shape, device="meta").stride()
            ):
                raise ValueError(
                    f"Bucketed MuonAdapter parameter {fqn!r} must have contiguous "
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
                    f"Bucketed MuonAdapter parameter {fqn!r} has local shape "
                    f"{tuple(local_param.shape)}, expected {tuple(expected_shape)}"
                )

            bindings.append(
                _UnassignedBinding(
                    fqn=fqn,
                    param=param,
                    group_index=group_index,
                    matrix_shape=matrix_shape,
                    global_shape=torch.Size(param.shape),
                    global_stride=tuple(param.stride()),
                    dtype=param.dtype,
                    device=local_param.device,
                    shard_numels=shard_numels,
                    shard_offsets=shard_offsets,
                )
            )

    if not bindings:
        raise ValueError("Bucketed MuonAdapter requires at least one parameter")
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
class _IdentityFlexShardBackend:
    """A validated no-op plan for pointwise optimizer compute."""

    assignments: tuple[BucketAssignment, ...]


def _build_identity_backend(
    optimizer: Optimizer,
    specs: Sequence[BucketSpec],
) -> FlexShardBackend:
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
    return _IdentityFlexShardBackend(assignments)


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
    local_params = [binding.param.to_local() for binding in bindings]
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
    )


class _BucketedOwnedMuonBackend:
    def __init__(
        self,
        optimizer: MuonAdapter,
        bucket_specs: Sequence[BucketSpec],
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "Bucketed MuonAdapter requires distributed initialization"
            )

        self._specs = tuple(bucket_specs)
        if any(spec.mesh.ndim != 1 for spec in self._specs):
            raise ValueError(
                "MuonAdapter flex_shard currently requires 1D bucket meshes"
            )
        setup_error = None
        try:
            self._initialize_local(optimizer)
        except Exception as error:
            setup_error = error
        self._synchronize_setup_error(setup_error)
        self._validate_plan_across_ranks(optimizer)

    def _initialize_local(self, optimizer: MuonAdapter) -> None:
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
                    matrix_shape=binding.matrix_shape,
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
        self._bucket_status = torch.empty(
            1, dtype=torch.int32, device=self._tensor_device
        )
        self._validate_groups(optimizer)

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
                "Bucketed MuonAdapter setup failed validation on another rank"
            )

    @staticmethod
    def _validate_groups(optimizer: MuonAdapter) -> None:
        errors = _BucketedOwnedMuonBackend._group_validation_errors(optimizer)
        if errors:
            raise ValueError(f"Invalid bucketed MuonAdapter group: {errors[0]}")

    def _validate_plan_across_ranks(self, optimizer: MuonAdapter) -> None:
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
                            binding.matrix_shape,
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
            raise RuntimeError("Bucketed MuonAdapter plans differ across ranks")

    @staticmethod
    def _group_validation_errors(optimizer: MuonAdapter) -> list[str]:
        errors = []
        for group_index, group in enumerate(optimizer.param_groups):
            ns_steps = group["ns_steps"]
            if (
                isinstance(ns_steps, bool)
                or not isinstance(ns_steps, int)
                or ns_steps < 0
                or ns_steps >= 100
            ):
                errors.append(
                    f"group {group_index} ns_steps must be an integer in [0, 100), "
                    f"got {ns_steps!r}"
                )
            coefficients = group["ns_coefficients"]
            if (
                not isinstance(coefficients, tuple)
                or len(coefficients) != 3
                or not all(isinstance(value, (int, float)) for value in coefficients)
            ):
                errors.append(
                    f"group {group_index} must have exactly three numeric "
                    "Newton-Schulz coefficients"
                )
            eps = group["eps"]
            if not isinstance(eps, (int, float)) or eps < 0:
                errors.append(
                    f"group {group_index} eps must be non-negative, got {eps!r}"
                )
            for name in ("lr", "momentum", "weight_decay"):
                value = group[name]
                if isinstance(value, torch.Tensor):
                    valid = value.numel() == 1
                else:
                    valid = isinstance(value, (int, float))
                if not valid:
                    errors.append(
                        f"group {group_index} {name} must be a number or scalar "
                        f"tensor, got {value!r}"
                    )
                    continue
                try:
                    nonnegative = bool(value >= 0)
                except (RuntimeError, TypeError, ValueError):
                    nonnegative = False
                if not nonnegative:
                    errors.append(
                        f"group {group_index} {name} must be non-negative, "
                        f"got {value!r}"
                    )
            if not isinstance(group["nesterov"], bool):
                errors.append(f"group {group_index} nesterov must be a bool")
            if group["adjust_lr_fn"] not in (
                None,
                "original",
                "match_rms_adamw",
                "spectral_unclamped",
            ):
                errors.append(
                    f"group {group_index} has unsupported adjust_lr_fn "
                    f"{group['adjust_lr_fn']!r}"
                )
        return errors

    @staticmethod
    def _group_config_hash(optimizer: MuonAdapter) -> int:
        def signature_value(value):
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    return ("invalid_tensor", tuple(value.shape))
                return ("tensor", value.detach().item())
            return value

        signature = [
            (
                signature_value(group["lr"]),
                signature_value(group["weight_decay"]),
                signature_value(group["momentum"]),
                group["nesterov"],
                group["ns_coefficients"],
                group["eps"],
                group["ns_steps"],
                group["adjust_lr_fn"],
                group.get("matrix_shape"),
            )
            for group in optimizer.param_groups
        ]
        digest = hashlib.sha256(repr(signature).encode("utf-8")).digest()
        return int.from_bytes(digest[:7], byteorder="little")

    def _validate_step_inputs(self, optimizer: MuonAdapter) -> dict[int, bool]:
        local_errors = self._group_validation_errors(optimizer)
        local_active = []
        for binding in self._bindings:
            if binding.group_index >= len(optimizer.param_groups):
                local_errors.append(f"parameter group for {binding.fqn!r} was removed")
                local_active.append(False)
                continue
            group = optimizer.param_groups[binding.group_index]
            param_index = next(
                (
                    index
                    for index, param in enumerate(group["params"])
                    if param is binding.param
                ),
                None,
            )
            if param_index is None:
                local_errors.append(f"parameter group for {binding.fqn!r} changed")
            else:
                param_names = group.get("param_names")
                if (
                    param_names is None
                    or len(param_names) != len(group["params"])
                    or param_names[param_index] != binding.fqn
                ):
                    local_errors.append(f"parameter FQN for {binding.fqn!r} changed")
            if group.get("matrix_shape") != binding.matrix_shape:
                local_errors.append(f"matrix_shape for {binding.fqn!r} changed")

            local_param = binding.param.to_local()
            if (
                torch.Size(binding.param.shape) != binding.global_shape
                or tuple(binding.param.stride()) != binding.global_stride
                or binding.param.dtype != binding.dtype
                or local_param.device != binding.device
                or not local_param.is_contiguous()
            ):
                local_errors.append(f"parameter layout for {binding.fqn!r} changed")

            grad = binding.param.grad
            local_active.append(grad is not None)
            if grad is None:
                continue
            if not isinstance(grad, DTensor):
                local_errors.append(f"gradient for {binding.fqn!r} is not a DTensor")
                continue
            if grad.is_sparse:
                local_errors.append(f"gradient for {binding.fqn!r} is sparse")
                continue
            if (
                torch.Size(grad.shape) != binding.global_shape
                or _mesh_ranks(grad.device_mesh)
                != _mesh_ranks(binding.param.device_mesh)
                or grad.placements != binding.param.placements
                or grad.to_local().shape != binding.param.to_local().shape
                or grad.to_local().dtype != binding.param.to_local().dtype
                or grad.to_local().device != binding.param.to_local().device
                or tuple(grad.stride()) != binding.global_stride
                or not grad.to_local().is_contiguous()
            ):
                local_errors.append(f"gradient layout for {binding.fqn!r} changed")

            momentum_buffer = optimizer.state.get(binding.param, {}).get(
                "momentum_buffer"
            )
            if momentum_buffer is not None and (
                not isinstance(momentum_buffer, DTensor)
                or torch.Size(momentum_buffer.shape) != binding.global_shape
                or momentum_buffer.placements != binding.param.placements
                or _mesh_ranks(momentum_buffer.device_mesh)
                != _mesh_ranks(binding.param.device_mesh)
                or momentum_buffer.to_local().shape != binding.param.to_local().shape
                or momentum_buffer.to_local().dtype != binding.dtype
                or momentum_buffer.to_local().device != binding.device
                or tuple(momentum_buffer.stride()) != binding.global_stride
                or not momentum_buffer.to_local().is_contiguous()
            ):
                local_errors.append(f"momentum layout for {binding.fqn!r} changed")

        status = torch.tensor(
            [
                *(int(active) for active in local_active),
                int(bool(local_errors)),
                self._group_config_hash(optimizer),
            ],
            dtype=torch.int64,
            device=self._tensor_device,
        )
        gathered_status = [torch.empty_like(status) for _ in range(self._world_size)]
        dist.all_gather(gathered_status, status, group=self._process_group)
        if any(rank_status[-2].item() for rank_status in gathered_status):
            detail = (
                local_errors[0] if local_errors else "error reported by another rank"
            )
            raise RuntimeError(f"Invalid bucketed MuonAdapter input: {detail}")
        if any(
            rank_status[-1].item() != status[-1].item()
            for rank_status in gathered_status
        ):
            raise RuntimeError(
                "Bucketed MuonAdapter parameter-group settings differ across ranks"
            )

        active_by_param = {}
        for binding_index, binding in enumerate(self._bindings):
            active_values = {
                rank_status[binding_index].item() for rank_status in gathered_status
            }
            if len(active_values) != 1:
                raise RuntimeError(
                    "Bucketed MuonAdapter gradient presence differs across ranks "
                    f"for {binding.fqn!r}"
                )
            active_by_param[id(binding.param)] = active_values == {1}
        return active_by_param

    @staticmethod
    def _compute_full_delta(
        optimizer: MuonAdapter,
        binding: _Binding,
        full_pre: Tensor,
    ) -> Tensor:
        group = optimizer.param_groups[binding.group_index]
        return optimizer._compute_owned_delta(
            full_pre,
            binding.matrix_shape,
            group,
        )

    def _synchronize_bucket_error(
        self,
        plan: _BucketPlan,
        error: Exception | None,
        phase: str,
    ) -> None:
        self._bucket_status.fill_(int(error is not None))
        dist.all_reduce(
            self._bucket_status,
            op=dist.ReduceOp.SUM,
            group=plan.process_group,
        )
        if self._bucket_status.item():
            message = (
                f"Bucketed MuonAdapter {phase} failed for bucket " f"{plan.spec.name!r}"
            )
            if error is not None:
                raise RuntimeError(message) from error
            raise RuntimeError(f"{message} on another rank")

    def _step_bucket(
        self,
        optimizer: MuonAdapter,
        plan: _BucketPlan,
        active_by_param: dict[int, bool],
    ) -> None:
        if not any(active_by_param[id(binding.param)] for binding in plan.bindings):
            return

        local_buffer = None
        owner_buffer = None
        pack_error = None
        try:
            local_buffer = torch.zeros(
                plan.local_buffer_numel,
                dtype=plan.dtype,
                device=plan.device,
            )
            owner_buffer = torch.empty(
                plan.owner_buffer_numel,
                dtype=plan.dtype,
                device=plan.device,
            )
            for binding_index, binding in enumerate(plan.bindings):
                if not active_by_param[id(binding.param)]:
                    continue
                grad = binding.param.grad
                assert isinstance(grad, DTensor)
                group = optimizer.param_groups[binding.group_index]
                pre_ns = optimizer._compute_owned_pre_ns(
                    binding.param,
                    grad,
                    group,
                )
                local_pre = pre_ns.to_local().contiguous().view(-1)
                local_offset = plan.local_offsets[binding_index]
                local_buffer[local_offset : local_offset + local_pre.numel()].copy_(
                    local_pre
                )
        except Exception as error:
            pack_error = error
        self._synchronize_bucket_error(plan, pack_error, "local packing")
        assert local_buffer is not None
        assert owner_buffer is not None

        dist.all_to_all_single(
            owner_buffer,
            local_buffer,
            output_split_sizes=plan.output_split_sizes,
            input_split_sizes=plan.input_split_sizes,
            group=plan.process_group,
        )

        compute_error = None
        try:
            for binding_index, binding in enumerate(plan.bindings):
                if (
                    binding.owner_group_rank != plan.group_rank
                    or not active_by_param[id(binding.param)]
                ):
                    continue
                full_pre = torch.empty(
                    binding.global_shape,
                    dtype=owner_buffer.dtype,
                    device=owner_buffer.device,
                )
                flat_pre = full_pre.view(-1)
                for source_rank in range(plan.world_size):
                    source_numel = binding.shard_numels[source_rank]
                    source_offset = binding.shard_offsets[source_rank]
                    owner_offset = plan.owner_offsets[(binding_index, source_rank)]
                    flat_pre[source_offset : source_offset + source_numel].copy_(
                        owner_buffer[owner_offset : owner_offset + source_numel]
                    )
                full_delta = self._compute_full_delta(optimizer, binding, full_pre)
                flat_delta = full_delta.view(-1)
                for destination_rank in range(plan.world_size):
                    destination_numel = binding.shard_numels[destination_rank]
                    destination_offset = binding.shard_offsets[destination_rank]
                    owner_offset = plan.owner_offsets[(binding_index, destination_rank)]
                    owner_buffer[owner_offset : owner_offset + destination_numel].copy_(
                        flat_delta[
                            destination_offset : destination_offset + destination_numel
                        ]
                    )
        except Exception as error:
            compute_error = error
        self._synchronize_bucket_error(plan, compute_error, "owner compute")

        dist.all_to_all_single(
            local_buffer,
            owner_buffer,
            output_split_sizes=plan.input_split_sizes,
            input_split_sizes=plan.output_split_sizes,
            group=plan.process_group,
        )

        for binding_index, binding in enumerate(plan.bindings):
            if not active_by_param[id(binding.param)]:
                continue
            local_param = binding.param.to_local()
            local_offset = plan.local_offsets[binding_index]
            local_delta = local_buffer[
                local_offset : local_offset + local_param.numel()
            ].view(local_param.shape)
            delta = DTensor.from_local(
                local_delta,
                device_mesh=binding.param.device_mesh,
                placements=binding.param.placements,
                run_check=False,
                shape=binding.global_shape,
                stride=binding.global_stride,
            )
            group = optimizer.param_groups[binding.group_index]
            optimizer._apply_owned_delta(binding.param, delta, group)

    def step(self, optimizer: MuonAdapter, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        active_by_param = self._validate_step_inputs(optimizer)
        for plan in self._plans:
            self._step_bucket(optimizer, plan, active_by_param)
        return loss


def _build_muon_backend(
    optimizer: Optimizer,
    specs: Sequence[BucketSpec],
) -> FlexShardBackend:
    from torchtitan.components.muon_adapter import MuonAdapter

    if not isinstance(optimizer, MuonAdapter):
        raise TypeError("Muon flex_shard backend requires a MuonAdapter")
    return _BucketedOwnedMuonBackend(optimizer, specs)


def flex_shard(
    optimizer: _OptimizerT,
    bucket_spec: Sequence[BucketSpec],
) -> _OptimizerT:
    """Apply an optimizer's registered storage-to-compute sharding plan.

    Unregistered optimizers use an identity backend: compute sharding equals
    storage sharding. Optimizers whose compute needs a different placement
    register a backend that owns temporary redistribution and writeback.
    """
    if getattr(optimizer, "_flex_shard_backend", None) is not None:
        raise RuntimeError(f"{type(optimizer).__name__} is already flex-sharded")

    specs = tuple(bucket_spec)
    if not all(isinstance(spec, BucketSpec) for spec in specs):
        raise TypeError("bucket_spec must contain only BucketSpec objects")

    factory = next(
        (
            _BACKEND_FACTORIES[optimizer_type]
            for optimizer_type in type(optimizer).__mro__
            if optimizer_type in _BACKEND_FACTORIES
        ),
        None,
    )
    if factory is None:
        factory = _build_identity_backend
    backend = factory(optimizer, specs)
    optimizer.__dict__["_flex_shard_backend"] = backend
    return optimizer
