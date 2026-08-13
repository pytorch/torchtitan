# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build tensor-region routes and packed collective schedules."""

from __future__ import annotations

import fnmatch
import hashlib
import math
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeAlias, TypeVar

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._utils import _compute_local_shape_and_global_offset

from .optimizer_reshard import BucketConfig, NoRedistribution


__all__: list[str] = []


_ItemT = TypeVar("_ItemT")


def _bucket_diagnostic_label(
    redistribution_mesh_axis_names: tuple[str, ...] | NoRedistribution,
    fqns: tuple[str, ...],
) -> str:
    return (
        "BucketConfig(redistribution_mesh_axis_names="
        f"{redistribution_mesh_axis_names!r}, fqns={fqns!r})"
    )


def _config_redistribution_mesh_axis_names(
    config: BucketConfig,
) -> tuple[str, ...]:
    axis_names = config.redistribution_mesh_axis_names
    return () if isinstance(axis_names, NoRedistribution) else axis_names


@dataclass(frozen=True, slots=True)
class _RedistributionGroup:
    process_group: dist.ProcessGroup
    participants: tuple[int, ...]
    participant_by_shard_index: tuple[int, ...]
    local_participant: int


@dataclass(frozen=True, slots=True)
class _BucketSpec:
    fqns: tuple[str, ...]
    redistribution_mesh_axis_names: tuple[str, ...] | NoRedistribution
    redistribution_group: _RedistributionGroup | None

    @property
    def diagnostic_label(self) -> str:
        return _bucket_diagnostic_label(
            self.redistribution_mesh_axis_names,
            self.fqns,
        )


def _bind_bucket_configs(
    configs: Sequence[BucketConfig],
    items: Sequence[_ItemT],
    *,
    get_fqn: Callable[[_ItemT], str],
    get_storage_dtensor: Callable[[_ItemT], DTensor],
    get_redistribution_storage_mesh_axes: Callable[[_ItemT], tuple[int, ...]],
    get_bucket_compatibility_key: Callable[[_ItemT], Hashable],
) -> tuple[_BucketSpec, ...]:
    """Bind each explicit physical config to at most one bucket spec."""
    cartesian_meshes: dict[tuple[str, tuple[int, ...]], DeviceMesh] = {}
    redistribution_axis_names_by_config = tuple(
        _config_redistribution_mesh_axis_names(config) for config in configs
    )
    matched_items_by_config: list[list[_ItemT]] = [[] for _ in configs]
    for item in items:
        fqn = get_fqn(item)
        pattern_matches = [
            index
            for index, config in enumerate(configs)
            if any(fnmatch.fnmatchcase(fqn, pattern) for pattern in config.patterns)
        ]
        if not pattern_matches:
            raise ValueError(f"optimizer parameter {fqn!r} must match a bucket pattern")

        resolved_redistribution_axis_names: tuple[str, ...] = ()
        raw_storage_mesh_axes = get_redistribution_storage_mesh_axes(item)
        if raw_storage_mesh_axes:
            storage_dtensor = get_storage_dtensor(item)
            storage_mesh = storage_dtensor.device_mesh
            mesh_axis_names = storage_mesh.mesh_dim_names
            if mesh_axis_names is None:
                raise RuntimeError(
                    "redistributed parameter has no resolved redistribution axes: "
                    f"{fqn!r}"
                )
            if len(set(raw_storage_mesh_axes)) != len(raw_storage_mesh_axes):
                raise RuntimeError(
                    "redistributed parameter has duplicate redistribution axes: "
                    f"{fqn!r}"
                )
            if any(
                mesh_axis < 0 or mesh_axis >= storage_mesh.ndim
                for mesh_axis in raw_storage_mesh_axes
            ):
                raise RuntimeError(
                    "redistributed parameter has an invalid redistribution axis: "
                    f"{fqn!r}"
                )
            resolved_redistribution_axis_names = tuple(
                mesh_axis_names[mesh_axis] for mesh_axis in raw_storage_mesh_axes
            )

        matches = [
            index
            for index in pattern_matches
            if len(redistribution_axis_names_by_config[index])
            == len(resolved_redistribution_axis_names)
            and set(redistribution_axis_names_by_config[index])
            == set(resolved_redistribution_axis_names)
        ]
        if len(matches) != 1:
            declared_alternatives = [
                (index, configs[index].redistribution_mesh_axis_names)
                for index in pattern_matches
            ]
            if not matches:
                raise ValueError(
                    f"optimizer parameter {fqn!r} requires redistribution axes "
                    f"{resolved_redistribution_axis_names}, but its matching "
                    "BucketConfigs declare "
                    f"{declared_alternatives} as (index, redistribution)"
                )
            raise ValueError(
                f"optimizer parameter {fqn!r} matches multiple BucketConfigs "
                f"{matches} for redistribution axes "
                f"{resolved_redistribution_axis_names}"
            )
        matched_items_by_config[matches[0]].append(item)

    specs = []
    for config, redistribution_axis_names, raw_matched_items in zip(
        configs,
        redistribution_axis_names_by_config,
        matched_items_by_config,
        strict=True,
    ):
        matched_items = tuple(sorted(raw_matched_items, key=get_fqn))
        if not matched_items:
            continue
        matched_fqns = tuple(get_fqn(item) for item in matched_items)
        diagnostic_label = _bucket_diagnostic_label(
            config.redistribution_mesh_axis_names,
            matched_fqns,
        )
        reference_mesh: DeviceMesh | None = None
        reference_participant_by_shard_index: tuple[int, ...] | None = None
        reference_dtype: torch.dtype | None = None
        reference_device: torch.device | None = None
        reference_compatibility_key: Hashable | None = None
        reference_fqn: str | None = None
        redistribution_axis_name_set = set(redistribution_axis_names)
        for item in matched_items:
            fqn = get_fqn(item)
            storage_dtensor = get_storage_dtensor(item)
            local_tensor = storage_dtensor.to_local()
            compatibility_key = get_bucket_compatibility_key(item)
            mesh = None
            participant_by_shard_index = None
            if redistribution_axis_names:
                storage_mesh = storage_dtensor.device_mesh
                storage_axis_names = storage_mesh.mesh_dim_names
                if storage_axis_names is None or any(
                    axis_name not in storage_axis_names
                    for axis_name in redistribution_axis_names
                ):
                    raise ValueError(
                        f"{diagnostic_label} redistribution axes "
                        f"{redistribution_axis_names} are not present in parameter "
                        f"{fqn!r} storage mesh axes {storage_axis_names}"
                    )
                if len(redistribution_axis_names) == 1:
                    mesh = storage_mesh[redistribution_axis_names[0]]
                    participant_by_shard_index = _device_mesh_ranks(mesh)
                else:
                    storage_ordered_axis_names = tuple(
                        storage_axis_name
                        for storage_axis_name in storage_axis_names
                        if storage_axis_name in redistribution_axis_name_set
                    )
                    storage_ordered_submesh = storage_mesh[storage_ordered_axis_names]
                    process_group_ranks = tuple(
                        sorted(storage_ordered_submesh.mesh.reshape(-1).tolist())
                    )
                    mesh_key = (storage_mesh.device_type, process_group_ranks)
                    mesh = cartesian_meshes.get(mesh_key)
                    if mesh is None:
                        # Only subgroup members construct this communicator. This
                        # keeps PP stages independent while config order gives
                        # overlapping groups a consistent creation order.
                        process_group = dist.new_group(
                            ranks=process_group_ranks,
                            use_local_synchronization=True,
                            group_desc="flex_shard_cartesian",
                        )
                        assert isinstance(process_group, dist.ProcessGroup)
                        mesh = DeviceMesh.from_group(
                            process_group,
                            storage_mesh.device_type,
                            mesh_dim_names=("flex_shard",),
                        )
                        cartesian_meshes[mesh_key] = mesh
                    declared_axis_permutation = tuple(
                        storage_ordered_axis_names.index(axis_name)
                        for axis_name in redistribution_axis_names
                    )
                    participant_by_shard_index = tuple(
                        storage_ordered_submesh.mesh.permute(declared_axis_permutation)
                        .reshape(-1)
                        .tolist()
                    )

            if reference_dtype is None:
                reference_mesh = mesh
                reference_participant_by_shard_index = participant_by_shard_index
                reference_dtype = local_tensor.dtype
                reference_device = local_tensor.device
                reference_compatibility_key = compatibility_key
                reference_fqn = fqn
                continue
            assert reference_fqn is not None
            if participant_by_shard_index != reference_participant_by_shard_index:
                raise ValueError(
                    f"{diagnostic_label} cannot form one packed collective: "
                    f"{reference_fqn!r} resolves to participant-by-shard-index "
                    f"mapping {reference_participant_by_shard_index}, but {fqn!r} "
                    f"resolves to {participant_by_shard_index}; use separate "
                    "BucketConfigs"
                )
            if (
                local_tensor.dtype != reference_dtype
                or local_tensor.device != reference_device
            ):
                raise ValueError(
                    f"{diagnostic_label} cannot form one packed collective: "
                    f"{reference_fqn!r} uses {reference_dtype} on "
                    f"{reference_device}, but {fqn!r} uses {local_tensor.dtype} "
                    f"on {local_tensor.device}; use separate BucketConfigs"
                )
            if compatibility_key != reference_compatibility_key:
                raise ValueError(
                    f"{diagnostic_label} cannot form one packed collective: "
                    f"{reference_fqn!r} and {fqn!r} have incompatible routes; "
                    "use separate BucketConfigs"
                )

        if reference_mesh is None:
            redistribution_group = None
        else:
            assert reference_participant_by_shard_index is not None
            redistribution_group = _redistribution_group(
                reference_mesh,
                reference_participant_by_shard_index,
            )
        specs.append(
            _BucketSpec(
                fqns=matched_fqns,
                redistribution_mesh_axis_names=config.redistribution_mesh_axis_names,
                redistribution_group=redistribution_group,
            )
        )
    return tuple(specs)


def _resolve_buckets(
    items: Sequence[_ItemT],
    specs: Sequence[_BucketSpec],
    *,
    get_fqn: Callable[[_ItemT], str],
) -> tuple[tuple[_ItemT, ...], ...]:
    resolved: list[list[_ItemT]] = [[] for _ in specs]
    for item in items:
        name = get_fqn(item)
        matches = [index for index, spec in enumerate(specs) if name in spec.fqns]
        if len(matches) != 1:
            raise ValueError(f"optimizer parameter {name!r} must match one bucket")
        resolved[matches[0]].append(item)
    return tuple(tuple(bucket) for bucket in resolved)


@dataclass(frozen=True, slots=True)
class _TensorRegion:
    """One rectangular tensor region."""

    offsets: tuple[int, ...]
    shape: tuple[int, ...]

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True, slots=True)
class _GroupLocalStorage:
    """Storage regions normalized to one transport subgroup's domain."""

    logical_shape: tuple[int, ...]
    regions: tuple[tuple[tuple[int, ...], _TensorRegion], ...]


@dataclass(frozen=True, slots=True)
class _ParticipantPartition:
    """One participant's tensor shape and its global logical regions."""

    participant: int
    tensor_shape: tuple[int, ...]
    logical_regions: tuple[_TensorRegion, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tensor_shape", tuple(self.tensor_shape))
        object.__setattr__(self, "logical_regions", tuple(self.logical_regions))


@dataclass(frozen=True, slots=True)
class _RouteEndpoint:
    """One route endpoint's local tensor region and participants."""

    tensor_region: _TensorRegion
    participants: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "participants", tuple(self.participants))


@dataclass(frozen=True, slots=True)
class _TensorRegionRoute:
    """Map one logical region between differently shaped endpoint tensors."""

    logical_region: _TensorRegion
    source: _RouteEndpoint
    destination: _RouteEndpoint

    def inverse(self) -> _TensorRegionRoute:
        return _TensorRegionRoute(
            logical_region=self.logical_region,
            source=self.destination,
            destination=self.source,
        )


class _RedistributionDirection(Enum):
    STORAGE_TO_COMPUTE = "storage-to-compute"
    COMPUTE_TO_STORAGE = "compute-to-storage"

    def routes(self, plan: _RedistributionPlan) -> tuple[_TensorRegionRoute, ...]:
        if self is _RedistributionDirection.STORAGE_TO_COMPUTE:
            return plan.storage_to_compute_routes
        return plan.compute_to_storage_routes


@dataclass(frozen=True, slots=True)
class _RedistributionPlan:
    """Transport-neutral exact region partitions in both directions."""

    participants: tuple[int, ...]
    logical_shape: tuple[int, ...]
    storage_partitions: tuple[_ParticipantPartition, ...]
    compute_partitions: tuple[_ParticipantPartition, ...]
    storage_to_compute_routes: tuple[_TensorRegionRoute, ...]
    compute_to_storage_routes: tuple[_TensorRegionRoute, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "participants", tuple(self.participants))
        object.__setattr__(self, "logical_shape", tuple(self.logical_shape))
        object.__setattr__(self, "storage_partitions", tuple(self.storage_partitions))
        object.__setattr__(self, "compute_partitions", tuple(self.compute_partitions))
        object.__setattr__(
            self, "storage_to_compute_routes", tuple(self.storage_to_compute_routes)
        )
        object.__setattr__(
            self,
            "compute_to_storage_routes",
            tuple(route.inverse() for route in self.storage_to_compute_routes),
        )
        _validate_redistribution_plan(self)

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


def _require_valid_plan(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"invalid redistribution plan: {message}")


def _validate_redistribution_plan(plan: _RedistributionPlan) -> None:
    """Validate the complete internal plan once, before building schedules."""
    participant_set = set(plan.participants)
    _require_valid_plan(
        len(participant_set) == len(plan.participants),
        "participants must be unique",
    )

    for name, partitions in (
        ("storage", plan.storage_partitions),
        ("compute", plan.compute_partitions),
    ):
        _require_valid_plan(
            tuple(partition.participant for partition in partitions)
            == plan.participants,
            f"{name} partitions must follow the redistribution participants",
        )
        for partition in partitions:
            _validate_participant_logical_partition(
                partition,
                plan.logical_shape,
                direction=f"{name} logical partition {partition.participant}",
            )

    routes = plan.storage_to_compute_routes
    for route in routes:
        _require_valid_plan(
            bool(route.source.participants and route.destination.participants),
            "routes require source and destination participants",
        )
        _require_valid_plan(
            len(set(route.source.participants)) == len(route.source.participants)
            and len(set(route.destination.participants))
            == len(route.destination.participants),
            "route endpoint participants must be unique",
        )
        _require_valid_plan(
            set(route.source.participants) <= participant_set
            and set(route.destination.participants) <= participant_set,
            "route references an unknown participant",
        )
        _require_valid_plan(
            route.logical_region.numel
            == route.source.tensor_region.numel
            == route.destination.tensor_region.numel,
            "route endpoint regions must have equal numel",
        )

    _validate_tensor_region_partition(
        tuple(route.logical_region for route in routes),
        plan.logical_shape,
        direction=_RedistributionDirection.STORAGE_TO_COMPUTE.value,
    )

    storage_by_participant = {
        partition.participant: partition for partition in plan.storage_partitions
    }
    compute_by_participant = {
        partition.participant: partition for partition in plan.compute_partitions
    }
    for participant in plan.participants:
        source_routes = tuple(
            route for route in routes if participant in route.source.participants
        )
        _validate_regions_cover_partition(
            tuple(route.logical_region for route in source_routes),
            storage_by_participant[participant].logical_regions,
            plan.logical_shape,
            direction=f"storage source {participant}",
        )
        _validate_tensor_region_partition(
            tuple(route.source.tensor_region for route in source_routes),
            storage_by_participant[participant].tensor_shape,
            direction=f"storage source tensor {participant}",
        )

        destination_routes = tuple(
            route for route in routes if participant in route.destination.participants
        )
        _validate_regions_cover_partition(
            tuple(route.logical_region for route in destination_routes),
            compute_by_participant[participant].logical_regions,
            plan.logical_shape,
            direction=f"compute destination {participant}",
        )
        _validate_tensor_region_partition(
            tuple(route.destination.tensor_region for route in destination_routes),
            compute_by_participant[participant].tensor_shape,
            direction=f"compute destination tensor {participant}",
        )


def _validate_disjoint_regions_in_bounds(
    regions: tuple[_TensorRegion, ...],
    bounds_shape: tuple[int, ...],
    *,
    direction: str,
) -> None:
    _require_valid_plan(
        not any(size < 0 for size in bounds_shape)
        and not any(
            len(region.offsets) != len(bounds_shape)
            or len(region.shape) != len(bounds_shape)
            or any(
                offset < 0 or size < 0 or offset + size > bounds_size
                for offset, size, bounds_size in zip(
                    region.offsets, region.shape, bounds_shape, strict=True
                )
            )
            for region in regions
        ),
        f"{direction} regions must be in bounds",
    )

    positive_regions = tuple(region for region in regions if region.numel)
    for index, first in enumerate(positive_regions):
        for second in positive_regions[index + 1 :]:
            _require_valid_plan(
                not all(
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
                ),
                "overlapping logical tensor regions are not supported",
            )


def _validate_tensor_region_partition(
    regions: tuple[_TensorRegion, ...],
    logical_shape: tuple[int, ...],
    *,
    direction: str,
) -> None:
    _validate_disjoint_regions_in_bounds(
        regions,
        logical_shape,
        direction=direction,
    )

    _require_valid_plan(
        sum(region.numel for region in regions) == math.prod(logical_shape),
        f"{direction} regions do not cover the logical tensor",
    )


def _validate_participant_logical_partition(
    partition: _ParticipantPartition,
    logical_shape: tuple[int, ...],
    *,
    direction: str,
) -> None:
    _validate_disjoint_regions_in_bounds(
        partition.logical_regions,
        logical_shape,
        direction=direction,
    )
    _require_valid_plan(
        sum(region.numel for region in partition.logical_regions)
        == math.prod(partition.tensor_shape),
        f"{direction} regions do not match the participant tensor shape",
    )


def _tensor_region_intersection_numel(
    first: _TensorRegion, second: _TensorRegion
) -> int:
    intersection = _tensor_region_intersection(first, second)
    return 0 if intersection is None else intersection.numel


def _tensor_region_intersection(
    first: _TensorRegion,
    second: _TensorRegion,
) -> _TensorRegion | None:
    if len(first.shape) != len(second.shape):
        return None
    intersection_offsets = tuple(
        max(first_offset, second_offset)
        for first_offset, second_offset in zip(
            first.offsets,
            second.offsets,
            strict=True,
        )
    )
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
    if not math.prod(intersection_shape):
        return None
    return _TensorRegion(
        offsets=intersection_offsets,
        shape=intersection_shape,
    )


def _validate_regions_cover_partition(
    regions: tuple[_TensorRegion, ...],
    expected: tuple[_TensorRegion, ...],
    logical_shape: tuple[int, ...],
    *,
    direction: str,
) -> None:
    """Validate that nonoverlapping regions exactly cover an expected partition."""
    _validate_disjoint_regions_in_bounds(
        regions,
        logical_shape,
        direction=direction,
    )

    _require_valid_plan(
        sum(region.numel for region in regions)
        == sum(region.numel for region in expected),
        f"{direction} regions do not cover the participant partition",
    )
    for region in regions:
        if not region.numel:
            continue
        _require_valid_plan(
            sum(
                _tensor_region_intersection_numel(region, expected_region)
                for expected_region in expected
            )
            == region.numel,
            f"{direction} regions leave the participant partition",
        )


def _build_whole_tensor_redistribution_plan(
    storage_regions: Sequence[tuple[tuple[int, ...], _TensorRegion]],
    *,
    participants: tuple[int, ...],
    compute_participants: tuple[int, ...],
    logical_shape: tuple[int, ...],
) -> _RedistributionPlan:
    """Build whole-tensor compute from a canonical region-to-holders mapping."""
    participants = tuple(participants)
    compute_participants = tuple(compute_participants)
    participant_set = set(participants)
    _require_valid_plan(
        bool(compute_participants)
        and len(set(compute_participants)) == len(compute_participants)
        and all(participant in participant_set for participant in compute_participants),
        "compute participants must be unique redistribution participants",
    )

    storage_mappings = []
    storage_mapping_by_participant = {}
    for raw_holders, logical_region in storage_regions:
        holders = tuple(raw_holders)
        _require_valid_plan(
            bool(holders)
            and len(set(holders)) == len(holders)
            and all(holder in participant_set for holder in holders),
            "storage region holders must be unique redistribution participants",
        )
        tensor_region = _TensorRegion(
            offsets=(0,) * len(logical_region.shape),
            shape=logical_region.shape,
        )
        storage_mappings.append((holders, logical_region, tensor_region))
        for holder in holders:
            _require_valid_plan(
                holder not in storage_mapping_by_participant,
                "multiple storage regions per participant are not supported",
            )
            storage_mapping_by_participant[holder] = (logical_region, tensor_region)
    _require_valid_plan(
        set(storage_mapping_by_participant) == participant_set,
        "storage regions must cover every redistribution participant",
    )

    storage_partitions = tuple(
        _ParticipantPartition(
            participant=participant,
            tensor_shape=storage_mapping_by_participant[participant][1].shape,
            logical_regions=(storage_mapping_by_participant[participant][0],),
        )
        for participant in participants
    )

    full_region = _TensorRegion(
        offsets=(0,) * len(logical_shape),
        shape=logical_shape,
    )
    compute_participant_set = set(compute_participants)
    compute_partitions = tuple(
        _ParticipantPartition(
            participant=participant,
            tensor_shape=(
                logical_shape if participant in compute_participant_set else (0,)
            ),
            logical_regions=(
                (full_region,) if participant in compute_participant_set else ()
            ),
        )
        for participant in participants
    )
    storage_to_compute_routes = tuple(
        _TensorRegionRoute(
            logical_region=logical_region,
            source=_RouteEndpoint(tensor_region, holders),
            destination=_RouteEndpoint(logical_region, compute_participants),
        )
        for holders, logical_region, tensor_region in storage_mappings
    )
    return _RedistributionPlan(
        participants=participants,
        logical_shape=logical_shape,
        storage_partitions=storage_partitions,
        compute_partitions=compute_partitions,
        storage_to_compute_routes=storage_to_compute_routes,
    )


def _build_owned_redistribution_plan(
    storage_regions: Sequence[tuple[tuple[int, ...], _TensorRegion]],
    *,
    participants: tuple[int, ...],
    owner_rank: int,
    logical_shape: tuple[int, ...],
) -> _RedistributionPlan:
    """Build mirrored routes to one whole-tensor compute owner."""
    return _build_whole_tensor_redistribution_plan(
        storage_regions,
        participants=participants,
        compute_participants=(owner_rank,),
        logical_shape=logical_shape,
    )


def _build_dim0_shard_redistribution_plan(
    storage_regions: Sequence[tuple[tuple[int, ...], _TensorRegion]],
    *,
    participants: tuple[int, ...],
    participant_by_shard_index: tuple[int, ...],
    logical_shape: tuple[int, ...],
) -> _RedistributionPlan:
    """Route arbitrary rectangular storage partitions to a dim-0 shard."""
    participants = tuple(participants)
    participant_by_shard_index = tuple(participant_by_shard_index)
    participant_set = set(participants)
    _require_valid_plan(
        len(participant_by_shard_index) == len(participants)
        and set(participant_by_shard_index) == participant_set,
        "participant-by-shard-index mapping must contain every redistribution "
        "participant",
    )
    storage_endpoints = []
    storage_by_participant = {}
    for raw_holders, logical_region in storage_regions:
        holders = tuple(raw_holders)
        _require_valid_plan(
            bool(holders)
            and len(set(holders)) == len(holders)
            and set(holders) <= participant_set,
            "storage region holders must be unique redistribution participants",
        )
        tensor_region = _TensorRegion(
            offsets=(0,) * len(logical_shape),
            shape=logical_region.shape,
        )
        storage_endpoints.append((holders, logical_region, tensor_region))
        for holder in holders:
            _require_valid_plan(
                holder not in storage_by_participant,
                "multiple storage regions per participant are not supported",
            )
            storage_by_participant[holder] = (logical_region, tensor_region)
    _require_valid_plan(
        set(storage_by_participant) == participant_set,
        "storage regions must cover every redistribution participant",
    )
    storage_partitions = tuple(
        _ParticipantPartition(
            participant=participant,
            tensor_shape=storage_by_participant[participant][1].shape,
            logical_regions=(storage_by_participant[participant][0],),
        )
        for participant in participants
    )

    compute_partitions = []
    compute_endpoints = []
    shard_index_by_participant = {
        participant: index
        for index, participant in enumerate(participant_by_shard_index)
    }
    for participant in participants:
        shard_index = shard_index_by_participant[participant]
        local_dim0, dim0_offset = Shard.local_shard_size_and_offset(
            logical_shape[0],
            len(participants),
            shard_index,
        )
        local_shape = (local_dim0, *logical_shape[1:])
        logical_region = _TensorRegion(
            offsets=(dim0_offset,) + (0,) * (len(logical_shape) - 1),
            shape=local_shape,
        )
        tensor_region = _TensorRegion(
            offsets=(0,) * len(logical_shape),
            shape=local_shape,
        )
        compute_partitions.append(
            _ParticipantPartition(
                participant=participant,
                tensor_shape=local_shape,
                logical_regions=(logical_region,),
            )
        )
        compute_endpoints.append(((participant,), logical_region, tensor_region))

    storage_to_compute_routes = []
    for source_holders, storage_region, storage_tensor_base in storage_endpoints:
        for (
            destination_holders,
            compute_region,
            compute_tensor_base,
        ) in compute_endpoints:
            logical_region = _tensor_region_intersection(
                storage_region,
                compute_region,
            )
            if logical_region is None:
                continue
            storage_tensor_region = _TensorRegion(
                offsets=tuple(
                    base_offset + logical_offset - storage_offset
                    for base_offset, logical_offset, storage_offset in zip(
                        storage_tensor_base.offsets,
                        logical_region.offsets,
                        storage_region.offsets,
                        strict=True,
                    )
                ),
                shape=logical_region.shape,
            )
            compute_tensor_region = _TensorRegion(
                offsets=tuple(
                    base_offset + logical_offset - compute_offset
                    for base_offset, logical_offset, compute_offset in zip(
                        compute_tensor_base.offsets,
                        logical_region.offsets,
                        compute_region.offsets,
                        strict=True,
                    )
                ),
                shape=logical_region.shape,
            )
            storage_to_compute_routes.append(
                _TensorRegionRoute(
                    logical_region=logical_region,
                    source=_RouteEndpoint(storage_tensor_region, source_holders),
                    destination=_RouteEndpoint(
                        compute_tensor_region,
                        destination_holders,
                    ),
                )
            )

    return _RedistributionPlan(
        participants=participants,
        logical_shape=logical_shape,
        storage_partitions=storage_partitions,
        compute_partitions=tuple(compute_partitions),
        storage_to_compute_routes=tuple(storage_to_compute_routes),
    )


@dataclass(frozen=True, slots=True)
class _PackedSpan:
    """Physical packed-buffer location for an endpoint tensor region."""

    region: _TensorRegion
    buffer_offset: int

    @property
    def numel(self) -> int:
        return self.region.numel


@dataclass(frozen=True, slots=True)
class _PackedAllToAllSchedule:
    process_group: dist.ProcessGroup
    participants: tuple[int, ...]
    local_participant: int
    has_remote_transfers: bool
    input_split_sizes: list[int]
    output_split_sizes: list[int]
    input_spans_by_parameter: tuple[tuple[_PackedSpan, ...], ...]
    output_spans_by_parameter: tuple[tuple[_PackedSpan, ...], ...]
    input_buffer_numel: int = field(init=False)
    output_buffer_numel: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_buffer_numel", sum(self.input_split_sizes))
        object.__setattr__(self, "output_buffer_numel", sum(self.output_split_sizes))
        if (
            not self.has_remote_transfers
            and self.input_buffer_numel != self.output_buffer_numel
        ):
            raise RuntimeError("local all-to-all schedule has mismatched buffers")


@dataclass(frozen=True, slots=True)
class _BucketPlanningContext(Generic[_ItemT]):
    items: tuple[_ItemT, ...]
    group: _RedistributionGroup


@dataclass(slots=True)
class _LocalBucketPlan(Generic[_ItemT]):
    items: tuple[_ItemT, ...]


@dataclass(slots=True)
class _RedistributionBucketPlan(Generic[_ItemT]):
    redistributed_items: tuple[_ItemT, ...]
    redistribution_plans: tuple[_RedistributionPlan, ...]
    group: _RedistributionGroup
    storage_to_compute_schedule: _PackedAllToAllSchedule
    compute_to_storage_schedule: _PackedAllToAllSchedule
    dtype: torch.dtype
    device: torch.device


@dataclass(slots=True)
class _BucketOverlapPlan(Generic[_ItemT]):
    redistribution_bucket: _RedistributionBucketPlan[_ItemT]
    local_buckets: tuple[_LocalBucketPlan[_ItemT], ...]


_SingleBucketPlan: TypeAlias = (
    _LocalBucketPlan[_ItemT] | _RedistributionBucketPlan[_ItemT]
)
_BucketExecutionPlan: TypeAlias = _SingleBucketPlan[_ItemT] | _BucketOverlapPlan[_ItemT]


@dataclass(frozen=True, slots=True)
class _BucketPlanningResult(Generic[_ItemT]):
    plans: tuple[_BucketExecutionPlan[_ItemT], ...]
    plan_items: tuple[_ItemT, ...]


def _compose_bucket_overlap_plans(
    plans: Sequence[_SingleBucketPlan[_ItemT]],
) -> tuple[_BucketExecutionPlan[_ItemT], ...]:
    """Overlap adjacent local buckets with redistribution when one exists."""
    first_redistribution_index = next(
        (
            index
            for index, plan in enumerate(plans)
            if isinstance(plan, _RedistributionBucketPlan)
        ),
        None,
    )
    if first_redistribution_index is None:
        return tuple(plans)

    leading_local_buckets = []
    for plan in plans[:first_redistribution_index]:
        assert isinstance(plan, _LocalBucketPlan)
        leading_local_buckets.append(plan)

    overlap_plans: list[_BucketExecutionPlan[_ItemT]] = []
    plan_index = first_redistribution_index
    while plan_index < len(plans):
        plan = plans[plan_index]
        assert isinstance(plan, _RedistributionBucketPlan)

        local_buckets = leading_local_buckets
        leading_local_buckets = []
        next_plan_index = plan_index + 1
        while next_plan_index < len(plans) and isinstance(
            plans[next_plan_index], _LocalBucketPlan
        ):
            local_buckets.append(plans[next_plan_index])
            next_plan_index += 1
        if local_buckets:
            overlap_plans.append(
                _BucketOverlapPlan(
                    redistribution_bucket=plan,
                    local_buckets=tuple(local_buckets),
                )
            )
        else:
            overlap_plans.append(plan)
        plan_index = next_plan_index
    return tuple(overlap_plans)


def _flatten_bucket_plan_items(
    plans: Sequence[_BucketExecutionPlan[_ItemT]],
) -> tuple[_ItemT, ...]:
    items = []
    for plan in plans:
        if isinstance(plan, _LocalBucketPlan):
            items.extend(plan.items)
        elif isinstance(plan, _BucketOverlapPlan):
            for local_bucket in plan.local_buckets:
                items.extend(local_bucket.items)
            items.extend(plan.redistribution_bucket.redistributed_items)
        else:
            items.extend(plan.redistributed_items)
    return tuple(items)


def _resolve_routes_to_transfers(
    routes: tuple[_TensorRegionRoute, ...], participants: tuple[int, ...]
) -> tuple[tuple[int, int, _TensorRegion, _TensorRegion], ...]:
    participant_order = {
        participant: index for index, participant in enumerate(participants)
    }
    transfers = []
    for route in routes:
        sources = tuple(
            sorted(route.source.participants, key=participant_order.__getitem__)
        )
        for destination in route.destination.participants:
            source = destination if destination in sources else sources[0]
            transfers.append(
                (
                    source,
                    destination,
                    route.source.tensor_region,
                    route.destination.tensor_region,
                )
            )
    return tuple(transfers)


def _packed_spans_by_parameter(
    indexed_spans: list[tuple[int, _PackedSpan]], num_parameters: int
) -> tuple[tuple[_PackedSpan, ...], ...]:
    return tuple(
        tuple(
            span
            for span_parameter_index, span in indexed_spans
            if span_parameter_index == parameter_index
        )
        for parameter_index in range(num_parameters)
    )


def _build_packed_all_to_all_schedule(
    redistribution_plans: tuple[_RedistributionPlan, ...],
    *,
    direction: _RedistributionDirection,
    process_group: dist.ProcessGroup,
    local_participant: int,
) -> _PackedAllToAllSchedule:
    """Build this participant's packed all-to-all schedule from route plans."""
    participants = redistribution_plans[0].participants
    routes_by_parameter = tuple(direction.routes(plan) for plan in redistribution_plans)
    transfers_by_parameter = tuple(
        _resolve_routes_to_transfers(routes, participants)
        for routes in routes_by_parameter
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
                source_region,
                _destination_region,
            ) in transfers:
                if source != local_participant or transfer_destination != destination:
                    continue
                input_spans.append(
                    (parameter_index, _PackedSpan(source_region, input_cursor))
                )
                input_cursor += source_region.numel
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
                _source_region,
                destination_region,
            ) in transfers:
                if transfer_source != source or destination != local_participant:
                    continue
                output_spans.append(
                    (parameter_index, _PackedSpan(destination_region, output_cursor))
                )
                output_cursor += destination_region.numel
        output_split_sizes.append(output_cursor - split_start)

    return _PackedAllToAllSchedule(
        process_group=process_group,
        participants=participants,
        local_participant=local_participant,
        has_remote_transfers=any(
            source != destination
            for transfers in transfers_by_parameter
            for source, destination, _source_region, _destination_region in transfers
        ),
        input_split_sizes=input_split_sizes,
        output_split_sizes=output_split_sizes,
        input_spans_by_parameter=_packed_spans_by_parameter(
            input_spans, len(redistribution_plans)
        ),
        output_spans_by_parameter=_packed_spans_by_parameter(
            output_spans, len(redistribution_plans)
        ),
    )


def _validate_in_place_compute_exchange(
    storage_to_compute: _PackedAllToAllSchedule,
    compute_to_storage: _PackedAllToAllSchedule,
) -> None:
    """Ensure writeback cannot overwrite another item's unread input."""
    reads = storage_to_compute.output_spans_by_parameter
    writes = compute_to_storage.input_spans_by_parameter
    for parameter_index, parameter_writes in enumerate(writes):
        future_reads = (
            span
            for parameter_reads in reads[parameter_index + 1 :]
            for span in parameter_reads
        )
        for read in future_reads:
            for write in parameter_writes:
                _require_valid_planner_result(
                    write.buffer_offset + write.numel <= read.buffer_offset
                    or read.buffer_offset + read.numel <= write.buffer_offset,
                    "compute writeback overlaps unread exchange input",
                )


def _device_mesh_ranks(mesh: DeviceMesh) -> tuple[int, ...]:
    # Process groups can canonicalize rank order, but DeviceMesh order defines
    # which logical shard each global rank holds.
    return tuple(mesh.mesh.flatten().tolist())


def _redistribution_group(
    mesh: DeviceMesh,
    participant_by_shard_index: tuple[int, ...],
) -> _RedistributionGroup:
    process_group = mesh.get_group()
    participants = tuple(dist.get_process_group_ranks(process_group))
    if len(participant_by_shard_index) != len(participants) or set(
        participant_by_shard_index
    ) != set(participants):
        raise ValueError("bucket mesh and process group participants do not match")
    return _RedistributionGroup(
        process_group=process_group,
        participants=participants,
        participant_by_shard_index=participant_by_shard_index,
        local_participant=participants[dist.get_rank(process_group)],
    )


def _dtensor_mesh_coordinate(
    tensor: DTensor,
    participant: int,
) -> list[int]:
    mesh_shape = tuple(tensor.device_mesh.shape)
    flat_mesh_index = _device_mesh_ranks(tensor.device_mesh).index(participant)
    mesh_coordinate = [0] * len(mesh_shape)
    for mesh_axis in range(len(mesh_shape) - 1, -1, -1):
        flat_mesh_index, mesh_coordinate[mesh_axis] = divmod(
            flat_mesh_index, mesh_shape[mesh_axis]
        )
    return mesh_coordinate


def _dtensor_storage_region_for_coordinate(
    tensor: DTensor,
    mesh_coordinate: list[int],
) -> _TensorRegion:
    local_shape, global_offsets = _compute_local_shape_and_global_offset(
        tensor.shape,
        tensor.device_mesh.shape,
        mesh_coordinate,
        tensor.placements,
    )
    return _TensorRegion(
        offsets=tuple(global_offsets),
        shape=tuple(local_shape),
    )


def _dtensor_storage_regions(
    tensor: DTensor,
    participants: tuple[int, ...],
    *,
    required_storage_mesh_axes: tuple[int, ...],
) -> _GroupLocalStorage:
    storage_mesh = tensor.device_mesh
    storage_ranks = storage_mesh.mesh
    if (
        not required_storage_mesh_axes
        or tuple(sorted(set(required_storage_mesh_axes))) != required_storage_mesh_axes
        or required_storage_mesh_axes[0] < 0
        or required_storage_mesh_axes[-1] >= storage_mesh.ndim
    ):
        raise ValueError(
            "required storage mesh axes must be unique, ordered, and in range"
        )
    reference_locations = (storage_ranks == participants[0]).nonzero()
    if tuple(reference_locations.shape) != (1, storage_mesh.ndim):
        raise ValueError(
            "bucket mesh participants must belong to the DTensor storage mesh"
        )

    reference_coordinate = reference_locations[0].tolist()
    transport_coordinate = list(reference_coordinate)
    for storage_mesh_axis in required_storage_mesh_axes:
        transport_coordinate[storage_mesh_axis] = slice(None)
    cartesian_participants = tuple(
        storage_ranks[tuple(transport_coordinate)].flatten().tolist()
    )
    participant_set = set(participants)
    if (
        len(cartesian_participants) != len(participants)
        or set(cartesian_participants) != participant_set
    ):
        raise ValueError(
            "bucket mesh participants do not match the required Cartesian "
            "storage-mesh axes"
        )

    transport_axis_set = set(required_storage_mesh_axes)
    for mesh_axis, placement in enumerate(tensor.placements):
        if mesh_axis in transport_axis_set:
            if type(placement) not in (Replicate, Shard):
                raise ValueError(
                    "redistributed optimizer storage requires exact Shard or "
                    "Replicate on the communication mesh axes"
                )
        elif storage_mesh.size(mesh_axis) != 1 and type(placement) not in (
            Replicate,
            Shard,
        ):
            raise ValueError(
                "redistributed optimizer storage requires exact Shard or "
                "Replicate outside the communication mesh axes"
            )

    domain_placements = list(tensor.placements)
    for storage_mesh_axis in required_storage_mesh_axes:
        domain_placements[storage_mesh_axis] = Replicate()
    domain_shape, domain_offsets = _compute_local_shape_and_global_offset(
        tensor.shape,
        storage_mesh.shape,
        reference_coordinate,
        domain_placements,
    )

    holders_by_region: dict[_TensorRegion, list[int]] = {}
    for participant in participants:
        coordinate = _dtensor_mesh_coordinate(tensor, participant)
        global_region = _dtensor_storage_region_for_coordinate(tensor, coordinate)
        region = _TensorRegion(
            offsets=tuple(
                offset - domain_offset
                for offset, domain_offset in zip(
                    global_region.offsets,
                    domain_offsets,
                    strict=True,
                )
            ),
            shape=global_region.shape,
        )
        holders_by_region.setdefault(region, []).append(participant)
    regions = tuple(
        (tuple(holders), region) for region, holders in holders_by_region.items()
    )
    _validate_tensor_region_partition(
        tuple(region for _holders, region in regions),
        tuple(domain_shape),
        direction="transport-subgroup storage",
    )
    return _GroupLocalStorage(
        logical_shape=tuple(domain_shape),
        regions=regions,
    )


def _require_valid_planner_result(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"invalid redistribution planner result: {message}")


def _build_bucket_plans(
    items: Sequence[_ItemT],
    specs: Sequence[_BucketSpec],
    *,
    get_fqn: Callable[[_ItemT], str],
    get_storage_dtensor: Callable[[_ItemT], DTensor],
    requires_redistribution: Callable[[_ItemT], bool],
    resolve_redistribution_plans: Callable[
        [tuple[_BucketPlanningContext[_ItemT], ...]],
        Sequence[Sequence[_RedistributionPlan | None]],
    ],
) -> _BucketPlanningResult[_ItemT]:
    """Build ordered optimizer bucket plans with and without redistribution.

    Every bound spec is either entirely local or entirely redistributed.
    Local buckets bypass optimizer-specific redistribution planning. For every
    communication bucket, ``resolve_redistribution_plans`` owns compute
    placement and returns one transport-neutral plan per item. Physical bucket
    membership and redistribution requirements must be rank-stable within
    every communication group.
    """
    resolved = _resolve_buckets(items, specs, get_fqn=get_fqn)
    buckets = []
    for spec, bucket in zip(specs, resolved, strict=True):
        if not bucket:
            continue
        sorted_bucket = tuple(sorted(bucket, key=get_fqn))
        redistribution_requirements = tuple(
            requires_redistribution(item) for item in sorted_bucket
        )
        buckets.append((spec, sorted_bucket, redistribution_requirements))
    contexts_list = []
    for spec, bucket, redistribution_requirements in buckets:
        if spec.redistribution_group is None:
            if any(redistribution_requirements):
                raise ValueError(
                    f"{spec.diagnostic_label} requires a communication mesh for "
                    "redistribution"
                )
            continue
        if not all(redistribution_requirements):
            raise ValueError(
                f"{spec.diagnostic_label} with a communication mesh contains "
                "compute-ready parameters"
            )
        contexts_list.append(
            _BucketPlanningContext(
                items=bucket,
                group=spec.redistribution_group,
            )
        )
    contexts = tuple(contexts_list)
    item_plans_by_bucket = (
        tuple(resolve_redistribution_plans(contexts)) if contexts else ()
    )
    _require_valid_planner_result(
        len(item_plans_by_bucket) == len(contexts),
        "expected one entry per bucket",
    )

    single_bucket_plans: list[_SingleBucketPlan[_ItemT]] = []
    redistribution_bucket_index = 0
    for spec, bucket, redistribution_requirements in buckets:
        if not any(redistribution_requirements):
            single_bucket_plans.append(_LocalBucketPlan(items=bucket))
            continue

        context = contexts[redistribution_bucket_index]
        diagnostic_label = spec.diagnostic_label
        raw_item_plans = item_plans_by_bucket[redistribution_bucket_index]
        redistribution_bucket_index += 1
        item_plans = tuple(raw_item_plans)
        _require_valid_planner_result(
            len(item_plans) == len(context.items),
            f"{diagnostic_label} expected one entry per item",
        )
        group = context.group

        redistributed_items_list = []
        redistribution_plans = []
        for item, item_plan in zip(
            context.items,
            item_plans,
            strict=True,
        ):
            if item_plan is None:
                raise RuntimeError(
                    "invalid redistribution planner result: "
                    f"{diagnostic_label} omitted required redistribution "
                    f"for parameter {get_fqn(item)!r}"
                )
            _require_valid_planner_result(
                item_plan.participants == group.participants,
                f"{diagnostic_label} participants do not match its process group",
            )
            local_tensor = get_storage_dtensor(item).to_local()
            storage_partition = item_plan.storage_partition(group.local_participant)
            _require_valid_planner_result(
                tuple(local_tensor.shape) == storage_partition.tensor_shape,
                f"{diagnostic_label} storage partition does not match its mesh",
            )
            redistributed_items_list.append(item)
            redistribution_plans.append(item_plan)

        redistributed_items = tuple(redistributed_items_list)
        assert redistributed_items

        storage_dtensors = [get_storage_dtensor(item) for item in redistributed_items]
        local_tensors = [tensor.to_local() for tensor in storage_dtensors]
        dtype = local_tensors[0].dtype
        device = local_tensors[0].device
        if any(
            tensor.dtype != dtype or tensor.device != device for tensor in local_tensors
        ):
            raise ValueError(f"{diagnostic_label} mixes dtype or device")

        redistribution_plans_tuple = tuple(redistribution_plans)
        storage_to_compute_schedule = _build_packed_all_to_all_schedule(
            redistribution_plans_tuple,
            direction=_RedistributionDirection.STORAGE_TO_COMPUTE,
            process_group=group.process_group,
            local_participant=group.local_participant,
        )
        compute_to_storage_schedule = _build_packed_all_to_all_schedule(
            redistribution_plans_tuple,
            direction=_RedistributionDirection.COMPUTE_TO_STORAGE,
            process_group=group.process_group,
            local_participant=group.local_participant,
        )
        _validate_in_place_compute_exchange(
            storage_to_compute_schedule,
            compute_to_storage_schedule,
        )
        single_bucket_plans.append(
            _RedistributionBucketPlan(
                redistributed_items=redistributed_items,
                redistribution_plans=redistribution_plans_tuple,
                group=group,
                storage_to_compute_schedule=storage_to_compute_schedule,
                compute_to_storage_schedule=compute_to_storage_schedule,
                dtype=dtype,
                device=device,
            )
        )

    plans = _compose_bucket_overlap_plans(single_bucket_plans)
    return _BucketPlanningResult(
        plans=plans,
        plan_items=_flatten_bucket_plan_items(plans),
    )


def _validate_bucket_plans_across_ranks(
    plans: Sequence[_BucketExecutionPlan[_ItemT]],
    *,
    item_signature: Callable[[_ItemT], tuple[Any, ...]],
) -> None:
    """Collectively verify rank-stable redistribution plans.

    Entirely local plans cannot desynchronize a runtime collective and do not
    require a process group. Every rank must provide redistribution plans in
    the same process-group order so all workers enter these validation
    collectives in the same sequence. Overlapped local-bucket membership is
    intentionally excluded because it may differ across ranks on an
    orthogonal mesh and has no collective ordering requirement.
    """
    for plan in plans:
        if isinstance(plan, _LocalBucketPlan):
            continue
        if isinstance(plan, _BucketOverlapPlan):
            redistribution_bucket = plan.redistribution_bucket
        else:
            redistribution_bucket = plan
        description = (
            str(redistribution_bucket.dtype),
            redistribution_bucket.device.type,
            tuple(
                _redistribution_plan_key(redistribution_plan)
                for redistribution_plan in redistribution_bucket.redistribution_plans
            ),
            [
                item_signature(item)
                for item in redistribution_bucket.redistributed_items
            ],
        )
        digest = hashlib.sha256(repr(description).encode()).digest()
        plan_hash = int.from_bytes(digest[:7], "little")
        local_hash = torch.tensor(
            plan_hash,
            dtype=torch.int64,
            device=redistribution_bucket.device,
        )
        process_group = redistribution_bucket.group.process_group
        gathered = [
            torch.empty_like(local_hash)
            for _ in range(dist.get_world_size(process_group))
        ]
        dist.all_gather(gathered, local_hash, group=process_group)
        if any(value.item() != plan_hash for value in gathered):
            raise RuntimeError("optimizer bucket plans differ across ranks")


def _redistribution_plan_key(plan: _RedistributionPlan) -> tuple[Any, ...]:
    def partition_key(partition: _ParticipantPartition) -> tuple[Any, ...]:
        return (
            partition.participant,
            partition.tensor_shape,
            tuple(
                (region.offsets, region.shape) for region in partition.logical_regions
            ),
        )

    def route_key(route: _TensorRegionRoute) -> tuple[Any, ...]:
        return (
            route.logical_region.offsets,
            route.logical_region.shape,
            route.source.tensor_region.offsets,
            route.source.tensor_region.shape,
            route.destination.tensor_region.offsets,
            route.destination.tensor_region.shape,
            route.source.participants,
            route.destination.participants,
        )

    return (
        plan.participants,
        plan.logical_shape,
        tuple(map(partition_key, plan.storage_partitions)),
        tuple(map(partition_key, plan.compute_partitions)),
        tuple(map(route_key, plan.storage_to_compute_routes)),
    )
