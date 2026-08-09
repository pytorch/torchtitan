# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muon storage-to-compute transitions and redistribution routes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard

from ..flex_optimizer_reshard import (
    _BucketPlanningContext,
    _build_single_participant_redistribution_plan,
    _dtensor_storage_regions,
    _ParticipantPartition,
    _RedistributionGroup,
    _RedistributionPlan,
    _TensorRegion,
    _TensorRegionRoute,
)
from ..work_assignment import assign_balanced_work


__all__ = ["Owned"]


@dataclass(frozen=True, slots=True)
class Owned:
    """Require complete 2D matrix compute on one balanced participant."""


@dataclass(frozen=True, slots=True)
class _PreparedParameterComputeView:
    compute_view_key: tuple[Any, ...]
    global_compute_shape: torch.Size
    local_storage_view: Tensor | None


@dataclass(frozen=True, slots=True)
class _ParameterComputeLayout:
    fqn: str
    param: DTensor
    group_index: int
    compute_view_key: tuple[Any, ...]
    global_compute_shape: torch.Size
    local_storage_view: Tensor | None
    local_storage_signature: tuple[Any, ...]
    compute_distribution: _ComputeDistribution
    storage_to_compute_transition: _StorageToComputeTransition
    redistribution_storage_mesh_axis: int | None

    @property
    def storage_is_compute_ready(self) -> bool:
        return isinstance(
            self.storage_to_compute_transition, _NoRedistributionTransition
        )


@dataclass(frozen=True, slots=True)
class _NoRedistributionTransition:
    pass


@dataclass(frozen=True, slots=True)
class _OwnedRedistributionTransition:
    pass


@dataclass(frozen=True, slots=True)
class _ShardedRedistributionTransition:
    pass


_StorageToComputeTransition = (
    _NoRedistributionTransition
    | _OwnedRedistributionTransition
    | _ShardedRedistributionTransition
)


@dataclass(frozen=True, slots=True)
class _ShardedCompute:
    dim: int


@dataclass(frozen=True, slots=True)
class _SingleRankCompute:
    pass


_ComputeDistribution = _ShardedCompute | _SingleRankCompute


def _compute_distribution_key(
    distribution: _ComputeDistribution,
) -> tuple[str, ...] | tuple[str, int]:
    if isinstance(distribution, _ShardedCompute):
        return ("shard", distribution.dim)
    assert isinstance(distribution, _SingleRankCompute)
    return ("single_rank",)


@dataclass(frozen=True, slots=True)
class _ResolvedStorageToComputeTransition:
    compute_distribution: _ComputeDistribution
    storage_to_compute_transition: _StorageToComputeTransition
    redistribution_storage_mesh_axis: int | None = None


def _resolve_muon_redistribution_plans(
    contexts: tuple[_BucketPlanningContext[_ParameterComputeLayout], ...],
) -> tuple[tuple[_RedistributionPlan | None, ...], ...]:
    """Resolve Muon compute placements directly into transport plans."""
    cumulative_loads_by_participants: dict[tuple[int, ...], tuple[int, ...]] = {}
    plans_by_bucket = []
    for context in contexts:
        participants = context.group.participants
        initial_loads = cumulative_loads_by_participants.setdefault(
            participants,
            (0,) * len(participants),
        )
        compute_participants, cumulative_loads = _assign_balanced_single_participants(
            context.items,
            participants=participants,
            cumulative_loads=initial_loads,
        )
        cumulative_loads_by_participants[participants] = cumulative_loads
        plans_by_bucket.append(
            tuple(
                _build_parameter_redistribution_plan(
                    layout,
                    context.group,
                    compute_participant,
                )
                for layout, compute_participant in zip(
                    context.items,
                    compute_participants,
                    strict=True,
                )
            )
        )
    return tuple(plans_by_bucket)


def _assign_balanced_single_participants(
    compute_layouts: Sequence[_ParameterComputeLayout],
    *,
    participants: tuple[int, ...],
    cumulative_loads: Sequence[int],
) -> tuple[tuple[int | None, ...], tuple[int, ...]]:
    """Balance single-participant compute within and across ordered buckets."""
    assignments: list[int | None] = [None] * len(compute_layouts)
    candidates = tuple(
        (index, layout)
        for index, layout in enumerate(compute_layouts)
        if isinstance(layout.compute_distribution, _SingleRankCompute)
    )
    candidate_partitions, updated_cumulative_loads = assign_balanced_work(
        candidates,
        num_partitions=len(participants),
        initial_loads=cumulative_loads,
        get_weight=lambda indexed_layout: indexed_layout[1].param.numel(),
        get_stable_key=lambda indexed_layout: indexed_layout[1].fqn,
    )
    for (index, _layout), partition in zip(
        candidates,
        candidate_partitions,
        strict=True,
    ):
        assignments[index] = participants[partition]
    return tuple(assignments), updated_cumulative_loads


def _build_parameter_redistribution_plan(
    compute_layout: _ParameterComputeLayout,
    group: _RedistributionGroup,
    compute_participant: int | None,
) -> _RedistributionPlan | None:
    transition = compute_layout.storage_to_compute_transition
    if isinstance(transition, _NoRedistributionTransition):
        return None

    storage_regions = _dtensor_storage_regions(
        compute_layout.param,
        group.participants,
        required_storage_mesh_axis=(compute_layout.redistribution_storage_mesh_axis),
    )
    if isinstance(transition, _OwnedRedistributionTransition):
        assert compute_participant is not None
        assert compute_participant in group.participants
        return _build_single_participant_redistribution_plan(
            storage_regions,
            participants=group.participants,
            compute_participant=compute_participant,
            logical_shape=tuple(compute_layout.param.shape),
        )

    assert compute_participant is None
    assert isinstance(transition, _ShardedRedistributionTransition)
    if tuple(compute_layout.global_compute_shape) == tuple(compute_layout.param.shape):
        return _build_replicated_to_dim0_shard_plan(
            storage_regions,
            participants=group.participants,
            logical_shape=tuple(compute_layout.global_compute_shape),
        )

    return _build_batched_matrix_redistribution_plan(
        storage_regions,
        participants=group.participants,
        storage_shape=tuple(compute_layout.param.shape),
        compute_shape=tuple(compute_layout.global_compute_shape),
    )


def _build_replicated_to_dim0_shard_plan(
    storage_regions: Sequence[tuple[tuple[int, ...], _TensorRegion]],
    *,
    participants: tuple[int, ...],
    logical_shape: tuple[int, ...],
) -> _RedistributionPlan:
    """Partition replicated native tensor storage along compute dimension 0."""
    full_region = _TensorRegion(
        offsets=(0,) * len(logical_shape),
        shape=logical_shape,
    )
    if tuple(storage_regions) != ((participants, full_region),):
        raise ValueError("dim-0 sharded compute requires replicated storage")

    storage_partitions = tuple(
        _ParticipantPartition(
            participant=participant,
            tensor_shape=logical_shape,
            logical_regions=(full_region,),
        )
        for participant in participants
    )
    compute_partitions = []
    storage_to_compute_routes = []
    compute_to_storage_routes = []
    for participant_index, participant in enumerate(participants):
        local_dim0, dim0_offset = Shard.local_shard_size_and_offset(
            logical_shape[0],
            len(participants),
            participant_index,
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
        if not logical_region.numel:
            continue
        storage_to_compute_routes.append(
            _TensorRegionRoute(
                logical_region=logical_region,
                source_region=logical_region,
                destination_region=tensor_region,
                source_participants=participants,
                destination_participants=(participant,),
            )
        )
        compute_to_storage_routes.append(
            _TensorRegionRoute(
                logical_region=logical_region,
                source_region=tensor_region,
                destination_region=logical_region,
                source_participants=(participant,),
                destination_participants=participants,
            )
        )

    return _RedistributionPlan(
        participants=participants,
        logical_shape=logical_shape,
        storage_partitions=storage_partitions,
        compute_partitions=tuple(compute_partitions),
        storage_to_compute_routes=tuple(storage_to_compute_routes),
        compute_to_storage_routes=tuple(compute_to_storage_routes),
    )


def _build_batched_matrix_redistribution_plan(
    storage_regions: Sequence[tuple[tuple[int, ...], _TensorRegion]],
    *,
    participants: tuple[int, ...],
    storage_shape: tuple[int, ...],
    compute_shape: tuple[int, ...],
) -> _RedistributionPlan:
    """Map flat row storage to sharded matrix batches."""
    if (
        len(storage_shape) != 2
        or len(compute_shape) != 3
        or storage_shape[0] != compute_shape[0] * compute_shape[1]
        or storage_shape[1] != compute_shape[2]
    ):
        raise ValueError("matrix-batch redistribution requires a flattened 2D view")

    num_matrices, matrix_rows, matrix_columns = compute_shape
    storage_by_participant = {}
    storage_endpoints = []
    for holders, logical_region in storage_regions:
        if (
            not holders
            or len(logical_region.shape) != 2
            or logical_region.offsets[1] != 0
            or logical_region.shape[1] != matrix_columns
        ):
            raise ValueError(
                "matrix-batch redistribution requires row-sharded 2D storage"
            )
        tensor_region = _TensorRegion(
            offsets=(0, 0),
            shape=logical_region.shape,
        )
        storage_endpoints.append((holders, logical_region, tensor_region))
        for participant in holders:
            if participant not in participants or participant in storage_by_participant:
                raise ValueError(
                    "matrix-batch storage holders must partition participants"
                )
            storage_by_participant[participant] = (logical_region, tensor_region)

    if set(storage_by_participant) != set(participants):
        raise ValueError("matrix-batch storage must cover every participant")

    storage_partitions = tuple(
        _ParticipantPartition(
            participant=participant,
            tensor_shape=storage_by_participant[participant][1].shape,
            logical_regions=(storage_by_participant[participant][0],),
        )
        for participant in participants
    )

    compute_endpoints = []
    compute_partitions_list = []
    for mesh_rank, participant in enumerate(participants):
        local_num_matrices, matrix_offset = Shard.local_shard_size_and_offset(
            num_matrices,
            len(participants),
            mesh_rank,
        )
        logical_region = _TensorRegion(
            offsets=(matrix_offset * matrix_rows, 0),
            shape=(local_num_matrices * matrix_rows, matrix_columns),
        )
        compute_endpoints.append(((participant,), matrix_offset, local_num_matrices))
        compute_partitions_list.append(
            _ParticipantPartition(
                participant=participant,
                tensor_shape=(local_num_matrices, matrix_rows, matrix_columns),
                logical_regions=(logical_region,),
            )
        )
    compute_partitions = tuple(compute_partitions_list)

    storage_to_compute_routes = []
    compute_to_storage_routes = []
    for source_holders, storage_region, storage_tensor_base in storage_endpoints:
        storage_row_offset = storage_region.offsets[0]
        storage_row_end = storage_row_offset + storage_region.shape[0]
        for (
            destination_holders,
            matrix_offset,
            local_num_matrices,
        ) in compute_endpoints:
            for local_matrix_index in range(local_num_matrices):
                matrix_index = matrix_offset + local_matrix_index
                matrix_row_offset = matrix_index * matrix_rows
                route_row_offset = max(storage_row_offset, matrix_row_offset)
                route_row_end = min(
                    storage_row_end,
                    matrix_row_offset + matrix_rows,
                )
                route_rows = route_row_end - route_row_offset
                if route_rows <= 0:
                    continue

                logical_region = _TensorRegion(
                    offsets=(route_row_offset, 0),
                    shape=(route_rows, matrix_columns),
                )
                storage_tensor_region = _TensorRegion(
                    offsets=(
                        storage_tensor_base.offsets[0]
                        + route_row_offset
                        - storage_row_offset,
                        0,
                    ),
                    shape=(route_rows, matrix_columns),
                )
                compute_tensor_region = _TensorRegion(
                    offsets=(
                        local_matrix_index,
                        route_row_offset - matrix_row_offset,
                        0,
                    ),
                    shape=(1, route_rows, matrix_columns),
                )
                storage_to_compute_routes.append(
                    _TensorRegionRoute(
                        logical_region=logical_region,
                        source_region=storage_tensor_region,
                        destination_region=compute_tensor_region,
                        source_participants=source_holders,
                        destination_participants=destination_holders,
                    )
                )
                compute_to_storage_routes.append(
                    _TensorRegionRoute(
                        logical_region=logical_region,
                        source_region=compute_tensor_region,
                        destination_region=storage_tensor_region,
                        source_participants=destination_holders,
                        destination_participants=source_holders,
                    )
                )

    return _RedistributionPlan(
        participants=participants,
        logical_shape=storage_shape,
        storage_partitions=storage_partitions,
        compute_partitions=compute_partitions,
        storage_to_compute_routes=tuple(storage_to_compute_routes),
        compute_to_storage_routes=tuple(compute_to_storage_routes),
    )


def _resolve_storage_to_compute_transition(
    fqn: str,
    param: DTensor,
    global_compute_shape: torch.Size,
    local_storage_view: Tensor | None,
    compute_placement: object,
) -> _ResolvedStorageToComputeTransition:
    """Validate one storage layout and resolve its concrete compute transition."""
    local = param.to_local()
    if (
        len(global_compute_shape) not in (2, 3)
        or torch.is_complex(param)
        or param.ndim < 2
        or not local.is_contiguous()
    ):
        raise ValueError(f"Muon parameter {fqn!r} has unsupported shape or storage")

    replicated_storage = _has_replicated_storage(param)
    storage_can_redistribute, storage_shard_axis = _redistribution_storage_shard_axis(
        param
    )
    mesh_size = param.device_mesh.size()
    if isinstance(compute_placement, Shard):
        if len(global_compute_shape) == 3 and (
            _normalize_dim(compute_placement.dim, len(global_compute_shape)) == 0
        ):
            if replicated_storage:
                return _ResolvedStorageToComputeTransition(
                    compute_distribution=_ShardedCompute(0),
                    storage_to_compute_transition=(
                        _NoRedistributionTransition()
                        if mesh_size == 1
                        else _ShardedRedistributionTransition()
                    ),
                )
            if (
                local_storage_view is not None
                and local_storage_view.shape[1:] == global_compute_shape[1:]
                and _has_dim0_sharded_storage(param)
            ):
                return _ResolvedStorageToComputeTransition(
                    compute_distribution=_ShardedCompute(0),
                    storage_to_compute_transition=_NoRedistributionTransition(),
                )
    elif (
        isinstance(compute_placement, Owned)
        and len(global_compute_shape) == 2
        and param.ndim == 2
    ):
        if storage_can_redistribute:
            return _ResolvedStorageToComputeTransition(
                compute_distribution=_SingleRankCompute(),
                storage_to_compute_transition=(
                    _NoRedistributionTransition()
                    if mesh_size == 1
                    else _OwnedRedistributionTransition()
                ),
                redistribution_storage_mesh_axis=storage_shard_axis,
            )
    raise ValueError(f"unsupported storage-to-compute layout for {fqn!r}")


def _has_replicated_storage(param: DTensor) -> bool:
    return all(type(placement) is Replicate for placement in param.placements)


def _has_dim0_sharded_storage(param: DTensor) -> bool:
    has_shard = False
    for placement in param.placements:
        # FSDP2 emits _StridedShard when a later TP/EP axis already shards
        # this dimension. Keep the allowlist exact so new placements fail closed.
        if type(placement) in (Shard, _StridedShard):
            shard = cast(Shard | _StridedShard, placement)
            if shard.dim % param.ndim != 0:
                return False
            has_shard = True
        elif type(placement) is not Replicate:
            return False
    return has_shard


def _redistribution_storage_shard_axis(param: DTensor) -> tuple[bool, int | None]:
    """Recognize storage replicated outside at most one exact Shard axis."""
    storage_shard_axis = None
    for mesh_axis, placement in enumerate(param.placements):
        if type(placement) is Replicate:
            continue
        if type(placement) is not Shard or storage_shard_axis is not None:
            return False, None
        storage_shard_axis = mesh_axis
    return True, storage_shard_axis


def _normalize_dim(dim: int, ndim: int) -> int:
    normalized = dim if dim >= 0 else dim + ndim
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"dimension {dim} is invalid for a rank-{ndim} tensor")
    return normalized
