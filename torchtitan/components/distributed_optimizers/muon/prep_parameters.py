# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public configuration and construction for DistributedMuon."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._utils import _compute_local_shape_and_global_offset

from ..flex_optimizer_reshard import BucketConfig
from .distributed_muon import DistributedMuon
from .storage_to_compute import _PreparedParameterComputeView, Owned


__all__ = [
    "BatchedMatrixComputeView",
    "MuonComputeShardingConfig",
    "Owned",
    "build_distributed_muon",
]


@dataclass(frozen=True, slots=True)
class BatchedMatrixComputeView:
    """View 2D storage as matrices with batch and rows flattened into dim 0."""

    num_matrices: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_matrices, bool)
            or not isinstance(self.num_matrices, int)
            or self.num_matrices <= 0
        ):
            raise ValueError("num_matrices must be a positive integer")

    def _resolve(self, storage_shape: torch.Size) -> _ResolvedBatchedMatrixView:
        if (
            len(storage_shape) != 2
            or storage_shape[0] == 0
            or storage_shape[0] % self.num_matrices
        ):
            raise ValueError(
                f"storage shape {tuple(storage_shape)} cannot be viewed as "
                f"{self.num_matrices} matrices"
            )
        return _ResolvedBatchedMatrixView(
            matrix_rows=storage_shape[0] // self.num_matrices,
            matrix_columns=storage_shape[1],
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class MuonComputeShardingConfig:
    """Define the logical Muon tensor and its compute placement.

    ``Owned`` balances complete 2D matrices across bucket participants.
    ``Shard(0)`` partitions rank-3 matrix batches along the batch dimension.
    Storage is redistributed when it does not already match the requested
    compute placement.
    """

    placement: Owned | Shard

    # Applied before compute placement, so placement dimensions refer to the
    # viewed tensor. A future view_after_placement mode can apply a local view
    # after redistribution; that ordering is not supported yet.
    view_before_placement: BatchedMatrixComputeView | None = None

    def __post_init__(self) -> None:
        if type(self.placement) not in (Owned, Shard):
            raise ValueError(
                "MuonComputeShardingConfig.placement must be Owned or Shard"
            )
        if (
            self.view_before_placement is not None
            and type(self.view_before_placement) is not BatchedMatrixComputeView
        ):
            raise ValueError(
                "MuonComputeShardingConfig.view_before_placement must be "
                "BatchedMatrixComputeView or None"
            )

    def to_dict(self) -> dict:
        """Serialize for JSON logging. Placements become repr strings."""
        return {"repr": repr(self)}


def build_distributed_muon(
    params: Iterable[dict[str, Any]],
    *,
    bucket_configs: Sequence[BucketConfig],
    num_pipeline_slots: int = 2,
    **kwargs: Any,
) -> DistributedMuon:
    """Prepare named DTensor parameter groups and construct DistributedMuon.

    Every group must provide aligned ``params`` and ``param_names`` plus one
    ``compute_sharding`` contract. Parameter groups, bucket configuration, and
    layouts are frozen after construction because optimizer state and
    collectives depend on them. ``num_pipeline_slots`` bounds the number of
    redistributed buckets held in the rolling prefetch window.
    """
    if type(num_pipeline_slots) is not int or num_pipeline_slots < 1:
        raise ValueError("num_pipeline_slots must be a positive integer")

    prepared_params = []
    parameters_to_prepare = []
    for param_group in params:
        group = dict(param_group)
        compute_sharding = group.pop("compute_sharding")
        compute_view = compute_sharding.view_before_placement
        group["_compute_placement"] = compute_sharding.placement
        raw_params = group.get("params", ())
        group_params = (
            (raw_params,) if isinstance(raw_params, Tensor) else tuple(raw_params)
        )
        raw_param_names = group.get("param_names")
        param_names = () if raw_param_names is None else tuple(raw_param_names)
        if raw_param_names is None or len(group_params) != len(param_names):
            raise ValueError("params and param_names must be aligned")
        group["params"] = group_params
        group["param_names"] = param_names

        for param, fqn in zip(group_params, param_names, strict=True):
            parameters_to_prepare.append((param, fqn, compute_view))
        prepared_params.append(group)

    prepared_compute_views = {}
    for param, fqn, compute_view in parameters_to_prepare:
        global_storage_shape = torch.Size(param.shape)
        resolved_view = None
        if compute_view is not None and any(
            type(placement) not in (Shard, Replicate)
            for placement in getattr(param, "placements", ())
        ):
            raise ValueError(
                f"batched-matrix Muon parameter {fqn!r} requires exact "
                "Shard or Replicate storage placements"
            )
        if compute_view is not None:
            resolved_view = compute_view._resolve(global_storage_shape)
            if isinstance(param, DTensor):
                _validate_batched_matrix_storage_alignment(
                    fqn,
                    param,
                    resolved_view,
                )
        local_storage = param.to_local() if isinstance(param, DTensor) else param
        local_storage_for_compute_view = (
            local_storage.detach() if isinstance(param, DTensor) else local_storage
        )
        local_storage_shape = torch.Size(local_storage.shape)
        if compute_view is None:
            compute_view_key = ("identity",)
            global_compute_shape = global_storage_shape
            local_storage_view = local_storage_for_compute_view
        else:
            compute_view_key = (
                "batched_matrix",
                compute_view.num_matrices,
                0,
            )
            assert resolved_view is not None
            global_compute_shape = torch.Size(
                (
                    compute_view.num_matrices,
                    resolved_view.matrix_rows,
                    resolved_view.matrix_columns,
                )
            )
            local_storage_view = local_storage_for_compute_view.view(
                resolved_view.compute_shape(local_storage_shape)
            )
        if len(global_compute_shape) not in (2, 3):
            raise ValueError(
                f"Muon parameter {fqn!r} compute shape "
                f"{tuple(global_compute_shape)} must be 2D or batch-first 3D"
            )
        prepared_compute_views[fqn] = _PreparedParameterComputeView(
            compute_view_key=compute_view_key,
            global_compute_shape=global_compute_shape,
            local_storage_view=local_storage_view,
        )
    return DistributedMuon(
        prepared_params,
        _bucket_configs=tuple(bucket_configs),
        _prepared_compute_views=prepared_compute_views,
        _num_pipeline_slots=num_pipeline_slots,
        **kwargs,
    )


@dataclass(frozen=True, slots=True)
class _ResolvedBatchedMatrixView:
    matrix_rows: int
    matrix_columns: int

    def compute_shape(self, storage_shape: torch.Size) -> torch.Size:
        if not (
            len(storage_shape) == 2
            and not storage_shape[0] % self.matrix_rows
            and storage_shape[1] == self.matrix_columns
        ):
            raise RuntimeError(
                "prepared batched-matrix view is internally inconsistent"
            )
        return torch.Size(
            (
                storage_shape[0] // self.matrix_rows,
                self.matrix_rows,
                self.matrix_columns,
            )
        )


def _validate_batched_matrix_storage_alignment(
    fqn: str,
    param: DTensor,
    resolved_view: _ResolvedBatchedMatrixView,
) -> None:
    """Validate every storage shard from globally identical DTensor metadata."""
    for placement in param.placements:
        if type(placement) is Replicate:
            continue
        assert type(placement) is Shard
        if placement.dim % param.ndim != 0:
            raise ValueError(
                f"batched-matrix Muon parameter {fqn!r} requires storage "
                "shards along tensor dimension 0"
            )

    matrix_rows = resolved_view.matrix_rows
    # Every rank must validate all coordinates before DistributedMuon performs
    # collectives; checking only the local shard could strand its peers.
    coordinates = product(
        *(range(mesh_axis_size) for mesh_axis_size in param.device_mesh.shape)
    )
    for coordinate in coordinates:
        local_shape, global_offset = _compute_local_shape_and_global_offset(
            param.shape,
            param.device_mesh.shape,
            list(coordinate),
            param.placements,
        )
        if local_shape[0] and (
            local_shape[0] % matrix_rows or global_offset[0] % matrix_rows
        ):
            raise ValueError(
                f"batched-matrix Muon parameter {fqn!r} storage shards are not "
                f"aligned to matrix rows of size {matrix_rows}"
            )
