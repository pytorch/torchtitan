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

from ..flex_optimizer_reshard import BucketConfig, BucketSpec
from .distributed_muon import DistributedMuon
from .storage_to_compute import _PreparedParameterComputeView, Owned


__all__ = [
    "BatchedMatrixComputeView",
    "MuonComputeSharding",
    "Owned",
    "build_distributed_muon",
]


@dataclass(frozen=True, slots=True)
class BatchedMatrixComputeView:
    """View 2D storage as matrices with batch and rows flattened into dim 0."""

    num_matrices: int
    matrices_flattened_into_dim: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_matrices, bool)
            or not isinstance(self.num_matrices, int)
            or self.num_matrices <= 0
        ):
            raise ValueError("num_matrices must be a positive integer")
        if self.matrices_flattened_into_dim != 0:
            raise ValueError("only matrices_flattened_into_dim=0 is supported")

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


@dataclass(frozen=True, slots=True, kw_only=True)
class MuonComputeSharding:
    """Define the logical Muon tensor and its compute placement.

    ``Owned`` balances complete 2D matrices across bucket participants.
    ``Shard(0)`` partitions rank-3 matrix batches along
    the matrix-batch dimension. Aligned storage computes locally; otherwise an
    exact one-dimensional ``Shard(0)`` storage layout is repartitioned.
    Storage is redistributed when it does not already match the requested
    compute placement.
    """

    # Applied before compute placement, so placement dimensions refer to the
    # viewed tensor. A future view_after_placement mode can apply a local view
    # after redistribution; that ordering is not supported yet.
    view_before_placement: BatchedMatrixComputeView | None = None
    placement: Owned | Shard

    def __post_init__(self) -> None:
        if type(self.placement) not in (Owned, Shard) or (
            self.view_before_placement is not None
            and type(self.view_before_placement) is not BatchedMatrixComputeView
        ):
            raise TypeError(
                "MuonComputeSharding requires a supported view and placement"
            )

    def to_dict(self) -> dict:
        return {"repr": repr(self)}


def build_distributed_muon(
    params: Iterable[dict[str, Any]],
    *,
    bucket_specs: Sequence[BucketSpec] | None = None,
    bucket_configs: Sequence[BucketConfig] | None = None,
    **kwargs: Any,
) -> DistributedMuon:
    """Prepare named DTensor parameter groups and construct DistributedMuon.

    Every group must provide aligned ``params`` and ``param_names`` plus one
    ``compute_sharding`` contract. Exactly one of ``bucket_specs`` or
    ``bucket_configs`` is required. Parameter groups and layouts are frozen
    after construction because optimizer state and collectives depend on them.
    """
    if (bucket_specs is None) == (bucket_configs is None):
        raise ValueError("provide exactly one of bucket_specs or bucket_configs")

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
        storage_shards_are_matrix_aligned = True
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
                storage_shards_are_matrix_aligned = (
                    _validate_batched_matrix_storage_alignment(
                        fqn,
                        param,
                        resolved_view,
                    )
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
                compute_view.matrices_flattened_into_dim,
            )
            assert resolved_view is not None
            global_compute_shape = torch.Size(
                (
                    compute_view.num_matrices,
                    resolved_view.matrix_rows,
                    resolved_view.matrix_columns,
                )
            )
            local_storage_view = (
                local_storage_for_compute_view.view(
                    resolved_view.compute_shape(local_storage_shape)
                )
                if storage_shards_are_matrix_aligned
                else None
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
        bucket_specs=None if bucket_specs is None else tuple(bucket_specs),
        _bucket_configs=(None if bucket_configs is None else tuple(bucket_configs)),
        _prepared_compute_views=prepared_compute_views,
        **kwargs,
    )


@dataclass(frozen=True, slots=True)
class _ResolvedBatchedMatrixView:
    matrix_rows: int
    matrix_columns: int

    def compute_shape(self, storage_shape: torch.Size) -> torch.Size:
        if (
            len(storage_shape) != 2
            or storage_shape[0] % self.matrix_rows
            or storage_shape[1] != self.matrix_columns
        ):
            raise ValueError(
                f"storage shape {tuple(storage_shape)} is not aligned to "
                f"matrix shape {(self.matrix_rows, self.matrix_columns)}"
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
) -> bool:
    """Validate every storage shard and return whether all preserve matrices."""
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
    storage_is_aligned = True
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
            storage_is_aligned = False

    if not storage_is_aligned and (
        len(param.device_mesh.shape) != 1 or param.placements != (Shard(0),)
    ):
        raise ValueError(
            f"batched-matrix Muon parameter {fqn!r} storage shards are not "
            f"aligned to matrix rows of size {matrix_rows}; unaligned storage "
            "requires an exact one-dimensional Shard(0) placement"
        )
    return storage_is_aligned
