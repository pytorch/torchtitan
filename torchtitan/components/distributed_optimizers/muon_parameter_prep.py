# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muon parameter views and pre-construction layout preparation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor, Replicate, Shard
from .bucketed_redistribution import (
    _bind_bucket_configs,
    BucketConfig,
    BucketSpec,
)
from .muon import (
    _PreparedParameterComputeView,
    DistributedMuon,
    Owned,
)


__all__ = [
    "BatchedMatrixComputeView",
    "build_distributed_muon",
    "MuonComputeSharding",
]


@dataclass(frozen=True, slots=True)
class BatchedMatrixComputeView:
    """Unflatten a storage dimension into a batch of matrices."""

    num_matrices: int
    matrices_flattened_into_dim: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_matrices, bool)
            or not isinstance(self.num_matrices, int)
            or self.num_matrices <= 0
        ):
            raise ValueError("num_matrices must be a positive integer")
        if isinstance(self.matrices_flattened_into_dim, bool) or not isinstance(
            self.matrices_flattened_into_dim, int
        ):
            raise ValueError("matrices_flattened_into_dim must be an integer")
        if self.matrices_flattened_into_dim != 0:
            raise ValueError("only matrices_flattened_into_dim=0 is supported")

    def _resolve(self, storage_shape: torch.Size) -> _ResolvedBatchedMatrixView:
        if len(storage_shape) != 2:
            raise ValueError("BatchedMatrixComputeView requires rank-2 storage")
        flattened_extent = storage_shape[self.matrices_flattened_into_dim]
        if flattened_extent == 0 or flattened_extent % self.num_matrices:
            raise ValueError(
                f"storage shape {tuple(storage_shape)} is not divisible into "
                f"{self.num_matrices} matrices"
            )
        return _ResolvedBatchedMatrixView(
            matrix_rows=flattened_extent // self.num_matrices,
            matrix_columns=storage_shape[1],
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class MuonComputeSharding:
    """Define the logical Muon compute tensor and its required placement."""

    # Applied before compute placement, so placement dimensions refer to the
    # viewed tensor. A future view_after_placement mode can apply a local view
    # after redistribution; that ordering is not supported yet.
    view_before_placement: BatchedMatrixComputeView | None = None
    placement: Owned | Shard

    def __post_init__(self) -> None:
        if not isinstance(self.placement, (Owned, Shard)):
            raise TypeError("placement must be Owned or Shard")
        if self.view_before_placement is not None and not isinstance(
            self.view_before_placement, BatchedMatrixComputeView
        ):
            raise TypeError(
                "view_before_placement must be a BatchedMatrixComputeView or None"
            )


@dataclass(frozen=True, slots=True)
class _ResolvedBatchedMatrixView:
    matrix_rows: int
    matrix_columns: int

    def compute_shape(self, storage_shape: torch.Size) -> torch.Size:
        if len(storage_shape) != 2:
            raise ValueError("batched-matrix compute view requires rank-2 storage")
        if (
            storage_shape[0] % self.matrix_rows
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


def build_distributed_muon(
    params: Iterable[Tensor] | Iterable[dict[str, Any]],
    *,
    bucket_spec: Sequence[BucketSpec] | None = None,
    bucket_configs: Sequence[BucketConfig] | None = None,
    **kwargs: Any,
) -> DistributedMuon:
    """Prepare parameter views and construct the DistributedMuon runtime."""
    if (bucket_spec is None) == (bucket_configs is None):
        raise ValueError("provide exactly one of bucket_spec or bucket_configs")

    prepared_params = []
    parameters_to_prepare = []
    for param_or_group in params:
        if not isinstance(param_or_group, dict):
            prepared_params.append(param_or_group)
            continue
        group = dict(param_or_group)
        compute_sharding = group.pop("compute_sharding", None)
        if not isinstance(compute_sharding, MuonComputeSharding):
            raise TypeError("compute_sharding must be a MuonComputeSharding")
        compute_view = compute_sharding.view_before_placement
        group["_compute_placement"] = compute_sharding.placement
        raw_params = group.get("params", ())
        group_params = (
            (raw_params,) if isinstance(raw_params, Tensor) else tuple(raw_params)
        )
        raw_param_names = group.get("param_names")
        param_names = (
            () if raw_param_names is None else tuple(raw_param_names)
        )
        if raw_param_names is None or len(group_params) != len(param_names):
            raise ValueError("params and param_names must be aligned")
        group["params"] = group_params
        group["param_names"] = param_names

        for param, fqn in zip(group_params, param_names, strict=True):
            parameters_to_prepare.append((param, fqn, compute_view))
        prepared_params.append(group)

    if bucket_configs is not None:
        storage_by_fqn = {
            fqn: param
            for param, fqn, _compute_view in parameters_to_prepare
            if isinstance(param, DTensor)
        }
        if len(storage_by_fqn) != len(parameters_to_prepare):
            raise TypeError("bucket_configs require named DTensor parameters")
        bucket_spec = _bind_bucket_configs(bucket_configs, storage_by_fqn)
    assert bucket_spec is not None
    bucket_spec = tuple(bucket_spec)

    prepared_compute_views = {}
    for param, fqn, compute_view in parameters_to_prepare:
        global_storage_shape = torch.Size(param.shape)
        if compute_view is not None and any(
            type(placement) not in (Shard, Replicate)
            for placement in getattr(param, "placements", ())
        ):
            raise ValueError(
                f"batched-matrix Muon parameter {fqn!r} requires exact "
                "Shard or Replicate storage placements"
            )
        local_storage = param.to_local() if isinstance(param, DTensor) else param
        compute_storage = (
            local_storage.detach() if isinstance(param, DTensor) else local_storage
        )
        local_storage_shape = torch.Size(local_storage.shape)
        if compute_view is None:
            global_compute_shape = global_storage_shape
            local_compute_tensor = compute_storage
        else:
            resolved_view = compute_view._resolve(global_storage_shape)
            global_compute_shape = resolved_view.compute_shape(
                global_storage_shape
            )
            local_compute_tensor = compute_storage.view(
                resolved_view.compute_shape(local_storage_shape)
            )
        prepared_compute_views[fqn] = _PreparedParameterComputeView(
            global_compute_shape=global_compute_shape,
            local_compute_tensor=local_compute_tensor,
        )

    return DistributedMuon(
        prepared_params,
        bucket_spec=bucket_spec,
        _prepared_compute_views=prepared_compute_views,
        **kwargs,
    )
