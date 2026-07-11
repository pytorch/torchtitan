# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Helpers for torchtitan's spmd_types backend."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from threading import local
from typing import Any, TYPE_CHECKING

import spmd_types as spmd
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Partial, Placement, Replicate, Shard

from torchtitan.distributed.utils import get_spmd_backend

# Avoid circular import: protocols.__init__ imports module.py, which imports us.
if TYPE_CHECKING:
    from torchtitan.distributed.parallel_dims import (
        MeshAxisName,
        ParallelDims,
        SpmdLayout,
    )

__all__ = [
    "annotate_input_spmd_types",
    "current_spmd_mesh",
    "spmd_dense_mesh",
    "spmd_sparse_mesh",
    "spmd_mesh_size",
    "spmd_distribute_tensor",
    "set_current_spmd_mesh",
    "set_spmd_meshes",
    "maybe_set_sparse_mesh",
    "spmd_layout_to_dtensor_placements",
]


_MESH_TLS = local()


def set_spmd_meshes(
    *,
    dense_mesh: DeviceMesh,
    sparse_mesh: DeviceMesh | None,
) -> None:
    """Register the SPMD meshes for dense and sparse runtime regions."""
    _MESH_TLS.dense_mesh = dense_mesh
    _MESH_TLS.sparse_mesh = sparse_mesh


def spmd_dense_mesh() -> DeviceMesh:
    """Return the registered dense SPMD mesh."""
    mesh = getattr(_MESH_TLS, "dense_mesh", None)
    assert mesh is not None, "SPMD dense mesh has not been registered"
    return mesh


def spmd_sparse_mesh() -> DeviceMesh | None:
    """Return the registered sparse SPMD mesh, if EP is enabled."""
    return getattr(_MESH_TLS, "sparse_mesh", None)


def _spmd_mesh_stack() -> list[DeviceMesh | None]:
    stack = getattr(_MESH_TLS, "mesh_stack", None)
    if stack is None:
        stack = []
        _MESH_TLS.mesh_stack = stack
    return stack


def current_spmd_mesh() -> DeviceMesh | None:
    """Return the current runtime mesh, or ``None`` if unset."""
    if get_spmd_backend() != "spmd_types":
        return None
    stack = _spmd_mesh_stack()
    if not stack:
        return None
    return stack[-1]


def spmd_mesh_size(axis_name: str) -> int:
    """Return the size of a mesh axis, or 1 if not active."""
    mesh = current_spmd_mesh()
    if mesh is None:
        return 1
    names = mesh.mesh_dim_names or ()
    if axis_name not in names:
        return 1
    return mesh.size(names.index(axis_name))


@contextlib.contextmanager
def set_current_spmd_mesh(mesh: DeviceMesh | None) -> Iterator[None]:
    """Set TorchTitan and spmd_types current mesh state for one runtime region."""
    assert get_spmd_backend() == "spmd_types", (
        "set_current_spmd_mesh() is only valid under spmd_types backend"
    )

    stack = _spmd_mesh_stack()
    stack.append(mesh)
    if mesh is None:
        try:
            yield
        finally:
            popped = stack.pop()
            assert popped is mesh
        return

    with spmd.set_current_mesh(mesh):
        try:
            yield
        finally:
            popped = stack.pop()
            assert popped is mesh


@contextlib.contextmanager
def maybe_set_sparse_mesh() -> Iterator[None]:
    """Activate the registered sparse mesh under spmd_types, otherwise no-op."""
    if get_spmd_backend() != "spmd_types" or (mesh := spmd_sparse_mesh()) is None:
        yield
        return

    with set_current_spmd_mesh(mesh):
        yield


def spmd_layout_to_dtensor_placements(
    layout: "SpmdLayout",
) -> dict["MeshAxisName", Placement]:
    """Convert an SPMD layout to DTensor placements keyed by mesh axis name."""
    from torchtitan.distributed.parallel_dims import MeshAxisName

    result: dict[MeshAxisName, Placement] = {}
    for axis_name, axis_type in layout.per_axis_spmd_types().items():
        if axis_type == spmd.R or axis_type == spmd.I:
            dtensor_placement: Placement = Replicate()
        elif axis_type == spmd.P:
            dtensor_placement = Partial()
        else:
            assert isinstance(axis_type, spmd.Shard)
            dtensor_placement = Shard(axis_type.dim)

        if axis_name == MeshAxisName.DP:
            result[MeshAxisName.DP_REPLICATE] = dtensor_placement
            result[MeshAxisName.DP_SHARD] = dtensor_placement
        else:
            result[axis_name] = dtensor_placement
    return result


def annotate_input_spmd_types(
    parallel_dims: "ParallelDims",
    inputs: torch.Tensor,
    labels: torch.Tensor,
    extra_kwargs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Annotate decoder inputs/labels with SPMD types.

    Hardcodes the standard decoder convention: inputs and positions are
    ``S(0)@DP, S(1)@CP, R@TP``; labels are ``S(0)@DP, S(1)@CP, I@TP``.
    Analogous to ``full_dtensor.parallelize_inputs()`` but for the
    ``spmd_types`` path.
    """
    from torchtitan.distributed.parallel_dims import MeshAxisName

    token_type = {
        MeshAxisName.DP: spmd.S(0),
        MeshAxisName.CP: spmd.S(1),
        MeshAxisName.TP: spmd.R,
    }
    label_type = {
        MeshAxisName.DP: spmd.S(0),
        MeshAxisName.CP: spmd.S(1),
        MeshAxisName.TP: spmd.I,
    }

    mesh = parallel_dims.spmd_dense_mesh()
    with set_current_spmd_mesh(mesh):
        spmd.assert_type(inputs, token_type)
        spmd.assert_type(labels, label_type)
        if "positions" in extra_kwargs and isinstance(
            extra_kwargs["positions"], torch.Tensor
        ):
            spmd.assert_type(extra_kwargs["positions"], token_type)
    return inputs, labels, extra_kwargs


def spmd_distribute_tensor(
    tensor: torch.Tensor,
    mesh: DeviceMesh,
    layout: SpmdLayout,
) -> torch.Tensor:
    """Materialize local state shards according to the declared SPMD layout.

    Direct ``S(dim)`` layouts are applied per axis. For ``V + PartitionSpec``
    layouts, raw PartitionSpec tuple order controls repeated sharding of the
    same tensor dim, e.g. ``(DP, CP)`` means shard by DP, then shard each DP
    slice by CP.
    """
    shard_types = layout.per_axis_spmd_types()
    if layout.partition_spec is None:
        axis_shard_dims = [
            (axis_name, axis_type.dim)
            for axis_name, axis_type in shard_types.items()
            if isinstance(axis_type, spmd.Shard)
        ]
    else:
        # When multiple mesh axes shard the same tensor dim, the raw
        # PartitionSpec tuple defines the slicing order. For example,
        # PartitionSpec((DP, CP), None) shards by DP first, then CP.
        axis_shard_dims = []
        for dim, entry in enumerate(layout.partition_spec):
            if entry is None:
                continue
            axes = entry if isinstance(entry, tuple) else (entry,)
            for axis_name in axes:
                axis_shard_dims.append((axis_name, dim))

    assert mesh.mesh_dim_names is not None, "DeviceMesh must have named axes"
    for axis_name, dim in axis_shard_dims:
        axis = axis_name.value
        axis_size = (
            mesh.size(mesh.mesh_dim_names.index(axis))
            if axis in mesh.mesh_dim_names
            else 1
        )
        if axis_size > 1:
            tensor = spmd.shard(
                tensor,
                mesh.get_group(axis),
                src=spmd.I,
                dst=spmd.S(dim),
            )
    return tensor
