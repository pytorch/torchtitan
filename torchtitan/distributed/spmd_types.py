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
    "spmd_mesh_size",
    "spmd_distribute_tensor",
    "spmd_redistribute_per_axis",
    "spmd_validate_redistributions",
    "set_current_spmd_mesh",
    "spmd_layout_to_dtensor_placements",
]


_MESH_TLS = local()


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
    assert (
        get_spmd_backend() == "spmd_types"
    ), "set_current_spmd_mesh() is only valid under spmd_types backend"

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

    mesh = parallel_dims._global_meshes["spmd_dense_for_fwdbwd"]
    with set_current_spmd_mesh(mesh):
        spmd.assert_type(inputs, token_type)
        spmd.assert_type(labels, label_type)
        if "positions" in extra_kwargs and isinstance(
            extra_kwargs["positions"], torch.Tensor
        ):
            spmd.assert_type(extra_kwargs["positions"], token_type)
    return inputs, labels, extra_kwargs


def spmd_validate_redistributions(sharding_config: Any) -> None:
    """Validate that SPMD redistributions fit the current runtime helper.

    ``spmd_redistribute_per_axis`` can issue at most one single-axis
    collective for a src/dst layout pair. It does not implement multi-axis
    moves, and it cannot express unshard/reshard reorderings such as
    ``PartitionSpec((DP, CP)) -> PartitionSpec((CP, DP))`` where per-axis
    shard types are unchanged but global shard order changes.

    TODO(pianpwk): this is transitional code while ShardingConfig-based
    redistributions are written in src/dst DTensor-style placements.
    A more general DTensor-style redistribute API should live in spmd_types,
    or we should write collective-based (not placement-based) redistributions
    after full_dtensor backend is removed.
    """

    def _normalize_partition_spec(
        axis_types: dict["MeshAxisName", spmd.PerMeshAxisSpmdType],
        *,
        ndim: int,
    ) -> tuple[tuple["MeshAxisName", ...], ...]:
        """Normalize per-axis-types w/ S(dim) -> PartitionSpec-style tuple."""
        entries: list[tuple["MeshAxisName", ...]] = [()] * ndim
        for axis_name, axis_type in axis_types.items():
            if not isinstance(axis_type, spmd.Shard):
                continue
            dim = axis_type.dim if axis_type.dim >= 0 else ndim + axis_type.dim
            if dim < 0 or dim >= ndim:
                raise ValueError(
                    f"Cannot compare SPMD layout with shard dim {axis_type.dim} "
                    f"against PartitionSpec of rank {ndim}."
                )
            entries[dim] = (axis_name,)
        return tuple(entries)

    def _validate_redistribute_spmd_pair(
        src: "SpmdLayout",
        dst: "SpmdLayout",
        *,
        name: str,
    ) -> None:
        """Validate a SPMD redistribution is expressible with one-axis collective."""
        # 1) Checks based on per_axis_spmd_types(), that only one axis mismatches.
        # Store the changed_axes so we know what to look for in PartitionSpec.
        src_types = src.per_axis_spmd_types()
        dst_types = dst.per_axis_spmd_types()
        if set(src_types) != set(dst_types):
            raise ValueError(
                "SpmdLayout-based redistribute axis keys do not match for "
                f"src: {src_types} -> dst: {dst_types}."
            )

        changed_axes = [
            axis_name
            for axis_name in src_types.keys() | dst_types.keys()
            if src_types.get(axis_name) != dst_types.get(axis_name)
        ]
        if len(changed_axes) > 1:
            raise ValueError(
                f"{name}: SpmdLayout-based redistribution changes multiple mesh "
                f"axes ({sorted(axis.value for axis in changed_axes)}). "
                "spmd_redistribute_per_axis only supports one single-axis "
                "redistribution."
            )

        # 2) If neither has PartitionSpec, comparing per_axis_spmd_types() is sufficient.
        if src.partition_spec is None and dst.partition_spec is None:
            return

        # 3) If one side has no PartitionSpec, synthesize the simple
        # one-axis-per-dim form from its S(dim) local types.
        ndim = (
            len(src.partition_spec)  # pyrefly: ignore [bad-argument-type]
            if dst.partition_spec is None
            else len(dst.partition_spec)
        )
        src_spec, dst_spec = src.partition_spec, dst.partition_spec
        if src_spec is None:
            src_spec = _normalize_partition_spec(src.axis_types, ndim=ndim)
        if dst_spec is None:
            dst_spec = _normalize_partition_spec(dst.axis_types, ndim=ndim)

        # A one-axis redistribute may only leave each tensor dim's shard axes
        # unchanged, add the changed axis as the innermost shard, or remove it
        # from the innermost position. For example, (DP) -> (DP, CP) is valid
        # when CP is the changed axis, but (DP) -> (CP, DP) changes shard order.
        changed_axis = changed_axes[0] if changed_axes else None
        for dim, (src_entry, dst_entry) in enumerate(zip(src_spec, dst_spec)):
            src_axes = (
                ()
                if src_entry is None
                else src_entry
                if isinstance(src_entry, tuple)
                else (src_entry,)
            )
            dst_axes = (
                ()
                if dst_entry is None
                else dst_entry
                if isinstance(dst_entry, tuple)
                else (dst_entry,)
            )
            if src_axes == dst_axes:
                continue
            if changed_axis is not None and dst_axes == src_axes + (changed_axis,):
                continue
            if changed_axis is not None and src_axes == dst_axes + (changed_axis,):
                continue
            raise ValueError(
                "SpmdLayout-based redistribution changes shard order for "
                f"tensor {name} dim {dim}, which is currently unsupported "
                "by spmd_redistribute_per_axis. Please write this as an "
                "explicit collective instead."
            )

    in_src = sharding_config.in_src_shardings or {}
    in_dst = sharding_config.in_dst_shardings or {}
    for name in in_src.keys() & in_dst.keys():
        _validate_redistribute_spmd_pair(
            in_src[name],
            in_dst[name],
            name=f"input {name!r}",
        )

    out_src = sharding_config.out_src_shardings
    out_dst = sharding_config.out_dst_shardings
    if out_src is not None and out_dst is not None:
        _validate_redistribute_spmd_pair(out_src, out_dst, name="output")


def spmd_redistribute_per_axis(
    x: torch.Tensor,
    mesh: DeviceMesh | None,
    src_types: spmd.PerMeshAxisSpmdTypes,
    dst_types: spmd.PerMeshAxisSpmdTypes,
) -> torch.Tensor:
    """Redistribute a local tensor along axes whose SPMD type changes.

    Iterates over *dst_types* and issues a per-axis ``spmd.redistribute``
    for each axis where src and dst differ. Each call is a single collective
    (all-reduce, reduce-scatter, or all-gather) on that axis's process group.

    TODO(pianpwk): Move into ``spmd_types`` as a version that takes
    per-axis types + ``PartitionSpec``, so the library handles multi-axis
    redistribute ordering internally.
    """
    if mesh is None:
        return x

    assert mesh.mesh_dim_names is not None, "DeviceMesh must have named axes"
    for axis_name, dst_t in dst_types.items():
        src_t = src_types.get(axis_name)
        axis = axis_name.value
        axis_size = (
            mesh.size(mesh.mesh_dim_names.index(axis))
            if axis in mesh.mesh_dim_names
            else 1
        )
        if src_t == dst_t or axis_size == 1:
            continue
        x = spmd.redistribute(
            x,
            mesh.get_group(axis),
            src=src_t,
            dst=dst_t,
            backward_options={"op_dtype": x.dtype},
        )
    return x


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
