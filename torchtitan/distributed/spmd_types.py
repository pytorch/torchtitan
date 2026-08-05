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
from torch.distributed.tensor import Placement

from torchtitan.distributed.parallel_dims import MeshAxisName, spmd_layout_per_axis_types
from torchtitan.distributed.utils import get_spmd_backend

# Avoid circular import: protocols.__init__ imports module.py, which imports us.
if TYPE_CHECKING:
    from torchtitan.distributed.parallel_dims import (
        ParallelDims,
        SpmdLayout,
    )
    from torchtitan.protocols.sharding import PerAxisRedistribution

__all__ = [
    "annotate_input_spmd_types",
    "current_spmd_mesh",
    "spmd_dense_mesh",
    "spmd_sparse_mesh",
    "spmd_mesh_size",
    "spmd_distribute_tensor",
    "validate_sharding_config",
    "set_current_spmd_mesh",
    "set_spmd_meshes",
    "maybe_set_sparse_mesh",
    "resolve_dtensor_dst_placements",
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
    from torchtitan.distributed.parallel_dims import (
        MeshAxisName,
        spmd_layout_per_axis_types,
    )

    result: dict[MeshAxisName, Placement] = {}
    for axis_name, axis_type in spmd_layout_per_axis_types(layout).items():
        dtensor_placement = spmd.spmd_type_to_dtensor_placement(axis_type)

        if axis_name == MeshAxisName.DP:
            result[MeshAxisName.DP_REPLICATE] = dtensor_placement
            result[MeshAxisName.DP_SHARD] = dtensor_placement
        else:
            result[axis_name] = dtensor_placement
    return result


def resolve_dtensor_dst_placements(
    layout: "SpmdLayout",
    redist: "PerAxisRedistribution.Config",
    mesh: DeviceMesh,
) -> tuple[Placement, ...]:
    """Resolve DTensor placements after applying one redistribution config."""
    from torchtitan.protocols.sharding import resolve_placements

    src_axis_type = spmd_layout_per_axis_types(layout).get(redist.axis)
    if src_axis_type is not None and src_axis_type != redist.src:
        raise ValueError(
            "PerAxisRedistribution src does not match source sharding for "
            f"axis {redist.axis.value!r}: {redist.src!r} vs {src_axis_type!r}."
        )

    placements = list(resolve_placements(layout, mesh))
    mesh_axis_names = mesh.mesh_dim_names or ()
    axis = redist.axis.value
    if axis not in mesh_axis_names:
        return tuple(placements)

    axis_idx = mesh_axis_names.index(axis)
    if mesh.size(axis_idx) > 1:
        placements[axis_idx] = spmd.spmd_type_to_dtensor_placement(redist.dst)
    return tuple(placements)


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


def validate_sharding_config(
    sharding_config: Any, *, module_name: str | None = None
) -> None:
    # Validate sharding config SPMD layouts, and explicit redistributions before runtime usage.
    def validate_layout(layout: "SpmdLayout", *, name: str) -> None:
        """
        Validates a single SpmdLayout.
        - SpmdLayout axes are subset of either dense, or sparse mesh.
        - PartitionSpec axes must be spmd.V.
        - Per-axis types don't contain spmd.S(dim) if PartitionSpec is set,
          and don't repeatedly shard the same tensor dim.
        """
        dense_axes = {MeshAxisName.DP, MeshAxisName.CP, MeshAxisName.TP}
        sparse_axes = {
            MeshAxisName.DP_REPLICATE,
            MeshAxisName.EFSDP,
            MeshAxisName.EP,
        }

        location = f"{module_name}: {name}" if module_name is not None else name
        prefix = f"{location}: "
        layout_axes = set(layout.local_type)

        # PartitionSpec axes must be explicit V entries in local_type.
        if layout.partition_spec is not None:
            for dim, spec_entry in enumerate(layout.partition_spec):
                if spec_entry is None:
                    continue
                dim_axes = (
                    spec_entry if isinstance(spec_entry, tuple) else (spec_entry,)
                )
                for axis in dim_axes:
                    layout_axes.add(axis)
                    axis_type = layout.local_type.get(axis)
                    if axis_type is not spmd.V:
                        raise ValueError(
                            f"{prefix}PartitionSpec axis {axis.value!r} must "
                            f"have local type V, got {axis_type!r} at tensor "
                            f"dim {dim}."
                        )

        # Placement axes must be subset of either dense or sparse mesh.
        if layout_axes and not (layout_axes <= dense_axes or layout_axes <= sparse_axes):
            raise ValueError(
                f"{prefix}SPMD layout axes must be a subset of either dense "
                "mesh axes ('dp', 'cp', 'tp') or sparse mesh axes "
                "('dp_replicate', 'efsdp', 'ep'), got "
                f"{tuple(axis.value for axis in sorted(layout_axes))}."
            )

        # Local type shouldn't contain spmd.S(dim) if PartitionSpec is set,
        # and repeated sharding should be expressed in PartitionSpec.
        shard_dims: set[int] = set()
        for axis, axis_type in layout.local_type.items():
            if not isinstance(axis_type, spmd.Shard):
                continue
            if layout.partition_spec is not None:
                raise ValueError(
                    f"{prefix}axis {axis.value!r} has local type {axis_type!r}; "
                    "use either V + PartitionSpec, or per-axis sharding without "
                    "repeated shard dims."
                )
            if axis_type.dim in shard_dims:
                raise ValueError(
                    f"{prefix}multiple axes directly shard tensor dim "
                    f"{axis_type.dim}; shard order must be expressed in "
                    f"PartitionSpec, with V in local_type: {layout.local_type}."
                )
            shard_dims.add(axis_type.dim)

    def validate_redistribution(
        src: "SpmdLayout",
        redist: "PerAxisRedistribution.Config",
        *,
        name: str | None = None,
        is_input: bool = False,
    ) -> None:
        """
        Validate one PerAxisRedistribution.
        - redist.src matches the source layout for redist.axis.
        - removing S(dim) from a PartitionSpec removes the innermost axis.
        - adding S(dim) appends it as the innermost axis.
        """
        name = f"input {name!r}" if is_input else "output"
        location = (
            f"{module_name}: {name}" if module_name is not None else name
        )

        # src type must match between redistribution & placement.
        src_types = spmd_layout_per_axis_types(src)
        src_axis_type = src_types.get(redist.axis)
        if src_axis_type is None:
            raise ValueError(
                f"{location}: PerAxisRedistribution axis {redist.axis.value!r} "
                "is not declared in source sharding."
            )
        if src_axis_type != redist.src:
            raise ValueError(
                f"{location}: PerAxisRedistribution src does not match source "
                f"sharding for axis {redist.axis.value!r}: "
                f"{redist.src!r} vs {src_axis_type!r}."
            )

        # redistribution is unambiguous if shard ordering not involved
        if src.partition_spec is None or not isinstance(redist.src, spmd.Shard):
            return

        # Find the mesh-axis order for the tensor dim being unsharded.
        shard_dim = redist.src.dim
        assert (spec_entry := src.partition_spec[shard_dim]) is not None
        if isinstance(spec_entry, tuple):
            dim_axes = spec_entry
        else:
            dim_axes = (spec_entry,)
        assert redist.axis in dim_axes

        # Only the innermost mesh axis can be unsharded.
        if dim_axes[-1] != redist.axis:
            raise ValueError(
                f"{location}: unsharding axis {redist.axis.value!r} from tensor "
                f"dim {shard_dim} is only valid for the innermost mesh axis; "
                f"source shard order is {tuple(axis.value for axis in dim_axes)}; "
                f"PartitionSpec is {src.partition_spec}."
            )

    out_src = sharding_config.out_src_shardings
    layouts = {
        **sharding_config.state_shardings,
        **(sharding_config.in_src_shardings or {}),
        **({"output": out_src} if out_src is not None else {}),
    }
    for name, layout in layouts.items():
        validate_layout(layout, name=name)

    in_src = sharding_config.in_src_shardings or {}
    in_redist = sharding_config.in_redist or {}
    for name in in_src.keys() & in_redist.keys():
        validate_redistribution(
            in_src[name],
            in_redist[name],
            name=name,
            is_input=True,
        )

    out_redist = sharding_config.out_redist
    if out_src is not None and out_redist is not None:
        validate_redistribution(out_src, out_redist)


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
    from torchtitan.distributed.parallel_dims import spmd_layout_per_axis_types

    shard_types = spmd_layout_per_axis_types(layout)
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
