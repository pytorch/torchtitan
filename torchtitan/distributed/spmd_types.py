# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Helpers for torchtitan's spmd_types backend."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from threading import local
from typing import Any

import spmd_types as spmd
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from torchtitan.distributed.parallel_dims import (
    MeshAxisName,
    ParallelDims,
    unfold_dp_axes,
)
from torchtitan.distributed.utils import get_spmd_backend


# TODO: Remove after spmd_types fixes deepcopy for its variadic tuple subclass.
# PartitionSpec is immutable, so sharing it across a model deepcopy is safe.
setattr(spmd.PartitionSpec, "__deepcopy__", lambda self, memo: self)  # noqa: B010

__all__ = [
    "annotate_input_spmd_types",
    "current_spmd_mesh",
    "dtensor_to_plain_tensor_state_dict",
    "spmd_axes",
    "maybe_set_sparse_mesh",
    "plain_tensor_to_dtensor_state_dict",
    "spmd_dense_mesh",
    "spmd_sparse_mesh",
    "spmd_mesh_size",
    "spmd_distribute_tensor",
    "spmd_redistribute_per_axis",
    "spmd_validate_redistributions",
    "set_current_spmd_mesh",
    "set_spmd_meshes",
]


_MESH_TLS = local()


def spmd_axes(layout: spmd.SpmdType) -> tuple[MeshAxisName, ...]:
    """Return and validate the named mesh axes used by an SPMD layout."""
    axes = []
    for axis in layout.local_type:
        if not isinstance(axis, str):
            raise TypeError(
                f"TorchTitan SPMD layouts require named mesh axes, got {axis!r}"
            )
        axes.append(MeshAxisName(axis))
    return tuple(axes)


def plain_tensor_to_dtensor_state_dict(
    state_dict: dict[str, Any],
    *,
    state_dict_layouts: Mapping[str, spmd.SpmdType],
    parallel_dims: ParallelDims,
) -> dict[str, Any]:
    """Represent plain local state tensors as DTensors for state transfer."""
    from torchtitan.protocols.sharding import resolve_placements

    dtensor_state_dict = dict(state_dict)
    with torch.no_grad():
        for name, target in state_dict.items():
            if not isinstance(target, torch.Tensor) or isinstance(target, DTensor):
                continue

            layout = state_dict_layouts.get(name)
            if layout is None:
                raise KeyError(f"{name} is missing SPMD layout metadata")

            mesh = parallel_dims.get_activated_mesh(unfold_dp_axes(spmd_axes(layout)))
            if mesh is None:
                continue

            dtensor_state_dict[name] = DTensor.from_local(
                target,
                mesh,
                resolve_placements(layout, mesh),
                run_check=False,
            )
    return dtensor_state_dict


def dtensor_to_plain_tensor_state_dict(
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Replace DTensor state-dict entries with their plain local tensors."""
    return {
        name: value.to_local() if isinstance(value, DTensor) else value
        for name, value in state_dict.items()
    }


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


def annotate_input_spmd_types(
    parallel_dims: "ParallelDims",
    input_dict: dict[str, Any],
    input_sharding: dict[str, spmd.SpmdType],
) -> dict[str, Any]:
    """Annotate named forward inputs with SPMD types from ``input_sharding``.

    ``input_dict`` maps each name ('input', 'labels', and extra forward kwargs)
    to its value. Each named tensor is asserted against its own layout.
    Non-tensor kwargs (e.g. ``attention_masks`` containers, ``special_tokens``)
    are left untouched. Every *tensor* input, however, must have a layout entry:
    a tensor with no entry raises rather than being silently left untyped.
    Tensors nested inside container kwargs are not reachable here and must
    be annotated at their construction site.
    """
    mesh = parallel_dims.spmd_dense_mesh()
    untyped: list[str] = []
    with set_current_spmd_mesh(mesh):
        for name, value in input_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            layout = input_sharding.get(name)
            if layout is None:
                untyped.append(name)
                continue
            spmd.assert_type(value, layout)
    if untyped:
        raise ValueError(
            "spmd_types backend requires an SPMD layout for every tensor input, "
            f"but these have no entry in input_sharding: {sorted(untyped)}. Add "
            "them to the input layout the model declares in ``preprocess_inputs``, "
            "or annotate nested/container tensors at their construction site."
        )
    return input_dict


def _per_axis_types(
    layout: spmd.SpmdType,
) -> dict[MeshAxisName, spmd.PerMeshAxisSpmdType]:
    result: dict[MeshAxisName, spmd.PerMeshAxisSpmdType] = {}
    for axis, axis_type in layout.local_type.items():
        if not isinstance(axis, str):
            raise TypeError(
                f"TorchTitan SPMD layouts require named mesh axes, got {axis!r}"
            )
        result[MeshAxisName(axis)] = axis_type
    if layout.partition_spec is not None:
        for dim, entry in enumerate(layout.partition_spec):
            axes = (
                () if entry is None else entry if isinstance(entry, tuple) else (entry,)
            )
            for axis in axes:
                if not isinstance(axis, str):
                    raise TypeError(
                        "TorchTitan SPMD layouts require named mesh axes, "
                        f"got {axis!r}"
                    )
                result[MeshAxisName(axis)] = spmd.S(dim)
    return result


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
    once the partial_dtensor backend is removed.
    """

    def _normalize_partition_spec(
        axis_types: Mapping[MeshAxisName, spmd.PerMeshAxisSpmdType],
        *,
        ndim: int,
    ) -> tuple[tuple[MeshAxisName, ...], ...]:
        """Normalize per-axis-types w/ S(dim) -> PartitionSpec-style tuple."""
        entries: list[tuple[MeshAxisName, ...]] = [()] * ndim
        for axis_name, axis_type in axis_types.items():
            if not isinstance(axis_type, spmd.Shard):
                continue
            if not isinstance(axis_name, str):
                raise TypeError("ShardingConfig SpmdType axes must be names")
            dim = axis_type.dim if axis_type.dim >= 0 else ndim + axis_type.dim
            if dim < 0 or dim >= ndim:
                raise ValueError(
                    f"Cannot compare SPMD layout with shard dim {axis_type.dim} "
                    f"against PartitionSpec of rank {ndim}."
                )
            entries[dim] = (MeshAxisName(axis_name),)
        return tuple(entries)

    def _validate_redistribute_spmd_pair(
        src: spmd.SpmdType,
        dst: spmd.SpmdType,
        *,
        name: str,
    ) -> None:
        """Validate a SPMD redistribution is expressible with one-axis collective."""
        # 1) Check that only one axis mismatches.
        # Store the changed_axes so we know what to look for in PartitionSpec.
        src_types = _per_axis_types(src)
        dst_types = _per_axis_types(dst)
        if set(src_types) != set(dst_types):
            raise ValueError(
                "SpmdType-based redistribute axis keys do not match for "
                f"src: {src_types} -> dst: {dst_types}."
            )

        changed_axes = [
            axis_name
            for axis_name in src_types.keys() | dst_types.keys()
            if src_types.get(axis_name) != dst_types.get(axis_name)
        ]
        if len(changed_axes) > 1:
            raise ValueError(
                f"{name}: SpmdType-based redistribution changes multiple mesh "
                f"axes ({sorted(str(axis) for axis in changed_axes)}). "
                "spmd_redistribute_per_axis only supports one single-axis "
                "redistribution."
            )
        if changed_axes and (
            src_types[changed_axes[0]] is spmd.V or dst_types[changed_axes[0]] is spmd.V
        ):
            axis = changed_axes[0]
            raise ValueError(
                f"{name}: SpmdType-based redistribution changes mesh axis "
                f"{str(axis)!r} with spmd.V as the source or destination type. "
                "Config-based redistribution requires non-V types; write an "
                "explicit collective when the value semantics are unclear."
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
            src_spec = _normalize_partition_spec(src_types, ndim=ndim)
        if dst_spec is None:
            dst_spec = _normalize_partition_spec(dst_types, ndim=ndim)

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
                "SpmdType-based redistribution changes shard order for "
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
    src: spmd.SpmdType,
    dst: spmd.SpmdType,
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

    src_types = _per_axis_types(src)
    dst_types = _per_axis_types(dst)
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
    layout: spmd.SpmdType,
) -> torch.Tensor:
    """Materialize local state shards according to the declared SPMD layout.

    Direct ``S(dim)`` layouts are applied per axis. For ``V + PartitionSpec``
    layouts, raw PartitionSpec tuple order controls repeated sharding of the
    same tensor dim, e.g. ``(DP, CP)`` means shard by DP, then shard each DP
    slice by CP.
    """
    if layout.partition_spec is None:
        axis_shard_dims = [
            (axis_name, axis_type.dim)
            for axis_name, axis_type in layout.local_type.items()
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
        if not isinstance(axis_name, str):
            raise TypeError(
                f"TorchTitan SPMD layouts require named mesh axes, got {axis_name!r}"
            )
        axis = MeshAxisName(axis_name).value
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
