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
from typing import Any, Protocol, runtime_checkable

import spmd_types as spmd
import torch
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.distributed.parallelism_context import MeshAxisName, ParallelismContext


# TODO: Remove after spmd_types fixes deepcopy for its variadic tuple subclass.
# PartitionSpec is immutable, so sharing it across a model deepcopy is safe.
setattr(spmd.PartitionSpec, "__deepcopy__", lambda self, memo: self)  # noqa: B010

__all__ = [
    "SpmdAnnotatedInput",
    "disable_dense_sp",
    "annotate_input_spmd_types",
    "annotate_replicated_parameters",
    "current_spmd_mesh",
    "spmd_axes",
    "spmd_local_context",
    "maybe_set_sparse_mesh",
    "spmd_dense_mesh",
    "spmd_dense_sp_enabled",
    "spmd_mesh_group",
    "spmd_sparse_mesh",
    "spmd_mesh_size",
    "spmd_distribute_tensor",
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


def set_spmd_meshes(
    *,
    dense_mesh: DeviceMesh,
    sparse_mesh: DeviceMesh | None,
    dense_sp_enabled: bool,
) -> None:
    """Register the SPMD meshes for dense and sparse runtime regions."""
    _MESH_TLS.dense_mesh = dense_mesh
    _MESH_TLS.sparse_mesh = sparse_mesh
    _MESH_TLS.dense_sp_enabled = dense_sp_enabled


def spmd_dense_mesh() -> DeviceMesh:
    """Return the registered dense SPMD mesh."""
    mesh = getattr(_MESH_TLS, "dense_mesh", None)
    assert mesh is not None, "SPMD dense mesh has not been registered"
    return mesh


def spmd_dense_sp_enabled() -> bool:
    """Return whether sequence parallelism is enabled in the dense region."""
    return getattr(_MESH_TLS, "dense_sp_enabled", False)


@contextlib.contextmanager
def disable_dense_sp() -> Iterator[None]:
    """Treat the dense region as having no sequence parallelism.

    For regions whose activations are never sequence-sharded even when the
    decoder uses SP, e.g. vision encoders, whose activations stay
    TP-``Invariant``. Inside, ``spmd_dense_sp_enabled()`` is False, so helpers
    that follow the decoder residual layout (``maybe_gather_tp_input``,
    ``RowParallelLinear``) use the non-SP ``Invariant`` layout.
    """
    previous = spmd_dense_sp_enabled()
    _MESH_TLS.dense_sp_enabled = False
    try:
        yield
    finally:
        _MESH_TLS.dense_sp_enabled = previous


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


def spmd_mesh_group(axis_name: str) -> torch.distributed.ProcessGroup | None:
    """Return a non-singleton process group from the current SPMD mesh."""
    mesh = current_spmd_mesh()
    if mesh is None:
        return None
    names = mesh.mesh_dim_names or ()
    if axis_name not in names:
        return None
    group = mesh.get_group(axis_name)
    return group if group.size() > 1 else None


def spmd_local_context(
    *local_axes: str,
) -> contextlib.AbstractContextManager[None]:
    """Context manager treating the named mesh axes as local axes.

    Local axes retain per-coordinate SPMD semantics during global type
    checking: each coordinate selects an independent tensor, and only the
    remaining axes describe that tensor's global sharding. This is a no-op for
    axes with size 1.
    """
    active_axes = tuple(
        dict.fromkeys(axis for axis in local_axes if spmd_mesh_size(axis) > 1)
    )
    if not active_axes:
        return contextlib.nullcontext()
    return spmd.set_current_mesh(local_axes=active_axes)


@contextlib.contextmanager
def set_current_spmd_mesh(mesh: DeviceMesh | None) -> Iterator[None]:
    """Set TorchTitan and spmd_types current mesh state for one runtime region."""
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
    """Activate the registered sparse mesh, if present."""
    if (mesh := spmd_sparse_mesh()) is None:
        yield
        return

    with set_current_spmd_mesh(mesh):
        yield


@runtime_checkable
class SpmdAnnotatedInput(Protocol):
    """A non-tensor forward input, e.g. attention metadata, whose nested tensors
    ``annotate_input_spmd_types`` can't see; it annotates them itself."""

    def annotate_spmd_types(self) -> None:
        ...


def annotate_input_spmd_types(
    parallelism_context: "ParallelismContext",
    input_dict: dict[str, Any],
    input_sharding: dict[str, spmd.SpmdType],
) -> dict[str, Any]:
    """Annotate named forward inputs with SPMD types from ``input_sharding``.

    ``input_dict`` maps each name ('input', 'labels', and extra forward kwargs)
    to its value. Each named tensor is asserted against its own layout, and
    must have a layout entry: a tensor with no entry raises rather than being
    silently left untyped. A non-tensor kwarg, or a value of a mapping kwarg
    (e.g. ``attention_metadata``), that is a ``SpmdAnnotatedInput`` annotates
    its own nested tensors; other non-tensor kwargs (e.g. ``special_tokens``)
    are left untouched.
    """
    mesh = parallelism_context.spmd_dense_mesh()
    untyped: list[str] = []
    with set_current_spmd_mesh(mesh):
        for name, value in input_dict.items():
            if not isinstance(value, torch.Tensor):
                nested = value.values() if isinstance(value, Mapping) else (value,)
                for item in nested:
                    if isinstance(item, SpmdAnnotatedInput):
                        item.annotate_spmd_types()
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
            "or make their container a ``SpmdAnnotatedInput``."
        )
    return input_dict


def annotate_replicated_parameters(
    module: torch.nn.Module,
    parallelism_context: ParallelismContext,
) -> None:
    """Annotate undistributed model parameters as replicated.

    Call this before state-sharding modules with ``Module._parallelize``. That
    replaces declared parameters with their model-parallel shards, while these
    annotations remain on parameters without a ``ShardingConfig`` for FSDP.
    """
    with set_current_spmd_mesh(parallelism_context.spmd_dense_mesh()):
        for param in module.parameters():
            spmd.assert_type(param, spmd.R)


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


def invariant_to_replicate(x: torch.Tensor, axis: MeshAxisName) -> torch.Tensor:
    """``spmd.invariant_to_replicate`` over ``axis``, if ``axis`` is in the SPMD mesh.

    A forward no-op whose backward all-reduces the gradient over ``axis``.
    Returns ``x`` unchanged when ``axis`` is not in the SPMD mesh.
    """
    group = spmd_mesh_group(axis)
    if group is None:
        return x
    return spmd.invariant_to_replicate(x, group, backward_options={"op_dtype": x.dtype})


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
