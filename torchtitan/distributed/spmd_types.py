# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SPMD type helpers for the local-tensor parallelization path.

Bridges torchtitan's ``PlacementLike`` sharding annotations with
``spmd_types`` runtime APIs (``assert_type``, ``redistribute``,
``PartitionSpec``).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import local
from typing import Any

import spmd_types as spmd
import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Partial, Placement, Replicate, Shard
from torch.utils._pytree import tree_map

# Avoid circular import — MeshAxisName, PlacementLike, and NamedPlacement
# live in protocols.*, whose __init__ imports module.py which imports us.
# Use TYPE_CHECKING for annotations; import at use-site for runtime.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchtitan.protocols.sharding import PlacementLike
    from torchtitan.protocols.types import MeshAxisName, NamedPlacement

__all__ = [
    "PlacementSpec",
    "current_mesh",
    "mesh_size",
    "annotate_input_spmd_types",
    "active_placement",
    "is_placement_like",
    "placement_axes",
    "placement_to_spmd_assert_type",
    "preserve_buffer_spmd",
    "redistribute_spmd_per_axis",
    "set_current_spmd_mesh",
    "set_spmd_backend",
]


# ---------------------------------------------------------------------------
# Mesh context TLS — manages the current SPMD mesh for the local-tensor path
# ---------------------------------------------------------------------------

_MESH_TLS = local()
_spmd_backend = "default"


def set_spmd_backend(spmd_backend: str) -> None:
    """Set the backend that controls whether current-mesh context is active."""
    global _spmd_backend
    _spmd_backend = spmd_backend


def _mesh_stack() -> list[DeviceMesh | None]:
    stack = getattr(_MESH_TLS, "mesh_stack", None)
    if stack is None:
        stack = []
        _MESH_TLS.mesh_stack = stack
    return stack


def current_mesh() -> DeviceMesh | None:
    """Return the current runtime mesh, or ``None`` if unset."""
    if _spmd_backend != "spmd":
        return None
    stack = _mesh_stack()
    if not stack:
        return None
    return stack[-1]


def mesh_size(axis_name: str) -> int:
    """Return the size of a mesh axis, or 1 if not active."""
    mesh = current_mesh()
    if mesh is None:
        return 1
    names = mesh.mesh_dim_names or ()
    if axis_name not in names:
        return 1
    return mesh.size(names.index(axis_name))


@contextlib.contextmanager
def set_current_spmd_mesh(mesh: DeviceMesh | None) -> Iterator[None]:
    """Set TorchTitan and spmd_types current mesh state for one runtime region."""
    if _spmd_backend != "spmd":
        yield
        return

    stack = _mesh_stack()
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


@dataclass(frozen=True, kw_only=True, slots=True)
class PlacementSpec:
    """Placement with explicit global partition spec ordering.

    Use this only when multiple mesh axes shard the same tensor dimension.
    ``placement`` carries per-axis runtime placement. ``partition_spec`` only
    disambiguates ordering for type assertions in the local-tensor SPMD path.
    """

    placement: spmd.PerMeshAxisSpmdTypes
    partition_spec: spmd.PartitionSpec


def _is_spmd_placement(value: object) -> bool:
    """Return ``True`` if *value* is an SPMD placement (not a DTensor one).

    Checks structure: must be a ``PlacementSpec``, or a non-empty dict with
    ``MeshAxisName`` keys and ``PerMeshAxisSpmdType`` values.
    """
    if isinstance(value, PlacementSpec):
        return True
    from torchtitan.protocols.types import MeshAxisName
    return (
        isinstance(value, dict)
        and bool(value)
        and all(
            isinstance(k, MeshAxisName) and isinstance(v, spmd.PerMeshAxisSpmdType)
            for k, v in value.items()
        )
    )


def is_placement_like(value: object) -> bool:
    """Return ``True`` if *value* is any kind of placement (SPMD or DTensor).

    A placement is a ``PlacementSpec``, or a non-empty dict with
    ``MeshAxisName`` keys and either ``PerMeshAxisSpmdType`` or DTensor
    ``Placement`` values.
    """
    if _is_spmd_placement(value):
        return True
    from torchtitan.protocols.types import MeshAxisName
    return (
        isinstance(value, dict)
        and bool(value)
        and all(isinstance(k, MeshAxisName) for k in value)
    )


def placement_axes(placement: PlacementLike) -> tuple[MeshAxisName, ...]:
    """Return the mesh axis names declared in *placement*."""
    named = placement.placement if isinstance(placement, PlacementSpec) else placement
    return tuple(named.keys())


def spmd_to_dtensor_placement(placement: PlacementLike) -> NamedPlacement | None:
    """Convert an SPMD placement to a DTensor ``NamedPlacement``.

    Returns ``None`` if *placement* is not an SPMD placement (i.e. it is
    already a DTensor ``NamedPlacement``).

    Conversion rules:
    - ``I`` / ``R`` → ``Replicate()`` (DTensor has no Invariant concept)
    - ``S(dim)`` → ``Shard(dim)``
    - ``P`` → ``Partial()``
    - ``DP`` axis → expanded to both ``DP_REPLICATE`` and ``DP_SHARD``
      (the SPMD path uses one logical DP axis, while the full-DTensor
      storage mesh still spells it as two)

    TODO(pianpwk): Remove after the full_dtensor path is deleted. Only
    exists to translate SPMD placements into DTensor placements so that
    ``resolve_placements`` can serve both backends while sharding configs
    are written in spmd_types.
    """
    from torchtitan.protocols.sharding import NamedPlacement as _NP
    from torchtitan.protocols.types import MeshAxisName

    if not _is_spmd_placement(placement):
        return None
    named = placement.placement if isinstance(placement, PlacementSpec) else placement

    result: _NP = {}
    for axis_name, axis_type in named.items():
        if axis_type == spmd.I or axis_type == spmd.R:
            dtensor_placement: Placement = Replicate()
        elif axis_type == spmd.P:
            dtensor_placement = Partial()
        elif isinstance(axis_type, spmd.Shard):
            dtensor_placement = Shard(axis_type.dim)
        else:
            raise ValueError(
                f"Unsupported SPMD placement for axis {axis_name!r}: {axis_type!r}."
            )
        if axis_name == MeshAxisName.DP:
            result[MeshAxisName.DP_REPLICATE] = dtensor_placement
            result[MeshAxisName.DP_SHARD] = dtensor_placement
        else:
            result[axis_name] = dtensor_placement
    return result


def active_placement(
    placement: PlacementLike,
) -> spmd.PerMeshAxisSpmdTypes:
    """Return the SPMD placement restricted to active current-mesh axes.

    Sharding configs declare placements for all possible axes (DP, CP, TP,
    etc.), but the runtime mesh may only have a subset active (e.g. TP-only
    when DP/CP are size 1). This filters the placement dict to only the
    axes present in ``spmd.current_mesh_names()``, so ``spmd.assert_type``
    and ``spmd.redistribute`` see only the axes they can resolve.

    Raises ``ValueError`` if *placement* is a DTensor placement (not SPMD).
    Returns an empty dict if no current mesh is set.
    """
    named = placement.placement if isinstance(placement, PlacementSpec) else placement
    if not _is_spmd_placement(placement):
        raise ValueError(
            "SPMD backend requires SPMD placements. "
            f"Got DTensor placement: {named!r}."
        )
    mesh_names = spmd.current_mesh_names()
    if mesh_names is None:
        return {}

    resolved: spmd.PerMeshAxisSpmdTypes = {}
    for axis_name, value in named.items():
        if axis_name in mesh_names:
            resolved[axis_name] = value
    return resolved


def placement_to_spmd_assert_type(
    placement: PlacementLike,
) -> tuple[spmd.PerMeshAxisSpmdTypes, spmd.PartitionSpec | None]:
    """Resolve a placement to ``assert_type`` args for local SPMD typechecking.

    Two normalizations are needed because sharding configs use ``S(dim)`` for
    readability, but ``spmd.assert_type`` expects ``V`` (varying) with shard
    ordering carried separately in a ``PartitionSpec``:

    1. ``S(dim)`` entries in the per-axis type dict are converted to ``V``.
    2. When a ``PlacementSpec`` carries a ``PartitionSpec`` with nested axis
       tuples (e.g. ``(DP, (CP, TP), None)``), inactive axes are filtered
       out based on ``current_mesh_names()``.

    TODO(pianpwk) before landing: Move this logic into ``spmd_types`` so it accepts
    ``S(*)`` directly and does the ``S`` → ``V`` + ``PartitionSpec``
    normalization internally.
    """
    local_type = active_placement(placement)
    if not isinstance(placement, PlacementSpec):
        return local_type, None

    mesh_names = spmd.current_mesh_names()
    if mesh_names is None:
        return local_type, None

    def active_partition_entry(entry):
        filtered = tree_map(
            lambda axis_name: axis_name if axis_name in mesh_names else None,
            entry,
        )
        if filtered is None:
            return None
        if not isinstance(filtered, tuple):
            return filtered

        active_axes = tuple(
            axis_name for axis_name in filtered if axis_name is not None
        )
        if not active_axes:
            return None
        return active_axes[0] if len(active_axes) == 1 else active_axes

    partition_spec = spmd.PartitionSpec(
        *(active_partition_entry(entry) for entry in placement.partition_spec)
    )
    return {
        axis_name: spmd.V if isinstance(value, spmd.Shard) else value
        for axis_name, value in local_type.items()
    }, partition_spec


def annotate_input_spmd_types(
    parallel_dims: Any,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    extra_inputs: dict[str, torch.Tensor],
    extra_kwargs: dict[str, Any],
) -> None:
    """Annotate decoder inputs/labels with local SPMD types.

    Hardcodes the standard decoder convention: inputs and positions are
    ``S(0)@DP, S(1)@CP, R@TP``; labels are ``S(0)@DP, S(1)@CP, I@TP``.
    Analogous to ``full_dtensor.parallelize_inputs()`` but for the
    local-tensor SPMD path.
    """
    from torchtitan.protocols.types import MeshAxisName

    DP = MeshAxisName.DP
    CP = MeshAxisName.CP
    TP = MeshAxisName.TP

    token_placement: spmd.PerMeshAxisSpmdTypes = {
        DP: spmd.S(0),
        CP: spmd.S(1),
        TP: spmd.R,
    }
    label_placement: spmd.PerMeshAxisSpmdTypes = {
        DP: spmd.S(0),
        CP: spmd.S(1),
        TP: spmd.I,
    }

    def assert_type(tensor: torch.Tensor, placement: spmd.PerMeshAxisSpmdTypes) -> None:
        local_type, partition_spec = placement_to_spmd_assert_type(placement)
        if local_type:
            spmd.assert_type(tensor, local_type, partition_spec=partition_spec)

    assert_type(inputs, token_placement)
    assert_type(labels, label_placement)
    if "positions" in extra_kwargs and isinstance(extra_kwargs["positions"], torch.Tensor):
        assert_type(extra_kwargs["positions"], token_placement)


@contextmanager
def preserve_buffer_spmd(model: nn.Module) -> Iterator[None]:
    """Context manager that saves and restores ``spmd_types`` annotations on buffers.

    ``Module.init_states()`` may replace buffer tensors (e.g. RoPE cache),
    which drops the ``spmd_types`` local-type metadata attached by
    ``parallelize()``.  Wrap the init call in this context manager so the
    annotations survive the replacement.
    """
    saved: dict[str, Any] = {}
    for fqn, buf in model.named_buffers():
        if spmd.has_local_type(buf):
            saved[fqn] = dict(spmd.get_local_type(buf))
    yield
    for fqn, buf in model.named_buffers():
        if fqn in saved and not spmd.has_local_type(buf):
            spmd.assert_type(buf, saved[fqn])


def redistribute_spmd_per_axis(
    x: torch.Tensor,
    src_types: spmd.PerMeshAxisSpmdTypes,
    dst_types: spmd.PerMeshAxisSpmdTypes,
) -> torch.Tensor:
    """Redistribute a local tensor along axes whose SPMD type changes.

    Iterates over *dst_types* and issues a per-axis ``spmd.redistribute``
    for each axis where src and dst differ. Each call is a single collective
    (all-reduce, reduce-scatter, or all-gather) on that axis's process group.

    TODO(pianpwk) before landing: Move into ``spmd_types`` as a version that takes
    per-axis types + ``PartitionSpec``, so the library handles multi-axis
    redistribute ordering internally.
    """
    mesh = current_mesh()
    if mesh is None:
        return x

    for axis_name, dst_t in dst_types.items():
        src_t = src_types.get(axis_name)
        if src_t is None or src_t == dst_t:
            continue
        x = spmd.redistribute(
            x,
            mesh.get_group(axis_name),
            src=src_t,
            dst=dst_t,
            backward_options={"op_dtype": x.dtype},
        )
    return x
