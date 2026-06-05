# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SPMD type helpers for the local-tensor parallelization path.

Bridges torchtitan's ``NamedPlacement`` sharding annotations with
``spmd_types`` runtime APIs (``assert_type``, ``redistribute``,
``PartitionSpec``).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from threading import local
from typing import Any

import spmd_types as spmd
import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Partial, Placement, Replicate, Shard

# Avoid circular import — MeshAxisName and NamedPlacement live in protocols.*,
# whose __init__ imports module.py which imports us.
# Use TYPE_CHECKING for annotations; import at use-site for runtime.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchtitan.protocols.types import NamedPlacement

__all__ = [
    "current_mesh",
    "mesh_size",
    "annotate_input_spmd_types",
    "placement_to_spmd_assert_type",
    "preserve_buffers_spmd",
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
    if _spmd_backend != "spmd_types":
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
def preserve_buffers_spmd(model: nn.Module) -> Iterator[None]:
    """
    Preserve local SPMD annotations on buffers across reinitialization.

    Previously we could have read types off self.freqs_cis, but after
    https://github.com/pytorch/torchtitan/pull/3458 changed to per-layer cache,
    the types disappear across to_empty() + _init_self_buffers().
    Implement it as a general context for all buffers in case other modules suffer from this.
    """
    if _spmd_backend != "spmd_types":
        yield
        return

    saved = {
        fqn: dict(spmd.get_local_type(buf))
        for fqn, buf in model.named_buffers()
        if spmd.has_local_type(buf)
    }
    try:
        yield
    finally:
        for fqn, buf in model.named_buffers():
            if fqn in saved and not spmd.has_local_type(buf):
                spmd.assert_type(buf, saved[fqn])


@contextlib.contextmanager
def set_current_spmd_mesh(mesh: DeviceMesh | None) -> Iterator[None]:
    """Set TorchTitan and spmd_types current mesh state for one runtime region."""
    if _spmd_backend != "spmd_types":
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


def spmd_to_dtensor_placement(placement: NamedPlacement) -> NamedPlacement:
    """Convert a placement with SPMD axis types to DTensor placements.

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
    from torchtitan.protocols.types import MeshAxisName, NamedPlacement

    result = {}
    for axis_name, axis_type in placement.placements.items():
        if isinstance(axis_type, Placement):
            dtensor_placement = axis_type
        elif axis_type == spmd.I or axis_type == spmd.R:
            dtensor_placement: Placement = Replicate()
        elif axis_type == spmd.P:
            dtensor_placement = Partial()
        elif isinstance(axis_type, spmd.Shard):
            dtensor_placement = Shard(axis_type.dim)
        else:
            raise ValueError(
                f"Unsupported placement for axis {axis_name!r}: {axis_type!r}."
            )
        # unfold DP axis
        if axis_name == MeshAxisName.DP:
            result[MeshAxisName.DP_REPLICATE] = dtensor_placement
            result[MeshAxisName.DP_SHARD] = dtensor_placement
        else:
            result[axis_name] = dtensor_placement
    return NamedPlacement(result)


def placement_to_spmd_assert_type(
    placement: NamedPlacement,
) -> tuple[spmd.PerMeshAxisSpmdTypes, spmd.PartitionSpec | None]:
    """Resolve a placement to ``assert_type`` args for local SPMD typechecking.

    If a ``NamedPlacement`` carries an explicit ``PartitionSpec``, shard
    entries are omitted from the per-axis type dict. ``spmd.assert_type``
    infers ``V`` for axes referenced by the normalized ``PartitionSpec``.
    Singleton axes are intentionally left in the returned values:
    ``spmd_types`` resolves names through ``current_mesh_all_names()`` and
    drops size-1 axes during normalization.
    """
    if not placement.is_spmd():
        raise ValueError(
            "SPMD backend requires SPMD placements. "
            f"Got DTensor placement: {placement!r}."
        )
    local_type = dict(placement.placements)
    if placement.partition_spec is None:
        return local_type, None

    partition_spec = spmd.normalize_partition_spec(
        spmd.PartitionSpec(*placement.partition_spec)
    )
    return {
        axis_name: value
        for axis_name, value in local_type.items()
        if not isinstance(value, spmd.Shard)
    }, partition_spec


def annotate_input_spmd_types(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    extra_kwargs: dict[str, Any],
) -> None:
    """Annotate decoder inputs/labels with local SPMD types.

    Hardcodes the standard decoder convention: inputs and positions are
    ``S(0)@DP, S(1)@CP, R@TP``; labels are ``S(0)@DP, S(1)@CP, I@TP``.
    Analogous to ``full_dtensor.parallelize_inputs()`` but for the
    ``spmd_types`` path.
    """
    from torchtitan.protocols.types import MeshAxisName, NamedPlacement

    DP = MeshAxisName.DP
    CP = MeshAxisName.CP
    TP = MeshAxisName.TP

    token_placement = NamedPlacement(
        {
            DP: spmd.S(0),
            CP: spmd.S(1),
            TP: spmd.R,
        }
    )
    label_placement = NamedPlacement(
        {
            DP: spmd.S(0),
            CP: spmd.S(1),
            TP: spmd.I,
        }
    )

    def assert_type(tensor: torch.Tensor, placement: NamedPlacement) -> None:
        local_type, partition_spec = placement_to_spmd_assert_type(placement)
        spmd.assert_type(tensor, local_type, partition_spec=partition_spec)

    assert_type(inputs, token_placement)
    assert_type(labels, label_placement)
    if "positions" in extra_kwargs and isinstance(extra_kwargs["positions"], torch.Tensor):
        assert_type(extra_kwargs["positions"], token_placement)

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

    for axis_types in (src_types, dst_types):
        for axis_name, axis_type in axis_types.items():
            if not isinstance(axis_type, spmd.PerMeshAxisSpmdType):
                raise ValueError(
                    "SPMD backend requires SPMD placements. "
                    f"Got {axis_type!r} for axis {axis_name!r}."
                )

    for axis_name, dst_t in dst_types.items():
        src_t = src_types.get(axis_name)
        if src_t is None or src_t == dst_t or mesh_size(axis_name) == 1:
            continue
        x = spmd.redistribute(
            x,
            mesh.get_group(axis_name),
            src=src_t,
            dst=dst_t,
            backward_options={"op_dtype": x.dtype},
        )
    return x
