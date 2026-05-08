# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Global SPMD state for the full_spmd_types path.

Populated once during ``ParallelDims.build_mesh()`` when
``full_spmd_types=True``.  Provides forward-time access to mesh axes
and process groups.

Usage::

    from torchtitan.distributed.spmd_state import mesh, spmd_state, is_spmd_active

    # Annotations — always valid, size-1 axes are no-ops
    spmd.assert_type(x, {mesh().tp: spmd.S(0)})

    # PGs for collectives
    state = spmd_state()
    pg = state.pgs["tp"]
"""

from dataclasses import dataclass, field
from typing import Any

from spmd_types import MeshAxis
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh


class MeshAxes:
    """Named collection of MeshAxis objects for active parallelism dimensions.

    Only stores axes with size > 1. ``getattr(mesh(), name, None)``
    returns ``None`` for disabled axes.
    """

    def __init__(self, **axes: MeshAxis) -> None:
        self._axes: dict[str, MeshAxis] = dict(axes)

    def __getattr__(self, name: str) -> MeshAxis | None:
        axes = self.__dict__.get("_axes")
        if axes is not None:
            return axes.get(name)
        return None


@dataclass
class SpmdState:
    """Torchtitan-specific derived SPMD state."""

    dp_axes: list[MeshAxis] = field(default_factory=list)
    all_axes: frozenset[MeshAxis] = field(default_factory=frozenset)
    pgs: dict[str, ProcessGroup] = field(default_factory=dict)

    def pg_for_axis(self, ax: MeshAxis) -> ProcessGroup | None:
        """Reverse lookup: MeshAxis → ProcessGroup."""
        if _MESH is None:
            return None
        for name, a in _MESH._axes.items():
            if a is ax:
                return self.pgs.get(name)
        return None


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_MESH: MeshAxes | None = None
_STATE: SpmdState | None = None


class _UninitializedMesh:
    """Sentinel for pre-init attribute access — raises a clear error."""

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(
            f"SPMD mesh not initialized. Cannot access mesh().{name}. "
            "Call init_spmd_state() during build_mesh()."
        )


_mesh_proxy: MeshAxes | Any = _UninitializedMesh()


def mesh() -> MeshAxes:
    """Return the global mesh axes. Always-valid axes after init."""
    return _mesh_proxy


def spmd_state() -> SpmdState:
    """Return the global SPMD state. Must be initialized first."""
    assert _STATE is not None, (
        "SPMD state not initialized. "
        "Call init_spmd_state() during build_mesh() with full_spmd_types=True."
    )
    return _STATE


def is_spmd_active() -> bool:
    """Check whether the SPMD path is active."""
    return _STATE is not None


_DP_AXIS_NAMES = ("dp_replicate", "dp_shard", "cp")


def init_spmd_state(dense_mesh: DeviceMesh) -> None:
    """Build MeshAxes and SpmdState from the dense mesh.

    Called by ``ParallelDims.build_mesh()``. Iterates dense_mesh axis names
    (skipping ``"pp"``), builds ``MeshAxis.of(pg)`` for size>1 axes and
    ``MeshAxis.of(1, 1)`` for size-1, derives ``dp_axes``, collects PGs.
    """
    global _MESH, _STATE, _mesh_proxy

    assert dense_mesh.mesh_dim_names is not None

    axes: dict[str, MeshAxis] = {}
    pgs: dict[str, ProcessGroup] = {}

    for i, name in enumerate(dense_mesh.mesh_dim_names):
        if name == "pp":
            continue
        size = dense_mesh.size(i)
        if size > 1:
            pg = dense_mesh.get_group(name)
            pg._set_group_desc(name)
            axes[name] = MeshAxis.of(pg)
            pgs[name] = pg

    dp_axes = [axes[n] for n in _DP_AXIS_NAMES if n in axes]
    all_axes = frozenset(axes.values())

    mesh_axes = MeshAxes(**axes)
    _MESH = mesh_axes
    _mesh_proxy = mesh_axes
    _STATE = SpmdState(dp_axes=dp_axes, all_axes=all_axes, pgs=pgs)
