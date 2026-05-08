# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Global SPMD state for the full_spmd_types path.

Populated once during ``ParallelDims.build_mesh()`` when
``full_spmd_types=True``. Provides forward-time access to mesh axes
and process groups.

Axis access uses the ``mesh`` singleton (always-valid axes — size-1
axes are trivial MeshAxis, automatically dropped by spmd_types).
PG access and derived state (dp_axes) live on ``SpmdState``.

Usage::

    from torchtitan.distributed.spmd_state import mesh, spmd_state, is_spmd_active

    # Annotations — always valid, size-1 axes are no-ops
    spmd.assert_type(x, {mesh.TP: spmd.S(0)})

    # PGs for collectives
    state = spmd_state()
    pg = state.pgs["tp"]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch.distributed import ProcessGroup


class MeshAxes:
    """Named collection of MeshAxis objects for every parallelism dimension.

    All axes are always valid — size-1 axes are trivial ``MeshAxis.of(1, 1)``.
    The spmd_types library drops size-1 axes automatically, so annotations
    like ``{mesh.TP: spmd.V}`` are no-ops when TP degree is 1.
    """

    def __init__(self, **axes: Any) -> None:
        self._axes: dict[str, Any] = dict(axes)

    def __getattr__(self, name: str) -> Any:
        axes = self.__dict__.get("_axes")
        if axes is not None and name in axes:
            return axes[name]
        raise AttributeError(f"MeshAxes has no axis {name!r}")


@dataclass
class SpmdState:
    """Torchtitan-specific derived SPMD state."""

    dp_axes: list[Any] = field(default_factory=list)
    all_axes: frozenset = field(default_factory=frozenset)
    pgs: dict[str, ProcessGroup] = field(default_factory=dict)

    def pg_for_axis(self, ax) -> ProcessGroup | None:
        """Reverse lookup: MeshAxis → ProcessGroup."""
        if _MESH is None:
            return None
        for name, a in _MESH._axes.items():
            if a is ax:
                return self.pgs.get(name)
        return None

    @property
    def tp_pg(self) -> ProcessGroup | None:
        return self.pgs.get("tp")


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_MESH: MeshAxes | None = None
_STATE: SpmdState | None = None


# Sentinel mesh for pre-init attribute access (raises clear error)
class _UninitializedMesh:
    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(
            f"SPMD mesh not initialized. Cannot access mesh.{name}. "
            "Call init_spmd_state() during build_mesh()."
        )


_mesh: MeshAxes | Any = _UninitializedMesh()


def mesh() -> MeshAxes:
    """Return the global mesh singleton. Always-valid axes after init."""
    return _mesh


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


def init_spmd_state(
    mesh_axes: MeshAxes,
    state: SpmdState,
) -> None:
    """Set the global SPMD mesh and state. Called once during mesh setup."""
    global _MESH, _STATE, _mesh
    _MESH = mesh_axes
    _mesh = mesh_axes
    _STATE = state
