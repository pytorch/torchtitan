# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Global SPMD state for the full_spmd_types path.

Populated once during ``ParallelDims.build_mesh()`` when
``full_spmd_types=True``. Read at forward time by any code that
needs mesh axes or process groups for SPMD collectives.

Usage::

    from torchtitan.distributed.spmd_state import spmd_state, is_spmd_active

    if is_spmd_active():
        state = spmd_state()
        pg = state.tp_pg
        dp_axes = state.dp_axes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch.distributed import ProcessGroup


@dataclass
class SpmdState:
    """Runtime SPMD state — axes, process groups, and derived info."""

    dp_axes: list[Any] = field(default_factory=list)
    tp_axis: Any | None = None
    cp_axis: Any | None = None
    all_axes: frozenset = field(default_factory=frozenset)
    _axes: dict[str, Any] = field(default_factory=dict)
    pgs: dict[str, ProcessGroup] = field(default_factory=dict)

    @property
    def tp_pg(self) -> ProcessGroup | None:
        return self.pgs.get("tp")

    @property
    def cp_active(self) -> bool:
        return self.cp_axis is not None and self.cp_axis.size() > 1

    @property
    def tp_active(self) -> bool:
        return self.tp_axis is not None and self.tp_axis.size() > 1

    def axis(self, name: str) -> Any:
        """Return the MeshAxis for a named axis, or a trivial axis if absent."""
        return self._axes[name]

    def pg_for_axis(self, ax: Any) -> ProcessGroup | None:
        """Reverse lookup: MeshAxis → ProcessGroup."""
        for name, a in self._axes.items():
            if a is ax:
                return self.pgs.get(name)
        return None


_STATE: SpmdState | None = None


def spmd_state() -> SpmdState:
    """Return the global SPMD state. Must be initialized first."""
    assert _STATE is not None, (
        "SPMD state not initialized. "
        "Call init_spmd_state() during build_mesh() with full_spmd_types=True."
    )
    return _STATE


def is_spmd_active() -> bool:
    """Check whether the SPMD path is active (SpmdState has been initialized)."""
    return _STATE is not None


def init_spmd_state(state: SpmdState) -> None:
    """Set the global SPMD state. Called once during mesh setup."""
    global _STATE
    _STATE = state
