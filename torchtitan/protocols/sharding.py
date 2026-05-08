# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding types for config-based parallelization.

``ShardingConfig`` is set on ``Module.Config`` by ``set_sharding_config()``
and read by ``Module.parallelize()``.  All placements use
``NamedPlacement`` (dict keyed by ``MeshAxisName``) so they are
self-documenting and support multi-dimensional meshes.

Placement values are either DTensor ``Placement`` objects or ``spmd_types``
types (``R``/``I``/``V``/``P``/``S(dim)``).  ``parallelize()`` duck-types
on the values to dispatch DTensor vs SPMD paths.
"""

from dataclasses import dataclass, field, fields

import spmd_types as spmd
from torch.distributed.tensor import Placement
from torch.utils._pytree import tree_leaves

from torchtitan.protocols.types import MeshAxisName

# Value is either:
#   - DTensor Placement (Shard, Replicate, Partial)
#   - spmd_types type (R, I, V, P, S(dim))
#   - (spmd_type, PartitionSpec) tuple for multi-axis-same-dim ordering
#
# parallelize() duck-types on the value to dispatch DTensor vs SPMD paths.
SpmdPlacement = (
    spmd.PerMeshAxisSpmdType
    | tuple[spmd.PerMeshAxisSpmdType, spmd.PartitionSpec]
)
NamedPlacement = dict[MeshAxisName, Placement | SpmdPlacement]


def _is_spmd_value(value) -> bool:
    """True if a single placement value is an spmd_types type."""
    if isinstance(value, tuple):
        return isinstance(value[0], (spmd.PerMeshAxisLocalSpmdType, spmd.Shard))
    return isinstance(value, (spmd.PerMeshAxisLocalSpmdType, spmd.Shard))


@dataclass(kw_only=True, slots=True)
class LocalMapConfig:
    """Spec for modules computing on local tensors.

    Wraps forward with ``local_map()`` (DTensor) or ``spmd.local_map()``
    (SPMD): strips distributed wrapper before forward, re-establishes it
    after.

    Placements are ``NamedPlacement`` (keyed by mesh axis name). At
    parallelize time they are resolved to positional tuples (DTensor) or
    per-axis type dicts (SPMD) matching the runtime mesh.

    Attributes:
        in_placements: Per-input NamedPlacements (positional: q, k, v).
        out_placements: Per-output NamedPlacements.
        in_grad_placements: Per-input-gradient NamedPlacements.
            Ignored for SPMD path (no backward typechecking).
    """

    in_placements: tuple[NamedPlacement, ...]
    out_placements: tuple[NamedPlacement, ...]
    in_grad_placements: tuple[NamedPlacement, ...] | None = None

    def to_dict(self) -> dict:
        return {"repr": repr(self)}


@dataclass(kw_only=True, slots=True)
class ShardingConfig:
    """Declarative sharding for a Module's states and activations.

    All placements use ``NamedPlacement`` (``dict[MeshAxisName, ...]``)
    keyed by mesh axis names.  Values are either DTensor ``Placement``
    objects or ``spmd_types`` types — ``parallelize()`` duck-types on
    them to dispatch the right path.

    Redistribution is expressed as a (source, destination) pair: src declares
    what the tensor's placement is entering the boundary, dst declares the
    desired placement after redistribution.

    Attributes:
        state_shardings: Parameter/buffer placements.
            DTensor path: used with ``distribute_tensor``.
            SPMD path: used with ``spmd.assert_type`` (+ physical TP shard).
            e.g. ``{"weight": {TP: Shard(0)}}`` or ``{"weight": {TP: S(0)}}``.
        in_src_shardings: Source placements of inputs, keyed by ``forward()``
            arg name. Declares what the input's placement is before any
            redistribution.
        in_dst_shardings: Desired input placements after redistribution,
            keyed by ``forward()`` arg name.
            ``None`` means no input redistribution.
        out_src_shardings: Source placement of outputs before redistribution.
            DTensor path: usually implicit in tensor's placements.
            SPMD path: required for ``spmd.redistribute``.
        out_dst_shardings: Desired output placement after redistribution.
            ``None`` means no output redistribution.
        local_map: If set, wraps forward with ``local_map()`` or
            ``spmd.local_map()``.
    """

    state_shardings: dict[str, NamedPlacement] = field(default_factory=dict)
    in_src_shardings: dict[str, NamedPlacement] | None = None
    in_dst_shardings: dict[str, NamedPlacement] | None = None
    out_src_shardings: NamedPlacement | None = None
    out_dst_shardings: NamedPlacement | None = None
    local_map: LocalMapConfig | None = None
    use_spmd: bool = field(default=False, init=False)

    def validate(self) -> None:
        """Check consistency and set ``use_spmd``.

        Called by ``parallelize()``. Ensures all NamedPlacement values are
        consistently DTensor or spmd_types.
        """

        def _is_np(v):
            return isinstance(v, dict) and v and isinstance(next(iter(v)), MeshAxisName)

        placements = [
            v
            for v in tree_leaves(
                [getattr(self, f.name) for f in fields(self) if f.name != "use_spmd"],
                is_leaf=_is_np,
            )
            if _is_np(v)
        ]
        if not placements:
            return

        all_values = [v for np in placements for v in np.values()]
        spmd_flags = [_is_spmd_value(v) for v in all_values]
        self.use_spmd = spmd_flags[0]
        if not all(f == self.use_spmd for f in spmd_flags[1:]):
            raise ValueError(
                "ShardingConfig mixes DTensor Placements and spmd_types values. "
                "All NamedPlacement values must be consistently one or the other."
            )
        if self.use_spmd and self.out_dst_shardings is not None and self.out_src_shardings is None:
            raise ValueError(
                "SPMD path requires out_src_shardings when out_dst_shardings is set."
            )

    def to_dict(self) -> dict:
        """Serialize for JSON logging. Placements become repr strings."""
        return {"repr": repr(self)}


def resolve_placements(
    named: NamedPlacement,
    mesh_axis_names: tuple[str, ...],
) -> tuple[Placement, ...]:
    """Resolve NamedPlacement against a mesh in axis order.

    Every sharding_config must explicitly declare a placement for every mesh axis
    it will be applied against. Missing declarations raise ``ValueError``;
    extra declarations (axes not in the mesh) are ignored.
    """
    result = []
    for axis_name in mesh_axis_names:
        key = MeshAxisName(axis_name)
        if key not in named:
            raise ValueError(
                f"Sharding sharding_config does not declare a placement for mesh axis "
                f"{axis_name!r}. Declared: "
                f"{sorted(k.value for k in named)}; "
                f"required: {list(mesh_axis_names)}."
            )
        result.append(named[key])
    return tuple(result)
