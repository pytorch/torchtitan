# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding types for config-based parallelization.

``ShardingConfig`` is set on ``Module.Config`` by ``set_sharding_config()``
and read by ``Module.parallelize(mesh)``.  All placements use
``NamedPlacement`` (dict keyed by ``MeshAxisName``) so they are
self-documenting and support multi-dimensional meshes.

Placement values are either DTensor ``Placement`` objects or ``spmd_types``
types (``R``/``I``/``V``/``P``/``S(dim)``).  ``parallelize()`` duck-types
on the values to dispatch DTensor vs SPMD paths.
"""

from dataclasses import dataclass, field

import spmd_types as spmd
from torch.distributed.tensor import Partial, Placement, Replicate, Shard

from torchtitan.distributed.spmd_state import is_spmd_active
from torchtitan.protocols.types import MeshAxisName

# Value is either a DTensor Placement (Shard, Replicate, Partial) or an
# spmd_types type (R, I, V, P, S(dim)).  parallelize() duck-types on the
# value to dispatch DTensor vs SPMD paths.
NamedPlacement = dict[MeshAxisName, Placement | spmd.PerMeshAxisSpmdType]


def S(dim: int) -> spmd.Shard | Shard:
    """Shard on ``dim``. Returns ``spmd.S(dim)`` or ``Shard(dim)``."""
    return spmd.S(dim) if is_spmd_active() else Shard(dim)


def R() -> spmd.PerMeshAxisLocalSpmdType | Replicate:
    """Replicate. Returns ``spmd.R`` or ``Replicate()``."""
    return spmd.R if is_spmd_active() else Replicate()


def Inv() -> spmd.PerMeshAxisLocalSpmdType | Replicate:
    """Invariant. Returns ``spmd.I`` or ``Replicate()`` (DTensor equivalent)."""
    return spmd.I if is_spmd_active() else Replicate()


def P() -> spmd.PerMeshAxisLocalSpmdType | Partial:
    """Partial. Returns ``spmd.P`` or ``Partial()``."""
    return spmd.P if is_spmd_active() else Partial()


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
        in_ndims: Per-input tensor ndim hints for SPMD PartitionSpec
            resolution. Required when multiple axes shard the same dim
            (e.g. HSDP). ``None`` means no hints (ok when no collisions).
        out_ndims: Per-output tensor ndim hints.
    """

    in_placements: tuple[NamedPlacement, ...]
    out_placements: tuple[NamedPlacement, ...]
    in_grad_placements: tuple[NamedPlacement, ...] | None = None
    in_ndims: tuple[int, ...] | None = None
    out_ndims: tuple[int, ...] | None = None

    def to_dict(self) -> dict:
        return {"repr": repr(self)}


@dataclass(kw_only=True, slots=True)
class ShardingConfig:
    """Declarative sharding for a Module's states and activations.

    All placements use ``NamedPlacement`` (``dict[MeshAxisName, ...]``)
    keyed by mesh axis names.  Values are either DTensor ``Placement``
    objects or ``spmd_types`` types — ``parallelize()`` duck-types on
    them to dispatch the right path.

    Completely dtype-agnostic at this moment — quantization (Float8/MXFP8) is
    orthogonal.

    Redistribution is expressed as a (source, destination) pair: src declares
    what the tensor's placement is entering the boundary, dst declares the
    desired placement after redistribution. For DTensor, the src is usually
    implicit in the tensor's ``placements``; declaring it explicitly keeps
    the contract uniform with future erased-type systems that require both
    sides of every redistribute.

    Attributes:
        state_shardings: Parameter/buffer placements.
            DTensor path: used with ``distribute_tensor``.
            SPMD path: used with ``spmd.assert_type`` (+ physical TP shard).
            e.g. ``{"weight": {TP: Shard(0)}}`` or ``{"weight": {TP: S(0)}}``.
        state_tp_ir: Local parameter names to convert from I@tp at rest to R@tp
            during forward compute. Temporary SPMD escape hatch for replicated
            parameters that need TP gradient reduction semantics.
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
    state_tp_ir: set[str] = field(default_factory=set)
    in_src_shardings: dict[str, NamedPlacement] | None = None
    in_dst_shardings: dict[str, NamedPlacement] | None = None
    out_src_shardings: NamedPlacement | None = None
    out_dst_shardings: NamedPlacement | None = None
    local_map: LocalMapConfig | None = None

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
