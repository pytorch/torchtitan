from __future__ import annotations

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
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from torch.distributed.tensor import Partial, Placement, Replicate, Shard

import spmd_types as spmd
from torchtitan.protocols.types import MeshAxisName

if TYPE_CHECKING:
    from torchtitan.distributed.parallel_dims import ParallelDims


NamedPlacement = dict[MeshAxisName, Placement]

NamedSpmdType = dict[MeshAxisName, spmd.PerMeshAxisSpmdType]


# ---------------------------------------------------------------------------
# SPMD annotation and config types
# ---------------------------------------------------------------------------


@dataclass(kw_only=True, slots=True)
class SpmdAnnotation:
    """Per-tensor SPMD type annotation with optional PartitionSpec.

    Stores unresolved ``MeshAxisName`` keys. Call ``resolve()`` at
    parallelize time to produce resolved ``{MeshAxis: spmd_type}``.

    Attributes:
        types: SPMD type per mesh axis.
        partition_spec: Optional partition spec template. Needed when
            multiple axes shard different tensor dims.
    """

    types: NamedSpmdType
    partition_spec: tuple | None = None

    def resolve(
        self, parallel_dims: ParallelDims,
    ) -> spmd.PerMeshAxisSpmdTypes | tuple[spmd.PerMeshAxisSpmdTypes, spmd.PartitionSpec]:
        resolved = {
            parallel_dims.get_spmd_axis(name.value): t
            for name, t in self.types.items()
            if parallel_dims.get_spmd_axis(name.value).size() > 1
        }
        if self.partition_spec is not None:
            from torch.utils._pytree import tree_map_only

            spec = tree_map_only(
                MeshAxisName,
                lambda n: parallel_dims.get_spmd_axis(n.value),
                self.partition_spec,
            )
            return (resolved, spmd.PartitionSpec(*spec))
        return resolved


@dataclass(kw_only=True, slots=True)
class SpmdRedist:
    """Source/destination pair for a single redistribute operation."""

    src: SpmdAnnotation
    dst: SpmdAnnotation

    def resolve(
        self, parallel_dims: ParallelDims,
    ) -> tuple[spmd.PerMeshAxisSpmdTypes, spmd.PerMeshAxisSpmdTypes]:
        src = self.src.resolve(parallel_dims)
        dst = self.dst.resolve(parallel_dims)
        if isinstance(src, tuple):
            src = src[0]
        if isinstance(dst, tuple):
            dst = dst[0]
        return (src, dst)


@dataclass(kw_only=True, slots=True)
class LocalSpmdConfig:
    """Spec for modules using ``spmd.local_map`` for local typechecking.

    Wraps a module's forward so that type checking is suspended inside the
    body; ``spmd.local_map`` re-establishes SPMD types at the boundary.

    Attributes:
        out: Output annotation.
        inputs: Per-input annotations (positional).
            ``None`` means ``Infer`` (inputs pass through unchecked).
    """

    out: SpmdAnnotation
    inputs: tuple[SpmdAnnotation, ...] | None = None

    def to_dict(self) -> dict:
        return {"repr": repr(self)}


@dataclass(kw_only=True, slots=True)
class GlobalSpmdConfig:
    """Redistribution specs for the SPMD path (positional).

    Attributes:
        inputs: Per-input redistribute specs (positional).
            ``None`` entries mean no redistribution for that arg.
        output: Output redistribute spec.
    """

    inputs: tuple[SpmdRedist | None, ...] | None = None
    output: SpmdRedist | None = None

    def to_dict(self) -> dict:
        return {"repr": repr(self)}


# ---------------------------------------------------------------------------
# DTensor sharding config types
# ---------------------------------------------------------------------------


@dataclass(kw_only=True, slots=True)
class LocalMapConfig:
    """Spec for modules computing on local tensors (DTensor path).

    Wraps forward with ``local_map()``: DTensor -> local before forward,
    local -> DTensor after forward.

    Attributes:
        in_placements: Per-input NamedPlacements (positional: q, k, v).
        out_placements: Per-output NamedPlacements.
        in_grad_placements: Per-input-gradient NamedPlacements.
    """

    in_placements: tuple[NamedPlacement, ...]
    out_placements: tuple[NamedPlacement, ...]
    in_grad_placements: tuple[NamedPlacement, ...]

    def to_dict(self) -> dict:
        return {"repr": repr(self)}


@dataclass(kw_only=True, slots=True)
class ShardingConfig:
    """Declarative sharding for a Module's states and activations.

    Used by both DTensor and SPMD paths. ``state_shardings`` applies to both.
    DTensor-specific fields: ``in_src/dst_shardings``, ``out_dst_shardings``,
    ``local_map``. SPMD config is stored separately on ``Module._spmd_config``.

    Attributes:
        state_shardings: Parameter/buffer placements. Used by both DTensor
            (``distribute_tensor``) and SPMD (``assert_type``) paths.
        in_src_shardings: Source placements of inputs (DTensor path).
        in_dst_shardings: Desired input placements after redistribution (DTensor path).
        out_dst_shardings: Desired output placement (DTensor path).
        local_map: DTensor local_map boundary.
    """

    state_shardings: dict[str, NamedPlacement] = field(default_factory=dict)
    in_src_shardings: dict[str, NamedPlacement] | None = None
    in_dst_shardings: dict[str, NamedPlacement] | None = None
    out_dst_shardings: NamedPlacement | None = None
    local_map: LocalMapConfig | None = None

    def to_dict(self) -> dict:
        """Serialize for JSON logging. Placements become repr strings."""
        return {"repr": repr(self)}


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def resolve_placements(
    named: NamedPlacement,
    mesh_axis_names: tuple[str, ...],
) -> tuple[Placement, ...]:
    """Resolve NamedPlacement against a mesh in axis order."""
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


def placement_to_spmd_type(
    placement: Placement,
    *,
    is_compute: bool = False,
) -> spmd.PerMeshAxisSpmdType:
    """Convert a DTensor Placement to an spmd_types type.

    Shard(dim) -> S(dim), Partial() -> P.
    Replicate() -> R in compute regions (activations: gradient needs reduction),
    I at rest (params: identical, no grad reduction by default).
    """
    if isinstance(placement, Shard):
        return spmd.S(placement.dim)
    if isinstance(placement, Partial):
        return spmd.P
    return spmd.R if is_compute else spmd.I


def resolve_spmd(
    named: NamedPlacement | None,
    parallel_dims: ParallelDims,
    *,
    is_compute: bool = False,
) -> spmd.PerMeshAxisSpmdTypes | None:
    """Resolve NamedPlacement to {MeshAxis: spmd_type}. Skips disabled axes."""
    if named is None:
        return None
    result: spmd.PerMeshAxisSpmdTypes = {}
    for axis_name, placement in named.items():
        axis = parallel_dims.get_spmd_axis(axis_name.value)
        if axis.size() <= 1:
            continue
        result[axis] = placement_to_spmd_type(placement, is_compute=is_compute)
    return result
