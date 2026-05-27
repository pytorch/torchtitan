# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding types for config-based SPMD parallelization.

``ShardingConfig`` is set on ``Module.Config`` by ``set_sharding_config()``
and read by ``Module.parallelize(parallel_dims)``. Placements are keyed by mesh
axis name so they are self-documenting and support multi-dimensional meshes.
"""

from dataclasses import dataclass, field, fields

import spmd_types as spmd
from torch.utils._pytree import tree_map_only

from torchtitan.protocols.types import MeshAxisName


# Per-axis SPMD type, keyed by mesh axis name. ``MeshAxisName`` is a StrEnum,
# so these keys can be passed directly to spmd_types APIs that accept strings.
NamedPlacement = dict[MeshAxisName, spmd.PerMeshAxisSpmdType]
NamedPartitionSpecEntry = MeshAxisName | tuple[MeshAxisName, ...] | None
NamedPartitionSpec = tuple[NamedPartitionSpecEntry, ...]
_MESH_AXIS_NAMES = {axis.value for axis in MeshAxisName}


def _mesh_axis_name(axis_name: object) -> str:
    return axis_name.value if hasattr(axis_name, "value") else str(axis_name)


def _is_mesh_axis_key(key: object) -> bool:
    return _mesh_axis_name(key) in _MESH_AXIS_NAMES


@dataclass(frozen=True, kw_only=True, slots=True)
class PlacementSpec:
    """Placement with explicit global partition spec ordering.

    Use this only when multiple mesh axes shard the same tensor dimension. The
    ``placement`` field must use local SPMD types (``R``, ``I``, ``V``, ``P``)
    and put ``V`` on every axis named by ``partition_spec``.
    """

    placement: NamedPlacement
    partition_spec: NamedPartitionSpec


PlacementLike = NamedPlacement | PlacementSpec


@dataclass(kw_only=True, slots=True)
class ShardingConfig:
    """Declarative sharding for a Module's states and activations.

    Placements are usually ``NamedPlacement``. Use ``PlacementSpec`` only for
    explicit ordering when multiple axes shard the same tensor dimension.

    Completely dtype-agnostic at this moment — quantization (Float8/MXFP8) is
    orthogonal.

    Redistribution is expressed as a (source, destination) pair: src declares
    what the tensor's placement is entering the boundary, dst declares the
    desired placement after redistribution.

    Attributes:
        state_shardings: Parameter/buffer placements. Outer dict keys are
            param/buffer names, e.g. ``{"weight": {TP: spmd.S(0)}}``.
        state_shardings_compute: Parameter/buffer placements for forward
            compute when they differ from ``state_shardings``. Outer dict keys
            are param/buffer names, e.g. ``{"weight": {TP: spmd.R}}`` for a
            norm weight that is ``I@TP`` at rest but ``R@TP`` during compute.
        in_src_shardings: Source placements of inputs, keyed by ``forward()``
            arg name. Declares what the input's placement is before any
            redistribution.
        in_dst_shardings: Desired input placements after redistribution,
            keyed by ``forward()`` arg name.
            ``None`` means no input redistribution.
        out_src_shardings: Source placement of outputs before redistribution.
            Required for ``spmd.redistribute``.
        out_dst_shardings: Desired output placement after redistribution.
            ``None`` means no output redistribution.
        local_spmd: Wrap forward for local SPMD typechecking.
    """

    state_shardings: dict[str, PlacementLike] = field(default_factory=dict)
    state_shardings_compute: dict[str, PlacementLike] = field(default_factory=dict)
    in_src_shardings: dict[str, PlacementLike] | None = None
    in_dst_shardings: dict[str, PlacementLike] | None = None
    out_src_shardings: PlacementLike | tuple[PlacementLike, ...] | None = None
    out_dst_shardings: PlacementLike | None = None
    local_spmd: bool = False

    def axes(self) -> set[str]:
        """Return mesh axes referenced by this sharding config."""
        axes: set[str] = set()

        def is_placement(value: object) -> bool:
            return (
                isinstance(value, PlacementSpec)
                or (
                    isinstance(value, dict)
                    and len(value) > 0
                    and all(_is_mesh_axis_key(key) for key in value)
                )
            )

        def collect_axes(placement: PlacementLike) -> PlacementLike:
            named = (
                placement.placement
                if isinstance(placement, PlacementSpec)
                else placement
            )
            axes.update(_mesh_axis_name(axis_name) for axis_name in named)
            if isinstance(placement, PlacementSpec):
                for entry in placement.partition_spec:
                    if entry is None:
                        continue
                    entry_axes = entry if isinstance(entry, tuple) else (entry,)
                    axes.update(_mesh_axis_name(axis_name) for axis_name in entry_axes)
            return placement

        tree_map_only(
            is_placement,
            collect_axes,
            tuple(getattr(self, field.name) for field in fields(self)),
            is_leaf=is_placement,
        )
        return axes

    def to_dict(self) -> dict:
        """Serialize for JSON logging. Placements become repr strings."""
        return {"repr": repr(self)}


@dataclass(kw_only=True, slots=True)
class SpmdInputConfig:
    """Model-owned trainer input annotations for the SPMD path."""

    inputs: PlacementLike | None = None
    labels: PlacementLike | None = None
    extra_inputs: dict[str, PlacementLike] = field(default_factory=dict)
    extra_kwargs: dict[str, PlacementLike] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"repr": repr(self)}
