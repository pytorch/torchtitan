# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public optimizer-reshard configuration."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard

__all__ = ["BucketConfig", "ComputeLayout"]


class _FrozenPlacementsByMeshAxis(Mapping[str, Replicate | Shard]):
    """Small immutable mapping that remains safe to copy with configs."""

    __slots__ = ("_items",)

    def __init__(
        self,
        placements_by_mesh_axis: Mapping[str, Replicate | Shard],
    ) -> None:
        self._items = tuple(placements_by_mesh_axis.items())

    def __getitem__(self, axis_name: str) -> Replicate | Shard:
        for candidate_axis_name, placement in self._items:
            if candidate_axis_name == axis_name:
                return placement
        raise KeyError(axis_name)

    def __iter__(self) -> Iterator[str]:
        return (axis_name for axis_name, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return hash(self._items)

    def __repr__(self) -> str:
        return repr(dict(self._items))

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenPlacementsByMeshAxis:
        return self


@dataclass(frozen=True, slots=True)
class ComputeLayout:
    """Describe a temporary compute layout on named storage-mesh axes.

    Placements override the named axes; omitted axes preserve their storage
    placement. Matrix ownership axes assign each complete logical matrix to one
    participant in their Cartesian group. Extra declarations may target mesh
    variants not used by a particular parameter, but at least one declaration
    must apply when the layout is resolved.

    Examples:
        Shard a viewed matrix batch across the EFSDP and EP axes::

            ComputeLayout(
                placements_by_mesh_axis={
                    "efsdp": Shard(0),
                    "ep": Shard(0),
                }
            )

        Assign each complete matrix to one rank along ``dp_shard``::

            ComputeLayout(matrix_ownership_axes=("dp_shard",))
    """

    placements_by_mesh_axis: Mapping[str, Replicate | Shard] = field(
        default_factory=dict
    )
    matrix_ownership_axes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        placements_by_mesh_axis = dict(self.placements_by_mesh_axis)
        matrix_ownership_axes = tuple(self.matrix_ownership_axes)
        if not placements_by_mesh_axis and not matrix_ownership_axes:
            raise ValueError(
                "ComputeLayout must declare a placement or matrix ownership axis"
            )
        for axis_name, placement in placements_by_mesh_axis.items():
            if not isinstance(axis_name, str):
                raise ValueError(
                    "ComputeLayout.placements_by_mesh_axis keys must be strings"
                )
            if type(placement) not in (Replicate, Shard):
                raise ValueError(
                    "ComputeLayout.placements_by_mesh_axis values must be "
                    "Replicate or Shard"
                )
        if any(not isinstance(axis_name, str) for axis_name in matrix_ownership_axes):
            raise ValueError(
                "ComputeLayout.matrix_ownership_axes entries must be strings"
            )
        if len(set(matrix_ownership_axes)) != len(matrix_ownership_axes):
            raise ValueError(
                "ComputeLayout.matrix_ownership_axes entries must be unique"
            )
        overlapping_axes = set(placements_by_mesh_axis).intersection(
            matrix_ownership_axes
        )
        if overlapping_axes:
            names = sorted(overlapping_axes)
            raise ValueError(
                "ComputeLayout axes cannot have both placement and matrix ownership: "
                f"{names}"
            )

        normalized_placements_by_mesh_axis = dict(
            sorted(placements_by_mesh_axis.items())
        )
        normalized_matrix_ownership_axes = tuple(sorted(matrix_ownership_axes))
        object.__setattr__(
            self,
            "placements_by_mesh_axis",
            _FrozenPlacementsByMeshAxis(normalized_placements_by_mesh_axis),
        )
        object.__setattr__(
            self,
            "matrix_ownership_axes",
            normalized_matrix_ownership_axes,
        )


@dataclass(frozen=True, slots=True)
class BucketConfig:
    """Ordered optimizer-work bucket selected by canonical FQN patterns.

    Compute layouts determine communication topology. A bucket controls only
    scheduling order and overlap. FlexShard internally splits its selected
    parameters into adjacent homogeneous physical transport buckets.
    """

    patterns: tuple[str, ...]
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "patterns", tuple(self.patterns))

    def _bind(
        self,
        mesh: DeviceMesh | None,
        fqns: tuple[str, ...],
    ) -> _BucketSpec:
        return _BucketSpec(
            fqns=fqns,
            mesh=mesh,
            name=self.name,
        )


@dataclass(frozen=True, slots=True)
class _BucketSpec:
    """One resolved physical optimizer-work bucket.

    ``fqns`` are bound from one public ``BucketConfig`` after compute layouts
    select concrete transport groups. ``mesh`` is the bucket's exact 1D
    communication mesh, which may flatten a Cartesian storage-mesh submesh,
    or ``None`` when every parameter is already compute-ready. ``name`` is
    diagnostic metadata only.
    """

    fqns: tuple[str, ...]
    mesh: DeviceMesh | None
    name: str = ""

    def __post_init__(self) -> None:
        if self.mesh is not None and self.mesh.ndim != 1:
            raise ValueError("bucket mesh must be one-dimensional")
        object.__setattr__(self, "fqns", tuple(self.fqns))
