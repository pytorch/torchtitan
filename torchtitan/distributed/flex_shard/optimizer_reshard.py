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

from torchtitan.distributed.parallel_dims import MeshAxisName


__all__ = ["BucketConfig", "ComputeLayout"]


class _FrozenAxisPlacements(Mapping[MeshAxisName, Replicate | Shard]):
    """Small immutable mapping that remains safe to copy with configs."""

    __slots__ = ("_items",)

    def __init__(
        self,
        placements: Mapping[MeshAxisName, Replicate | Shard],
    ) -> None:
        self._items = tuple(placements.items())

    def __getitem__(self, axis_name: MeshAxisName) -> Replicate | Shard:
        for candidate_axis_name, placement in self._items:
            if candidate_axis_name == axis_name:
                return placement
        raise KeyError(axis_name)

    def __iter__(self) -> Iterator[MeshAxisName]:
        return (axis_name for axis_name, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return hash(self._items)

    def __repr__(self) -> str:
        return repr(dict(self._items))

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenAxisPlacements:
        return self


@dataclass(frozen=True, slots=True)
class ComputeLayout:
    """Describe a temporary compute layout on named storage-mesh axes.

    Placements override the named axes; omitted axes preserve their storage
    placement. Ownership axes assign each complete logical matrix to one
    participant in their Cartesian group. Extra declarations may target mesh
    variants not used by a particular parameter, but at least one declaration
    must apply when the layout is resolved.
    """

    axis_placements: Mapping[MeshAxisName, Replicate | Shard] = field(
        default_factory=dict
    )
    owner_mesh_axis_names: tuple[MeshAxisName, ...] = ()

    def __post_init__(self) -> None:
        axis_placements = dict(self.axis_placements)
        owner_mesh_axis_names = tuple(self.owner_mesh_axis_names)
        if not axis_placements and not owner_mesh_axis_names:
            raise ValueError("ComputeLayout must declare a placement or owner axis")
        for axis_name, placement in axis_placements.items():
            if type(axis_name) is not MeshAxisName:
                raise ValueError("ComputeLayout placement axes must use MeshAxisName")
            if type(placement) not in (Replicate, Shard):
                raise ValueError("ComputeLayout placements must be Replicate or Shard")
        if any(
            type(axis_name) is not MeshAxisName for axis_name in owner_mesh_axis_names
        ):
            raise ValueError("ComputeLayout owner axes must use MeshAxisName")
        if len(set(owner_mesh_axis_names)) != len(owner_mesh_axis_names):
            raise ValueError("ComputeLayout owner axes must be unique")
        overlapping_axes = set(axis_placements).intersection(owner_mesh_axis_names)
        if overlapping_axes:
            names = sorted(axis_name.value for axis_name in overlapping_axes)
            raise ValueError(
                "ComputeLayout axes cannot have both placement and ownership: "
                f"{names}"
            )

        normalized_axis_placements = dict(
            sorted(axis_placements.items(), key=lambda item: item[0].value)
        )
        normalized_owner_axes = tuple(
            sorted(owner_mesh_axis_names, key=lambda axis_name: axis_name.value)
        )
        object.__setattr__(
            self,
            "axis_placements",
            _FrozenAxisPlacements(normalized_axis_placements),
        )
        object.__setattr__(self, "owner_mesh_axis_names", normalized_owner_axes)


@dataclass(frozen=True, slots=True)
class BucketConfig:
    """Ordered optimizer-work bucket selected by canonical FQN patterns.

    Compute layouts determine communication topology. A bucket controls only
    scheduling order and overlap; all redistributed parameters it selects must
    currently resolve to one homogeneous transport group.
    """

    patterns: tuple[str, ...]
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "patterns", tuple(self.patterns))

    def _bind(self, mesh: DeviceMesh | None) -> _BucketSpec:
        return _BucketSpec(
            patterns=self.patterns,
            mesh=mesh,
            name=self.name,
        )


@dataclass(frozen=True, slots=True)
class _BucketSpec:
    """One ordered optimizer-work bucket selected by canonical FQN.

    Patterns use case-sensitive ``fnmatch`` syntax. Every optimizer FQN must
    match exactly one bucket, and sequence order controls execution order.
    ``mesh`` is the bucket's exact one-dimensional communication mesh, or
    ``None`` when every matched parameter is already compute-ready. ``name`` is
    diagnostic metadata only.
    """

    patterns: tuple[str, ...]
    mesh: DeviceMesh | None
    name: str = ""

    def __post_init__(self) -> None:
        if self.mesh is not None and self.mesh.ndim != 1:
            raise ValueError("bucket mesh must be one-dimensional")
        object.__setattr__(self, "patterns", tuple(self.patterns))
