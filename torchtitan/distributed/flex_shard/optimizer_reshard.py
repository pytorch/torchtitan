# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public optimizer-reshard configuration."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard

__all__ = ["BucketConfig", "ComputeLayout", "Owned"]


@dataclass(frozen=True, slots=True)
class Owned:
    """Assign the complete subgroup-local logical tensor to one owner rank.

    The owner is selected dynamically from the communication submesh and exists
    only during the compute phase; this is not persistent DTensor storage
    ownership. Multiple mesh axes using ``Owned`` select one owner rank from
    their Cartesian product.
    """

    pass


class _FrozenShardingsByMeshAxis(Mapping[str, Owned | Replicate | Shard]):
    """Small immutable mapping that remains safe to copy with configs."""

    __slots__ = ("_items",)

    def __init__(
        self,
        shardings_by_mesh_axis: Mapping[str, Owned | Replicate | Shard],
    ) -> None:
        self._items = tuple(shardings_by_mesh_axis.items())

    def __getitem__(self, axis_name: str) -> Owned | Replicate | Shard:
        for candidate_axis_name, sharding in self._items:
            if candidate_axis_name == axis_name:
                return sharding
        raise KeyError(axis_name)

    def __iter__(self) -> Iterator[str]:
        return (axis_name for axis_name, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return hash(self._items)

    def __repr__(self) -> str:
        return repr(dict(self._items))

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenShardingsByMeshAxis:
        return self


@dataclass(frozen=True, slots=True)
class ComputeLayout:
    """Describe temporary compute shardings on named storage-mesh axes.

    Each named axis uses one of three compute shardings: ``Owned`` assigns the
    complete subgroup-local logical tensor to one dynamically selected owner rank,
    ``Replicate`` assigns it to every rank, and ``Shard`` partitions one
    tensor dimension.
    Multiple ``Owned`` shardings select one owner rank from their joint Cartesian
    group.
    Omitted axes preserve their storage placement. Extra declarations may target
    mesh variants not used by a particular parameter, but at least one declaration
    must apply when the layout is resolved.

    Examples:
        Shard a viewed matrix batch across the EFSDP and EP axes::

            ComputeLayout(
                shardings_by_mesh_axis={
                    "efsdp": Shard(0),
                    "ep": Shard(0),
                }
            )

        Assign the complete subgroup-local logical tensor to one owner rank
        along ``dp_shard``::

            ComputeLayout(
                shardings_by_mesh_axis={
                    "dp_shard": Owned(),
                }
            )
    """

    shardings_by_mesh_axis: Mapping[str, Owned | Replicate | Shard]

    def __post_init__(self) -> None:
        shardings_by_mesh_axis = dict(self.shardings_by_mesh_axis)
        if not shardings_by_mesh_axis:
            raise ValueError("ComputeLayout must declare a compute sharding")
        for axis_name, sharding in shardings_by_mesh_axis.items():
            if not isinstance(axis_name, str):
                raise ValueError(
                    "ComputeLayout.shardings_by_mesh_axis keys must be strings"
                )
            if type(sharding) not in (Owned, Replicate, Shard):
                raise ValueError(
                    "ComputeLayout.shardings_by_mesh_axis values must be "
                    "Owned, Replicate, or Shard"
                )
        normalized_shardings_by_mesh_axis = dict(sorted(shardings_by_mesh_axis.items()))
        object.__setattr__(
            self,
            "shardings_by_mesh_axis",
            _FrozenShardingsByMeshAxis(normalized_shardings_by_mesh_axis),
        )


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
