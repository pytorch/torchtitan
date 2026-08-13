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

__all__ = ["BucketConfig", "ComputeLayout", "SingleParticipant"]


@dataclass(frozen=True, slots=True)
class SingleParticipant:
    """Assign the complete subgroup-local logical tensor to one participant.

    This is a temporary compute distribution, not a DTensor storage placement.
    The consuming optimizer chooses the participant.
    """

    pass


class _FrozenDistributionByMeshAxis(
    Mapping[str, SingleParticipant | Replicate | Shard]
):
    """Small immutable mapping that remains safe to copy with configs."""

    __slots__ = ("_items",)

    def __init__(
        self,
        distribution_by_mesh_axis: Mapping[str, SingleParticipant | Replicate | Shard],
    ) -> None:
        self._items = tuple(distribution_by_mesh_axis.items())

    def __getitem__(self, axis_name: str) -> SingleParticipant | Replicate | Shard:
        for candidate_axis_name, distribution in self._items:
            if candidate_axis_name == axis_name:
                return distribution
        raise KeyError(axis_name)

    def __iter__(self) -> Iterator[str]:
        return (axis_name for axis_name, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return hash(self._items)

    def __repr__(self) -> str:
        return repr(dict(self._items))

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenDistributionByMeshAxis:
        return self


@dataclass(frozen=True, slots=True)
class ComputeLayout:
    """Describe a temporary compute layout on named storage-mesh axes.

    Each named axis uses one of three compute distributions: ``SingleParticipant``
    assigns the complete subgroup-local logical tensor to one participant,
    ``Replicate`` assigns it to every participant, and ``Shard`` partitions one
    tensor dimension.
    Omitted axes preserve their storage placement. Extra declarations may target
    mesh variants not used by a particular parameter, but at least one declaration
    must apply when the layout is resolved.

    Examples:
        Shard a viewed matrix batch across the EFSDP and EP axes::

            ComputeLayout(
                distribution_by_mesh_axis={
                    "efsdp": Shard(0),
                    "ep": Shard(0),
                }
            )

        Assign the complete subgroup-local logical tensor to one participant
        along ``dp_shard``::

            ComputeLayout(
                distribution_by_mesh_axis={
                    "dp_shard": SingleParticipant(),
                }
            )
    """

    distribution_by_mesh_axis: Mapping[str, SingleParticipant | Replicate | Shard]

    def __post_init__(self) -> None:
        distribution_by_mesh_axis = dict(self.distribution_by_mesh_axis)
        if not distribution_by_mesh_axis:
            raise ValueError("ComputeLayout must declare a compute distribution")
        for axis_name, distribution in distribution_by_mesh_axis.items():
            if not isinstance(axis_name, str):
                raise ValueError(
                    "ComputeLayout.distribution_by_mesh_axis keys must be strings"
                )
            if type(distribution) not in (SingleParticipant, Replicate, Shard):
                raise ValueError(
                    "ComputeLayout.distribution_by_mesh_axis values must be "
                    "SingleParticipant, Replicate, or Shard"
                )
        normalized_distribution_by_mesh_axis = dict(
            sorted(distribution_by_mesh_axis.items())
        )
        object.__setattr__(
            self,
            "distribution_by_mesh_axis",
            _FrozenDistributionByMeshAxis(normalized_distribution_by_mesh_axis),
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
