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

from torch.distributed.tensor import Replicate, Shard

__all__ = [
    "BlockShard",
    "BucketConfig",
    "ComputeLayout",
    "NoRedistribution",
    "Owned",
]


@dataclass(frozen=True, slots=True)
class Owned:
    """Assign the complete subgroup-local logical tensor to one owner rank.

    The owner is selected dynamically from the communication submesh and exists
    only during the compute phase; this is not persistent DTensor storage
    ownership. Multiple mesh axes using ``Owned`` select one owner rank from
    their Cartesian product.
    """

    pass


@dataclass(frozen=True, slots=True)
class BlockShard:
    """Shard fixed-size blocks without changing the tensor shape.

    The selected tensor dimension is partitioned into contiguous blocks of
    ``block_size`` elements, and those blocks are sharded using the same
    contiguous partitioning as ``Shard``. A block is never split between
    participants. ``BlockShard`` describes only distribution; it does not
    reshape or reinterpret the tensor.
    """

    dim: int
    block_size: int

    def __post_init__(self) -> None:
        if isinstance(self.dim, bool) or not isinstance(self.dim, int):
            raise ValueError("BlockShard.dim must be an integer")
        if (
            isinstance(self.block_size, bool)
            or not isinstance(self.block_size, int)
            or self.block_size <= 0
        ):
            raise ValueError("BlockShard.block_size must be a positive integer")


@dataclass(frozen=True, slots=True)
class NoRedistribution:
    """Declare a bucket whose storage shards are already compute-ready.

    Parameters in this bucket may still use sharded DTensor storage. The marker
    means only that optimizer compute needs no storage-to-compute redistribution.
    """

    pass


class _FrozenShardingsByMeshAxis(Mapping[str, Owned | Replicate | Shard | BlockShard]):
    """Small immutable mapping that remains safe to copy with configs."""

    __slots__ = ("_items",)

    def __init__(
        self,
        shardings_by_mesh_axis: Mapping[str, Owned | Replicate | Shard | BlockShard],
    ) -> None:
        self._items = tuple(shardings_by_mesh_axis.items())

    def __getitem__(self, axis_name: str) -> Owned | Replicate | Shard | BlockShard:
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

    These shardings apply before any optimizer-local tensor view.

    Each named axis uses one of four compute shardings: ``Owned`` assigns the
    complete subgroup-local logical tensor to one dynamically selected owner rank,
    ``Replicate`` assigns it to every rank, ``Shard`` partitions one tensor
    dimension, and ``BlockShard`` partitions fixed-size blocks without changing
    the tensor shape.
    Multiple ``Owned`` shardings select one owner rank from their joint Cartesian
    group.
    Omitted axes preserve their storage placement. Extra declarations may target
    mesh variants not used by a particular parameter, but at least one declaration
    must apply when the layout is resolved.

    Examples:
        Shard a logical batch of matrices across the EFSDP and EP axes::

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

        Shard complete four-row blocks along tensor dimension 0::

            ComputeLayout(
                shardings_by_mesh_axis={
                    "dp_shard": BlockShard(dim=0, block_size=4),
                }
            )
    """

    shardings_by_mesh_axis: Mapping[str, Owned | Replicate | Shard | BlockShard]

    def __post_init__(self) -> None:
        shardings_by_mesh_axis = dict(self.shardings_by_mesh_axis)
        if not shardings_by_mesh_axis:
            raise ValueError("ComputeLayout must declare a compute sharding")
        for axis_name, sharding in shardings_by_mesh_axis.items():
            if not isinstance(axis_name, str):
                raise ValueError(
                    "ComputeLayout.shardings_by_mesh_axis keys must be strings"
                )
            if type(sharding) not in (Owned, Replicate, Shard, BlockShard):
                raise ValueError(
                    "ComputeLayout.shardings_by_mesh_axis values must be "
                    "Owned, Replicate, Shard, or BlockShard"
                )
        normalized_shardings_by_mesh_axis = dict(sorted(shardings_by_mesh_axis.items()))
        object.__setattr__(
            self,
            "shardings_by_mesh_axis",
            _FrozenShardingsByMeshAxis(normalized_shardings_by_mesh_axis),
        )

    def to_dict(self) -> dict:
        """Serialize for JSON logging."""
        return {"repr": repr(self)}


@dataclass(frozen=True, slots=True)
class BucketConfig:
    """One physical optimizer-redistribution bucket.

    ``redistribution_mesh_axis_names`` declares the ordered storage-mesh axes
    whose Cartesian product forms this bucket's exact redistribution group.
    ``NoRedistribution`` declares local-only work. A config is never split
    automatically.

    Multiple configs may use the same patterns with different redistribution
    choices. This explicitly describes runtime-topology alternatives; each
    local parameter must resolve to exactly one matching pattern-and-choice
    config.
    """

    patterns: tuple[str, ...]
    redistribution_mesh_axis_names: tuple[str, ...] | NoRedistribution

    def __post_init__(self) -> None:
        object.__setattr__(self, "patterns", tuple(self.patterns))
        redistribution = self.redistribution_mesh_axis_names
        if isinstance(redistribution, NoRedistribution):
            return
        if not isinstance(redistribution, tuple):
            raise ValueError(
                "BucketConfig.redistribution_mesh_axis_names must be a tuple "
                "or NoRedistribution"
            )
        redistribution_mesh_axis_names = tuple(redistribution)
        if not redistribution_mesh_axis_names:
            raise ValueError(
                "BucketConfig.redistribution_mesh_axis_names must be nonempty; "
                "use NoRedistribution() for local-only work"
            )
        for axis_name in redistribution_mesh_axis_names:
            if not isinstance(axis_name, str) or not axis_name:
                raise ValueError(
                    "BucketConfig.redistribution_mesh_axis_names values must be "
                    "nonempty strings"
                )
        if len(set(redistribution_mesh_axis_names)) != len(
            redistribution_mesh_axis_names
        ):
            raise ValueError(
                "BucketConfig.redistribution_mesh_axis_names must not contain "
                "duplicate axes"
            )
        object.__setattr__(
            self,
            "redistribution_mesh_axis_names",
            redistribution_mesh_axis_names,
        )
