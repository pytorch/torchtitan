# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public optimizer-reshard configuration."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast, TypeVar

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard

__all__ = [
    "BlockShard",
    "BucketConfig",
    "ComputeLayout",
    "Owned",
    "build_transport_mesh",
    "precreate_transport_meshes",
]


# TODO: Use public root/flatten APIs when available
# (https://github.com/pytorch/pytorch/issues/159013).
_RootKey = tuple[str, tuple[str, ...], tuple[int, ...], tuple[int, ...]]
_TransportKey = tuple[_RootKey, tuple[str, ...], int]


def _root_key(root: DeviceMesh) -> _RootKey:
    """Identify a named mesh root consistently across world ranks."""
    return (
        root.device_type,
        root.mesh_dim_names or (),
        tuple(root.mesh.shape),
        tuple(root.mesh.flatten().tolist()),
    )


def _transport_name(axes: tuple[str, ...], max_ctas: int | None) -> str:
    """Return the native cache name for one transport and CTA policy."""
    joined = "_".join(axes)
    return joined if max_ctas is None else f"optimizer_{joined}_max_ctas_{max_ctas}"


def _flatten_transport_mesh(
    transport_mesh: DeviceMesh, max_ctas: int | None, backend: str
) -> DeviceMesh:
    """Use the root's native flatten cache, with a separate name for each cap."""
    name = _transport_name(
        cast(tuple[str, ...], transport_mesh.mesh_dim_names), max_ctas
    )
    if max_ctas is None:
        return transport_mesh._flatten(name)
    options = dist.ProcessGroupNCCL.Options()
    options.config.max_ctas = max_ctas
    cast(Any, options.config).split_share = 0
    # Native flatten caches by name/layout, not backend options. Keep distinct
    # caps distinct and avoid the existing-axis shortcut for a capped 1D mesh.
    return transport_mesh._flatten(
        name,
        backend_override=(backend, options),
    )


def precreate_transport_meshes(
    requests: Sequence[tuple[DeviceMesh, tuple[str, ...], int | None]],
    *,
    storage_meshes: Iterable[DeviceMesh],
    validation_errors: Sequence[str] = (),
) -> None:
    """Create a WORLD-wide union using existing local storage-mesh views.

    All WORLD ranks call in the same order, including ranks without requests.
    Requests contain storage meshes, ordered axes, and optional positive NCCL
    CTA caps. Two metadata exchanges agree on requests and validate local views
    and native cache hits before any group is created. ``validation_errors``
    lets the owning optimizer report local layout failures collectively.

    Each requested root must cover WORLD, and ``storage_meshes`` must contain
    an unambiguous view with its requested axes in root rank order on every rank.
    Missing/unsorted views, conflicting roots, and inconsistent cache hits raise
    ``ValueError`` on all ranks. Otherwise native ``DeviceMesh._flatten`` owns
    group creation/reuse.
    Supply precreated ``BucketConfig.mesh`` values for unsupported layouts or
    unequal optimizer-construction sequences. Fake groups skip the exchange.
    """
    if dist.get_backend() == "fake":
        if validation_errors:
            raise ValueError("; ".join(validation_errors))
        return
    views = tuple(storage_meshes)
    local_errors = list(validation_errors)
    if any(cap is not None and cap <= 0 for _, _, cap in requests):
        local_errors.append("Transport CTA caps must be positive when supplied")
    local_requests: set[_TransportKey] = {
        (_root_key(mesh._get_root_mesh()), axes, cap or 0)
        for mesh, axes, cap in requests
        if (len(axes) > 1 or cap is not None) and (cap is None or cap > 0)
    }
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, (local_requests, tuple(local_errors)))
    errors = sorted({error for _, rank_errors in gathered for error in rank_errors})
    if errors:
        raise ValueError("DistMuon transport preparation failed: " + "; ".join(errors))
    union = sorted({request for requests, _ in gathered for request in requests})
    if not union:
        return
    names = {(key, _transport_name(axes, cap or None)) for key, axes, cap in union}
    if len(names) != len(union):
        raise ValueError("Optimizer transport names must identify unique axis requests")

    resolved = []
    cache_hits = []
    for key, axes, cap in union:
        candidates = []
        for mesh in views:
            if _root_key(mesh._get_root_mesh()) != key or not all(
                axis in (mesh.mesh_dim_names or ()) for axis in axes
            ):
                continue
            try:
                candidates.append(mesh[axes])
            except (KeyError, ValueError, RuntimeError) as error:
                errors.append(f"Cannot resolve transport view {axes!r}: {error}")
        if not candidates or set(key[3]) != set(range(dist.get_world_size())):
            errors.append(
                f"Transport {axes!r} needs a WORLD-covering storage view on every "
                "rank; supply precreated BucketConfig.mesh values"
            )
            continue
        selected = candidates[0]
        root = selected._get_root_mesh()
        root_order = {rank: index for index, rank in enumerate(key[3])}
        positions = [root_order[rank] for rank in selected.mesh.flatten().tolist()]
        if positions != sorted(positions):
            errors.append(
                f"Transport {axes!r} must follow root rank order; "
                "supply precreated BucketConfig.mesh values for unsorted views"
            )
        if any(
            candidate._get_root_mesh() is not root or candidate != selected
            for candidate in candidates[1:]
        ):
            errors.append(f"Ambiguous storage views for transport {axes!r}")
        if cap and dist.get_backend(selected.get_group(0)) != "nccl":
            errors.append("Transport CTA caps must be positive and require NCCL")
        name = _transport_name(axes, cap or None)
        if name in (root.mesh_dim_names or ()):
            errors.append(f"Transport name {name!r} conflicts with a root axis")
        try:
            cached = root[name]
        except KeyError:
            cached = None
            # An unsorted native cache entry exists but cannot be sliced.
            try:
                root.get_group(name)
            except KeyError:
                pass
            else:
                errors.append(
                    f"Cannot resolve native cache layout for transport {axes!r}"
                )
        if cached is not None and not torch.equal(
            cached.mesh.flatten(), selected.mesh.flatten()
        ):
            errors.append(f"Conflicting native cache layout for transport {axes!r}")
        cache_hits.append(cached is not None)
        resolved.append(selected)
    dist.all_gather_object(gathered, (tuple(errors), tuple(cache_hits)))
    errors = sorted({error for rank_errors, _ in gathered for error in rank_errors})
    if errors:
        raise ValueError("DistMuon transport preparation failed: " + "; ".join(errors))
    if any(hits != gathered[0][1] for _, hits in gathered):
        raise ValueError(
            "Optimizer transport cache state must agree on every WORLD rank"
        )
    for selected, (_, axes, cap) in zip(resolved, union, strict=True):
        build_transport_mesh(selected, axes, max_ctas=cap or None)


def build_transport_mesh(
    storage_mesh: DeviceMesh,
    mesh_axis_names: tuple[str, ...],
    *,
    max_ctas: int | None = None,
) -> DeviceMesh:
    """Flatten storage axes into a native, cached optimizer transport mesh.

    First creation is collective over the default process group: all ranks must
    request the same axes and cap in the same order. Subsequent calls reuse the
    root mesh's cached communicator, including across optimizer instances.
    The returned 1D mesh uses the joined-axis name and logical storage order.

    ``max_ctas`` must be positive when supplied and requires NCCL. It creates a
    resource-independent capped communicator without modifying storage groups.
    """
    transport_mesh = storage_mesh[
        mesh_axis_names[0] if len(mesh_axis_names) == 1 else mesh_axis_names
    ]
    axis_name = "_".join(mesh_axis_names)
    if max_ctas is not None and max_ctas <= 0:
        raise ValueError("max_ctas must be positive when supplied")
    backend = dist.get_backend(transport_mesh.get_group(0))
    if max_ctas is not None and backend != dist.Backend.NCCL:
        raise ValueError(f"max_ctas requires NCCL, got {backend!r}")
    flattened = _flatten_transport_mesh(transport_mesh, max_ctas, str(backend))
    if max_ctas is None:
        return flattened
    # Rename only the logical axis, without constructing another communicator.
    # The list-group form preserves logical mesh order independently of PG order.
    return DeviceMesh.from_group(
        [flattened.get_group()],
        storage_mesh.device_type,
        mesh=flattened.mesh,
        mesh_dim_names=(axis_name,),
    )


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
    """Shard complete contiguous blocks without changing the tensor shape.

    The selected tensor dimension is partitioned into contiguous blocks of
    sizes from the repeating sequence ``block_sizes``, sharded using the same
    contiguous partitioning as ``Shard``. A block is never split between
    participants. ``BlockShard`` describes only distribution; it does not
    reshape or reinterpret the tensor.

    ``block_sizes`` must be a nonempty tuple of positive integers. Use ``(R,)``
    for uniform blocks of size R. For example, ``(128, 64)`` partitions 384 rows
    into four independently shardable blocks of 128, 64, 128, and 64 rows. The
    dimension must contain whole repetitions of this sequence. DistMuon treats
    each block as one independent matrix.
    """

    dim: int
    block_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if isinstance(self.dim, bool) or not isinstance(self.dim, int):
            raise ValueError("BlockShard.dim must be an integer")
        if (
            not isinstance(self.block_sizes, tuple)
            or not self.block_sizes
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in self.block_sizes
            )
        ):
            raise ValueError(
                "BlockShard.block_sizes must be a nonempty tuple of positive integers"
            )

    def num_blocks(self, dim_size: int) -> int:
        """Return the block count for a dimension containing whole repetitions."""
        period = sum(self.block_sizes)
        num_periods, remainder = divmod(dim_size, period)
        if remainder:
            raise ValueError(
                f"BlockShard dimension size {dim_size} must be divisible by "
                f"block-size period {period}"
            )
        return num_periods * len(self.block_sizes)

    def block_start(self, block_index: int) -> int:
        """Return a block boundary, including the final dimension endpoint."""
        num_periods, offset = divmod(block_index, len(self.block_sizes))
        return num_periods * sum(self.block_sizes) + sum(self.block_sizes[:offset])

    def block_index(self, dim_offset: int) -> int:
        """Return the block index at a dimension offset on a block boundary."""
        if dim_offset < 0:
            raise ValueError("BlockShard dimension offsets must be non-negative")
        num_periods, period_offset = divmod(dim_offset, sum(self.block_sizes))
        block_offset = 0
        for block_index, block_size in enumerate(self.block_sizes):
            if period_offset == block_offset:
                return num_periods * len(self.block_sizes) + block_index
            block_offset += block_size
        raise ValueError(
            f"BlockShard dimension offset {dim_offset} must be a block boundary"
        )


_ComputeSharding = Owned | Replicate | Shard | BlockShard

_KeyT = TypeVar("_KeyT")
_ValueT = TypeVar("_ValueT")


class _FrozenConfigMapping(Mapping[_KeyT, _ValueT]):
    """Small immutable mapping that remains safe to copy with configs.

    ``ComputeLayout`` normalizes both of its mapping fields into this type:
    ``shardings_by_mesh_axis`` keyed by mesh axis name, and
    ``shard_order_by_tensor_dim`` keyed by tensor dimension. Both must stay
    hashable and must survive a deep copy of the surrounding config without
    becoming a mutable alias.
    """

    __slots__ = ("_items",)

    def __init__(self, items: Mapping[_KeyT, _ValueT]) -> None:
        self._items = tuple(items.items())

    def __getitem__(self, key: _KeyT) -> _ValueT:
        for candidate_key, value in self._items:
            if candidate_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[_KeyT]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return hash(self._items)

    def __repr__(self) -> str:
        return repr(dict(self._items))

    def __deepcopy__(
        self, memo: dict[int, Any]
    ) -> _FrozenConfigMapping[_KeyT, _ValueT]:
        return self


_ShardingsByMeshAxis = _FrozenConfigMapping[str, _ComputeSharding]
_ShardOrderByTensorDim = _FrozenConfigMapping[int, tuple[str, ...]]

_DEFAULT_SHARD_ORDER = _ShardOrderByTensorDim({})


@dataclass(frozen=True, slots=True)
class ComputeLayout:
    """Describe temporary compute shardings on named storage-mesh axes.

    These shardings apply before any optimizer-local tensor view.

    Each named axis uses one of four compute shardings: ``Owned`` assigns the
    complete subgroup-local logical tensor to one dynamically selected owner rank,
    ``Replicate`` assigns it to every rank, ``Shard`` partitions one tensor
    dimension, and ``BlockShard`` partitions complete blocks without changing
    the tensor shape.
    Multiple ``Owned`` shardings select one owner rank from their joint Cartesian
    group.
    Omitted axes preserve their storage placement. Extra declarations may target
    mesh variants not used by a particular parameter, but at least one declaration
    must apply when the layout is resolved.

    ``shard_order_by_tensor_dim`` maps a tensor dimension to the mesh axes that
    shard it, outermost first: the leading axis partitions the whole dimension
    and every following axis partitions its predecessor's shard. Declaring an
    order is only necessary when it differs from the default, which applies the
    axes in storage-mesh order. Every axis in one order must declare the same
    sharding type: either ``Shard`` on that tensor dimension or ``BlockShard``
    with the same tensor dimension and block-size sequence.

    ``compute_dtype`` is the dtype the optimizer computes in for parameters
    that are redistributed; ``None`` keeps the storage dtype.

    Examples:
        Shard a logical batch of matrices over EP first, then split each
        EP-local batch over ``edp_shard`` even though ``edp_shard`` precedes EP in the storage
        mesh::

            ComputeLayout(
                shardings_by_mesh_axis={
                    "edp_shard": Shard(0),
                    "ep": Shard(0),
                },
                shard_order_by_tensor_dim={0: ("ep", "edp_shard")},
            )

        Preserve TP head ownership, then distribute complete blocks over DP::

            ComputeLayout(
                shardings_by_mesh_axis={
                    "dp_shard": BlockShard(dim=0, block_sizes=(head_dim,)),
                    "tp": BlockShard(dim=0, block_sizes=(head_dim,)),
                },
                shard_order_by_tensor_dim={0: ("tp", "dp_shard")},
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
                    "dp_shard": BlockShard(dim=0, block_sizes=(4,)),
                }
            )
    """

    shardings_by_mesh_axis: Mapping[str, _ComputeSharding]
    shard_order_by_tensor_dim: Mapping[int, tuple[str, ...]] = _DEFAULT_SHARD_ORDER
    compute_dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        if self.compute_dtype is not None and not isinstance(
            self.compute_dtype, torch.dtype
        ):
            raise ValueError("compute_dtype must be a torch.dtype or None")
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
            _ShardingsByMeshAxis(normalized_shardings_by_mesh_axis),
        )
        object.__setattr__(
            self,
            "shard_order_by_tensor_dim",
            _ShardOrderByTensorDim(
                self._validated_shard_order(normalized_shardings_by_mesh_axis)
            ),
        )

    def _validated_shard_order(
        self,
        shardings_by_mesh_axis: Mapping[str, _ComputeSharding],
    ) -> dict[int, tuple[str, ...]]:
        validated_shard_order: dict[int, tuple[str, ...]] = {}
        for tensor_dim, axis_names in dict(self.shard_order_by_tensor_dim).items():
            if (
                isinstance(tensor_dim, bool)
                or not isinstance(tensor_dim, int)
                or tensor_dim < 0
            ):
                raise ValueError(
                    "ComputeLayout.shard_order_by_tensor_dim keys must be "
                    f"non-negative tensor dimensions; got {tensor_dim!r}"
                )
            if isinstance(axis_names, str) or not isinstance(axis_names, Sequence):
                raise ValueError(
                    "ComputeLayout.shard_order_by_tensor_dim values must be "
                    f"sequences of mesh axis names; got {axis_names!r}"
                )
            ordered_axis_names = tuple(axis_names)
            if len(ordered_axis_names) < 2:
                raise ValueError(
                    "ComputeLayout.shard_order_by_tensor_dim must order at least "
                    f"two mesh axes; tensor dimension {tensor_dim} lists "
                    f"{list(ordered_axis_names)}"
                )
            if len(set(ordered_axis_names)) != len(ordered_axis_names):
                raise ValueError(
                    "ComputeLayout.shard_order_by_tensor_dim must not repeat a "
                    f"mesh axis; tensor dimension {tensor_dim} lists "
                    f"{list(ordered_axis_names)}"
                )
            ordered_shardings = []
            for axis_name in ordered_axis_names:
                sharding = shardings_by_mesh_axis.get(axis_name)
                if sharding is None:
                    raise ValueError(
                        "ComputeLayout.shard_order_by_tensor_dim names mesh axis "
                        f"{axis_name!r}, which shardings_by_mesh_axis does not "
                        "declare"
                    )
                if (
                    type(sharding) is not Shard and type(sharding) is not BlockShard
                ) or sharding.dim != tensor_dim:
                    raise ValueError(
                        "ComputeLayout.shard_order_by_tensor_dim requires "
                        f"Shard({tensor_dim}) or BlockShard({tensor_dim}, ...) "
                        f"on mesh axis {axis_name!r}; got {sharding!r}"
                    )
                ordered_shardings.append(sharding)
            first_sharding = ordered_shardings[0]
            if any(
                type(sharding) is not type(first_sharding)
                for sharding in ordered_shardings
            ):
                raise ValueError(
                    "ComputeLayout.shard_order_by_tensor_dim cannot mix Shard "
                    "and BlockShard"
                )
            if type(first_sharding) is BlockShard and any(
                sharding != first_sharding for sharding in ordered_shardings
            ):
                raise ValueError(
                    "ComputeLayout.shard_order_by_tensor_dim requires equal "
                    "BlockShard block-size sequences"
                )
            validated_shard_order[tensor_dim] = ordered_axis_names
        return dict(sorted(validated_shard_order.items()))

    def to_dict(self) -> dict:
        """Serialize for JSON logging."""
        return {"repr": repr(self)}


@dataclass(frozen=True, slots=True)
class BucketConfig:
    """Ordered optimizer-work bucket selected by canonical FQN patterns.

    ``mesh`` may override the process groups used for redistribution. It must
    have the same ranks in the same order as the selected storage axes. For
    one transport axis, its name must match and a storage-shaped mesh is also
    accepted. For multiple axes, supply a 1D mesh named by joining the axis
    names, for example ``dp_shard_tp``, as returned by ``build_transport_mesh``.
    ``None`` lets DistMuon prepare the WORLD-wide union once per constructor.
    All WORLD ranks must build the same sequence of automatically configured
    DistMuon instances and provide local views of every requested transport.
    For unequal construction sequences, precreate every bucket's ``mesh``
    collectively and pass only explicit meshes.
    """

    patterns: tuple[str, ...]
    name: str = ""
    mesh: DeviceMesh | None = None

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
