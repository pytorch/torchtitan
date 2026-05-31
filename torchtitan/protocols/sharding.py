# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding types for config-based parallelization.

``ShardingConfig`` is set on ``Module.Config`` by ``set_sharding_config()``
and read by ``Module.parallelize(parallel_dims)``.  All placements use
``NamedPlacement`` (dict keyed by ``MeshAxisName``) so they are
self-documenting and support multi-dimensional meshes.
"""

from dataclasses import dataclass, field

import spmd_types as spmd
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Partial, Placement, Replicate, Shard
from torch.utils._pytree import tree_map

from torchtitan.protocols.types import MeshAxisName


__all__ = [
    "LocalMapConfig",
    "NamedPartitionSpec",
    "NamedPartitionSpecEntry",
    "NamedPlacement",
    "NamedPlacementSpmd",
    "PlacementLike",
    "PlacementSpec",
    "ShardingConfig",
    "SpmdInputConfig",
    "active_spmd_placement",
    "is_placement_like",
    "is_spmd_placement",
    "placement_axes",
    "placement_to_spmd_assert_type",
    "resolve_placements",
]

NamedPlacement = dict[MeshAxisName, Placement]
NamedPlacementSpmd = dict[MeshAxisName, spmd.PerMeshAxisSpmdType]
NamedPartitionSpecEntry = MeshAxisName | tuple[MeshAxisName, ...] | None
NamedPartitionSpec = tuple[NamedPartitionSpecEntry, ...]


@dataclass(frozen=True, kw_only=True, slots=True)
class PlacementSpec:
    """Placement with explicit global partition spec ordering.

    Use this only when multiple mesh axes shard the same tensor dimension.
    ``placement`` carries per-axis runtime placement. ``partition_spec`` only
    disambiguates ordering for type assertions in the local-tensor SPMD path.
    """

    placement: NamedPlacementSpmd
    partition_spec: NamedPartitionSpec


PlacementLike = NamedPlacement | NamedPlacementSpmd | PlacementSpec


def _named_placement(placement: PlacementLike) -> NamedPlacement | NamedPlacementSpmd:
    return placement.placement if isinstance(placement, PlacementSpec) else placement


def is_placement_like(value: object) -> bool:
    return isinstance(value, PlacementSpec) or (
        isinstance(value, dict)
        and bool(value)
        and all(MeshAxisName.has_axis(key) for key in value)
    )


def is_spmd_placement(placement: PlacementLike) -> bool:
    if isinstance(placement, PlacementSpec):
        return True
    return all(
        isinstance(value, spmd.PerMeshAxisSpmdType) for value in placement.values()
    )


def placement_axes(placement: PlacementLike) -> tuple[MeshAxisName, ...]:
    return tuple(_named_placement(placement).keys())


@dataclass(kw_only=True, slots=True)
class LocalMapConfig:
    """Spec for modules computing on local tensors.

    Wraps forward with ``local_map()``: DTensor -> local before forward,
    local -> DTensor after forward.

    Input placements come from ``ShardingConfig.in_dst_shardings``
    (already aligned by ``_redistribute_inputs``); output placements from
    ``ShardingConfig.out_src_shardings``. ``LocalMapConfig`` only carries
    ``in_grad_placements`` since there's no equivalent slot on
    ``ShardingConfig`` today.

    Attributes:
        in_grad_placements: Per-input-gradient NamedPlacements (positional,
            ordered by ``forward`` args).
    """

    in_grad_placements: tuple[PlacementLike | None, ...]

    def to_dict(self) -> dict:
        return {"repr": repr(self)}


@dataclass(kw_only=True, slots=True)
class ShardingConfig:
    """Declarative sharding for a Module's states and activations.

    All placements use ``NamedPlacement`` (``dict[MeshAxisName, Placement]``)
    keyed by mesh axis names.  At ``parallelize()`` time, NamedPlacements
    are resolved to ``tuple[Placement, ...]`` in mesh axis order.

    Completely dtype-agnostic at this moment — quantization (Float8/MXFP8) is
    orthogonal.

    Redistribution is expressed as a (source, destination) pair: src declares
    what the tensor's placement is entering the boundary, dst declares the
    desired placement after redistribution. For DTensor, the src is usually
    implicit in the tensor's ``placements``; declaring it explicitly keeps
    the contract uniform with future erased-type systems that require both
    sides of every redistribute.

    Attributes:
        state_shardings: Parameter/buffer placements for ``distribute_tensor``.
            Outer dict keys are param names.
            e.g. ``{"weight": {TP: Shard(0)}}`` for colwise.
        state_shardings_compute: Parameter/buffer placements used only during
            forward compute when they differ from ``state_shardings``. This is
            for local SPMD type changes such as an ``I@TP`` norm weight that
            must be read as ``R@TP`` while computing.
        in_src_shardings: Source placements of inputs, keyed by ``forward()``
            arg name. Used to annotate plain tensors as DTensors via
            ``DTensor.from_local`` when inputs arrive plain (e.g. from
            dataloader or FSDP-only path). Also declares the src side of
            the input redistribute pair.
            e.g. ``{"x": {TP: Shard(1)}}``.
        in_dst_shardings: Desired input placements after redistribution,
            keyed by ``forward()`` arg name.
            e.g. ``{"x": {TP: Replicate()}}`` for all-gather.
            ``None`` means no input redistribution.
        out_src_shardings: Source placement of the forward's output as a
            DTensor. When ``local_map`` is set this also tells ``local_map``
            what to wrap the local output back to. Accepts a single
            ``NamedPlacement`` (single-output case) or a tuple (multi-
            output case, e.g. attention with ``return_lse=True``). ``None``
            means "infer from the output" (it's already a DTensor at the
            right placement, or there's no local_map to drive).
            e.g. ``{TP: Partial()}`` for the MoE wrapper.
        out_dst_shardings: Desired output placement after redistribution.
            e.g. ``{TP: Shard(1)}`` for reduce-scatter to sequence-parallel.
            ``None`` means no output redistribution.
        local_map: If set, wraps forward with ``local_map()``. Input and
            output placements come from ``in_dst_shardings`` and
            ``out_src_shardings``; ``LocalMapConfig`` only carries
            ``in_grad_placements``.
    """

    state_shardings: dict[str, PlacementLike] = field(default_factory=dict)
    state_shardings_compute: dict[str, PlacementLike] = field(default_factory=dict)
    in_src_shardings: dict[str, PlacementLike] | None = None
    in_dst_shardings: dict[str, PlacementLike] | None = None
    out_src_shardings: PlacementLike | tuple[PlacementLike, ...] | None = None
    out_dst_shardings: PlacementLike | None = None
    local_map: LocalMapConfig | None = None

    def to_dict(self) -> dict:
        """Serialize for JSON logging. Placements become repr strings."""
        return {"repr": repr(self)}


@dataclass(kw_only=True, slots=True)
class SpmdInputConfig:
    """Model-owned trainer input annotations for the local SPMD path."""

    inputs: PlacementLike | None = None
    labels: PlacementLike | None = None
    extra_inputs: dict[str, PlacementLike] = field(default_factory=dict)
    extra_kwargs: dict[str, PlacementLike] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"repr": repr(self)}


def resolve_placements(
    placement: PlacementLike,
    mesh: DeviceMesh,
) -> tuple[Placement, ...]:
    """Resolve NamedPlacement against a mesh in axis order.

    Every sharding_config must explicitly declare a placement for every mesh axis
    it will be applied against. Missing declarations raise ``ValueError``;
    extra declarations (axes not in the mesh) are ignored.

    ``Shard(d)`` on a size-1 mesh axis is normalized to ``Replicate()`` --
    the two are operationally identical on a 1-rank axis, but DTensor's op
    rules (placement-equality, view/reshape strict mode, ...) treat them
    as distinct and reject ``Shard`` in places where ``Replicate`` would
    work.
    """
    # TODO(fegin): remove the ``Shard(d)`` on a size-1 mesh to ``Replicate()``
    # conversion once FlexShard replaces ``fully_shard``.
    named = _named_placement(placement)
    if is_spmd_placement(placement):
        named = _spmd_to_dtensor_placement(placement)
    assert mesh.mesh_dim_names is not None, "DeviceMesh must have named axes"
    result = []
    for i, axis_name in enumerate(mesh.mesh_dim_names):
        key = MeshAxisName(axis_name)
        if key not in named:
            raise ValueError(
                f"ShardingConfig does not declare a placement for mesh axis "
                f"{axis_name!r}. Declared: "
                f"{sorted(k.value for k in named)}; "
                f"required: {list(mesh.mesh_dim_names)}."
            )
        p = named[key]
        if isinstance(p, Shard) and mesh.size(i) == 1:
            p = Replicate()
        assert isinstance(
            p, Placement
        ), f"Expected a DTensor Placement for axis {axis_name!r}, got {p!r}."
        result.append(p)
    return tuple(result)


def _spmd_to_dtensor_placement(placement: PlacementLike) -> NamedPlacement:
    named = _named_placement(placement)
    if not is_spmd_placement(placement):
        raise ValueError(f"Expected an SPMD placement, got {named!r}.")

    result: NamedPlacement = {}
    for axis_name, axis_type in named.items():
        if axis_type == spmd.I or axis_type == spmd.R:
            dtensor_placement: Placement = Replicate()
        elif axis_type == spmd.P:
            dtensor_placement = Partial()
        elif isinstance(axis_type, spmd.Shard):
            dtensor_placement = Shard(axis_type.dim)
        else:
            raise ValueError(
                f"Unsupported SPMD placement for axis {axis_name!r}: {axis_type!r}."
            )
        if axis_name == MeshAxisName.DP:
            # Temporary bridge: local SPMD uses one logical DP axis, while the
            # current full-DTensor storage mesh still spells it as two axes.
            result[MeshAxisName.DP_REPLICATE] = dtensor_placement
            result[MeshAxisName.DP_SHARD] = dtensor_placement
        else:
            result[axis_name] = dtensor_placement
    return result


def active_spmd_placement(placement: PlacementLike) -> NamedPlacementSpmd:
    """Return the SPMD placement restricted to active current-mesh axes."""
    if not is_spmd_placement(placement):
        raise ValueError(
            "SPMD backend requires SPMD placements. "
            f"Got DTensor placement: {_named_placement(placement)!r}."
        )

    named = _named_placement(placement)
    mesh_names = spmd.current_mesh_names()
    if mesh_names is None:
        return {}

    resolved: NamedPlacementSpmd = {}
    for axis_name, value in named.items():
        if axis_name in mesh_names:
            resolved[axis_name] = value
    return resolved


def placement_to_spmd_assert_type(
    placement: PlacementLike,
) -> tuple[NamedPlacementSpmd, spmd.PartitionSpec | None]:
    """Resolve a placement to ``assert_type`` args for local SPMD typechecking."""
    local_type = active_spmd_placement(placement)
    if not isinstance(placement, PlacementSpec):
        return local_type, None

    mesh_names = spmd.current_mesh_names()
    if mesh_names is None:
        return local_type, None

    def active_partition_entry(entry: NamedPartitionSpecEntry):
        filtered = tree_map(
            lambda axis_name: axis_name if axis_name in mesh_names else None,
            entry,
        )
        if filtered is None:
            return None
        if not isinstance(filtered, tuple):
            return filtered

        active_axes = tuple(
            axis_name for axis_name in filtered if axis_name is not None
        )
        if not active_axes:
            return None
        return active_axes[0] if len(active_axes) == 1 else active_axes

    partition_spec = spmd.PartitionSpec(
        *(active_partition_entry(entry) for entry in placement.partition_spec)
    )
    return {
        axis_name: spmd.V if isinstance(value, spmd.Shard) else value
        for axis_name, value in local_type.items()
    }, partition_spec
