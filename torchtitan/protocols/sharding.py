# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding types for config-based parallelization.

``ShardingSpec`` is set on ``Module.Config`` by ``set_sharding_spec()``
and read by ``Module.parallelize(mesh)``.  All placements use
``NamedPlacement`` (dict keyed by ``MeshDimName``) so they are
self-documenting and support multi-dim meshes.
"""

from dataclasses import dataclass, field
from enum import Enum

from torch.distributed.tensor import Placement


class StrEnum(str, Enum):
    """str + Enum for Python < 3.11 compatibility."""

    pass


class MeshDimName(StrEnum):
    """Mesh dimension names for parallelism."""

    DP = "dp"
    DP_REPLICATE = "dp_replicate"
    DP_SHARD = "dp_shard"
    FSDP = "fsdp"
    TP = "tp"
    CP = "cp"
    PP = "pp"
    EP = "ep"
    ETP = "etp"
    EFSDP = "efsdp"


# Placement per mesh dim, keyed by MeshDimName.
# Example: {MeshDimName.TP: Shard(0), MeshDimName.DP_SHARD: Replicate()}
# Every spec must declare a placement for every mesh dim it will be applied
# against; ``resolve_placements`` errors otherwise.
NamedPlacement = dict[MeshDimName, Placement]


@dataclass(kw_only=True, slots=True)
class LocalMapSpec:
    """Spec for modules computing on local tensors.

    Wraps forward with ``local_map()``: DTensor -> local before forward,
    local -> DTensor after forward.

    Placements are ``NamedPlacement`` (keyed by mesh dim name). At
    parallelize time they are resolved to positional tuples matching the
    runtime mesh's dim order.

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
class ShardingSpec:
    """Declarative sharding for a Module's states and activations.

    All placements use ``NamedPlacement`` (``dict[MeshDimName, Placement]``)
    keyed by mesh dim names.  At ``parallelize()`` time, NamedPlacements
    are resolved to ``tuple[Placement, ...]`` in mesh dim order.

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
        out_dst_shardings: Desired output placement after redistribution.
            e.g. ``{TP: Shard(1)}`` for reduce-scatter to sequence-parallel.
            ``None`` means no output redistribution.
        local_map: If set, wraps forward with ``local_map()``.

    TODO: add ``out_src_shardings`` to declare the output's source placement
    when integrating with spmd_type (erased types), which requires both src
    and dst for every redistribute.
    """

    state_shardings: dict[str, NamedPlacement] = field(default_factory=dict)
    in_src_shardings: dict[str, NamedPlacement] | None = None
    in_dst_shardings: dict[str, NamedPlacement] | None = None
    out_dst_shardings: NamedPlacement | None = None
    local_map: LocalMapSpec | None = None

    def to_dict(self) -> dict:
        """Serialize for JSON logging. Placements become repr strings."""
        return {"repr": repr(self)}


def resolve_placements(
    named: NamedPlacement,
    mesh_dim_names: tuple[str, ...],
) -> tuple[Placement, ...]:
    """Resolve NamedPlacement against a mesh in dim order.

    Every spec must explicitly declare a placement for every mesh dim it
    will be applied against. Missing declarations raise ``ValueError``;
    extra declarations (dims not in the mesh) are ignored.
    """
    result = []
    for dim_name in mesh_dim_names:
        key = MeshDimName(dim_name)
        if key not in named:
            raise ValueError(
                f"Sharding spec does not declare a placement for mesh dim "
                f"{dim_name!r}. Declared: "
                f"{sorted(k.value for k in named)}; "
                f"required: {list(mesh_dim_names)}."
            )
        result.append(named[key])
    return tuple(result)
