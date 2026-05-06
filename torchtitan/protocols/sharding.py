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
from typing import Any, TYPE_CHECKING

from torch.distributed.tensor import Partial, Placement, Replicate, Shard

from torchtitan.protocols.types import MeshAxisName

if TYPE_CHECKING:
    from torchtitan.distributed.parallel_dims import ParallelDims


# Placement per mesh axis, keyed by MeshAxisName.
# Example: {MeshAxisName.TP: Shard(0), MeshAxisName.DP_SHARD: Replicate()}
# Every sharding_config must declare a placement for every mesh axis it will be applied
# against; ``resolve_placements`` errors otherwise.
#
# Shard order: we implicitly assume the trivial outer -> inner order matching
# the mesh axis order. The only non-trivial case is FSDP + TP both sharding on
# tensor dim 0, but it doesn't need to be annotated today.
# TODO: integrate with global spmd types (e.g., ``TP: V`` + ``PartitionSpec``
# carrying explicit shard-order info) once that lands.
NamedPlacement = dict[MeshAxisName, Placement]

# spmd_types type per mesh axis, keyed by MeshAxisName.
# Values are spmd_types types (R, I, V, P, S(dim)).
NamedSpmdType = dict[MeshAxisName, Any]


@dataclass(kw_only=True, slots=True)
class LocalMapConfig:
    """Spec for modules computing on local tensors.

    Wraps forward with ``local_map()``: DTensor -> local before forward,
    local -> DTensor after forward.

    Placements are ``NamedPlacement`` (keyed by mesh axis name). At
    parallelize time they are resolved to positional tuples matching the
    runtime mesh's axis order.

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
class LocalSpmdConfig:
    """Spec for modules that switch from global to local SPMD typechecking.

    At the input boundary, asserts types and optionally redistributes
    (e.g. all-gather k/v on CP). Then switches to local-only typechecking.
    At the output boundary, re-annotates with global types.

    Attributes:
        in_types: Per-input NamedSpmdTypes (positional, e.g. q, k, v).
        in_dst_types: Optional per-input destination types. If set,
            redistribute from in_types[i] to in_dst_types[i] at entry.
        out_types: Per-output NamedSpmdTypes.
        in_partition_specs: Optional per-input partition spec templates.
        out_partition_specs: Same format, for output re-annotation.
    """

    in_types: tuple[NamedSpmdType, ...]
    in_dst_types: tuple[NamedSpmdType, ...] | None = None
    out_types: tuple[NamedSpmdType, ...]
    in_partition_specs: tuple[tuple, ...] | None = None
    out_partition_specs: tuple[tuple, ...] | None = None

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
        out_src_shardings: Source placement of the output before redistribution.
            Declares what the forward naturally produces. Required for
            spmd_types (type erasure). DTensor path ignores it.
            e.g. ``{TP: Partial()}`` for rowwise linear.
        out_dst_shardings: Desired output placement after redistribution.
            e.g. ``{TP: Shard(1)}`` for reduce-scatter to sequence-parallel.
            ``None`` means no output redistribution.
        local_input_grad_placements: Per-input gradient placement.
            Disambiguates ``Replicate()`` → ``R`` vs ``I`` for spmd_types.
        local_output_grad_placements: Output gradient placement.
            Same disambiguation for outputs.
        local_map: If set, wraps forward with ``local_map()``.
        local_spmd: If set, wraps forward with spmd_types local
            typechecking boundary. Mutually exclusive with ``local_map``.
    """

    state_shardings: dict[str, NamedPlacement] = field(default_factory=dict)
    in_src_shardings: dict[str, NamedPlacement] | None = None
    in_dst_shardings: dict[str, NamedPlacement] | None = None
    out_src_shardings: NamedPlacement | None = None
    out_dst_shardings: NamedPlacement | None = None
    local_input_grad_placements: dict[str, NamedPlacement] | None = None
    local_output_grad_placements: NamedPlacement | None = None
    local_map: LocalMapConfig | None = None
    local_spmd: LocalSpmdConfig | None = None

    def to_dict(self) -> dict:
        """Serialize for JSON logging. Placements become repr strings."""
        return {"repr": repr(self)}


def resolve_placements(
    named: NamedPlacement,
    mesh_axis_names: tuple[str, ...],
) -> tuple[Placement, ...]:
    """Resolve NamedPlacement against a mesh in axis order.

    Every sharding_config must explicitly declare a placement for every mesh axis
    it will be applied against. Missing declarations raise ``ValueError``;
    extra declarations (axes not in the mesh) are ignored.
    """
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


# ---------------------------------------------------------------------------
# spmd_types resolution
# ---------------------------------------------------------------------------


def placement_to_spmd_type(
    placement: Placement,
    grad_placement: Placement | None = None,
) -> Any:
    """Convert a DTensor Placement to an spmd_types type.

    Shard(dim) → S(dim), Partial() → P, Replicate() → I by default.
    Replicate() with Partial() grad → R (gradient needs reduction).
    """
    from spmd_types import I, P, R, S

    if isinstance(placement, Shard):
        return S(placement.dim)
    if isinstance(placement, Partial):
        return P
    if grad_placement is not None and isinstance(grad_placement, Partial):
        return R
    return I


def resolve_spmd(
    named: NamedPlacement | None,
    parallel_dims: "ParallelDims",
    grad_named: NamedPlacement | None = None,
) -> dict | None:
    """Resolve NamedPlacement to {MeshAxis: spmd_type}. Skips disabled axes."""
    if named is None:
        return None
    result = {}
    for axis_name, placement in named.items():
        axis = parallel_dims.get_spmd_axis(axis_name.value)
        if axis.size() <= 1:
            continue
        grad_p = grad_named.get(axis_name) if grad_named else None
        result[axis] = placement_to_spmd_type(placement, grad_p)
    return result


def resolve_spmd_types(
    named: NamedSpmdType,
    parallel_dims: "ParallelDims",
) -> dict:
    """Resolve NamedSpmdType to {MeshAxis: spmd_type}. Skips disabled axes."""
    return {
        parallel_dims.get_spmd_axis(name.value): spmd_type
        for name, spmd_type in named.items()
        if parallel_dims.get_spmd_axis(name.value).size() > 1
    }


def resolve_partition_spec(
    template: tuple,
    parallel_dims: "ParallelDims",
) -> Any:
    """Resolve a partition spec template to a real PartitionSpec.

    Template entries: None, a MeshAxisName, or a tuple of MeshAxisNames.
    """
    from spmd_types.types import PartitionSpec
    from torch.utils._pytree import tree_map_only

    resolved = tree_map_only(
        MeshAxisName,
        lambda name: parallel_dims.get_spmd_axis(name.value),
        template,
    )
    return PartitionSpec(*resolved)
