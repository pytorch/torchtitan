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

from torch.distributed.tensor import Placement

from torchtitan.protocols.types import MeshAxisName


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
        out_dst_shardings: Desired output placement after redistribution.
            e.g. ``{TP: Shard(1)}`` for reduce-scatter to sequence-parallel.
            ``None`` means no output redistribution.
        local_input_grad_placements: Per-input ``NamedPlacement`` declaring
            the placement of the **input gradient** when the plain input is
            wrapped via ``DTensor.from_local`` at the input boundary. Keyed
            by ``forward()`` arg name. Mirrors
            ``ColwiseParallelWithGradPlacement(local_input_grad_placements=...)``.
            Only effective for inputs that arrive as plain tensors and have
            an entry in ``in_src_shardings``.
            e.g. ``{"x": {TP: Partial()}}`` to keep d_x as Partial in
            backward instead of all-reducing to Replicate.
            ``None`` means default ``from_local`` grad behavior.
        local_output_grad_placements: ``NamedPlacement`` declaring the
            placement of the **output gradient** when the DTensor output is
            unwrapped via ``to_local(grad_placements=...)`` at the output
            boundary. When set, the module's effective return type becomes
            a local tensor (the wrapper calls ``to_local`` after any
            ``out_dst_shardings`` redistribute). Mirrors
            ``NoParallel(local_output_grad_placements=...)``.
            e.g. ``{TP: Partial()}`` to receive the upstream local d_output
            as Partial on TP in backward.
            ``None`` leaves the output as DTensor.
        local_map: If set, wraps forward with ``local_map()``.

    TODO: add ``out_src_shardings`` to declare the output's source placement
    when integrating with spmd_type (erased types), which requires both src
    and dst for every redistribute.
    """

    state_shardings: dict[str, NamedPlacement] = field(default_factory=dict)
    in_src_shardings: dict[str, NamedPlacement] | None = None
    in_dst_shardings: dict[str, NamedPlacement] | None = None
    out_dst_shardings: NamedPlacement | None = None
    local_input_grad_placements: dict[str, NamedPlacement] | None = None
    local_output_grad_placements: NamedPlacement | None = None
    local_map: LocalMapConfig | None = None

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
