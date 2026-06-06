# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding types for config-based parallelization.

``ShardingConfig`` is set on ``Module.Config`` by ``set_sharding_config()``
and read by ``Module.parallelize(parallel_dims)``.  All placements use
``SpmdLayout`` so they are self-documenting and support multi-dimensional
meshes.
"""

from dataclasses import dataclass, field

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Placement, Replicate, Shard

from torchtitan.distributed.spmd_types import spmd_layout_to_dtensor_placements
from torchtitan.protocols.types import MeshAxisName, SpmdLayout


__all__ = [
    "LocalMapConfig",
    "ShardingConfig",
    "SpmdLayout",
    "resolve_placements",
]


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
        in_grad_placements: Per-input-gradient SpmdLayouts (positional,
            ordered by ``forward`` args). Use ``None`` for non-tensor args.
    """

    in_grad_placements: tuple[SpmdLayout | None, ...]

    def to_dict(self) -> dict:
        return {"repr": repr(self)}


@dataclass(kw_only=True, slots=True)
class ShardingConfig:
    """Declarative sharding for a Module's states and activations.

    All placements use ``SpmdLayout`` keyed by mesh axis names.  At
    ``parallelize()`` time, SpmdLayouts are resolved to
    ``tuple[Placement, ...]`` in mesh axis order.

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
        out_src_shardings: Source placement of the forward's output as a
            DTensor. When ``local_map`` is set this also tells ``local_map``
            what to wrap the local output back to. Accepts a single
            ``SpmdLayout`` (single-output case) or a tuple (multi-
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

    state_shardings: dict[str, SpmdLayout] = field(default_factory=dict)
    in_src_shardings: dict[str, SpmdLayout] | None = None
    in_dst_shardings: dict[str, SpmdLayout] | None = None
    out_src_shardings: SpmdLayout | tuple[SpmdLayout, ...] | None = None
    out_dst_shardings: SpmdLayout | None = None
    local_map: LocalMapConfig | None = None

    def to_dict(self) -> dict:
        """Serialize for JSON logging. Placements become repr strings."""
        return {"repr": repr(self)}


def resolve_placements(
    layout: SpmdLayout,
    mesh: DeviceMesh,
) -> tuple[Placement, ...]:
    """Resolve SpmdLayout against a mesh in axis order.

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
    assert mesh.mesh_dim_names is not None, "DeviceMesh must have named axes"
    placements = spmd_layout_to_dtensor_placements(layout)
    result = []
    for i, axis_name in enumerate(mesh.mesh_dim_names):
        key = MeshAxisName(axis_name)
        if key not in placements:
            raise ValueError(
                f"ShardingConfig does not declare a placement for mesh axis "
                f"{axis_name!r}. Declared: "
                f"{sorted(k.value for k in layout.axes())}; "
                f"required: {list(mesh.mesh_dim_names)}."
            )
        p = placements[key]
        if isinstance(p, Shard) and mesh.size(i) == 1:
            p = Replicate()
        result.append(p)
    return tuple(result)
