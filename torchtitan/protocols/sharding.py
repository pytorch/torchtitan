# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding types for config-based parallelization.

``ShardingConfig`` is set on ``Module.Config`` by ``set_sharding_config()``
and read by ``Module.parallelize(parallel_dims)``. All placements use
``SpmdType`` so they are self-documenting and support multi-dimensional
meshes.
"""

from dataclasses import dataclass, field

import spmd_types as spmd
from spmd_types import SpmdType
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Partial, Placement, Replicate, Shard

from torchtitan.distributed.parallel_dims import MeshAxisName, unfold_dp_axis
from torchtitan.distributed.spmd_types import _per_axis_types, spmd_axes


__all__ = [
    "ShardingConfig",
    "resolve_placements",
]


@dataclass(kw_only=True, slots=True)
class ShardingConfig:
    """Declarative sharding for a Module's states and activations.

    All placements use ``SpmdType`` keyed by mesh axis names. At
    ``parallelize()`` time, parameters and buffers are locally sharded and
    annotated, while activation layouts are checked at module boundaries.

    Completely dtype-agnostic at this moment -- quantization (Float8/MXFP8) is
    orthogonal.

    Attributes:
        state_shardings: Parameter/buffer SPMD layouts. Outer dict keys are
            state names.
            e.g. ``{"weight": {TP: Shard(0)}}`` for colwise.
        in_shardings: Expected input placements, keyed by ``forward()``
            argument name.
            e.g. ``{"x": {TP: Shard(1)}}``.
        out_shardings: Expected output placements. Accepts a single
            ``SpmdType`` or a pytree matching the output structure.
        local_spmd: If true, wraps forward with ``spmd.no_typecheck()`` using
            ``in_shardings`` and ``out_shardings`` as its boundary types.
    """

    state_shardings: dict[str, SpmdType] = field(default_factory=dict)
    in_shardings: dict[str, SpmdType] | None = None
    out_shardings: SpmdType | tuple[SpmdType | None, ...] | None = None
    local_spmd: bool = False

    def to_dict(self) -> dict:
        """Serialize for JSON logging. Placements become repr strings."""
        return {"repr": repr(self)}


def resolve_placements(
    layout: SpmdType,
    mesh: DeviceMesh,
) -> tuple[Placement, ...]:
    """Resolve an SPMD type against a mesh in axis order.

    Every sharding_config must explicitly declare a placement for every mesh axis
    it will be applied against. Missing declarations raise ``ValueError``;
    extra declarations (axes not in the mesh) are ignored.

    ``Shard(d)`` or ``Partial`` on a size-1 mesh axis is normalized to
    ``Replicate()`` -- all three are operationally identical on a 1-rank axis
    (no data is split, and a sum over a single rank is the identity), but
    DTensor's op rules (placement-equality, view/reshape strict mode, ...)
    treat them as distinct and reject ``Shard``/``Partial`` in places where
    ``Replicate`` would work.
    """
    # TODO(fegin): remove the size-1 ``Shard(d)``/``Partial`` to ``Replicate()``
    # conversion once FlexShard replaces ``fully_shard``.
    assert mesh.mesh_dim_names is not None, "DeviceMesh must have named axes"
    concrete_axis_types = {}
    for axis_name, axis_type in _per_axis_types(layout).items():
        for concrete_axis_name in unfold_dp_axis(axis_name):
            concrete_axis_types[concrete_axis_name] = axis_type

    result = []
    for i, axis_name in enumerate(mesh.mesh_dim_names):
        key = MeshAxisName(axis_name)
        if key not in concrete_axis_types:
            raise ValueError(
                f"ShardingConfig does not declare a placement for mesh axis "
                f"{axis_name!r}. Declared: "
                f"{sorted(k.value for k in spmd_axes(layout))}; "
                f"required: {list(mesh.mesh_dim_names)}."
            )
        p = spmd.spmd_type_to_dtensor_placement(concrete_axis_types[key])
        if isinstance(p, (Shard, Partial)) and mesh.size(i) == 1:
            p = Replicate()
        result.append(p)
    return tuple(result)
