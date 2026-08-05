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

import spmd_types as spmd
import torch

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Partial, Placement, Replicate, Shard

from torchtitan.config import Configurable, Function
from torchtitan.distributed.parallel_dims import (
    MeshAxisName,
    SpmdLayout,
)

from torchtitan.distributed.spmd_types import (
    spmd_layout_to_dtensor_placements,
    spmd_mesh_size,
)


__all__ = [
    "LocalMapConfig",
    "PerAxisRedistribution",
    "ShardingConfig",
    "SpmdLayout",
    "resolve_placements",
]

# Shard order: we implicitly assume the trivial outer -> inner order matching
# the mesh axis order. The only non-trivial case is FSDP + TP both sharding on
# tensor dim 0, but it doesn't need to be annotated today.
# TODO: integrate with global spmd types (e.g., ``TP: V`` + ``PartitionSpec``
# carrying explicit shard-order info) once that lands.


@dataclass(kw_only=True, slots=True)
class LocalMapConfig:
    """Spec for modules computing on local tensors.

    Wraps forward with ``local_map()``: DTensor -> local before forward,
    local -> DTensor after forward.

    Output placements come from ``ShardingConfig.out_src_shardings``.
    ``LocalMapConfig`` only carries ``in_grad_placements`` since input
    placements are inferred at the local_map boundary. Set it to ``None`` to
    omit the local_map argument when input gradients are irrelevant.

    Attributes:
        in_grad_placements: Per-input-gradient SpmdLayouts (positional,
            ordered by ``forward`` args). Use ``None`` to omit
            input-gradient placements or for non-tensor args.
    """

    in_grad_placements: tuple[SpmdLayout | None, ...] | None

    def to_dict(self) -> dict:
        return {"repr": repr(self)}


class PerAxisRedistribution(Function):
    """Apply one ``spmd.redistribute`` over a single mesh axis.

    The config names the mesh axis to redistribute and the per-axis source and
    destination SPMD types. Optional dtype fields are forwarded to
    ``spmd.redistribute`` for forward and backward collectives.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        axis: MeshAxisName
        src: spmd.PerMeshAxisSpmdType
        dst: spmd.PerMeshAxisSpmdType
        fwd_op_dtype: torch.dtype | None = None
        fwd_out_dtype: torch.dtype | None = None
        bwd_op_dtype: torch.dtype | None = None
        bwd_out_dtype: torch.dtype | None = None

    def __init__(self, config: Config):
        self.config = config

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        axis = self.config.axis.value
        if self.config.src == self.config.dst or spmd_mesh_size(axis) == 1:
            return x
        backward_options = {"op_dtype": self.config.bwd_op_dtype or x.dtype}
        if self.config.bwd_out_dtype is not None:
            backward_options["out_dtype"] = self.config.bwd_out_dtype
        return spmd.redistribute(
            x,
            axis,
            src=self.config.src,
            dst=self.config.dst,
            op_dtype=self.config.fwd_op_dtype,
            out_dtype=self.config.fwd_out_dtype,
            backward_options=backward_options,
        )


@dataclass(kw_only=True, slots=True)
class ShardingConfig(Configurable.Config):
    """Declarative sharding for a Module's states and activations.

    All placements use ``SpmdLayout`` keyed by mesh axis names.  At
    ``parallelize()`` time, SpmdLayouts are resolved to
    ``tuple[Placement, ...]`` in mesh axis order.

    Completely dtype-agnostic at this moment — quantization (Float8/MXFP8) is
    orthogonal.

    Redistribution is expressed as a ``PerAxisRedistribution``: ``src``
    declares what the tensor's per-axis SPMD type is entering the boundary,
    and ``dst`` declares the desired type after redistribution.

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
        in_redist: Per-input redistribution specs, keyed by ``forward()``
            arg name. Each spec changes one mesh axis from ``src`` to ``dst``;
            the same input name must be present in ``in_src_shardings`` so the
            source layout and mesh are explicit.
        out_src_shardings: Source placement of the forward's output as a
            DTensor. When ``local_map`` is set this also tells ``local_map``
            what to wrap the local output back to. ``None`` means "infer from
            the output" (it's already a DTensor at the right placement, or
            there's no local_map to drive).
            e.g. ``{TP: Partial()}`` for the MoE wrapper.
        out_redist: Output redistribution spec. Changes one mesh axis from
            ``src`` to ``dst`` after forward; requires single-tensor
            ``out_src_shardings`` so the source layout and mesh are explicit.
        local_map: If set, wraps forward with ``local_map()``. Input and
            output placements are inferred from runtime inputs and
            ``out_src_shardings``; ``LocalMapConfig`` only carries
            ``in_grad_placements``.
    """

    state_shardings: dict[str, SpmdLayout] = field(default_factory=dict)
    in_src_shardings: dict[str, SpmdLayout] | None = None
    in_redist: dict[str, PerAxisRedistribution.Config] | None = None
    out_src_shardings: SpmdLayout | None = None
    out_redist: PerAxisRedistribution.Config | None = None
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
    placements = spmd_layout_to_dtensor_placements(layout)
    result = []
    for i, axis_name in enumerate(mesh.mesh_dim_names):
        key = MeshAxisName(axis_name)
        if key not in placements:
            raise ValueError(
                f"ShardingConfig does not declare a placement for mesh axis "
                f"{axis_name!r}. Declared: "
                f"{sorted(k.value for k in layout.local_type)}; "
                f"required: {list(mesh.mesh_dim_names)}."
            )
        p = placements[key]
        if isinstance(p, (Shard, Partial)) and mesh.size(i) == 1:
            p = Replicate()
        result.append(p)
    return tuple(result)
