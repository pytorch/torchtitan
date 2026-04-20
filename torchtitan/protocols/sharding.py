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
# Example: {MeshDimName.TP: Shard(0), MeshDimName.DP: Replicate()}
# Unspecified mesh dims default to Replicate() at resolve time.
NamedPlacement = dict[MeshDimName, Placement]


class Unconstrained(Placement):
    """Preserve existing placement — no redistribution."""

    def __init__(self) -> None:
        super().__init__()


@dataclass(kw_only=True, slots=True)
class LocalMapSpec:
    """Spec for modules computing on local tensors.

    Wraps forward with ``local_map()``: DTensor -> local before forward,
    local -> DTensor after forward.

    Attributes:
        in_placements: Per-input placements (positional: q, k, v).
        out_placements: Per-output placements.
        in_grad_placements: Per-input gradient placements.
    """

    in_placements: tuple[tuple[Placement, ...], ...]
    out_placements: tuple[tuple[Placement, ...], ...]
    in_grad_placements: tuple[tuple[Placement, ...], ...]

    def to_dict(self) -> dict:
        return {"repr": repr(self)}


@dataclass(kw_only=True, slots=True)
class ShardingSpec:
    """Declarative sharding for a Module's states and activations.

    All placements use ``NamedPlacement`` (``dict[MeshDimName, Placement]``)
    keyed by mesh dim names.  At ``parallelize()`` time, NamedPlacements
    are resolved to ``tuple[Placement, ...]`` in mesh dim order.

    Completely dtype-agnostic at this memoent — quantization (Float8/MXFP8) is
    orthogonal.

    Attributes:
        state_shardings: Parameter/buffer placements for ``distribute_tensor``.
            Outer dict keys are param names.
            e.g. ``{"weight": {TP: Shard(0)}}`` for colwise.
        input_layouts: A workaround to annotate plain tensor inputs as DTensors,
            keyed by ``forward()`` arg name. Used when inputs arrive as
            plain tensors (e.g., from dataloader or FSDP-only path).
            e.g. ``{"x": {TP: Shard(1)}}`` means the plain tensor is a
            local shard on TP dim. ``None`` means no annotation needed.
            Will be unnecessary once full DTensor is adopted.
        in_shardings: Desired input placements after redistribution,
            keyed by ``forward()`` arg name.
            e.g. ``{"x": {TP: Replicate()}}`` for all-gather.
            ``None`` means no input redistribution.
        out_shardings: Desired output placement.
            e.g. ``{TP: Shard(1)}`` for reduce-scatter to sequence-parallel.
            ``None`` means no output redistribution.
        local_map: If set, wraps forward with ``local_map()``.
    """

    state_shardings: dict[str, NamedPlacement] = field(default_factory=dict)
    # TODO: Remove once all inputs flow as DTensors (full DTensor regime).
    input_layouts: dict[str, NamedPlacement] | None = None
    in_shardings: dict[str, NamedPlacement] | None = None
    out_shardings: NamedPlacement | None = None
    local_map: LocalMapSpec | None = None

    def to_dict(self) -> dict:
        """Serialize for JSON logging. Placements become repr strings."""
        return {"repr": repr(self)}


def resolve_placements(
    named: NamedPlacement,
    mesh_dim_names: tuple[str, ...],
) -> tuple[Placement, ...]:
    """Convert NamedPlacement to ``tuple[Placement, ...]`` in mesh dim order.

    Unspecified mesh dims default to ``Unconstrained()``. Each caller decides
    how to resolve ``Unconstrained`` for its context (state distribution
    needs a concrete placement, input redistribution preserves existing).
    """
    return tuple(
        named.get(MeshDimName(dim_name), Unconstrained()) for dim_name in mesh_dim_names
    )
