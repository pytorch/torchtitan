# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding types for config-based SPMD parallelization.

``ShardingConfig`` is set on ``Module.Config`` by ``set_sharding_config()``
and read by ``Module.parallelize(parallel_dims)``. All placements use
``NamedPlacement`` (dict keyed by mesh axis name) so they are self-documenting
and support multi-dimensional meshes.
"""

from dataclasses import dataclass, field

import spmd_types as spmd

from torchtitan.protocols.types import MeshAxisName


# Per-axis SPMD type, keyed by mesh axis name. ``MeshAxisName`` is a StrEnum,
# so these keys can be passed directly to spmd_types APIs that accept strings.
NamedPlacement = dict[MeshAxisName, spmd.PerMeshAxisSpmdType]


def _mesh_axis_name(axis_name: object) -> str:
    return axis_name.value if hasattr(axis_name, "value") else str(axis_name)


@dataclass(kw_only=True, slots=True)
class LocalSpmdConfig:
    """Spec for modules computing on local tensors.

    Wraps forward with local SPMD typechecking. Input placements come from
    ``ShardingConfig.in_dst_shardings`` and output placements from
    ``ShardingConfig.out_src_shardings``.
    """

    def to_dict(self) -> dict:
        return {"repr": repr(self)}


@dataclass(kw_only=True, slots=True)
class ShardingConfig:
    """Declarative sharding for a Module's states and activations.

    All placements use ``NamedPlacement`` keyed by mesh axis name.

    Completely dtype-agnostic at this moment — quantization (Float8/MXFP8) is
    orthogonal.

    Redistribution is expressed as a (source, destination) pair: src declares
    what the tensor's placement is entering the boundary, dst declares the
    desired placement after redistribution.

    Attributes:
        state_shardings: Parameter/buffer placements. Outer dict keys are
            param/buffer names, e.g. ``{"weight": {TP: spmd.S(0)}}``.
        state_tp_ir: Local parameter names to convert from I@tp at rest to R@tp
            during forward compute. Temporary SPMD escape hatch for replicated
            parameters that need TP gradient reduction semantics.
        in_src_shardings: Source placements of inputs, keyed by ``forward()``
            arg name. Declares what the input's placement is before any
            redistribution.
        in_dst_shardings: Desired input placements after redistribution,
            keyed by ``forward()`` arg name.
            ``None`` means no input redistribution.
        out_src_shardings: Source placement of outputs before redistribution.
            Required for ``spmd.redistribute``.
        out_dst_shardings: Desired output placement after redistribution.
            ``None`` means no output redistribution.
        local_spmd: If set, wraps forward for local SPMD typechecking.
    """

    state_shardings: dict[str, NamedPlacement] = field(default_factory=dict)
    state_tp_ir: set[str] = field(default_factory=set)
    in_src_shardings: dict[str, NamedPlacement] | None = None
    in_dst_shardings: dict[str, NamedPlacement] | None = None
    out_src_shardings: NamedPlacement | tuple[NamedPlacement, ...] | None = None
    out_dst_shardings: NamedPlacement | None = None
    local_spmd: LocalSpmdConfig | None = None

    def axes(self) -> set[str]:
        """Return mesh axes referenced by this sharding config."""
        def add_axes(named: NamedPlacement | None) -> None:
            if named is None:
                return
            axes.update(_mesh_axis_name(axis_name) for axis_name in named)

        axes: set[str] = set()
        for named in self.state_shardings.values():
            add_axes(named)
        for shardings in (self.in_src_shardings, self.in_dst_shardings):
            if shardings is not None:
                for named in shardings.values():
                    add_axes(named)
        if isinstance(self.out_src_shardings, tuple):
            for named in self.out_src_shardings:
                add_axes(named)
        else:
            add_axes(self.out_src_shardings)
        add_axes(self.out_dst_shardings)
        return axes

    def to_dict(self) -> dict:
        """Serialize for JSON logging. Placements become repr strings."""
        return {"repr": repr(self)}


@dataclass(kw_only=True, slots=True)
class SpmdInputConfig:
    """Model-owned trainer input annotations for the SPMD path."""

    inputs: NamedPlacement | None = None
    labels: NamedPlacement | None = None
    extra_inputs: dict[str, NamedPlacement] = field(default_factory=dict)
    extra_kwargs: dict[str, NamedPlacement] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"repr": repr(self)}
