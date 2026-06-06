# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SPMD type helpers for the local-tensor parallelization path.

Bridges TorchTitan's ``SpmdLayout`` sharding annotations with ``spmd_types``
runtime APIs (``assert_type``, ``redistribute``, ``PartitionSpec``).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from threading import local
from typing import Any, TYPE_CHECKING

import spmd_types as spmd
import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Partial, Placement, Replicate, Shard

# Avoid circular import: protocols.__init__ imports module.py, which imports us.
if TYPE_CHECKING:
    from torchtitan.protocols.types import MeshAxisName, SpmdLayout

__all__ = [
    "annotate_input_spmd_types",
    "current_mesh",
    "mesh_size",
    "preserve_buffers_spmd",
    "redistribute_spmd_per_axis",
    "set_current_spmd_mesh",
    "set_spmd_backend",
    "spmd_layout_to_assert_type",
    "spmd_layout_to_dtensor_placements",
]


_MESH_TLS = local()
_spmd_backend = "default"


def set_spmd_backend(spmd_backend: str) -> None:
    """Set the backend that controls whether current-mesh context is active."""
    global _spmd_backend
    _spmd_backend = spmd_backend


def _mesh_stack() -> list[DeviceMesh | None]:
    stack = getattr(_MESH_TLS, "mesh_stack", None)
    if stack is None:
        stack = []
        _MESH_TLS.mesh_stack = stack
    return stack


def current_mesh() -> DeviceMesh | None:
    """Return the current runtime mesh, or ``None`` if unset."""
    if _spmd_backend != "spmd_types":
        return None
    stack = _mesh_stack()
    if not stack:
        return None
    return stack[-1]


def mesh_size(axis_name: str) -> int:
    """Return the size of a mesh axis, or 1 if not active."""
    mesh = current_mesh()
    if mesh is None:
        return 1
    names = mesh.mesh_dim_names or ()
    if axis_name not in names:
        return 1
    return mesh.size(names.index(axis_name))


@contextlib.contextmanager
def preserve_buffers_spmd(model: nn.Module) -> Iterator[None]:
    """
    Preserve local SPMD annotations on buffers across reinitialization.

    Previously we could have read types off self.freqs_cis, but after
    https://github.com/pytorch/torchtitan/pull/3458 changed to per-layer cache,
    the types disappear across to_empty() + _init_self_buffers().
    Implement it as a general context for all buffers in case other modules suffer from this.
    """
    if _spmd_backend != "spmd_types":
        yield
        return

    saved = {
        fqn: dict(spmd.get_local_type(buf))
        for fqn, buf in model.named_buffers()
        if spmd.has_local_type(buf)
    }
    try:
        yield
    finally:
        for fqn, buf in model.named_buffers():
            if fqn in saved and not spmd.has_local_type(buf):
                spmd.assert_type(buf, saved[fqn])


@contextlib.contextmanager
def set_current_spmd_mesh(mesh: DeviceMesh | None) -> Iterator[None]:
    """Set TorchTitan and spmd_types current mesh state for one runtime region."""
    if _spmd_backend != "spmd_types":
        yield
        return

    stack = _mesh_stack()
    stack.append(mesh)
    if mesh is None:
        try:
            yield
        finally:
            popped = stack.pop()
            assert popped is mesh
        return

    with spmd.set_current_mesh(mesh):
        try:
            yield
        finally:
            popped = stack.pop()
            assert popped is mesh


def spmd_layout_to_dtensor_placements(
    layout: "SpmdLayout",
) -> dict["MeshAxisName", Placement]:
    """Convert an SPMD layout to DTensor placements keyed by mesh axis name."""
    from torchtitan.protocols.types import MeshAxisName

    result: dict[MeshAxisName, Placement] = {}
    for axis_name, axis_type in layout.shard_types().items():
        if axis_type == spmd.R or axis_type == spmd.I or axis_type == spmd.V:
            dtensor_placement: Placement = Replicate()
        elif axis_type == spmd.P:
            dtensor_placement = Partial()
        elif isinstance(axis_type, spmd.Shard):
            dtensor_placement = Shard(axis_type.dim)
        else:
            raise ValueError(
                f"Unsupported SPMD type for axis {axis_name.value!r}: {axis_type!r}."
            )
        if axis_name == MeshAxisName.DP:
            result[MeshAxisName.DP_REPLICATE] = dtensor_placement
            result[MeshAxisName.DP_SHARD] = dtensor_placement
        else:
            result[axis_name] = dtensor_placement
    return result


def spmd_layout_to_assert_type(
    layout: "SpmdLayout",
) -> tuple[spmd.PerMeshAxisSpmdTypes, spmd.PartitionSpec | None]:
    """Resolve a layout to ``assert_type`` args for local SPMD typechecking."""
    local_type = dict(layout.axis_types)
    if layout.partition_spec is None:
        return local_type, None
    return {
        axis_name: axis_type
        for axis_name, axis_type in local_type.items()
        if not isinstance(axis_type, spmd.Shard)
    }, layout.partition_spec


def annotate_input_spmd_types(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    extra_kwargs: dict[str, Any],
) -> None:
    """Annotate decoder inputs/labels with local SPMD types.

    Hardcodes the standard decoder convention: inputs and positions are
    ``S(0)@DP, S(1)@CP, R@TP``; labels are ``S(0)@DP, S(1)@CP, I@TP``.
    Analogous to ``full_dtensor.parallelize_inputs()`` but for the
    ``spmd_types`` path.
    """
    from torchtitan.protocols.types import MeshAxisName, SpmdLayout

    token_layout = SpmdLayout(
        {
            MeshAxisName.DP: spmd.S(0),
            MeshAxisName.CP: spmd.S(1),
            MeshAxisName.TP: spmd.R,
        }
    )
    label_layout = SpmdLayout(
        {
            MeshAxisName.DP: spmd.S(0),
            MeshAxisName.CP: spmd.S(1),
            MeshAxisName.TP: spmd.I,
        }
    )

    def assert_type(tensor: torch.Tensor, layout: SpmdLayout) -> None:
        local_type, partition_spec = spmd_layout_to_assert_type(layout)
        spmd.assert_type(tensor, local_type, partition_spec=partition_spec)

    assert_type(inputs, token_layout)
    assert_type(labels, label_layout)
    if "positions" in extra_kwargs and isinstance(
        extra_kwargs["positions"], torch.Tensor
    ):
        assert_type(extra_kwargs["positions"], token_layout)


def redistribute_spmd_per_axis(
    x: torch.Tensor,
    src_types: spmd.PerMeshAxisSpmdTypes,
    dst_types: spmd.PerMeshAxisSpmdTypes,
) -> torch.Tensor:
    """Redistribute a local tensor along axes whose SPMD type changes.

    Iterates over *dst_types* and issues a per-axis ``spmd.redistribute``
    for each axis where src and dst differ. Each call is a single collective
    (all-reduce, reduce-scatter, or all-gather) on that axis's process group.

    TODO(pianpwk) before landing: Move into ``spmd_types`` as a version that takes
    per-axis types + ``PartitionSpec``, so the library handles multi-axis
    redistribute ordering internally.
    """
    mesh = current_mesh()
    if mesh is None:
        return x

    for axis_types in (src_types, dst_types):
        for axis_name, axis_type in axis_types.items():
            if not isinstance(axis_type, spmd.PerMeshAxisSpmdType):
                raise ValueError(
                    "SPMD backend requires SPMD placements. "
                    f"Got {axis_type!r} for axis {axis_name!r}."
                )

    for axis_name, dst_t in dst_types.items():
        src_t = src_types.get(axis_name)
        if src_t is None or src_t == dst_t or mesh_size(axis_name.value) == 1:
            continue
        x = spmd.redistribute(
            x,
            mesh.get_group(axis_name.value),
            src=src_t,
            dst=dst_t,
            backward_options={"op_dtype": x.dtype},
        )
    return x
