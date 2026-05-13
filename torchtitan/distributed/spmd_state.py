# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scoped SPMD runtime state for the full_spmd_types path."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

import spmd_types as spmd
from torch.distributed import ProcessGroup

if TYPE_CHECKING:
    from torchtitan.distributed.parallel_dims import ParallelDims


_CURRENT_PARALLEL_DIMS: ContextVar["ParallelDims | None"] = ContextVar(
    "torchtitan_current_parallel_dims", default=None
)
_DP_AXIS_NAMES = ("dp_replicate", "dp_shard", "cp")


def is_spmd_active() -> bool:
    """Check whether the current context is inside the SPMD path."""
    return _CURRENT_PARALLEL_DIMS.get() is not None


@contextmanager
def set_current_parallel_dims(parallel_dims: "ParallelDims") -> Iterator[None]:
    """Set Torchtitan and spmd_types current mesh state for one runtime region."""
    token = _CURRENT_PARALLEL_DIMS.set(parallel_dims)
    with spmd.set_current_mesh(parallel_dims._global_meshes["dense"]):
        try:
            yield
        finally:
            _CURRENT_PARALLEL_DIMS.reset(token)


def _current_parallel_dims() -> "ParallelDims":
    parallel_dims = _CURRENT_PARALLEL_DIMS.get()
    assert parallel_dims is not None, (
        "No current ParallelDims is set. Use set_current_parallel_dims() around "
        "SPMD runtime code."
    )
    return parallel_dims


def current_pg(axis_name: str) -> ProcessGroup | None:
    """Return the current process group for an active mesh axis."""
    if not is_spmd_active():
        return None
    mesh = _current_parallel_dims().get_optional_mesh(axis_name)
    if mesh is None or mesh.size() == 1:
        return None
    pg = mesh.get_group()
    pg._set_group_desc(axis_name)
    return pg


def current_dp_axis_names() -> tuple[str, ...]:
    """Return active data-parallel-like axes used for loss partial semantics."""
    parallel_dims = _current_parallel_dims()
    return tuple(
        axis_name
        for axis_name in _DP_AXIS_NAMES
        if (mesh := parallel_dims.get_optional_mesh(axis_name)) is not None
        and mesh.size() > 1
    )
