# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scoped SPMD runtime mesh state."""

from collections.abc import Iterator
from contextlib import contextmanager
from threading import local

import spmd_types as spmd
from torch.distributed.device_mesh import DeviceMesh


_TLS = local()


def _mesh_stack() -> list[DeviceMesh]:
    stack = getattr(_TLS, "mesh_stack", None)
    if stack is None:
        stack = []
        _TLS.mesh_stack = stack
    return stack


def is_spmd_active() -> bool:
    """Check whether the current context is inside the SPMD path."""
    return bool(_mesh_stack())


def current_mesh() -> DeviceMesh | None:
    """Return the current SPMD runtime mesh, or ``None`` if unset."""
    stack = _mesh_stack()
    if not stack:
        return None
    return stack[-1]


@contextmanager
def set_current_mesh(mesh: DeviceMesh | None) -> Iterator[None]:
    """Set Torchtitan and spmd_types current mesh state for one runtime region."""
    if mesh is None or current_mesh() is mesh:
        yield
        return

    stack = _mesh_stack()
    stack.append(mesh)
    with spmd.set_current_mesh(mesh):
        try:
            yield
        finally:
            popped = stack.pop()
            assert popped is mesh
