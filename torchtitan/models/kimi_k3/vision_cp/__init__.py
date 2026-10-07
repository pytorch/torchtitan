# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dynamic context parallelism for the vision tower: large images split by rows over sub-CP groups."""

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.distributed.parallelism_context import MeshAxisName, ParallelismContext

from .attention import VisionCPAttention, VisionCPLayout
from .encoder import MoonViTCPEncoder

__all__ = [
    "build_cp_subgroups",
    "install_vision_cp",
    "MoonViTCPEncoder",
    "VisionCPAttention",
    "VisionCPLayout",
]


def build_cp_subgroups(cp_mesh: DeviceMesh) -> dict[int, dist.ProcessGroup]:
    """This rank's group for every equal split of its CP mesh, keyed by the number of sub-groups."""
    cp_size = cp_mesh.size()
    subgroups = {1: cp_mesh.get_group()}
    for num in range(2, cp_size):
        if cp_size % num:
            continue
        index, sub = f"cp_sub{num}_index", f"cp_sub{num}"
        mesh = cp_mesh._unflatten(
            0, (num, cp_size // num), (index, sub), backend_override={index: "fake"}
        )
        subgroups[num] = mesh[sub].get_group()
    return subgroups


def install_vision_cp(
    tower: MoonViTCPEncoder | None, parallelism_context: ParallelismContext
) -> None:
    """Under context parallelism, build the sub-CP groups and hand them to the tower."""
    if not parallelism_context.cp_enabled:
        return
    # Building groups is collective, so ranks without the tower build them too.
    subgroups = build_cp_subgroups(parallelism_context.get_mesh(MeshAxisName.CP))
    if tower is not None:
        tower.set_cp_subgroups(subgroups)
