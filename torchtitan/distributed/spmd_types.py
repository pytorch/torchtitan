# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Helpers for bridging TorchTitan SPMD layouts to DTensor placements."""

from typing import TYPE_CHECKING

import spmd_types as spmd
from torch.distributed.tensor import Partial, Placement, Replicate, Shard

if TYPE_CHECKING:
    from torchtitan.distributed.parallel_dims import MeshAxisName, SpmdLayout


def spmd_layout_to_dtensor_placements(
    layout: "SpmdLayout",
) -> dict["MeshAxisName", Placement]:
    """Convert an SPMD layout to DTensor placements keyed by mesh axis name."""
    result: dict[MeshAxisName, Placement] = {}
    for axis_name, axis_type in layout.shard_types().items():
        if axis_type == spmd.R or axis_type == spmd.I:
            dtensor_placement: Placement = Replicate()
        elif axis_type == spmd.P:
            dtensor_placement = Partial()
        else:
            assert isinstance(axis_type, spmd.Shard)
            dtensor_placement = Shard(axis_type.dim)
        result[axis_name] = dtensor_placement
    return result
