# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from .placements import Placement


# Module attribute names for storing DStorage
_DSTORAGE_ATTR = "_dstorage"
_DSTORAGES_ATTR = "_dstorages"

# Hidden attribute names for placement metadata on plain tensors
_PLACEMENTS_ATTR = "_placements"
_GLOBAL_SHAPE_ATTR = "_global_shape"
_GLOBAL_STRIDE_ATTR = "_global_stride"
_MESH_ATTR = "_mesh"
_EAGER_BATCHED_HOOK_REGISTERED_ATTR = "_flex_shard_eager_batched_hook_registered"
_EAGER_COMM_CONTEXTS_ATTR = "_flex_shard_eager_comm_contexts"
_PARAM_FQN_ATTR = "_flex_shard_param_fqn"
_BUCKET_FQN_ATTR = "_flex_shard_bucket_fqn"
_EAGER_AUTOGRAD_BUCKET_UNSHARD_ATTR = "_flex_shard_eager_autograd_bucket_unshard"


def set_sharding_info(
    tensor: torch.Tensor,
    placements: tuple[Placement, ...],
    global_shape: torch.Size,
    global_stride: tuple[int, ...],
    mesh: DeviceMesh,
) -> None:
    """Annotate a tensor with FlexShard placement metadata."""
    tensor._placements = placements
    tensor._global_shape = global_shape
    tensor._global_stride = global_stride
    tensor._mesh = mesh


def get_placements(tensor: torch.Tensor) -> tuple[Placement, ...] | None:
    """Get FlexShard placements from a tensor, or None if not annotated."""
    return getattr(tensor, _PLACEMENTS_ATTR, None)


def get_global_shape(tensor: torch.Tensor) -> torch.Size | None:
    """Get the global (unsharded) shape from a tensor, or None if not annotated."""
    return getattr(tensor, _GLOBAL_SHAPE_ATTR, None)


def is_flex_shard_param(tensor: torch.Tensor) -> bool:
    """Check if a tensor has FlexShard placement metadata."""
    return hasattr(tensor, _PLACEMENTS_ATTR)
