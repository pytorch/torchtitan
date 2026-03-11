# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.distributed.varlen_cp.dispatch_ops import (
    compute_local_cu_seqlens,
    dispatch,
    undispatch,
)
from torchtitan.distributed.varlen_cp.dispatch_solver import (
    ChunkPair,
    DispatchPlan,
    MagiDispatchPlan,
    solve_dispatch,
    solve_magi_dispatch,
)
from torchtitan.distributed.varlen_cp.mask_primitives import (
    AttnSlice,
    cu_seqlens_to_attn_slices,
    make_slice_mask,
    MaskType,
    split_slice_at_chunk_boundary,
)
from torchtitan.distributed.varlen_cp.ring_attention import (
    merge_with_lse,
    varlen_ring_attention,
)

__all__ = [
    "AttnSlice",
    "ChunkPair",
    "DispatchPlan",
    "MagiDispatchPlan",
    "MaskType",
    "compute_local_cu_seqlens",
    "cu_seqlens_to_attn_slices",
    "dispatch",
    "make_slice_mask",
    "merge_with_lse",
    "solve_dispatch",
    "solve_magi_dispatch",
    "split_slice_at_chunk_boundary",
    "undispatch",
    "varlen_ring_attention",
]
