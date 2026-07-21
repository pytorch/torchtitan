# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Selective K/V gather for context parallel.

For sliding-window / sparse attention under CP, most of an all-gathered K/V is
masked out. Selective gather fetches only the blocks attention actually reads.

The package exposes one differentiable API -- ``selective_gather`` over a
``SelectiveGatherContext`` -- on top of a transport-agnostic plan/metadata layer
(``topology``). The context picks a backend: the CuTeDSL LSA kernels (nccl4py +
CuTeDSL + NCCL windows, Hopper+; intra-node NVLink, fastest) or the portable
``batch_isend_irecv`` P2P baseline (anywhere NCCL/RCCL point-to-point works,
incl. AMD). The CuTeDSL/nccl4py imports are lazy, so this package imports on
P2P-only hosts.
"""

# Apply the cutlass-dsl / nccl4py version shim before any lazy nccl import
# (transport host API, the lsa device kernels). No-op without cutlass.
from . import _compat  # noqa: F401

from .autograd import selective_gather
from .backend import select_backend
from .p2p import run_p2p_gather, run_p2p_gather_backward
from .topology import (
    backward_staging_map,
    BlockGatherPlan,
    build_plan_metadata,
    full_plan,
    PlanMetadata,
    sliding_window_plan,
)
from .transport import SelectiveGatherContext

__all__ = [
    "BlockGatherPlan",
    "SelectiveGatherContext",
    "PlanMetadata",
    "full_plan",
    "sliding_window_plan",
    "backward_staging_map",
    "build_plan_metadata",
    "run_p2p_gather",
    "run_p2p_gather_backward",
    "selective_gather",
    "select_backend",
]
