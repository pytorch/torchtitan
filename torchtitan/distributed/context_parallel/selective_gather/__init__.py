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
(``topology``). The context picks a backend by CP-group topology: the CuTeDSL
kernels (nccl4py + CuTeDSL + NCCL windows, Hopper+) -- LSA intra-node over
NVLink, GIN inter-node over RDMA -- or the portable ``batch_isend_irecv`` P2P
baseline (anywhere NCCL/RCCL point-to-point works, incl. AMD). The
CuTeDSL/nccl4py imports are lazy, so this package imports on P2P-only hosts.
"""

# Apply the cutlass-dsl / nccl4py version shim before any lazy nccl import
# (transport host API, lsa/gin device kernels). No-op without cutlass.
from . import _compat  # noqa: F401

from .autograd import selective_gather
from .backend import select_backend
from .p2p import run_p2p_gather, run_p2p_gather_backward
from .topology import (
    backward_staging_map,
    BlockGatherPlan,
    build_gin_metadata,
    build_plan_metadata,
    full_plan,
    GINMetadata,
    PlanMetadata,
    sliding_window_plan,
)
from .transport import SelectiveGatherContext

# gin_kernel is intentionally NOT imported here: it imports nccl.core.device.cute
# at module load, so keep it lazy (like lsa_kernel) to stay importable on hosts
# without CuTeDSL/nccl4py. Import it as selective_gather.gin_kernel where needed.

__all__ = [
    "BlockGatherPlan",
    "SelectiveGatherContext",
    "PlanMetadata",
    "GINMetadata",
    "full_plan",
    "sliding_window_plan",
    "backward_staging_map",
    "build_plan_metadata",
    "build_gin_metadata",
    "run_p2p_gather",
    "run_p2p_gather_backward",
    "selective_gather",
    "select_backend",
]
