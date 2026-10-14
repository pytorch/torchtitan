# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exact router token counts without materializing dense expert masks."""

import torch
import triton
import triton.language as tl

# Shape suffix legend:
#   T = num tokens, K = eight selected experts, E = 256 experts,
#   R = routed assignments, P = partial histograms


@triton.jit
def _partial_counts_kernel(
    topk_expert_ids,
    partial_counts,
    num_assignments: tl.constexpr,
    block_assignments: tl.constexpr,
):
    partial_id = tl.program_id(0)
    assignment_ids_R = partial_id * block_assignments + tl.arange(0, block_assignments)
    valid_R = assignment_ids_R < num_assignments
    expert_ids_R = tl.load(topk_expert_ids + assignment_ids_R, valid_R, other=0).to(
        tl.int32
    )
    partial_counts_E = tl.histogram(expert_ids_R, 256, mask=valid_R)
    expert_ids_E = tl.arange(0, 256)
    tl.store(partial_counts + partial_id * 256 + expert_ids_E, partial_counts_E)


@triton.jit
def _sum_counts_kernel(
    partial_counts,
    counts_int64,
    counts_float32,
    num_partials: tl.constexpr,
    block_partials: tl.constexpr,
    block_experts: tl.constexpr,
):
    partial_ids_P = tl.arange(0, block_partials)
    expert_ids_E = tl.program_id(0) * block_experts + tl.arange(0, block_experts)
    partial_counts_PE = tl.load(
        partial_counts + partial_ids_P[:, None] * 256 + expert_ids_E[None, :],
        partial_ids_P[:, None] < num_partials,
        other=0,
    )
    counts_E = tl.sum(partial_counts_PE, 0)
    tl.store(counts_int64 + expert_ids_E, counts_E.to(tl.int64))
    tl.store(counts_float32 + expert_ids_E, counts_E.to(tl.float32))


def token_counts(
    topk_expert_ids_TK: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return int64 and FP32 per-expert counts for a graph-proven topk output.

    The input must be contiguous CUDA int64 [T, 8], with distinct expert
    IDs in [0, 256) within every row. The graph pass must establish that the
    IDs come from topk before replacing a dense scatter-mask reduction.
    Duplicate IDs have different dense-scatter semantics and are unsupported.
    T is positive and at most 2**24 so every FP32 count is exact. Both
    reductions are integer arithmetic, with no nondeterministic FP32 atomics.
    """
    if (
        topk_expert_ids_TK.ndim != 2
        or topk_expert_ids_TK.shape[1] != 8
        or not 0 < topk_expert_ids_TK.shape[0] <= 2**24
        or topk_expert_ids_TK.dtype != torch.int64
        or not topk_expert_ids_TK.is_cuda
        or not topk_expert_ids_TK.is_contiguous()
    ):
        raise ValueError(
            "Topk expert IDs must be contiguous CUDA int64 [T, 8], "
            "with 0 < T <= 2**24."
        )
    num_assignments = topk_expert_ids_TK.numel()
    num_partials = triton.cdiv(num_assignments, 512)
    partial_counts_PE = torch.empty(
        (num_partials, 256), device=topk_expert_ids_TK.device, dtype=torch.int32
    )
    counts_int64_E = torch.empty(
        (256,), device=topk_expert_ids_TK.device, dtype=torch.int64
    )
    counts_float32_E = torch.empty(
        (256,), device=topk_expert_ids_TK.device, dtype=torch.float32
    )
    _partial_counts_kernel[(num_partials,)](
        topk_expert_ids_TK,
        partial_counts_PE,
        num_assignments,
        512,
        num_warps=4,
    )
    _sum_counts_kernel[(8,)](
        partial_counts_PE,
        counts_int64_E,
        counts_float32_E,
        num_partials,
        triton.next_power_of_2(num_partials),
        32,
        num_warps=4,
    )
    return counts_int64_E, counts_float32_E
