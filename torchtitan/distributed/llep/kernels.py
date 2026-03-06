# Copyright (c) Nous Research and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Triton kernels for LLEP distributed expert parallelism.

Includes:
- Fused GPU imbalance ratio computation
- Parallel token-to-GPU assignment
- Pad/unpad for grouped_mm alignment
- Vectorized send matrix computation
"""

import torch
import triton
import triton.language as tl


# =============================================================================
# Fused GPU Imbalance Ratio Kernel
# =============================================================================
# Replaces: counts.view(ep,nle).sum(1).float() -> mean/max/div + .item()
# 1 kernel launch instead of 6 PyTorch op launches.


@triton.jit
def _imbalance_ratio_kernel(
    counts_ptr,  # (num_experts,) int64
    output_ptr,  # (1,) float32 - the ratio
    NUM_EXPERTS: tl.constexpr,
    NUM_LOCAL: tl.constexpr,
    EP_SIZE: tl.constexpr,
):
    """Single-program kernel: compute max_gpu_load / mean_gpu_load."""
    # Load all expert counts
    offs = tl.arange(0, NUM_EXPERTS)
    counts = tl.load(counts_ptr + offs, mask=offs < NUM_EXPERTS, other=0).to(tl.float32)

    # Compute per-GPU loads via segmented sum
    # gpu_id = offs // NUM_LOCAL
    max_load = 0.0
    total_load = 0.0
    for g in tl.static_range(EP_SIZE):
        start = g * NUM_LOCAL
        g_offs = tl.arange(0, NUM_LOCAL)
        g_mask = (start + g_offs) < NUM_EXPERTS
        g_counts = tl.load(counts_ptr + start + g_offs, mask=g_mask, other=0).to(
            tl.float32
        )
        g_load = tl.sum(g_counts)
        total_load += g_load
        max_load = tl.where(g_load > max_load, g_load, max_load)

    mean_load = total_load / EP_SIZE
    ratio = tl.where(mean_load > 0.0, max_load / mean_load, 1.0)
    tl.store(output_ptr, ratio)


def triton_imbalance_ratio(
    expert_counts: torch.Tensor,
    ep_size: int,
    num_local_experts: int,
) -> float:
    """
    Fused GPU imbalance ratio: 1 Triton kernel instead of 6 PyTorch ops.

    Computes max_gpu_load / mean_gpu_load. Returns 1.0 for perfect balance.
    """
    num_experts = ep_size * num_local_experts
    counts = (
        expert_counts[:num_experts]
        if expert_counts.size(0) > num_experts
        else expert_counts
    )

    output = torch.empty(1, dtype=torch.float32, device=counts.device)

    # Constexprs must be powers of 2 for tl.arange
    # Round up num_experts and num_local to next power of 2
    import math

    NE_CONST = 1 << math.ceil(math.log2(max(num_experts, 1)))
    NL_CONST = 1 << math.ceil(math.log2(max(num_local_experts, 1)))

    _imbalance_ratio_kernel[(1,)](
        counts,
        output,
        NUM_EXPERTS=NE_CONST,
        NUM_LOCAL=NL_CONST,
        EP_SIZE=ep_size,
    )
    return output.item()


# =============================================================================
# Fused Token-to-GPU Assignment Kernel
# =============================================================================
# Replaces: Python loop over lpt_plan + per-GPU scatter in assign_tokens_to_gpus.
# Processes all tokens in parallel: each thread looks up its expert's plan,
# computes global position, finds matching assignment, writes target GPU.


@triton.jit
def _assign_tokens_kernel(
    # Sorted token data
    sort_perm_ptr,  # (N,) int64 - original positions of sorted tokens
    sorted_experts_ptr,  # (N,) int64 - expert IDs in sorted order
    # Offset tables
    local_offsets_ptr,  # (num_experts+1,) int64 - cumsum of local expert counts
    global_offsets_ptr,  # (num_experts,) int64 - this rank's offset into global tokens
    # LPT plan encoded as tensors
    plan_gpu_ids_ptr,  # (num_experts, MAX_ASSIGN) int32 - GPU ID per assignment
    plan_starts_ptr,  # (num_experts, MAX_ASSIGN) int64 - start token per assignment
    plan_ends_ptr,  # (num_experts, MAX_ASSIGN) int64 - end token per assignment
    plan_count_ptr,  # (num_experts,) int32 - num assignments per expert
    # Output
    target_gpus_ptr,  # (N,) int64 - target GPU per token
    # Sizes
    N,  # total number of token-expert pairs
    num_local_experts,
    MAX_ASSIGN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Parallel token-to-GPU assignment.

    Each program processes BLOCK_SIZE tokens from the sorted order.
    For each token: look up expert's LPT plan, compute global position,
    find matching (gpu, start, end) range, write target GPU.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load sorted expert IDs and original positions
    expert = tl.load(sorted_experts_ptr + offs, mask=mask, other=0)
    orig_pos = tl.load(sort_perm_ptr + offs, mask=mask, other=0)

    # Default routing: expert_id // num_local_experts
    target = expert // num_local_experts

    # Check which tokens have LPT plan entries
    n_assign = tl.load(plan_count_ptr + expert, mask=mask, other=0)
    has_plan = n_assign > 0

    # Compute global position for planned tokens
    local_start = tl.load(local_offsets_ptr + expert, mask=mask & has_plan, other=0)
    global_off = tl.load(global_offsets_ptr + expert, mask=mask & has_plan, other=0)
    # offs is the index in sorted order; offs - local_start = position within expert's group
    global_pos = global_off + (offs - local_start)

    # Search through plan assignments (unrolled loop)
    for a in tl.static_range(MAX_ASSIGN):
        a_mask = mask & has_plan & (a < n_assign)
        base = expert * MAX_ASSIGN + a
        gpu = tl.load(plan_gpu_ids_ptr + base, mask=a_mask, other=-1).to(tl.int64)
        start = tl.load(plan_starts_ptr + base, mask=a_mask, other=0)
        end = tl.load(plan_ends_ptr + base, mask=a_mask, other=0)
        in_range = a_mask & (global_pos >= start) & (global_pos < end) & (gpu >= 0)
        target = tl.where(in_range, gpu, target)

    # Scatter-write to original positions
    tl.store(target_gpus_ptr + orig_pos, target, mask=mask)


def triton_assign_tokens(
    flat_experts: torch.Tensor,  # (N,) int64
    local_expert_counts: torch.Tensor,  # (num_experts,) int64
    global_offsets: torch.Tensor,  # (num_experts,) int64 - this rank's cumulative offset
    lpt_plan: dict,  # expert_id -> [(gpu_id, start, end), ...]
    num_local_experts: int,
    num_experts: int,
) -> torch.Tensor:
    """
    Fused token-to-GPU assignment using Triton kernel.

    Replaces the Python loop + per-GPU scatter with a single parallel kernel.
    Returns target_gpus tensor (N,) with GPU assignment per token.
    """
    N = flat_experts.shape[0]
    device = flat_experts.device

    # Sort tokens by expert (still needed for contiguous expert groups)
    sorted_experts, sort_perm = flat_experts.sort(stable=True)

    # Compute local offsets (cumsum)
    local_offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    local_offsets[1:] = local_expert_counts.cumsum(0)

    # Encode LPT plan as tensors — build on CPU (numpy), single H2D transfer
    max_assign = max((len(a) for a in lpt_plan.values()), default=1) if lpt_plan else 1
    max_assign = max(max_assign, 1)

    import numpy as np

    gpu_ids_np = np.full((num_experts, max_assign), -1, dtype=np.int32)
    starts_np = np.zeros((num_experts, max_assign), dtype=np.int64)
    ends_np = np.zeros((num_experts, max_assign), dtype=np.int64)
    count_np = np.zeros(num_experts, dtype=np.int32)

    if lpt_plan:
        for eid, assignments in lpt_plan.items():
            count_np[eid] = len(assignments)
            for j, (gpu_id, start, end) in enumerate(assignments):
                gpu_ids_np[eid, j] = gpu_id
                starts_np[eid, j] = start
                ends_np[eid, j] = end

    # Single H2D: pack into one contiguous buffer, transfer, slice
    plan_gpu_ids = torch.from_numpy(gpu_ids_np).to(device, non_blocking=True)
    plan_starts = torch.from_numpy(starts_np).to(device, non_blocking=True)
    plan_ends = torch.from_numpy(ends_np).to(device, non_blocking=True)
    plan_count = torch.from_numpy(count_np).to(device, non_blocking=True)

    # Allocate output
    target_gpus = torch.empty(N, dtype=torch.int64, device=device)

    if N == 0:
        return target_gpus

    # Round max_assign up to power of 2 for tl.static_range
    import math

    MA_CONST = 1 << math.ceil(math.log2(max(max_assign, 1)))

    BLOCK_SIZE = 1024
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _assign_tokens_kernel[grid](
        sort_perm,
        sorted_experts,
        local_offsets,
        global_offsets,
        plan_gpu_ids,
        plan_starts,
        plan_ends,
        plan_count,
        target_gpus,
        N,
        num_local_experts,
        MAX_ASSIGN=MA_CONST,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return target_gpus


# =============================================================================
# Fused Pad for Grouped MM Kernel (v2: row-parallel with precomputed mapping)
# =============================================================================
# Replaces: vectorized PyTorch scatter with cumsum + broadcast + masked_select
# Strategy: precompute dst_index for each src row using repeat_interleave,
# then copy rows with a row-parallel Triton kernel.
# Why not binary search? repeat_interleave is a single CUDA kernel and avoids
# Triton compile-time static_range issues with variable expert counts.


@triton.jit
def _copy_rows_kernel(
    src_ptr,  # (num_rows, dim) source tensor
    dst_ptr,  # (num_dst_rows, dim) destination tensor
    index_ptr,  # (num_rows,) int64 - dst row index per src row
    num_rows,
    dim,
    BLOCK_D: tl.constexpr,
):
    """
    Row-parallel copy: each program copies one src row to dst[index[row]].
    Grid: (num_rows, num_col_blocks).
    """
    row_idx = tl.program_id(0)
    col_block = tl.program_id(1)

    if row_idx >= num_rows:
        return

    dst_row = tl.load(index_ptr + row_idx)

    col_start = col_block * BLOCK_D
    col_offs = col_start + tl.arange(0, BLOCK_D)
    col_mask = col_offs < dim

    vals = tl.load(src_ptr + row_idx * dim + col_offs, mask=col_mask, other=0.0)
    tl.store(dst_ptr + dst_row * dim + col_offs, vals, mask=col_mask)


def triton_pad_for_grouped_mm(
    x_sorted: torch.Tensor,  # (total_tokens, dim)
    counts: torch.Tensor,  # (n,) int64
    counts_padded: torch.Tensor,  # (n,) int64 - already computed aligned counts
) -> torch.Tensor:
    """
    Triton-accelerated padding for grouped_mm.

    Precomputes a dst-index mapping via repeat_interleave, then uses a
    row-parallel kernel to scatter rows to their padded positions.
    """
    n = counts.shape[0]
    dim = x_sorted.shape[1]
    device = x_sorted.device
    total_tokens = x_sorted.shape[0]

    total_padded = counts_padded.sum().item()
    x_padded = torch.zeros(total_padded, dim, device=device, dtype=x_sorted.dtype)

    if n == 0 or total_tokens == 0:
        return x_padded

    # Build dst-index mapping: for each src row, compute which dst row it maps to
    # dst_index[src_offsets[i] + j] = dst_offsets[i] + j, for j in [0, counts[i])
    #
    # = repeat_interleave(dst_offsets, counts) + position_within_group
    # position_within_group = arange(total) - repeat_interleave(src_offsets, counts)
    src_offsets = torch.zeros(n, dtype=torch.int64, device=device)
    if n > 1:
        src_offsets[1:] = counts[:-1].cumsum(0)
    dst_offsets = torch.zeros(n, dtype=torch.int64, device=device)
    if n > 1:
        dst_offsets[1:] = counts_padded[:-1].cumsum(0)

    # Only process non-empty experts (skip zero-count experts)
    nonempty = counts > 0
    if not nonempty.all():
        ne_counts = counts[nonempty]
        ne_src_offsets = src_offsets[nonempty]
        ne_dst_offsets = dst_offsets[nonempty]
    else:
        ne_counts = counts
        ne_src_offsets = src_offsets
        ne_dst_offsets = dst_offsets

    # repeat_interleave: O(total_tokens) single CUDA kernel
    src_base = torch.repeat_interleave(ne_src_offsets, ne_counts)
    dst_base = torch.repeat_interleave(ne_dst_offsets, ne_counts)

    src_indices = torch.arange(total_tokens, device=device, dtype=torch.int64)
    dst_index = dst_base + (src_indices - src_base)

    BLOCK_D = 1024
    if dim > 1024:
        BLOCK_D = 2048
    if dim > 2048:
        BLOCK_D = 4096

    num_col_blocks = (dim + BLOCK_D - 1) // BLOCK_D
    grid = (total_tokens, num_col_blocks)

    _copy_rows_kernel[grid](
        x_sorted,
        x_padded,
        dst_index,
        total_tokens,
        dim,
        BLOCK_D=BLOCK_D,
    )

    return x_padded


# =============================================================================
# Fused Unpad Output Kernel (v2: row-parallel with precomputed mapping)
# =============================================================================


def triton_unpad_output(
    out_padded: torch.Tensor,  # (total_padded, dim)
    counts: torch.Tensor,  # (n,) int64
    counts_padded: torch.Tensor,  # (n,) int64
) -> torch.Tensor:
    """
    Triton-accelerated unpadding: reverse of pad_for_grouped_mm.

    Precomputes src-index mapping, uses row-parallel kernel to gather rows.
    """
    n = counts.shape[0]
    dim = out_padded.shape[1]
    device = out_padded.device

    total_tokens = counts.sum().item()
    out = torch.empty(total_tokens, dim, device=device, dtype=out_padded.dtype)

    if n == 0 or total_tokens == 0:
        return out

    # Build src-index mapping: for each dst row, which padded src row to read from
    # src_index[dst_offsets[i] + j] = padded_offsets[i] + j
    dst_offsets = torch.zeros(n, dtype=torch.int64, device=device)
    if n > 1:
        dst_offsets[1:] = counts[:-1].cumsum(0)
    padded_offsets = torch.zeros(n, dtype=torch.int64, device=device)
    if n > 1:
        padded_offsets[1:] = counts_padded[:-1].cumsum(0)

    nonempty = counts > 0
    if not nonempty.all():
        ne_counts = counts[nonempty]
        ne_dst_offsets = dst_offsets[nonempty]
        ne_padded_offsets = padded_offsets[nonempty]
    else:
        ne_counts = counts
        ne_dst_offsets = dst_offsets
        ne_padded_offsets = padded_offsets

    dst_base = torch.repeat_interleave(ne_dst_offsets, ne_counts)
    src_base = torch.repeat_interleave(ne_padded_offsets, ne_counts)

    row_indices = torch.arange(total_tokens, device=device, dtype=torch.int64)
    src_index = src_base + (row_indices - dst_base)  # padded row for each output row

    BLOCK_D = 1024
    if dim > 1024:
        BLOCK_D = 2048
    if dim > 2048:
        BLOCK_D = 4096

    num_col_blocks = (dim + BLOCK_D - 1) // BLOCK_D
    grid = (total_tokens, num_col_blocks)

    # Reuse _copy_rows_kernel: src=out_padded, dst=out, index=src_index
    # But we need reverse: read from src_index[row], write to row
    # Use a "gather" variant
    _gather_rows_kernel[grid](
        out_padded,
        out,
        src_index,
        total_tokens,
        dim,
        BLOCK_D=BLOCK_D,
    )

    return out


@triton.jit
def _gather_rows_kernel(
    src_ptr,  # (num_src_rows, dim) source tensor
    dst_ptr,  # (num_rows, dim) destination tensor
    index_ptr,  # (num_rows,) int64 - src row index per dst row
    num_rows,
    dim,
    BLOCK_D: tl.constexpr,
):
    """
    Row-parallel gather: each program reads src[index[row]] and writes to dst[row].
    Grid: (num_rows, num_col_blocks).
    """
    row_idx = tl.program_id(0)
    col_block = tl.program_id(1)

    if row_idx >= num_rows:
        return

    src_row = tl.load(index_ptr + row_idx)

    col_start = col_block * BLOCK_D
    col_offs = col_start + tl.arange(0, BLOCK_D)
    col_mask = col_offs < dim

    vals = tl.load(src_ptr + src_row * dim + col_offs, mask=col_mask, other=0.0)
    tl.store(dst_ptr + row_idx * dim + col_offs, vals, mask=col_mask)


# =============================================================================
# Vectorized Send Matrix Computation
# =============================================================================
# Replaces nested Python loops in assign_tokens_to_gpus send_matrix calculation.
# Uses vectorized numpy operations instead of per-expert Python iteration.


def compute_send_matrix_vectorized(
    all_counts_np,  # (ep_size, num_experts) numpy
    cum_counts_np,  # (ep_size+1, num_experts) numpy
    lpt_plan: dict,  # expert_id -> [(gpu_id, start, end), ...]
    ep_size: int,
    num_local_experts: int,
    num_experts: int,
) -> "np.ndarray":
    """
    Vectorized send_matrix computation using numpy broadcasting.

    For non-LPT experts: vectorized sum per destination GPU.
    For LPT experts: still needs per-assignment loop but uses vectorized overlap.
    """
    import numpy as np

    send_matrix = np.zeros((ep_size, ep_size), dtype=np.int64)

    if not lpt_plan:
        # Simple default routing
        return all_counts_np.reshape(ep_size, ep_size, num_local_experts).sum(axis=2)

    lpt_expert_ids = np.array(list(lpt_plan.keys()), dtype=np.int64)

    # --- Non-LPT experts: vectorized ---
    is_lpt = np.zeros(num_experts, dtype=bool)
    is_lpt[lpt_expert_ids] = True
    non_lpt_mask = ~is_lpt

    expert_owners = np.arange(num_experts) // num_local_experts

    non_lpt_counts = all_counts_np[:, non_lpt_mask]
    non_lpt_owners = expert_owners[non_lpt_mask]

    for dst in range(ep_size):
        dst_mask = non_lpt_owners == dst
        if dst_mask.any():
            send_matrix[:, dst] += non_lpt_counts[:, dst_mask].sum(axis=1)

    # --- LPT experts: vectorized per-expert overlap ---
    for expert_id, assignments in lpt_plan.items():
        src_starts = cum_counts_np[:-1, expert_id]
        src_ends = cum_counts_np[1:, expert_id]

        for dst_gpu, dst_start, dst_end in assignments:
            overlap_starts = np.maximum(src_starts, dst_start)
            overlap_ends = np.minimum(src_ends, dst_end)
            overlaps = np.maximum(overlap_ends - overlap_starts, 0)
            send_matrix[:, dst_gpu] += overlaps

    return send_matrix
