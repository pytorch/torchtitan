##############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
##############################################################################
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import math

import torch
import triton
import triton.language as tl


# Assign a block to a row([1,topk]), generate a local routing map([1,num_of_local_experts])
@triton.jit
def _indices_to_multihot_kernel(
    indices_ptr,
    probs_in_indices_ptr,
    multihot_indices_ptr,  # bool
    probs_in_multihot_ptr,
    position_map_ptr,
    num_of_local_experts: tl.constexpr,
    num_of_local_experts_next_power_of_2: tl.constexpr,
    topk: tl.constexpr,
    topk_next_power_of_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for converting indices to multihot representation.
    Input:
        indices: [num_of_tokens, topk]
        probs_in_indices: [num_of_tokens, topk]
    Output:
        multihot_indices: [num_of_tokens, num_of_local_experts]
        probs_in_multihot: [num_of_tokens, num_of_local_experts]
    Assume that topk = 2 , num_of_local_experts = 4, num_of_tokens = 2,
    then the kernel can process the following conversion:
    Input Example:
        indices = [
                [0, 1],
                [1, 2]
            ]
        probs_in_indices = [
                [0.1, 0.2],
                [0.3, 0.4]
            ]
    Output Example:
        multihot_indices = [
                [1, 1, -1, -1],
                [-1, 1, 1, -1]
            ]
        probs_in_multihot = [
                [0.1, 0.2, 0.0, 0.0],
                [0.0, 0.3, 0.4, 0.0]
            ]
    """
    # Prepare the [0, topk) row
    topk_row = tl.arange(0, topk_next_power_of_2)
    topk_row = tl.where(topk_row < topk, topk_row, -1)
    topk_row_mask = topk_row != -1
    # Prepare the [0, num_of_local_experts) row
    num_exp_row = tl.arange(0, num_of_local_experts_next_power_of_2)
    num_exp_row = tl.where(num_exp_row < num_of_local_experts, num_exp_row, -1)
    num_exp_row_mask = num_exp_row != -1

    # Load a [1, topk] row from the indices buffer
    row_idx = tl.program_id(0)
    indices_row = tl.load(indices_ptr + row_idx * topk + topk_row, mask=topk_row_mask)
    indices_row = tl.where(topk_row_mask, indices_row, -1)
    probs_row = tl.load(
        probs_in_indices_ptr + row_idx * topk + topk_row, mask=topk_row_mask
    )

    # Get the position of the each index in the indices_row, which is saved for backwards
    position_row = tl.where(indices_row != -1, topk_row, -1)
    # Mask of the valid indices
    mask = (indices_row != -1) & (indices_row < num_of_local_experts)

    row_idx_offset = row_idx * num_of_local_experts
    # Store to initialize
    tl.store(
        multihot_indices_ptr + row_idx_offset + num_exp_row, 0, mask=num_exp_row_mask
    )
    tl.store(
        probs_in_multihot_ptr + row_idx_offset + num_exp_row, 0, mask=num_exp_row_mask
    )
    tl.store(position_map_ptr + row_idx_offset + num_exp_row, -1, mask=num_exp_row_mask)
    # Use barrier to make sure the initialization is done
    tl.debug_barrier()
    # Store the indices and probs_in_indices
    tl.store(multihot_indices_ptr + row_idx_offset + indices_row, 1, mask)
    tl.store(probs_in_multihot_ptr + row_idx_offset + indices_row, probs_row, mask)
    # Store the position of the position_row for backwards
    tl.store(position_map_ptr + row_idx_offset + indices_row, position_row, mask)


# Assign a block to a row([1,topk]), generate a probs_indices([1,topk])
@triton.jit
def _multihot_to_indices_kernel(
    probs_in_multihot_ptr,
    position_map_ptr,
    probs_indices_ptr,
    num_of_local_experts: tl.constexpr,
    num_of_local_experts_next_power_of_2: tl.constexpr,
    topk: tl.constexpr,
    topk_next_power_of_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for converting multihot representation to indices.
    Input:
        probs_in_multihot: [num_of_tokens, num_of_local_experts]
        position_map: [num_of_tokens, num_of_local_experts]
    Output:
        probs_indices: [num_of_tokens, topk]
    Assume that topk = 2 , num_of_local_experts = 4, num_of_tokens = 2,
    then the kernel can process the following conversion:
    Input Example:
        probs_in_multihot = [
                [0.7, 0.8, 0.0, 0.0],
                [0.0, 0.1, 0.9, 0.0]
            ]
        position_map = [
                [1, 1, -1, -1],
                [-1, 1, 1, -1]
            ]
    Output Example:
        probs_indices = [
                [0.7, 0.8],
                [0.1, 0.9]
            ]
    """
    # Prepare the [0, topk) row
    topk_row = tl.arange(0, topk_next_power_of_2)
    topk_row = tl.where(topk_row < topk, topk_row, -1)
    topk_row_mask = topk_row != -1
    # Prepare the [0, num_of_local_experts) row
    num_exp_row = tl.arange(0, num_of_local_experts_next_power_of_2)
    num_exp_row = tl.where(num_exp_row < num_of_local_experts, num_exp_row, -1)
    num_exp_row_mask = num_exp_row != -1

    # Load a [1, num_of_local_experts] row from the local routing map
    row_idx = tl.program_id(0)
    ptr_offset = row_idx * num_of_local_experts + num_exp_row
    probs_in_multihot_row = tl.load(
        probs_in_multihot_ptr + ptr_offset, mask=num_exp_row_mask
    )

    # Get the original position of the valid value in the the indices
    position_map_row = tl.load(position_map_ptr + ptr_offset, mask=num_exp_row_mask)
    position_map_row = tl.where(num_exp_row_mask, position_map_row, -1)
    mask = position_map_row != -1

    # Store to initialize
    tl.store(probs_indices_ptr + row_idx * topk + topk_row, 0, mask=topk_row_mask)
    # Use barrier to make sure the initialization is done
    tl.debug_barrier()
    # Restore the indices and probs_indices
    tl.store(
        probs_indices_ptr + row_idx * topk + position_map_row,
        probs_in_multihot_row,
        mask,
    )


class IndicesToMultihot(torch.autograd.Function):
    """Convert moe topk indices to multihot representation.
    This class implements a custom forward and backward propagation
    operation for efficiently converting indices to multihot
    representation.
    It is an experimental feature and may change in future versions.
    """

    @staticmethod
    def forward(ctx, indices, probs_indices, num_of_local_experts):
        """Forward function for IndicesToMultihot
        Convert indices to multihot representation.
        Args:
            indices: [num_of_tokens, topk]
            probs_indices: [num_of_tokens, topk]
            num_of_local_experts: int
        Returns:
            multihot_indices: [num_of_tokens, num_of_local_experts]
            probs_in_multihot: [num_of_tokens, num_of_local_experts]
        """
        num_of_tokens = indices.shape[0]
        assert (
            indices.shape == probs_indices.shape
        ), "indices and probs_indices must have the same shape"
        topk = indices.shape[1]
        multihot_indices = torch.empty(
            (num_of_tokens, num_of_local_experts), dtype=torch.bool, device="cuda"
        )
        probs_in_multihot = torch.empty(
            (num_of_tokens, num_of_local_experts),
            dtype=probs_indices.dtype,
            device="cuda",
        )
        position_map = torch.empty(
            (num_of_tokens, num_of_local_experts), dtype=torch.int32, device="cuda"
        )
        # Compute the next power of 2 for the topk and num_of_local_experts
        topk_next_power_of_2 = 2 ** int(math.ceil(math.log2(topk)))
        num_of_local_experts_next_power_of_2 = 2 ** int(
            math.ceil(math.log2(num_of_local_experts))
        )
        grid = (num_of_tokens,)
        _indices_to_multihot_kernel[grid](
            indices,
            probs_indices,
            multihot_indices,
            probs_in_multihot,
            position_map,
            num_of_local_experts,
            num_of_local_experts_next_power_of_2,
            topk,
            topk_next_power_of_2,
            BLOCK_SIZE=32,  # use only 1 warp per block
            num_warps=1,
        )

        ctx.save_for_backward(position_map)
        ctx.num_of_tokens = num_of_tokens
        ctx.num_of_local_experts = num_of_local_experts
        ctx.topk = topk
        return multihot_indices, probs_in_multihot

    @staticmethod
    def backward(ctx, grad_multihot_indices, grad_probs_in_multihot):
        """Backward function for IndicesToMultihot
        Convert multihot probs representation to indices.
        indices is ignored in the backward function.
        Args:
            grad_multihot_indices: [num_of_tokens, num_of_local_experts]
            grad_probs_in_multihot: [num_of_tokens, num_of_local_experts]
        Returns:
            grad_probs_indices: [num_of_tokens, topk]
        """
        position_map = ctx.saved_tensors[0]
        num_of_tokens = ctx.num_of_tokens
        num_of_local_experts = ctx.num_of_local_experts
        topk = ctx.topk

        # Initialize the gradient of the indices and probs_indices
        grad_probs_indices = torch.empty(
            (num_of_tokens, topk), dtype=grad_probs_in_multihot.dtype, device="cuda"
        )
        # Compute the next power of 2 for the topk and num_of_local_experts
        topk_next_power_of_2 = 2 ** int(math.ceil(math.log2(topk)))
        num_of_local_experts_next_power_of_2 = 2 ** int(
            math.ceil(math.log2(num_of_local_experts))
        )

        grid = (num_of_tokens,)
        _multihot_to_indices_kernel[grid](
            # if the grad_probs_in_multihot is all-one/all-zero,
            # overlapping stride will cause error without contiguous()
            grad_probs_in_multihot.contiguous(),
            position_map,
            grad_probs_indices,
            num_of_local_experts,
            num_of_local_experts_next_power_of_2,
            topk,
            topk_next_power_of_2,
            BLOCK_SIZE=32,  # use only 1 warp per block
            num_warps=1,
        )
        return None, grad_probs_indices, None, None


def fused_indices_to_multihot(indices, probs_indices, num_of_local_experts):
    """Convert moe topk indices to multihot representation.
    This function is an experimental feature and may change in future versions.
    """
    return IndicesToMultihot.apply(indices, probs_indices, num_of_local_experts)


# =============================================================================
# NEW: Padded version for grouped_mm alignment (fixes empty expert gradient bug)
# =============================================================================

# Minimum group size / alignment in M (rows) for grouped_mm.
# For bf16/fp16 we need 16 byte alignment → 8 elements.
ALIGNMENT_M = 8


@triton.jit
def _indices_to_multihot_with_counts_kernel(
    indices_ptr,
    probs_in_indices_ptr,
    multihot_indices_ptr,  # bool
    probs_in_multihot_ptr,
    position_map_ptr,
    tokens_per_expert_ptr,  # int32 [num_of_local_experts]
    num_of_local_experts: tl.constexpr,
    num_of_local_experts_next_power_of_2: tl.constexpr,
    topk: tl.constexpr,
    topk_next_power_of_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for converting indices to multihot representation
    with per-expert token counting via atomic_add.

    Input:
        indices: [num_of_tokens, topk]
        probs_in_indices: [num_of_tokens, topk]
    Output:
        multihot_indices: [num_of_tokens, num_of_local_experts]
        probs_in_multihot: [num_of_tokens, num_of_local_experts]
        position_map: [num_of_tokens, num_of_local_experts]
        tokens_per_expert: [num_of_local_experts] (via atomic_add)
    """
    # Prepare the [0, topk) row
    topk_row = tl.arange(0, topk_next_power_of_2)
    topk_row = tl.where(topk_row < topk, topk_row, -1)
    topk_row_mask = topk_row != -1

    # Prepare the [0, num_of_local_experts) row
    num_exp_row = tl.arange(0, num_of_local_experts_next_power_of_2)
    num_exp_row = tl.where(num_exp_row < num_of_local_experts, num_exp_row, -1)
    num_exp_row_mask = num_exp_row != -1

    # Load a [1, topk] row from the indices buffer
    row_idx = tl.program_id(0)
    indices_row = tl.load(
        indices_ptr + row_idx * topk + topk_row,
        mask=topk_row_mask,
    )
    indices_row = tl.where(topk_row_mask, indices_row, -1)

    probs_row = tl.load(
        probs_in_indices_ptr + row_idx * topk + topk_row,
        mask=topk_row_mask,
    )

    # Get the position of each index in the indices_row, which is saved for backwards
    position_row = tl.where(indices_row != -1, topk_row, -1)

    # Mask of the valid indices
    mask = (indices_row != -1) & (indices_row < num_of_local_experts)

    row_idx_offset = row_idx * num_of_local_experts

    # Store to initialize this token's row
    tl.store(
        multihot_indices_ptr + row_idx_offset + num_exp_row,
        0,
        mask=num_exp_row_mask,
    )
    tl.store(
        probs_in_multihot_ptr + row_idx_offset + num_exp_row,
        0,
        mask=num_exp_row_mask,
    )
    tl.store(
        position_map_ptr + row_idx_offset + num_exp_row,
        -1,
        mask=num_exp_row_mask,
    )
    # Use barrier to make sure the initialization is done
    tl.debug_barrier()

    # Scatter the indices and probs_in_indices to their expert positions
    tl.store(
        multihot_indices_ptr + row_idx_offset + indices_row,
        1,
        mask=mask,
    )
    tl.store(
        probs_in_multihot_ptr + row_idx_offset + indices_row,
        probs_row,
        mask=mask,
    )
    tl.store(
        position_map_ptr + row_idx_offset + indices_row,
        position_row,
        mask=mask,
    )

    # Accumulate per-expert token counts.
    # tokens_per_expert_ptr is [num_of_local_experts], zero-initialized.
    tl.atomic_add(tokens_per_expert_ptr + indices_row, 1, mask=mask)


class IndicesToMultihotWithPadding(torch.autograd.Function):
    """Convert moe topk indices to multihot representation with 8-alignment metadata.

    This version computes per-expert token counts inside the kernel (GPU-side, no extra pass)
    and returns 8-aligned sizes for grouped_mm compatibility.

    This fixes the empty expert gradient bug where torch._grouped_mm produces garbage
    gradients for experts with 0 tokens.
    """

    @staticmethod
    def forward(ctx, indices, probs_indices, num_of_local_experts):
        """Forward function for IndicesToMultihotWithPadding.

        Args:
            indices: [num_of_tokens, topk]
            probs_indices: [num_of_tokens, topk]
            num_of_local_experts: int

        Returns:
            multihot_indices: [num_of_tokens, num_of_local_experts] (bool)
            probs_in_multihot: [num_of_tokens, num_of_local_experts]
            tokens_per_expert: [num_of_local_experts] (int32, raw counts)
            m_sizes: [num_of_local_experts] (int32, >= ALIGNMENT_M and aligned)
            m_offsets: [num_of_local_experts] (int32, cumsum of m_sizes)
        """
        num_of_tokens = indices.shape[0]
        assert (
            indices.shape == probs_indices.shape
        ), "indices and probs_indices must have the same shape"
        topk = indices.shape[1]

        device = indices.device

        multihot_indices = torch.empty(
            (num_of_tokens, num_of_local_experts),
            dtype=torch.bool,
            device=device,
        )
        probs_in_multihot = torch.empty(
            (num_of_tokens, num_of_local_experts),
            dtype=probs_indices.dtype,
            device=device,
        )
        position_map = torch.empty(
            (num_of_tokens, num_of_local_experts),
            dtype=torch.int32,
            device=device,
        )

        # Per-expert token counts (accumulated atomically in the kernel)
        tokens_per_expert = torch.zeros(
            (num_of_local_experts,),
            dtype=torch.int32,
            device=device,
        )

        # Compute the next power of 2 for the topk and num_of_local_experts
        topk_next_power_of_2 = 2 ** int(math.ceil(math.log2(topk)))
        num_of_local_experts_next_power_of_2 = 2 ** int(
            math.ceil(math.log2(num_of_local_experts))
        )
        grid = (num_of_tokens,)

        _indices_to_multihot_with_counts_kernel[grid](
            indices,
            probs_indices,
            multihot_indices,
            probs_in_multihot,
            position_map,
            tokens_per_expert,
            num_of_local_experts,
            num_of_local_experts_next_power_of_2,
            topk,
            topk_next_power_of_2,
            BLOCK_SIZE=32,  # use only 1 warp per block
            num_warps=1,
        )

        # ------------------------------------------------------------------
        # 8-padding / alignment metadata for grouped_mm, all on GPU
        # ------------------------------------------------------------------
        alignment = ALIGNMENT_M

        # 1. clamp empty experts up to alignment (avoid 0-row GEMMs)
        tokens_per_expert_clamped = torch.clamp_min(tokens_per_expert, alignment)

        # 2. round up to multiple of alignment (CUTLASS/TMA requirement)
        m_sizes = (
            (tokens_per_expert_clamped + alignment - 1) // alignment * alignment
        ).to(torch.int32)

        # 3. prefix sum for flattened padded layout (optional but useful)
        m_offsets = torch.cumsum(m_sizes, dim=0).to(torch.int32)

        # Save for backward (only need position_map + metadata)
        ctx.save_for_backward(position_map)
        ctx.num_of_tokens = num_of_tokens
        ctx.num_of_local_experts = num_of_local_experts
        ctx.topk = topk

        # Extra outputs (tokens_per_expert, m_sizes, m_offsets) are
        # treated as non-differentiable by autograd.
        return (
            multihot_indices,
            probs_in_multihot,
            tokens_per_expert,
            m_sizes,
            m_offsets,
        )

    @staticmethod
    def backward(
        ctx,
        grad_multihot_indices,
        grad_probs_in_multihot,
        grad_tokens_per_expert=None,
        grad_m_sizes=None,
        grad_m_offsets=None,
    ):
        """Backward function for IndicesToMultihotWithPadding.

        Args:
            grad_multihot_indices: [num_of_tokens, num_of_local_experts]
            grad_probs_in_multihot: [num_of_tokens, num_of_local_experts]

        Returns:
            grad_indices: None
            grad_probs_indices: [num_of_tokens, topk]
            grad_num_of_local_experts: None
        """
        (position_map,) = ctx.saved_tensors
        num_of_tokens = ctx.num_of_tokens
        num_of_local_experts = ctx.num_of_local_experts
        topk = ctx.topk

        # Initialize the gradient of probs_indices
        grad_probs_indices = torch.empty(
            (num_of_tokens, topk),
            dtype=grad_probs_in_multihot.dtype,
            device=grad_probs_in_multihot.device,
        )

        # Compute the next power of 2 for the topk and num_of_local_experts
        topk_next_power_of_2 = 2 ** int(math.ceil(math.log2(topk)))
        num_of_local_experts_next_power_of_2 = 2 ** int(
            math.ceil(math.log2(num_of_local_experts))
        )

        grid = (num_of_tokens,)
        _multihot_to_indices_kernel[grid](
            # if the grad_probs_in_multihot is all-one/all-zero,
            # overlapping stride will cause error without contiguous()
            grad_probs_in_multihot.contiguous(),
            position_map,
            grad_probs_indices,
            num_of_local_experts,
            num_of_local_experts_next_power_of_2,
            topk,
            topk_next_power_of_2,
            BLOCK_SIZE=32,  # use only 1 warp per block
            num_warps=1,
        )

        # Gradients:
        #  - indices: None (non-differentiable)
        #  - probs_indices: grad_probs_indices
        #  - num_of_local_experts: None
        return None, grad_probs_indices, None


def fused_indices_to_multihot_with_padding(
    indices, probs_indices, num_of_local_experts
):
    """Convert moe topk indices to multihot representation with 8-alignment metadata.

    This version fixes the empty expert gradient bug by:
    1. Computing per-expert token counts inside the Triton kernel (no extra pass)
    2. Returning 8-aligned sizes (m_sizes) for grouped_mm compatibility
    3. All computation on GPU with no D2H sync

    Returns:
        multihot_indices:  [num_tokens, num_experts] (bool)
        probs_in_multihot: [num_tokens, num_experts]
        tokens_per_expert: [num_experts] (int32, raw counts)
        m_sizes:           [num_experts] (int32, >= ALIGNMENT_M, aligned to 8)
        m_offsets:         [num_experts] (int32, cumsum of m_sizes)
    """
    return IndicesToMultihotWithPadding.apply(
        indices, probs_indices, num_of_local_experts
    )
