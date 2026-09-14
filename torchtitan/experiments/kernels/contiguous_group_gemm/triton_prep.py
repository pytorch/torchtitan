# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _token_expand_kernel(
    # Input pointers
    tokens_ptr,  # [batch_size * seq_len, hidden_dim]
    top_k_indices_ptr,  # [batch_size * seq_len, top_k]
    top_k_probs_ptr,  # [batch_size * seq_len, top_k]
    # Output pointers
    expanded_tokens_ptr,  # [M_total, hidden_dim]
    token_expert_indices_ptr,  # [M_total]
    token_weights_ptr,  # [M_total]
    original_indices_ptr,  # [M_total]
    # Dimensions
    total_original_tokens,  # batch_size * seq_len
    hidden_dim,  # Token hidden dimension
    top_k: tl.constexpr,  # Number of experts per token
    BLOCK_SIZE_H: tl.constexpr,  # Block size for hidden dimension
):
    """
    Kernel to expand tokens for top-k routing and populate the initial expanded arrays.
    This kernel copies tokens in blocks to handle arbitrary hidden dimensions.
    """
    pid = tl.program_id(0)  # token index

    # Only process if within bounds
    if pid < total_original_tokens:
        # Calculate base indices in expanded arrays
        base_expanded_idx = pid * top_k

        # Process each of the top-k experts for this token
        for k in range(top_k):
            # Calculate the expanded index
            expanded_idx = base_expanded_idx + k

            # Load expert index and probability for this (token, expert) pair
            expert_idx = tl.load(top_k_indices_ptr + pid * top_k + k)
            prob = tl.load(top_k_probs_ptr + pid * top_k + k)

            # Store metadata
            tl.store(token_expert_indices_ptr + expanded_idx, expert_idx)
            tl.store(token_weights_ptr + expanded_idx, prob)
            tl.store(original_indices_ptr + expanded_idx, pid)

            # Copy token data in blocks of size BLOCK_SIZE_H
            for h_start in range(0, hidden_dim, BLOCK_SIZE_H):
                # Calculate actual block size (to handle non-multiple hidden dims)
                h_end = min(h_start + BLOCK_SIZE_H, hidden_dim)
                h_size = h_end - h_start

                # Use tl.arange with a fixed size and mask for the valid portion
                h_offsets = tl.arange(0, BLOCK_SIZE_H)
                h_mask = h_offsets < h_size

                # Load token block with mask
                token_ptrs = tokens_ptr + pid * hidden_dim + h_start + h_offsets
                token_block = tl.load(token_ptrs, mask=h_mask, other=0.0)

                # Store token block with mask
                expanded_tokens_ptrs = (
                    expanded_tokens_ptr
                    + expanded_idx * hidden_dim
                    + h_start
                    + h_offsets
                )
                tl.store(expanded_tokens_ptrs, token_block, mask=h_mask)


@triton.jit
def _expert_counters_kernel(
    # Input pointers
    token_expert_indices_ptr,  # [M_total]
    # Output pointers
    expert_counters_ptr,  # [num_experts]
    # Dimensions
    M_total,  # Total number of expanded tokens
    num_experts: tl.constexpr,  # Number of experts
    BLOCK_SIZE: tl.constexpr,  # Block size for processing
):
    """
    Kernel to count tokens per expert using atomic operations.
    """
    pid = tl.program_id(0)  # Block index
    block_start = pid * BLOCK_SIZE

    # Calculate offsets within this block
    offsets = tl.arange(0, BLOCK_SIZE)
    token_idx = block_start + offsets

    # Create mask for valid tokens
    mask = token_idx < M_total

    # Load expert indices for this block
    expert_indices = tl.load(token_expert_indices_ptr + token_idx, mask=mask, other=0)

    # Process each token in parallel using a mask
    for e in range(num_experts):
        # Count tokens assigned to this expert
        is_expert_e = expert_indices == e
        # Only count valid tokens (within bounds)
        valid_expert_e = is_expert_e & mask
        # Count how many tokens match this expert
        count = tl.sum(valid_expert_e)
        # Atomically add the count to the counter
        if count > 0:
            tl.atomic_add(expert_counters_ptr + e, count)


def prepare_tokens_triton(
    tokens: torch.Tensor,  # [batch_size, seq_len, hidden_dim]
    router_logits: torch.Tensor,  # [batch_size, seq_len, num_experts]
    top_k: int = 6,  # Number of experts per token
    group_size_m: int = 128,  # Size of contiguous token blocks
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    Accelerated version of prepare_tokens_for_cg_gemm_topk using Triton kernels.
    hybrid approach with computationally expensive parts in Triton kernels
    and complex control flow in PyTorch.

    Args:
        tokens: Input token embeddings [batch_size, seq_len, hidden_dim]
        router_logits: Router logits [batch_size, seq_len, num_experts]
        top_k: Number of experts per token (default: 6)
        group_size_m: Size of contiguous token blocks (default: 128)

    Returns:
        Same as prepare_tokens_for_cg_gemm_topk:
        - padded_tokens: Tokens arranged in contiguous blocks by expert [total_padded_tokens, hidden_dim]
        - expanded_expert_indices: Expert indices [total_padded_tokens]
        - padded_weights: Token weights [total_padded_tokens]
        - metadata: Dictionary with metadata for output reconstruction
    """
    device = tokens.device
    dtype = tokens.dtype
    batch_size, seq_len, hidden_dim = tokens.shape
    _, _, num_experts = router_logits.shape

    # Get top-k experts and their probabilities for each token
    router_probs = torch.softmax(router_logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(router_probs, k=top_k, dim=-1)

    # Normalize the top-k probabilities
    top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

    # Flatten batch and sequence dimensions
    flat_tokens = tokens.reshape(-1, hidden_dim)  # [batch_size*seq_len, hidden_dim]
    flat_top_k_indices = top_k_indices.reshape(-1, top_k)  # [batch_size*seq_len, top_k]
    flat_top_k_probs = top_k_probs.reshape(-1, top_k)  # [batch_size*seq_len, top_k]

    total_original_tokens = flat_tokens.shape[0]
    M_total = total_original_tokens * top_k

    # Create arrays to hold expanded tokens and their metadata
    expanded_tokens = torch.zeros((M_total, hidden_dim), device=device, dtype=dtype)
    token_expert_indices = torch.zeros(M_total, dtype=torch.int64, device=device)
    token_weights = torch.zeros(M_total, device=device)
    original_indices = torch.zeros(M_total, dtype=torch.int64, device=device)

    # Determine block sizes - needs to be power of 2 for Triton
    # Find the largest power of 2 less than or equal to hidden_dim, capped at 128
    max_block_h = min(128, hidden_dim)
    block_size_h = 2 ** int(torch.log2(torch.tensor(max_block_h)).item())

    # Don't use meta reference for token expand kernel
    # Launch kernel to expand tokens
    n_tokens = total_original_tokens
    grid = (n_tokens,)
    _token_expand_kernel[grid](
        flat_tokens,
        flat_top_k_indices,
        flat_top_k_probs,
        expanded_tokens,
        token_expert_indices,
        token_weights,
        original_indices,
        total_original_tokens,
        hidden_dim,
        top_k,
        block_size_h,
    )

    # Count tokens assigned to each expert
    expert_counts = torch.zeros(num_experts, dtype=torch.int64, device=device)
    block_size = 128  # Processing block size (power of 2)

    # Calculate number of blocks needed
    n_blocks = triton.cdiv(M_total, block_size)
    grid = (n_blocks,)
    _expert_counters_kernel[grid](
        token_expert_indices, expert_counts, M_total, num_experts, block_size
    )

    # Calculate padding needed for each expert to reach a multiple of group_size_m
    padded_expert_counts = (
        torch.ceil(expert_counts.float() / group_size_m) * group_size_m
    ).to(torch.int64)
    total_padded_tokens = padded_expert_counts.sum().item()

    # Create padded arrays
    padded_tokens = torch.zeros(
        (total_padded_tokens, hidden_dim), device=device, dtype=dtype
    )
    padded_weights = torch.zeros(total_padded_tokens, device=device)
    padded_original_indices = (
        torch.ones(total_padded_tokens, dtype=torch.int64, device=device) * -1
    )  # -1 indicates padding
    expanded_expert_indices = torch.zeros(
        total_padded_tokens, dtype=torch.int32, device=device
    )

    # For the  regrouping and padding logic, we use PyTorch operations

    # Step 1: Sort tokens by expert
    sorted_indices = torch.argsort(token_expert_indices)
    sorted_tokens = expanded_tokens[sorted_indices]
    sorted_expert_indices = token_expert_indices[sorted_indices]
    sorted_weights = token_weights[sorted_indices]
    sorted_original_indices = original_indices[sorted_indices]

    # Step 2: Create the final padded layout
    current_pos = 0
    for e in range(num_experts):
        expert_count = expert_counts[e].item()
        padded_count = padded_expert_counts[e].item()

        if expert_count > 0:
            # Find all tokens assigned to this expert
            expert_mask = sorted_expert_indices == e
            expert_indices = torch.nonzero(expert_mask).squeeze(1)

            # Copy actual tokens to padded arrays
            padded_tokens[current_pos : current_pos + expert_count] = sorted_tokens[
                expert_indices
            ]
            padded_weights[current_pos : current_pos + expert_count] = sorted_weights[
                expert_indices
            ]
            padded_original_indices[current_pos : current_pos + expert_count] = (
                sorted_original_indices[expert_indices]
            )

        # Fill expert indices for all tokens in this expert's groups (including padding)
        expanded_expert_indices[current_pos : current_pos + padded_count] = e

        # Move to next position accounting for padding
        current_pos += padded_count

    # Prepare metadata for output reconstruction
    metadata = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
        "top_k": top_k,
        "original_indices": padded_original_indices,
        "num_original_tokens": total_original_tokens,
    }

    return padded_tokens, expanded_expert_indices, padded_weights, metadata
