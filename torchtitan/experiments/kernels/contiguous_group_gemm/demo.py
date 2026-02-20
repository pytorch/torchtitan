# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from cg_forward import cg_grouped_gemm_forward
from triton_prep import prepare_tokens_triton


def prepare_tokens_for_cg_gemm_topk(
    tokens: torch.Tensor,  # [batch_size, seq_len, hidden_dim]
    router_logits: torch.Tensor,  # [batch_size, seq_len, num_experts]
    top_k: int = 6,  # Number of experts per token
    group_size_m: int = 128,  # Size of contiguous token blocks
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    Prepare tokens for contiguous grouped GEMM with top-k routing.
    This function processes tokens to be used with the _kernel_cg_forward_aligned kernel.

    For each token, we select the top-k experts, and duplicate the token k times in the output.
    Tokens are organized in contiguous blocks for each expert to enable efficient GEMM.

    Args:
        tokens: Input token embeddings [batch_size, seq_len, hidden_dim]
        router_logits: Router logits [batch_size, seq_len, num_experts]
        top_k: Number of experts per token (default: 6)
        group_size_m: Size of contiguous token blocks (default: 128)

    Returns:
        Tuple of (
            expanded_tokens: [M_total, hidden_dim] where M_total = batch_size * seq_len * top_k,
            expert_indices: [M_total] expanded indices matching each token to its expert,
            token_weights: [M_total] weights for each token-expert combination,
            metadata: Dictionary with metadata for restoring the original order
        )
    """
    device = tokens.device
    dtype = tokens.dtype
    batch_size, seq_len, hidden_dim = tokens.shape
    _, _, num_experts = router_logits.shape

    # Get top-k experts and their probabilities for each token
    router_probs = F.softmax(router_logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(router_probs, k=top_k, dim=-1)

    # Normalize the top-k probabilities
    top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

    # Flatten batch and sequence dimensions
    flat_tokens = tokens.reshape(-1, hidden_dim)  # [batch_size*seq_len, hidden_dim]
    flat_top_k_indices = top_k_indices.reshape(-1, top_k)  # [batch_size*seq_len, top_k]
    flat_top_k_probs = top_k_probs.reshape(-1, top_k)  # [batch_size*seq_len, top_k]

    total_original_tokens = flat_tokens.shape[0]

    # We'll create expanded tokens where each original token appears k times, once for each expert
    M_total = total_original_tokens * top_k

    # Create arrays to hold expanded tokens and their metadata
    expanded_tokens = torch.zeros((M_total, hidden_dim), device=device, dtype=dtype)
    token_expert_indices = torch.zeros(M_total, dtype=torch.int64, device=device)
    token_weights = torch.zeros(M_total, device=device)
    original_indices = torch.zeros(M_total, dtype=torch.int64, device=device)

    # Fill in the expanded arrays
    for i in range(total_original_tokens):
        for j in range(top_k):
            idx = i * top_k + j
            expanded_tokens[idx] = flat_tokens[i]
            token_expert_indices[idx] = flat_top_k_indices[i, j]
            token_weights[idx] = flat_top_k_probs[i, j]
            original_indices[idx] = i

    # Step 1: Sort all tokens by their expert assignment
    # This will group tokens destined for the same expert together
    sorted_indices = torch.argsort(token_expert_indices)

    # Reorder all arrays according to this sorting
    sorted_tokens = expanded_tokens[sorted_indices]
    sorted_expert_indices = token_expert_indices[sorted_indices]
    sorted_weights = token_weights[sorted_indices]
    sorted_original_indices = original_indices[sorted_indices]

    # Step 2: Ensure all groups have a size that's a multiple of group_size_m
    # Count tokens assigned to each expert
    expert_counts = torch.zeros(num_experts, dtype=torch.int64, device=device)
    for e in range(num_experts):
        expert_counts[e] = torch.sum(sorted_expert_indices == e)

    # Calculate padding needed for each expert to reach a multiple of group_size_m
    padded_expert_counts = (
        torch.ceil(expert_counts.float() / group_size_m) * group_size_m
    )
    padded_expert_counts = padded_expert_counts.to(torch.int64)

    # Create padded arrays
    total_padded_tokens = padded_expert_counts.sum().item()
    padded_tokens = torch.zeros(
        (total_padded_tokens, hidden_dim), device=device, dtype=dtype
    )
    padded_weights = torch.zeros(total_padded_tokens, device=device)
    padded_original_indices = (
        torch.ones(total_padded_tokens, dtype=torch.int64, device=device) * -1
    )  # -1 indicates padding

    # Create expanded expert indices with shape [M_total]
    # This is different from the original code which created [num_groups]
    expanded_expert_indices = torch.zeros(
        total_padded_tokens, dtype=torch.int32, device=device
    )

    # Fill in the padded arrays
    current_pos = 0
    next_pos = 0
    for e in range(num_experts):
        expert_count = expert_counts[e].item()
        padded_count = padded_expert_counts[e].item()

        next_pos = current_pos + expert_count

        # Copy actual tokens
        if expert_count > 0:
            expert_mask = sorted_expert_indices == e
            expert_indices = torch.nonzero(expert_mask).squeeze(1)

            padded_tokens[current_pos : current_pos + expert_count] = sorted_tokens[
                expert_indices
            ]
            padded_weights[current_pos : current_pos + expert_count] = sorted_weights[
                expert_indices
            ]
            padded_original_indices[current_pos : current_pos + expert_count] = (
                sorted_original_indices[expert_indices]
            )

            # Fill expert indices for all tokens in this expert's groups
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


def restore_output_from_cg_gemm_topk(
    output: torch.Tensor,  # [M_total, hidden_dim]
    weights: torch.Tensor,  # [M_total]
    metadata: Dict,  # Metadata from preparation
) -> torch.Tensor:
    """
    Restore the output from contiguous grouped GEMM to original token order.

    Args:
        output: Output tensor from CG GEMM [M_total, hidden_dim]
        weights: Token-expert weights [M_total]
        metadata: Metadata from the preparation function

    Returns:
        Reconstructed output [batch_size, seq_len, hidden_dim]
    """
    batch_size = metadata["batch_size"]
    seq_len = metadata["seq_len"]
    hidden_dim = metadata["hidden_dim"]
    top_k = metadata["top_k"]
    original_indices = metadata["original_indices"]
    num_original_tokens = metadata["num_original_tokens"]

    device = output.device
    dtype = output.dtype

    # Initialize accumulator for final output
    final_output = torch.zeros(
        (num_original_tokens, hidden_dim), device=device, dtype=dtype
    )
    weight_accumulator = torch.zeros(num_original_tokens, device=device)

    # Apply weights to output
    weighted_output = output * weights.unsqueeze(1)

    # Accumulate results for each original token
    valid_mask = original_indices >= 0
    valid_indices = original_indices[valid_mask]
    valid_outputs = weighted_output[valid_mask]
    valid_weights = weights[valid_mask]

    # Use scatter_add to accumulate outputs for each original token
    index_tensor = valid_indices.unsqueeze(1).expand(-1, hidden_dim)
    final_output.scatter_add_(0, index_tensor, valid_outputs)
    weight_accumulator.scatter_add_(0, valid_indices, valid_weights)

    # Ensure no division by zero
    weight_accumulator = torch.clamp(weight_accumulator, min=1e-10)

    # Normalize by accumulated weights
    final_output = final_output / weight_accumulator.unsqueeze(1)

    # Reshape to original dimensions
    final_output = final_output.reshape(batch_size, seq_len, hidden_dim)

    return final_output


def example_moe_with_cg_gemm_topk(
    tokens: torch.Tensor,  # [batch_size, seq_len, hidden_dim]
    router: torch.nn.Linear,  # Router network
    expert_weights: torch.Tensor,  # [num_experts, output_dim, hidden_dim]
    top_k: int = 6,  # Number of experts per token
    group_size_m: int = 128,  # Size of contiguous blocks
    cg_forward_fn: callable = None,  # Function that implements the CG GEMM forward pass
) -> torch.Tensor:
    """
    Example of using contiguous grouped GEMM with top-k routing.

    Args:
        tokens: Input token embeddings [batch_size, seq_len, hidden_dim]
        router: Router network (Linear layer)
        expert_weights: Expert weights [num_experts, output_dim, hidden_dim]
        top_k: Number of experts per token (default: 6)
        group_size_m: Size of contiguous blocks (default: 128)
        cg_forward_fn: Function that implements the CG GEMM forward pass

    Returns:
        Output tensor [batch_size, seq_len, hidden_dim]
    """
    if cg_forward_fn is None:
        raise ValueError(
            "Must provide a function that implements the CG GEMM forward pass"
        )

    # Get routing logits
    router_logits = router(tokens)  # [batch_size, seq_len, num_experts]

    # Prepare tokens for contiguous grouped GEMM
    # expanded_tokens_ref, expert_indices_ref, token_weights_ref, metadata_ref = (
    #    prepare_tokens_for_cg_gemm_topk(
    #        tokens, router_logits, top_k=top_k, group_size_m=group_size_m
    #    )
    # )

    expanded_tokens, expert_indices, token_weights, metadata = (
        # prepare_tokens_for_cg_gemm_topk(  #
        prepare_tokens_triton(
            tokens, router_logits, top_k=top_k, group_size_m=group_size_m
        )
    )

    """assert (
        expanded_tokens.shape == expanded_tokens_ref.shape
    ), f"{expanded_tokens.shape} vs {expanded_tokens_ref.shape}"
    assert (
        expert_indices.shape == expert_indices_ref.shape
    ), f"{expert_indices.shape} vs {expert_indices_ref.shape}"
    assert (
        token_weights.shape == token_weights_ref.shape
    ), f"{token_weights.shape} vs {token_weights_ref.shape}"
    assert metadata == metadata_ref
    """
    # Run contiguous grouped GEMM
    output = cg_forward_fn(
        expanded_tokens, expert_weights, expert_indices, group_size_m=group_size_m
    )
    print(f"CG GEMM output shape: {output.shape}")
    print(f"CG GEMM output dtype: {output.dtype}")

    # Restore original token order
    from triton_restore import restore_output_triton

    # final_output2 = restore_output_triton(output, token_weights, metadata)
    final_output = restore_output_from_cg_gemm_topk(output, token_weights, metadata)

    # assert torch.allclose(final_output, final_output2, atol=1e-3, rtol=1e-3)

    return final_output


def demo_usage():
    """
    Demonstration of how to use the data preparation utilities with a CG GEMM implementation.

    """
    # Toggle with actual implementation

    def pytorch_cg_gemm_forward(
        inputs, expert_weights, expert_indices, group_size_m=128
    ):
        """Placeholder for actual CG GEMM implementation"""
        M_total, hidden_dim = inputs.shape
        num_experts, output_dim, _ = expert_weights.shape

        print(
            f"CG GEMM with M_total={M_total}, hidden_dim={hidden_dim}, num_experts={num_experts}"
        )
        print(f"Number of groups: {expert_indices.shape[0]}")

        # Simulate kernel by just doing a simple matmul for each expert
        output = torch.zeros(
            (M_total, output_dim), device=inputs.device, dtype=inputs.dtype
        )

        for g in range(expert_indices.shape[0]):
            start_idx = g * group_size_m
            end_idx = start_idx + group_size_m
            expert_idx = expert_indices[g].item()

            # Simple matmul instead of optimized kernel
            group_inputs = inputs[start_idx:end_idx]
            group_output = torch.matmul(group_inputs, expert_weights[expert_idx].t())
            output[start_idx:end_idx] = group_output

        return output

    # Parameters
    batch_size = 2
    seq_len = 1024
    hidden_dim = 768
    num_experts = 8
    top_k = 6
    group_size_m = 128
    device = "cuda"  # if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # Create sample inputs
    tokens = torch.randn((batch_size, seq_len, hidden_dim), device=device, dtype=dtype)
    router = torch.nn.Linear(hidden_dim, num_experts, device=device, dtype=dtype)
    expert_weights = torch.randn(
        (num_experts, hidden_dim, hidden_dim), device=device, dtype=dtype
    )

    # from flatcg import cg_grouped_gemm_forward_flat

    # Run example
    output = example_moe_with_cg_gemm_topk(
        tokens=tokens,
        router=router,
        expert_weights=expert_weights,
        top_k=top_k,
        group_size_m=group_size_m,
        cg_forward_fn=cg_grouped_gemm_forward,  # pytorch_cg_gemm_forward
    )

    print(f"Output shape: {output.shape}")

    return output


if __name__ == "__main__":
    demo_usage()
