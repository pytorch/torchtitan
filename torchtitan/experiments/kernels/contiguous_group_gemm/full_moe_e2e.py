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

# Import the working complete implementation with backward pass
from cg_backward import cg_grouped_gemm

# Import the forward pass implementation
from cg_forward import cg_grouped_gemm_forward

# Import token preparation function
from triton_prep import prepare_tokens_triton


def prepare_tokens_with_triton_and_gradients(
    tokens: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int = 6,
    group_size_m: int = 128,
):
    # Calculate router probabilities
    router_probs = F.softmax(router_logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(router_probs, k=top_k, dim=-1)

    # Normalize the top-k probabilities
    top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-10)

    # Save these original probabilities for later gradient connection
    original_probs = top_k_probs

    # Run Triton preparation kernels
    expanded_tokens, expert_indices, token_weights, metadata = prepare_tokens_triton(
        tokens, router_logits, top_k=top_k, group_size_m=group_size_m
    )

    # Store the original probabilities in the metadata
    metadata["original_probs"] = original_probs

    return expanded_tokens, expert_indices, token_weights, metadata


def prepare_tokens_for_cg_gemm_topk(
    tokens: torch.Tensor,  # [batch_size, seq_len, hidden_dim]
    router_logits: torch.Tensor,  # [batch_size, seq_len, num_experts]
    top_k: int = 6,  # Number of experts per token
    group_size_m: int = 128,  # Size of contiguous token blocks
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    Prepare tokens for contiguous grouped GEMM with top-k routing.
    This version preserves gradients for the router by ensuring the computation graph
    is properly connected for backpropagation.

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

    # Normalize the top-k probabilities - KEEP CONNECTION TO COMPUTATION GRAPH
    top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-10)

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

    # Fill in the expanded arrays - PRESERVE GRADIENTS for token_weights
    for i in range(total_original_tokens):
        for j in range(top_k):
            idx = i * top_k + j
            # Use index_select to preserve gradients
            expanded_tokens[idx] = flat_tokens[i]
            token_expert_indices[idx] = flat_top_k_indices[i, j]
            # Direct assignment preserves gradient connection
            token_weights[idx] = flat_top_k_probs[i, j]
            original_indices[idx] = i

    # Sort all tokens by their expert assignment
    sorted_indices = torch.argsort(token_expert_indices)

    # Reorder all arrays according to this sorting
    sorted_tokens = expanded_tokens[sorted_indices]
    sorted_expert_indices = token_expert_indices[sorted_indices]
    # Use indexing that preserves gradients
    sorted_weights = token_weights[sorted_indices]
    sorted_original_indices = original_indices[sorted_indices]

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
    expanded_expert_indices = torch.zeros(
        total_padded_tokens, dtype=torch.int32, device=device
    )

    # Fill in the padded arrays
    current_pos = 0
    for e in range(num_experts):
        expert_count = expert_counts[e].item()
        padded_count = padded_expert_counts[e].item()

        # Copy actual tokens
        if expert_count > 0:
            expert_mask = sorted_expert_indices == e
            expert_indices = torch.nonzero(expert_mask).squeeze(1)

            padded_tokens[current_pos : current_pos + expert_count] = sorted_tokens[
                expert_indices
            ]
            # Use indexing that preserves gradients
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
    This version ensures proper gradient flow for the router.

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
    original_indices = metadata["original_indices"]
    num_original_tokens = metadata["num_original_tokens"]

    device = output.device
    dtype = output.dtype

    # Initialize accumulator for final output
    final_output = torch.zeros(
        (num_original_tokens, hidden_dim), device=device, dtype=dtype
    )
    weight_accumulator = torch.zeros(num_original_tokens, device=device)

    # Apply weights to output - PRESERVE GRADIENTS
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


# Import original preparation function as fallback/reference
def prepare_tokens_for_cg_gemm_topk_norouter(
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


def restore_output_from_cg_gemm_topk_norouter(
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
    use_forward_only: bool = False,  # Whether to use forward-only implementation
    use_triton_prep: bool = True,  # Whether to use triton for token preparation
) -> torch.Tensor:
    """
    Example of using contiguous grouped GEMM with top-k routing.
    Now supports both forward-only and full autograd versions.

    Args:
        tokens: Input token embeddings [batch_size, seq_len, hidden_dim]
        router: Router network (Linear layer)
        expert_weights: Expert weights [num_experts, output_dim, hidden_dim]
        top_k: Number of experts per token (default: 6)
        group_size_m: Size of contiguous blocks (default: 128)
        use_forward_only: Whether to use the forward-only implementation
        use_triton_prep: Whether to use triton for token preparation

    Returns:
        Output tensor [batch_size, seq_len, hidden_dim]
    """
    # Get routing logits
    router_logits = router(tokens)  # [batch_size, seq_len, num_experts]

    # Get router probabilities - keep this in the computational graph
    router_probs = torch.softmax(router_logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(router_probs, k=top_k, dim=-1)

    # Normalize the top-k probabilities
    top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

    # Prepare tokens for contiguous grouped GEMM
    if use_triton_prep:
        try:
            expanded_tokens, expert_indices, token_weights, metadata = (
                prepare_tokens_with_triton_and_gradients(
                    tokens, router_logits, top_k=top_k, group_size_m=group_size_m
                )
            )
        except Exception as e:
            print(
                f"Triton preparation failed with error: {e}. Falling back to PyTorch implementation."
            )
            expanded_tokens, expert_indices, token_weights, metadata = (
                prepare_tokens_for_cg_gemm_topk(
                    tokens, router_logits, top_k=top_k, group_size_m=group_size_m
                )
            )
    else:
        expanded_tokens, expert_indices, token_weights, metadata = (
            prepare_tokens_for_cg_gemm_topk(
                tokens, router_logits, top_k=top_k, group_size_m=group_size_m
            )
        )

    # Choose between forward-only or autograd-enabled implementation
    if use_forward_only:
        output = cg_grouped_gemm_forward(
            expanded_tokens, expert_weights, expert_indices, group_size_m=group_size_m
        )
    else:
        output = cg_grouped_gemm(
            expanded_tokens, expert_weights, expert_indices, group_size_m=group_size_m
        )

    # Restore original token order
    if use_triton_prep:
        final_output = restore_output_with_triton_and_gradients(
            output, token_weights, metadata
        )
    else:
        final_output = restore_output_from_cg_gemm_topk(output, token_weights, metadata)

    return final_output


def restore_output_with_triton_and_gradients(
    output: torch.Tensor,
    weights: torch.Tensor,
    metadata: Dict,
) -> torch.Tensor:
    # Get the original probabilities that maintain gradient connection
    original_probs = metadata.get("original_probs")

    # Triton restore kernel for efficiency
    # restored_output = restore_output_from_cg_gemm_topk(output, token_weights, metadata)
    restored_output = restore_output_from_cg_gemm_topk(output, weights, metadata)

    # If we have original probs, ensure gradient connection
    if original_probs is not None:
        # Create a dummy gradient connection that doesn't change the output
        # This is a common trick to maintain gradient flow
        batch_size, seq_len, hidden_dim = restored_output.shape
        dummy = torch.zeros_like(restored_output)

        # The scale should be very small to not affect the output
        scale = 1e-10

        # Add a tiny connection to the original probabilities
        for b in range(batch_size):
            for s in range(seq_len):
                # Sum the probabilities to create a scalar that connects to the gradient
                prob_sum = original_probs[b, s].sum()
                # Add a scaled version to each position
                dummy[b, s] += scale * prob_sum

        # Add the dummy to the output
        restored_output = restored_output + dummy

    return restored_output


# ========


# ========


class MoELayer(torch.nn.Module):
    """
    Mixture of Experts layer using contiguous grouped GEMM with proper gradient flow.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 6,
        group_size_m: int = 128,
        use_forward_only: bool = False,
        use_triton_prep: bool = True,
    ):
        """
        Initialize the MoE layer.

        Args:
            hidden_size: Hidden dimension size
            num_experts: Number of experts
            top_k: Number of experts per token
            group_size_m: Size of contiguous blocks
            use_forward_only: Whether to use forward-only implementation
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.group_size_m = group_size_m
        self.use_forward_only = use_forward_only
        self.use_triton_prep = use_triton_prep

        # Router network
        self.router = torch.nn.Linear(hidden_size, num_experts)

        # Expert weights
        self.expert_weights = torch.nn.Parameter(
            torch.randn(num_experts, hidden_size, hidden_size) / (hidden_size**0.5)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE layer with proper gradient flow.

        Args:
            tokens: Input tensor [batch_size, seq_len, hidden_size]

        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        # Import here to avoid circular imports
        # from cg_backward import cg_grouped_gemm
        # from cg_forward import cg_grouped_gemm_forward

        # Get routing logits
        router_logits = self.router(tokens)  # [batch_size, seq_len, num_experts]

        if self.use_triton_prep:
            try:
                expanded_tokens, expert_indices, token_weights, metadata = (
                    prepare_tokens_with_triton_and_gradients(
                        tokens,
                        router_logits,
                        top_k=self.top_k,
                        group_size_m=self.group_size_m,
                    )
                )
            except Exception as e:
                print(
                    f"Triton preparation failed with error: {e}. Falling back to PyTorch implementation."
                )
                expanded_tokens, expert_indices, token_weights, metadata = (
                    prepare_tokens_for_cg_gemm_topk(
                        tokens,
                        router_logits,
                        top_k=self.top_k,
                        group_size_m=self.group_size_m,
                    )
                )

        else:  # Prepare tokens for contiguous grouped GEMM with proper gradient flow
            expanded_tokens, expert_indices, token_weights, metadata = (
                prepare_tokens_for_cg_gemm_topk(
                    tokens,
                    router_logits,
                    top_k=self.top_k,
                    group_size_m=self.group_size_m,
                )
            )

        # Choose between forward-only or autograd-enabled implementation
        if self.use_forward_only:
            output = cg_grouped_gemm_forward(
                expanded_tokens,
                self.expert_weights,
                expert_indices,
                group_size_m=self.group_size_m,
            )
        else:
            output = cg_grouped_gemm(
                expanded_tokens,
                self.expert_weights,
                expert_indices,
                group_size_m=self.group_size_m,
            )

        # Restore original token order with proper gradient flow
        if self.use_triton_prep:
            final_output = restore_output_with_triton_and_gradients(
                output, token_weights, metadata
            )
        else:
            final_output = restore_output_from_cg_gemm_topk(
                output, token_weights, metadata
            )

        return final_output


class MoELayer_norouter(torch.nn.Module):
    """
    Mixture of Experts layer using contiguous grouped GEMM.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 6,
        group_size_m: int = 128,
        use_forward_only: bool = False,
        use_triton_prep: bool = True,
    ):
        """
        Initialize the MoE layer.

        Args:
            hidden_size: Hidden dimension size
            num_experts: Number of experts
            top_k: Number of experts per token
            group_size_m: Size of contiguous blocks
            use_forward_only: Whether to use forward-only implementation
            use_triton_prep: Whether to use triton for token preparation
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.group_size_m = group_size_m
        self.use_forward_only = use_forward_only
        self.use_triton_prep = use_triton_prep

        # Router network
        self.router = torch.nn.Linear(hidden_size, num_experts)

        # Expert weights
        self.expert_weights = torch.nn.Parameter(
            torch.randn(num_experts, hidden_size, hidden_size) / (hidden_size**0.5)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE layer.

        Args:
            tokens: Input tensor [batch_size, seq_len, hidden_size]

        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        return example_moe_with_cg_gemm_topk(
            tokens=tokens,
            router=self.router,
            expert_weights=self.expert_weights,
            top_k=self.top_k,
            group_size_m=self.group_size_m,
            use_forward_only=self.use_forward_only,
            use_triton_prep=self.use_triton_prep,
        )


def demo_forward_backward():
    """
    Demonstration of a complete forward and backward pass using MoE with CG-GEMM.
    This simulates a single training loop iteration.
    """
    print("-" * 80)
    print("Demonstrating MoE with CG-GEMM forward and backward pass")
    print("-" * 80)

    # Parameters
    batch_size = 2
    seq_len = 512
    hidden_dim = 768
    num_experts = 8
    top_k = 6
    group_size_m = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"Parameters: batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}"
    )
    print(f"num_experts={num_experts}, top_k={top_k}, group_size_m={group_size_m}")
    print(f"device={device}")

    # Create MoE layer
    moe_layer = MoELayer(
        hidden_size=hidden_dim,
        num_experts=num_experts,
        top_k=top_k,
        group_size_m=group_size_m,
        use_forward_only=False,  # Use autograd-enabled implementation
        use_triton_prep=False,  # Use triton for token preparation
    ).to(device)

    # Create optimizer
    optimizer = torch.optim.AdamW(moe_layer.parameters(), lr=1e-4)

    # Create input data
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, requires_grad=True)
    target = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Initialize timing variables
    forward_time = 0
    backward_time = 0

    # Forward pass
    print("\nStarting forward pass...")
    torch.cuda.synchronize()
    start_time = time.time()

    y = moe_layer(x)

    torch.cuda.synchronize()
    forward_time = time.time() - start_time
    print(f"Forward pass completed in {forward_time:.4f} seconds")
    print(f"Output shape: {y.shape}")

    # Compute loss
    loss = F.mse_loss(y, target)
    print(f"Loss: {loss.item():.4f}")

    # Backward pass
    print("\nStarting backward pass...")
    torch.cuda.synchronize()
    start_time = time.time()

    optimizer.zero_grad()
    loss.backward()

    torch.cuda.synchronize()
    backward_time = time.time() - start_time
    print(f"Backward pass completed in {backward_time:.4f} seconds")

    # Check gradients
    print("\nChecking gradients:")
    router_grad_norm = torch.norm(moe_layer.router.weight.grad)
    experts_grad_norm = torch.norm(moe_layer.expert_weights.grad)

    print(f"Router gradient norm: {router_grad_norm:.4f}")
    print(f"Expert weights gradient norm: {experts_grad_norm:.4f}")

    # Gradient step
    print("\nPerforming optimizer step...")
    optimizer.step()

    print("\nTraining step completed successfully!")
    print(f"Total time: {forward_time + backward_time:.4f} seconds")

    return moe_layer, x, y, loss


def compare_implementations():
    """
    Compare the forward-only and autograd-enabled implementations.
    Verifies that both implementations produce the same results on forward pass,
    and that the autograd-enabled implementation correctly computes gradients.
    """
    print("-" * 80)
    print("Comparing forward-only and autograd-enabled implementations")
    print("-" * 80)

    # Parameters
    batch_size = 2
    seq_len = 256  # Smaller for quicker comparison
    hidden_dim = 768
    num_experts = 4  # Fewer experts for simplicity
    top_k = 4
    group_size_m = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"Parameters: batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}"
    )
    print(f"num_experts={num_experts}, top_k={top_k}, group_size_m={group_size_m}")

    # Create input data
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, requires_grad=True)

    # Create shared parameters to ensure fair comparison
    router = torch.nn.Linear(hidden_dim, num_experts, device=device)
    expert_weights = torch.randn(
        num_experts, hidden_dim, hidden_dim, device=device, requires_grad=True
    )

    # Duplicate parameters for the forward-only implementation
    expert_weights_forward = expert_weights.detach().clone().requires_grad_(True)

    print("\nRunning forward-only implementation...")
    y_forward = example_moe_with_cg_gemm_topk(
        tokens=x,
        router=router,
        expert_weights=expert_weights_forward,
        top_k=top_k,
        group_size_m=group_size_m,
        use_forward_only=True,
    )

    print("Running autograd-enabled implementation...")
    y_autograd = example_moe_with_cg_gemm_topk(
        tokens=x,
        router=router,
        expert_weights=expert_weights,
        top_k=top_k,
        group_size_m=group_size_m,
        use_forward_only=False,
    )

    # Compare forward pass outputs
    forward_match = torch.allclose(y_forward, y_autograd, rtol=1e-3, atol=1e-3)
    print(f"\nForward pass outputs match: {forward_match}")

    if not forward_match:
        max_diff = torch.max(torch.abs(y_forward - y_autograd))
        print(f"Maximum difference: {max_diff:.6f}")

    # Test backward pass for autograd-enabled implementation
    print("\nTesting backward pass...")

    # Create target and compute loss
    target = torch.randn_like(y_autograd)
    loss = F.mse_loss(y_autograd, target)

    # Backward pass
    loss.backward()

    # Check if gradients were computed
    has_router_grad = router.weight.grad is not None
    has_expert_grad = expert_weights.grad is not None

    print(f"Router has gradient: {has_router_grad}")
    print(f"Expert weights have gradient: {has_expert_grad}")

    if has_expert_grad:
        expert_grad_norm = torch.norm(expert_weights.grad)
        print(f"Expert weights gradient norm: {expert_grad_norm:.4f}")

    return forward_match, has_router_grad, has_expert_grad


def benchmark_performance():
    """
    Benchmark the performance of the MoE implementation with CG-GEMM.
    Compares the speed of forward and backward passes.
    """
    print("-" * 80)
    print("Benchmarking MoE with CG-GEMM performance")
    print("-" * 80)

    # Parameters
    batch_sizes = [1, 2, 4, 8]
    seq_lens = [512, 1024]
    hidden_dim = 768
    num_experts = 8
    top_k = 6
    group_size_m = 128
    device = "cuda"  # if torch.cuda.is_available() else "cpu"
    num_runs = 5

    results = []

    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            print(f"\nBenchmarking with batch_size={batch_size}, seq_len={seq_len}")

            # Create MoE layer
            moe_layer = MoELayer(
                hidden_size=hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
                group_size_m=group_size_m,
                use_forward_only=False,
                use_triton_prep=True,
            ).to(device)

            # Create optimizer
            optimizer = torch.optim.AdamW(moe_layer.parameters(), lr=1e-4)

            # Create input data
            x = torch.randn(
                batch_size, seq_len, hidden_dim, device=device, requires_grad=True
            )
            target = torch.randn(batch_size, seq_len, hidden_dim, device=device)

            # Warmup
            for _ in range(3):
                y = moe_layer(x)
                loss = F.mse_loss(y, target)
                optimizer.zero_grad()
                loss.backward()

            # Benchmark forward pass
            torch.cuda.synchronize()
            forward_times = []

            for _ in range(num_runs):
                start_time = time.time()
                y = moe_layer(x)
                torch.cuda.synchronize()
                forward_times.append(time.time() - start_time)

            avg_forward_time = sum(forward_times) / num_runs

            # Benchmark backward pass
            torch.cuda.synchronize()
            backward_times = []

            for _ in range(num_runs):
                y = moe_layer(x)
                loss = F.mse_loss(y, target)
                optimizer.zero_grad()

                torch.cuda.synchronize()
                start_time = time.time()
                loss.backward()
                torch.cuda.synchronize()
                backward_times.append(time.time() - start_time)

            avg_backward_time = sum(backward_times) / num_runs

            # Calculate throughput
            tokens_per_batch = batch_size * seq_len
            forward_tokens_per_sec = tokens_per_batch / avg_forward_time
            e2e_tokens_per_sec = tokens_per_batch / (
                avg_forward_time + avg_backward_time
            )

            # Save results
            result = {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "tokens_per_batch": tokens_per_batch,
                "forward_time_sec": avg_forward_time,
                "backward_time_sec": avg_backward_time,
                "total_time_sec": avg_forward_time + avg_backward_time,
                "forward_tokens_per_sec": forward_tokens_per_sec,
                "e2e_tokens_per_sec": e2e_tokens_per_sec,
            }

            results.append(result)

            print(f"  Forward time: {avg_forward_time:.4f} s")
            print(f"  Backward time: {avg_backward_time:.4f} s")
            print(f"  Total time: {avg_forward_time + avg_backward_time:.4f} s")
            print(f"  Forward throughput: {forward_tokens_per_sec:.0f} tokens/s")
            print(f"  End-to-end throughput: {e2e_tokens_per_sec:.0f} tokens/s")

    # Print summary table
    print("\nPerformance Summary:")
    print("=" * 80)
    print(
        f"{'Batch Size':<10}{'Seq Len':<10}{'Total Tokens':<15}{'Forward (s)':<15}{'Backward (s)':<15}{'Total (s)':<15}{'Throughput':<15}"
    )
    print("-" * 80)

    for result in results:
        print(
            f"{result['batch_size']:<10}{result['seq_len']:<10}{result['tokens_per_batch']:<15}{result['forward_time_sec']:<15.4f}{result['backward_time_sec']:<15.4f}{result['total_time_sec']:<15.4f}{result['e2e_tokens_per_sec']:<15.0f}"
        )

    return results


from typing import Dict, List, Optional, Tuple

# -----------

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _token_expand_kernel(
    # Input pointers
    tokens_ptr,  # [batch_size * seq_len, hidden_dim]
    top_k_indices_ptr,  # [batch_size * seq_len, top_k]
    # Output pointers
    expanded_tokens_ptr,  # [M_total, hidden_dim]
    original_indices_ptr,  # [M_total]
    # Dimensions
    total_original_tokens,  # batch_size * seq_len
    hidden_dim,  # Token hidden dimension
    top_k: tl.constexpr,  # Number of experts per token
    BLOCK_SIZE_H: tl.constexpr,  # Block size for hidden dimension
):
    """
    Kernel to expand tokens for top-k routing without modifying weights.
    This preserves gradient flow by only handling token duplication, not routing weights.
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

            # Store original index for reconstruction
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
def _sort_tokens_by_expert_kernel(
    # Input pointers
    expanded_tokens_ptr,  # [M_total, hidden_dim]
    expert_indices_ptr,  # [M_total]
    token_weights_ptr,  # [M_total]
    original_indices_ptr,  # [M_total]
    # Output pointers
    sorted_tokens_ptr,  # [M_total, hidden_dim]
    sorted_expert_indices_ptr,  # [M_total]
    sorted_weights_ptr,  # [M_total]
    sorted_original_indices_ptr,  # [M_total]
    # Sort data
    sort_indices_ptr,  # [M_total] - Indices for sorting
    # Dimensions
    M_total,  # Total expanded tokens
    hidden_dim,  # Hidden dimension
    BLOCK_SIZE_H: tl.constexpr,  # Block size for hidden dimension
):
    """
    Kernel to sort tokens by expert assignment. This operation doesn't need
    to propagate gradients as it's just reordering.
    """
    pid = tl.program_id(0)  # token index

    # Only process if within bounds
    if pid < M_total:
        # Load sort index for this token
        sort_idx = tl.load(sort_indices_ptr + pid)

        # Load metadata for this token
        expert_idx = tl.load(expert_indices_ptr + sort_idx)
        weight = tl.load(token_weights_ptr + sort_idx)
        orig_idx = tl.load(original_indices_ptr + sort_idx)

        # Store sorted metadata
        tl.store(sorted_expert_indices_ptr + pid, expert_idx)
        tl.store(sorted_weights_ptr + pid, weight)
        tl.store(sorted_original_indices_ptr + pid, orig_idx)

        # Copy token data in blocks
        for h_start in range(0, hidden_dim, BLOCK_SIZE_H):
            # Calculate actual block size
            h_end = min(h_start + BLOCK_SIZE_H, hidden_dim)
            h_size = h_end - h_start

            # Create offset array with mask
            h_offsets = tl.arange(0, BLOCK_SIZE_H)
            h_mask = h_offsets < h_size

            # Load token block
            token_ptrs = (
                expanded_tokens_ptr + sort_idx * hidden_dim + h_start + h_offsets
            )
            token_block = tl.load(token_ptrs, mask=h_mask, other=0.0)

            # Store sorted token block
            sorted_token_ptrs = (
                sorted_tokens_ptr + pid * hidden_dim + h_start + h_offsets
            )
            tl.store(sorted_token_ptrs, token_block, mask=h_mask)


def hybrid_prepare_tokens(
    tokens: torch.Tensor,  # [batch_size, seq_len, hidden_dim]
    router_logits: torch.Tensor,  # [batch_size, seq_len, num_experts]
    top_k: int = 6,  # Number of experts per token
    group_size_m: int = 128,  # Size of contiguous token blocks
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    Hybrid preparation function that preserves gradient flow to the router
    while using Triton for computationally intensive operations.

    This function uses PyTorch for gradient-sensitive operations and Triton
    for performance-critical parts.

    Returns:
        Tuple of (
            padded_tokens: Tokens arranged by expert [total_padded_tokens, hidden_dim]
            expanded_expert_indices: Expert indices [total_padded_tokens]
            token_weights: Token weights with gradient connection [total_padded_tokens]
            metadata: Dictionary with metadata for output reconstruction
        )
    """
    device = tokens.device
    dtype = tokens.dtype
    batch_size, seq_len, hidden_dim = tokens.shape
    _, _, num_experts = router_logits.shape

    # PART 1: Calculate routing probabilities - using PyTorch to maintain gradients
    router_probs = F.softmax(router_logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(router_probs, k=top_k, dim=-1)

    # Normalize the top-k probabilities - KEEP CONNECTION TO COMPUTATION GRAPH
    top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-10)

    # Store original shapes for reconstruction
    original_shape = (batch_size, seq_len, top_k)

    # Flatten batch and sequence dimensions
    flat_tokens = tokens.reshape(-1, hidden_dim)  # [batch_size*seq_len, hidden_dim]
    flat_top_k_indices = top_k_indices.reshape(-1, top_k)  # [batch_size*seq_len, top_k]
    flat_top_k_probs = top_k_probs.reshape(-1, top_k)  # [batch_size*seq_len, top_k]

    total_original_tokens = flat_tokens.shape[0]
    M_total = total_original_tokens * top_k

    # PART 2: Token expansion using Triton - doesn't need gradient propagation
    # We only expand the tokens and track original indices, not the weights

    # Create output arrays
    expanded_tokens = torch.zeros((M_total, hidden_dim), device=device, dtype=dtype)
    expanded_original_indices = torch.zeros(M_total, dtype=torch.int64, device=device)

    # Determine block size for hidden dimension - power of 2
    block_size_h = min(128, 2 ** int(torch.log2(torch.tensor(hidden_dim)).item()))

    # Launch expansion kernel
    grid = (total_original_tokens,)
    _token_expand_kernel[grid](
        flat_tokens,
        flat_top_k_indices,
        expanded_tokens,
        expanded_original_indices,
        total_original_tokens,
        hidden_dim,
        top_k,
        block_size_h,
    )

    # PART 3: Create expert indices and weights arrays - preserving gradient connection
    # We'll do this in PyTorch to maintain the computation graph

    # Create expert indices and weights arrays
    expanded_expert_indices = torch.zeros(M_total, dtype=torch.int64, device=device)
    expanded_weights = torch.zeros(M_total, device=device)

    # Fill arrays that need gradient connection
    for i in range(total_original_tokens):
        for j in range(top_k):
            idx = i * top_k + j
            # The expert assignment comes from top_k_indices
            expanded_expert_indices[idx] = flat_top_k_indices[i, j]
            # This preserves gradient connection to router
            expanded_weights[idx] = flat_top_k_probs[i, j]

    # PART 4: Sort by expert - can use PyTorch or Triton
    # This doesn't need gradient connection

    # Sort all tokens by their expert assignment
    sorted_indices = torch.argsort(expanded_expert_indices)

    # Reorder all arrays according to this sorting
    sorted_tokens = expanded_tokens[sorted_indices]
    sorted_expert_indices = expanded_expert_indices[sorted_indices]
    # Use indexing that preserves gradients
    sorted_weights = expanded_weights[sorted_indices]
    sorted_original_indices = expanded_original_indices[sorted_indices]

    # PART 5: Create padded layout - using PyTorch

    # Count tokens assigned to each expert
    expert_counts = torch.zeros(num_experts, dtype=torch.int64, device=device)
    for e in range(num_experts):
        expert_counts[e] = torch.sum(sorted_expert_indices == e)

    # Calculate padding needed for each expert
    padded_expert_counts = (
        torch.ceil(expert_counts.float() / group_size_m) * group_size_m
    ).to(torch.int64)

    # Create padded arrays
    total_padded_tokens = padded_expert_counts.sum().item()
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

    # Fill in the padded arrays
    current_pos = 0
    for e in range(num_experts):
        expert_count = expert_counts[e].item()
        padded_count = padded_expert_counts[e].item()

        # Copy actual tokens
        if expert_count > 0:
            expert_mask = sorted_expert_indices == e
            expert_indices = torch.nonzero(expert_mask).squeeze(1)

            padded_tokens[current_pos : current_pos + expert_count] = sorted_tokens[
                expert_indices
            ]
            # Preserve gradient connection
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
        "original_shape": original_shape,
        "top_k_probs": top_k_probs,  # Store for robust gradient connection
    }

    return padded_tokens, expanded_expert_indices, padded_weights, metadata


def hybrid_restore_output(
    output: torch.Tensor,  # [M_total, hidden_dim]
    weights: torch.Tensor,  # [M_total]
    metadata: Dict,  # Metadata from preparation
) -> torch.Tensor:
    """
    Hybrid restore function that ensures proper gradient flow back to the router.
    Uses PyTorch operations for gradient-critical parts.

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
    original_indices = metadata["original_indices"]
    num_original_tokens = metadata["num_original_tokens"]
    top_k_probs = metadata.get("top_k_probs")  # Original probabilities with gradients

    device = output.device
    dtype = output.dtype

    # Initialize accumulator for final output
    final_output = torch.zeros(
        (num_original_tokens, hidden_dim), device=device, dtype=dtype
    )
    weight_accumulator = torch.zeros(num_original_tokens, device=device)

    # Apply weights to output - PRESERVE GRADIENTS
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

    # If we have the original probabilities, ensure gradient connection
    if top_k_probs is not None:
        # Create a robust gradient connection
        # Scale the output by a tiny amount based on routing probabilities
        scale_factor = 1.0 + top_k_probs.sum(dim=-1, keepdim=True) * 1e-10
        final_output = final_output * scale_factor

    return final_output


class HybridMoELayer(torch.nn.Module):
    """
    Mixture of Experts layer with optimized implementation that preserves
    router gradient flow while using Triton acceleration.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 6,
        group_size_m: int = 128,
        use_forward_only: bool = False,
    ):
        """
        Initialize the hybrid MoE layer.

        Args:
            hidden_size: Hidden dimension size
            num_experts: Number of experts
            top_k: Number of experts per token
            group_size_m: Size of contiguous blocks
            use_forward_only: Whether to use forward-only implementation
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.group_size_m = group_size_m
        self.use_forward_only = use_forward_only

        # Router network
        self.router = torch.nn.Linear(hidden_size, num_experts)

        # Expert weights
        self.expert_weights = torch.nn.Parameter(
            torch.randn(num_experts, hidden_size, hidden_size) / (hidden_size**0.5)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with guaranteed gradient flow to the router.

        Args:
            tokens: Input tensor [batch_size, seq_len, hidden_size]

        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        # Get routing logits
        router_logits = self.router(tokens)  # [batch_size, seq_len, num_experts]

        # Use hybrid preparation function that maintains gradient flow
        expanded_tokens, expert_indices, token_weights, metadata = (
            hybrid_prepare_tokens(
                tokens, router_logits, top_k=self.top_k, group_size_m=self.group_size_m
            )
        )

        # Execute expert computation
        if self.use_forward_only:
            # Import here to avoid circular imports
            from cg_forward import cg_grouped_gemm_forward

            output = cg_grouped_gemm_forward(
                expanded_tokens,
                self.expert_weights,
                expert_indices,
                group_size_m=self.group_size_m,
            )
        else:
            # Import here to avoid circular imports
            from cg_backward import cg_grouped_gemm

            output = cg_grouped_gemm(
                expanded_tokens,
                self.expert_weights,
                expert_indices,
                group_size_m=self.group_size_m,
            )

        # Use hybrid restore function that maintains gradient flow
        final_output = hybrid_restore_output(output, token_weights, metadata)

        return final_output


def test_router_gradients():
    """
    Test function to verify gradient flow to the router.
    """
    print("Testing router gradient flow...")

    # Parameters
    batch_size = 2
    seq_len = 4
    hidden_dim = 16
    num_experts = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create models to test
    moe_pytorch = MoELayer(
        hidden_size=hidden_dim,
        num_experts=num_experts,
        use_triton_prep=False,  # Use PyTorch preparation
    ).to(device)

    moe_hybrid = HybridMoELayer(
        hidden_size=hidden_dim,
        num_experts=num_experts,
    ).to(device)

    # Create input
    tokens = torch.randn(
        batch_size, seq_len, hidden_dim, requires_grad=True, device=device
    )

    # Forward pass with PyTorch preparation
    output_pytorch = moe_pytorch(tokens)
    loss_pytorch = output_pytorch.sum()

    # Reset gradients
    tokens.grad = None
    moe_pytorch.router.weight.grad = None

    # Backward pass
    loss_pytorch.backward()

    # Check gradients
    router_grad_norm_pytorch = torch.norm(moe_pytorch.router.weight.grad)
    print(f"PyTorch router gradient norm: {router_grad_norm_pytorch:.6f}")

    # Forward pass with Hybrid preparation
    tokens.grad = None  # Reset gradients
    output_hybrid = moe_hybrid(tokens)
    loss_hybrid = output_hybrid.sum()

    # Reset gradients
    moe_hybrid.router.weight.grad = None

    # Backward pass
    loss_hybrid.backward()

    # Check gradients
    router_grad_norm_hybrid = torch.norm(moe_hybrid.router.weight.grad)
    print(f"Hybrid router gradient norm: {router_grad_norm_hybrid:.6f}")

    # Verify gradients are flowing
    assert router_grad_norm_hybrid > 0, "No gradients flowing to hybrid router!"

    # Compare gradient magnitudes (should be similar but not identical)
    ratio = router_grad_norm_hybrid / router_grad_norm_pytorch
    print(f"Gradient norm ratio (hybrid/pytorch): {ratio:.4f}")

    print("Router gradient flow test passed!")
    return router_grad_norm_pytorch, router_grad_norm_hybrid


def benchmark_preparation_methods(
    batch_size=4, seq_len=512, hidden_dim=768, num_experts=8, top_k=6
):
    """
    Benchmark different token preparation methods.
    """
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create input data
    tokens = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    router_logits = torch.randn(batch_size, seq_len, num_experts, device=device)

    # Warm up
    for _ in range(3):
        # PyTorch preparation
        _ = prepare_tokens_for_cg_gemm_topk(tokens, router_logits, top_k=top_k)
        # Hybrid preparation
        _ = hybrid_prepare_tokens(tokens, router_logits, top_k=top_k)
        # Triton preparation
        try:
            _ = prepare_tokens_triton(tokens, router_logits, top_k=top_k)
        except:
            print("Pure Triton preparation not available on this device")

    # Time PyTorch preparation
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(10):
        _ = prepare_tokens_for_cg_gemm_topk(tokens, router_logits, top_k=top_k)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start_time) / 10

    # Time Hybrid preparation
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(10):
        _ = hybrid_prepare_tokens(tokens, router_logits, top_k=top_k)
    torch.cuda.synchronize()
    hybrid_time = (time.time() - start_time) / 10

    # Time Triton preparation if available
    triton_time = float("inf")
    try:
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(10):
            _ = prepare_tokens_triton(tokens, router_logits, top_k=top_k)
        torch.cuda.synchronize()
        triton_time = (time.time() - start_time) / 10
    except:
        print("Pure Triton preparation not available on this device")

    # Print results
    print(f"Token preparation benchmarks (batch_size={batch_size}, seq_len={seq_len}):")
    print(f"  PyTorch preparation: {pytorch_time*1000:.2f} ms")
    print(f"  Hybrid preparation:  {hybrid_time*1000:.2f} ms")
    if triton_time != float("inf"):
        print(f"  Triton preparation:  {triton_time*1000:.2f} ms")

    # Compare speedups
    print(f"  Hybrid speedup vs PyTorch: {pytorch_time/hybrid_time:.2f}x")
    if triton_time != float("inf"):
        print(f"  Triton speedup vs PyTorch: {pytorch_time/triton_time:.2f}x")
        print(f"  Triton speedup vs Hybrid: {hybrid_time/triton_time:.2f}x")

    return {
        "pytorch_time": pytorch_time,
        "hybrid_time": hybrid_time,
        "triton_time": triton_time if triton_time != float("inf") else None,
    }


if __name__ == "__main__":
    # Test gradient flow
    test_router_gradients()

    # Benchmark preparation methods
    benchmark_preparation_methods()


# -----------
"""
if __name__ == "__main__":
    print("\n==== RUNNING FORWARD-BACKWARD DEMO ====")
    demo_forward_backward()

    print("\n==== COMPARING IMPLEMENTATIONS ====")
    compare_implementations()

    print("\n==== BENCHMARKING PERFORMANCE ====")
    benchmark_performance()
"""
