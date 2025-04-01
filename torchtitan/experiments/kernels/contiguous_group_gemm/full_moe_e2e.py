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


# Import original preparation function as fallback/reference
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
                prepare_tokens_triton(
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
    final_output = restore_output_from_cg_gemm_topk(output, token_weights, metadata)

    return final_output


class MoELayer(torch.nn.Module):
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
        use_triton_prep=True,  # Use triton for token preparation
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
    # router_grad_norm = torch.norm(moe_layer.router.weight.grad)
    experts_grad_norm = torch.norm(moe_layer.expert_weights.grad)

    # print(f"Router gradient norm: {router_grad_norm:.4f}")
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


if __name__ == "__main__":
    print("\n==== RUNNING FORWARD-BACKWARD DEMO ====")
    demo_forward_backward()

    print("\n==== COMPARING IMPLEMENTATIONS ====")
    compare_implementations()

    print("\n==== BENCHMARKING PERFORMANCE ====")
    benchmark_performance()
