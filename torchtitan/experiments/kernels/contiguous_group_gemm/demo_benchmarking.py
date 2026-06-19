# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# this demo compares performance of pytorch with triton prep


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

    # create expanded tokens where each original token appears k times, once for each expert
    M_total = total_original_tokens * top_k

    # arrays to hold expanded tokens and their metadata
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
    use_triton_prep: bool = False,  # Whether to use Triton for token preparation
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
        use_triton_prep: Whether to use Triton for token preparation

    Returns:
        Output tensor [batch_size, seq_len, hidden_dim]
    """
    if cg_forward_fn is None:
        raise ValueError(
            "Must provide a function that implements the CG GEMM forward pass"
        )

    # Get routing logits
    router_logits = router(tokens)  # [batch_size, seq_len, num_experts]

    # Prepare tokens for contiguous grouped GEMM (using Triton or PyTorch)
    if use_triton_prep:
        expanded_tokens, expert_indices, token_weights, metadata = (
            prepare_tokens_triton(
                tokens, router_logits, top_k=top_k, group_size_m=group_size_m
            )
        )
    else:
        expanded_tokens, expert_indices, token_weights, metadata = (
            prepare_tokens_for_cg_gemm_topk(
                tokens, router_logits, top_k=top_k, group_size_m=group_size_m
            )
        )

    # Run contiguous grouped GEMM
    output = cg_forward_fn(
        expanded_tokens, expert_weights, expert_indices, group_size_m=group_size_m
    )

    # Restore original token order
    final_output = restore_output_from_cg_gemm_topk(output, token_weights, metadata)

    return final_output


def pytorch_cg_gemm_forward(
    inputs: torch.Tensor,  # [M_total, K]
    expert_weights: torch.Tensor,  # [num_experts, N, K]
    expert_indices: torch.Tensor,  # [M_total]
    group_size_m: int = 128,
) -> torch.Tensor:
    """Reference implementation of grouped GEMM using PyTorch operations."""
    M_total, K = inputs.shape
    num_experts, N, _ = expert_weights.shape

    # Create output tensor
    output = torch.zeros((M_total, N), device=inputs.device, dtype=inputs.dtype)

    # Process each group
    for g in range(0, M_total, group_size_m):
        end = min(g + group_size_m, M_total)

        # Get expert index for this group
        expert_idx = expert_indices[g].item()

        # Get expert weights
        expert_weight = expert_weights[expert_idx]

        # Compute output for this group
        output[g:end] = torch.matmul(inputs[g:end], expert_weight.t())

    return output


def benchmark_token_preparation(
    tokens: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    group_size_m: int,
    num_runs: int = 10,
) -> dict:
    """
    Benchmark token preparation implementations.

    Args:
        tokens: Input tokens
        router_logits: Router logits
        top_k: Number of experts per token
        group_size_m: Group size
        num_runs: Number of benchmark runs

    Returns:
        Dictionary with benchmark results
    """
    # Benchmark PyTorch implementation
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(num_runs):
        # Warmup run
        pytorch_tokens, pytorch_indices, pytorch_weights, pytorch_metadata = (
            prepare_tokens_for_cg_gemm_topk(
                tokens, router_logits, top_k=top_k, group_size_m=group_size_m
            )
        )
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_runs):
        pytorch_tokens, pytorch_indices, pytorch_weights, pytorch_metadata = (
            prepare_tokens_for_cg_gemm_topk(
                tokens, router_logits, top_k=top_k, group_size_m=group_size_m
            )
        )
        torch.cuda.synchronize()

    pytorch_time = (time.time() - start) / num_runs * 1000  # ms

    # Benchmark Triton implementation
    torch.cuda.synchronize()
    for _ in range(num_runs):
        # Warmup run
        triton_tokens, triton_indices, triton_weights, triton_metadata = (
            prepare_tokens_triton(
                tokens, router_logits, top_k=top_k, group_size_m=group_size_m
            )
        )
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_runs):
        triton_tokens, triton_indices, triton_weights, triton_metadata = (
            prepare_tokens_triton(
                tokens, router_logits, top_k=top_k, group_size_m=group_size_m
            )
        )
        torch.cuda.synchronize()

    triton_time = (time.time() - start) / num_runs * 1000  # ms

    # Verify the results match
    outputs_match = (
        torch.allclose(pytorch_tokens, triton_tokens)
        and torch.all(pytorch_indices == triton_indices)
        and torch.allclose(pytorch_weights, triton_weights)
    )

    # Calculate speedup
    speedup = pytorch_time / triton_time

    return {
        "pytorch_time_ms": pytorch_time,
        "triton_time_ms": triton_time,
        "speedup": speedup,
        "outputs_match": outputs_match,
        "pytorch_shape": pytorch_tokens.shape,
        "triton_shape": triton_tokens.shape,
    }


def benchmark_forward_pass(
    tokens: torch.Tensor,
    router: torch.nn.Linear,
    expert_weights: torch.Tensor,
    top_k: int,
    group_size_m: int,
    num_runs: int = 10,
) -> dict:
    """
    Benchmark forward pass implementations.

    Args:
        tokens: Input tokens
        router: Router network
        expert_weights: Expert weights
        top_k: Number of experts per token
        group_size_m: Group size
        num_runs: Number of benchmark runs

    Returns:
        Dictionary with benchmark results
    """
    # Get routing logits
    router_logits = router(tokens)

    # Prepare tokens once for all benchmarks
    expanded_tokens, expert_indices, token_weights, metadata = prepare_tokens_triton(
        tokens, router_logits, top_k=top_k, group_size_m=group_size_m
    )

    # Benchmark PyTorch reference implementation
    torch.cuda.synchronize()
    for _ in range(num_runs):
        # Warmup run
        pytorch_output = pytorch_cg_gemm_forward(
            expanded_tokens, expert_weights, expert_indices, group_size_m=group_size_m
        )
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_runs):
        pytorch_output = pytorch_cg_gemm_forward(
            expanded_tokens, expert_weights, expert_indices, group_size_m=group_size_m
        )
        torch.cuda.synchronize()

    pytorch_time = (time.time() - start) / num_runs * 1000  # ms

    # Benchmark Triton implementation
    torch.cuda.synchronize()
    for _ in range(num_runs):
        # Warmup run
        triton_output = cg_grouped_gemm_forward(
            expanded_tokens, expert_weights, expert_indices, group_size_m=group_size_m
        )
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_runs):
        triton_output = cg_grouped_gemm_forward(
            expanded_tokens, expert_weights, expert_indices, group_size_m=group_size_m
        )
        torch.cuda.synchronize()

    triton_time = (time.time() - start) / num_runs * 1000  # ms

    # Verify the results match - soft matching for now
    outputs_match = torch.allclose(pytorch_output, triton_output, rtol=1e-1, atol=1e-1)

    # Calculate speedup
    speedup = pytorch_time / triton_time

    # Calculate TFLOPS
    M_total, K = expanded_tokens.shape
    _, N, _ = expert_weights.shape  # Calculate TFLOPS
    M_total, K = expanded_tokens.shape
    _, N, _ = expert_weights.shape

    # Each group uses one expert's weights
    # For each token in the group, we do NK multiply-adds (2NK FLOPs)
    # So total FLOPs is approximately M_total * 2 * N * K
    flops = M_total * 2 * N * K

    triton_tflops = flops / (triton_time / 1000) / 1e12
    pytorch_tflops = flops / (pytorch_time / 1000) / 1e12

    return {
        "pytorch_time_ms": pytorch_time,
        "triton_time_ms": triton_time,
        "speedup": speedup,
        "outputs_match": outputs_match,
        "pytorch_tflops": pytorch_tflops,
        "triton_tflops": triton_tflops,
    }


def benchmark_end_to_end(
    tokens: torch.Tensor,
    router: torch.nn.Linear,
    expert_weights: torch.Tensor,
    top_k: int,
    group_size_m: int,
    num_runs: int = 10,
) -> dict:
    """
    Benchmark end-to-end implementations.

    Args:
        tokens: Input tokens
        router: Router network
        expert_weights: Expert weights
        top_k: Number of experts per token
        group_size_m: Group size
        num_runs: Number of benchmark runs

    Returns:
        Dictionary with benchmark results
    """
    # Benchmark PyTorch preparation + PyTorch forward implementation
    torch.cuda.synchronize()
    for _ in range(num_runs // 2):  # Fewer warmup runs for end-to-end
        # Warmup run
        output_pytorch_pytorch = example_moe_with_cg_gemm_topk(
            tokens=tokens,
            router=router,
            expert_weights=expert_weights,
            top_k=top_k,
            group_size_m=group_size_m,
            cg_forward_fn=pytorch_cg_gemm_forward,
            use_triton_prep=False,
        )
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_runs):
        output_pytorch_pytorch = example_moe_with_cg_gemm_topk(
            tokens=tokens,
            router=router,
            expert_weights=expert_weights,
            top_k=top_k,
            group_size_m=group_size_m,
            cg_forward_fn=pytorch_cg_gemm_forward,
            use_triton_prep=False,
        )
        torch.cuda.synchronize()

    pytorch_pytorch_time = (time.time() - start) / num_runs * 1000  # ms

    # Benchmark Triton preparation + PyTorch forward implementation
    torch.cuda.synchronize()
    for _ in range(num_runs // 2):  # Fewer warmup runs
        # Warmup run
        output_triton_pytorch = example_moe_with_cg_gemm_topk(
            tokens=tokens,
            router=router,
            expert_weights=expert_weights,
            top_k=top_k,
            group_size_m=group_size_m,
            cg_forward_fn=pytorch_cg_gemm_forward,
            use_triton_prep=True,
        )
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_runs):
        output_triton_pytorch = example_moe_with_cg_gemm_topk(
            tokens=tokens,
            router=router,
            expert_weights=expert_weights,
            top_k=top_k,
            group_size_m=group_size_m,
            cg_forward_fn=pytorch_cg_gemm_forward,
            use_triton_prep=True,
        )
        torch.cuda.synchronize()

    triton_pytorch_time = (time.time() - start) / num_runs * 1000  # ms

    # Benchmark PyTorch preparation + Triton forward implementation
    torch.cuda.synchronize()
    for _ in range(num_runs // 2):  # Fewer warmup runs
        # Warmup run
        output_pytorch_triton = example_moe_with_cg_gemm_topk(
            tokens=tokens,
            router=router,
            expert_weights=expert_weights,
            top_k=top_k,
            group_size_m=group_size_m,
            cg_forward_fn=cg_grouped_gemm_forward,
            use_triton_prep=False,
        )
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_runs):
        output_pytorch_triton = example_moe_with_cg_gemm_topk(
            tokens=tokens,
            router=router,
            expert_weights=expert_weights,
            top_k=top_k,
            group_size_m=group_size_m,
            cg_forward_fn=cg_grouped_gemm_forward,
            use_triton_prep=False,
        )
        torch.cuda.synchronize()

    pytorch_triton_time = (time.time() - start) / num_runs * 1000  # ms

    # Benchmark Triton preparation + Triton forward implementation
    torch.cuda.synchronize()
    for _ in range(num_runs // 2):  # Fewer warmup runs
        # Warmup run
        output_triton_triton = example_moe_with_cg_gemm_topk(
            tokens=tokens,
            router=router,
            expert_weights=expert_weights,
            top_k=top_k,
            group_size_m=group_size_m,
            cg_forward_fn=cg_grouped_gemm_forward,
            use_triton_prep=True,
        )
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_runs):
        output_triton_triton = example_moe_with_cg_gemm_topk(
            tokens=tokens,
            router=router,
            expert_weights=expert_weights,
            top_k=top_k,
            group_size_m=group_size_m,
            cg_forward_fn=cg_grouped_gemm_forward,
            use_triton_prep=True,
        )
        torch.cuda.synchronize()

    triton_triton_time = (time.time() - start) / num_runs * 1000  # ms

    # Verify the results match
    tolerance_rtol = 1e-1
    tolerance_atol = 1e-1
    all_match = (
        torch.allclose(
            output_pytorch_pytorch,
            output_triton_pytorch,
            rtol=tolerance_rtol,
            atol=tolerance_atol,
        )
        and torch.allclose(
            output_pytorch_pytorch,
            output_pytorch_triton,
            rtol=tolerance_rtol,
            atol=tolerance_atol,
        )
        and torch.allclose(
            output_pytorch_pytorch,
            output_triton_triton,
            rtol=tolerance_rtol,
            atol=tolerance_atol,
        )
    )

    # Calculate speedups relative to PyTorch baseline
    speedup_triton_prep = pytorch_pytorch_time / triton_pytorch_time
    speedup_triton_forward = pytorch_pytorch_time / pytorch_triton_time
    speedup_full_triton = pytorch_pytorch_time / triton_triton_time

    return {
        "pytorch_pytorch_time_ms": pytorch_pytorch_time,
        "triton_pytorch_time_ms": triton_pytorch_time,
        "pytorch_triton_time_ms": pytorch_triton_time,
        "triton_triton_time_ms": triton_triton_time,
        "speedup_triton_prep": speedup_triton_prep,
        "speedup_triton_forward": speedup_triton_forward,
        "speedup_full_triton": speedup_full_triton,
        "outputs_match": all_match,
    }


def run_benchmarks(
    batch_size_range=[1, 2, 4, 8], seq_len_range=[128, 256, 512, 1024], hidden_dim=768
):
    """
    Run benchmarks with different batch sizes and sequence lengths.

    Args:
        batch_size_range: List of batch sizes to test
        seq_len_range: List of sequence lengths to test
        hidden_dim: Hidden dimension size

    Returns:
        Dictionary with benchmark results
    """
    num_experts = 8
    top_k = 6
    group_size_m = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    results = []

    for batch_size in batch_size_range:
        for seq_len in seq_len_range:
            print(
                f"\nBenchmarking batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}"
            )

            # Create sample inputs
            tokens = torch.randn(
                (batch_size, seq_len, hidden_dim), device=device, dtype=dtype
            )
            router = torch.nn.Linear(
                hidden_dim, num_experts, device=device, dtype=dtype
            )
            expert_weights = torch.randn(
                (num_experts, hidden_dim, hidden_dim), device=device, dtype=dtype
            )

            # Benchmark token preparation
            print("Benchmarking token preparation...")
            router_logits = router(tokens)
            prep_results = benchmark_token_preparation(
                tokens, router_logits, top_k, group_size_m
            )

            print(f"  PyTorch: {prep_results['pytorch_time_ms']:.2f} ms")
            print(f"  Triton: {prep_results['triton_time_ms']:.2f} ms")
            print(f"  Speedup: {prep_results['speedup']:.2f}x")
            print(f"  Outputs match: {prep_results['outputs_match']}")

            # Benchmark forward pass
            print("Benchmarking forward pass...")
            forward_results = benchmark_forward_pass(
                tokens, router, expert_weights, top_k, group_size_m
            )

            print(
                f"  PyTorch: {forward_results['pytorch_time_ms']:.2f} ms, {forward_results['pytorch_tflops']:.2f} TFLOPS"
            )
            print(
                f"  Triton: {forward_results['triton_time_ms']:.2f} ms, {forward_results['triton_tflops']:.2f} TFLOPS"
            )
            print(f"  Speedup: {forward_results['speedup']:.2f}x")
            print(f"  Outputs match: {forward_results['outputs_match']}")

            # Benchmark end-to-end
            print("Benchmarking end-to-end...")
            e2e_results = benchmark_end_to_end(
                tokens, router, expert_weights, top_k, group_size_m
            )

            print(
                f"  PyTorch prep + PyTorch forward: {e2e_results['pytorch_pytorch_time_ms']:.2f} ms"
            )
            print(
                f"  Triton prep + PyTorch forward: {e2e_results['triton_pytorch_time_ms']:.2f} ms"
            )
            print(
                f"  PyTorch prep + Triton forward: {e2e_results['pytorch_triton_time_ms']:.2f} ms"
            )
            print(
                f"  Triton prep + Triton forward: {e2e_results['triton_triton_time_ms']:.2f} ms"
            )
            print(
                f"  Speedup with Triton prep: {e2e_results['speedup_triton_prep']:.2f}x"
            )
            print(
                f"  Speedup with Triton forward: {e2e_results['speedup_triton_forward']:.2f}x"
            )
            print(
                f"  Speedup with full Triton: {e2e_results['speedup_full_triton']:.2f}x"
            )
            print(f"  All outputs match: {e2e_results['outputs_match']}")

            # Combine results
            result = {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "hidden_dim": hidden_dim,
                "prep": prep_results,
                "forward": forward_results,
                "e2e": e2e_results,
            }

            results.append(result)

    return results


def demo_usage():
    """
    Demonstration of how to use the data preparation utilities with a CG GEMM implementation.
    Includes benchmarking of different implementations.
    """
    # Parameters
    batch_size = 2
    seq_len = 1024
    hidden_dim = 768
    num_experts = 8
    top_k = 6
    group_size_m = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    print(
        f"Running demo with batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}"
    )
    print(f"num_experts={num_experts}, top_k={top_k}, group_size_m={group_size_m}")
    print(f"device={device}, dtype={dtype}")

    # Create sample inputs
    tokens = torch.randn((batch_size, seq_len, hidden_dim), device=device, dtype=dtype)
    router = torch.nn.Linear(hidden_dim, num_experts, device=device, dtype=dtype)
    expert_weights = torch.randn(
        (num_experts, hidden_dim, hidden_dim), device=device, dtype=dtype
    )

    # Run with PyTorch preparation and Triton forward pass
    print("\nRunning with PyTorch preparation and Triton forward pass...")
    output_pytorch_triton = example_moe_with_cg_gemm_topk(
        tokens=tokens,
        router=router,
        expert_weights=expert_weights,
        top_k=top_k,
        group_size_m=group_size_m,
        cg_forward_fn=cg_grouped_gemm_forward,
        use_triton_prep=False,
    )
    print(f"Output shape: {output_pytorch_triton.shape}")

    # Run with Triton preparation and Triton forward pass
    print("\nRunning with Triton preparation and Triton forward pass...")
    output_triton_triton = example_moe_with_cg_gemm_topk(
        tokens=tokens,
        router=router,
        expert_weights=expert_weights,
        top_k=top_k,
        group_size_m=group_size_m,
        cg_forward_fn=cg_grouped_gemm_forward,
        use_triton_prep=True,
    )
    print(f"Output shape: {output_triton_triton.shape}")

    # Verify outputs match
    outputs_match = torch.allclose(
        output_pytorch_triton, output_triton_triton, rtol=1e-2, atol=1e-2
    )
    print(f"Outputs match: {outputs_match}")

    # Run benchmarks
    print("\nRunning full benchmarks...")
    results = run_benchmarks(
        batch_size_range=[1, 2, 4],
        seq_len_range=[256, 512, 1024],
        hidden_dim=hidden_dim,
    )

    # Print a summary of the results
    print("\nBenchmark Summary:")
    print("================================================")
    print("| Batch | Seq Len | Prep Speedup | Forward Speedup | E2E Speedup |")
    print("------------------------------------------------")

    for result in results:
        bs = result["batch_size"]
        sl = result["seq_len"]
        prep_speedup = result["prep"]["speedup"]
        forward_speedup = result["forward"]["speedup"]
        e2e_speedup = result["e2e"]["speedup_full_triton"]

        print(
            f"| {bs:5d} | {sl:7d} | {prep_speedup:12.2f}x | {forward_speedup:15.2f}x | {e2e_speedup:11.2f}x |"
        )

    print("================================================")

    return output_triton_triton


if __name__ == "__main__":
    demo_usage()
