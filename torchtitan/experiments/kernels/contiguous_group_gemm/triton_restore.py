# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _kernel_restore_output_accumulate_fixed(
    # Input pointers
    output_ptr,  # [M_total, hidden_dim]
    weights_ptr,  # [M_total]
    original_indices_ptr,  # [M_total]
    # Output pointers
    final_output_ptr,  # [num_original_tokens, hidden_dim]
    weight_accumulator_ptr,  # [num_original_tokens]
    # Dimensions
    M_total,  # Total number of tokens after expansion
    hidden_dim,  # Hidden dimension size
    num_original_tokens,  # Original number of tokens (batch_size * seq_len)
    BLOCK_SIZE_M: tl.constexpr,  # Block size for token dimension
    BLOCK_SIZE_H: tl.constexpr,  # Block size for hidden dimension
):
    """
    Fixed kernel to accumulate the weighted outputs back to their original token positions.
    This version simplifies the logic to avoid compilation issues.
    """
    pid = tl.program_id(0)  # Block index

    # Calculate starting position for this block
    start_idx = pid * BLOCK_SIZE_M

    # Only process if in bounds
    if start_idx < M_total:
        # Create offsets for this block
        offs_m = tl.arange(0, BLOCK_SIZE_M) + start_idx

        # Create mask for valid elements
        mask_m = offs_m < M_total

        # Load original token indices and weights
        original_indices = tl.load(original_indices_ptr + offs_m, mask=mask_m, other=-1)
        weights = tl.load(weights_ptr + offs_m, mask=mask_m, other=0.0)

        # Create a mask for valid indices (not padding)
        valid_mask = original_indices >= 0

        # Process hidden dimension in blocks
        for h_start in range(0, hidden_dim, BLOCK_SIZE_H):
            # Calculate bounds for hidden dimension
            h_size = min(BLOCK_SIZE_H, hidden_dim - h_start)
            offs_h = tl.arange(0, BLOCK_SIZE_H) + h_start
            mask_h = offs_h < hidden_dim

            # Load output values
            output_ptrs = output_ptr + offs_m[:, None] * hidden_dim + offs_h[None, :]
            output_vals = tl.load(
                output_ptrs, mask=mask_m[:, None] & mask_h[None, :], other=0.0
            )

            # Apply weights to outputs
            weighted_outputs = output_vals * weights[:, None]

            # Process all tokens in parallel using masks
            for m_idx in range(BLOCK_SIZE_M):
                # Create a mask that's true only for the current m_idx
                m_selector = offs_m == (start_idx + m_idx)
                # Only process if this is a valid token (not padding)
                m_valid = m_selector & valid_mask & mask_m

                if tl.sum(m_valid) > 0:
                    # Get original index using masked load - avoids dynamic indexing
                    # This loads the same value for all lanes, but only one will be valid
                    orig_idx = tl.load(original_indices_ptr + start_idx + m_idx)
                    w = tl.load(weights_ptr + start_idx + m_idx)

                    # Accumulate weight
                    tl.atomic_add(weight_accumulator_ptr + orig_idx, w)

                    # Accumulate weighted output for each hidden dimension element
                    for h_idx in range(BLOCK_SIZE_H):
                        if h_idx < h_size:
                            val = tl.load(
                                output_ptr
                                + (start_idx + m_idx) * hidden_dim
                                + h_start
                                + h_idx
                            )
                            tl.atomic_add(
                                final_output_ptr
                                + orig_idx * hidden_dim
                                + h_start
                                + h_idx,
                                val * w,
                            )


@triton.jit
def _kernel_restore_output_normalize_fixed(
    # Input/output pointers
    final_output_ptr,  # [num_original_tokens, hidden_dim]
    weight_accumulator_ptr,  # [num_original_tokens]
    # Dimensions
    num_original_tokens,  # Original number of tokens (batch_size * seq_len)
    hidden_dim,  # Hidden dimension size
    BLOCK_SIZE_M: tl.constexpr,  # Block size for token dimension
    BLOCK_SIZE_H: tl.constexpr,  # Block size for hidden dimension
    eps: tl.constexpr = 1e-10,  # Epsilon to avoid division by zero
):
    """
    Fixed kernel to normalize the accumulated outputs by the accumulated weights.
    Simplified to avoid compilation issues.
    """
    pid = tl.program_id(0)  # Block index

    # Calculate starting position for this block
    start_idx = pid * BLOCK_SIZE_M

    # Only process if in bounds
    if start_idx < num_original_tokens:
        # Create offsets for this block
        offs_m = tl.arange(0, BLOCK_SIZE_M) + start_idx

        # Create mask for valid elements
        mask_m = offs_m < num_original_tokens

        # Load accumulated weights
        weights = tl.load(weight_accumulator_ptr + offs_m, mask=mask_m, other=1.0)

        # Ensure no division by zero
        weights = tl.maximum(weights, eps)

        # Process hidden dimension in blocks
        for h_start in range(0, hidden_dim, BLOCK_SIZE_H):
            # Calculate bounds for hidden dimension
            offs_h = tl.arange(0, BLOCK_SIZE_H) + h_start
            mask_h = offs_h < hidden_dim

            # Combined mask
            mask = mask_m[:, None] & mask_h[None, :]

            # Load accumulated outputs
            output_ptrs = (
                final_output_ptr + offs_m[:, None] * hidden_dim + offs_h[None, :]
            )
            output_vals = tl.load(output_ptrs, mask=mask, other=0.0)

            # Normalize by weights
            normalized_output = output_vals / weights[:, None]

            # Store normalized outputs
            tl.store(output_ptrs, normalized_output, mask=mask)


def restore_output_triton_fixed(
    output: torch.Tensor,  # [M_total, hidden_dim]
    weights: torch.Tensor,  # [M_total]
    metadata: Dict,  # Metadata from preparation
) -> torch.Tensor:
    """
    Fixed version of restore_output_triton using corrected kernels.
    This function restores the output from contiguous grouped GEMM to original token order.

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

    # Make sure all inputs are contiguous
    output = output.contiguous()
    weights = weights.contiguous()
    original_indices = original_indices.contiguous()

    # Step 1: Initialize accumulator for final output
    final_output = torch.zeros(
        (num_original_tokens, hidden_dim), device=device, dtype=dtype
    )
    weight_accumulator = torch.zeros(num_original_tokens, device=device)

    # Step 2: Accumulate outputs and weights using Triton kernel
    # Determine block sizes - powers of 2 for better performance
    block_size_m = 16  # Smaller block size to avoid register pressure
    block_size_h = 32  # Smaller block size for hidden dimension

    # Launch kernel for accumulation
    grid = (triton.cdiv(output.shape[0], block_size_m),)
    _kernel_restore_output_accumulate_fixed[grid](
        output,
        weights,
        original_indices,
        final_output,
        weight_accumulator,
        output.shape[0],
        hidden_dim,
        num_original_tokens,
        block_size_m,
        block_size_h,
    )

    # Step 3: Normalize by accumulated weights
    grid = (triton.cdiv(num_original_tokens, block_size_m),)
    _kernel_restore_output_normalize_fixed[grid](
        final_output,
        weight_accumulator,
        num_original_tokens,
        hidden_dim,
        block_size_m,
        block_size_h,
    )

    # Reshape to original dimensions
    final_output = final_output.reshape(batch_size, seq_len, hidden_dim)

    return final_output


# Alternative implementation using PyTorch operations but with smaller chunks for better performance
def restore_output_hybrid(
    output: torch.Tensor,  # [M_total, hidden_dim]
    weights: torch.Tensor,  # [M_total]
    metadata: Dict,  # Metadata from preparation
) -> torch.Tensor:
    """
    Hybrid implementation that uses PyTorch operations but processes data in chunks
    for better memory efficiency. This serves as a reliable fallback.

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

    # Make sure all inputs are contiguous
    output = output.contiguous()
    weights = weights.contiguous()
    original_indices = original_indices.contiguous()

    # Initialize accumulator for final output
    final_output = torch.zeros(
        (num_original_tokens, hidden_dim), device=device, dtype=dtype
    )
    weight_accumulator = torch.zeros(num_original_tokens, device=device)

    # Process in chunks to avoid excessive memory usage
    chunk_size = 1024  # Adjust based on available GPU memory

    for start_idx in range(0, output.shape[0], chunk_size):
        end_idx = min(start_idx + chunk_size, output.shape[0])

        # Get chunk data
        chunk_output = output[start_idx:end_idx]
        chunk_weights = weights[start_idx:end_idx]
        chunk_indices = original_indices[start_idx:end_idx]

        # Apply weights
        chunk_weighted_output = chunk_output * chunk_weights.unsqueeze(1)

        # Filter valid indices (not padding)
        valid_mask = chunk_indices >= 0
        valid_indices = chunk_indices[valid_mask]
        valid_outputs = chunk_weighted_output[valid_mask]
        valid_weights = chunk_weights[valid_mask]

        # Accumulate with scatter_add
        if valid_mask.any():
            # For outputs
            index_tensor = valid_indices.unsqueeze(1).expand(-1, hidden_dim)
            final_output.scatter_add_(0, index_tensor, valid_outputs)

            # For weights
            weight_accumulator.scatter_add_(0, valid_indices, valid_weights)

    # Ensure no division by zero
    weight_accumulator = torch.clamp(weight_accumulator, min=1e-10)

    # Normalize by accumulated weights
    final_output = final_output / weight_accumulator.unsqueeze(1)

    # Reshape to original dimensions
    final_output = final_output.reshape(batch_size, seq_len, hidden_dim)

    return final_output


def restore_output_triton(
    output: torch.Tensor,  # [M_total, hidden_dim]
    weights: torch.Tensor,  # [M_total]
    metadata: Dict,  # Metadata from preparation
) -> torch.Tensor:
    """
    Wrapper function that tries the Triton implementation first and falls back to the hybrid
    implementation if Triton fails.

    Args:
        output: Output tensor from CG GEMM [M_total, hidden_dim]
        weights: Token-expert weights [M_total]
        metadata: Metadata from the preparation function

    Returns:
        Reconstructed output [batch_size, seq_len, hidden_dim]
    """
    try:
        return restore_output_triton_fixed(output, weights, metadata)
    except Exception as e:
        print(f"Triton implementation failed with error: {e}")
        print("Falling back to hybrid implementation")
        return restore_output_hybrid(output, weights, metadata)


def verify_restore_output(
    M_total=1024,
    hidden_dim=768,
    batch_size=2,
    seq_len=512,
    top_k=6,
    device="cuda",
    atol=1e-1,
    rtol=1e-1,
):
    """
    Verify that the Triton implementation produces the same results as the PyTorch implementation.
    """
    import time

    # Calculate original tokens
    num_original_tokens = batch_size * seq_len

    # Create test tensors
    torch.manual_seed(0)
    output = torch.randn((M_total, hidden_dim), device=device)
    weights = torch.rand(M_total, device=device)

    # Create original indices mapping
    original_indices = torch.randint(0, num_original_tokens, (M_total,), device=device)
    # Add some padding (-1) indices
    padding_mask = torch.rand(M_total, device=device) < 0.1  # 10% padding
    original_indices = torch.where(
        padding_mask, torch.tensor(-1, device=device), original_indices
    )

    # Create metadata
    metadata = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
        "top_k": top_k,
        "original_indices": original_indices,
        "num_original_tokens": num_original_tokens,
    }

    # Pure PyTorch reference implementation
    def restore_output_pytorch():
        # Initialize accumulator for final output
        final_output = torch.zeros(
            (num_original_tokens, hidden_dim), device=device, dtype=output.dtype
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

    # Helper function for timing
    class timed:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            torch.cuda.synchronize()
            self.start = time.time()
            return self

        def __exit__(self, type, value, traceback):
            torch.cuda.synchronize()
            end = time.time()
            print(f"{self.name}: {(end - self.start) * 1000:.3f} ms")

    # Warm up
    pytorch_output = restore_output_pytorch()

    try:
        # Try both implementations
        triton_output = restore_output_triton_fixed(output, weights, metadata)
        hybrid_output = restore_output_hybrid(output, weights, metadata)
        print(f"{triton_output.shape=}, {hybrid_output.shape=}")
        print(f"{triton_output=}")
        print(f"{hybrid_output=}")
        # Check both implementations
        triton_match = torch.allclose(
            pytorch_output, triton_output, atol=atol, rtol=rtol
        )
        hybrid_match = torch.allclose(
            pytorch_output, hybrid_output, atol=atol, rtol=rtol
        )

        print(f"Triton implementation matches PyTorch: {triton_match}")
        print(f"Hybrid implementation matches PyTorch: {hybrid_match}")

        # Timing comparison
        with timed("PyTorch implementation"):
            pytorch_output = restore_output_pytorch()

        with timed("Triton implementation"):
            triton_output = restore_output_triton_fixed(output, weights, metadata)

        with timed("Hybrid implementation"):
            hybrid_output = restore_output_hybrid(output, weights, metadata)

        # Choose the fastest working implementation
        if triton_match:
            print("Using Triton implementation (matches PyTorch and likely faster)")
            return triton_output
        elif hybrid_match:
            print("Using Hybrid implementation (matches PyTorch but may be slower)")
            return hybrid_output
        else:
            print("Using PyTorch implementation (neither alternative matched)")
            return pytorch_output

    except Exception as e:
        print(f"Triton implementation failed with error: {e}")
        print("Testing hybrid implementation only")

        hybrid_output = restore_output_hybrid(output, weights, metadata)
        hybrid_match = torch.allclose(
            pytorch_output, hybrid_output, atol=atol, rtol=rtol
        )

        print(f"Hybrid implementation matches PyTorch: {hybrid_match}")

        # Timing comparison
        with timed("PyTorch implementation"):
            pytorch_output = restore_output_pytorch()

        with timed("Hybrid implementation"):
            hybrid_output = restore_output_hybrid(output, weights, metadata)

        if hybrid_match:
            print("Using Hybrid implementation (matches PyTorch)")
            return hybrid_output
        else:
            print("Using PyTorch implementation (hybrid didn't match)")
            return pytorch_output


if __name__ == "__main__":
    import time

    print("Testing restore_output implementations...")
    result = verify_restore_output()
