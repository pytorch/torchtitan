from typing import Dict, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _kernel_restore_output_accumulate(
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
    Kernel to accumulate the weighted outputs back to their original token positions.
    Uses atomic operations for the accumulation to handle the case where multiple expanded
    tokens map back to the same original token.
    """
    pid = tl.program_id(0)  # Block index

    # Calculate the range of tokens this block will process
    start_idx = pid * BLOCK_SIZE_M

    # Check if our block is within bounds
    if start_idx < M_total:
        # Offsets for token dimension with bounds checking
        offs_m = tl.arange(0, BLOCK_SIZE_M) + start_idx
        mask_m = offs_m < M_total

        # Load original token indices
        original_idx = tl.load(original_indices_ptr + offs_m, mask=mask_m, other=-1)

        # Load weights
        weights = tl.load(weights_ptr + offs_m, mask=mask_m, other=0.0)

        # Process only valid tokens (where original_idx >= 0)
        valid_mask = original_idx >= 0

        # Process each valid token in this block
        for m_offset in range(BLOCK_SIZE_M):
            if m_offset < M_total - start_idx and valid_mask[m_offset]:
                m_idx = start_idx + m_offset
                orig_idx = original_idx[m_offset]
                weight = weights[m_offset]

                # Accumulate weight for this original token
                tl.atomic_add(weight_accumulator_ptr + orig_idx, weight)

                # Process hidden dimension in blocks
                for h_start in range(0, hidden_dim, BLOCK_SIZE_H):
                    # Calculate actual block size (handle boundary)
                    h_end = min(h_start + BLOCK_SIZE_H, hidden_dim)
                    h_size = h_end - h_start

                    # Create offsets and mask for hidden dimension
                    offs_h = tl.arange(0, BLOCK_SIZE_H) + h_start
                    mask_h = offs_h < h_size

                    # Load output values
                    output_ptrs = output_ptr + m_idx * hidden_dim + offs_h
                    output_vals = tl.load(output_ptrs, mask=mask_h, other=0.0)

                    # Apply weight
                    weighted_output = output_vals * weight

                    # Atomic accumulate to final output
                    final_output_ptrs = (
                        final_output_ptr + orig_idx * hidden_dim + offs_h
                    )
                    tl.atomic_add(final_output_ptrs, weighted_output, mask=mask_h)


@triton.jit
def _kernel_restore_output_normalize(
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
    Kernel to normalize the accumulated outputs by the accumulated weights.
    """
    pid = tl.program_id(0)  # Block index

    # Calculate the range of tokens this block will process
    start_idx = pid * BLOCK_SIZE_M

    # Check if our block is within bounds
    if start_idx < num_original_tokens:
        # Offsets for token dimension with bounds checking
        offs_m = tl.arange(0, BLOCK_SIZE_M) + start_idx
        mask_m = offs_m < num_original_tokens

        # Load accumulated weights
        weights = tl.load(weight_accumulator_ptr + offs_m, mask=mask_m, other=1.0)

        # Ensure no division by zero
        weights = tl.maximum(weights, eps)

        # Process hidden dimension in blocks
        for h_start in range(0, hidden_dim, BLOCK_SIZE_H):
            # Calculate actual block size (handle boundary)
            h_end = min(h_start + BLOCK_SIZE_H, hidden_dim)
            h_size = h_end - h_start

            # Create offsets and mask for hidden dimension
            offs_h = tl.arange(0, BLOCK_SIZE_H) + h_start
            mask_h = offs_h < h_size

            # Create combined mask
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


def restore_output_triton(
    output: torch.Tensor,  # [M_total, hidden_dim]
    weights: torch.Tensor,  # [M_total]
    metadata: Dict,  # Metadata from preparation
) -> torch.Tensor:
    """
    Accelerated version of restore_output_from_cg_gemm_topk using Triton kernels.
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
    top_k = metadata["top_k"]
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
    # We use a smaller block size for M to allow more parallel blocks
    block_size_m = min(128, triton.next_power_of_2(output.shape[0] // 8))
    block_size_h = min(128, triton.next_power_of_2(hidden_dim))

    # Launch kernel for accumulation
    grid = (triton.cdiv(output.shape[0], block_size_m),)
    _kernel_restore_output_accumulate[grid](
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
    _kernel_restore_output_normalize[grid](
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


def verify_restore_output(
    M_total=1024,
    hidden_dim=768,
    batch_size=2,
    seq_len=512,
    top_k=6,
    device="cuda",
    atol=1e-6,
    rtol=1e-6,
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

    # Run PyTorch implementation
    import time

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

    # Warm up both implementations
    restore_output_pytorch()
    restore_output_triton(output, weights, metadata)

    # Run both implementations with timing
    with timed("PyTorch implementation"):
        pytorch_output = restore_output_pytorch()

    with timed("Triton implementation"):
        triton_output = restore_output_triton(output, weights, metadata)

    # Check if the results match
    match = torch.allclose(pytorch_output, triton_output, atol=atol, rtol=rtol)
    print(f"Results match: {match}")

    if not match:
        # Show differences
        max_diff = torch.max(torch.abs(pytorch_output - triton_output))
        mean_diff = torch.mean(torch.abs(pytorch_output - triton_output))
        print(f"Max difference: {max_diff}")
        print(f"Mean difference: {mean_diff}")

    # Benchmark both implementations
    num_runs = 10

    # Benchmark PyTorch
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        pytorch_output = restore_output_pytorch()
        torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_runs * 1000  # ms

    # Benchmark Triton
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        triton_output = restore_output_triton(output, weights, metadata)
        torch.cuda.synchronize()
    triton_time = (time.time() - start) / num_runs * 1000  # ms

    speedup = pytorch_time / triton_time

    print(f"PyTorch time: {pytorch_time:.3f} ms")
    print(f"Triton time: {triton_time:.3f} ms")
    print(f"Speedup: {speedup:.2f}x")

    return {
        "match": match,
        "pytorch_time": pytorch_time,
        "triton_time": triton_time,
        "speedup": speedup,
    }


# Helper function for timing code blocks
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


if __name__ == "__main__":
    import time

    # Run verification
    print("\nVerifying restore_output implementation...")
    results = verify_restore_output()

    # Benchmark with different sizes
    if results["match"]:
        print("\nRunning benchmarks with different sizes...")

        sizes = [
            # (M_total, hidden_dim, batch_size, seq_len, top_k)
            (5120, 768, 10, 512, 6),  # Small
            (10240, 1024, 20, 512, 6),  # Medium
            (20480, 1536, 40, 512, 6),  # Large
        ]

        for M_total, hidden_dim, batch_size, seq_len, top_k in sizes:
            print(f"\nBenchmarking with M_total={M_total}, hidden_dim={hidden_dim}")
            result = verify_restore_output(
                M_total=M_total,
                hidden_dim=hidden_dim,
                batch_size=batch_size,
                seq_len=seq_len,
                top_k=top_k,
            )
