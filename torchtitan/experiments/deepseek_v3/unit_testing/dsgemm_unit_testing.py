# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Any, Dict, Tuple

import deep_gemm
import torch
from deep_gemm import calc_diff, get_col_major_tma_aligned_tensor


def create_m_indices_fast(m_sizes: torch.Tensor) -> torch.Tensor:
    """
    Fast implementation to create m_indices when tokens are already contiguous.

    Args:
        m_sizes: Tensor containing the number of rows for each group

    Returns:
        m_indices: Tensor mapping each row to its group index
    """
    # Use cumulative sum to create offsets
    total_size = m_sizes.sum().item()

    # Pre-allocate output tensor
    indices = torch.empty(total_size, device=m_sizes.device, dtype=torch.int32)

    # Fill indices directly
    offset = 0
    for i, size in enumerate(m_sizes):
        size_val = size.item()
        if size_val > 0:
            indices[offset : offset + size_val] = i
            offset += size_val

    return indices


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert tensor to FP8 format with per-token scaling.

    Args:
        x: Input tensor of shape [m, k]

    Returns:
        Tuple of (FP8 tensor, scale factors)
    """
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(
        m, n
    ), (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert tensor to FP8 format with per-block scaling.

    Args:
        x: Input tensor

    Returns:
        Tuple of (FP8 tensor, scale factors)
    """
    assert x.dim() == 2
    m, n = x.shape
    m_padded = ((m + 127) // 128) * 128
    n_padded = ((n + 127) // 128) * 128

    x_padded = torch.zeros((m_padded, n_padded), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x

    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)

    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(
        x_view.size(0), x_view.size(2)
    )


def compute_reference_with_scaling(
    lhs: torch.Tensor,
    lhs_scales: torch.Tensor,
    rhs: torch.Tensor,
    rhs_scales: torch.Tensor,
    m_indices: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """
    Compute reference output that matches FP8 scaling behavior.

    Args:
        lhs: Left-hand side FP8 tensor
        lhs_scales: Per-token scales for LHS
        rhs: Right-hand side FP8 tensor
        rhs_scales: Per-block scales for RHS
        m_indices: Group indices for each row
        num_groups: Total number of groups

    Returns:
        Reference output tensor
    """
    m, k = lhs.shape
    n = rhs.shape[1]
    out = torch.zeros((m, n), device=lhs.device, dtype=torch.bfloat16)

    for group_idx in range(num_groups):
        group_mask = m_indices == group_idx
        if not torch.any(group_mask):
            continue  # Skip empty groups

        # Get the rows for this group
        lhs_group = lhs[group_mask].float()
        lhs_scale_group = lhs_scales[group_mask].float()

        # Get the RHS matrix for this group
        rhs_group = rhs[group_idx].float()
        rhs_scale_group = rhs_scales[group_idx].float()

        # Scale back the LHS - per token scaling
        lhs_descaled = lhs_group * lhs_scale_group.unsqueeze(2).expand(
            -1, -1, 128
        ).reshape_as(lhs_group)

        # Scale back the RHS - per block scaling
        rhs_group_view = rhs_group.view((n + 127) // 128, 128, (k + 127) // 128, 128)
        rhs_scale_group_view = rhs_scale_group.view(
            (n + 127) // 128, 1, (k + 127) // 128, 1
        )
        rhs_descaled = rhs_group_view * rhs_scale_group_view.expand(-1, 128, -1, 128)
        rhs_descaled = rhs_descaled.reshape(n, k)

        # Perform matrix multiplication
        result = lhs_descaled @ rhs_descaled.T

        # Convert to bfloat16 and store
        out[group_mask] = result.to(torch.bfloat16)

    return out


def test_m_grouped_gemm_contiguous_with_empty_groups() -> Dict[str, Any]:
    """
    Test case 1: Testing DeepSeek grouped contiguous GEMM with empty groups.
    This test properly accounts for FP8 scaling in the reference computation.

    Returns:
        A dictionary with test results
    """
    print("\n==== TEST 1: Contiguous GEMM with Empty Groups (With Proper Scaling) ====")

    # Create a scenario with 5 groups, with some empty groups
    num_groups = 5
    k = 512  # Multiple of 128 for simplicity
    n = 384  # Multiple of 128 for simplicity

    # Define group sizes with some empty groups
    m_sizes = torch.tensor([128, 0, 256, 0, 128], device="cuda", dtype=torch.int32)

    # Create m_offsets from m_sizes
    m_offsets = torch.zeros(m_sizes.size(0) + 1, device="cuda", dtype=torch.int32)
    torch.cumsum(m_sizes, dim=0, out=m_offsets[1:])

    # Total number of rows (sum of all group sizes)
    total_m = int(m_offsets[-1].item())

    print(f"Group sizes: {m_sizes}")
    print(f"Group offsets: {m_offsets}")
    print(f"Total rows: {total_m}")

    # Create m_indices using our fast method
    m_indices = create_m_indices_fast(m_sizes)
    print(f"m_indices shape: {m_indices.shape}")
    print(f"m_indices sample: {m_indices[:10]} ... {m_indices[-10:]}")

    # Create input data for LHS (left-hand side)
    lhs = torch.randn((total_m, k), device="cuda", dtype=torch.bfloat16)

    # Create input data for RHS (right-hand side) - one per group
    rhs = torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16)

    # Create output tensor
    out = torch.zeros((total_m, n), device="cuda", dtype=torch.bfloat16)

    # Convert to FP8 format
    lhs_fp8, lhs_scales = per_token_cast_to_fp8(lhs)

    # For RHS, handle each group separately
    rhs_fp8 = torch.empty_like(rhs, dtype=torch.float8_e4m3fn)
    rhs_scales = torch.empty(
        (num_groups, (n + 127) // 128, (k + 127) // 128),
        device="cuda",
        dtype=torch.float32,
    )

    for i in range(num_groups):
        rhs_fp8[i], rhs_scales[i] = per_block_cast_to_fp8(rhs[i])

    # Align scales for TMA loads
    lhs_scales_aligned = get_col_major_tma_aligned_tensor(lhs_scales)

    # Pack inputs for the GEMM operation
    lhs_tuple = (lhs_fp8, lhs_scales_aligned)
    rhs_tuple = (rhs_fp8, rhs_scales)

    # Compute reference output with proper scaling
    ref_out = compute_reference_with_scaling(
        lhs_fp8, lhs_scales, rhs_fp8, rhs_scales, m_indices, num_groups
    )

    print(
        f"Reference output computed for {torch.sum(m_sizes > 0).item()} non-empty groups"
    )

    # Call the actual DeepSeek GEMM function
    deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
        lhs_tuple, rhs_tuple, out, m_indices
    )

    # Check that output matches reference
    diff = calc_diff(out, ref_out)
    print(f"Max difference from reference: {diff}")
    print("Test passed!" if diff < 0.001 else "Test failed!")

    return {
        "m_sizes": m_sizes,
        "m_offsets": m_offsets,
        "m_indices": m_indices,
        "ref_out": ref_out,
        "out": out,
        "diff": diff,
    }


def test_m_grouped_gemm_contiguous_all_empty_but_one() -> Dict[str, Any]:
    """
    Test case 2: Testing DeepSeek grouped contiguous GEMM with all groups empty except one.
    This test properly accounts for FP8 scaling in the reference computation.

    Returns:
        A dictionary with test results
    """
    print(
        "\n==== TEST 2: Contiguous GEMM with All Groups Empty Except One (With Proper Scaling) ===="
    )

    # Create an extreme case with many groups, but only one non-empty
    num_groups = 8
    k = 512  # Multiple of 128 for simplicity
    n = 384  # Multiple of 128 for simplicity

    # Define group sizes with only one non-empty group
    m_sizes = torch.zeros(num_groups, device="cuda", dtype=torch.int32)
    m_sizes[3] = 256  # Only group 3 has elements

    # Create m_offsets from m_sizes
    m_offsets = torch.zeros(m_sizes.size(0) + 1, device="cuda", dtype=torch.int32)
    torch.cumsum(m_sizes, dim=0, out=m_offsets[1:])

    # Total number of rows (sum of all group sizes)
    total_m = int(m_offsets[-1].item())

    print(f"Group sizes: {m_sizes}")
    print(f"Group offsets: {m_offsets}")
    print(f"Total rows: {total_m}")

    # Create m_indices using our fast method
    m_indices = create_m_indices_fast(m_sizes)
    print(f"m_indices shape: {m_indices.shape}")
    print(f"m_indices unique values: {torch.unique(m_indices)}")

    # Create input data for LHS (left-hand side)
    lhs = torch.randn((total_m, k), device="cuda", dtype=torch.bfloat16)

    # Create input data for RHS (right-hand side) - one per group
    rhs = torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16)

    # Create output tensor
    out = torch.zeros((total_m, n), device="cuda", dtype=torch.bfloat16)

    # Convert to FP8 format
    lhs_fp8, lhs_scales = per_token_cast_to_fp8(lhs)

    # For RHS, handle each group separately
    rhs_fp8 = torch.empty_like(rhs, dtype=torch.float8_e4m3fn)
    rhs_scales = torch.empty(
        (num_groups, (n + 127) // 128, (k + 127) // 128),
        device="cuda",
        dtype=torch.float32,
    )

    for i in range(num_groups):
        rhs_fp8[i], rhs_scales[i] = per_block_cast_to_fp8(rhs[i])

    # Align scales for TMA loads
    lhs_scales_aligned = get_col_major_tma_aligned_tensor(lhs_scales)

    # Pack inputs for the GEMM operation
    lhs_tuple = (lhs_fp8, lhs_scales_aligned)
    rhs_tuple = (rhs_fp8, rhs_scales)

    # Compute reference output with proper scaling
    ref_out = compute_reference_with_scaling(
        lhs_fp8, lhs_scales, rhs_fp8, rhs_scales, m_indices, num_groups
    )

    print(
        f"Reference output computed for {torch.sum(m_sizes > 0).item()} non-empty groups"
    )

    # Call the actual DeepSeek GEMM function
    deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
        lhs_tuple, rhs_tuple, out, m_indices
    )

    # Check that output matches reference
    diff = calc_diff(out, ref_out)
    print(f"Max difference from reference: {diff}")
    print("Test passed!" if diff < 0.001 else "Test failed!")

    return {
        "m_sizes": m_sizes,
        "m_offsets": m_offsets,
        "m_indices": m_indices,
        "ref_out": ref_out,
        "out": out,
        "diff": diff,
    }


def test_m_grouped_gemm_contiguous_with_scaling_edge_cases() -> Dict[str, Any]:
    """
    Test case 3: Test edge cases with scaling (small dimensions, minimum block sizes).

    Returns:
        A dictionary with test results
    """
    print("\n==== TEST 3: Testing Edge Cases with Different Scaling Patterns ====")

    # Create a scenario with minimal block sizes for edge case testing
    num_groups = 4
    k = 128  # Minimal multiple of 128
    n = 128  # Minimal multiple of 128

    # Define group sizes with minimal dimensions
    m_sizes = torch.tensor([128, 0, 128, 0], device="cuda", dtype=torch.int32)

    # Create m_offsets from m_sizes
    m_offsets = torch.zeros(m_sizes.size(0) + 1, device="cuda", dtype=torch.int32)
    torch.cumsum(m_sizes, dim=0, out=m_offsets[1:])

    # Total number of rows (sum of all group sizes)
    total_m = int(m_offsets[-1].item())

    print(f"Group sizes: {m_sizes}")
    print(f"Group offsets: {m_offsets}")
    print(f"Total rows: {total_m}")

    # Create m_indices using our fast method
    m_indices = create_m_indices_fast(m_sizes)
    print(f"m_indices shape: {m_indices.shape}")

    # Create input data with extreme values to test scaling
    lhs = torch.randn((total_m, k), device="cuda", dtype=torch.bfloat16) * 100.0
    rhs = torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16) * 50.0

    # Create output tensor
    out = torch.zeros((total_m, n), device="cuda", dtype=torch.bfloat16)

    # Convert to FP8 format
    lhs_fp8, lhs_scales = per_token_cast_to_fp8(lhs)

    # For RHS, handle each group separately
    rhs_fp8 = torch.empty_like(rhs, dtype=torch.float8_e4m3fn)
    rhs_scales = torch.empty(
        (num_groups, (n + 127) // 128, (k + 127) // 128),
        device="cuda",
        dtype=torch.float32,
    )

    for i in range(num_groups):
        rhs_fp8[i], rhs_scales[i] = per_block_cast_to_fp8(rhs[i])

    # Align scales for TMA loads
    lhs_scales_aligned = get_col_major_tma_aligned_tensor(lhs_scales)

    # Pack inputs for the GEMM operation
    lhs_tuple = (lhs_fp8, lhs_scales_aligned)
    rhs_tuple = (rhs_fp8, rhs_scales)

    # Compute reference output with proper scaling
    ref_out = compute_reference_with_scaling(
        lhs_fp8, lhs_scales, rhs_fp8, rhs_scales, m_indices, num_groups
    )

    print(
        f"Reference output computed for {torch.sum(m_sizes > 0).item()} non-empty groups"
    )

    # Call the actual DeepSeek GEMM function
    deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
        lhs_tuple, rhs_tuple, out, m_indices
    )

    # Check that output matches reference
    diff = calc_diff(out, ref_out)
    print(f"Max difference from reference: {diff}")
    print("Test passed!" if diff < 0.001 else "Test failed!")

    return {
        "m_sizes": m_sizes,
        "m_offsets": m_offsets,
        "m_indices": m_indices,
        "ref_out": ref_out,
        "out": out,
        "diff": diff,
    }


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # Enable TF32 for better performance (if available)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Run all tests
    results = {
        "test1": test_m_grouped_gemm_contiguous_with_empty_groups(),
        "test2": test_m_grouped_gemm_contiguous_all_empty_but_one(),
        "test3": test_m_grouped_gemm_contiguous_with_scaling_edge_cases(),
    }

    # Print overall summary
    print("\n==== Test Summary ====")
    for test_name, test_results in results.items():
        diff = test_results["diff"]
        print(f"{test_name}: {'PASSED' if diff < 0.001 else 'FAILED'} (diff = {diff})")
