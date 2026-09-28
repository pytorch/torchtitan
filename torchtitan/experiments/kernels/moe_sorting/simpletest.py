"""
// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.
"""

import torch

try:
    import token_sorting_cuda
except ImportError:
    print(f"unable to import token_sorting_cuda extension...")
    raise

import argparse

import numpy as np


def pytorch_sort_tokens(topk_ids, x, n_experts):
    """Original PyTorch implementation for comparison"""
    with torch.no_grad():
        # [seq_len, n_experts]
        cnts = topk_ids.new_zeros((topk_ids.shape[0], n_experts))
        # Fill 1 to the selected experts
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        # Token indices for each expert
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens_shape = idxs.shape + x.shape[1:]
    sorted_tokens = x[idxs // topk_ids.shape[1]]

    return sorted_tokens, idxs, tokens_per_expert


def cuda_sort_tokens(topk_ids, x, n_experts):
    """CUDA optimized implementation"""
    # topk_int_ids = topk_ids.to(torch.int32)
    # topk_int_ids = topk_ids.to(torch.int32)
    # print(f"Original dtype: {topk_ids.dtype}, Converted dtype: {topk_int_ids.dtype}")

    # print(f"{topk_int_ids=}, {x=}, {n_experts=}")

    sorted_tokens, sorted_indices, tokens_per_expert = (
        token_sorting_cuda.sort_tokens_by_expert(topk_ids, x, n_experts)
    )

    return sorted_tokens, sorted_indices, tokens_per_expert


def test_simple_case():
    """Test with a simple example where we know the expected output"""
    device = torch.device("cuda")

    # Create small test case
    seq_len = 4
    k = 2
    hidden_dim = 3
    n_experts = 3

    # Create expert assignments: [[0,1], [1,2], [0,2], [1,0]]
    topk_ids = torch.tensor(
        [[0, 1], [1, 2], [0, 2], [1, 0]], device=device, dtype=torch.int64
    )

    # Create token features with recognizable values
    x = torch.tensor(
        [
            [1.0, 1.1, 1.2],  # token 0
            [2.0, 2.1, 2.2],  # token 1
            [3.0, 3.1, 3.2],  # token 2
            [4.0, 4.1, 4.2],  # token 3
        ],
        device=device,
        dtype=torch.float32,
    )

    print("\n===== SIMPLE TEST CASE =====")
    print(f"Input topk_ids:\n{topk_ids}")
    print(f"Input tokens:\n{x}")

    # Run implementations
    pt_sorted, pt_indices, pt_counts = pytorch_sort_tokens(topk_ids, x, n_experts)

    cuda_sorted, cuda_indices, cuda_counts = cuda_sort_tokens(topk_ids, x, n_experts)

    # Display results
    print("\nToken counts per expert:")
    print(f"PyTorch: {pt_counts}")
    print(f"CUDA:    {cuda_counts}")
    print(f"Match:   {torch.allclose(pt_counts, cuda_counts)}")

    print("\nSorted indices:")
    print(f"PyTorch: {pt_indices}")
    print(f"CUDA:    {cuda_indices}")
    print(f"Shapes match: {pt_indices.shape == cuda_indices.shape}")

    print("\nSorted tokens (first few):")
    print(f"PyTorch:\n{pt_sorted[:5]}")
    print(f"CUDA:\n{cuda_sorted[:5]}")
    print(f"Shapes match: {pt_sorted.shape == cuda_sorted.shape}")

    if pt_sorted.shape == cuda_sorted.shape:
        tokens_match = torch.allclose(pt_sorted, cuda_sorted, rtol=1e-5, atol=1e-5)
        print(f"Values match: {tokens_match}")

    overall_match = (
        torch.allclose(pt_counts, cuda_counts)
        and pt_indices.shape == cuda_indices.shape
        and pt_sorted.shape == cuda_sorted.shape
        and torch.allclose(pt_sorted, cuda_sorted, rtol=1e-5, atol=1e-5)
    )

    print(f"\nOverall match: {overall_match}")
    return overall_match


def test_random_case(seq_len=16, hidden_dim=8, n_experts=4, k=2):
    """Test with random inputs of specified dimensions"""
    torch.manual_seed(42)  # For reproducibility
    device = torch.device("cuda")

    # Create random inputs
    topk_ids = torch.randint(
        0, n_experts, (seq_len, k), device=device, dtype=torch.int64
    )
    x = torch.randn(seq_len, hidden_dim, device=device)

    print(f"\n===== RANDOM TEST CASE =====")
    print(f"seq_len={seq_len}, hidden_dim={hidden_dim}, n_experts={n_experts}, k={k}")

    # Run implementations
    pt_sorted, pt_indices, pt_counts = pytorch_sort_tokens(topk_ids, x, n_experts)
    cuda_sorted, cuda_indices, cuda_counts = cuda_sort_tokens(topk_ids, x, n_experts)

    # Display results
    print("\nToken counts per expert:")
    print(f"PyTorch: {pt_counts}")
    print(f"CUDA:    {cuda_counts}")
    print(f"Match:   {torch.allclose(pt_counts, cuda_counts)}")

    print("\nSorted indices shapes:")
    print(f"PyTorch: {pt_indices.shape}")
    print(f"CUDA:    {cuda_indices.shape}")
    print(f"Match:   {pt_indices.shape == cuda_indices.shape}")

    print("\nSorted tokens shapes:")
    print(f"PyTorch: {pt_sorted.shape}")
    print(f"CUDA:    {cuda_sorted.shape}")
    print(f"Match:   {pt_sorted.shape == cuda_sorted.shape}")

    if pt_sorted.shape == cuda_sorted.shape:
        tokens_match = torch.allclose(pt_sorted, cuda_sorted, rtol=1e-5, atol=1e-5)
        print(f"Values match: {tokens_match}")

    overall_match = (
        torch.allclose(pt_counts, cuda_counts)
        and pt_indices.shape == cuda_indices.shape
        and pt_sorted.shape == cuda_sorted.shape
        and torch.allclose(pt_sorted, cuda_sorted, rtol=1e-5, atol=1e-5)
    )

    print(f"\nOverall match: {overall_match}")
    return overall_match


def debug_equality(pt_sorted, cuda_sorted, pt_indices, cuda_indices):
    """Debug why tensors might not be equal"""
    print("\n===== DEBUGGING EQUALITY =====")

    if pt_sorted.shape != cuda_sorted.shape:
        print(f"Shape mismatch: PyTorch {pt_sorted.shape} vs CUDA {cuda_sorted.shape}")
        return

    # Check for NaN or Inf values
    print(f"PyTorch has NaN: {torch.isnan(pt_sorted).any()}")
    print(f"CUDA has NaN: {torch.isnan(cuda_sorted).any()}")
    print(f"PyTorch has Inf: {torch.isinf(pt_sorted).any()}")
    print(f"CUDA has Inf: {torch.isinf(cuda_sorted).any()}")

    # Check differences
    if not torch.allclose(pt_sorted, cuda_sorted, rtol=1e-5, atol=1e-5):
        diff = torch.abs(pt_sorted - cuda_sorted)
        max_diff = torch.max(diff).item()
        max_diff_idx = torch.argmax(diff.view(-1)).item()
        print(f"Max difference: {max_diff} at index {max_diff_idx}")

        # Find rows with largest differences
        row_diffs = torch.sum(diff, dim=1)
        top_diff_rows = torch.topk(row_diffs, min(5, len(row_diffs)))
        print("Top 5 rows with largest differences:")
        for i, idx in enumerate(top_diff_rows.indices):
            print(f"Row {idx}:")
            print(f"  PyTorch: {pt_sorted[idx]}")
            print(f"  CUDA:    {cuda_sorted[idx]}")
            print(f"  Diff:    {diff[idx]}")

    # Check if indices are different
    if not torch.equal(pt_indices, cuda_indices):
        print("\nIndices don't match")
        print(f"First 10 PyTorch indices: {pt_indices[:10]}")
        print(f"First 10 CUDA indices:    {cuda_indices[:10]}")

        # Check distribution of indices
        print(f"\nPyTorch indices min: {pt_indices.min()}, max: {pt_indices.max()}")
        print(f"CUDA indices min: {cuda_indices.min()}, max: {cuda_indices.max()}")

        # Check uniqueness of indices
        pt_unique = torch.unique(pt_indices)
        cuda_unique = torch.unique(cuda_indices)
        print(f"PyTorch unique indices count: {len(pt_unique)}")
        print(f"CUDA unique indices count: {len(cuda_unique)}")


def main():
    parser = argparse.ArgumentParser(description="Test token sorting implementations")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length")
    parser.add_argument(
        "--hidden-dim", type=int, default=8, help="Hidden dimension size"
    )
    parser.add_argument("--experts", type=int, default=4, help="Number of experts")
    parser.add_argument(
        "--k", type=int, default=2, help="Number of expert assignments per token"
    )
    args = parser.parse_args()

    print("=" * 50)
    print("Token Sorting Tests")
    print("=" * 50)

    # Run the simple test case first
    simple_match = test_simple_case()

    # Run the random test case with configurable dimensions
    random_match = test_random_case(
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        n_experts=args.experts,
        k=args.k,
    )

    if not simple_match or not random_match:
        print("\n⚠️  Some tests failed. Collecting debug information...")

        # Run a debug test case and collect detailed comparison
        device = torch.device("cuda")
        topk_ids = torch.randint(
            0, args.experts, (args.seq_len, args.k), device=device, dtype=torch.int64
        )
        x = torch.randn(args.seq_len, args.hidden_dim, device=device)

        pt_sorted, pt_indices, pt_counts = pytorch_sort_tokens(
            topk_ids, x, args.experts
        )
        cuda_sorted, cuda_indices, cuda_counts = cuda_sort_tokens(
            topk_ids, x, args.experts
        )

        debug_equality(pt_sorted, cuda_sorted, pt_indices, cuda_indices)

    print("\n" + "=" * 50)
    print(f"Simple test result: {'✅ PASS' if simple_match else '❌ FAIL'}")
    print(f"Random test result: {'✅ PASS' if random_match else '❌ FAIL'}")
    print("=" * 50)

    return 0 if simple_match and random_match else 1


if __name__ == "__main__":
    main()
