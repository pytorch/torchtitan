# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys

import torch

# Add PyTorch's library path to system path
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "../../../")
os.environ["LD_LIBRARY_PATH"] = (
    f"{torch_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
)

import time

# Import our CUDA extension
# note - requires torch to go first...this may be another way:
# export LD_LIBRARY_PATH=$(python -c "import torch; print(torch.__path__[0])")/../../../:$LD_LIBRARY_PATH

import moe_permutation

import numpy as np

# import torch


def compute_permutation_indices_pytorch(
    tokens_per_expert_group, experts_per_rank, ep_size, alignment=128, pad_value=-1
):
    """
    PyTorch implementation of compute_permutation_indices for comparison.
    """
    device = tokens_per_expert_group.device
    n_routed_experts = tokens_per_expert_group.shape[0]

    # Compute offsets using prefix sum
    offsets = torch.zeros(
        n_routed_experts + 1, dtype=torch.int32, device=device
    )  # +1 for exclusive scan
    for i in range(n_routed_experts):
        offsets[i + 1] = offsets[i] + tokens_per_expert_group[i]

    # Create indices chunk by chunk - ensure device is specified
    indices = [
        torch.arange(
            offsets[i].item(), offsets[i + 1].item(), device=device, dtype=torch.int32
        )
        for i in range(n_routed_experts)
    ]

    # Print the generated indices for debugging
    print("\nPyTorch generated indices by expert:")
    for i, idx in enumerate(indices):
        print(f"  Expert {i}: {idx}")

    # For each local expert, collect indices from all ranks
    permuted_indices = []
    m_sizes = []

    # Process each local expert
    for e in range(experts_per_rank):
        indices_for_e = []
        len_for_e = 0

        print(f"\nProcessing local expert {e}:")
        # For each remote rank
        for r in range(ep_size):
            expert_idx = r * experts_per_rank + e
            if expert_idx < n_routed_experts:
                print(
                    f"  Adding tokens from rank {r}, expert {expert_idx}: {indices[expert_idx]}"
                )
                indices_for_e.append(indices[expert_idx])
                len_for_e += indices[expert_idx].shape[0]

        # Add padding
        fill_len = (alignment - len_for_e % alignment) % alignment
        fill = torch.full((fill_len,), pad_value, dtype=torch.int32, device=device)
        indices_for_e.append(fill)
        print(f"  Adding {fill_len} padding elements")

        # Store the sizes and indices
        padded_size = len_for_e + fill_len
        m_sizes.append(padded_size)
        concatenated = torch.cat(indices_for_e)
        permuted_indices.append(concatenated)
        print(f"  Total size for expert {e}: {padded_size}")

    # Concatenate all permuted indices and convert sizes to tensor
    permuted_indices = torch.cat(permuted_indices)
    m_sizes = torch.tensor(m_sizes, dtype=torch.int32, device=device)

    return permuted_indices, m_sizes


def apply_permutation_pytorch(token_gather_buf, permuted_indices, output_shape):
    """
    PyTorch implementation of apply_permutation for comparison.
    """
    print("Running PyTorch implementation...")
    print(f"{token_gather_buf.shape=}")
    print(f"{permuted_indices.shape=}")
    print(f"{output_shape=}")

    # Create output tensor with zeros
    permuted_tokens = torch.zeros(
        (output_shape),
        dtype=token_gather_buf.dtype,
        device=token_gather_buf.device,
    )

    # Apply permutation safely - ensure indices are within bounds
    for i in range(len(permuted_indices)):
        idx = permuted_indices[i].item()
        if idx >= 0 and idx < token_gather_buf.shape[0]:
            # permuted_tokens[i] = token_gather_buf[idx]
            if i < permuted_tokens.shape[0]:
                permuted_tokens[i] = token_gather_buf[idx]

    return permuted_tokens


def test_with_controlled_data(use_cuda=True):
    """
    Test with manually controlled data to ensure deterministic behavior.
    """
    print(f"Running controlled test case with {'CUDA' if use_cuda else 'PyTorch'}...")
    device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"

    # Create a very simple test case with small numbers
    experts_per_rank = 2  # 2 experts per rank
    ep_size = 2  # 2 ranks
    alignment = 8  # Smaller alignment for easier debugging

    # Create tokens_per_expert with a small number of tokens per expert
    # Keep total tokens small to avoid any potential issues
    tokens_per_expert = torch.tensor([2, 2, 2, 2], dtype=torch.int32, device=device)

    # Create a simple token buffer
    total_tokens = tokens_per_expert.sum().item()  # Should be 8
    hidden_dim = 4  # Small hidden dimension for simplicity
    token_gather_buf = torch.arange(
        0, total_tokens * hidden_dim, dtype=torch.float32, device=device
    ).reshape(total_tokens, hidden_dim)

    print("Input configuration:")
    print(f"  tokens_per_expert: {tokens_per_expert}")
    print(f"  total_tokens: {total_tokens}")
    print(f"  token shape: {token_gather_buf.shape}")

    if use_cuda:
        # Run CUDA implementation
        print("\nRunning CUDA implementation...")
        try:
            # Try to import the CUDA extension
            import moe_permutation

            has_cuda_extension = True
            print("Successfully imported CUDA extension")

            # Use the CUDA implementation
            permuted_indices, m_sizes = moe_permutation.compute_permutation_indices(
                tokens_per_expert, experts_per_rank, ep_size, alignment, -1
            )

            permuted_tokens = moe_permutation.apply_permutation(
                token_gather_buf, permuted_indices, token_gather_buf.shape
            )
        except ImportError:
            # Fall back to PyTorch implementation if CUDA extension not available
            print("CUDA extension not found, using PyTorch implementation")
            has_cuda_extension = False

            # Use PyTorch implementation
            permuted_indices, m_sizes = compute_permutation_indices_pytorch(
                tokens_per_expert, experts_per_rank, ep_size, alignment, -1
            )

            # Apply permutation
            permuted_tokens = apply_permutation_pytorch(
                token_gather_buf, permuted_indices, token_gather_buf.shape
            )
    else:
        # Run PyTorch implementation for comparison
        print("\nRunning PyTorch implementation...")
        with torch.no_grad():
            permuted_indices, m_sizes = compute_permutation_indices_pytorch(
                tokens_per_expert, experts_per_rank, ep_size, alignment, -1
            )

            permuted_tokens = apply_permutation_pytorch(
                token_gather_buf, permuted_indices, token_gather_buf.shape
            )

    # Print results for examination
    print("\nPermuted indices:", permuted_indices)
    print("m_sizes:", m_sizes)

    # Print a sample of the permuted tokens
    print("\nSample of permuted tokens:")
    print(permuted_tokens[:4])

    return permuted_tokens, m_sizes, permuted_indices


def benchmark(
    batch_size=2,
    seq_len=128,
    hidden_dim=256,
    n_experts=8,
    experts_per_rank=4,
    ep_size=2,
    top_k=2,
    alignment=128,
    num_runs=3,
):
    """
    Benchmark the CUDA implementation against PyTorch.
    """
    print(
        f"Running benchmark with batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # Try to import CUDA extension
        import moe_permutation

        has_cuda_extension = True
        print("CUDA extension available")
    except ImportError:
        has_cuda_extension = False
        print("CUDA extension not available, will only benchmark PyTorch")

    # Create test data with evenly distributed tokens to avoid out-of-bounds
    tokens_per_expert = torch.ones(n_experts, dtype=torch.int32, device=device) * (
        batch_size * seq_len * top_k // n_experts
    )

    # Make sure the total is exactly what we want
    remainder = batch_size * seq_len * top_k - tokens_per_expert.sum().item()
    if remainder > 0:
        # Add remaining tokens to the first few experts
        for i in range(remainder):
            tokens_per_expert[i % n_experts] += 1

    total_tokens = tokens_per_expert.sum().item()
    token_gather_buf = torch.randn(total_tokens, hidden_dim, device=device)

    print(
        f"Created test data with {total_tokens} total tokens distributed across {n_experts} experts"
    )

    # PyTorch implementation
    pytorch_times = []
    torch.cuda.synchronize()  # Ensure GPU is synchronized before timing

    for run in range(num_runs):
        print(f"PyTorch run {run+1}/{num_runs}...")
        start = time.time()
        with torch.no_grad():
            permuted_indices, m_sizes = compute_permutation_indices_pytorch(
                tokens_per_expert, experts_per_rank, ep_size, alignment, -1
            )

            permuted_tokens = apply_permutation_pytorch(
                token_gather_buf, permuted_indices, token_gather_buf.shape
            )
        torch.cuda.synchronize()  # Ensure GPU is done before stopping timer
        end = time.time()
        pytorch_times.append(end - start)

    # Calculate PyTorch statistics
    pytorch_mean = np.mean(pytorch_times)
    pytorch_std = np.std(pytorch_times)

    print(
        f"PyTorch implementation: {pytorch_mean*1000:.2f}ms ± {pytorch_std*1000:.2f}ms"
    )

    # CUDA implementation (only if extension is available)
    if has_cuda_extension:
        cuda_times = []
        for run in range(num_runs):
            print(f"CUDA run {run+1}/{num_runs}...")
            torch.cuda.synchronize()
            start = time.time()

            permuted_tokens, m_sizes = moe_permutation.optimized_token_permutation(
                token_gather_buf,
                tokens_per_expert,
                experts_per_rank,
                ep_size,
                alignment,
            )

            torch.cuda.synchronize()
            end = time.time()
            cuda_times.append(end - start)

        # Calculate CUDA statistics
        cuda_mean = np.mean(cuda_times)
        cuda_std = np.std(cuda_times)

        print(f"CUDA implementation: {cuda_mean*1000:.2f}ms ± {cuda_std*1000:.2f}ms")
        print(f"Speedup: {pytorch_mean / cuda_mean:.2f}x")
    else:
        print("CUDA implementation skipped - extension not available")


if __name__ == "__main__":
    # Parse command-line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Test MoE token permutation")
    parser.add_argument("--test", action="store_true", help="Run controlled test")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Batch size for benchmark"
    )
    parser.add_argument(
        "--seq-len", type=int, default=128, help="Sequence length for benchmark"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=256, help="Hidden dimension for benchmark"
    )
    args = parser.parse_args()

    # Set CUDA device if available
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU")

    # Run tests based on command-line arguments
    if args.test or (not args.benchmark and not args.test):
        print("\n========== Running controlled test with PyTorch ==========")
        pt_tokens, pt_m_sizes, pt_indices = test_with_controlled_data(use_cuda=False)

        print("\n========== Running controlled test with CUDA ==========")
        cuda_tokens, cuda_m_sizes, cuda_indices = test_with_controlled_data(
            use_cuda=True
        )

        # Compare results
        # pt_indices.to(cuda_indices.device)

        # Compare results
        indices_match = torch.all(
            pt_indices.to(cuda_indices.device) == cuda_indices[: len(pt_indices)]
        )
        m_sizes_match = torch.all(pt_m_sizes.to(cuda_m_sizes.device) == cuda_m_sizes)
        tokens_match = torch.allclose(pt_tokens.to(cuda_tokens.device), cuda_tokens)

        # print(f"verify pti: {pt_indices.device=}, {cuda_indices.device}")
        # indices_match = torch.all(pt_indices == cuda_indices[: len(pt_indices)])
        # m_sizes_match = torch.all(pt_m_sizes == cuda_m_sizes)
        # tokens_match = torch.allclose(pt_tokens, cuda_tokens)

        print("\n========== Comparison ==========")
        print(f"Indices match: {indices_match}")
        print(f"m_sizes match: {m_sizes_match}")
        print(f"Tokens match: {tokens_match}")

    if args.benchmark:
        print("\n========== Running benchmark ==========")
        benchmark(
            batch_size=args.batch_size, seq_len=args.seq_len, hidden_dim=args.hidden_dim
        )

    print("\nDone!")
