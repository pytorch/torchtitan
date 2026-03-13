import os
import sys
import time
import unittest
from typing import List, Tuple

import numpy as np
import torch


"""
perf and current failures:

Performance Test:
Test data: 8x512 sequence, 1024 dims, 16 experts
Total tokens: 8192
PyTorch: 136.76 ± 1.29 ms
CUDA:    0.14 ± 0.05 ms
Speedup: 951.89x

======================================================================
ERROR: test_various_alignments (unit_test.TestMoEPermutationCUDA.test_various_alignments)
Test with various alignment values
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/data/users/less/torchtitan/torchtitan/experiments/kernels/contiguous_group_gemm/permute/unit_test.py", line 436, in test_various_alignments
    torch.all(ref_indices[valid_mask] == cuda_indices[valid_mask]),
                                         ~~~~~~~~~~~~^^^^^^^^^^^^
IndexError: The shape of the mask [16] at index 0 does not match the shape of the indexed tensor [24] at index 0

======================================================================
FAIL: test_compute_permutation_indices_medium (unit_test.TestMoEPermutationCUDA.test_compute_permutation_indices_medium)
Test compute_permutation_indices with medium-sized inputs
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/data/users/less/torchtitan/torchtitan/experiments/kernels/contiguous_group_gemm/permute/unit_test.py", line 247, in test_compute_permutation_indices_medium
    self.assertEqual(
AssertionError: torch.Size([768]) != torch.Size([1152]) : Indices shape mismatch

======================================================================
FAIL: test_compute_permutation_indices_small (unit_test.TestMoEPermutationCUDA.test_compute_permutation_indices_small)
Test compute_permutation_indices with small inputs
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/data/users/less/torchtitan/torchtitan/experiments/kernels/contiguous_group_gemm/permute/unit_test.py", line 203, in test_compute_permutation_indices_small
    self.assertEqual(
AssertionError: torch.Size([16]) != torch.Size([24]) : Indices shape mismatch

----------------------------------------------------------------------
Ran 9 tests in 3.034s

FAILED (failures=2, errors=1)

"""


# Add PyTorch's library path to system path
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "../../../")
os.environ["LD_LIBRARY_PATH"] = (
    f"{torch_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
)

# Import the CUDA extension - will fail if not available
import moe_permutation


def compute_permutation_indices_pytorch(
    tokens_per_expert_group: torch.Tensor,
    experts_per_rank: int,
    ep_size: int,
    alignment: int = 128,
    pad_value: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch reference implementation of compute_permutation_indices.
    """
    device = tokens_per_expert_group.device
    n_routed_experts = tokens_per_expert_group.shape[0]

    # Compute offsets using prefix sum (exclusive scan)
    offsets = torch.zeros(n_routed_experts + 1, dtype=torch.int32, device=device)
    for i in range(n_routed_experts):
        offsets[i + 1] = offsets[i] + tokens_per_expert_group[i]

    # Create indices chunk by chunk
    indices = [
        torch.arange(
            offsets[i].item(), offsets[i + 1].item(), device=device, dtype=torch.int32
        )
        for i in range(n_routed_experts)
    ]

    # For each local expert, collect indices from all ranks
    permuted_indices = []
    m_sizes = []

    # Process each local expert
    for e in range(experts_per_rank):
        indices_for_e = []
        len_for_e = 0

        # For each remote rank
        for r in range(ep_size):
            expert_idx = r * experts_per_rank + e
            if expert_idx < n_routed_experts:
                indices_for_e.append(indices[expert_idx])
                len_for_e += indices[expert_idx].shape[0]

        # Add padding
        fill_len = (alignment - len_for_e % alignment) % alignment
        fill = torch.full((fill_len,), pad_value, dtype=torch.int32, device=device)
        indices_for_e.append(fill)

        # Store the sizes and indices
        padded_size = len_for_e + fill_len
        m_sizes.append(padded_size)
        concatenated = torch.cat(indices_for_e)
        permuted_indices.append(concatenated)

    # Concatenate all permuted indices and convert sizes to tensor
    permuted_indices = torch.cat(permuted_indices)
    m_sizes = torch.tensor(m_sizes, dtype=torch.int32, device=device)

    return permuted_indices, m_sizes


def apply_permutation_pytorch(
    token_gather_buf: torch.Tensor,
    permuted_indices: torch.Tensor,
    output_shape: torch.Size,
) -> torch.Tensor:
    """
    PyTorch reference implementation of apply_permutation.
    """
    # Create output tensor with zeros
    permuted_tokens = torch.zeros(
        output_shape, dtype=token_gather_buf.dtype, device=token_gather_buf.device
    )

    # Apply permutation safely
    for i in range(min(len(permuted_indices), permuted_tokens.shape[0])):
        idx = permuted_indices[i].item()
        if idx >= 0 and idx < token_gather_buf.shape[0]:
            permuted_tokens[i] = token_gather_buf[idx]

    return permuted_tokens


def create_test_data(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create realistic test data for MoE token routing.

    Returns:
        Tuple of (token_gather_buf, tokens_per_expert)
    """
    # Create a distribution of tokens per expert
    tokens_per_expert = torch.ones(num_experts, dtype=torch.int32, device=device)

    # Calculate expected total tokens
    total_tokens = batch_size * seq_len * top_k

    # Distribute tokens evenly, then add remainder to first experts
    tokens_per_expert.fill_(total_tokens // num_experts)
    remainder = total_tokens - (total_tokens // num_experts) * num_experts
    if remainder > 0:
        tokens_per_expert[:remainder] += 1

    # Verify the total
    assert (
        tokens_per_expert.sum().item() == total_tokens
    ), f"Expected {total_tokens} tokens, got {tokens_per_expert.sum().item()}"

    # Create token buffer with sequential values for easy verification
    token_gather_buf = torch.arange(
        0, total_tokens * hidden_dim, device=device, dtype=torch.float32
    ).reshape(total_tokens, hidden_dim)

    return token_gather_buf, tokens_per_expert


class TestMoEPermutationCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Check if CUDA is available - fail if not
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required for these tests")

        # Set up CUDA
        torch.cuda.set_device(0)
        cls.device = torch.device("cuda:0")
        print(f"Running tests on: {torch.cuda.get_device_name(0)}")

    def setUp(self):
        # Small test case for quick validation
        self.tokens_per_expert_small = torch.tensor(
            [2, 2, 2, 2], dtype=torch.int32, device=self.device
        )
        self.experts_per_rank_small = 2
        self.ep_size_small = 2
        self.alignment_small = 8

        # Create corresponding token buffer
        total_tokens_small = self.tokens_per_expert_small.sum().item()
        self.token_gather_buf_small = torch.arange(
            0, total_tokens_small * 4, device=self.device, dtype=torch.float32
        ).reshape(total_tokens_small, 4)

        # Medium-sized test case (realistic sizes)
        self.tokens_per_expert_med = torch.tensor(
            [128, 64, 96, 32, 64, 128, 32, 96], dtype=torch.int32, device=self.device
        )
        self.experts_per_rank_med = 4
        self.ep_size_med = 2
        self.alignment_med = 128

        # Create corresponding token buffer
        total_tokens_med = self.tokens_per_expert_med.sum().item()
        self.token_gather_buf_med = torch.randn(
            total_tokens_med, 64, device=self.device, dtype=torch.float32
        )

    def test_compute_permutation_indices_small(self):
        """Test compute_permutation_indices with small inputs"""
        # Reference implementation
        ref_indices, ref_m_sizes = compute_permutation_indices_pytorch(
            self.tokens_per_expert_small,
            self.experts_per_rank_small,
            self.ep_size_small,
            self.alignment_small,
            -1,
        )

        # CUDA implementation
        cuda_indices, cuda_m_sizes = moe_permutation.compute_permutation_indices(
            self.tokens_per_expert_small,
            self.experts_per_rank_small,
            self.ep_size_small,
            self.alignment_small,
            -1,
        )

        # Check that shapes match
        self.assertEqual(
            ref_indices.shape, cuda_indices.shape, "Indices shape mismatch"
        )
        self.assertEqual(
            ref_m_sizes.shape, cuda_m_sizes.shape, "M_sizes shape mismatch"
        )

        # Check content equality for valid indices (not padding)
        valid_mask = ref_indices >= 0
        self.assertTrue(
            torch.all(ref_indices[valid_mask] == cuda_indices[valid_mask]),
            "Valid indices values don't match",
        )

        # Check that padding is in the same positions
        self.assertTrue(
            torch.all((ref_indices == -1) == (cuda_indices == -1)),
            "Padding positions don't match",
        )

        # Check m_sizes
        self.assertTrue(torch.all(ref_m_sizes == cuda_m_sizes), "M_sizes don't match")

    def test_compute_permutation_indices_medium(self):
        """Test compute_permutation_indices with medium-sized inputs"""
        # Reference implementation
        ref_indices, ref_m_sizes = compute_permutation_indices_pytorch(
            self.tokens_per_expert_med,
            self.experts_per_rank_med,
            self.ep_size_med,
            self.alignment_med,
            -1,
        )

        # CUDA implementation
        cuda_indices, cuda_m_sizes = moe_permutation.compute_permutation_indices(
            self.tokens_per_expert_med,
            self.experts_per_rank_med,
            self.ep_size_med,
            self.alignment_med,
            -1,
        )

        # Check that shapes match
        self.assertEqual(
            ref_indices.shape, cuda_indices.shape, "Indices shape mismatch"
        )
        self.assertEqual(
            ref_m_sizes.shape, cuda_m_sizes.shape, "M_sizes shape mismatch"
        )

        # Check content equality for valid indices (not padding)
        valid_mask = ref_indices >= 0
        self.assertTrue(
            torch.all(ref_indices[valid_mask] == cuda_indices[valid_mask]),
            "Valid indices values don't match",
        )

        # Check m_sizes
        self.assertTrue(torch.all(ref_m_sizes == cuda_m_sizes), "M_sizes don't match")

    def test_apply_permutation_small(self):
        """Test apply_permutation with small inputs"""
        # Get indices from reference implementation
        ref_indices, _ = compute_permutation_indices_pytorch(
            self.tokens_per_expert_small,
            self.experts_per_rank_small,
            self.ep_size_small,
            self.alignment_small,
            -1,
        )

        # Reference implementation
        ref_tokens = apply_permutation_pytorch(
            self.token_gather_buf_small, ref_indices, self.token_gather_buf_small.shape
        )

        # CUDA implementation
        cuda_tokens = moe_permutation.apply_permutation(
            self.token_gather_buf_small,
            ref_indices,  # Use the same indices from PyTorch for fair comparison
            self.token_gather_buf_small.shape,
        )

        # Check shape match
        self.assertEqual(ref_tokens.shape, cuda_tokens.shape, "Tokens shape mismatch")

        # Check content match
        self.assertTrue(
            torch.allclose(ref_tokens, cuda_tokens, rtol=1e-5, atol=1e-5),
            "Permuted tokens don't match",
        )

    def test_apply_permutation_medium(self):
        """Test apply_permutation with medium-sized inputs"""
        # Get indices from reference implementation
        ref_indices, _ = compute_permutation_indices_pytorch(
            self.tokens_per_expert_med,
            self.experts_per_rank_med,
            self.ep_size_med,
            self.alignment_med,
            -1,
        )

        # Reference implementation
        ref_tokens = apply_permutation_pytorch(
            self.token_gather_buf_med, ref_indices, self.token_gather_buf_med.shape
        )

        # CUDA implementation
        cuda_tokens = moe_permutation.apply_permutation(
            self.token_gather_buf_med,
            ref_indices,  # Use the same indices from PyTorch for fair comparison
            self.token_gather_buf_med.shape,
        )

        # Check shape match
        self.assertEqual(ref_tokens.shape, cuda_tokens.shape, "Tokens shape mismatch")

        # Check content match with tolerance
        self.assertTrue(
            torch.allclose(ref_tokens, cuda_tokens, rtol=1e-5, atol=1e-5),
            "Permuted tokens don't match",
        )

    def test_optimized_token_permutation_small(self):
        """Test the full optimized_token_permutation pipeline with small inputs"""
        # Reference implementation
        ref_indices, ref_m_sizes = compute_permutation_indices_pytorch(
            self.tokens_per_expert_small,
            self.experts_per_rank_small,
            self.ep_size_small,
            self.alignment_small,
            -1,
        )

        ref_tokens = apply_permutation_pytorch(
            self.token_gather_buf_small, ref_indices, self.token_gather_buf_small.shape
        )

        # CUDA implementation
        cuda_tokens, cuda_m_sizes = moe_permutation.optimized_token_permutation(
            self.token_gather_buf_small,
            self.tokens_per_expert_small,
            self.experts_per_rank_small,
            self.ep_size_small,
            self.alignment_small,
        )

        # Check shapes match
        self.assertEqual(ref_tokens.shape, cuda_tokens.shape, "Tokens shape mismatch")
        self.assertEqual(
            ref_m_sizes.shape, cuda_m_sizes.shape, "M_sizes shape mismatch"
        )

        # Check content match
        self.assertTrue(
            torch.allclose(ref_tokens, cuda_tokens, rtol=1e-5, atol=1e-5),
            "Permuted tokens don't match",
        )

        # Check m_sizes
        self.assertTrue(torch.all(ref_m_sizes == cuda_m_sizes), "M_sizes don't match")

    def test_optimized_token_permutation_medium(self):
        """Test the full optimized_token_permutation pipeline with medium inputs"""
        # Reference implementation
        ref_indices, ref_m_sizes = compute_permutation_indices_pytorch(
            self.tokens_per_expert_med,
            self.experts_per_rank_med,
            self.ep_size_med,
            self.alignment_med,
            -1,
        )

        ref_tokens = apply_permutation_pytorch(
            self.token_gather_buf_med, ref_indices, self.token_gather_buf_med.shape
        )

        # CUDA implementation
        cuda_tokens, cuda_m_sizes = moe_permutation.optimized_token_permutation(
            self.token_gather_buf_med,
            self.tokens_per_expert_med,
            self.experts_per_rank_med,
            self.ep_size_med,
            self.alignment_med,
        )

        # Check shapes match
        self.assertEqual(ref_tokens.shape, cuda_tokens.shape, "Tokens shape mismatch")
        self.assertEqual(
            ref_m_sizes.shape, cuda_m_sizes.shape, "M_sizes shape mismatch"
        )

        # Check content match (with some tolerance for floating point)
        self.assertTrue(
            torch.allclose(ref_tokens, cuda_tokens, rtol=1e-5, atol=1e-5),
            "Permuted tokens don't match",
        )

        # Check m_sizes
        self.assertTrue(torch.all(ref_m_sizes == cuda_m_sizes), "M_sizes don't match")

    def test_various_alignments(self):
        """Test with various alignment values"""
        for alignment in [8, 16, 32, 64, 128]:
            # Reference implementation
            ref_indices, ref_m_sizes = compute_permutation_indices_pytorch(
                self.tokens_per_expert_small,
                self.experts_per_rank_small,
                self.ep_size_small,
                alignment,
                -1,
            )

            # CUDA implementation
            cuda_indices, cuda_m_sizes = moe_permutation.compute_permutation_indices(
                self.tokens_per_expert_small,
                self.experts_per_rank_small,
                self.ep_size_small,
                alignment,
                -1,
            )

            # Check m_sizes match
            self.assertTrue(
                torch.all(ref_m_sizes == cuda_m_sizes),
                f"M_sizes don't match for alignment={alignment}",
            )

            # Check valid indices match
            valid_mask = ref_indices >= 0
            self.assertTrue(
                torch.all(ref_indices[valid_mask] == cuda_indices[valid_mask]),
                f"Valid indices don't match for alignment={alignment}",
            )

    def test_performance(self):
        """Benchmark test comparing PyTorch and CUDA implementations"""
        print("\nPerformance Test:")

        # Generate larger test data
        batch_size = 8
        seq_len = 512
        hidden_dim = 1024
        num_experts = 16
        experts_per_rank = 8
        ep_size = 2
        top_k = 2
        alignment = 128
        num_runs = 5

        # Create test data
        token_gather_buf, tokens_per_expert = create_test_data(
            batch_size, seq_len, hidden_dim, num_experts, top_k, self.device
        )

        print(
            f"Test data: {batch_size}x{seq_len} sequence, {hidden_dim} dims, {num_experts} experts"
        )
        print(f"Total tokens: {token_gather_buf.shape[0]}")

        # Benchmark PyTorch implementation
        torch.cuda.synchronize()
        pytorch_times = []

        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.time()

            with torch.no_grad():
                permuted_indices, m_sizes = compute_permutation_indices_pytorch(
                    tokens_per_expert, experts_per_rank, ep_size, alignment, -1
                )

                _ = apply_permutation_pytorch(
                    token_gather_buf, permuted_indices, token_gather_buf.shape
                )

            torch.cuda.synchronize()
            end = time.time()
            pytorch_times.append(end - start)

        # Benchmark CUDA implementation
        torch.cuda.synchronize()
        cuda_times = []

        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.time()

            _, _ = moe_permutation.optimized_token_permutation(
                token_gather_buf,
                tokens_per_expert,
                experts_per_rank,
                ep_size,
                alignment,
            )

            torch.cuda.synchronize()
            end = time.time()
            cuda_times.append(end - start)

        # Calculate statistics
        pt_mean = np.mean(pytorch_times) * 1000  # ms
        pt_std = np.std(pytorch_times) * 1000  # ms
        cuda_mean = np.mean(cuda_times) * 1000  # ms
        cuda_std = np.std(cuda_times) * 1000  # ms
        speedup = pt_mean / cuda_mean

        print(f"PyTorch: {pt_mean:.2f} ± {pt_std:.2f} ms")
        print(f"CUDA:    {cuda_mean:.2f} ± {cuda_std:.2f} ms")
        print(f"Speedup: {speedup:.2f}x")

        # Assert CUDA is faster (at least 2x)
        self.assertGreater(
            speedup, 2.0, "CUDA implementation should be at least 2x faster"
        )

    def test_large_scale(self):
        """Test with larger models to ensure CUDA kernels can handle realistic loads"""
        # Skip if less than 8GB VRAM to avoid OOM
        if torch.cuda.get_device_properties(0).total_memory < 8 * 1024 * 1024 * 1024:
            print("\nSkipping large scale test due to insufficient GPU memory")
            return

        print("\nLarge Scale Test:")

        # Large test case (LLM scale)
        batch_size = 1
        seq_len = 2048
        hidden_dim = 4096
        num_experts = 64
        experts_per_rank = 8
        ep_size = 8
        top_k = 2
        alignment = 128

        # Create test data (evenly distributed tokens)
        token_gather_buf, tokens_per_expert = create_test_data(
            batch_size, seq_len, hidden_dim, num_experts, top_k, self.device
        )

        print(
            f"Running large scale test with {token_gather_buf.shape[0]} tokens, hidden_dim={hidden_dim}"
        )

        # Test CUDA implementation
        try:
            # This should not error out
            permuted_tokens, m_sizes = moe_permutation.optimized_token_permutation(
                token_gather_buf,
                tokens_per_expert,
                experts_per_rank,
                ep_size,
                alignment,
            )

            # Check basic correctness
            self.assertEqual(
                permuted_tokens.shape,
                token_gather_buf.shape,
                "Permuted tokens shape should match input shape",
            )
            self.assertEqual(
                m_sizes.shape[0],
                experts_per_rank,
                "M_sizes should have length equal to experts_per_rank",
            )

            print("Large scale test passed successfully")
        except Exception as e:
            self.fail(f"Large scale test failed with error: {str(e)}")


if __name__ == "__main__":
    unittest.main()
