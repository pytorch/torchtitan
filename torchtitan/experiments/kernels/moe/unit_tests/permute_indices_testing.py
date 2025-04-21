# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from typing import Tuple

import numpy as np
import torch

try:
    from torchtitan.experiments.kernels.moe.indices import fill_indices_wrapper

except ImportError:
    print("unable to import required function(s) from indices.py...")
    raise


def fill_indices_cpu(
    tokens_per_expert_group: torch.Tensor,
    start_index_values: torch.Tensor,
    write_offsets: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
):
    """CPU implementation for reference."""
    # We need to preallocate the output
    device = tokens_per_expert_group.device
    permuted_indices = torch.full((max_len,), -1, dtype=torch.int32, device=device)
    # Fill the permuted indices
    # For each local expert
    for e in range(experts_per_rank):
        write_start = write_offsets[e].item()
        # For each remote rank
        for r in range(num_ranks):
            i = r * experts_per_rank + e
            start_index = start_index_values[i].item()
            length = tokens_per_expert_group[i].item()
            # Fill in the indices
            if length > 0:
                end_idx = min(write_start + length, max_len)
                # Add this check to prevent empty ranges
                if write_start < end_idx:
                    permuted_indices[write_start:end_idx] = torch.arange(
                        start_index,
                        start_index + (end_idx - write_start),
                        dtype=torch.int32,
                        device=device,
                    )
            write_start += length
    return permuted_indices


class TestOptimizedKernel(unittest.TestCase):
    """Test cases for the permute Triton kernel."""

    def setUp(self):
        """Set up common test parameters."""
        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            self.skipTest("WARNING:  No GPU available, skipping Triton kernel tests")

        # Set random seed for reproducible tests
        torch.manual_seed(2020)
        np.random.seed(2020)

        # Total experts to test
        self.total_experts = 256

    def create_test_data(
        self,
        experts_per_rank: int,
        num_ranks: int,
        token_range: Tuple[int, int] = (1, 16),
        alignment: int = 32,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Create test data"""
        # Create token counts
        tokens_per_expert_group = torch.randint(
            token_range[0],
            token_range[1] + 1,
            (num_ranks * experts_per_rank,),
            dtype=torch.int32,
            device=self.device,
        )

        # Calc start indices
        start_index_values = (
            torch.cumsum(tokens_per_expert_group, 0) - tokens_per_expert_group
        )

        # Calcu chunk sizes
        chunk_size_per_expert = tokens_per_expert_group.view(num_ranks, -1).sum(0)
        m_sizes = ((chunk_size_per_expert + alignment - 1) // alignment * alignment).to(
            torch.int32
        )

        # Calc write offsets
        write_offsets = torch.cumsum(m_sizes, 0) - m_sizes

        # Calcu max_len
        total_tokens = tokens_per_expert_group.sum().item()
        max_len = total_tokens * 2  # Give some extra space

        return tokens_per_expert_group, start_index_values, write_offsets, max_len

    def test_fixed_total_experts_varying_ranks(self):
        """Test with 256 total experts (deepseek v3) distributed across different numbers of ranks."""
        # Test with different rank configurations
        rank_configs = [1, 2, 4, 8, 16, 32, 64, 128]

        for num_ranks in rank_configs:
            # Calculate experts per rank to maintain fixed total experts
            experts_per_rank = self.total_experts // num_ranks

            # Skip if experts_per_rank is less than 1
            if experts_per_rank < 1:
                continue

            # Actual total experts (may differ slightly due to integer division)
            actual_total_experts = experts_per_rank * num_ranks

            with self.subTest(
                f"ranks={num_ranks}, experts_per_rank={experts_per_rank}, total={actual_total_experts}"
            ):
                print(
                    f"Testing {experts_per_rank} experts per rank across {num_ranks} ranks (total: {actual_total_experts})"
                )

                (
                    tokens_per_expert_group,
                    start_index_values,
                    write_offsets,
                    max_len,
                ) = self.create_test_data(experts_per_rank, num_ranks)

                # Run CPU implementation
                cpu_result = fill_indices_cpu(
                    tokens_per_expert_group,
                    start_index_values,
                    write_offsets,
                    experts_per_rank,
                    num_ranks,
                    max_len,
                )

                # Run optimized implementation
                optimized_result = fill_indices_wrapper(
                    tokens_per_expert_group,
                    start_index_values,
                    write_offsets,
                    experts_per_rank,
                    num_ranks,
                    max_len,
                    block_size=128,
                )

                # Verify results match
                self.assertTrue(
                    torch.equal(cpu_result, optimized_result),
                    f"Triton kernel output does not match CPU implementation for "
                    f"{experts_per_rank} experts per rank, {num_ranks} ranks",
                )

                # Check that all valid positions match
                valid_positions = cpu_result != -1
                if valid_positions.sum() > 0:
                    valid_cpu = cpu_result[valid_positions]
                    valid_optimized = optimized_result[valid_positions]
                    self.assertTrue(
                        torch.equal(valid_cpu, valid_optimized),
                        f"Mismatch in valid positions for {experts_per_rank} experts per rank, {num_ranks} ranks",
                    )

    def test_different_block_sizes_with_fixed_experts(self):
        """Test different block sizes with fixed total experts."""
        # Use a moderate rank configuration
        num_ranks = 8
        experts_per_rank = self.total_experts // num_ranks  # 32 experts per rank

        (
            tokens_per_expert_group,
            start_index_values,
            write_offsets,
            max_len,
        ) = self.create_test_data(experts_per_rank, num_ranks)

        # Run CPU implementation for reference
        cpu_result = fill_indices_cpu(
            tokens_per_expert_group,
            start_index_values,
            write_offsets,
            experts_per_rank,
            num_ranks,
            max_len,
        )

        # Test different block sizes
        block_sizes = [32, 64, 128, 256, 512]
        for block_size in block_sizes:
            with self.subTest(f"block_size={block_size}"):
                optimized_result = fill_indices_wrapper(
                    tokens_per_expert_group,
                    start_index_values,
                    write_offsets,
                    experts_per_rank,
                    num_ranks,
                    max_len,
                    block_size=block_size,
                )

                # Verify results match
                self.assertTrue(
                    torch.equal(cpu_result, optimized_result),
                    f"Block size {block_size} produces incorrect results",
                )

    def test_edge_cases_with_fixed_experts(self):
        """Test edge cases with configurations that maintain fixed total experts."""
        # For fixed total experts = 256
        # Test with different token ranges
        token_ranges = [(1, 1), (0, 0), (16, 16), (1, 32)]

        num_ranks = 8
        experts_per_rank = self.total_experts // num_ranks  # 32 experts per rank

        for token_range in token_ranges:
            with self.subTest(f"token_range={token_range}"):
                (
                    tokens_per_expert_group,
                    start_index_values,
                    write_offsets,
                    max_len,
                ) = self.create_test_data(experts_per_rank, num_ranks, token_range)

                # Run CPU implementation
                cpu_result = fill_indices_cpu(
                    tokens_per_expert_group,
                    start_index_values,
                    write_offsets,
                    experts_per_rank,
                    num_ranks,
                    max_len,
                )

                # Run optimized implementation
                optimized_result = fill_indices_wrapper(
                    tokens_per_expert_group,
                    start_index_values,
                    write_offsets,
                    experts_per_rank,
                    num_ranks,
                    max_len,
                    block_size=128,
                )

                # Verify results match
                self.assertTrue(
                    torch.equal(cpu_result, optimized_result),
                    f"Triton kernel failed for token range {token_range}",
                )

    def test_max_blocks_with_large_experts(self):
        """Test cases where the number of experts exceeds the maximum number of blocks.

        This test verifies that the kernel correctly processes all experts even when
        the number of experts is significantly larger than the maximum blocks allowed,
        forcing multiple experts to be processed by each block.
        """
        # Test configurations where experts greatly exceed max_blocks
        test_configs = [
            {
                "experts_per_rank": 512,
                "num_ranks": 4,
                "max_blocks": 64,
            },  # 8 experts per block
            {
                "experts_per_rank": 1024,
                "num_ranks": 2,
                "max_blocks": 32,
            },  # 32 experts per block
            {
                "experts_per_rank": 2048,
                "num_ranks": 1,
                "max_blocks": 16,
            },  # 128 experts per block
        ]

        for config in test_configs:
            experts_per_rank = config["experts_per_rank"]
            num_ranks = config["num_ranks"]
            max_blocks = config["max_blocks"]

            # Calculate experts per block for reporting
            experts_per_block = (experts_per_rank + max_blocks - 1) // max_blocks

            with self.subTest(
                f"experts_per_rank={experts_per_rank}, num_ranks={num_ranks}, "
                f"max_blocks={max_blocks}, experts_per_block={experts_per_block}"
            ):
                print(
                    f"Testing with {experts_per_rank} experts per rank, {num_ranks} ranks, "
                    f"max_blocks={max_blocks} (approx. {experts_per_block} experts per block)"
                )

                # Create test data
                (
                    tokens_per_expert_group,
                    start_index_values,
                    write_offsets,
                    max_len,
                ) = self.create_test_data(
                    experts_per_rank,
                    num_ranks,
                    token_range=(
                        1,
                        8,
                    ),  # Use smaller token range for large expert counts
                )

                # Run CPU implementation for reference
                cpu_result = fill_indices_cpu(
                    tokens_per_expert_group,
                    start_index_values,
                    write_offsets,
                    experts_per_rank,
                    num_ranks,
                    max_len,
                )

                # Run optimized implementation with max_blocks cap
                optimized_result = fill_indices_wrapper(
                    tokens_per_expert_group,
                    start_index_values,
                    write_offsets,
                    experts_per_rank,
                    num_ranks,
                    max_len,
                    block_size=128,
                    max_blocks=max_blocks,
                )

                # Verify results match
                self.assertTrue(
                    torch.equal(cpu_result, optimized_result),
                    f"Triton kernel output doesn't match CPU implementation with max_blocks={max_blocks}",
                )

                # Additional verification for valid entries
                valid_positions = cpu_result != -1
                if valid_positions.sum() > 0:
                    valid_cpu = cpu_result[valid_positions]
                    valid_optimized = optimized_result[valid_positions]
                    self.assertTrue(
                        torch.equal(valid_cpu, valid_optimized),
                        f"Mismatch in valid positions with max_blocks={max_blocks}",
                    )

    def test_extreme_max_blocks_limit(self):
        """Test with extremely small max_blocks limits to stress-test the looping mechanism."""
        # Use moderately large number of experts
        experts_per_rank = 256
        num_ranks = 4

        # Test with extremely small max_blocks values
        max_blocks_values = [8, 4, 2, 1]  # Test down to just a single block

        for max_blocks in max_blocks_values:
            experts_per_block = (experts_per_rank + max_blocks - 1) // max_blocks

            with self.subTest(
                f"experts_per_rank={experts_per_rank}, max_blocks={max_blocks}, "
                f"experts_per_block={experts_per_block}"
            ):
                print(
                    f"Testing extreme case with {experts_per_rank} experts per rank, "
                    f"max_blocks={max_blocks} (approx. {experts_per_block} experts per block)"
                )

                # Create test data
                (
                    tokens_per_expert_group,
                    start_index_values,
                    write_offsets,
                    max_len,
                ) = self.create_test_data(experts_per_rank, num_ranks)

                # Run CPU implementation for reference
                cpu_result = fill_indices_cpu(
                    tokens_per_expert_group,
                    start_index_values,
                    write_offsets,
                    experts_per_rank,
                    num_ranks,
                    max_len,
                )

                # Run optimized implementation with extremely limited max_blocks
                optimized_result = fill_indices_wrapper(
                    tokens_per_expert_group,
                    start_index_values,
                    write_offsets,
                    experts_per_rank,
                    num_ranks,
                    max_len,
                    block_size=128,
                    max_blocks=max_blocks,
                )

                # Verify results match
                self.assertTrue(
                    torch.equal(cpu_result, optimized_result),
                    f"Triton kernel failed with extreme max_blocks limit of {max_blocks}",
                )


if __name__ == "__main__":
    unittest.main()
