# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import triton
import triton.language as tl


# parallelized kernel
@triton.jit
def _fill_indices_kernel(
    tokens_per_expert_group_ptr,
    start_index_values_ptr,
    write_offsets_ptr,
    output_ptr,
    experts_per_rank,
    num_ranks,
    BLOCK_SIZE: tl.constexpr,  # Number of threads per block
):
    pid = tl.program_id(axis=0)

    # each program handles one expert
    expert_id = pid

    # only process if valid expert
    if expert_id < experts_per_rank:
        # read this experts write offset
        write_offset = tl.load(write_offsets_ptr + expert_id)

        # loop over all ranks
        for r in range(num_ranks):
            # index into tokens_per_expert_group array
            i = r * experts_per_rank + expert_id

            # load start index and number of tokens for this expert-rank pair
            start_index = tl.load(start_index_values_ptr + i)
            length = tl.load(tokens_per_expert_group_ptr + i)

            # each thread in block processes tokens in parallel
            offsets = tl.arange(0, BLOCK_SIZE)

            # tokens are processed in chunks of BLOCK_SIZE
            for chunk_start in range(0, length, BLOCK_SIZE):
                chunk_offsets = chunk_start + offsets

                # mask valid indices
                mask = chunk_offsets < length

                values = start_index + chunk_offsets

                # destination
                dest_indices = write_offset + chunk_offsets

                # store
                tl.store(output_ptr + dest_indices, values, mask=mask)

            # update write offset for next rank
            write_offset += length


# ==============
# wrapper
# ==============


def fill_indices_wrapper(
    tokens_per_expert_group: torch.Tensor,
    start_index_values: torch.Tensor,
    write_offsets: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
    block_size: int = 128,
):
    # preallocate output
    permuted_indices = torch.full(
        (max_len,), -1, dtype=torch.int32, device=tokens_per_expert_group.device
    )

    # grid = one block per expert
    grid = (experts_per_rank,)

    # launch kernel
    _fill_indices_kernel[grid](
        tokens_per_expert_group,
        start_index_values,
        write_offsets,
        permuted_indices,
        experts_per_rank,
        num_ranks,
        BLOCK_SIZE=block_size,
    )
    return permuted_indices


# reference
def fill_indices_cpu(
    tokens_per_expert_group: torch.Tensor,
    start_index_values: torch.Tensor,
    write_offsets: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
):
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
                permuted_indices[write_start:end_idx] = torch.arange(
                    start_index,
                    start_index + (end_idx - write_start),
                    dtype=torch.int32,
                    device=device,
                )
            write_start += length
    return permuted_indices


def generate_permute_indices(
    tokens_per_expert_group: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
    alignment: int,
    use_cpu: bool = False,
    block_size: int = 128,
):
    """
    Prepare permutation indices and the number of tokens for each expert.

    Args:
        tokens_per_expert_group: number of tokens for each expert from all ranks.
        experts_per_rank: number of experts per rank.
        num_ranks: number of ranks.
        max_len: maximum length of the output index vector.
        alignment: alignment for each returned element in `m_sizes`.
        use_cpu: whether to use CPU implementation.
        use_optimized: whether to use optimized Triton implementation.
        block_size: block size for optimized implementation.

    Returns:
        permuted_indices: permutation indices.
        m_sizes: number of tokens for each expert.
    """
    # prefix sum to get start index of each expert (parallel scan kernel in future?)
    start_index_values = (
        torch.cumsum(tokens_per_expert_group, 0) - tokens_per_expert_group
    )

    # chunk sizes for each expert
    chunk_size_per_expert = tokens_per_expert_group.view(num_ranks, -1).sum(0)

    # align the chunk sizes (cdiv)
    m_sizes = ((chunk_size_per_expert + alignment - 1) // alignment * alignment).to(
        torch.int32
    )

    # additional prefix sum to get write offset of each expert in permuted_indices
    write_offsets = torch.cumsum(m_sizes, 0) - m_sizes

    # Select the implementation to use
    if use_cpu:
        permuted_indices = fill_indices_cpu(
            tokens_per_expert_group,
            start_index_values,
            write_offsets,
            experts_per_rank,
            num_ranks,
            max_len,
        )
    else:
        permuted_indices = fill_indices_wrapper(
            tokens_per_expert_group,
            start_index_values,
            write_offsets,
            experts_per_rank,
            num_ranks,
            max_len,
            block_size=block_size,
        )

    return permuted_indices, m_sizes


def verify_correctness(
    experts_per_rank: int = 8,
    num_ranks: int = 8,
    token_range: Tuple[int, int] = (1, 32),
    max_len_factor: int = 4,
    alignment: int = 32,
    seed: int = 2020,
):
    torch.manual_seed(seed)

    # original sequential kernel
    from indices import generate_permute_indices as original_permute_indices

    # generate test data
    # Create test data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokens_per_expert_group = torch.randint(
        token_range[0],
        token_range[1] + 1,
        (num_ranks * experts_per_rank,),
        dtype=torch.int32,
        device=device,
    )

    # Calculate max_len based on token counts
    total_tokens = tokens_per_expert_group.sum().item()
    max_len = total_tokens * max_len_factor

    # Generate permutation indices using different implementations
    cpu_indices, cpu_sizes = generate_permute_indices(
        tokens_per_expert_group,
        experts_per_rank,
        num_ranks,
        max_len,
        alignment,
        use_cpu=True,
    )

    original_indices, original_sizes = original_permute_indices(
        tokens_per_expert_group,
        experts_per_rank,
        num_ranks,
        max_len,
        alignment,
        use_cpu=False,
    )

    optimized_indices, optimized_sizes = generate_permute_indices(
        tokens_per_expert_group,
        experts_per_rank,
        num_ranks,
        max_len,
        alignment,
        use_cpu=False,
    )

    # Check if results match
    cpu_original_match = torch.equal(cpu_indices, original_indices)
    cpu_optimized_match = torch.equal(cpu_indices, optimized_indices)
    orig_optimized_match = torch.equal(original_indices, optimized_indices)

    sizes_match = torch.equal(cpu_sizes, original_sizes) and torch.equal(
        cpu_sizes, optimized_sizes
    )

    all_match = (
        cpu_original_match
        and cpu_optimized_match
        and orig_optimized_match
        and sizes_match
    )

    if not all_match:
        print(
            f"Correctness test failed for experts_per_rank={experts_per_rank}, num_ranks={num_ranks}"
        )
        if not cpu_original_match:
            print("  CPU vs Original mismatch")
        if not cpu_optimized_match:
            print("  CPU vs Optimized mismatch")
        if not orig_optimized_match:
            print("  Original vs Optimized mismatch")
        if not sizes_match:
            print("  Sizes mismatch")

        # Find first mismatch (if results don't match)
        if not orig_optimized_match:
            mismatch_indices = (original_indices != optimized_indices).nonzero(
                as_tuple=True
            )[0]
            if len(mismatch_indices) > 0:
                first_mismatch = mismatch_indices[0].item()
                print(
                    f"  First mismatch at index {first_mismatch}: "
                    f"Original={original_indices[first_mismatch].item()}, "
                    f"Optimized={optimized_indices[first_mismatch].item()}"
                )
    print(f"{optimized_indices=}")
    return all_match


if __name__ == "__main__":
    res = verify_correctness()
    if res:
        print("Success - results match reference!")
    else:
        print("Warning:  see details above - results do not match reference!")
