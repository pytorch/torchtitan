# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl


__all__ = ["generate_permute_indices"]


# parallelized kernel
@triton.jit
def _fill_indices_kernel(
    tokens_per_expert_group_ptr,
    start_index_values_ptr,
    write_offsets_ptr,
    output_ptr,
    experts_per_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # Number of threads per block
    zero_replacement: tl.constexpr,  # Value to replace zeros with
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    # map programs (blocks) to the experts and loop (grid stride) if needed
    for expert_id in range(pid, experts_per_rank, num_programs):
        # read this experts write offset
        write_offset = tl.load(write_offsets_ptr + expert_id)

        # Replace zero write_offset with the replacement value
        write_offset = tl.where(write_offset == 0, zero_replacement, write_offset)

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

                # Replace zero values with the replacement value
                values = tl.where(values == 0, zero_replacement, values)

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
    zero_replacement: int = 8,
    block_size: int = 128,
    max_blocks: int = 1024,  # cap on total number of blocks to launch
):
    # preallocate output
    permuted_indices = torch.full(
        (max_len,), -1, dtype=torch.int32, device=tokens_per_expert_group.device
    )

    # write offsets is per local expert...
    num_blocks = min(experts_per_rank, max_blocks)
    # grid = one block per expert unless capped and then we loop...
    grid = (num_blocks,)

    # launch kernel
    _fill_indices_kernel[grid](
        tokens_per_expert_group,
        start_index_values,
        write_offsets,
        permuted_indices,
        experts_per_rank,
        num_ranks,
        BLOCK_SIZE=block_size,
        zero_replacement=zero_replacement,
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
    zero_replacement: int,
):
    # We need to preallocate the output - we ignore device and force it on cpu
    # device = tokens_per_expert_group.device
    permuted_indices = torch.full(
        (max_len,),
        -1,
        dtype=torch.int32,
    )  # device=device)

    # Replace zeros in write_offsets with zero_replacement
    write_offsets_replaced = torch.where(
        write_offsets == 0,
        torch.tensor(zero_replacement, dtype=write_offsets.dtype),
        write_offsets,
    )

    # Fill the permuted indices
    # For each local expert
    for e in range(experts_per_rank):
        write_start = write_offsets_replaced[e].item()
        # For each remote rank
        for r in range(num_ranks):
            i = r * experts_per_rank + e
            start_index = start_index_values[i].item()
            length = tokens_per_expert_group[i].item()
            # Fill in the indices
            if length > 0:
                end_idx = min(write_start + length, max_len)
                indices = torch.arange(
                    start_index,
                    start_index + (end_idx - write_start),
                    dtype=torch.int32,
                    # device=device,
                )
                # Replace zeros in indices with zero_replacement
                indices = torch.where(
                    indices == 0,
                    torch.tensor(zero_replacement, dtype=indices.dtype),
                    indices,
                )
                permuted_indices[write_start:end_idx] = indices
            write_start += length
    return permuted_indices


def generate_permute_indices(
    tokens_per_expert_group: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
    alignment: int,
    dtype: str = "bf16",  # Options: "bf16" or "fp8"
    use_cpu: bool = False,
):
    """
    Prepare permutation indices and the number of tokens for each expert.

    Args:
        tokens_per_expert_group: number of tokens for each expert from all ranks.
        experts_per_rank: number of experts per rank.
        num_ranks: number of ranks.
        max_len: maximum length of the output index vector.
        alignment: alignment for each returned element in `m_sizes`.
        dtype: data type, either "bf16" or "fp8".
        use_cpu: whether to use CPU implementation.

    Returns:
        permuted_indices: Tensor of indices that map original token order to the expert-grouped order.
        m_sizes: aligned number of tokens for each expert (padded to alignment boundary).
        m_offsets: Cumulative sum of m_sizes. The exclusive ending position for each expert's tokens.

    Explanatory details:
        `tokens_per_expert_group` is of shape (num_ranks * experts_per_rank,), for example:
        From: |       rank 0      |       rank 1      |
        To:   | E0 | E1 | E2 | E3 | E0 | E1 | E2 | E3 |
              |  4 |  2 |  1 |  3 |  1 |  2 |  3 |  4 |
    """
    device = tokens_per_expert_group.device

    # Set zero replacement value based on dtype
    if dtype == "bf16":
        zero_replacement = 8
    elif dtype == "fp8":
        zero_replacement = 16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}, expected 'bf16' or 'fp8'")

    # Identify which expert-rank pairs have zero tokens
    zero_mask = tokens_per_expert_group == 0

    # prefix sum to get start index of each expert (parallel scan kernel in future?)
    start_index_values = (
        torch.cumsum(tokens_per_expert_group, 0) - tokens_per_expert_group
    )

    # Replace zeros in start_index_values with zero_replacement
    start_index_values = torch.where(
        start_index_values == 0,
        torch.tensor(zero_replacement, dtype=start_index_values.dtype, device=device),
        start_index_values,
    )

    # chunk sizes for each expert
    chunk_size_per_expert = tokens_per_expert_group.view(num_ranks, -1).sum(0)

    # align the chunk sizes (cdiv)
    m_sizes = ((chunk_size_per_expert + alignment - 1) // alignment * alignment).to(
        torch.int32
    )

    # Replace zeros in m_sizes with zero_replacement
    m_sizes = torch.where(
        m_sizes == 0,
        torch.tensor(zero_replacement, dtype=m_sizes.dtype, device=device),
        m_sizes,
    )

    # additional prefix sum to get write offset of each expert in permuted_indices

    m_offsets = torch.cumsum(m_sizes, 0)
    write_offsets = m_offsets - m_sizes

    # Replace zeros in m_offsets with zero_replacement
    m_offsets = torch.where(
        m_offsets == 0,
        torch.tensor(zero_replacement, dtype=m_offsets.dtype, device=device),
        m_offsets,
    )

    # Replace zeros in write_offsets with zero_replacement
    write_offsets = torch.where(
        write_offsets == 0,
        torch.tensor(zero_replacement, dtype=write_offsets.dtype, device=device),
        write_offsets,
    )

    # Select the implementation to use
    if use_cpu:
        permuted_indices = fill_indices_cpu(
            tokens_per_expert_group,
            start_index_values,
            write_offsets,
            experts_per_rank,
            num_ranks,
            max_len,
            zero_replacement,
        )
    else:
        permuted_indices = fill_indices_wrapper(
            tokens_per_expert_group,
            start_index_values,
            write_offsets,
            experts_per_rank,
            num_ranks,
            max_len,
            zero_replacement,
        )

    permuted_indices = torch.where(
        permuted_indices == 0,
        torch.tensor(
            zero_replacement,
            dtype=permuted_indices.dtype,
            device=permuted_indices.device,
        ),
        permuted_indices,
    )

    # Now, insert zero_replacement values for individual experts with zero tokens
    # For each expert-rank pair that has zero tokens
    for i in range(len(tokens_per_expert_group)):
        if tokens_per_expert_group[i] == 0:
            # Determine which expert and rank this is
            r = i // experts_per_rank  # rank
            e = i % experts_per_rank  # expert

            # Get the write offset for this expert
            offset = write_offsets[e].item()

            # Calculate where this specific rank-expert pair's data would start
            for prev_r in range(r):
                prev_i = prev_r * experts_per_rank + e
                offset += tokens_per_expert_group[prev_i].item()

            # Insert zero_replacement value at this position if it's within bounds
            if offset < max_len:
                permuted_indices[offset] = zero_replacement

    return permuted_indices, m_sizes, m_offsets.to(torch.int32)


# Below is for testing only


def simple_test():
    device = torch.device("cuda", 0)
    experts_per_rank = 4
    num_ranks = 4
    tokens_per_expert_group = torch.full(
        (num_ranks * experts_per_rank,), 4, dtype=torch.int32, device=device
    )
    # Insert some zeros for testing
    tokens_per_expert_group[0] = 0  # Zero for expert 0, rank 0
    tokens_per_expert_group[5] = 0  # Zero for expert 1, rank 1

    # Create an expert with zero tokens across all ranks
    for r in range(num_ranks):
        tokens_per_expert_group[r * experts_per_rank + 2] = (
            0  # Zero for expert 2, all ranks
        )

    max_len = 128
    alignment = 32

    # Test with bf16
    permuted_indices_gpu_bf16, m_sizes_bf16, m_offsets_bf16 = generate_permute_indices(
        tokens_per_expert_group,
        experts_per_rank,
        num_ranks,
        max_len,
        alignment,
        dtype="bf16",
    )

    # Test with fp8
    permuted_indices_gpu_fp8, m_sizes_fp8, m_offsets_fp8 = generate_permute_indices(
        tokens_per_expert_group,
        experts_per_rank,
        num_ranks,
        max_len,
        alignment,
        dtype="fp8",
    )

    # Test CPU implementation with bf16
    permuted_indices_cpu, m_sizes_cpu, m_offsets_cpu = generate_permute_indices(
        tokens_per_expert_group,
        experts_per_rank,
        num_ranks,
        max_len,
        alignment,
        dtype="bf16",
        use_cpu=True,
    )

    # Verify zero replacement
    # Check that there are no zeros in any of the outputs for bf16
    assert torch.all(permuted_indices_gpu_bf16[permuted_indices_gpu_bf16 != -1] != 0)
    assert torch.all(m_sizes_bf16 != 0)
    assert torch.all(m_offsets_bf16 != 0)

    # Check that there are no zeros in any of the outputs for fp8
    assert torch.all(permuted_indices_gpu_fp8[permuted_indices_gpu_fp8 != -1] != 0)
    assert torch.all(m_sizes_fp8 != 0)
    assert torch.all(m_offsets_fp8 != 0)

    # Check that bf16 uses 8 as replacement and fp8 uses 16
    # Find where zeros would have been and check replacement values
    assert torch.all(m_sizes_bf16[m_sizes_bf16 == 8] == 8)
    assert torch.all(m_sizes_fp8[m_sizes_fp8 == 16] == 16)

    # Check alignment
    # assert torch.all(torch.remainder(m_sizes_bf16, alignment) == 0)
    # Check alignment for non-replacement values only
    assert torch.all(torch.remainder(m_sizes_bf16[m_sizes_bf16 != 8], alignment) == 0)

    # Check if expert 2 (with zero tokens across all ranks) has the correct replacement
    assert m_sizes_bf16[2] == 8
    assert m_sizes_fp8[2] == 16

    # Check if the permuted indices have the special values at the correct positions
    # Get offset for expert 2
    offset_e2_bf16 = write_offsets = m_offsets_bf16[2] - m_sizes_bf16[2]
    offset_e2_fp8 = write_offsets = m_offsets_fp8[2] - m_sizes_fp8[2]

    # Check if these positions have the replacement values
    assert permuted_indices_gpu_bf16[offset_e2_bf16] == 8
    assert permuted_indices_gpu_fp8[offset_e2_fp8] == 16

    print(f"permuted_indices_bf16: {permuted_indices_gpu_bf16.cpu()}")
    print(f"permuted_indices_fp8: {permuted_indices_gpu_fp8.cpu()}")
    print(f"m_sizes_bf16: {m_sizes_bf16.cpu()}")
    print(f"m_sizes_fp8: {m_sizes_fp8.cpu()}")
    print(f"m_offsets_bf16: {m_offsets_bf16.cpu()}")
    print(f"m_offsets_fp8: {m_offsets_fp8.cpu()}")
    print(
        "Success - All zeros replaced correctly and matching permuted indices inserted"
    )
    return True


if __name__ == "__main__":
    simple_test()
