# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl


__all__ = ["generate_permute_indices"]


@triton.jit
def fill_indices_kernel(
    tokens_per_expert_group_ptr,  # *Pointer* to first input vector.
    start_index_values_ptr,  # *Pointer* to second input vector.
    write_offsets_ptr,  # *Pointer* to third input vector.
    output_ptr,  # *Pointer* to output vector.
    experts_per_rank,  # Number of experts per rank.
    num_ranks,  # Number of expert ranks.
):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # The total number of programs in the launch grid.
    num_programs = tl.num_programs(axis=0)
    # We map the programs (blocks) to the experts.
    for expert_id in tl.range(pid, experts_per_rank, step=num_programs):
        # Read this expert's write offset.
        write_offset = tl.load(write_offsets_ptr + expert_id)
        # Loop over the ranks.
        for r in tl.range(num_ranks):
            # Slot in the tokens_per_expert_group array.
            i = r * experts_per_rank + expert_id
            start_index = tl.load(start_index_values_ptr + i)
            length = tl.load(tokens_per_expert_group_ptr + i)
            # Write the indices.
            for l in tl.range(length):
                val = start_index + l
                tl.store(output_ptr + write_offset + l, val)
            write_offset += length


def fill_indices(
    tokens_per_expert_group: torch.Tensor,
    start_index_values: torch.Tensor,
    write_offsets: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
):
    # We need to preallocate the output.
    permuted_indices = torch.full(
        (max_len,), -1, dtype=torch.int32, device=tokens_per_expert_group.device
    )
    # Analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks (TODO: bump this value).
    grid = lambda meta: (1,)
    #  Each torch.tensor object is implicitly converted into a pointer to its first element.
    fill_indices_kernel[grid](
        tokens_per_expert_group,
        start_index_values,
        write_offsets,
        permuted_indices,
        experts_per_rank,
        num_ranks,
    )
    return permuted_indices


def fill_indices_cpu(
    tokens_per_expert_group: torch.Tensor,
    start_index_values: torch.Tensor,
    write_offsets: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
):
    # We need to preallocate the output.
    permuted_indices = torch.full((max_len,), -1, dtype=torch.int32)
    # Fill the permuted indices
    # For each local expert
    for e in range(experts_per_rank):
        write_start = write_offsets[e]
        # For each remote rank
        for r in range(num_ranks):
            i = r * experts_per_rank + e
            start_index = start_index_values[i]
            length = tokens_per_expert_group[i]
            # Fill in the indices
            permuted_indices[write_start : write_start + length] = torch.arange(
                start_index, start_index + length
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
):
    # Prepare permutation indices and the number of tokens for each expert.  The
    # permutation indices are the indices of the tokens for each expert.  The
    # number of tokens for each expert is the sum of the number of tokens for
    # such experts from all ranks. This number is aligned to the provided
    # alignment requirement (usually comes from group gemm).

    # Args:
    #     tokens_per_expert_group: number of tokens for each expert from all ranks.
    #     experts_per_rank: number of experts per rank.
    #     num_ranks: number of ranks.
    #     max_len: maximum length of the output index vector. If greater than
    #     total number of tokens, the remaining indices are set to -1.
    #     alignment: alignment for each returned element in `m_sizes`.
    #     use_cpu: whether to use cpu or gpu.
    # Returns:
    #     permuted_indices: permutation indices.
    #     m_sizes: number of tokens for each expert.

    # `tokens_per_expert_group` is of shape (num_ranks * experts_per_rank,), for example:
    # From: |       rank 0      |       rank 1      |
    # To:   | E0 | E1 | E2 | E3 | E0 | E1 | E2 | E3 |
    #       |  4 |  2 |  1 |  3 |  1 |  2 |  3 |  4 |

    # Prefix sum to get the start index value of each expert
    start_index_values = (
        torch.cumsum(tokens_per_expert_group, 0) - tokens_per_expert_group
    )
    # Chunk sizes for each expert
    chunk_size_per_expert = tokens_per_expert_group.view(num_ranks, -1).sum(0)
    # Align the chunk sizes to the given alignment
    m_sizes = ((chunk_size_per_expert + alignment - 1) // alignment * alignment).to(
        torch.int32
    )
    # Perform another prefix sum to get the write offset of each expert in `permuted_indices`
    m_offsets = torch.cumsum(m_sizes, 0)
    write_offsets = m_offsets - m_sizes
    # Select the method to fill the permuted indices
    fill_fn = fill_indices_cpu if use_cpu else fill_indices
    # Fill the permuted indices
    permuted_indices = fill_fn(
        tokens_per_expert_group,
        start_index_values,
        write_offsets,
        experts_per_rank,
        num_ranks,
        max_len,
    )
    return permuted_indices, m_sizes, m_offsets.to(torch.int32)


# Below is for testing only


def test():
    device = torch.device("cuda", 0)
    experts_per_rank = 4
    num_ranks = 4
    tokens_per_expert_group = torch.full(
        (num_ranks * experts_per_rank,), 4, dtype=torch.int32, device=device
    )
    max_len = 128
    alignment = 32
    # Use the GPU kernel
    permuted_indices_gpu, m_sizes, _ = generate_permute_indices(
        tokens_per_expert_group, experts_per_rank, num_ranks, max_len, alignment
    )
    # Use the CPU method
    permuted_indices_cpu, _, _ = generate_permute_indices(
        tokens_per_expert_group,
        experts_per_rank,
        num_ranks,
        max_len,
        alignment,
        use_cpu=True,
    )
    # Check that the results are the same
    assert torch.equal(permuted_indices_gpu.cpu(), permuted_indices_cpu)
    assert torch.equal(
        torch.remainder(m_sizes, alignment),
        torch.zeros(experts_per_rank, device=device),
    )
    # Print the results
    print(permuted_indices_gpu)
    print(m_sizes)
    print("Success")


if __name__ == "__main__":
    test()
