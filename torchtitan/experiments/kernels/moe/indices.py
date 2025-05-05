import torch
import triton
import triton.language as tl


__all__ = ["generate_permute_indices"]


# Updated Triton kernel with min_slots_per_expert support
@triton.jit
def _fill_indices_kernel(
    tokens_per_expert_group_ptr,
    start_index_values_ptr,
    write_offsets_ptr,
    output_ptr,
    chunk_size_per_expert_ptr,  # Added to check for zero tokens
    experts_per_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    min_slots_per_expert: tl.constexpr,  # New parameter
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    # map programs (blocks) to the experts and loop (grid stride) if needed
    for expert_id in range(pid, experts_per_rank, num_programs):
        # read this expert's write offset
        write_offset = tl.load(write_offsets_ptr + expert_id)

        # Get total tokens for this expert across all ranks
        total_expert_tokens = tl.load(chunk_size_per_expert_ptr + expert_id)

        # If total tokens is zero, we don't need to do anything
        # The slots are already initialized to -1
        # We just ensure min_slots_per_expert slots are allocated, which happens in generate_permute_indices
        if total_expert_tokens > 0:
            # loop over all ranks
            for r in range(num_ranks):
                # index into tokens_per_expert_group array
                i = r * experts_per_rank + expert_id

                # load start index and number of tokens for this expert-rank pair
                start_index = tl.load(start_index_values_ptr + i)
                length = tl.load(tokens_per_expert_group_ptr + i)

                # Skip if no tokens for this rank-expert pair
                if length > 0:
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


# Updated wrapper function
def fill_indices_wrapper(
    tokens_per_expert_group: torch.Tensor,
    start_index_values: torch.Tensor,
    write_offsets: torch.Tensor,
    chunk_size_per_expert: torch.Tensor,  # Added parameter
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
    min_slots_per_expert: int = 8,  # Added parameter
    block_size: int = 128,
    max_blocks: int = 1024,
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
        chunk_size_per_expert,  # Pass the tensor to check for zero tokens
        experts_per_rank,
        num_ranks,
        min_slots_per_expert,  # Pass the minimum slots parameter
        BLOCK_SIZE=block_size,
    )
    return permuted_indices


# Updated CPU reference implementation
def fill_indices_cpu(
    tokens_per_expert_group: torch.Tensor,
    start_index_values: torch.Tensor,
    write_offsets: torch.Tensor,
    chunk_size_per_expert: torch.Tensor,  # Added parameter
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
    min_slots_per_expert: int = 8,  # Added parameter
):
    # We need to preallocate the output
    permuted_indices = torch.full(
        (max_len,),
        -1,
        dtype=torch.int32,
    )

    # Fill the permuted indices
    # For each local expert
    for e in range(experts_per_rank):
        write_start = write_offsets[e].item()
        total_tokens = chunk_size_per_expert[e].item()

        # Skip if this expert has no tokens (will have min_slots_per_expert filled with -1)
        if total_tokens > 0:
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
                    )
                    write_start += length

    return permuted_indices


# Updated main function
def generate_permute_indices(
    tokens_per_expert_group: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
    alignment: int,
    min_slots_per_expert: int = 8,  # Added parameter
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
        min_slots_per_expert: minimum number of slots to allocate for an expert, even if it has 0 tokens.
        use_cpu: whether to use CPU implementation.

    Returns:
        permuted_indices: Tensor of indices that map original token order to the expert-grouped order.
        m_sizes: aligned number of tokens for each expert (padded to alignment boundary).
        m_offsets: Cumulative sum of m_sizes. The exclusive ending position for each expert's tokens.
    """
    # prefix sum to get start index of each expert
    start_index_values = (
        torch.cumsum(tokens_per_expert_group, 0) - tokens_per_expert_group
    )

    # chunk sizes for each expert
    chunk_size_per_expert = tokens_per_expert_group.view(num_ranks, -1).sum(0)

    # Apply minimum slots per expert - ensure at least min_slots_per_expert
    # even for experts with zero tokens
    padded_chunk_size_per_expert = torch.maximum(
        chunk_size_per_expert,
        torch.full_like(chunk_size_per_expert, min_slots_per_expert),
    )

    # align the chunk sizes (cdiv)
    m_sizes = (
        (padded_chunk_size_per_expert + alignment - 1) // alignment * alignment
    ).to(torch.int32)

    # additional prefix sum to get write offset of each expert in permuted_indices
    m_offsets = torch.cumsum(m_sizes, 0)
    write_offsets = m_offsets - m_sizes

    # Select the implementation to use
    if use_cpu:
        permuted_indices = fill_indices_cpu(
            tokens_per_expert_group,
            start_index_values,
            write_offsets,
            chunk_size_per_expert,  # Pass to check for zero tokens
            experts_per_rank,
            num_ranks,
            max_len,
            min_slots_per_expert,
        )
    else:
        permuted_indices = fill_indices_wrapper(
            tokens_per_expert_group,
            start_index_values,
            write_offsets,
            chunk_size_per_expert,  # Pass to check for zero tokens
            experts_per_rank,
            num_ranks,
            max_len,
            min_slots_per_expert,
        )

    return permuted_indices, m_sizes, m_offsets.to(torch.int32)


# Updated test function
def test_with_zero_tokens():
    device = torch.device("cuda", 0)
    experts_per_rank = 4
    num_ranks = 2

    # Create a test case where some experts have zero tokens
    tokens_per_expert_group = torch.tensor(
        [4, 0, 2, 3, 1, 0, 0, 5],  # Some experts have zero tokens
        dtype=torch.int32,
        device=device,
    )

    max_len = 128
    alignment = 32
    min_slots_per_expert = 8  # Ensure at least 8 slots for experts with zero tokens

    # Use the GPU kernel
    permuted_indices_gpu, m_sizes, m_offsets = generate_permute_indices(
        tokens_per_expert_group,
        experts_per_rank,
        num_ranks,
        max_len,
        alignment,
        min_slots_per_expert,
    )

    # Use the CPU method
    permuted_indices_cpu, m_sizes_cpu, m_offsets_cpu = generate_permute_indices(
        tokens_per_expert_group,
        experts_per_rank,
        num_ranks,
        max_len,
        alignment,
        min_slots_per_expert,
        use_cpu=True,
    )

    # Check that the results are the same
    assert torch.equal(permuted_indices_gpu.cpu(), permuted_indices_cpu)
    assert torch.equal(m_sizes, m_sizes_cpu)

    # Verify that experts with zero tokens have at least min_slots_per_expert
    chunk_size_per_expert = tokens_per_expert_group.view(num_ranks, -1).sum(0)
    zero_token_experts = chunk_size_per_expert == 0
    if zero_token_experts.any():
        assert (m_sizes[zero_token_experts] >= min_slots_per_expert).all()

    # Check alignment
    assert torch.equal(
        torch.remainder(m_sizes, alignment),
        torch.zeros(experts_per_rank, device=device),
    )

    # Print the results
    print(f"tokens_per_expert_group = {tokens_per_expert_group}")
    print(f"chunk_size_per_expert = {chunk_size_per_expert}")
    print(f"m_sizes = {m_sizes}")
    print(f"m_offsets = {m_offsets}")
    print(f"permuted_indices = {permuted_indices_gpu[:sum(m_sizes).item()]}")

    # Check that experts with zero tokens have -1 in their slots
    for e in range(experts_per_rank):
        start = (m_offsets[e] - m_sizes[e]).item()
        end = m_offsets[e].item()
        expert_indices = permuted_indices_gpu[start:end]
        if chunk_size_per_expert[e] == 0:
            assert (
                expert_indices == -1
            ).all(), f"Expert {e} with zero tokens should have all -1 indices"
            assert (
                expert_indices.size(0) >= min_slots_per_expert
            ), f"Expert {e} with zero tokens should have at least {min_slots_per_expert} slots"
            print(
                f"Expert {e} has zero tokens and {expert_indices.size(0)} slots with all -1"
            )

    print("All tests passed successfully!")
    return True


if __name__ == "__main__":
    test_with_zero_tokens()
