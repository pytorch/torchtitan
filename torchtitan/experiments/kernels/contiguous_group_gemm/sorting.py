import torch
import triton
import triton.language as tl


@triton.heuristics(
    values={"INPUT_ALIGNED": lambda args: args["M_total"] % args["BLOCK_SIZE"] == 0}
)
@triton.jit
def _improved_counting_sort_kernel(
    # Input pointer
    expert_indices_ptr,  # [M_total]
    # Output pointers
    sort_indices_ptr,  # [M_total]
    expert_offsets_ptr,  # [num_experts + 1]
    # Dimensions
    M_total,
    num_experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INPUT_ALIGNED: tl.constexpr,
):
    """
    H100-optimized kernel for sorting token indices by expert using counting sort.

    Key optimizations:
    1. Uses shared memory histogram with power-of-2 sized buffers
    2. Efficient prefix sum computation with proper synchronization
    3. Careful bounds checking throughout
    4. Proper use of atomics for global memory updates only
    """
    # Get program ID and number of blocks
    pid = tl.program_id(0)
    n_blocks = tl.num_programs(0)

    # Calculate start and end indices for this block
    items_per_block = tl.cdiv(M_total, n_blocks)
    start_idx = pid * items_per_block

    # Calculate offsets for this block
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M_total

    # Ensure histogram buffer is power-of-2 sized
    COUNT_SIZE: tl.constexpr = triton.next_power_of_2(num_experts + 1)

    # Allocate shared memory for local counts
    local_counts = tl.zeros([COUNT_SIZE], dtype=tl.int32)

    # Load expert IDs for this block with masking
    expert_ids = tl.load(
        expert_indices_ptr + offsets,
        mask=mask,
        other=num_experts,  # Use num_experts as padding value that won't be counted
    )

    # Count occurrences of each expert ID in this block
    # Create a mask for each expert in parallel
    expert_masks = tl.arange(0, COUNT_SIZE)[:, None] == expert_ids[None, :]
    # Apply bounds mask to each expert
    valid_expert_masks = expert_masks & mask[None, :]
    # Count tokens per expert
    expert_counts = tl.sum(valid_expert_masks.to(tl.int32), axis=1)
    # Update local counts
    local_counts += expert_counts

    # First block initializes global counts to 0
    if pid == 0:
        for e in range(num_experts + 1):
            tl.store(expert_offsets_ptr + e, 0)

    # Ensure initialization is complete
    tl.debug_barrier()

    # Atomically add local counts to global counts (offset+1 for exclusive prefix sum)
    for e in range(num_experts):
        # Create a mask for the current expert
        is_expert_e = tl.arange(0, COUNT_SIZE) == e
        # Get the count for this expert (will be 0 for all other positions)
        count = tl.sum(local_counts * is_expert_e)
        if count > 0:  # Only process if there are tokens for this expert
            tl.atomic_add(expert_offsets_ptr + e + 1, count)

    # Wait for all blocks to finish counting
    tl.debug_barrier()

    # One block computes prefix sum
    if pid == 0:
        running_sum = 0
        for e in range(num_experts + 1):
            current = tl.load(expert_offsets_ptr + e)
            tl.store(expert_offsets_ptr + e, running_sum)
            running_sum += current

    # Ensure prefix sum is complete
    tl.debug_barrier()

    # Each block handles its own portion of input data
    # Each block handles its own portion of input data
    for i in range(BLOCK_SIZE):
        # Create a mask for the current position
        is_pos_i = tl.arange(0, BLOCK_SIZE) == i
        # Check if this position is within bounds
        valid_pos = is_pos_i & mask

        if tl.sum(valid_pos) > 0:
            # Get the expert ID for this position using the mask
            expert_id = tl.sum(expert_ids * is_pos_i)
            # Check if expert ID is valid
            if expert_id < num_experts:
                # Get current position for this expert and increment atomically
                position = tl.atomic_add(expert_offsets_ptr + expert_id, 1)
                # Store the original index
                idx = start_idx + i
                tl.store(sort_indices_ptr + position, idx)


def _counting_sort(expert_indices: torch.Tensor) -> torch.Tensor:
    """
    Optimized implementation of counting sort for H100 GPUs.

    Args:
        expert_indices: Expert indices tensor [M_total]

    Returns:
        Sorted token indices tensor [M_total]
    """
    device = expert_indices.device
    M_total = expert_indices.shape[0]

    # Handle empty tensor case
    if M_total == 0:
        return torch.zeros(0, dtype=torch.int64, device=device)

    # Find the maximum expert index to determine num_experts
    num_experts = int(expert_indices.max().item()) + 1

    # Create output tensors
    sort_indices = torch.empty(M_total, dtype=torch.int64, device=device)
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)

    # Determine optimal block size and grid
    BLOCK_SIZE = 256  # Smaller block size for better occupancy
    grid = (triton.cdiv(M_total, BLOCK_SIZE),)

    # Launch kernel
    with torch.cuda.device(device):
        _improved_counting_sort_kernel[grid](
            expert_indices,
            sort_indices,
            expert_offsets,
            M_total,
            num_experts,
            BLOCK_SIZE,
        )

    return sort_indices


def test_counting_sort(
    M_total=24,
    num_experts=8,
    seed=2020,
    device="cuda",
    verbose=True,
):
    """
    Test the correctness of improved counting sort against PyTorch argsort.
    """
    torch.manual_seed(seed)

    # Create random expert indices
    expert_indices = torch.randint(0, num_experts, (M_total,), device=device)

    # Get reference result using PyTorch
    pytorch_result = torch.argsort(expert_indices)

    # Get result using our implementation
    triton_result = _counting_sort(expert_indices)

    # Check if sorting is correct
    pytorch_sorted = expert_indices[pytorch_result]
    triton_sorted = expert_indices[triton_result]

    # Check if experts are correctly sorted
    experts_match = torch.all(pytorch_sorted == triton_sorted).item()

    # Check stability by direct comparison with a test case
    # Explicitly set up a test case with repeated expert IDs
    if M_total >= 10:
        test_experts = torch.tensor([3, 1, 3, 2, 1, 0, 2, 3, 1, 0], device=device)
        test_indices = torch.arange(len(test_experts), device=device)

        # Get reference stable sort
        pytorch_stable = torch.argsort(test_experts, stable=True)

        # Get our result
        triton_stable = _counting_sort(test_experts)

        # Check stability by comparing expert order and index order
        pytorch_sorted_experts = test_experts[pytorch_stable]
        triton_sorted_experts = test_experts[triton_stable]

        # For each expert ID, check if the original indices appear in the same order
        stable_sort = True
        for expert_id in range(num_experts):
            # Find positions where this expert appears in both sorted results
            pytorch_positions = (
                (pytorch_sorted_experts == expert_id).nonzero().flatten()
            )
            triton_positions = (triton_sorted_experts == expert_id).nonzero().flatten()

            # Skip if this expert doesn't appear in the test case
            if len(pytorch_positions) == 0 or len(triton_positions) == 0:
                continue

            # Check if we have the same number of occurrences
            if len(pytorch_positions) != len(triton_positions):
                stable_sort = False
                if verbose:
                    print(
                        f"Expert {expert_id}: Different number of occurrences - PyTorch: {len(pytorch_positions)}, Triton: {len(triton_positions)}"
                    )
                break

            # Get original indices in these positions
            pytorch_orig_indices = pytorch_stable[pytorch_positions]
            triton_orig_indices = triton_stable[triton_positions]

            # Check if original indices appear in the same order
            if not torch.all(pytorch_orig_indices == triton_orig_indices):
                stable_sort = False
                if verbose:
                    print(f"Expert {expert_id}: Different order of indices")
                    print(f"  PyTorch: {pytorch_orig_indices}")
                    print(f"  Triton: {triton_orig_indices}")
                break
    else:
        stable_sort = True  # Skip stability check for small test cases

    if verbose:
        print(f"Testing with {M_total} tokens and {num_experts} experts:")
        print(f"  - Experts correctly sorted: {experts_match}")
        print(f"  - Stability maintained: {stable_sort}")

    return {
        "experts_match": experts_match,
        "stability_match": stable_sort,
        "success": experts_match and stable_sort,
    }


def test_improved_counting_sort_2(
    M_total=10000,
    num_experts=128,
    seed=42,
    device="cuda",
    verbose=True,
):
    """
    Test the correctness of improved counting sort against PyTorch argsort.
    """
    torch.manual_seed(seed)

    # Create random expert indices
    expert_indices = torch.randint(0, num_experts, (M_total,), device=device)

    # Get reference result using PyTorch
    pytorch_result = torch.argsort(expert_indices)

    # Get result using our implementation
    triton_result = improved_counting_sort(expert_indices)

    # Check if sorting is correct
    pytorch_sorted = expert_indices[pytorch_result]
    triton_sorted = expert_indices[triton_result]

    # Check if experts are correctly sorted
    experts_match = torch.all(pytorch_sorted == triton_sorted).item()

    # Check stability by direct comparison with a test case
    # Explicitly set up a test case with repeated expert IDs
    if M_total >= 10:
        test_experts = torch.tensor([3, 1, 3, 2, 1, 0, 2, 3, 1, 0], device=device)
        test_indices = torch.arange(len(test_experts), device=device)

        # Get reference stable sort
        pytorch_stable = torch.argsort(test_experts, stable=True)

        # Get our result
        triton_stable = improved_counting_sort(test_experts)

        # Check stability by comparing expert order and index order
        pytorch_sorted_experts = test_experts[pytorch_stable]
        triton_sorted_experts = test_experts[triton_stable]

        # For each expert ID, check if the original indices appear in the same order
        stable_sort = True
        for expert_id in range(num_experts):
            # Find positions where this expert appears in both sorted results
            pytorch_positions = (
                (pytorch_sorted_experts == expert_id).nonzero().flatten()
            )
            triton_positions = (triton_sorted_experts == expert_id).nonzero().flatten()

            # Get original indices in these positions
            pytorch_orig_indices = pytorch_stable[pytorch_positions]
            triton_orig_indices = triton_stable[triton_positions]

            # Check if original indices appear in the same order
            if not torch.all(pytorch_orig_indices == triton_orig_indices):
                stable_sort = False
                break
    else:
        stable_sort = True  # Skip stability check for small test cases

    if verbose:
        print(f"Testing with {M_total} tokens and {num_experts} experts:")
        print(f"  - Experts correctly sorted: {experts_match}")
        print(f"  - Stability maintained: {stable_sort}")

    return {
        "experts_match": experts_match,
        "stability_match": stable_sort,
        "success": experts_match and stable_sort,
    }


if __name__ == "__main__":
    test_counting_sort()
