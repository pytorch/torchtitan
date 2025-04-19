import torch
import triton
import triton.language as tl

# Create m_indices tensor using pre-computed m_offsets tensor.
# DSGemm requires m_indices to be contiguous mapping of all rows.
# this file is an exploration of different methods to create m_indices tensor.


def create_m_indices_from_offsets_torch(m_offsets: torch.Tensor) -> torch.Tensor:
    """
    Create m_indices tensor using pre-computed m_offsets tensor.
    Pure PyTorch implementation without CPU-GPU synchronization.

    Args:
        m_offsets: Tensor of shape [num_groups + 1] containing cumulative row counts
                  (m_offsets[i] is the starting index for group i)

    Returns:
        m_indices: Tensor mapping each row to its group index
    """
    num_groups = m_offsets.size(0) - 1
    total_rows = int(m_offsets[-1].item())  # Single .item() call

    # Skip if no rows
    if total_rows == 0:
        return torch.empty(0, device=m_offsets.device, dtype=torch.int32)

    # Create output tensor
    m_indices = torch.empty(total_rows, device=m_offsets.device, dtype=torch.int32)

    # For each group, fill the corresponding slice with the group index
    for group_idx in range(num_groups):
        start_idx = m_offsets[group_idx]
        end_idx = m_offsets[group_idx + 1]

        # If this group has rows, fill them with the group index
        if end_idx > start_idx:
            m_indices[start_idx:end_idx] = group_idx

    return m_indices


@triton.jit
def create_m_indices_from_offsets_kernel(
    m_indices_ptr,  # Pointer to output indices tensor
    m_offsets_ptr,  # Pointer to offsets tensor
    num_groups,  # Number of groups
    total_rows,  # Total number of rows
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelization
):
    # Get program ID for this thread
    pid = tl.program_id(axis=0)

    # Compute which row this thread is handling
    row_start = pid * BLOCK_SIZE
    row_idx = row_start + tl.arange(0, BLOCK_SIZE)

    # Create mask for valid rows
    valid_mask = row_idx < total_rows

    # Initialize group IDs
    group_id = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    # For each group, check if the rows belong to that group
    for i in range(num_groups):
        start_offset = tl.load(m_offsets_ptr + i)
        end_offset = tl.load(m_offsets_ptr + i + 1)

        # If row_idx is within this group's range, assign the group ID
        in_group_mask = (row_idx >= start_offset) & (row_idx < end_offset)
        group_id = tl.where(in_group_mask, i, group_id)

    # Store the result using the valid mask
    tl.store(m_indices_ptr + row_idx, group_id, mask=valid_mask)


def create_m_indices_from_offsets_triton(m_offsets: torch.Tensor) -> torch.Tensor:
    """
    Create m_indices tensor using pre-computed m_offsets tensor using Triton.

    Args:
        m_offsets: Tensor of shape [num_groups + 1] containing cumulative row counts
                  (m_offsets[i] is the starting index for group i)

    Returns:
        m_indices: Tensor mapping each row to its group index
    """
    num_groups = m_offsets.size(0) - 1
    total_rows = int(m_offsets[-1].item())  # Single .item() call

    # Skip if no rows
    if total_rows == 0:
        return torch.empty(0, device=m_offsets.device, dtype=torch.int32)

    # Create output tensor
    m_indices = torch.empty(total_rows, device=m_offsets.device, dtype=torch.int32)

    # Determine grid and block sizes
    BLOCK_SIZE = 128  # Can be tuned
    grid = (triton.cdiv(total_rows, BLOCK_SIZE),)

    # Launch kernel
    create_m_indices_from_offsets_kernel[grid](
        m_indices,
        m_offsets,
        num_groups=num_groups,
        total_rows=total_rows,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return m_indices


# More efficient kernel using binary search with offsets
@triton.jit
def create_m_indices_binary_search_kernel(
    m_indices_ptr,  # Pointer to output indices tensor
    m_offsets_ptr,  # Pointer to offsets tensor
    num_groups,  # Number of groups
    total_rows,  # Total number of rows
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelization
):
    # Get program ID for this thread
    pid = tl.program_id(axis=0)

    # Compute which row this thread is handling
    row_start = pid * BLOCK_SIZE
    row_idx = row_start + tl.arange(0, BLOCK_SIZE)

    # Create mask for valid rows
    valid_mask = row_idx < total_rows

    # Initialize group IDs
    group_id = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    # Process each row in the block using binary search
    for i in range(BLOCK_SIZE):
        # Skip invalid rows with a conditional
        if i + row_start < total_rows:
            # Get the current row
            row = row_start + i

            # Binary search variables
            lo = 0
            hi = num_groups

            # Binary search with fixed iterations
            for _ in range(16):  # 32 - log2 of max possible groups
                mid = (lo + hi) // 2
                offset = tl.load(m_offsets_ptr + mid)

                # Update search bounds
                lo_new = tl.where(offset <= row, mid + 1, lo)
                hi_new = tl.where(offset <= row, hi, mid)

                # Update lo and hi
                lo = lo_new
                hi = hi_new

                # Early termination (without break)
                converged = lo >= hi
                lo = tl.where(converged, lo, lo)
                hi = tl.where(converged, hi, hi)

            # Group index is (lo-1)
            result = tl.where(lo > 0, lo - 1, 0)

            # Update only for this position in the block
            mask = (tl.arange(0, BLOCK_SIZE) == i) & valid_mask
            group_id = tl.where(mask, result, group_id)

    # Store results
    tl.store(m_indices_ptr + row_idx, group_id, mask=valid_mask)


def create_m_indices_from_offsets_binary(m_offsets: torch.Tensor) -> torch.Tensor:
    """
    Create m_indices tensor using pre-computed m_offsets tensor using binary search.
    This is optimized for cases with many groups.

    Args:
        m_offsets: Tensor of shape [num_groups + 1] containing cumulative row counts
                  (m_offsets[i] is the starting index for group i)

    Returns:
        m_indices: Tensor mapping each row to its group index
    """
    num_groups = m_offsets.size(0) - 1
    total_rows = int(m_offsets[-1].item())  # Single .item() call

    # Skip if no rows
    if total_rows == 0:
        return torch.empty(0, device=m_offsets.device, dtype=torch.int32)

    # Create output tensor
    m_indices = torch.empty(total_rows, device=m_offsets.device, dtype=torch.int32)

    # Determine grid and block sizes
    BLOCK_SIZE = 256  # Can be tuned
    grid = (triton.cdiv(total_rows, BLOCK_SIZE),)

    # Launch kernel
    create_m_indices_binary_search_kernel[grid](
        m_indices,
        m_offsets,
        num_groups=num_groups,
        total_rows=total_rows,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return m_indices


# Fast implementation directly generating the indices from sizes
def direct_create_m_indices(m_sizes: torch.Tensor) -> torch.Tensor:
    """
    A direct and efficient implementation that creates m_indices in a single pass.

    Args:
        m_sizes: Tensor containing the number of rows for each group

    Returns:
        m_indices: Tensor mapping each row to its group index
    """
    # Calculate total rows and offsets
    m_offsets = torch.zeros(
        m_sizes.size(0) + 1, device=m_sizes.device, dtype=m_sizes.dtype
    )
    torch.cumsum(m_sizes, dim=0, out=m_offsets[1:])
    total_rows = int(m_offsets[-1].item())

    # Early exit if no rows
    if total_rows == 0:
        return torch.empty(0, device=m_sizes.device, dtype=torch.int32)

    # Create the indices directly using arange and repeat
    indices = torch.zeros(total_rows, device=m_sizes.device, dtype=torch.int32)

    # Only process non-zero sizes
    non_zero_mask = m_sizes > 0
    non_zero_indices = torch.nonzero(non_zero_mask, as_tuple=True)[0]

    for idx in non_zero_indices:
        size = m_sizes[idx]
        start = m_offsets[idx]
        indices[start : start + size] = idx

    return indices


# Benchmark function specifically for offset-based methods
def benchmark_offset_based_methods(m_sizes, num_repeats=2):
    """Compare performance of different offset-based m_indices creation methods"""
    import time

    # Calculate offsets once
    m_offsets = torch.zeros(
        m_sizes.size(0) + 1, device=m_sizes.device, dtype=m_sizes.dtype
    )
    torch.cumsum(m_sizes, dim=0, out=m_offsets[1:])

    # Ensure CUDA is synchronized before measuring
    torch.cuda.synchronize()

    # PyTorch with offsets
    start = time.time()
    for _ in range(num_repeats):
        indices_torch = create_m_indices_from_offsets_torch(m_offsets)
        torch.cuda.synchronize()
    torch_time = (time.time() - start) / num_repeats

    # Triton linear with offsets
    start = time.time()
    for _ in range(num_repeats):
        indices_triton = create_m_indices_from_offsets_triton(m_offsets)
        torch.cuda.synchronize()
    triton_time = (time.time() - start) / num_repeats

    # Triton binary with offsets
    start = time.time()
    for _ in range(num_repeats):
        indices_binary = create_m_indices_from_offsets_binary(m_offsets)
        torch.cuda.synchronize()
    binary_time = (time.time() - start) / num_repeats

    # Direct implementation
    start = time.time()
    for _ in range(num_repeats):
        indices_direct = direct_create_m_indices(m_sizes)
        torch.cuda.synchronize()
    direct_time = (time.time() - start) / num_repeats

    # Verify all methods produce the same result
    assert torch.all(
        indices_torch == indices_triton
    ), "Linear search result doesn't match PyTorch"
    assert torch.all(
        indices_torch == indices_binary
    ), "Binary search result doesn't match PyTorch"
    assert torch.all(
        indices_torch == indices_direct
    ), "Direct method result doesn't match PyTorch"

    print(f"PyTorch with offsets:          {torch_time*1000:.3f} ms")
    print(f"Triton linear with offsets:    {triton_time*1000:.3f} ms")
    print(f"Triton binary with offsets:    {binary_time*1000:.3f} ms")
    print(f"Direct method:                 {direct_time*1000:.3f} ms")

    return {
        "torch": indices_torch,
        "triton_linear": indices_triton,
        "triton_binary": indices_binary,
        "direct": indices_direct,
        "timings": {
            "torch_ms": torch_time * 1000,
            "triton_linear_ms": triton_time * 1000,
            "triton_binary_ms": binary_time * 1000,
            "direct_ms": direct_time * 1000,
        },
    }


# Example usage
def demo_with_example_data():
    # Example data
    m_sizes = torch.tensor(
        [128, 0, 128, 128, 256, 128, 128, 128, 128, 128, 256, 128, 128, 0, 128, 0],
        device="cuda",
        dtype=torch.int32,
    )
    m_offsets = torch.tensor(
        [
            0,
            128,
            128,
            256,
            384,
            640,
            768,
            896,
            1024,
            1152,
            1280,
            1536,
            1664,
            1792,
            1792,
            1920,
            1920,
        ],
        device="cuda",
        dtype=torch.int32,
    )

    print("Using example data:")
    print(f"m_sizes: {m_sizes}")
    print(f"m_offsets: {m_offsets}")

    # Create indices using different methods
    indices_from_sizes = direct_create_m_indices(m_sizes)
    indices_from_offsets = create_m_indices_from_offsets_torch(m_offsets)

    # Verify they match
    match = torch.all(indices_from_sizes == indices_from_offsets)
    print(f"\nResults match between size-based and offset-based methods: {match}")

    # Show part of the results
    print(f"Total indices: {indices_from_offsets.shape[0]}")
    print(f"First 10 indices: {indices_from_offsets[:10]}")
    print(f"Last 10 indices: {indices_from_offsets[-10:]}")

    # Run benchmark
    print("\nBenchmarking with example data:")
    benchmark_offset_based_methods(m_sizes, num_repeats=50)

    return indices_from_offsets


if __name__ == "__main__":
    # Example usage
    indices_from_offsets = demo_with_example_data()
    print(indices_from_offsets)
