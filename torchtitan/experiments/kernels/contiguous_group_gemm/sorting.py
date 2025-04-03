import time

import matplotlib
import torch
import triton
import triton.language as tl

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# ===================================================================
# Counting Sort Kernel
# ===================================================================


@triton.jit
def _triton_counting_sort_kernel(
    # Input pointer
    expert_indices_ptr,  # [M_total]
    # Output pointers
    sort_indices_ptr,  # [M_total]
    expert_offsets_ptr,  # [num_experts + 1]
    # Dimensions
    M_total,
    num_experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    kernel for sorting token indices by expert
    """
    # Get program ID and number of blocks
    pid = tl.program_id(0)
    n_blocks = tl.num_programs(0)

    # start and end indices for this block
    items_per_block = tl.cdiv(M_total, n_blocks)
    start_idx = pid * items_per_block
    end_idx = min(start_idx + items_per_block, M_total)

    # power of 2 size for local counts (requires constexpr)
    COUNT_SIZE: tl.constexpr = triton.next_power_of_2(num_experts + 1)

    # shared memory
    local_counts = tl.zeros([COUNT_SIZE], dtype=tl.int32)

    # local histogram in shared memory
    for idx in range(start_idx, end_idx):
        if idx < M_total:  # Bounds check
            expert_idx = tl.load(expert_indices_ptr + idx)

            is_expert = tl.arange(0, COUNT_SIZE) == expert_idx

            valid_expert = is_expert & (tl.arange(0, COUNT_SIZE) < num_experts)
            # Add to counts / add 1 to the matching expert index
            local_counts = local_counts + valid_expert.to(tl.int32)

    # First block initializes global offsets
    if pid == 0:
        for e in range(num_experts + 1):
            tl.store(expert_offsets_ptr + e, 0)

    # Wait for initialization
    tl.debug_barrier()

    # Atomically add local counts to global counts

    for e in range(num_experts):
        # mask for the current expert
        is_expert_e = tl.arange(0, COUNT_SIZE) == e
        # count for this expert (will be 0 for all other positions)
        count = tl.sum(local_counts * is_expert_e)
        if count > 0:
            tl.atomic_add(expert_offsets_ptr + e + 1, count)

    # Wait for counting phase
    tl.debug_barrier()

    # Single block computes prefix sum
    if pid == 0:
        running_sum = 0
        for e in range(num_experts + 1):
            current = tl.load(expert_offsets_ptr + e)
            tl.store(expert_offsets_ptr + e, running_sum)
            running_sum += current

    # Barrier for prefix sum
    tl.debug_barrier()

    # Re-use local_counts for storing the current offset for each expert
    # Load offsets from global memory to local array (only up to num_experts)
    offsets = tl.zeros([COUNT_SIZE], dtype=tl.int32)
    for e in range(num_experts):
        # mask for the current expert
        is_expert_e = tl.arange(0, COUNT_SIZE) == e
        # Load the offset for this expert
        offset = tl.load(expert_offsets_ptr + e)
        # Store it in the local array
        offsets = offsets + is_expert_e.to(tl.int32) * offset

    # Place elements in their correct positions
    for idx in range(start_idx, end_idx):
        if idx < M_total:  # Add bounds check
            expert_idx = tl.load(expert_indices_ptr + idx)

            is_this_expert = tl.arange(0, COUNT_SIZE) == expert_idx
            position = tl.sum(offsets * is_this_expert)

            # Update the counter for this expert
            offsets = offsets + is_this_expert.to(tl.int32)

            # Store the sorted index
            tl.store(sort_indices_ptr + position, idx)


def triton_counting_sort(expert_indices: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of counting sort.

    Args:
        expert_indices: Expert indices for each token [M_total]

    Returns:
        Sorted indices [M_total] with stable ordering
    """
    device = expert_indices.device
    M_total = expert_indices.shape[0]

    # Handle empty tensor case
    if M_total == 0:
        return torch.zeros(0, dtype=torch.int64, device=device)

    # Find the maximum expert index
    num_experts = int(expert_indices.max().item()) + 1

    # Create output tensor for sorted indices
    sort_indices = torch.empty(M_total, dtype=torch.int64, device=device)

    # Create tensor for expert offsets
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)

    # Calculate optimal block size - power of 2 is important for compatibility
    block_size = 1024
    grid = (triton.cdiv(M_total, block_size),)

    # Launch kernel
    _triton_counting_sort_kernel[grid](
        expert_indices,
        sort_indices,
        expert_offsets,
        M_total,
        num_experts,
        block_size,
    )

    return sort_indices


# ===== Testing and Benchmarking Functions ===========


def test_counting_sort_correctness(
    M_total=10000,
    num_experts=128,
    seed=42,
    device="cuda",
    verbose=True,
):
    """
    Test correctness of counting sort implementation compared to PyTorch argsort.

    Args:
        M_total: Number of tokens
        num_experts: Number of experts
        seed: Random seed
        device: Device to run on
        verbose: Whether to print results

    Returns:
        Dictionary with test results
    """
    torch.manual_seed(seed)

    # Create random expert indices
    expert_indices = torch.randint(0, num_experts, (M_total,), device=device)

    # Handle empty tensor case
    if M_total == 0:
        if verbose:
            print("Testing with empty tensor")
        return {"experts_match": True, "stability_match": True, "success": True}

    # Pytorch reference result
    pytorch_result = torch.argsort(expert_indices)

    # kernel result
    triton_result = triton_counting_sort(expert_indices)

    # Check if sorting is correct
    pytorch_sorted = expert_indices[pytorch_result]
    triton_sorted = expert_indices[triton_result]

    # experts are correctly sorted?
    experts_match = torch.all(pytorch_sorted == triton_sorted).item()

    # Check stability (equal experts maintain original order)
    stability_match = True

    # Group the sorted indices by expert
    expert_groups = {}
    for i, expert in enumerate(pytorch_sorted.cpu().numpy()):
        if expert not in expert_groups:
            expert_groups[expert] = []
        expert_groups[expert].append((i, pytorch_result[i].item()))

    # Check each expert group
    for expert, group in expert_groups.items():
        if len(group) <= 1:
            continue

        # Get the Triton indices for this expert
        triton_indices = []
        expert_mask = triton_sorted == expert
        expert_positions = torch.nonzero(expert_mask).squeeze(1)
        for pos in expert_positions:
            triton_indices.append(triton_result[pos].item())

        # Check that the order of original indices matches
        pytorch_original_indices = [g[1] for g in group]

        # verify lengths
        if len(pytorch_original_indices) != len(triton_indices):
            stability_match = False
            break

        # Check if the relative order is preserved
        # need to compare the actual order, not the sorted order
        if not all(a == b for a, b in zip(pytorch_original_indices, triton_indices)):
            stability_match = False
            break

    if verbose:
        print(
            f"Testing counting sort correctness with {M_total} tokens and {num_experts} experts:"
        )
        print(f"  - Experts correctly sorted: {experts_match}")
        print(f"  - Stability maintained: {stability_match}")

    return {
        "experts_match": experts_match,
        "stability_match": stability_match,
        "success": experts_match and stability_match,
    }


def benchmark_sorting_methods(
    M_totals=[10_000, 50_000, 100_000, 500_000, 1_000_000],
    num_experts_list=[16, 64, 128, 256],  # Test various expert counts
    device="cuda",
    runs=20,  #  for better statistics
    warmup=5,
):
    """
    Benchmark sorting methods.

    Args:
        M_totals: List of token counts to benchmark
        num_experts: Number of experts
        device: Device to run on
        runs: Number of benchmark runs
        warmup: Number of warmup runs

    Returns:
        Dictionary with benchmark results
    """
    # Results structure for multiple expert counts
    results = {
        "M_totals": M_totals,
        "num_experts_list": num_experts_list,
        "pytorch_times": {},
        "triton_times": {},
        "speedups": {},
    }

    for num_experts in num_experts_list:
        results["pytorch_times"][num_experts] = []
        results["triton_times"][num_experts] = []
        results["speedups"][num_experts] = []

        for M_total in M_totals:
            print(f"\nBenchmarking with {M_total} tokens and {num_experts} experts:")

            # Create random expert indices
            torch.manual_seed(0)
            expert_indices = torch.randint(0, num_experts, (M_total,), device=device)

            # Warmup
            for _ in range(warmup):
                # Force complete CUDA execution
                torch.cuda.synchronize()
                _ = torch.argsort(expert_indices)
                torch.cuda.synchronize()
                _ = triton_counting_sort(expert_indices)
                torch.cuda.synchronize()

            # Benchmark PyTorch argsort
            torch.cuda.synchronize()
            pytorch_times = []
            for _ in range(runs):
                start_time = time.time()
                _ = torch.argsort(expert_indices)
                torch.cuda.synchronize()
                pytorch_times.append(time.time() - start_time)
            pytorch_time = sum(pytorch_times) / len(pytorch_times)

            # Benchmark our implementation
            torch.cuda.synchronize()
            triton_times = []
            for _ in range(runs):
                start_time = time.time()
                _ = triton_counting_sort(expert_indices)
                torch.cuda.synchronize()
                triton_times.append(time.time() - start_time)
            triton_time = sum(triton_times) / len(triton_times)

            # Calculate speedup
            speedup = pytorch_time / triton_time

            # Store results
            results["pytorch_times"][num_experts].append(
                pytorch_time * 1000
            )  # Convert to ms
            results["triton_times"][num_experts].append(
                triton_time * 1000
            )  # Convert to ms
            results["speedups"][num_experts].append(speedup)

            # Print results
            print(f"  PyTorch argsort:      {pytorch_time*1000:.3f} ms")
            print(f"  Triton kernel sort:   {triton_time*1000:.3f} ms")
            print(f"  Speedup:              {speedup:.2f}x")

            # Verify correctness
            pytorch_sorted = expert_indices[torch.argsort(expert_indices)]
            triton_sorted = expert_indices[triton_counting_sort(expert_indices)]
            is_correct = torch.all(pytorch_sorted == triton_sorted)
            print(f"  Correct sorting:      {is_correct}")

            # Additional info
            print(
                f"  PyTorch time std dev: {torch.tensor(pytorch_times).std() * 1000:.3f} ms"
            )
            print(
                f"  Triton time std dev:  {torch.tensor(triton_times).std() * 1000:.3f} ms"
            )

    return results


def run_comprehensive_tests():
    """
    Run comprehensive correctness tests for various input sizes and expert counts.
    """
    print("Running comprehensive correctness tests...")

    test_configs = [
        # Sample test cases
        {"M_total": 100, "num_experts": 10},
        {"M_total": 1000, "num_experts": 32},
        {"M_total": 10000, "num_experts": 64},
        {"M_total": 20000, "num_experts": 128},
        # {"M_total": 100000, "num_experts": 256},
        # Edge cases
        {"M_total": 1, "num_experts": 1},  # Single element
        {"M_total": 1024, "num_experts": 1},  # All same expert
        {"M_total": 1024, "num_experts": 1024},  # Each token unique expert (up to max)
        # causes Aten assert in Pytorch:
        # {"M_total": 255, "num_experts": 256},  # Non-power-of-2 tokens
        # Power of 2 boundary cases
        {"M_total": 256, "num_experts": 16},  # Both powers of 2
        {"M_total": 257, "num_experts": 16},  # Just over power of 2
        {"M_total": 511, "num_experts": 16},  # Just under next power of 2
        {"M_total": 127, "num_experts": 16},  # Just under power of 2
        # {"M_total": 500000, "num_experts": 128},
        # {"M_total": 1000000, "num_experts": 256},
    ]

    all_passed = True
    results = []

    for config in test_configs:
        M_total = config["M_total"]
        num_experts = config["num_experts"]

        print(f"\nTesting with {M_total} tokens and {num_experts} experts:")

        # Run multiple seeds for robustness
        seeds = [2020, 123, 456]
        config_passed = True

        for seed in seeds:
            result = test_counting_sort_correctness(
                M_total=M_total, num_experts=num_experts, seed=seed, verbose=False
            )

            if not result["success"]:
                config_passed = False
                print(f"  - FAILED with seed {seed}:")
                print(f"    * Experts match: {result['experts_match']}")
                print(f"    * Stability match: {result['stability_match']}")

        if config_passed:
            print(f"  - PASSED all seeds")
        else:
            all_passed = False

        results.append(
            {"M_total": M_total, "num_experts": num_experts, "passed": config_passed}
        )

    print("\nOverall correctness test result:", "PASSED" if all_passed else "FAILED")
    return results


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Print device information if on CUDA
    if device.type == "cuda":
        device_props = torch.cuda.get_device_properties(device)
        print(f"Device: {device_props.name}")
        print(f"CUDA Capability: {device_props.major}.{device_props.minor}")
        print(f"Total memory: {device_props.total_memory / (1024**3):.2f} GB")
        print(f"CUDA Cores: {device_props.multi_processor_count} SMs")

    try:
        # Run comprehensive correctness tests
        test_results = run_comprehensive_tests()

        # Run benchmarks if on CUDA device
        if device.type == "cuda":
            # Use smaller sizes for faster benchmarking, with multiple expert counts
            benchmark_sizes = [10_000, 50_000, 100_000, 200_000, 500_000]
            expert_counts = [16, 64, 128, 256]  # Test multiple expert counts

            benchmark_results = benchmark_sorting_methods(
                M_totals=benchmark_sizes, num_experts_list=expert_counts
            )

        else:
            print("\nSkipping benchmarks as CUDA is not available.")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()
