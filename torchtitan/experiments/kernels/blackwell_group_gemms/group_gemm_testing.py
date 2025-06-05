#!/usr/bin/env python3
"""
Grouped GEMM example using the TensorCuteConverter utility.

This example demonstrates how to:
1. Use the GroupedGemmConverter to create and manage grouped GEMM tensors
2. Execute a grouped GEMM operation with proper tensor handling
3. Verify results against PyTorch reference implementation
"""

import time

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True

# Import CUTLASS components
try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import cutlass.utils as utils
    from cute_grouped_gemm import GroupedGemmKernel
    from cutlass.cute.runtime import from_dlpack
    from tensor_cute_converter import GroupedGemmConverter, TensorMapConverter

    HAS_CUTLASS = True
except ImportError:
    HAS_CUTLASS = False


def simple_grouped_gemm_example():
    """
    Grouped GEMM example using the TensorCuteConverter utility.
    """

    print("=== Grouped GEMM Example with TensorCuteConverter ===")

    if not HAS_CUTLASS:
        print("CUTLASS not available")
        return

    # Define 3 groups with different problem sizes
    problem_sizes = [
        (512, 256, 128),  # Group 0: Small
        (512, 256, 128),
        (512, 256, 128),
        # (1024, 512, 256),  # Group 1: Medium
        # (768, 384, 192),  # Group 2: Different aspect ratio
    ]

    device = torch.device("cuda")
    dtype_torch = torch.float16
    dtype_cutlass = cutlass.Float16

    print(f"Number of groups: {len(problem_sizes)}")
    for i, (M, N, K) in enumerate(problem_sizes):
        print(f"  Group {i}: C[{M},{N}] = A[{M},{K}] @ B[{K},{N}]")

    # 1. Create tensors using GroupedGemmConverter
    print("\n1. Creating tensors using GroupedGemmConverter...")

    # Initialize the converter
    converter = GroupedGemmConverter(device=device, alignment=16)

    # Create grouped tensors with consistent majorness across all groups
    # For CUTLASS grouped GEMM, the majorness must be the same across all groups
    # A is K-major (row-major), B is K-major (column-major after transpose), C is N-major (row-major)
    print("\nEnsuring consistent majorness across all groups:")
    print("  A: K-major (row-major)")
    print("  B: K-major (column-major after transpose)")
    print("  C: N-major (row-major)")

    # Create grouped tensors
    grouped_result = converter.create_grouped_tensors(
        problem_sizes=problem_sizes, dtype=dtype_torch, fill_random=True
    )

    # Debug: Print strides and pointers for each group
    print("\nDebug - Strides for each group:")
    for i, stride_group in enumerate(grouped_result["metadata_tensors"]["strides"]):
        print(
            f"  Group {i} strides: A={stride_group[0]}, B={stride_group[1]}, C={stride_group[2]}"
        )

    print("\nDebug - Pointers for each group:")
    for i, ptr_group in enumerate(grouped_result["metadata_tensors"]["pointers"]):
        print(
            f"  Group {i} pointers: A=0x{ptr_group[0]:x}, B=0x{ptr_group[1]:x}, C=0x{ptr_group[2]:x}"
        )

    print("\nDebug - Problem sizes for each group:")
    for i, size_group in enumerate(grouped_result["metadata_tensors"]["problem_sizes"]):
        print(f"  Group {i} problem size: {size_group.tolist()}")

    # Extract components from the conversion result
    num_groups = grouped_result["num_groups"]
    original_tensors = grouped_result["original_tensors"]
    cute_tensors = grouped_result["cute_tensors"]

    # Get metadata tensors and their CUTE versions
    problem_sizes_tensor = grouped_result["metadata_tensors"]["problem_sizes"]
    strides_tensor = grouped_result["metadata_tensors"]["strides"]
    pointers_tensor = grouped_result["metadata_tensors"]["pointers"]

    problem_sizes_cute = grouped_result["metadata"]["problem_sizes"]
    strides_cute = grouped_result["metadata"]["strides"]
    pointers_cute = grouped_result["metadata"]["pointers"]

    print(f"  Successfully created {num_groups} tensor groups")
    print(f"  Problem sizes tensor: {problem_sizes_tensor.shape}")
    print(f"  Strides tensor: {strides_tensor.shape}")
    print(f"  Pointers tensor: {pointers_tensor.shape}")
    print(f"  Actual strides values:\n{strides_tensor}")

    # 2. Create tensormap buffer using TensorMapConverter
    print("\n2. Creating tensormap buffer...")

    hardware_info = utils.HardwareInfo()
    sm_count = hardware_info.get_max_active_clusters(1)

    # Use TensorMapConverter to create the tensormap buffer
    tensormap_converter = TensorMapConverter(device=device, alignment=16)
    tensormap_result = tensormap_converter.create_tensormap_buffer(
        sm_count=sm_count, tensor_count=3, element_size=8  # A, B, C  # int64
    )

    tensormap_tensor = tensormap_result["buffer"]
    tensormap_cute = tensormap_result["cute"]

    print(f"  Tensormap buffer: {tensormap_tensor.shape}")

    # 4. Setup GroupedGemmKernel
    print("\n4. Setting up GroupedGemmKernel...")

    grouped_gemm = GroupedGemmKernel(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=False,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        tensormap_update_mode=utils.TensorMapUpdateMode.SMEM,
    )

    # 5. Compute grid parameters
    # 5. Computing grid parameters...

    def compute_total_clusters():
        cta_tile_m = 128
        cta_tile_n = 128
        cluster_m = 4
        cluster_n = 4

        cluster_tile_m = cta_tile_m * cluster_m
        cluster_tile_n = cta_tile_n * cluster_n

        total = 0
        for M, N, K in problem_sizes:
            clusters_m = (M + cluster_tile_m - 1) // cluster_tile_m
            clusters_n = (N + cluster_tile_n - 1) // cluster_tile_n
            total += clusters_m * clusters_n
        return total

    total_clusters = compute_total_clusters()
    max_active_clusters = hardware_info.get_max_active_clusters(1)

    print(f"  Total clusters: {total_clusters}")
    print(f"  Max active clusters: {max_active_clusters}")

    # 6. Choose initial tensors
    # 6. Selecting initial tensors...

    # Instead of using smallest tensors, always use tensors from group 0
    # This is a workaround to see if the issue is with the initial tensors
    group_idx = 0

    # Access the CUTE tensors from group 0
    initial_A = cute_tensors[group_idx]["A"]
    initial_B = cute_tensors[group_idx]["B"]
    initial_C = cute_tensors[group_idx]["C"]

    print(f"\nDebug - Initial tensor selection:")
    print(f"  Using A, B, C from group {group_idx}")
    print(f"  A shape: {problem_sizes[group_idx][0]}x{problem_sizes[group_idx][2]}")
    print(f"  B shape: {problem_sizes[group_idx][2]}x{problem_sizes[group_idx][1]}")
    print(f"  C shape: {problem_sizes[group_idx][0]}x{problem_sizes[group_idx][1]}")

    # Compile and run kernel
    print("\n7. Compiling and executing kernel...")

    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    try:
        # Compile
        compiled_kernel = cute.compile(
            grouped_gemm,
            initial_A,
            initial_B,
            initial_C,
            num_groups,
            problem_sizes_cute,
            strides_cute,
            pointers_cute,
            total_clusters,
            tensormap_cute,
            max_active_clusters,
            stream,
        )

        print("  ✓ Kernel compiled successfully")

        # Execute
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        compiled_kernel(
            initial_A,
            initial_B,
            initial_C,
            problem_sizes_cute,
            strides_cute,
            pointers_cute,
            tensormap_cute,
            stream,
        )

        torch.cuda.synchronize()
        execution_time = time.perf_counter() - start_time

        print(f"  ✓ Kernel executed in {execution_time*1000:.2f} ms")

    except Exception as e:
        print(f"  ✗ Kernel failed: {e}")
        return

    # 8. Verify results
    # 8. Verifying results...

    all_correct = True
    total_flops = 0

    for i, (original_group, (M, N, K)) in enumerate(
        zip(original_tensors, problem_sizes)
    ):
        print(f"  Group {i} original_group type: {type(original_group)}")
        print(f"  Group {i} original_group keys: {original_group.keys()}")

        # Access tensors from the dictionary
        A = original_group["A"]
        B = original_group["B"]
        C = original_group["C"]

        # Print tensor shapes for debugging
        print(f"    A shape: {A.shape}, B shape: {B.shape}, C shape: {C.shape}")

        # Debug: Print tensor strides
        print(
            f"    A strides: {A.stride()}, B strides: {B.stride()}, C strides: {C.stride()}"
        )

        # Compute reference with PyTorch
        C_ref = torch.matmul(A, B)

        # Compare with CUTLASS result
        norm_diff = torch.norm(C - C_ref).item()
        norm_ref = torch.norm(C_ref).item()
        relative_error = norm_diff / norm_ref if norm_ref > 0 else float("inf")

        print(
            f"  Group {i}: norm_diff = {norm_diff:.2e}, rel_error = {relative_error:.2e}"
        )

        # Print first few elements for debugging
        print(f"    C_ref first 3 elements: {C_ref.flatten()[:3].tolist()}")
        print(f"    C result first 3 elements: {C.flatten()[:3].tolist()}")

        # Print more detailed debug info for failing groups
        if relative_error > 1e-2:
            # Check where the differences are
            diff = torch.abs(C - C_ref)
            max_diff_idx = torch.argmax(diff.flatten())
            max_diff_val = diff.flatten()[max_diff_idx]
            max_diff_pos = np.unravel_index(max_diff_idx.item(), C.shape)

            print(f"    Max difference: {max_diff_val:.4f} at position {max_diff_pos}")
            print(f"    C_ref at max diff: {C_ref[max_diff_pos].item():.4f}")
            print(f"    C result at max diff: {C[max_diff_pos].item():.4f}")

            # Check if there are any NaN or Inf values
            nan_count = torch.isnan(C).sum().item()
            inf_count = torch.isinf(C).sum().item()
            zero_count = (C == 0).sum().item()

            print(
                f"    NaN count: {nan_count}, Inf count: {inf_count}, Zero count: {zero_count}"
            )
            print(f"    ✗ Group {i} failed tolerance check")
            all_correct = False
        else:
            print(f"    ✓ Group {i} passed")

        total_flops += 2 * M * N * K

    # 9. Performance summary
    print(f"\n9. Performance Summary:")
    print(f"    Execution time: {execution_time*1000:.2f} ms")
    print(f"    Total FLOPs: {total_flops/1e9:.2f} GFLOP")
    print(f"    Throughput: {total_flops/execution_time/1e12:.2f} TFLOP/s")
    print(f"    All results correct: {all_correct}")

    if all_correct:
        print("\n🎉 Grouped GEMM completed successfully!")
        print("All output tensors are ready for continued PyTorch operations!")
    else:
        print("\n❌ Some results were incorrect")


if __name__ == "__main__":
    simple_grouped_gemm_example()
