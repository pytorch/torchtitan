#!/usr/bin/env python3
"""
Simple example showing how to use GroupedGemmKernel with PyTorch tensors.

This example demonstrates the key differences from DenseGemmKernel:
1. Multiple problem sizes in one kernel launch
2. Persistent scheduling across groups
3. Tensormap updates for different tensor configurations
"""

import time

import torch

# Import CUTLASS components
try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import cutlass.utils as utils
    from cute_grouped_gemm import GroupedGemmKernel
    from cutlass.cute.runtime import from_dlpack

    HAS_CUTLASS = True
except ImportError:
    HAS_CUTLASS = False


def simple_grouped_gemm_example():
    """
    Simple example with 3 different-sized GEMMs in one kernel launch.
    """

    print("=== Simple Grouped GEMM Example ===")

    if not HAS_CUTLASS:
        print("CUTLASS not available")
        return

    # Define 3 groups with different problem sizes
    # Format: (M, N, K, L) where L must be 1 for grouped GEMM
    problem_sizes = [
        (512, 256, 128, 1),  # Group 0: Small
        (1024, 512, 256, 1),  # Group 1: Medium
        (768, 384, 192, 1),  # Group 2: Different aspect ratio
    ]

    num_groups = len(problem_sizes)
    device = "cuda"

    print(f"Number of groups: {num_groups}")
    for i, (M, N, K, L) in enumerate(problem_sizes):
        print(f"  Group {i}: C[{M},{N}] = A[{M},{K}] @ B[{K},{N}]")

    # 1. Create PyTorch tensors for each group
    print("\n1. Creating PyTorch tensors...")

    torch_tensors = []
    cute_tensors = []
    strides = []
    pointers = []

    for i, (M, N, K, L) in enumerate(problem_sizes):
        # Create standard PyTorch tensors
        A = torch.randn(M, K, dtype=torch.float16, device=device)
        B = torch.randn(K, N, dtype=torch.float16, device=device)
        C = torch.zeros(M, N, dtype=torch.float16, device=device)

        # Convert to MNKL format by appending L=1 dimension
        A_mnkl = A.unsqueeze(-1).contiguous()  # (M, K) -> (M, K, 1)
        B_mnkl = B.transpose(0, 1).unsqueeze(-1).contiguous()  # (K, N) -> (N, K, 1)
        C_mnkl = C.unsqueeze(-1).contiguous()  # (M, N) -> (M, N, 1)

        # Create CUTE tensors
        A_cute = from_dlpack(A_mnkl, assumed_align=16)
        B_cute = from_dlpack(B_mnkl, assumed_align=16)
        C_cute = from_dlpack(C_mnkl, assumed_align=16)

        # Set CUTE properties
        A_cute.element_type = cutlass.Float16
        B_cute.element_type = cutlass.Float16
        C_cute.element_type = cutlass.Float16

        A_cute = A_cute.mark_layout_dynamic(leading_dim=1)
        B_cute = B_cute.mark_layout_dynamic(leading_dim=1)
        C_cute = C_cute.mark_layout_dynamic(leading_dim=1)

        # Store data
        torch_tensors.append((A, B, C))  # Original PyTorch tensors
        cute_tensors.append((A_cute, B_cute, C_cute))

        # Strides for GroupedGemm: [A_strides, B_strides, C_strides]
        # A: K-major (stride_k=1, stride_m=K), B: K-major (stride_k=1, stride_n=K), C: N-major (stride_n=1, stride_m=N)
        strides.append([(1, K), (1, K), (1, N)])

        # Pointers to tensor data
        pointers.append([A_mnkl.data_ptr(), B_mnkl.data_ptr(), C_mnkl.data_ptr()])

    # 2. Convert metadata to tensors
    print("\n2. Converting metadata to tensors...")

    # Problem sizes tensor: (num_groups, 4)
    problem_sizes_tensor = torch.tensor(problem_sizes, dtype=torch.int32, device=device)
    problem_sizes_cute = from_dlpack(problem_sizes_tensor, assumed_align=16)

    # Strides tensor: (num_groups, 3, 2)
    strides_tensor = torch.tensor(strides, dtype=torch.int32, device=device)
    strides_cute = from_dlpack(strides_tensor, assumed_align=16)

    # Pointers tensor: (num_groups, 3)
    pointers_tensor = torch.tensor(pointers, dtype=torch.int64, device=device)
    pointers_cute = from_dlpack(pointers_tensor, assumed_align=16)

    print(f"  Problem sizes tensor: {problem_sizes_tensor.shape}")
    print(f"  Strides tensor: {strides_tensor.shape}")
    print(f"  Pointers tensor: {pointers_tensor.shape}")

    # 3. Create tensormap buffer
    print("\n3. Creating tensormap buffer...")

    hardware_info = utils.HardwareInfo()
    sm_count = hardware_info.get_max_active_clusters(1)

    tensormap_tensor = torch.zeros(
        (sm_count, 3, 128 // 8),  # (SMs, num_tensormaps, bytes_per_tensormap // 8)
        dtype=torch.int64,
        device=device,
    )
    tensormap_cute = from_dlpack(tensormap_tensor, assumed_align=16)

    print(f"  Tensormap buffer: {tensormap_tensor.shape}")

    # 4. Setup GroupedGemmKernel
    print("\n4. Setting up GroupedGemmKernel...")

    grouped_gemm = GroupedGemmKernel(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=False,
        mma_tiler_mn=(128, 64),
        cluster_shape_mn=(1, 1),
        tensormap_update_mode=utils.TensorMapUpdateMode.SMEM,
    )

    # 5. Compute grid parameters
    print("\n5. Computing grid parameters...")

    # Calculate total clusters needed
    def compute_total_clusters():
        cta_tile_m = 128  # mma_tiler_mn[0] / (2 if use_2cta_instrs else 1)
        cta_tile_n = 64  # mma_tiler_mn[1]
        cluster_m = 1  # cluster_shape_mn[0]
        cluster_n = 1  # cluster_shape_mn[1]

        cluster_tile_m = cta_tile_m * cluster_m
        cluster_tile_n = cta_tile_n * cluster_n

        total = 0
        for M, N, K, L in problem_sizes:
            clusters_m = (M + cluster_tile_m - 1) // cluster_tile_m
            clusters_n = (N + cluster_tile_n - 1) // cluster_tile_n
            total += clusters_m * clusters_n
        return total

    total_clusters = compute_total_clusters()
    max_active_clusters = hardware_info.get_max_active_clusters(
        1
    )  # cluster_shape product

    print(f"  Total clusters: {total_clusters}")
    print(f"  Max active clusters: {max_active_clusters}")

    # 6. Choose initial tensors (smallest ones for tensormap initialization)
    print("\n6. Selecting initial tensors...")

    # Use tensors from group with smallest sizes
    sizes = [(M * K, N * K, M * N) for M, N, K, L in problem_sizes]
    min_a_idx = min(range(num_groups), key=lambda i: sizes[i][0])
    min_b_idx = min(range(num_groups), key=lambda i: sizes[i][1])
    min_c_idx = min(range(num_groups), key=lambda i: sizes[i][2])

    initial_A = cute_tensors[min_a_idx][0]
    initial_B = cute_tensors[min_b_idx][1]
    initial_C = cute_tensors[min_c_idx][2]

    print(
        f"  Using A from group {min_a_idx}, B from group {min_b_idx}, C from group {min_c_idx}"
    )

    # 7. Compile and run kernel
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

    # 8. Compare results# 8. Verify results...
    print("\n8. Verifying results...")

    for i, ((A, B, C), (M, N, K, L)) in enumerate(zip(torch_tensors, problem_sizes)):
        # Compute reference with PyTorch
        C_ref = torch.mm(A, B)

        # Compare with CUTLASS result using norm-based comparison
        norm_diff = torch.norm(C - C_ref).item()
        norm_ref = torch.norm(C_ref).item()
        relative_error = norm_diff / norm_ref if norm_ref > 0 else float("inf")
        total_flops = 0
        print(
            f"  Group {i}: norm_diff = {norm_diff:.2e}, rel_error = {relative_error:.2e}"
        )

        # Print first few elements for debugging
        print(f"    C_ref first 3 elements: {C_ref.flatten()[:3].tolist()}")
        print(f"    C result first 3 elements: {C.flatten()[:3].tolist()}")

        if relative_error > 1e-2:  # Using 1% relative error as tolerance
            print(f"    ✗ Group {i} failed tolerance check")
            all_correct = False
        else:
            print(f"    ✓ Group {i} passed")

        total_flops += 2 * M * N * K

    # 9. Continue using results in PyTorch
    print("\n9. Using results in PyTorch operations...")

    for i, (A, B, C) in enumerate(torch_tensors):
        # C now contains the GEMM result and can be used directly in PyTorch!
        if torch.all(C == 0):
            print(
                f"  ⚠️ Warning: Group {i} C tensor is all zeros - kernel may not have written to it"
            )
        # Example operations
        C_relu = torch.relu(C)
        C_norm = torch.norm(C)

        # Use in another computation
        bias = torch.ones_like(C) * 0.1
        C_biased = C + bias

        print(
            f"  Group {i}: norm={C_norm.item():.3f}, after_bias_norm={torch.norm(C_biased).item():.3f}"
        )

    # 10. Performance summary
    print(f"\n10. Performance Summary:")
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
