#!/usr/bin/env python3
"""
Fixed standalone CUTLASS backward pass test.
Corrects the dimensional issues in the matrix operations.
"""

import torch
import torch.nn as nn

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass.cute.runtime import from_dlpack
    from torchtitan.experiments.kernels.blackwell.cute_grouped_gemm import (
        GroupedGemmKernel,
    )

    HAS_CUTLASS = True
    print("✅ CUTLASS imports successful")
except ImportError as e:
    HAS_CUTLASS = False
    print(f"❌ CUTLASS import failed: {e}")
    exit(1)


class FixedCutlassStrategy:
    """Fixed CUTLASS strategy with correct dimension handling"""

    def __init__(self):
        print("🔧 Initializing fixed CUTLASS strategy...")

        # Force CUDA context creation
        dummy = torch.zeros(1, device="cuda")
        dummy.cpu()

        self.DTYPE_TORCH = torch.bfloat16
        self.DTYPE_CUTLASS = cutlass.BFloat16
        self.ACC_DTYPE = cutlass.Float32
        self.ALIGNMENT = 16

        # Initialize kernel
        self.grouped_gemm = GroupedGemmKernel(
            acc_dtype=self.ACC_DTYPE,
            use_2cta_instrs=False,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
            tensormap_update_mode=utils.TensorMapUpdateMode.SMEM,
        )

        # Initialize hardware info
        self.hardware_info = utils.HardwareInfo()
        self.max_active_clusters = self.hardware_info.get_max_active_clusters(1)

        torch_stream = torch.cuda.current_stream()
        self.stream = cuda.CUstream(torch_stream.cuda_stream)

        self._compiled_kernels = {}
        self._tensormap_buffers = {}

        print(
            f"✅ Strategy initialized (max_active_clusters: {self.max_active_clusters})"
        )

    def _get_tensormap_buffer(self, device):
        if device not in self._tensormap_buffers:
            sm_count = self.hardware_info.get_max_active_clusters(1)
            tensormap_tensor = torch.zeros(
                (sm_count, 3, 128 // 8),
                dtype=torch.int64,
                device=device,
            )
            self._tensormap_buffers[device] = from_dlpack(
                tensormap_tensor, assumed_align=self.ALIGNMENT
            )
        return self._tensormap_buffers[device]

    def _compute_total_clusters(self, problem_sizes):
        cluster_tile_m = 128
        cluster_tile_n = 128

        total = 0
        for M, N, K, L in problem_sizes:
            clusters_m = (M + cluster_tile_m - 1) // cluster_tile_m
            clusters_n = (N + cluster_tile_n - 1) // cluster_tile_n
            total += clusters_m * clusters_n
        return total

    def _create_initial_tensors(self, problem_shape, device):
        M, N, K, L = problem_shape

        tensors = [
            torch.randn(M, K, dtype=self.DTYPE_TORCH, device=device),
            torch.randn(N, K, dtype=self.DTYPE_TORCH, device=device),
            torch.zeros(M, N, dtype=self.DTYPE_TORCH, device=device),
        ]

        cute_tensors = []
        for tensor in tensors:
            mnkl_tensor = tensor.unsqueeze(-1).contiguous()
            cute_tensor = from_dlpack(mnkl_tensor, assumed_align=self.ALIGNMENT)
            cute_tensor.element_type = self.DTYPE_CUTLASS
            cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=1)
            cute_tensors.append(cute_tensor)

        return cute_tensors

    def execute_cutlass_gemm_basic(self, A, B, C, operation_name="gemm"):
        """Execute basic CUTLASS GEMM: C = A @ B^T"""
        M, K = A.shape
        N, K_B = B.shape

        assert K == K_B, f"Inner dimension mismatch: {K} != {K_B}"
        assert C.shape == (
            M,
            N,
        ), f"Output shape mismatch: expected ({M}, {N}), got {C.shape}"

        print(f"  Executing {operation_name}: A{A.shape} @ B^T{B.shape} = C{C.shape}")

        L = 1
        device = A.device

        # Convert to MNKL format
        A_mnkl = A.unsqueeze(-1).contiguous()
        B_mnkl = B.unsqueeze(-1).contiguous()
        C_mnkl = C.unsqueeze(-1).contiguous()

        # Problem setup
        problem_sizes = [[M, N, K, L]]
        strides_abc = [
            [
                list(A_mnkl.stride()[:2]),
                list(B_mnkl.stride()[:2]),
                list(C_mnkl.stride()[:2]),
            ]
        ]
        ptrs_abc = [[A.data_ptr(), B.data_ptr(), C.data_ptr()]]

        print(f"    Problem: {problem_sizes[0]}")
        print(f"    Strides: {strides_abc[0]}")

        # Execute kernel
        self._execute_kernel(
            problem_sizes, strides_abc, ptrs_abc, device, operation_name
        )

        return C

    def compute_input_gradient(self, grad_output, weight, operation_name="input_grad"):
        """
        Compute input gradient: dX = dY @ W

        Args:
            grad_output: [M, N] - upstream gradient
            weight: [N, K] - weight matrix

        Returns:
            grad_input: [M, K] - input gradient
        """
        M, N = grad_output.shape
        N_w, K = weight.shape

        assert N == N_w, f"Dimension mismatch: grad_output has {N}, weight has {N_w}"

        print(
            f"  Computing input gradient: dY{grad_output.shape} @ W{weight.shape} = dX[{M}, {K}]"
        )

        # Since CUTLASS computes A @ B^T, and we want dY @ W:
        # We can compute this directly as dY @ W where CUTLASS treats W as B^T
        # So A = dY [M, N], B = W^T [K, N] (so B^T = W [N, K])
        weight_for_cutlass = (
            weight.t().contiguous()
        )  # [K, N] - this will be transposed to [N, K]
        grad_input = torch.zeros(
            M, K, dtype=self.DTYPE_TORCH, device=grad_output.device
        )

        print(
            f"    CUTLASS setup: dY{grad_output.shape} @ (W^T)^T{weight_for_cutlass.shape} = dX{grad_input.shape}"
        )

        return self.execute_cutlass_gemm_basic(
            grad_output, weight_for_cutlass, grad_input, operation_name
        )

    def compute_weight_gradient(
        self, grad_output, input_tokens, operation_name="weight_grad"
    ):
        """
        Compute weight gradient: dW = dY^T @ X

        Args:
            grad_output: [M, N] - upstream gradient
            input_tokens: [M, K] - input tokens

        Returns:
            grad_weight: [N, K] - weight gradient
        """
        M, N = grad_output.shape
        M_i, K = input_tokens.shape

        assert M == M_i, f"Dimension mismatch: grad_output has {M}, input has {M_i}"

        print(
            f"  Computing weight gradient: dY^T{grad_output.shape} @ X{input_tokens.shape} = dW[{N}, {K}]"
        )

        # Since CUTLASS computes A @ B^T, and we want dY^T @ X:
        # A = dY^T [N, M], B = X^T [K, M] (so B^T = X [M, K])
        grad_output_T = grad_output.t().contiguous()  # [N, M]
        input_for_cutlass = (
            input_tokens.t().contiguous()
        )  # [K, M] - this will be transposed to [M, K]
        grad_weight = torch.zeros(
            N, K, dtype=self.DTYPE_TORCH, device=grad_output.device
        )

        print(
            f"    CUTLASS setup: dY^T{grad_output_T.shape} @ (X^T)^T{input_for_cutlass.shape} = dW{grad_weight.shape}"
        )

        return self.execute_cutlass_gemm_basic(
            grad_output_T, input_for_cutlass, grad_weight, operation_name
        )

    def _execute_kernel(
        self, problem_sizes, strides_abc, ptrs_abc, device, operation_name
    ):
        """Execute the CUTLASS kernel"""
        num_groups = len(problem_sizes)

        # Convert to tensors
        problem_sizes_tensor = torch.tensor(
            problem_sizes, dtype=torch.int32, device=device
        )
        strides_tensor = torch.tensor(strides_abc, dtype=torch.int32, device=device)
        ptrs_tensor = torch.tensor(ptrs_abc, dtype=torch.int64, device=device)

        # Convert to CUTE tensors
        problem_sizes_cute = from_dlpack(
            problem_sizes_tensor, assumed_align=self.ALIGNMENT
        )
        strides_cute = from_dlpack(strides_tensor, assumed_align=self.ALIGNMENT)
        ptrs_cute = from_dlpack(ptrs_tensor, assumed_align=self.ALIGNMENT)

        # Get buffers
        tensormap_cute = self._get_tensormap_buffer(device)
        total_clusters = self._compute_total_clusters(problem_sizes)
        initial_tensors = self._create_initial_tensors(problem_sizes[0], device)

        # Compile kernel if needed
        cache_key = (num_groups, total_clusters, tuple(problem_sizes[0][:3]))

        if cache_key not in self._compiled_kernels:
            print(f"    Compiling kernel for {operation_name}...")
            self._compiled_kernels[cache_key] = cute.compile(
                self.grouped_gemm,
                *initial_tensors,
                num_groups,
                problem_sizes_cute,
                strides_cute,
                ptrs_cute,
                total_clusters,
                tensormap_cute,
                self.max_active_clusters,
                self.stream,
            )
            print(f"    ✅ Kernel compiled")

        # Execute
        compiled_kernel = self._compiled_kernels[cache_key]
        compiled_kernel(
            *initial_tensors,
            problem_sizes_cute,
            strides_cute,
            ptrs_cute,
            tensormap_cute,
            self.stream,
        )
        torch.cuda.synchronize()
        print(f"    ✅ {operation_name} executed")


def test_basic_cutlass_gemm():
    """Test basic CUTLASS GEMM operation"""
    print("\n🔍 Testing Basic CUTLASS GEMM")
    print("=" * 40)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    strategy = FixedCutlassStrategy()

    # Test matrices
    M, N, K = 64, 128, 256
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(N, K, dtype=dtype, device=device)
    C = torch.zeros(M, N, dtype=dtype, device=device)

    print(f"Testing: A{A.shape} @ B^T{B.shape} = C{C.shape}")

    # Reference result
    C_ref = torch.mm(A, B.t())
    print(f"Reference norm: {C_ref.norm().item():.4f}")

    # CUTLASS result
    strategy.execute_cutlass_gemm_basic(A, B, C, "basic_test")
    print(f"CUTLASS norm: {C.norm().item():.4f}")

    # Compare
    diff = torch.abs(C - C_ref).max().item()
    rel_diff = diff / C_ref.abs().max().item()

    print(f"Max difference: {diff:.2e}")
    print(f"Relative difference: {rel_diff:.2e}")

    if rel_diff < 1e-2:
        print("✅ Basic CUTLASS GEMM works!")
        return True
    else:
        print("❌ Basic CUTLASS GEMM failed!")
        return False


def test_input_gradient():
    """Test input gradient computation: dX = dY @ W"""
    print("\n🔍 Testing Input Gradient")
    print("=" * 30)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    strategy = FixedCutlassStrategy()

    # Problem: dX = dY @ W where dY:[M,N], W:[N,K] -> dX:[M,K]
    M, N, K = 32, 64, 128
    dY = torch.randn(M, N, dtype=dtype, device=device)
    W = torch.randn(N, K, dtype=dtype, device=device)

    print(f"Computing: dX = dY{dY.shape} @ W{W.shape}")

    # Reference PyTorch
    dX_ref = torch.mm(dY, W)  # [M,N] @ [N,K] = [M,K]
    print(f"Reference dX: {dX_ref.shape}, norm: {dX_ref.norm().item():.4f}")

    # CUTLASS computation
    dX_cutlass = strategy.compute_input_gradient(dY, W, "input_gradient")
    print(f"CUTLASS dX: {dX_cutlass.shape}, norm: {dX_cutlass.norm().item():.4f}")

    # Compare
    diff = torch.abs(dX_cutlass - dX_ref).max().item()
    rel_diff = diff / dX_ref.abs().max().item()

    print(f"Max difference: {diff:.2e}")
    print(f"Relative difference: {rel_diff:.2e}")

    if rel_diff < 1e-2:
        print("✅ Input gradient works!")
        return True
    else:
        print("❌ Input gradient failed!")
        print(f"First few elements - Ref: {dX_ref.flatten()[:5]}")
        print(f"First few elements - CUTLASS: {dX_cutlass.flatten()[:5]}")
        return False


def test_weight_gradient():
    """Test weight gradient computation: dW = dY^T @ X"""
    print("\n🔍 Testing Weight Gradient")
    print("=" * 30)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    strategy = FixedCutlassStrategy()

    # Problem: dW = dY^T @ X where dY:[M,N], X:[M,K] -> dW:[N,K]
    M, N, K = 32, 64, 128
    dY = torch.randn(M, N, dtype=dtype, device=device)
    X = torch.randn(M, K, dtype=dtype, device=device)

    print(f"Computing: dW = dY^T{dY.shape} @ X{X.shape}")

    # Reference PyTorch
    dW_ref = torch.mm(dY.t(), X)  # [N,M] @ [M,K] = [N,K]
    print(f"Reference dW: {dW_ref.shape}, norm: {dW_ref.norm().item():.4f}")

    # CUTLASS computation
    dW_cutlass = strategy.compute_weight_gradient(dY, X, "weight_gradient")
    print(f"CUTLASS dW: {dW_cutlass.shape}, norm: {dW_cutlass.norm().item():.4f}")

    # Compare
    diff = torch.abs(dW_cutlass - dW_ref).max().item()
    rel_diff = diff / dW_ref.abs().max().item()

    print(f"Max difference: {diff:.2e}")
    print(f"Relative difference: {rel_diff:.2e}")

    if rel_diff < 1e-2:
        print("✅ Weight gradient works!")
        return True
    else:
        print("❌ Weight gradient failed!")
        print(f"First few elements - Ref: {dW_ref.flatten()[:5]}")
        print(f"First few elements - CUTLASS: {dW_cutlass.flatten()[:5]}")
        return False


def test_complete_backward():
    """Test complete backward pass for a single expert"""
    print("\n🔍 Testing Complete Backward Pass")
    print("=" * 40)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    strategy = FixedCutlassStrategy()

    # Problem setup: Y = X @ W^T, given dY, compute dX and dW
    M, N, K = 32, 64, 128
    X = torch.randn(M, K, dtype=dtype, device=device)  # Input [M, K]
    W = torch.randn(N, K, dtype=dtype, device=device)  # Weight [N, K]
    dY = torch.randn(M, N, dtype=dtype, device=device)  # Upstream grad [M, N]

    print(f"Forward was: Y = X{X.shape} @ W^T{W.shape}")
    print(f"Given upstream grad dY{dY.shape}")
    print(f"Computing dX and dW...")

    # Reference PyTorch backward
    dX_ref = torch.mm(dY, W)  # [M,N] @ [N,K] = [M,K]
    dW_ref = torch.mm(dY.t(), X)  # [N,M] @ [M,K] = [N,K]

    print(f"Reference dX: {dX_ref.shape}, norm: {dX_ref.norm().item():.4f}")
    print(f"Reference dW: {dW_ref.shape}, norm: {dW_ref.norm().item():.4f}")

    # CUTLASS backward
    print("\nCUTLASS computation:")

    # Input gradient: dX = dY @ W
    dX_cutlass = strategy.compute_input_gradient(dY, W, "backward_input")

    # Weight gradient: dW = dY^T @ X
    dW_cutlass = strategy.compute_weight_gradient(dY, X, "backward_weight")

    print(f"CUTLASS dX: {dX_cutlass.shape}, norm: {dX_cutlass.norm().item():.4f}")
    print(f"CUTLASS dW: {dW_cutlass.shape}, norm: {dW_cutlass.norm().item():.4f}")

    # Compare both gradients
    dX_diff = torch.abs(dX_cutlass - dX_ref).max().item()
    dX_rel_diff = dX_diff / dX_ref.abs().max().item()

    dW_diff = torch.abs(dW_cutlass - dW_ref).max().item()
    dW_rel_diff = dW_diff / dW_ref.abs().max().item()

    print(f"\nComparison:")
    print(f"dX max diff: {dX_diff:.2e} (relative: {dX_rel_diff:.2e})")
    print(f"dW max diff: {dW_diff:.2e} (relative: {dW_rel_diff:.2e})")

    success = dX_rel_diff < 1e-2 and dW_rel_diff < 1e-2

    if success:
        print("✅ Complete backward pass works!")
    else:
        print("❌ Complete backward pass failed!")
        if dX_rel_diff >= 1e-2:
            print("   Input gradient has large errors")
        if dW_rel_diff >= 1e-2:
            print("   Weight gradient has large errors")

    return success


def test_grouped_backward():
    """Test backward pass with multiple experts (minimal version)"""
    print("\n🔍 Testing Grouped Backward (2 experts)")
    print("=" * 45)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    strategy = FixedCutlassStrategy()

    # Setup: 2 experts, simple token distribution
    num_experts = 2
    tokens_per_expert = 16
    total_tokens = num_experts * tokens_per_expert
    in_features = 64
    out_features = 128

    # Create data
    X = torch.randn(total_tokens, in_features, dtype=dtype, device=device)
    W = torch.randn(num_experts, out_features, in_features, dtype=dtype, device=device)
    dY = torch.randn(total_tokens, out_features, dtype=dtype, device=device)

    print(f"Setup: {num_experts} experts, {tokens_per_expert} tokens each")
    print(f"X: {X.shape}, W: {W.shape}, dY: {dY.shape}")

    # Reference PyTorch grouped backward
    dX_ref = torch.zeros_like(X)
    dW_ref = torch.zeros_like(W)

    for expert_idx in range(num_experts):
        start_idx = expert_idx * tokens_per_expert
        end_idx = start_idx + tokens_per_expert

        expert_X = X[start_idx:end_idx]  # [tokens_per_expert, in_features]
        expert_W = W[expert_idx]  # [out_features, in_features]
        expert_dY = dY[start_idx:end_idx]  # [tokens_per_expert, out_features]

        # Compute gradients for this expert
        expert_dX = torch.mm(
            expert_dY, expert_W
        )  # [tokens, in] = [tokens, out] @ [out, in]
        expert_dW = torch.mm(
            expert_dY.t(), expert_X
        )  # [out, in] = [out, tokens] @ [tokens, in]

        dX_ref[start_idx:end_idx] = expert_dX
        dW_ref[expert_idx] = expert_dW

    print(f"Reference dX norm: {dX_ref.norm().item():.4f}")
    print(f"Reference dW norm: {dW_ref.norm().item():.4f}")

    # CUTLASS grouped backward
    dX_cutlass = torch.zeros_like(X)
    dW_cutlass = torch.zeros_like(W)

    for expert_idx in range(num_experts):
        start_idx = expert_idx * tokens_per_expert
        end_idx = start_idx + tokens_per_expert

        expert_X = X[start_idx:end_idx]
        expert_W = W[expert_idx]
        expert_dY = dY[start_idx:end_idx]

        print(f"\nExpert {expert_idx}:")

        # Input gradient: dX = dY @ W
        expert_dX_cutlass = strategy.compute_input_gradient(
            expert_dY, expert_W, f"expert_{expert_idx}_input"
        )
        dX_cutlass[start_idx:end_idx] = expert_dX_cutlass

        # Weight gradient: dW = dY^T @ X
        expert_dW_cutlass = strategy.compute_weight_gradient(
            expert_dY, expert_X, f"expert_{expert_idx}_weight"
        )
        dW_cutlass[expert_idx] = expert_dW_cutlass

    print(f"\nCUTLASS dX norm: {dX_cutlass.norm().item():.4f}")
    print(f"CUTLASS dW norm: {dW_cutlass.norm().item():.4f}")

    # Compare
    dX_diff = torch.abs(dX_cutlass - dX_ref).max().item()
    dX_rel_diff = dX_diff / dX_ref.abs().max().item()

    dW_diff = torch.abs(dW_cutlass - dW_ref).max().item()
    dW_rel_diff = dW_diff / dW_ref.abs().max().item()

    print(f"\nComparison:")
    print(f"dX max diff: {dX_diff:.2e} (relative: {dX_rel_diff:.2e})")
    print(f"dW max diff: {dW_diff:.2e} (relative: {dW_rel_diff:.2e})")

    success = dX_rel_diff < 1e-2 and dW_rel_diff < 1e-2

    if success:
        print("✅ Grouped backward pass works!")
    else:
        print("❌ Grouped backward pass failed!")

    return success


def main():
    """Main test sequence"""
    print("🧪 Fixed Standalone CUTLASS Backward Test")
    print("=" * 55)

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return

    if not HAS_CUTLASS:
        print("❌ CUTLASS not available")
        return

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")

    tests = [
        ("Basic CUTLASS GEMM", test_basic_cutlass_gemm),
        ("Input Gradient", test_input_gradient),
        ("Weight Gradient", test_weight_gradient),
        ("Complete Backward", test_complete_backward),
        ("Grouped Backward", test_grouped_backward),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n" + "=" * 60)
        try:
            success = test_func()
            results.append((test_name, "✅ PASS" if success else "❌ FAIL"))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            import traceback

            traceback.print_exc()
            results.append((test_name, "💥 CRASH"))

    # Summary
    print(f"\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)

    for test_name, result in results:
        print(f"{test_name:<20} {result}")

    # Count successes
    passes = sum(1 for _, result in results if "PASS" in result)
    total = len(results)

    print(f"\nOverall: {passes}/{total} tests passed")

    if passes == total:
        print("🎉 All tests passed! CUTLASS backward is working correctly.")
    else:
        print("🔧 Some tests failed. Check the specific failures above.")


if __name__ == "__main__":
    main()
