#!/usr/bin/env python3
"""
Standalone CUTLASS backward pass test.
Self-contained with no external dependencies beyond basic CUTLASS.


current:

CUTLASS computation:
  Executing backward_input: Atorch.Size([32, 64]) @ B^Ttorch.Size([64, 128]) = Ctorch.Size([32, 128])
    Problem: [32, 128, 64, 1]
    Strides: [[64, 1], [128, 1], [128, 1]]
max_dynamic_shared_memory: 232448
max_active_blocks: 1
    Compiling kernel for backward_input...
    ✅ Kernel compiled
    ✅ backward_input executed
❌ Complete Backward crashed: Inner dimension mismatch: 32 != 128
Traceback (most recent call last):
  File "/data/users/less/torchtitan/torchtitan/experiments/deepseek_v3/dsl_back_standalone.py", line 576, in main
    success = test_func()
              ^^^^^^^^^^^
  File "/data/users/less/torchtitan/torchtitan/experiments/deepseek_v3/dsl_back_standalone.py", line 409, in test_complete_backward
    strategy.execute_cutlass_gemm(dY_T, X, dW_cutlass, "backward_weight")
  File "/data/users/less/torchtitan/torchtitan/experiments/deepseek_v3/dsl_back_standalone.py", line 123, in execute_cutlass_gemm
    assert K == K_B, f"Inner dimension mismatch: {K} != {K_B}"
           ^^^^^^^^
AssertionError: Inner dimension mismatch: 32 != 128

============================================================

🔍 Testing Grouped Backward (2 experts)
=============================================
🔧 Initializing standalone CUTLASS strategy...
cute hardware - device_id 0
cute hardware - driver_version 12080
max_dynamic_shared_memory: 232448
max_active_blocks: 1
✅ Strategy initialized (max_active_clusters: 148)
Setup: 2 experts, 16 tokens each
X: torch.Size([32, 64]), W: torch.Size([2, 128, 64]), dY: torch.Size([32, 128])
Reference dX norm: 520.0000
Reference dW norm: 520.0000

Expert 0:
❌ Grouped Backward crashed: Inner dimension mismatch: 128 != 16
Traceback (most recent call last):
  File "/data/users/less/torchtitan/torchtitan/experiments/deepseek_v3/dsl_back_standalone.py", line 576, in main
    success = test_func()
              ^^^^^^^^^^^
  File "/data/users/less/torchtitan/torchtitan/experiments/deepseek_v3/dsl_back_standalone.py", line 509, in test_grouped_backward
    strategy.execute_cutlass_gemm(W_T, dY_T, dX_T, f"expert_{expert_idx}_input")
  File "/data/users/less/torchtitan/torchtitan/experiments/deepseek_v3/dsl_back_standalone.py", line 123, in execute_cutlass_gemm
    assert K == K_B, f"Inner dimension mismatch: {K} != {K_B}"
           ^^^^^^^^
AssertionError: Inner dimension mismatch: 128 != 16

============================================================
📊 FINAL RESULTS
============================================================
Basic CUTLASS GEMM   ✅ PASS
Input Gradient       ✅ PASS
Weight Gradient      ✅ PASS
Complete Backward    💥 CRASH
Grouped Backward     💥 CRASH

Overall: 3/5 tests passed
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


class StandaloneCutlassStrategy:
    """Self-contained CUTLASS strategy for testing"""

    def __init__(self):
        print("🔧 Initializing standalone CUTLASS strategy...")

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

    def execute_cutlass_gemm(self, A, B, C, operation_name="gemm"):
        """Execute a single CUTLASS GEMM: C = A @ B^T"""
        M, K = A.shape
        N, K_B = B.shape

        # For input gradient computation: dX = dY @ W
        # dY is [M, N] and W is [N, K], so we need to handle this special case
        if operation_name == "backward_input":
            # For backward_input, we expect A=dY [M,N] and B=W [N,K]
            # The inner dimensions should match (N == N)
            assert K == N, f"Inner dimension mismatch for backward_input: {K} != {N}"
            # Swap K_B and N for the assertion below
            K_B, N = N, K_B

        assert K == K_B, f"Inner dimension mismatch: {K} != {K_B}"
        assert C.shape == (
            M,
            N,
        ), f"Output shape mismatch: expected ({M}, {N}), got {C.shape}"

        L = 1
        device = A.device

        print(f"  Executing {operation_name}: A{A.shape} @ B^T{B.shape} = C{C.shape}")

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

    strategy = StandaloneCutlassStrategy()

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
    strategy.execute_cutlass_gemm(A, B, C, "basic_test")
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

    strategy = StandaloneCutlassStrategy()

    # Problem: dX = dY @ W where dY:[M,N], W:[N,K] -> dX:[M,K]
    M, N, K = 32, 64, 128
    dY = torch.randn(M, N, dtype=dtype, device=device)
    W = torch.randn(N, K, dtype=dtype, device=device)

    print(f"Computing: dX = dY{dY.shape} @ W{W.shape}")

    # Reference PyTorch
    dX_ref = torch.mm(dY, W)  # [M,N] @ [N,K] = [M,K]
    print(f"Reference dX: {dX_ref.shape}, norm: {dX_ref.norm().item():.4f}")

    # CUTLASS approach: reformulate as dX^T = W^T @ dY^T
    print("CUTLASS approach: dX^T = W^T @ dY^T")

    W_T = W.t().contiguous()  # [K, N]
    dY_T = dY.t().contiguous()  # [N, M]
    dX_T = torch.zeros(K, M, dtype=dtype, device=device)  # [K, M]

    print(f"  W^T{W_T.shape} @ (dY^T)^T{dY_T.shape} = dX^T{dX_T.shape}")
    print(f"  Note: CUTLASS computes W^T @ dY^T^T = W^T @ dY")

    # Execute: W^T @ dY^T^T (CUTLASS transposes second operand)
    strategy.execute_cutlass_gemm(W_T, dY, dX_T, "input_gradient")

    # Transpose back to get dX
    dX_cutlass = dX_T.t()  # [M, K]
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

    strategy = StandaloneCutlassStrategy()

    # Problem: dW = dY^T @ X where dY:[M,N], X:[M,K] -> dW:[N,K]
    M, N, K = 32, 64, 128
    dY = torch.randn(M, N, dtype=dtype, device=device)
    X = torch.randn(M, K, dtype=dtype, device=device)

    print(f"Computing: dW = dY^T{dY.shape} @ X{X.shape}")

    # Reference PyTorch
    dW_ref = torch.mm(dY.t(), X)  # [N,M] @ [M,K] = [N,K]
    print(f"Reference dW: {dW_ref.shape}, norm: {dW_ref.norm().item():.4f}")

    # CUTLASS approach: dW = dY^T @ X
    # Since CUTLASS computes A @ B^T, we use A = dY^T, B^T = X^T
    # So CUTLASS computes dY^T @ (X^T)^T = dY^T @ X = dW
    print("CUTLASS approach: dY^T @ X using A @ B^T format")

    dY_T = dY.t().contiguous()  # [N, M]
    X_T = X.t().contiguous()  # [K, M]
    dW_cutlass = torch.zeros(N, K, dtype=dtype, device=device)  # [N, K]

    print(f"  dY^T{dY_T.shape} @ (X^T)^T{X_T.shape} = dW{dW_cutlass.shape}")
    print(f"  Note: CUTLASS computes dY^T @ X^T^T = dY^T @ X")

    # Execute: dY^T @ X^T^T (CUTLASS transposes second operand)
    strategy.execute_cutlass_gemm(dY_T, X_T, dW_cutlass, "weight_gradient")

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

    strategy = StandaloneCutlassStrategy()

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
    dX_cutlass = torch.zeros(M, K, dtype=dtype, device=device)

    # For input gradient, we need to handle the special case in execute_cutlass_gemm
    strategy.execute_cutlass_gemm(dY, W, dX_cutlass, "backward_input")

    # Weight gradient: dW = dY^T @ X
    dY_T = dY.t().contiguous()  # [N, M]
    dW_cutlass = torch.zeros(N, K, dtype=dtype, device=device)

    strategy.execute_cutlass_gemm(dY_T, X, dW_cutlass, "backward_weight")

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

    strategy = StandaloneCutlassStrategy()

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

        # Input gradient: dX^T = W^T @ dY^T
        W_T = expert_W.t().contiguous()
        dY_T = expert_dY.t().contiguous()
        dX_T = torch.zeros(in_features, tokens_per_expert, dtype=dtype, device=device)

        strategy.execute_cutlass_gemm(W_T, dY_T, dX_T, f"expert_{expert_idx}_input")
        dX_cutlass[start_idx:end_idx] = dX_T.t()

        # Weight gradient: dW = dY^T @ X
        dY_T = expert_dY.t().contiguous()
        X_T = expert_X.t().contiguous()
        expert_dW_cutlass = torch.zeros(
            out_features, in_features, dtype=dtype, device=device
        )

        strategy.execute_cutlass_gemm(
            dY_T, X_T, expert_dW_cutlass, f"expert_{expert_idx}_weight"
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
    print("🧪 Standalone CUTLASS Backward Test")
    print("=" * 50)

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
