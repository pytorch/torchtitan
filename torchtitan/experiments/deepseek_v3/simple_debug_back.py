#!/usr/bin/env python3
"""
Simple debug script to isolate CUTLASS backward pass issues.
Run this to identify exactly where the numerical problems occur.
"""

import numpy as np
import torch
import torch.nn as nn


def test_single_expert_operations():
    """Test operations on a single expert to isolate issues"""
    print("🔍 Testing Single Expert Operations")
    print("=" * 50)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Simple test case
    M, N, K = 32, 64, 128  # Small sizes for debugging

    # Create test data
    X = torch.randn(M, K, dtype=dtype, device=device, requires_grad=True)  # Input
    W = torch.randn(N, K, dtype=dtype, device=device, requires_grad=True)  # Weight

    print(f"Input X: {X.shape}")
    print(f"Weight W: {W.shape}")

    # Forward pass: Y = X @ W^T
    Y = torch.mm(X, W.t())  # [M, N]
    print(f"Output Y: {Y.shape}")

    # Create upstream gradient
    dY = torch.randn_like(Y)
    print(f"Upstream grad dY: {dY.shape}")

    # Compute reference gradients
    print("\n📊 Reference PyTorch Gradients:")
    Y_ref = torch.mm(X, W.t())
    Y_ref.backward(dY, retain_graph=True)

    dX_ref = X.grad.clone()
    dW_ref = W.grad.clone()

    print(f"dX_ref norm: {dX_ref.norm().item():.4f}")
    print(f"dW_ref norm: {dW_ref.norm().item():.4f}")

    # Clear gradients
    X.grad = None
    W.grad = None

    # Manual gradient computation
    print("\n🧮 Manual Gradient Computation:")
    dX_manual = torch.mm(dY, W)  # [M, N] @ [N, K] = [M, K]
    dW_manual = torch.mm(dY.t(), X)  # [N, M] @ [M, K] = [N, K]

    print(f"dX_manual norm: {dX_manual.norm().item():.4f}")
    print(f"dW_manual norm: {dW_manual.norm().item():.4f}")

    # Check manual vs reference
    dX_diff = torch.abs(dX_manual - dX_ref).max().item()
    dW_diff = torch.abs(dW_manual - dW_ref).max().item()

    print(f"\n✅ Manual vs Reference:")
    print(f"dX difference: {dX_diff:.2e}")
    print(f"dW difference: {dW_diff:.2e}")

    if dX_diff < 1e-3 and dW_diff < 1e-3:
        print("✅ Manual gradients match reference!")
    else:
        print("❌ Manual gradients don't match!")
        return False

    return dX_manual, dW_manual, X, W, dY


def test_cutlass_simple_operations():
    """Test CUTLASS operations step by step"""
    print("\n🔍 Testing CUTLASS Simple Operations")
    print("=" * 50)

    try:
        from cutlass_backwards_debug import (
            CUTLASSBackwardGroupGemmDebug,
            CUTLASSGroupedGemmStrategyDebug,
        )
    except ImportError:
        print("❌ Cannot import debug modules")
        return False

    # Get reference data from single expert test
    dX_ref, dW_ref, X, W, dY = test_single_expert_operations()

    device = X.device
    dtype = X.dtype

    # Create debug strategy
    strategy = CUTLASSGroupedGemmStrategyDebug(
        debug_mode=True,
        backward_method="approach_3",  # Single expert debugging
        use_2cta_instrs=False,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
    )

    print(f"\n🔧 Testing CUTLASS Single Expert Operations:")

    # Test single expert CUTLASS operations
    try:
        dX_cutlass, dW_cutlass = (
            CUTLASSBackwardGroupGemmDebug._test_single_expert_cutlass(
                dY, X, W, strategy
            )
        )

        # Compare with reference
        dX_cutlass_diff = torch.abs(dX_cutlass - dX_ref).max().item()
        dW_cutlass_diff = torch.abs(dW_cutlass - dW_ref).max().item()

        print(f"\n📊 CUTLASS vs Reference:")
        print(f"dX difference: {dX_cutlass_diff:.2e}")
        print(f"dW difference: {dW_cutlass_diff:.2e}")

        if dX_cutlass_diff < 1e-2 and dW_cutlass_diff < 1e-2:
            print("✅ CUTLASS single expert operations working!")
            return True
        else:
            print("❌ CUTLASS single expert operations have issues")
            return False

    except Exception as e:
        print(f"❌ CUTLASS single expert test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_grouped_operations():
    """Test full grouped operations"""
    print("\n🔍 Testing Grouped Operations")
    print("=" * 50)

    try:
        from cutlass_backwards_debug import (
            CUTLASSGroupedGemmStrategyDebug,
            CUTLASSGroupedLinearDebug,
        )
    except ImportError:
        print("❌ Cannot import debug modules")
        return False

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Test parameters
    num_experts = 4
    in_features = 256
    out_features = 512
    total_tokens = 128

    # Test different approaches
    approaches = ["approach_1", "approach_2", "approach_3"]

    for approach in approaches:
        print(f"\n🔧 Testing grouped operations with {approach}")

        try:
            # Create strategy
            strategy = CUTLASSGroupedGemmStrategyDebug(
                debug_mode=True,
                backward_method=approach,
                use_2cta_instrs=False,
                mma_tiler_mn=(128, 128),
                cluster_shape_mn=(1, 1),
            )

            # Create test data
            input_tokens = torch.randn(
                total_tokens,
                in_features,
                dtype=dtype,
                device=device,
                requires_grad=True,
            )
            expert_assignments = torch.randint(
                0, num_experts, (total_tokens,), device=device
            )

            # Create layer
            layer = CUTLASSGroupedLinearDebug(
                num_experts, in_features, out_features, strategy, dtype=dtype
            )
            layer = layer.to(device)

            # Forward pass
            output = layer(input_tokens, expert_assignments)

            # Backward pass
            loss = output.sum()
            loss.backward()

            print(f"✅ {approach} completed successfully")

            # Check if gradients exist and are reasonable
            if input_tokens.grad is not None:
                input_grad_norm = input_tokens.grad.norm().item()
                print(f"   Input grad norm: {input_grad_norm:.4f}")
            else:
                print("   ❌ No input gradient!")

            if layer.weight.grad is not None:
                weight_grad_norm = layer.weight.grad.norm().item()
                print(f"   Weight grad norm: {weight_grad_norm:.4f}")
            else:
                print("   ❌ No weight gradient!")

        except Exception as e:
            print(f"❌ {approach} failed: {e}")
            import traceback

            traceback.print_exc()


def debug_matrix_operations():
    """Debug the core matrix operations used in backward pass"""
    print("\n🔍 Debugging Core Matrix Operations")
    print("=" * 50)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Test data
    M, N, K = 16, 32, 64

    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(N, K, dtype=dtype, device=device)
    dC = torch.randn(M, N, dtype=dtype, device=device)

    print(f"A: {A.shape}, B: {B.shape}, dC: {dC.shape}")

    # Test different formulations of the same operations
    print("\n📊 Testing Input Gradient Formulations:")

    # Method 1: Direct computation dA = dC @ B
    dA_direct = torch.mm(dC, B)  # [M, N] @ [N, K] = [M, K]
    print(f"Method 1 (direct): {dA_direct.shape}, norm: {dA_direct.norm().item():.4f}")

    # Method 2: Transpose formulation dA^T = B^T @ dC^T
    dA_transpose = torch.mm(B.t(), dC.t()).t()  # [K, N] @ [N, M] = [K, M] -> [M, K]
    print(
        f"Method 2 (transpose): {dA_transpose.shape}, norm: {dA_transpose.norm().item():.4f}"
    )

    # Check if they're the same
    dA_diff = torch.abs(dA_direct - dA_transpose).max().item()
    print(f"Difference: {dA_diff:.2e}")

    print("\n📊 Testing Weight Gradient Formulations:")

    # Method 1: Direct computation dB = dC^T @ A
    dB_direct = torch.mm(dC.t(), A)  # [N, M] @ [M, K] = [N, K]
    print(f"Method 1 (direct): {dB_direct.shape}, norm: {dB_direct.norm().item():.4f}")

    # Method 2: Using transpose dB^T = A^T @ dC, then transpose
    dB_transpose = torch.mm(A.t(), dC).t()  # [K, M] @ [M, N] = [K, N] -> [N, K]
    print(
        f"Method 2 (transpose): {dB_transpose.shape}, norm: {dB_transpose.norm().item():.4f}"
    )

    # Check if they're the same
    dB_diff = torch.abs(dB_direct - dB_transpose).max().item()
    print(f"Difference: {dB_diff:.2e}")

    if dA_diff < 1e-5 and dB_diff < 1e-5:
        print("✅ All formulations are mathematically equivalent!")
        return True
    else:
        print("❌ Formulations don't match - there's a mathematical error!")
        return False


def main():
    """Main debug sequence"""
    print("🧪 CUTLASS Backward Pass Debug Suite")
    print("=" * 60)

    # Step 4: Test grouped operations
    print("\n" + "=" * 60)
    test_grouped_operations()

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")

    # Step 1: Test mathematical formulations
    print("\n" + "=" * 60)
    if not debug_matrix_operations():
        print("❌ Mathematical formulations failed - fix before continuing")
        return

    # Step 2: Test single expert operations
    print("\n" + "=" * 60)
    single_expert_result = test_single_expert_operations()
    if single_expert_result is False:
        print("❌ Single expert operations failed")
        return

    # Step 3: Test CUTLASS single expert operations
    print("\n" + "=" * 60)
    if not test_cutlass_simple_operations():
        print("❌ CUTLASS single expert operations failed")
        return

    # Step 4: Test grouped operations
    print("\n" + "=" * 60)
    test_grouped_operations()

    print("\n" + "=" * 60)
    print("🎯 Debug sequence completed!")
    print("\nNext steps based on results:")
    print("1. If single expert works but grouped fails -> issue in batching/metadata")
    print("2. If CUTLASS single expert fails -> issue in CUTLASS setup")
    print("3. If mathematical formulations fail -> fundamental math error")


if __name__ == "__main__":
    main()
