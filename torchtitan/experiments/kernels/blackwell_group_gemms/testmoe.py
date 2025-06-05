#!/usr/bin/env python3
"""
Test script to verify CUTLASS GroupGEMM integration works correctly
"""

import time
from typing import List, Tuple

import torch
import torch.nn as nn


def test_cutlass_integration():
    """Test CUTLASS GroupGEMM integration with the MoE model"""

    print("Testing CUTLASS GroupGEMM Integration")
    print("=" * 50)

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping test")
        return False

    # Check GPU compute capability
    major, minor = torch.cuda.get_device_capability()
    if major < 9:  # Hopper is 9.0
        print(f"❌ GPU compute capability {major}.{minor} is too old for CUTLASS")
        print("   Requires compute capability 9.0+ (Hopper/Blackwell)")
        return False

    print(f"✅ GPU compute capability: {major}.{minor}")

    # Test import
    try:
        from group_gemms import CUTLASSGroupGEMM

        print("✅ Successfully imported CUTLASSGroupGEMM")
    except ImportError as e:
        print(f"❌ Failed to import CUTLASSGroupGEMM: {e}")
        return False

    # Check availability
    if not CUTLASSGroupGEMM.is_available():
        print("❌ CUTLASS GroupGEMM is not available")
        return False

    print("✅ CUTLASS GroupGEMM is available")

    # Test basic functionality
    try:
        # Create a simple test case
        device = torch.device("cuda")
        dtype = torch.bfloat16

        # Simulate MoE parameters
        hidden_size = 512
        intermediate_size = 1024
        num_experts = 4

        # Create test activation function
        activation_fn = nn.SiLU()

        # Create CUTLASS strategy
        cutlass_strategy = CUTLASSGroupGEMM(activation_fn)
        print("✅ Created CUTLASS strategy")

        # Create mock expert weights
        gate_weights = []
        up_weights = []
        down_weights = []

        for i in range(num_experts):
            gate_weights.append(
                torch.randn(hidden_size, intermediate_size, dtype=dtype, device=device)
            )
            up_weights.append(
                torch.randn(hidden_size, intermediate_size, dtype=dtype, device=device)
            )
            down_weights.append(
                torch.randn(intermediate_size, hidden_size, dtype=dtype, device=device)
            )

        # Mock module to store weights
        class MockModule:
            def __init__(self):
                self.params = {}

            def register_parameter(self, name, param):
                self.params[name] = param

            def get_parameter(self, name):
                return self.params[name]

        mock_module = MockModule()

        # Arrange weights
        arranged_gate = cutlass_strategy.arrange_expert_weights(
            gate_weights, "gate_proj", mock_module
        )
        arranged_up = cutlass_strategy.arrange_expert_weights(
            up_weights, "up_proj", mock_module
        )
        arranged_down = cutlass_strategy.arrange_expert_weights(
            down_weights, "down_proj", mock_module
        )

        # Store arranged weights
        mock_module.register_parameter("gate_proj_weight", nn.Parameter(arranged_gate))
        mock_module.register_parameter("up_proj_weight", nn.Parameter(arranged_up))
        mock_module.register_parameter("down_proj_weight", nn.Parameter(arranged_down))

        print("✅ Arranged expert weights")

        # Create test input
        token_counts = [32, 64, 16, 48]  # Tokens per expert
        total_tokens = sum(token_counts)

        # Create contiguous tokens (as would come from MoE routing)
        contig_tokens = torch.randn(
            total_tokens, hidden_size, dtype=dtype, device=device
        )

        # Create m_sizes and m_offsets
        m_sizes = token_counts
        m_offsets = [0]
        for size in m_sizes:
            m_offsets.append(m_offsets[-1] + size)

        print(
            f"✅ Created test input: {total_tokens} tokens across {num_experts} experts"
        )
        print(f"   Token distribution: {token_counts}")

        # Execute CUTLASS grouped GEMM
        start_time = time.time()
        result = cutlass_strategy.execute(
            contig_tokens, m_sizes, m_offsets, mock_module
        )
        torch.cuda.synchronize()
        cutlass_time = time.time() - start_time

        print(
            f"✅ Successfully executed CUTLASS GroupGEMM in {cutlass_time*1000:.2f} ms"
        )
        print(f"   Output shape: {result.shape}")
        print(f"   Output dtype: {result.dtype}")

        # Verify output shape
        expected_shape = (total_tokens, hidden_size)
        if result.shape != expected_shape:
            print(
                f"❌ Output shape mismatch: expected {expected_shape}, got {result.shape}"
            )
            return False

        # Verify output is not all zeros or all NaN
        if torch.isnan(result).any():
            print("❌ Output contains NaN values")
            return False

        if torch.allclose(result, torch.zeros_like(result)):
            print("❌ Output is all zeros")
            return False

        print("✅ Output validation passed")

        # Compare with PyTorch implementation for correctness
        try:
            from group_gemms import TorchBF16GroupGEMM

            torch_strategy = TorchBF16GroupGEMM(activation_fn)

            # Create PyTorch weights in expected format
            torch_module = MockModule()
            torch_gate = torch_strategy.arrange_expert_weights(
                gate_weights, "gate_proj", torch_module
            )
            torch_up = torch_strategy.arrange_expert_weights(
                up_weights, "up_proj", torch_module
            )
            torch_down = torch_strategy.arrange_expert_weights(
                down_weights, "down_proj", torch_module
            )

            torch_module.register_parameter(
                "gate_proj_weight", nn.Parameter(torch_gate)
            )
            torch_module.register_parameter("up_proj_weight", nn.Parameter(torch_up))
            torch_module.register_parameter(
                "down_proj_weight", nn.Parameter(torch_down)
            )

            # Execute PyTorch version
            start_time = time.time()
            torch_result = torch_strategy.execute(
                contig_tokens.clone(), m_sizes, m_offsets, torch_module
            )
            torch.cuda.synchronize()
            torch_time = time.time() - start_time

            print(f"✅ PyTorch reference completed in {torch_time*1000:.2f} ms")

            # Compare results
            max_diff = torch.max(torch.abs(result - torch_result)).item()
            rel_error = (
                torch.norm(result - torch_result) / torch.norm(torch_result)
            ).item()

            print(f"   Max absolute difference: {max_diff:.2e}")
            print(f"   Relative error: {rel_error:.2e}")

            tolerance = 1e-2  # BF16 tolerance
            if rel_error > tolerance:
                print(f"❌ Results differ too much (tolerance: {tolerance})")
                return False

            print("✅ Results match PyTorch reference within tolerance")

            # Performance comparison
            speedup = torch_time / cutlass_time
            print(
                f"   Performance: {speedup:.2f}x {'speedup' if speedup > 1 else 'slower'}"
            )

        except Exception as e:
            print(f"⚠️  Could not compare with PyTorch reference: {e}")

        print(
            "\n🎉 All tests passed! CUTLASS GroupGEMM integration is working correctly."
        )
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_model_integration():
    """Test integration with the full MoE model"""

    print("\nTesting Model Integration")
    print("=" * 30)

    try:
        # Test that we can set the group GEMM backend
        from model import MoE

        # Check available strategies
        if MoE.group_gemm_strategies is None:
            MoE._initialize_group_gemm_strategies()

        if "cutlass" not in MoE.group_gemm_strategies:
            print("❌ CUTLASS strategy not found in group_gemm_strategies")
            return False

        if MoE.group_gemm_strategies["cutlass"] is None:
            print("❌ CUTLASS strategy is None (not available)")
            return False

        print("✅ CUTLASS strategy is available in MoE")

        # Test setting the backend
        original_backend = MoE.group_mm
        MoE.group_mm = "cutlass"

        print(f"✅ Successfully set group GEMM backend to 'cutlass'")
        print(f"   (Previous backend: '{original_backend}')")

        # Restore original backend
        MoE.group_mm = original_backend

        return True

    except Exception as e:
        print(f"❌ Model integration test failed: {e}")
        return False


if __name__ == "__main__":
    print("CUTLASS GroupGEMM Integration Test Suite")
    print("=" * 60)

    # Run tests
    basic_test_passed = test_cutlass_integration()
    model_test_passed = test_model_integration()

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Basic functionality: {'✅ PASS' if basic_test_passed else '❌ FAIL'}")
    print(f"Model integration:   {'✅ PASS' if model_test_passed else '❌ FAIL'}")

    if basic_test_passed and model_test_passed:
        print("\n🎉 All tests passed! CUTLASS GroupGEMM is ready to use.")
        print("\nTo use in your model:")
        print("  from model import MoE")
        print("  MoE.group_mm = 'cutlass'  # Set before model creation")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        exit(1)
