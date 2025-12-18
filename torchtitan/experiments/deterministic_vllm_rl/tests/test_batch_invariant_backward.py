# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test batch_invariant_backward module to ensure it works correctly.
"""

import torch

from torchtitan.experiments.deterministic_vllm_rl.batch_invariant_backward import (
    disable_batch_invariant_backward_mode,
    enable_batch_invariant_backward_mode,
    linear_batch_invariant_backward,
    mm_batch_invariant_backward,
)


def test_mm_backward():
    """Test matrix multiplication with backward."""
    print("\n" + "=" * 80)
    print("Testing mm_batch_invariant_backward...")
    print("=" * 80)

    # Enable mode
    from vllm.model_executor.layers.batch_invariant import disable_batch_invariant_mode

    disable_batch_invariant_mode()
    enable_batch_invariant_backward_mode()

    # Create test tensors
    a = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    # Forward pass
    c = mm_batch_invariant_backward(a, b)
    print("Forward pass: a @ b = c")
    print(f"  a shape: {a.shape}, b shape: {b.shape}, c shape: {c.shape}")

    # Backward pass
    loss = c.sum()
    loss.backward()

    print("Backward pass successful!")
    print(f"  grad_a shape: {a.grad.shape if a.grad is not None else None}")
    print(f"  grad_b shape: {b.grad.shape if b.grad is not None else None}")

    assert a.grad is not None, "grad_a should not be None"
    assert b.grad is not None, "grad_b should not be None"
    print("✅ mm_backward test passed!")

    disable_batch_invariant_backward_mode()


def test_linear_backward():
    """Test linear layer with backward."""
    print("\n" + "=" * 80)
    print("Testing linear_batch_invariant_backward...")
    print("=" * 80)

    # Enable mode
    from vllm.model_executor.layers.batch_invariant import disable_batch_invariant_mode

    disable_batch_invariant_mode()
    enable_batch_invariant_backward_mode()

    # Create test tensors (3D input for realistic case)
    input = torch.randn(
        2, 10, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    weight = torch.randn(
        128, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    bias = torch.randn(128, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    # Forward pass
    output = linear_batch_invariant_backward(input, weight, bias)
    print("Forward pass: linear(input, weight, bias) = output")
    print(f"  input shape: {input.shape}")
    print(f"  weight shape: {weight.shape}")
    print(f"  output shape: {output.shape}")

    # Backward pass
    loss = output.sum()
    loss.backward()

    print("Backward pass successful!")
    print(f"  grad_input shape: {input.grad.shape if input.grad is not None else None}")
    print(
        f"  grad_weight shape: {weight.grad.shape if weight.grad is not None else None}"
    )
    print(f"  grad_bias shape: {bias.grad.shape if bias.grad is not None else None}")

    assert input.grad is not None, "grad_input should not be None"
    assert weight.grad is not None, "grad_weight should not be None"
    assert bias.grad is not None, "grad_bias should not be None"
    print("✅ linear_backward test passed!")

    disable_batch_invariant_backward_mode()


def test_deterministic_forward():
    """Test that forward passes are deterministic."""
    print("\n" + "=" * 80)
    print("Testing deterministic forward passes...")
    print("=" * 80)

    # Enable mode
    from vllm.model_executor.layers.batch_invariant import disable_batch_invariant_mode

    disable_batch_invariant_mode()
    enable_batch_invariant_backward_mode()

    # Create test tensors
    a = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

    # Run forward pass twice
    c1 = mm_batch_invariant_backward(a, b)
    c2 = mm_batch_invariant_backward(a, b)

    # Check if results are identical
    diff = (c1 - c2).abs().max().item()
    print(f"Forward pass 1 result: {c1[0, :5]}")
    print(f"Forward pass 2 result: {c2[0, :5]}")
    print(f"Max absolute difference: {diff}")

    assert diff == 0.0, f"Forward passes should be deterministic, but diff={diff}"
    print("✅ Deterministic forward test passed!")

    disable_batch_invariant_backward_mode()


def main():
    """Run all tests."""
    print("=" * 80)
    print("Testing batch_invariant_backward module")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping tests")
        return

    try:
        test_mm_backward()
        test_linear_backward()
        test_deterministic_forward()

        print("\n" + "=" * 80)
        print("✅ All tests passed!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
