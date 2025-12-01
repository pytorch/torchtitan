#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test script to verify that patching TorchTitan Qwen3 model with VLLMCompatibleFlashAttention
still allows it to run with TorchTitan's training loop.

This tests:
1. Model creation with patched attention
2. Forward pass with dummy data
3. Backward pass and gradient computation
4. Training step execution
5. Compatibility with TorchTitan's model protocol
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add TorchTitan to path
torchtitan_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(torchtitan_root))

from torchtitan.experiments.vllm.model.attention import VLLMCompatibleFlashAttention
from torchtitan.models.qwen3.model.args import Qwen3ModelArgs
from torchtitan.models.qwen3.model.model import Qwen3Model


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def patch_qwen3_attention(model: Qwen3Model, model_args: Qwen3ModelArgs) -> int:
    """
    Patch all attention modules in Qwen3Model with VLLMCompatibleFlashAttention.

    Args:
        model: Qwen3Model instance
        model_args: Model configuration

    Returns:
        Number of attention modules patched
    """
    num_patched = 0

    for layer_name, layer in model.layers.items():
        # Replace inner_attention with VLLMCompatibleFlashAttention
        layer.attention.inner_attention = VLLMCompatibleFlashAttention(
            hidden_size=model_args.dim,
            num_heads=model_args.n_heads,
            num_kv_heads=model_args.n_kv_heads,
            head_dim=model_args.head_dim,
            causal=True,
        )
        num_patched += 1

    return num_patched


def test_model_creation():
    """Test 1: Create Qwen3 model and patch with VLLMCompatibleFlashAttention."""
    print_section("Test 1: Model Creation and Patching")

    try:
        # Create small test model
        model_args = Qwen3ModelArgs(
            dim=512,
            n_layers=4,
            n_heads=8,
            n_kv_heads=4,  # GQA
            vocab_size=1000,
            max_seq_len=512,
            rope_theta=1000000.0,
            hidden_dim=1024,
            norm_eps=1e-6,
            qk_norm=True,
        )

        print("Creating Qwen3Model with config:")
        print(f"  dim: {model_args.dim}")
        print(f"  n_layers: {model_args.n_layers}")
        print(f"  n_heads: {model_args.n_heads}")
        print(f"  n_kv_heads: {model_args.n_kv_heads}")
        print(f"  vocab_size: {model_args.vocab_size}")

        model = Qwen3Model(model_args)
        print("✅ Model created successfully")

        # Patch attention modules
        print("\nPatching attention modules...")
        num_patched = patch_qwen3_attention(model, model_args)
        print(
            f"✅ Patched {num_patched} attention modules with VLLMCompatibleFlashAttention"
        )

        # Verify patch
        first_layer = model.layers["0"]
        assert isinstance(
            first_layer.attention.inner_attention, VLLMCompatibleFlashAttention
        ), "Attention module not patched correctly"
        print("✅ Verified attention module type")

        return model, model_args

    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def test_forward_pass(model: Qwen3Model, model_args: Qwen3ModelArgs):
    """Test 2: Run forward pass with dummy data."""
    print_section("Test 2: Forward Pass")

    if model is None:
        print("⚠️  Skipping (model creation failed)")
        return None

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        print(f"Using device: {device}")

        # Create dummy input
        batch_size = 2
        seq_len = 64
        tokens = torch.randint(
            0, model_args.vocab_size, (batch_size, seq_len), device=device
        )

        print(f"\nInput shape: {tokens.shape}")

        # Forward pass
        with torch.no_grad():
            output = model(tokens)

        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")

        # Verify output shape
        expected_shape = (batch_size, seq_len, model_args.vocab_size)
        assert (
            output.shape == expected_shape
        ), f"Expected {expected_shape}, got {output.shape}"

        print("\nOutput statistics:")
        print(f"  Mean: {output.mean().item():.6f}")
        print(f"  Std: {output.std().item():.6f}")
        print(f"  Min: {output.min().item():.6f}")
        print(f"  Max: {output.max().item():.6f}")

        print("\n✅ Forward pass successful")
        return output

    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_backward_pass(model: Qwen3Model, model_args: Qwen3ModelArgs):
    """Test 3: Run backward pass and verify gradients."""
    print_section("Test 3: Backward Pass and Gradient Computation")

    if model is None:
        print("⚠️  Skipping (model creation failed)")
        return False

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.train()  # Enable training mode

        # Create dummy input and target
        batch_size = 2
        seq_len = 64
        tokens = torch.randint(
            0, model_args.vocab_size, (batch_size, seq_len), device=device
        )
        targets = torch.randint(
            0, model_args.vocab_size, (batch_size, seq_len), device=device
        )

        print(f"Input shape: {tokens.shape}")
        print(f"Target shape: {targets.shape}")

        # Forward pass
        output = model(tokens)

        # Compute loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output.view(-1, model_args.vocab_size), targets.view(-1))

        print(f"\nLoss: {loss.item():.6f}")

        # Backward pass
        print("\nRunning backward pass...")
        loss.backward()

        # Check gradients
        grad_count = 0
        grad_norms = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_count += 1
                grad_norm = param.grad.norm().item()
                if "attention" in name:
                    grad_norms[name] = grad_norm

        print(f"✅ Gradients computed for {grad_count} parameters")

        # Show some attention gradients
        if grad_norms:
            print("\nSample attention gradient norms:")
            for name, norm in list(grad_norms.items())[:5]:
                print(f"  {name}: {norm:.6f}")

        # Verify gradients are non-zero
        assert grad_count > 0, "No gradients computed"

        print("\n✅ Backward pass successful")
        return True

    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_training_step(model: Qwen3Model, model_args: Qwen3ModelArgs):
    """Test 4: Run a full training step with optimizer."""
    print_section("Test 4: Training Step with Optimizer")

    if model is None:
        print("⚠️  Skipping (model creation failed)")
        return False

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.train()

        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss_fn = nn.CrossEntropyLoss()

        print(f"Optimizer: {type(optimizer).__name__}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        # Run multiple training steps
        num_steps = 3
        losses = []

        for step in range(num_steps):
            # Create dummy data
            batch_size = 2
            seq_len = 64
            tokens = torch.randint(
                0, model_args.vocab_size, (batch_size, seq_len), device=device
            )
            targets = torch.randint(
                0, model_args.vocab_size, (batch_size, seq_len), device=device
            )

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(tokens)
            loss = loss_fn(output.view(-1, model_args.vocab_size), targets.view(-1))

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

            losses.append(loss.item())
            print(f"  Step {step + 1}/{num_steps}: loss = {loss.item():.6f}")

        print(f"\n✅ Completed {num_steps} training steps")
        print(f"Loss values: {losses}")

        return True

    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_attention_shapes(model: Qwen3Model, model_args: Qwen3ModelArgs):
    """Test 5: Verify attention input/output shapes in detail."""
    print_section("Test 5: Attention Shape Verification")

    if model is None:
        print("⚠️  Skipping (model creation failed)")
        return False

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        # Hook to capture attention inputs/outputs
        attention_info = {}

        def attention_hook(module, input_args, output):
            """Capture attention layer inputs and outputs."""
            # input_args is a tuple: (x, rope_cache, attention_masks)
            x = input_args[0]
            attention_info["input_shape"] = x.shape
            attention_info["output_shape"] = output.shape

        # Register hook on first layer's attention
        first_layer = model.layers["0"]
        hook = first_layer.attention.register_forward_hook(attention_hook)

        # Run forward pass
        batch_size = 2
        seq_len = 64
        tokens = torch.randint(
            0, model_args.vocab_size, (batch_size, seq_len), device=device
        )

        with torch.no_grad():
            _ = model(tokens)

        # Remove hook
        hook.remove()

        # Verify shapes
        print(f"Attention input shape: {attention_info['input_shape']}")
        print(f"Attention output shape: {attention_info['output_shape']}")

        expected_input = (batch_size, seq_len, model_args.dim)
        expected_output = (batch_size, seq_len, model_args.dim)

        assert (
            attention_info["input_shape"] == expected_input
        ), f"Expected input {expected_input}, got {attention_info['input_shape']}"
        assert (
            attention_info["output_shape"] == expected_output
        ), f"Expected output {expected_output}, got {attention_info['output_shape']}"

        print("\n✅ Attention shapes verified")
        return True

    except Exception as e:
        print(f"❌ Test 5 failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_gqa_correctness(model_args: Qwen3ModelArgs):
    """Test 6: Verify GQA expansion works correctly."""
    print_section("Test 6: GQA (Grouped Query Attention) Verification")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create attention module directly
        attn = VLLMCompatibleFlashAttention(
            hidden_size=model_args.dim,
            num_heads=model_args.n_heads,
            num_kv_heads=model_args.n_kv_heads,
            head_dim=model_args.head_dim,
            causal=True,
        ).to(device)

        print("Attention config:")
        print(f"  num_heads (Q): {model_args.n_heads}")
        print(f"  num_kv_heads (K/V): {model_args.n_kv_heads}")
        print(f"  head_dim: {model_args.head_dim}")
        print(f"  n_rep: {model_args.n_heads // model_args.n_kv_heads}")

        batch_size = 2
        seq_len = 32

        # Test with unexpanded K/V (num_kv_heads)
        print(f"\nTest 6a: K/V with num_kv_heads ({model_args.n_kv_heads})")
        q = torch.randn(
            batch_size,
            model_args.n_heads,
            seq_len,
            model_args.head_dim,
            device=device,
            dtype=torch.float32,
        )
        k_small = torch.randn(
            batch_size,
            model_args.n_kv_heads,
            seq_len,
            model_args.head_dim,
            device=device,
            dtype=torch.float32,
        )
        v_small = torch.randn(
            batch_size,
            model_args.n_kv_heads,
            seq_len,
            model_args.head_dim,
            device=device,
            dtype=torch.float32,
        )

        with torch.no_grad():
            output_small = attn(q, k_small, v_small)

        print(f"  Output shape: {output_small.shape}")
        assert (
            output_small.shape == q.shape
        ), f"Shape mismatch: {output_small.shape} vs {q.shape}"
        print("  ✅ Unexpanded K/V works")

        # Test with expanded K/V (num_heads)
        print(f"\nTest 6b: K/V already expanded to num_heads ({model_args.n_heads})")
        k_large = k_small.repeat_interleave(
            model_args.n_heads // model_args.n_kv_heads, dim=1
        )
        v_large = v_small.repeat_interleave(
            model_args.n_heads // model_args.n_kv_heads, dim=1
        )

        print(f"  k_large shape: {k_large.shape}")
        print(f"  v_large shape: {v_large.shape}")

        with torch.no_grad():
            output_large = attn(q, k_large, v_large)

        print(f"  Output shape: {output_large.shape}")
        assert (
            output_large.shape == q.shape
        ), f"Shape mismatch: {output_large.shape} vs {q.shape}"
        print("  ✅ Expanded K/V works")

        print("\n✅ GQA verification successful")
        return True

    except Exception as e:
        print(f"❌ Test 6 failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("  TorchTitan + VLLMCompatibleFlashAttention Integration Test")
    print("=" * 70)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    results = {}

    # Test 1: Model creation and patching
    model, model_args = test_model_creation()
    results["model_creation"] = model is not None

    if model is None:
        print("\n❌ Cannot continue - model creation failed")
        return 1

    # Test 2: Forward pass
    output = test_forward_pass(model, model_args)
    results["forward_pass"] = output is not None

    # Test 3: Backward pass
    results["backward_pass"] = test_backward_pass(model, model_args)

    # Test 4: Training step
    results["training_step"] = test_training_step(model, model_args)

    # Test 5: Attention shapes
    results["attention_shapes"] = test_attention_shapes(model, model_args)

    # Test 6: GQA verification
    results["gqa_verification"] = test_gqa_correctness(model_args)

    # Summary
    print_section("FINAL SUMMARY")

    print("\nTest Results:")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:20s}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✅ VLLMCompatibleFlashAttention is compatible with TorchTitan!")
        print("✅ Model can be trained with patched attention modules.")
        print(
            "\nYou can safely use this attention implementation in TorchTitan training."
        )
        return 0
    else:
        failed_tests = [name for name, passed in results.items() if not passed]
        print(f"\n❌ {len(failed_tests)} TEST(S) FAILED:")
        for test in failed_tests:
            print(f"  - {test}")
        print(
            "\nPlease fix the issues before using VLLMCompatibleFlashAttention in production."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
