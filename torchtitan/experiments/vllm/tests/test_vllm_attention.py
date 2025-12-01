#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple test script for vLLM's Attention layer.
Tests the high-level Attention module with KV cache management.
"""

import torch
from vllm.attention import Attention
from vllm.config import CacheConfig


def test_vllm_attention_basic():
    """Test vLLM Attention layer with basic inputs (no KV cache)."""
    print("=" * 70)
    print("Test 1: Basic Attention (No KV Cache)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model parameters
    num_heads = 16
    num_kv_heads = 8  # GQA
    head_dim = 128
    hidden_size = num_heads * head_dim

    # Create minimal vLLM config
    # Note: In production, this comes from get_current_vllm_config()
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        swap_space=4,
        cache_dtype="auto",
    )

    # Create Attention layer
    print("\nCreating Attention layer:")
    print(f"  num_heads: {num_heads}")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  scale: {1.0 / (head_dim ** 0.5)}")

    attn = Attention(
        num_heads=num_heads,
        head_size=head_dim,
        scale=1.0 / (head_dim**0.5),
        num_kv_heads=num_kv_heads,
        cache_config=cache_config,
        quant_config=None,
        prefix="layers.0",
    )
    attn = attn.to(device)
    attn.eval()

    # Create dummy inputs
    batch_size = 2
    seq_len = 128
    total_tokens = batch_size * seq_len

    # Format: [total_tokens, num_heads, head_dim]
    q = torch.randn(
        total_tokens, num_heads, head_dim, dtype=torch.float16, device=device
    )
    k = torch.randn(
        total_tokens, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v = torch.randn(
        total_tokens, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

    print("\nInput shapes:")
    print(f"  q: {q.shape}")
    print(f"  k: {k.shape}")
    print(f"  v: {v.shape}")

    # Forward pass
    try:
        with torch.no_grad():
            output = attn(q, k, v)

        print(f"\nOutput shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        print(f"Output device: {output.device}")
        print("\nOutput statistics:")
        print(f"  Mean: {output.mean().item():.6f}")
        print(f"  Std: {output.std().item():.6f}")
        print(f"  Min: {output.min().item():.6f}")
        print(f"  Max: {output.max().item():.6f}")

        print("\n✅ Test 1 passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test 1 failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_vllm_attention_gqa_expansion():
    """Test that GQA expansion works correctly."""
    print("\n" + "=" * 70)
    print("Test 2: GQA Expansion Test")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_heads = 16
    num_kv_heads = 8
    head_dim = 128

    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        swap_space=4,
        cache_dtype="auto",
    )

    attn = Attention(
        num_heads=num_heads,
        head_size=head_dim,
        scale=1.0 / (head_dim**0.5),
        num_kv_heads=num_kv_heads,
        cache_config=cache_config,
        quant_config=None,
        prefix="layers.1",
    )
    attn = attn.to(device)
    attn.eval()

    # Test with both unexpanded and expanded k/v
    total_tokens = 64

    q = torch.randn(
        total_tokens, num_heads, head_dim, device=device, dtype=torch.float16
    )

    print(f"\nTest 2a: K/V with num_kv_heads ({num_kv_heads})")
    k_small = torch.randn(
        total_tokens, num_kv_heads, head_dim, device=device, dtype=torch.float16
    )
    v_small = torch.randn(
        total_tokens, num_kv_heads, head_dim, device=device, dtype=torch.float16
    )

    try:
        with torch.no_grad():
            output_small = attn(q, k_small, v_small)
        print(f"  Output shape: {output_small.shape}")
        print("  ✅ GQA with num_kv_heads works!")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

    print(f"\nTest 2b: K/V already expanded to num_heads ({num_heads})")
    # Simulate what TorchTitan does - expand k/v to num_heads
    k_large = k_small.repeat_interleave(num_heads // num_kv_heads, dim=1)
    v_large = v_small.repeat_interleave(num_heads // num_kv_heads, dim=1)
    print(f"  k_large shape: {k_large.shape}")
    print(f"  v_large shape: {v_large.shape}")

    try:
        with torch.no_grad():
            output_large = attn(q, k_large, v_large)
        print(f"  Output shape: {output_large.shape}")
        print("  ✅ Already-expanded K/V works!")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

    print("\n✅ Test 2 passed!")
    return True


def test_vllm_attention_shapes():
    """Test various input shapes."""
    print("\n" + "=" * 70)
    print("Test 3: Various Input Shapes")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_heads = 8
    num_kv_heads = 8  # MHA
    head_dim = 64

    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        swap_space=4,
        cache_dtype="auto",
    )

    attn = Attention(
        num_heads=num_heads,
        head_size=head_dim,
        scale=1.0 / (head_dim**0.5),
        num_kv_heads=num_kv_heads,
        cache_config=cache_config,
        quant_config=None,
        prefix="layers.2",
    )
    attn = attn.to(device)
    attn.eval()

    test_cases = [
        (1, "Single token"),
        (32, "Small batch"),
        (256, "Medium batch"),
        (1024, "Large batch"),
    ]

    for total_tokens, description in test_cases:
        print(
            f"\nTest 3.{test_cases.index((total_tokens, description)) + 1}: {description} ({total_tokens} tokens)"
        )
        q = torch.randn(
            total_tokens, num_heads, head_dim, device=device, dtype=torch.float16
        )
        k = torch.randn(
            total_tokens, num_kv_heads, head_dim, device=device, dtype=torch.float16
        )
        v = torch.randn(
            total_tokens, num_kv_heads, head_dim, device=device, dtype=torch.float16
        )

        try:
            with torch.no_grad():
                output = attn(q, k, v)
            assert (
                output.shape[0] == total_tokens
            ), f"Expected {total_tokens} tokens, got {output.shape[0]}"
            print(f"  ✅ Shape: {output.shape}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            return False

    print("\n✅ Test 3 passed!")
    return True


def test_integration_with_torchtitan_format():
    """Test integration with TorchTitan's tensor format."""
    print("\n" + "=" * 70)
    print("Test 4: TorchTitan Format Integration")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_heads = 16
    num_kv_heads = 8
    head_dim = 128

    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        swap_space=4,
        cache_dtype="auto",
    )

    attn = Attention(
        num_heads=num_heads,
        head_size=head_dim,
        scale=1.0 / (head_dim**0.5),
        num_kv_heads=num_kv_heads,
        cache_config=cache_config,
        quant_config=None,
        prefix="layers.3",
    )
    attn = attn.to(device)
    attn.eval()

    # Simulate TorchTitan format: [batch, num_heads, seq_len, head_dim]
    batch_size = 2
    seq_len = 64

    print(
        f"\nTorchTitan input format: [batch={batch_size}, num_heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}]"
    )

    q_tt = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16
    )
    k_tt = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16
    )
    v_tt = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16
    )

    print(f"  q_tt: {q_tt.shape}")
    print(f"  k_tt: {k_tt.shape}")
    print(f"  v_tt: {v_tt.shape}")

    # Convert to vLLM format: [total_tokens, num_heads, head_dim]
    total_tokens = batch_size * seq_len
    q_vllm = q_tt.transpose(1, 2).reshape(total_tokens, num_heads, head_dim)
    k_vllm = k_tt.transpose(1, 2).reshape(total_tokens, num_heads, head_dim)
    v_vllm = v_tt.transpose(1, 2).reshape(total_tokens, num_heads, head_dim)

    print(
        f"\nvLLM input format: [total_tokens={total_tokens}, num_heads={num_heads}, head_dim={head_dim}]"
    )
    print(f"  q_vllm: {q_vllm.shape}")
    print(f"  k_vllm: {k_vllm.shape}")
    print(f"  v_vllm: {v_vllm.shape}")

    try:
        with torch.no_grad():
            output_vllm = attn(q_vllm, k_vllm, v_vllm)

        print(f"\nvLLM output: {output_vllm.shape}")

        # Convert back to TorchTitan format
        output_tt = output_vllm.reshape(
            batch_size, seq_len, num_heads, head_dim
        ).transpose(1, 2)
        print(f"TorchTitan output: {output_tt.shape}")

        assert (
            output_tt.shape == q_tt.shape
        ), f"Output shape mismatch: {output_tt.shape} vs {q_tt.shape}"
        print("\n✅ Test 4 passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test 4 failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("Testing vLLM Attention Layer")
    print("=" * 70)

    all_passed = True

    # Run all tests
    all_passed &= test_vllm_attention_basic()
    all_passed &= test_vllm_attention_gqa_expansion()
    all_passed &= test_vllm_attention_shapes()
    all_passed &= test_integration_with_torchtitan_format()

    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 All tests passed successfully!")
    else:
        print("❌ Some tests failed!")
    print("=" * 70)
