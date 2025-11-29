#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple test script for flash_attn_varlen_func forward pass.
Tests the variable-length flash attention function from vLLM.
"""

import torch
from vllm.attention.utils.fa_utils import flash_attn_varlen_func


def test_flash_attn_varlen_func():
    """Test flash_attn_varlen_func with simple dummy inputs."""

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parameters
    batch_size = 2
    num_heads = 8
    head_dim = 64
    seq_len_q = 128
    seq_len_k = 128

    # Total tokens (for variable length)
    total_tokens_q = batch_size * seq_len_q
    total_tokens_k = batch_size * seq_len_k

    # Create input tensors
    # Shape: (total_tokens, num_heads, head_dim)
    q = torch.randn(
        total_tokens_q, num_heads, head_dim, dtype=torch.float16, device=device
    )
    k = torch.randn(
        total_tokens_k, num_heads, head_dim, dtype=torch.float16, device=device
    )
    v = torch.randn(
        total_tokens_k, num_heads, head_dim, dtype=torch.float16, device=device
    )

    # Create cumulative sequence lengths
    # cu_seqlens_q and cu_seqlens_k indicate the start position of each sequence
    # For uniform sequences: [0, seq_len, 2*seq_len, ...]
    cu_seqlens_q = torch.tensor(
        [0, seq_len_q, 2 * seq_len_q], dtype=torch.int32, device=device
    )
    cu_seqlens_k = torch.tensor(
        [0, seq_len_k, 2 * seq_len_k], dtype=torch.int32, device=device
    )

    # Maximum sequence lengths
    max_seqlen_q = seq_len_q
    max_seqlen_k = seq_len_k

    # Softmax scale (typically 1/sqrt(head_dim))
    softmax_scale = 1.0 / (head_dim**0.5)

    print("\nInput shapes:")
    print(f"  q: {q.shape}")
    print(f"  k: {k.shape}")
    print(f"  v: {v.shape}")
    print(f"  cu_seqlens_q: {cu_seqlens_q}")
    print(f"  cu_seqlens_k: {cu_seqlens_k}")
    print(f"  max_seqlen_q: {max_seqlen_q}")
    print(f"  max_seqlen_k: {max_seqlen_k}")
    print(f"  softmax_scale: {softmax_scale}")

    try:
        # Call flash_attn_varlen_func
        print("\nCalling flash_attn_varlen_func...")
        output = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=False,  # Set to True for causal attention
        )

        print(f"\nOutput shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        print(f"Output device: {output.device}")
        print("\nOutput statistics:")
        print(f"  Mean: {output.mean().item():.6f}")
        print(f"  Std: {output.std().item():.6f}")
        print(f"  Min: {output.min().item():.6f}")
        print(f"  Max: {output.max().item():.6f}")

        print("\n✓ Test passed successfully!")
        return output

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        raise


def test_flash_attn_varlen_func_causal():
    """Test flash_attn_varlen_func with causal attention."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print("Testing with causal attention")
    print(f"{'=' * 60}")
    print(f"Using device: {device}")

    # Smaller test for causal
    batch_size = 1
    num_heads = 4
    head_dim = 32
    seq_len = 64

    total_tokens = batch_size * seq_len

    q = torch.randn(
        total_tokens, num_heads, head_dim, dtype=torch.float16, device=device
    )
    k = torch.randn(
        total_tokens, num_heads, head_dim, dtype=torch.float16, device=device
    )
    v = torch.randn(
        total_tokens, num_heads, head_dim, dtype=torch.float16, device=device
    )

    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

    print("\nInput shapes:")
    print(f"  q, k, v: {q.shape}")
    print(f"  cu_seqlens: {cu_seqlens}")

    try:
        output = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
            causal=True,
        )

        print(f"\nCausal output shape: {output.shape}")
        print("Output statistics:")
        print(f"  Mean: {output.mean().item():.6f}")
        print(f"  Std: {output.std().item():.6f}")

        print("\n✓ Causal test passed successfully!")
        return output

    except Exception as e:
        print(f"\n✗ Causal test failed with error: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    print("=" * 60)
    print("Testing flash_attn_varlen_func")
    print("=" * 60)

    # Test 1: Basic forward pass
    test_flash_attn_varlen_func()

    # Test 2: Causal attention
    test_flash_attn_varlen_func_causal()

    print("\n" + "=" * 60)
    print("All tests completed successfully! ✓")
    print("=" * 60)
