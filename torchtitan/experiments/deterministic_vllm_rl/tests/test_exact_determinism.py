# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test if batch_invariant operations are EXACTLY deterministic.

This runs the same operation multiple times and checks if results are bit-for-bit identical.
"""

import torch
from vllm.model_executor.layers.batch_invariant import disable_batch_invariant_mode

from torchtitan.experiments.deterministic_vllm_rl.batch_invariant_backward import (
    enable_batch_invariant_backward_mode,
)

print("Enabling batch_invariant_backward mode...")
disable_batch_invariant_mode()
enable_batch_invariant_backward_mode()


def test_mm_exact_determinism():
    """Test if mm is exactly deterministic."""
    print("\n" + "=" * 80)
    print("Testing mm exact determinism")
    print("=" * 80)

    # Create random inputs
    torch.manual_seed(42)
    a = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

    # Run multiple times
    results = []
    for i in range(5):
        c = torch.mm(a, b)
        results.append(c.clone())
        print(
            f"Run {i + 1}: mean={c.float().mean().item():.10f}, "
            f"std={c.float().std().item():.10f}"
        )

    # Check if all results are identical
    all_same = True
    for i in range(1, len(results)):
        if not torch.equal(results[0], results[i]):
            diff = (results[0] - results[i]).abs().max().item()
            print(f"  ✗ Run 1 vs Run {i + 1}: MAX DIFF = {diff}")
            all_same = False

    if all_same:
        print("  ✓ All runs produce IDENTICAL results (bit-for-bit)")
    else:
        print("  ✗ Results differ across runs")

    return all_same


def test_flash_attention_determinism():
    """Test if Flash Attention is exactly deterministic."""
    print("\n" + "=" * 80)
    print("Testing Flash Attention exact determinism")
    print("=" * 80)

    from vllm.vllm_flash_attn import flash_attn_varlen_func

    # Create random inputs
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 16
    num_heads = 32
    head_dim = 128

    q = torch.randn(
        batch_size * seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(
        batch_size * seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn(
        batch_size * seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )

    cu_seqlens = torch.arange(
        0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device="cuda"
    )

    # Run multiple times
    results = []
    for i in range(5):
        output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
            softmax_scale=1.0 / (head_dim**0.5),
            causal=True,
            num_splits=1,  # Deterministic mode
        )
        results.append(output.clone())
        print(
            f"Run {i + 1}: mean={output.float().mean().item():.10f}, "
            f"std={output.float().std().item():.10f}"
        )

    # Check if all results are identical
    all_same = True
    for i in range(1, len(results)):
        if not torch.equal(results[0], results[i]):
            diff = (results[0] - results[i]).abs().max().item()
            print(f"  ✗ Run 1 vs Run {i + 1}: MAX DIFF = {diff}")
            all_same = False

    if all_same:
        print("  ✓ All runs produce IDENTICAL results (bit-for-bit)")
    else:
        print("  ✗ Results differ across runs")

    return all_same


def test_log_softmax_determinism():
    """Test if log_softmax is exactly deterministic."""
    print("\n" + "=" * 80)
    print("Testing log_softmax exact determinism")
    print("=" * 80)

    import torch.nn.functional as F

    # Create random inputs
    torch.manual_seed(42)
    x = torch.randn(32, 151936, device="cuda", dtype=torch.bfloat16)

    # Run multiple times
    results = []
    for i in range(5):
        # Convert to float32 before log_softmax (as we do in training)
        output = F.log_softmax(x.float(), dim=-1)
        results.append(output.clone())
        print(
            f"Run {i + 1}: mean={output.mean().item():.10f}, "
            f"std={output.std().item():.10f}"
        )

    # Check if all results are identical
    all_same = True
    for i in range(1, len(results)):
        if not torch.equal(results[0], results[i]):
            diff = (results[0] - results[i]).abs().max().item()
            print(f"  ✗ Run 1 vs Run {i + 1}: MAX DIFF = {diff}")
            all_same = False

    if all_same:
        print("  ✓ All runs produce IDENTICAL results (bit-for-bit)")
    else:
        print("  ✗ Results differ across runs")

    return all_same


def main():
    print("Testing exact determinism of operations")
    print("=" * 80)

    results = {}
    results["mm"] = test_mm_exact_determinism()
    results["flash_attention"] = test_flash_attention_determinism()
    results["log_softmax"] = test_log_softmax_determinism()

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    for op, is_deterministic in results.items():
        status = "✓ DETERMINISTIC" if is_deterministic else "✗ NON-DETERMINISTIC"
        print(f"{op:<20}: {status}")

    if all(results.values()):
        print("\n✓ All operations are exactly deterministic!")
    else:
        print(
            "\n✗ Some operations are not deterministic - this explains the vLLM/TorchTitan difference"
        )


if __name__ == "__main__":
    main()
