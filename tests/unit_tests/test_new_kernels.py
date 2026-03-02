#!/usr/bin/env python3
# Copyright (c) Nous Research and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Correctness + Benchmark tests for new Triton kernels:
1. triton_pad_for_grouped_mm
2. triton_unpad_output
3. compute_send_matrix_vectorized
"""

import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/phuc/workspace/moe/small_prs/pr008_saleforce_lbs/torchtitan")

from torchtitan.distributed.llep import _pad_for_grouped_mm, _TOKEN_ALIGN, _unpad_output
from torchtitan.distributed.llep_kernels import (
    compute_send_matrix_vectorized,
    triton_pad_for_grouped_mm,
    triton_unpad_output,
)

DEVICE = "cuda"
WARMUP = 5
ITERS = 20

# Training sizes
DIM = 3072
NUM_EXPERTS = 256
EP_SIZE = 8
NUM_LOCAL = 32
N = 393216  # tokens * top_k


def timer(fn, warmup=WARMUP, iters=ITERS, label=""):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / iters * 1000
    print(f"  {label:50s} {elapsed:8.3f} ms")
    return elapsed


def make_counts(num_experts, total_tokens):
    """Create realistic expert counts."""
    rng = np.random.default_rng(42)
    weights = 1.0 / np.arange(1, num_experts + 1) ** 1.5
    weights /= weights.sum()
    counts = (weights * total_tokens).astype(np.int64)
    counts = np.maximum(counts, 1)
    diff = total_tokens - counts.sum()
    counts[0] += diff
    return torch.tensor(counts, dtype=torch.int64, device=DEVICE)


def test_pad_correctness():
    """Test triton_pad_for_grouped_mm matches _pad_for_grouped_mm."""
    print("\n=== PAD CORRECTNESS ===")

    for n_experts, total_tok in [(8, 1000), (32, 10000), (64, 50000), (256, N)]:
        counts = make_counts(n_experts, total_tok)
        x_sorted = torch.randn(total_tok, DIM, device=DEVICE, dtype=torch.bfloat16)

        # Compute aligned counts (same logic as _pad_for_grouped_mm)
        counts_padded = ((counts + _TOKEN_ALIGN - 1) // _TOKEN_ALIGN * _TOKEN_ALIGN).to(
            torch.int64
        )
        counts_padded = torch.clamp_min(counts_padded, _TOKEN_ALIGN)

        # Reference
        ref, ref_cp = _pad_for_grouped_mm(x_sorted, counts)

        # Triton
        tri = triton_pad_for_grouped_mm(x_sorted, counts, counts_padded)

        # Check padded counts match
        assert torch.equal(
            ref_cp, counts_padded
        ), f"Padded counts mismatch for {n_experts} experts"

        # Check values match
        max_diff = (ref.float() - tri.float()).abs().max().item()
        assert (
            max_diff == 0.0
        ), f"Pad mismatch for {n_experts} experts: max_diff={max_diff}"
        print(f"  {n_experts:4d} experts, {total_tok:7d} tokens: PASS (max_diff=0.0)")

    print("  ALL PAD CORRECTNESS TESTS PASSED")


def test_unpad_correctness():
    """Test triton_unpad_output matches _unpad_output."""
    print("\n=== UNPAD CORRECTNESS ===")

    for n_experts, total_tok in [(8, 1000), (32, 10000), (64, 50000), (256, N)]:
        counts = make_counts(n_experts, total_tok)
        x_sorted = torch.randn(total_tok, DIM, device=DEVICE, dtype=torch.bfloat16)

        counts_padded = ((counts + _TOKEN_ALIGN - 1) // _TOKEN_ALIGN * _TOKEN_ALIGN).to(
            torch.int64
        )
        counts_padded = torch.clamp_min(counts_padded, _TOKEN_ALIGN)

        # Create padded tensor
        x_padded = triton_pad_for_grouped_mm(x_sorted, counts, counts_padded)

        # Reference
        ref = _unpad_output(x_padded, counts, counts_padded)

        # Triton
        tri = triton_unpad_output(x_padded, counts, counts_padded)

        max_diff = (ref.float() - tri.float()).abs().max().item()
        assert (
            max_diff == 0.0
        ), f"Unpad mismatch for {n_experts} experts: max_diff={max_diff}"
        print(f"  {n_experts:4d} experts, {total_tok:7d} tokens: PASS (max_diff=0.0)")

    print("  ALL UNPAD CORRECTNESS TESTS PASSED")


def test_pad_roundtrip():
    """Test pad -> unpad gives back original data."""
    print("\n=== PAD/UNPAD ROUNDTRIP ===")

    counts = make_counts(NUM_EXPERTS, N)
    x_sorted = torch.randn(N, DIM, device=DEVICE, dtype=torch.bfloat16)

    counts_padded = ((counts + _TOKEN_ALIGN - 1) // _TOKEN_ALIGN * _TOKEN_ALIGN).to(
        torch.int64
    )
    counts_padded = torch.clamp_min(counts_padded, _TOKEN_ALIGN)

    x_padded = triton_pad_for_grouped_mm(x_sorted, counts, counts_padded)
    x_recovered = triton_unpad_output(x_padded, counts, counts_padded)

    max_diff = (x_sorted.float() - x_recovered.float()).abs().max().item()
    assert max_diff == 0.0, f"Roundtrip mismatch: max_diff={max_diff}"
    print(f"  256 experts, {N} tokens: PASS (roundtrip exact)")


def test_send_matrix_correctness():
    """Test compute_send_matrix_vectorized matches reference."""
    print("\n=== SEND MATRIX CORRECTNESS ===")
    from torchtitan.distributed.llep import compute_llep_lpt_plan

    rng = np.random.default_rng(42)

    for trial in range(3):
        # Create random counts
        total_global = N * EP_SIZE
        global_counts = make_counts(NUM_EXPERTS, total_global)

        # Create per-rank counts
        all_counts = []
        remaining = global_counts.clone()
        for r in range(EP_SIZE):
            if r < EP_SIZE - 1:
                rc = (remaining / (EP_SIZE - r)).to(torch.int64).clamp(min=0)
            else:
                rc = remaining
            all_counts.append(rc.clone())
            remaining -= rc

        all_counts_np = torch.stack(all_counts).cpu().numpy()
        cum_counts_np = np.zeros((EP_SIZE + 1, NUM_EXPERTS), dtype=np.int64)
        cum_counts_np[1:] = np.cumsum(all_counts_np, axis=0)

        # Get LPT plan
        plan = compute_llep_lpt_plan(
            global_counts,
            EP_SIZE,
            0,
            NUM_LOCAL,
            1.1,
            1024,
        )

        # Reference: the loop-based send_matrix from assign_tokens_to_gpus
        ref_send = np.zeros((EP_SIZE, EP_SIZE), dtype=np.int64)
        lpt_expert_set = set(plan.lpt_plan.keys())

        for eid in range(NUM_EXPERTS):
            if eid not in lpt_expert_set:
                owner = eid // NUM_LOCAL
                ref_send[:, owner] += all_counts_np[:, eid]

        for expert_id, assignments in plan.lpt_plan.items():
            for src_rank in range(EP_SIZE):
                src_start = int(cum_counts_np[src_rank, expert_id])
                src_end = int(cum_counts_np[src_rank + 1, expert_id])
                if src_start == src_end:
                    continue
                for dst_gpu, dst_start, dst_end in assignments:
                    os_ = max(src_start, dst_start)
                    oe_ = min(src_end, dst_end)
                    if os_ < oe_:
                        ref_send[src_rank, dst_gpu] += oe_ - os_

        # Vectorized
        vec_send = compute_send_matrix_vectorized(
            all_counts_np,
            cum_counts_np,
            plan.lpt_plan,
            EP_SIZE,
            NUM_LOCAL,
            NUM_EXPERTS,
        )

        assert np.array_equal(
            ref_send, vec_send
        ), f"Trial {trial}: send_matrix mismatch!\nRef:\n{ref_send}\nVec:\n{vec_send}"
        print(f"  Trial {trial}: PASS (LPT experts: {len(plan.lpt_plan)})")

    print("  ALL SEND MATRIX CORRECTNESS TESTS PASSED")


def bench_pad():
    """Benchmark pad at training sizes."""
    print("\n=== PAD BENCHMARK (256 experts, 393K tokens, dim=3072) ===")

    counts = make_counts(NUM_EXPERTS, N)
    x_sorted = torch.randn(N, DIM, device=DEVICE, dtype=torch.bfloat16)

    counts_padded = ((counts + _TOKEN_ALIGN - 1) // _TOKEN_ALIGN * _TOKEN_ALIGN).to(
        torch.int64
    )
    counts_padded = torch.clamp_min(counts_padded, _TOKEN_ALIGN)

    ref_time = timer(
        lambda: _pad_for_grouped_mm(x_sorted, counts),
        label="PyTorch vectorized _pad_for_grouped_mm",
    )
    tri_time = timer(
        lambda: triton_pad_for_grouped_mm(x_sorted, counts, counts_padded),
        label="Triton triton_pad_for_grouped_mm",
    )
    print(f"  Speedup: {ref_time / tri_time:.2f}x")


def bench_unpad():
    """Benchmark unpad at training sizes."""
    print("\n=== UNPAD BENCHMARK (256 experts, 393K tokens, dim=3072) ===")

    counts = make_counts(NUM_EXPERTS, N)
    x_sorted = torch.randn(N, DIM, device=DEVICE, dtype=torch.bfloat16)

    counts_padded = ((counts + _TOKEN_ALIGN - 1) // _TOKEN_ALIGN * _TOKEN_ALIGN).to(
        torch.int64
    )
    counts_padded = torch.clamp_min(counts_padded, _TOKEN_ALIGN)

    x_padded, _ = _pad_for_grouped_mm(x_sorted, counts)

    ref_time = timer(
        lambda: _unpad_output(x_padded, counts, counts_padded),
        label="PyTorch vectorized _unpad_output",
    )
    tri_time = timer(
        lambda: triton_unpad_output(x_padded, counts, counts_padded),
        label="Triton triton_unpad_output",
    )
    print(f"  Speedup: {ref_time / tri_time:.2f}x")


def bench_send_matrix():
    """Benchmark send_matrix computation."""
    print("\n=== SEND MATRIX BENCHMARK ===")
    from torchtitan.distributed.llep import compute_llep_lpt_plan

    total_global = N * EP_SIZE
    global_counts = make_counts(NUM_EXPERTS, total_global)

    all_counts = []
    remaining = global_counts.clone()
    for r in range(EP_SIZE):
        if r < EP_SIZE - 1:
            rc = (remaining / (EP_SIZE - r)).to(torch.int64).clamp(min=0)
        else:
            rc = remaining
        all_counts.append(rc.clone())
        remaining -= rc

    all_counts_np = torch.stack(all_counts).cpu().numpy()
    cum_counts_np = np.zeros((EP_SIZE + 1, NUM_EXPERTS), dtype=np.int64)
    cum_counts_np[1:] = np.cumsum(all_counts_np, axis=0)

    plan = compute_llep_lpt_plan(global_counts, EP_SIZE, 0, NUM_LOCAL, 1.1, 1024)

    # Reference loop
    def ref_send():
        send_matrix_np = np.zeros((EP_SIZE, EP_SIZE), dtype=np.int64)
        lpt_expert_set = set(plan.lpt_plan.keys())
        for eid in range(NUM_EXPERTS):
            if eid not in lpt_expert_set:
                owner = eid // NUM_LOCAL
                send_matrix_np[:, owner] += all_counts_np[:, eid]
        for expert_id, assignments in plan.lpt_plan.items():
            for src_rank in range(EP_SIZE):
                src_start = int(cum_counts_np[src_rank, expert_id])
                src_end = int(cum_counts_np[src_rank + 1, expert_id])
                if src_start == src_end:
                    continue
                for dst_gpu, dst_start, dst_end in assignments:
                    os_ = max(src_start, dst_start)
                    oe_ = min(src_end, dst_end)
                    if os_ < oe_:
                        send_matrix_np[src_rank, dst_gpu] += oe_ - os_
        return send_matrix_np

    # Time both
    ref_time = timer(ref_send, label="Reference (nested Python loops)", iters=50)
    vec_time = timer(
        lambda: compute_send_matrix_vectorized(
            all_counts_np,
            cum_counts_np,
            plan.lpt_plan,
            EP_SIZE,
            NUM_LOCAL,
            NUM_EXPERTS,
        ),
        label="Vectorized numpy",
        iters=50,
    )
    print(f"  Speedup: {ref_time / vec_time:.2f}x")


if __name__ == "__main__":
    print("=" * 70)
    print("New Kernel Tests: Correctness + Benchmark")
    print("=" * 70)

    # Correctness
    test_pad_correctness()
    test_unpad_correctness()
    test_pad_roundtrip()
    test_send_matrix_correctness()

    # Benchmarks
    bench_pad()
    bench_unpad()
    bench_send_matrix()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
