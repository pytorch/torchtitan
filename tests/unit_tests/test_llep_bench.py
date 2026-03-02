#!/usr/bin/env python3
# Copyright (c) Nous Research and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark + correctness tests for LLEP optimizations.

Compares OLD (original) implementations vs NEW (optimized) implementations for:
1. _pad_for_grouped_mm  — vectorized GPU scatter vs Python for-loop
2. _unpad_output        — vectorized GPU gather vs Python for-loop
3. _pack_expert_weights — tensor indexing vs per-element .item() loop
4. assign_tokens_to_gpus — batched D2H sync vs N separate syncs
5. compute_llep_lpt_plan — hoisted closure vs re-created closure
6. fused_silu_gate       — cached import vs try/import per call
7. llep_moe_forward      — cached env lookups vs os.environ.get per call

Usage:
    .venv/bin/python torchtitan/tests/unit_tests/test_llep_bench.py
    .venv/bin/python torchtitan/tests/unit_tests/test_llep_bench.py --cuda   # GPU benchmarks
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# OLD reference implementations (copied from pre-optimization code)
# ---------------------------------------------------------------------------
_TOKEN_ALIGN = 8


def _pad_for_grouped_mm_OLD(x_sorted, counts):
    """Original Python for-loop version."""
    counts_padded = ((counts + _TOKEN_ALIGN - 1) // _TOKEN_ALIGN * _TOKEN_ALIGN).to(
        torch.int64
    )
    counts_padded = torch.clamp_min(counts_padded, _TOKEN_ALIGN)

    total_padded = counts_padded.sum().item()
    dim = x_sorted.shape[1]
    device = x_sorted.device
    dtype = x_sorted.dtype

    x_padded = torch.zeros(total_padded, dim, device=device, dtype=dtype)

    src_offset = 0
    dst_offset = 0
    counts_list = counts.tolist()
    counts_padded_list = counts_padded.tolist()
    for i in range(len(counts_list)):
        c = int(counts_list[i])
        cp = int(counts_padded_list[i])
        if c > 0:
            x_padded[dst_offset : dst_offset + c] = x_sorted[
                src_offset : src_offset + c
            ]
        src_offset += c
        dst_offset += cp

    return x_padded, counts_padded


def _unpad_output_OLD(out_padded, counts, counts_padded):
    """Original Python for-loop version."""
    total_tokens = counts.sum().item()
    dim = out_padded.shape[1]
    device = out_padded.device
    dtype = out_padded.dtype

    out = torch.empty(total_tokens, dim, device=device, dtype=dtype)

    src_offset = 0
    dst_offset = 0
    counts_list = counts.tolist()
    counts_padded_list = counts_padded.tolist()
    for i in range(len(counts_list)):
        c = int(counts_list[i])
        cp = int(counts_padded_list[i])
        if c > 0:
            out[dst_offset : dst_offset + c] = out_padded[src_offset : src_offset + c]
        src_offset += cp
        dst_offset += c

    return out


def _pack_expert_weights_OLD(
    unique_experts,
    w1_local,
    w2_local,
    w3_local,
    foreign_w1,
    foreign_w2,
    foreign_w3,
    foreign_w1_stacked,
    foreign_w2_stacked,
    foreign_w3_stacked,
    foreign_expert_id_mapping,
    ep_rank,
    num_local_experts,
):
    """Original per-element .item() loop version."""
    native_start = ep_rank * num_local_experts
    native_end = native_start + num_local_experts
    use_stacked = foreign_expert_id_mapping is not None

    w1_list = []
    w2_list = []
    w3_list = []
    valid_mask = []

    for eid_tensor in unique_experts:
        eid = eid_tensor.item()
        if native_start <= eid < native_end:
            local_idx = eid - native_start
            w1_list.append(w1_local[local_idx])
            w2_list.append(w2_local[local_idx])
            w3_list.append(w3_local[local_idx])
            valid_mask.append(True)
        elif use_stacked:
            stacked_idx = foreign_expert_id_mapping[eid].item()
            if stacked_idx >= 0:
                w1_list.append(foreign_w1_stacked[stacked_idx])
                w2_list.append(foreign_w2_stacked[stacked_idx])
                w3_list.append(foreign_w3_stacked[stacked_idx])
                valid_mask.append(True)
            else:
                w1_list.append(torch.zeros_like(w1_local[0]))
                w2_list.append(torch.zeros_like(w2_local[0]))
                w3_list.append(torch.zeros_like(w3_local[0]))
                valid_mask.append(False)
        elif foreign_w1 is not None and eid in foreign_w1:
            w1_list.append(foreign_w1[eid])
            w2_list.append(foreign_w2[eid])
            w3_list.append(foreign_w3[eid])
            valid_mask.append(True)
        else:
            w1_list.append(torch.zeros_like(w1_local[0]))
            w2_list.append(torch.zeros_like(w2_local[0]))
            w3_list.append(torch.zeros_like(w3_local[0]))
            valid_mask.append(False)

    return torch.stack(w1_list), torch.stack(w2_list), torch.stack(w3_list), valid_mask


def _assign_tokens_d2h_OLD(all_expert_counts):
    """OLD: N separate D2H syncs."""
    return np.stack([ec.cpu().numpy() for ec in all_expert_counts])


def _assign_tokens_d2h_NEW(all_expert_counts):
    """NEW: single D2H sync."""
    return torch.stack(all_expert_counts).cpu().numpy()


# ---------------------------------------------------------------------------
# Benchmarking utilities
# ---------------------------------------------------------------------------
def bench(fn, warmup=5, iters=50, sync_cuda=False):
    """Run fn() for warmup+iters iterations, return median time in ms."""
    for _ in range(warmup):
        fn()
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    median = times[len(times) // 2]
    mean = sum(times) / len(times)
    return median, mean


def report(name, old_ms, new_ms):
    """Print speedup report."""
    speedup = old_ms / new_ms if new_ms > 0 else float("inf")
    print(
        f"  {name:<35s}  OLD={old_ms:8.3f}ms  NEW={new_ms:8.3f}ms  speedup={speedup:.2f}x"
    )


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------
def test_pad_correctness(device):
    """Verify _pad_for_grouped_mm produces identical output."""
    from torchtitan.distributed.llep import _pad_for_grouped_mm

    print("\n=== Correctness: _pad_for_grouped_mm ===")
    torch.manual_seed(42)

    for n_experts in [4, 16, 64, 128]:
        for total_tokens in [128, 1024, 8192]:
            # Random expert counts
            counts = torch.randint(
                0, total_tokens // n_experts * 3, (n_experts,), device=device
            )
            # Make sure at least some are zero
            counts[0] = 0
            counts[-1] = 0
            actual_total = counts.sum().item()
            if actual_total == 0:
                counts[1] = 100
                actual_total = 100

            dim = 256
            x_sorted = torch.randn(
                actual_total, dim, device=device, dtype=torch.bfloat16
            )

            out_old, cp_old = _pad_for_grouped_mm_OLD(x_sorted, counts)
            out_new, cp_new = _pad_for_grouped_mm(x_sorted, counts)

            assert torch.equal(
                cp_old, cp_new
            ), f"counts_padded mismatch for {n_experts} experts"
            assert torch.equal(out_old, out_new), (
                f"x_padded mismatch for {n_experts} experts, {actual_total} tokens, "
                f"max_diff={( out_old - out_new).abs().max().item()}"
            )

    print("  PASSED (all sizes)")


def test_unpad_correctness(device):
    """Verify _unpad_output produces identical output."""
    from torchtitan.distributed.llep import _unpad_output

    print("\n=== Correctness: _unpad_output ===")
    torch.manual_seed(42)

    for n_experts in [4, 16, 64, 128]:
        counts = torch.randint(0, 200, (n_experts,), device=device)
        counts[0] = 0  # ensure at least one empty
        if counts.sum().item() == 0:
            counts[1] = 50

        counts_padded = ((counts + _TOKEN_ALIGN - 1) // _TOKEN_ALIGN * _TOKEN_ALIGN).to(
            torch.int64
        )
        counts_padded = torch.clamp_min(counts_padded, _TOKEN_ALIGN)

        total_padded = counts_padded.sum().item()
        dim = 256
        out_padded = torch.randn(total_padded, dim, device=device, dtype=torch.bfloat16)

        out_old = _unpad_output_OLD(out_padded, counts, counts_padded)
        out_new = _unpad_output(out_padded, counts, counts_padded)

        assert torch.equal(out_old, out_new), (
            f"_unpad_output mismatch for {n_experts} experts, "
            f"max_diff={(out_old - out_new).abs().max().item()}"
        )

    print("  PASSED (all sizes)")


def test_pack_weights_correctness(device):
    """Verify _pack_expert_weights produces identical output."""
    from torchtitan.distributed.llep import _pack_expert_weights

    print("\n=== Correctness: _pack_expert_weights ===")
    torch.manual_seed(42)

    dim = 128
    hidden_dim = 256
    num_local_experts = 8
    ep_rank = 1
    native_start = ep_rank * num_local_experts
    num_experts = 32

    w1_local = torch.randn(
        num_local_experts, hidden_dim, dim, device=device, dtype=torch.bfloat16
    )
    w2_local = torch.randn(
        num_local_experts, dim, hidden_dim, device=device, dtype=torch.bfloat16
    )
    w3_local = torch.randn(
        num_local_experts, hidden_dim, dim, device=device, dtype=torch.bfloat16
    )

    # Test 1: Stacked foreign path
    print("  Stacked foreign path...")
    num_foreign = 5
    foreign_w1_stacked = torch.randn(
        num_foreign, hidden_dim, dim, device=device, dtype=torch.bfloat16
    )
    foreign_w2_stacked = torch.randn(
        num_foreign, dim, hidden_dim, device=device, dtype=torch.bfloat16
    )
    foreign_w3_stacked = torch.randn(
        num_foreign, hidden_dim, dim, device=device, dtype=torch.bfloat16
    )

    mapping = torch.full((num_experts,), -1, dtype=torch.long, device=device)
    # Map some non-native experts to stacked indices
    foreign_eids = [0, 1, 2, 20, 25]
    for i, eid in enumerate(foreign_eids):
        mapping[eid] = i

    # Mix of native and foreign experts
    unique = torch.tensor(
        [0, 1, native_start, native_start + 1, native_start + 3, 20, 25, 30],
        device=device,
        dtype=torch.long,
    )

    args = (
        unique,
        w1_local,
        w2_local,
        w3_local,
        None,
        None,
        None,
        foreign_w1_stacked,
        foreign_w2_stacked,
        foreign_w3_stacked,
        mapping,
        ep_rank,
        num_local_experts,
    )

    w1_old, w2_old, w3_old, vm_old = _pack_expert_weights_OLD(*args)
    w1_new, w2_new, w3_new, vm_new = _pack_expert_weights(*args)

    assert vm_old == vm_new, f"valid_mask mismatch: {vm_old} vs {vm_new}"
    assert torch.equal(
        w1_old, w1_new
    ), f"w1 mismatch, max_diff={(w1_old - w1_new).abs().max().item()}"
    assert torch.equal(
        w2_old, w2_new
    ), f"w2 mismatch, max_diff={(w2_old - w2_new).abs().max().item()}"
    assert torch.equal(
        w3_old, w3_new
    ), f"w3 mismatch, max_diff={(w3_old - w3_new).abs().max().item()}"

    # Test 2: Dict foreign path
    print("  Dict foreign path...")
    foreign_w1_dict = {
        0: torch.randn(hidden_dim, dim, device=device, dtype=torch.bfloat16),
        20: torch.randn(hidden_dim, dim, device=device, dtype=torch.bfloat16),
    }
    foreign_w2_dict = {
        0: torch.randn(dim, hidden_dim, device=device, dtype=torch.bfloat16),
        20: torch.randn(dim, hidden_dim, device=device, dtype=torch.bfloat16),
    }
    foreign_w3_dict = {
        0: torch.randn(hidden_dim, dim, device=device, dtype=torch.bfloat16),
        20: torch.randn(hidden_dim, dim, device=device, dtype=torch.bfloat16),
    }

    unique2 = torch.tensor(
        [0, native_start, native_start + 2, 20, 30], device=device, dtype=torch.long
    )
    args2 = (
        unique2,
        w1_local,
        w2_local,
        w3_local,
        foreign_w1_dict,
        foreign_w2_dict,
        foreign_w3_dict,
        None,
        None,
        None,
        None,
        ep_rank,
        num_local_experts,
    )

    w1_old2, w2_old2, w3_old2, vm_old2 = _pack_expert_weights_OLD(*args2)
    w1_new2, w2_new2, w3_new2, vm_new2 = _pack_expert_weights(*args2)

    assert vm_old2 == vm_new2, f"valid_mask mismatch (dict): {vm_old2} vs {vm_new2}"
    assert torch.equal(w1_old2, w1_new2), f"w1 dict mismatch"
    assert torch.equal(w2_old2, w2_new2), f"w2 dict mismatch"
    assert torch.equal(w3_old2, w3_new2), f"w3 dict mismatch"

    # Test 3: No foreign experts at all
    print("  No foreign experts path...")
    unique3 = torch.tensor(
        [native_start, native_start + 1, native_start + 5],
        device=device,
        dtype=torch.long,
    )
    args3 = (
        unique3,
        w1_local,
        w2_local,
        w3_local,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        ep_rank,
        num_local_experts,
    )

    w1_old3, w2_old3, w3_old3, vm_old3 = _pack_expert_weights_OLD(*args3)
    w1_new3, w2_new3, w3_new3, vm_new3 = _pack_expert_weights(*args3)

    assert (
        vm_old3 == vm_new3
    ), f"valid_mask mismatch (no foreign): {vm_old3} vs {vm_new3}"
    assert torch.equal(w1_old3, w1_new3), f"w1 no-foreign mismatch"

    print("  PASSED (all paths)")


def test_d2h_sync_correctness(device):
    """Verify batched D2H produces identical numpy arrays."""
    print("\n=== Correctness: D2H sync batching ===")

    ep_size = 8
    num_experts = 64
    all_expert_counts = [
        torch.randint(0, 500, (num_experts,), dtype=torch.int64, device=device)
        for _ in range(ep_size)
    ]

    old = _assign_tokens_d2h_OLD(all_expert_counts)
    new = _assign_tokens_d2h_NEW(all_expert_counts)

    assert np.array_equal(old, new), f"D2H sync mismatch!"
    print("  PASSED")


def test_lpt_plan_correctness():
    """Verify hoisted closure doesn't change LPT plan output."""
    from torchtitan.distributed.llep import compute_llep_lpt_plan

    print("\n=== Correctness: compute_llep_lpt_plan (hoisted closure) ===")

    # Run the function with various inputs to verify it still works
    for seed in range(5):
        torch.manual_seed(seed)
        num_experts = 16
        ep_size = 4
        num_local_experts = 4

        counts = torch.randint(0, 2000, (num_experts,), dtype=torch.int64)
        # Make one expert very hot
        counts[seed % num_experts] = 5000

        plan = compute_llep_lpt_plan(
            counts,
            ep_size,
            ep_rank=0,
            num_local_experts=num_local_experts,
            max_tokens_factor=1.1,
            min_tokens_per_gemm=100,
        )

        # Verify basic invariants
        total_assigned = plan.gpu_loads.sum().item()
        total_tokens = counts.sum().item()
        assert (
            total_assigned == total_tokens
        ), f"Token count mismatch: assigned={total_assigned} vs total={total_tokens}"

        # Verify no expert exceeds its token count
        for eid, assignments in plan.lpt_plan.items():
            total_for_expert = sum(end - start for _, start, end in assignments)
            assert (
                total_for_expert == counts[eid].item()
            ), f"Expert {eid}: assigned {total_for_expert} != count {counts[eid].item()}"

    print("  PASSED (5 random seeds)")


def test_swiglu_ffn_correctness(device):
    """Verify full SwiGLU FFN (grouped_mm path) produces identical output."""
    from torchtitan.distributed.llep import _llep_swiglu_ffn_forloop, llep_swiglu_ffn

    print("\n=== Correctness: SwiGLU FFN (grouped_mm vs forloop) ===")

    torch.manual_seed(42)
    dim = 128
    hidden_dim = 256
    num_experts = 8
    ep_rank = 0
    num_local_experts = 8

    w1 = torch.randn(num_experts, hidden_dim, dim, device=device, dtype=torch.bfloat16)
    w2 = torch.randn(num_experts, dim, hidden_dim, device=device, dtype=torch.bfloat16)
    w3 = torch.randn(num_experts, hidden_dim, dim, device=device, dtype=torch.bfloat16)

    for num_tokens in [32, 128, 512, 2048]:
        x = torch.randn(num_tokens, dim, device=device, dtype=torch.bfloat16)
        expert_ids = torch.randint(0, num_experts, (num_tokens,), device=device)

        # Forloop (reference)
        out_ref = _llep_swiglu_ffn_forloop(
            x,
            expert_ids,
            w1,
            w2,
            w3,
            {},
            {},
            {},
            ep_rank,
            num_local_experts,
        )

        # grouped_mm (optimized)
        if device.type == "cuda":
            out_opt = llep_swiglu_ffn(
                x,
                expert_ids,
                w1,
                w2,
                w3,
                {},
                {},
                {},
                ep_rank,
                num_local_experts,
            )
            # bf16 grouped_mm uses different tiling/accumulation than matmul,
            # so we check relative tolerance (not absolute)
            abs_diff = (out_ref - out_opt).abs()
            max_diff = abs_diff.max().item()
            ref_scale = out_ref.abs().mean().item() + 1e-6
            rel_diff = max_diff / ref_scale
            assert rel_diff < 0.05, (
                f"FFN mismatch for {num_tokens} tokens: max_diff={max_diff}, "
                f"rel_diff={rel_diff:.4f}"
            )
            print(
                f"  {num_tokens:5d} tokens: max_diff={max_diff:.2f}, rel={rel_diff:.4f} OK"
            )
        else:
            # CPU: both use forloop
            out_opt = llep_swiglu_ffn(
                x,
                expert_ids,
                w1,
                w2,
                w3,
                {},
                {},
                {},
                ep_rank,
                num_local_experts,
            )
            assert torch.equal(out_ref, out_opt)
            print(f"  {num_tokens:5d} tokens: exact match (CPU forloop) OK")

    print("  PASSED")


# ---------------------------------------------------------------------------
# Performance benchmarks
# ---------------------------------------------------------------------------
def bench_pad_for_grouped_mm(device, sync_cuda):
    """Benchmark _pad_for_grouped_mm."""
    from torchtitan.distributed.llep import _pad_for_grouped_mm

    print("\n=== Benchmark: _pad_for_grouped_mm ===")

    configs = [
        (8, 1024, 256, "small"),
        (32, 4096, 512, "medium"),
        (64, 16384, 512, "large"),
        (128, 32768, 256, "xlarge"),
    ]

    for n_experts, total_tokens, dim, label in configs:
        torch.manual_seed(0)
        counts = torch.randint(
            1, total_tokens // n_experts * 3, (n_experts,), device=device
        )
        counts[0] = 0  # at least one empty
        actual_total = counts.sum().item()
        x_sorted = torch.randn(actual_total, dim, device=device, dtype=torch.bfloat16)

        old_median, _ = bench(
            lambda: _pad_for_grouped_mm_OLD(x_sorted, counts), sync_cuda=sync_cuda
        )
        new_median, _ = bench(
            lambda: _pad_for_grouped_mm(x_sorted, counts), sync_cuda=sync_cuda
        )
        report(
            f"pad [{label}: {n_experts}exp, {actual_total}tok]", old_median, new_median
        )


def bench_unpad_output(device, sync_cuda):
    """Benchmark _unpad_output."""
    from torchtitan.distributed.llep import _unpad_output

    print("\n=== Benchmark: _unpad_output ===")

    configs = [
        (8, 1024, 256, "small"),
        (32, 4096, 512, "medium"),
        (64, 16384, 512, "large"),
        (128, 32768, 256, "xlarge"),
    ]

    for n_experts, total_tokens, dim, label in configs:
        torch.manual_seed(0)
        counts = torch.randint(
            1, total_tokens // n_experts * 3, (n_experts,), device=device
        )
        counts[0] = 0
        counts_padded = ((counts + _TOKEN_ALIGN - 1) // _TOKEN_ALIGN * _TOKEN_ALIGN).to(
            torch.int64
        )
        counts_padded = torch.clamp_min(counts_padded, _TOKEN_ALIGN)
        total_padded = counts_padded.sum().item()
        dim_val = dim
        out_padded = torch.randn(
            total_padded, dim_val, device=device, dtype=torch.bfloat16
        )

        old_median, _ = bench(
            lambda: _unpad_output_OLD(out_padded, counts, counts_padded),
            sync_cuda=sync_cuda,
        )
        new_median, _ = bench(
            lambda: _unpad_output(out_padded, counts, counts_padded),
            sync_cuda=sync_cuda,
        )
        report(f"unpad [{label}: {n_experts}exp]", old_median, new_median)


def bench_pack_expert_weights(device, sync_cuda):
    """Benchmark _pack_expert_weights."""
    from torchtitan.distributed.llep import _pack_expert_weights

    print("\n=== Benchmark: _pack_expert_weights ===")

    configs = [
        (8, 4, 1, 128, 256, "small 8exp"),
        (16, 8, 1, 256, 512, "medium 16exp"),
        (32, 8, 2, 256, 512, "large 32exp"),
        (64, 8, 4, 512, 1024, "xlarge 64exp"),
    ]

    for num_experts, num_local, ep_rank, dim, hidden_dim, label in configs:
        torch.manual_seed(0)
        w1_local = torch.randn(
            num_local, hidden_dim, dim, device=device, dtype=torch.bfloat16
        )
        w2_local = torch.randn(
            num_local, dim, hidden_dim, device=device, dtype=torch.bfloat16
        )
        w3_local = torch.randn(
            num_local, hidden_dim, dim, device=device, dtype=torch.bfloat16
        )

        native_start = ep_rank * num_local
        num_foreign = min(8, num_experts - num_local)
        fw1 = torch.randn(
            num_foreign, hidden_dim, dim, device=device, dtype=torch.bfloat16
        )
        fw2 = torch.randn(
            num_foreign, dim, hidden_dim, device=device, dtype=torch.bfloat16
        )
        fw3 = torch.randn(
            num_foreign, hidden_dim, dim, device=device, dtype=torch.bfloat16
        )

        mapping = torch.full((num_experts,), -1, dtype=torch.long, device=device)
        foreign_eids = [
            i
            for i in range(num_experts)
            if not (native_start <= i < native_start + num_local)
        ][:num_foreign]
        for i, eid in enumerate(foreign_eids):
            mapping[eid] = i

        # Mix of native and foreign
        native_eids = list(range(native_start, native_start + num_local))
        unique = torch.tensor(
            sorted(native_eids + foreign_eids), device=device, dtype=torch.long
        )

        args = (
            unique,
            w1_local,
            w2_local,
            w3_local,
            None,
            None,
            None,
            fw1,
            fw2,
            fw3,
            mapping,
            ep_rank,
            num_local,
        )

        old_median, _ = bench(
            lambda: _pack_expert_weights_OLD(*args), sync_cuda=sync_cuda
        )
        new_median, _ = bench(lambda: _pack_expert_weights(*args), sync_cuda=sync_cuda)
        report(f"pack [{label}]", old_median, new_median)


def bench_d2h_sync(device, sync_cuda):
    """Benchmark D2H sync batching."""
    print("\n=== Benchmark: D2H sync batching ===")

    configs = [
        (2, 16, "2 GPUs, 16 experts"),
        (4, 32, "4 GPUs, 32 experts"),
        (8, 64, "8 GPUs, 64 experts"),
        (16, 128, "16 GPUs, 128 experts"),
    ]

    for ep_size, num_experts, label in configs:
        all_expert_counts = [
            torch.randint(0, 500, (num_experts,), dtype=torch.int64, device=device)
            for _ in range(ep_size)
        ]

        old_median, _ = bench(
            lambda: _assign_tokens_d2h_OLD(all_expert_counts),
            sync_cuda=sync_cuda,
        )
        new_median, _ = bench(
            lambda: _assign_tokens_d2h_NEW(all_expert_counts),
            sync_cuda=sync_cuda,
        )
        report(f"d2h [{label}]", old_median, new_median)


def bench_lpt_plan(device):
    """Benchmark compute_llep_lpt_plan."""
    from torchtitan.distributed.llep import compute_llep_lpt_plan

    print("\n=== Benchmark: compute_llep_lpt_plan ===")

    configs = [
        (8, 2, 4, "8exp/2gpu"),
        (16, 4, 4, "16exp/4gpu"),
        (64, 8, 8, "64exp/8gpu"),
        (128, 8, 16, "128exp/8gpu"),
        (256, 8, 32, "256exp/8gpu"),
    ]

    for num_experts, ep_size, num_local, label in configs:
        torch.manual_seed(0)
        counts = torch.randint(0, 2000, (num_experts,), dtype=torch.int64)
        counts[0] = 10000  # hot expert

        median, mean = bench(
            lambda: compute_llep_lpt_plan(
                counts,
                ep_size,
                ep_rank=0,
                num_local_experts=num_local,
                max_tokens_factor=1.1,
                min_tokens_per_gemm=100,
            ),
            warmup=3,
            iters=30,
        )
        print(f"  {label:<25s}  median={median:8.3f}ms  mean={mean:8.3f}ms")


def bench_fused_silu_gate(device, sync_cuda):
    """Benchmark cached vs uncached fused_silu_gate import."""
    print("\n=== Benchmark: fused_silu_gate (cached import) ===")

    import torchtitan.distributed.llep as llep_mod

    configs = [
        (1024, 512, "1K tokens"),
        (4096, 512, "4K tokens"),
        (16384, 1024, "16K tokens"),
    ]

    for num_tokens, hidden_dim, label in configs:
        x1 = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16)
        x3 = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16)

        # Old: try/import every time
        def old_fn():
            try:
                from torchtitan.distributed.llep_kernels import fused_silu_gate

                if device.type == "cpu":
                    # Triton kernels don't work on CPU
                    return F.silu(x1) * x3
                return fused_silu_gate(x1, x3)
            except (ImportError, RuntimeError):
                return F.silu(x1) * x3

        # New: cached
        def new_fn():
            return (
                llep_mod._fused_silu_gate_fn(x1, x3)
                if llep_mod._fused_silu_gate_fn
                else F.silu(x1) * x3
            )

        # Initialize the cache
        if llep_mod._fused_silu_gate_fn is None:
            try:
                from torchtitan.distributed.llep_kernels import fused_silu_gate

                llep_mod._fused_silu_gate_fn = fused_silu_gate
            except (ImportError, RuntimeError):
                llep_mod._fused_silu_gate_fn = lambda x1, x3: F.silu(x1) * x3

        old_median, _ = bench(old_fn, sync_cuda=sync_cuda)
        new_median, _ = bench(new_fn, sync_cuda=sync_cuda)
        report(f"silu_gate [{label}]", old_median, new_median)


def bench_swiglu_ffn(device, sync_cuda):
    """Benchmark full SwiGLU FFN (end-to-end)."""
    from torchtitan.distributed.llep import _llep_swiglu_ffn_forloop, llep_swiglu_ffn

    print("\n=== Benchmark: SwiGLU FFN (full pipeline) ===")

    configs = [
        (8, 128, 256, 512, "8exp/512tok"),
        (8, 128, 256, 2048, "8exp/2Ktok"),
        (8, 128, 256, 8192, "8exp/8Ktok"),
        (16, 256, 512, 2048, "16exp/2Ktok"),
        (16, 256, 512, 8192, "16exp/8Ktok"),
    ]

    for num_experts, dim, hidden_dim, num_tokens, label in configs:
        torch.manual_seed(0)
        w1 = torch.randn(
            num_experts, hidden_dim, dim, device=device, dtype=torch.bfloat16
        )
        w2 = torch.randn(
            num_experts, dim, hidden_dim, device=device, dtype=torch.bfloat16
        )
        w3 = torch.randn(
            num_experts, hidden_dim, dim, device=device, dtype=torch.bfloat16
        )
        x = torch.randn(num_tokens, dim, device=device, dtype=torch.bfloat16)
        expert_ids = torch.randint(0, num_experts, (num_tokens,), device=device)

        # Forloop baseline
        old_median, _ = bench(
            lambda: _llep_swiglu_ffn_forloop(
                x,
                expert_ids,
                w1,
                w2,
                w3,
                {},
                {},
                {},
                0,
                num_experts,
            ),
            sync_cuda=sync_cuda,
        )

        if device.type == "cuda":
            new_median, _ = bench(
                lambda: llep_swiglu_ffn(
                    x,
                    expert_ids,
                    w1,
                    w2,
                    w3,
                    {},
                    {},
                    {},
                    0,
                    num_experts,
                ),
                sync_cuda=sync_cuda,
            )
        else:
            new_median = old_median

        report(f"ffn [{label}]", old_median, new_median)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LLEP optimization benchmark")
    parser.add_argument(
        "--cuda", action="store_true", help="Run on CUDA (default: CPU)"
    )
    parser.add_argument(
        "--bench-only", action="store_true", help="Skip correctness tests"
    )
    parser.add_argument(
        "--correctness-only", action="store_true", help="Skip benchmarks"
    )
    args = parser.parse_args()

    if args.cuda and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.cuda = False

    device = torch.device("cuda:0" if args.cuda else "cpu")
    sync_cuda = args.cuda

    print(f"Device: {device}")
    if args.cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ---- Correctness ----
    if not args.bench_only:
        print("=" * 60)
        print("CORRECTNESS TESTS")
        print("=" * 60)

        test_pad_correctness(device)
        test_unpad_correctness(device)
        test_pack_weights_correctness(device)
        test_d2h_sync_correctness(device)
        test_lpt_plan_correctness()
        test_swiglu_ffn_correctness(device)

        print("\n" + "=" * 60)
        print("ALL CORRECTNESS TESTS PASSED")
        print("=" * 60)

    # ---- Benchmarks ----
    if not args.correctness_only:
        print("\n" + "=" * 60)
        print("PERFORMANCE BENCHMARKS")
        print("=" * 60)

        bench_pad_for_grouped_mm(device, sync_cuda)
        bench_unpad_output(device, sync_cuda)
        bench_pack_expert_weights(device, sync_cuda)
        bench_d2h_sync(device, sync_cuda)
        bench_lpt_plan(device)
        bench_fused_silu_gate(device, sync_cuda)

        if args.cuda:
            bench_swiglu_ffn(device, sync_cuda)

        print("\n" + "=" * 60)
        print("BENCHMARKS COMPLETE")
        print("=" * 60)


if __name__ == "__main__":
    main()
