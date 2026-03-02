#!/usr/bin/env python3
# Copyright (c) Nous Research and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Correctness tests for LLEP optimization implementations.

Tests that the grouped GEMM + fused Triton kernel optimizations produce
numerically correct results compared to reference implementations.

Test categories:
1. Grouped MM vs For-Loop (single-process, 7 tests)
2. Triton fused_silu_gate kernel (single-process, 4 tests)
3. Numerical stability (4 tests)
4. Performance benchmarks (2 tests)

Run with torchrun (requires >= 1 GPU):
    torchrun --nproc_per_node=1 tests/unit_tests/test_llep_correctness.py

Run specific test:
    torchrun --nproc_per_node=1 tests/unit_tests/test_llep_correctness.py --test grouped_vs_forloop_native

List all tests:
    torchrun --nproc_per_node=1 tests/unit_tests/test_llep_correctness.py --list
"""

import argparse
import sys
import traceback

import torch
import torch.distributed as dist
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Category 1: Grouped MM vs For-Loop (single-process)
# ---------------------------------------------------------------------------
def _test_grouped_vs_forloop(
    num_tokens: int = 32,
    dim: int = 64,
    hidden_dim: int = 128,
    num_local_experts: int = 4,
    ep_rank: int = 0,
    num_foreign: int = 0,
    routing: str = "balanced",
    atol: float = 1e-3,
    seed: int = 42,
    dtype: torch.dtype = torch.bfloat16,
):
    """Compare grouped_mm FFN path vs for-loop path on the same data."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    device = torch.device(f"cuda:{rank}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    w1 = (
        torch.randn(num_local_experts, hidden_dim, dim, device=device, dtype=dtype)
        * 0.02
    )
    w2 = (
        torch.randn(num_local_experts, dim, hidden_dim, device=device, dtype=dtype)
        * 0.02
    )
    w3 = (
        torch.randn(num_local_experts, hidden_dim, dim, device=device, dtype=dtype)
        * 0.02
    )

    x = torch.randn(num_tokens, dim, device=device, dtype=dtype) * 0.1

    # Create expert ids (only native experts for simplicity)
    total_experts = num_local_experts + num_foreign
    native_start = ep_rank * num_local_experts

    if routing == "balanced":
        expert_ids = (
            torch.arange(num_tokens, device=device) % num_local_experts + native_start
        )
    elif routing == "imbalanced":
        # 70% to first 2 experts
        num_hot = int(num_tokens * 0.7)
        expert_ids = torch.zeros(num_tokens, dtype=torch.int64, device=device)
        expert_ids[:num_hot] = torch.arange(num_hot, device=device) % 2 + native_start
        expert_ids[num_hot:] = (
            torch.arange(num_tokens - num_hot, device=device) % num_local_experts
            + native_start
        )
    elif routing == "extreme":
        expert_ids = torch.full(
            (num_tokens,), native_start, dtype=torch.int64, device=device
        )
    else:
        expert_ids = torch.randint(
            native_start, native_start + num_local_experts, (num_tokens,), device=device
        )

    # Foreign weights (stacked tensor interface)
    foreign_w1_stacked = None
    foreign_w2_stacked = None
    foreign_w3_stacked = None
    foreign_expert_id_mapping = None

    if num_foreign > 0:
        foreign_w1_stacked = (
            torch.randn(num_foreign, hidden_dim, dim, device=device, dtype=dtype) * 0.02
        )
        foreign_w2_stacked = (
            torch.randn(num_foreign, dim, hidden_dim, device=device, dtype=dtype) * 0.02
        )
        foreign_w3_stacked = (
            torch.randn(num_foreign, hidden_dim, dim, device=device, dtype=dtype) * 0.02
        )
        foreign_expert_id_mapping = torch.full(
            (total_experts * 2,), -1, dtype=torch.long, device=device
        )
        # Map foreign experts to stacked indices (ids beyond native range)
        for i in range(num_foreign):
            foreign_id = native_start + num_local_experts + i
            foreign_expert_id_mapping[foreign_id] = i
            # Also route some tokens to foreign experts
            num_foreign_tokens = max(1, num_tokens // (num_foreign * 4))
            start_idx = num_tokens - num_foreign_tokens * (num_foreign - i)
            end_idx = min(num_tokens, start_idx + num_foreign_tokens)
            if start_idx < end_idx:
                expert_ids[start_idx:end_idx] = foreign_id

    from torchtitan.distributed.llep import (
        _llep_swiglu_ffn_forloop,
        _llep_swiglu_ffn_grouped_mm,
    )

    # For-loop path (reference)
    out_forloop = _llep_swiglu_ffn_forloop(
        x.clone(),
        expert_ids.clone(),
        w1,
        w2,
        w3,
        None,
        None,
        None,
        ep_rank,
        num_local_experts,
        foreign_w1_stacked=foreign_w1_stacked,
        foreign_w2_stacked=foreign_w2_stacked,
        foreign_w3_stacked=foreign_w3_stacked,
        foreign_expert_id_mapping=foreign_expert_id_mapping,
    )

    # Grouped MM path
    out_grouped = _llep_swiglu_ffn_grouped_mm(
        x.clone(),
        expert_ids.clone(),
        w1,
        w2,
        w3,
        None,
        None,
        None,
        ep_rank,
        num_local_experts,
        foreign_w1_stacked=foreign_w1_stacked,
        foreign_w2_stacked=foreign_w2_stacked,
        foreign_w3_stacked=foreign_w3_stacked,
        foreign_expert_id_mapping=foreign_expert_id_mapping,
    )

    max_diff = (out_forloop - out_grouped).abs().max().item()
    mean_diff = (out_forloop - out_grouped).abs().mean().item()
    return max_diff, mean_diff, atol


def test_grouped_vs_forloop_native_only():
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print(f"\n{'='*70}\n  Grouped vs ForLoop: Native Only\n{'='*70}")
    max_diff, mean_diff, atol = _test_grouped_vs_forloop(
        num_tokens=32, dim=64, hidden_dim=128, num_local_experts=4, routing="balanced"
    )
    if rank == 0:
        print(f"  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")
    assert max_diff < atol, f"max_diff={max_diff:.6f} > {atol}"
    if rank == 0:
        print(f"  PASSED")


def test_grouped_vs_forloop_foreign_stacked():
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print(
            f"\n{'='*70}\n  Grouped vs ForLoop: With Foreign Experts (stacked)\n{'='*70}"
        )
    max_diff, mean_diff, atol = _test_grouped_vs_forloop(
        num_tokens=32,
        dim=64,
        hidden_dim=128,
        num_local_experts=2,
        num_foreign=2,
        routing="balanced",
    )
    if rank == 0:
        print(f"  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")
    assert max_diff < atol, f"max_diff={max_diff:.6f} > {atol}"
    if rank == 0:
        print(f"  PASSED")


def test_grouped_vs_forloop_balanced():
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print(f"\n{'='*70}\n  Grouped vs ForLoop: Balanced\n{'='*70}")
    max_diff, mean_diff, atol = _test_grouped_vs_forloop(routing="balanced")
    if rank == 0:
        print(f"  max_diff={max_diff:.6f}")
    assert max_diff < atol, f"max_diff={max_diff:.6f} > {atol}"
    if rank == 0:
        print(f"  PASSED")


def test_grouped_vs_forloop_imbalanced():
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print(f"\n{'='*70}\n  Grouped vs ForLoop: Imbalanced\n{'='*70}")
    max_diff, mean_diff, atol = _test_grouped_vs_forloop(routing="imbalanced")
    if rank == 0:
        print(f"  max_diff={max_diff:.6f}")
    assert max_diff < atol, f"max_diff={max_diff:.6f} > {atol}"
    if rank == 0:
        print(f"  PASSED")


def test_grouped_vs_forloop_extreme():
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print(f"\n{'='*70}\n  Grouped vs ForLoop: Extreme (all to expert 0)\n{'='*70}")
    max_diff, mean_diff, atol = _test_grouped_vs_forloop(routing="extreme")
    if rank == 0:
        print(f"  max_diff={max_diff:.6f}")
    assert max_diff < atol, f"max_diff={max_diff:.6f} > {atol}"
    if rank == 0:
        print(f"  PASSED")


def test_grouped_vs_forloop_various_dims():
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print(f"\n{'='*70}\n  Grouped vs ForLoop: Various Dims\n{'='*70}")
    for dim, hidden_dim in [(64, 128), (128, 256), (256, 768)]:
        max_diff, mean_diff, atol = _test_grouped_vs_forloop(
            dim=dim, hidden_dim=hidden_dim, num_tokens=32, atol=1e-3
        )
        if rank == 0:
            print(f"  dim={dim}, hidden={hidden_dim}: max_diff={max_diff:.6f}")
        assert max_diff < atol, f"dim={dim}: max_diff={max_diff:.6f} > {atol}"
    if rank == 0:
        print(f"  PASSED")


def test_grouped_vs_forloop_kimi_k2_dims():
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print(
            f"\n{'='*70}\n  Grouped vs ForLoop: Kimi K2 Scale (dim=7168, hidden=2048)\n{'='*70}"
        )
    max_diff, mean_diff, atol = _test_grouped_vs_forloop(
        dim=7168, hidden_dim=2048, num_tokens=64, num_local_experts=6, atol=1e-2
    )
    if rank == 0:
        print(f"  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")
    assert max_diff < atol, f"max_diff={max_diff:.6f} > {atol}"
    if rank == 0:
        print(f"  PASSED")


# ---------------------------------------------------------------------------
# Category 2: Triton Kernel Unit Tests (single-process)
# ---------------------------------------------------------------------------
def test_fused_silu_gate_vs_pytorch():
    rank = dist.get_rank() if dist.is_initialized() else 0
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"\n{'='*70}\n  Triton: fused_silu_gate vs PyTorch\n{'='*70}")

    from torchtitan.distributed.llep_kernels import fused_silu_gate

    torch.manual_seed(42)
    x1 = torch.randn(256, 2048, device=device, dtype=torch.bfloat16)
    x3 = torch.randn(256, 2048, device=device, dtype=torch.bfloat16)

    ref = F.silu(x1.float()) * x3.float()
    ref = ref.to(torch.bfloat16)

    out = fused_silu_gate(x1, x3)
    max_diff = (out - ref).abs().max().item()

    if rank == 0:
        print(f"  max_diff={max_diff:.6f}")
    assert max_diff < 1e-3, f"max_diff={max_diff:.6f}"
    if rank == 0:
        print(f"  PASSED")


def test_fused_silu_gate_backward():
    rank = dist.get_rank() if dist.is_initialized() else 0
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"\n{'='*70}\n  Triton: fused_silu_gate backward\n{'='*70}")

    from torchtitan.distributed.llep_kernels import fused_silu_gate

    torch.manual_seed(42)
    x1 = torch.randn(32, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    x3 = torch.randn(32, 128, device=device, dtype=torch.bfloat16, requires_grad=True)

    x1_ref = x1.detach().clone().requires_grad_(True)
    x3_ref = x3.detach().clone().requires_grad_(True)

    # Forward
    out_fused = fused_silu_gate(x1, x3)
    out_ref = (F.silu(x1_ref.float()) * x3_ref.float()).to(torch.bfloat16)

    # Backward
    grad_out = torch.randn_like(out_fused)
    out_fused.backward(grad_out)
    out_ref.backward(grad_out.clone())

    grad_x1_diff = (x1.grad - x1_ref.grad).abs().max().item()
    grad_x3_diff = (x3.grad - x3_ref.grad).abs().max().item()

    if rank == 0:
        print(f"  grad_x1 max_diff={grad_x1_diff:.6f}")
        print(f"  grad_x3 max_diff={grad_x3_diff:.6f}")

    assert grad_x1_diff < 2e-2, f"grad_x1 diff={grad_x1_diff:.6f}"
    assert grad_x3_diff < 2e-2, f"grad_x3 diff={grad_x3_diff:.6f}"

    if rank == 0:
        print(f"  PASSED")


def test_fused_silu_gate_various_shapes():
    rank = dist.get_rank() if dist.is_initialized() else 0
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"\n{'='*70}\n  Triton: fused_silu_gate various shapes\n{'='*70}")

    from torchtitan.distributed.llep_kernels import fused_silu_gate

    for num_tokens, hidden_size in [
        (32, 2048),
        (256, 2048),
        (1024, 2048),
        (4096, 2048),
    ]:
        torch.manual_seed(42)
        x1 = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
        x3 = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)

        ref = (F.silu(x1.float()) * x3.float()).to(torch.bfloat16)
        out = fused_silu_gate(x1, x3)
        max_diff = (out - ref).abs().max().item()

        if rank == 0:
            print(f"  ({num_tokens}, {hidden_size}): max_diff={max_diff:.6f}")
        assert (
            max_diff < 1e-3
        ), f"shape=({num_tokens},{hidden_size}): max_diff={max_diff:.6f}"

    if rank == 0:
        print(f"  PASSED")


def test_fused_silu_gate_determinism():
    rank = dist.get_rank() if dist.is_initialized() else 0
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"\n{'='*70}\n  Triton: fused_silu_gate determinism\n{'='*70}")

    from torchtitan.distributed.llep_kernels import fused_silu_gate

    torch.manual_seed(42)
    x1 = torch.randn(256, 2048, device=device, dtype=torch.bfloat16)
    x3 = torch.randn(256, 2048, device=device, dtype=torch.bfloat16)

    outputs = [fused_silu_gate(x1.clone(), x3.clone()) for _ in range(5)]
    all_match = all(torch.equal(outputs[0], outputs[i]) for i in range(1, 5))

    if rank == 0:
        print(f"  5 runs bitwise identical: {all_match}")
    assert all_match, "Determinism check failed"
    if rank == 0:
        print(f"  PASSED")


# ---------------------------------------------------------------------------
# Category 3: Numerical Stability
# ---------------------------------------------------------------------------
def test_numerical_bf16_vs_fp32():
    rank = dist.get_rank() if dist.is_initialized() else 0
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"\n{'='*70}\n  Numerical: bf16 vs fp32 reference\n{'='*70}")

    from torchtitan.distributed.llep import _llep_swiglu_ffn_forloop

    torch.manual_seed(42)
    dim, hidden_dim, num_local_experts, num_tokens = 64, 128, 4, 32

    w1 = (
        torch.randn(
            num_local_experts, hidden_dim, dim, device=device, dtype=torch.float32
        )
        * 0.02
    )
    w2 = (
        torch.randn(
            num_local_experts, dim, hidden_dim, device=device, dtype=torch.float32
        )
        * 0.02
    )
    w3 = (
        torch.randn(
            num_local_experts, hidden_dim, dim, device=device, dtype=torch.float32
        )
        * 0.02
    )
    x = torch.randn(num_tokens, dim, device=device, dtype=torch.float32) * 0.1
    expert_ids = torch.arange(num_tokens, device=device) % num_local_experts

    # fp32 reference
    out_fp32 = _llep_swiglu_ffn_forloop(
        x.clone(),
        expert_ids,
        w1,
        w2,
        w3,
        None,
        None,
        None,
        0,
        num_local_experts,
    )

    # bf16
    out_bf16 = _llep_swiglu_ffn_forloop(
        x.bfloat16(),
        expert_ids,
        w1.bfloat16(),
        w2.bfloat16(),
        w3.bfloat16(),
        None,
        None,
        None,
        0,
        num_local_experts,
    )

    max_diff = (out_fp32 - out_bf16.float()).abs().max().item()
    if rank == 0:
        print(f"  bf16 vs fp32: max_diff={max_diff:.6f}")
    assert max_diff < 0.05, f"bf16 vs fp32 too large: {max_diff:.6f}"
    if rank == 0:
        print(f"  PASSED")


def test_numerical_large_values():
    rank = dist.get_rank() if dist.is_initialized() else 0
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"\n{'='*70}\n  Numerical: Large input values\n{'='*70}")

    from torchtitan.distributed.llep import _llep_swiglu_ffn_forloop

    torch.manual_seed(42)
    dim, hidden_dim, num_local_experts, num_tokens = 64, 128, 4, 32

    w1 = (
        torch.randn(
            num_local_experts, hidden_dim, dim, device=device, dtype=torch.bfloat16
        )
        * 0.02
    )
    w2 = (
        torch.randn(
            num_local_experts, dim, hidden_dim, device=device, dtype=torch.bfloat16
        )
        * 0.02
    )
    w3 = (
        torch.randn(
            num_local_experts, hidden_dim, dim, device=device, dtype=torch.bfloat16
        )
        * 0.02
    )
    x = (
        torch.randn(num_tokens, dim, device=device, dtype=torch.bfloat16) * 10.0
    )  # Large values
    expert_ids = torch.arange(num_tokens, device=device) % num_local_experts

    out = _llep_swiglu_ffn_forloop(
        x,
        expert_ids,
        w1,
        w2,
        w3,
        None,
        None,
        None,
        0,
        num_local_experts,
    )

    has_nan = torch.isnan(out).any().item()
    has_inf = torch.isinf(out).any().item()
    if rank == 0:
        print(f"  NaN: {has_nan}, Inf: {has_inf}")
    assert not has_nan, "Output contains NaN"
    assert not has_inf, "Output contains Inf"
    if rank == 0:
        print(f"  PASSED")


def test_numerical_near_zero():
    rank = dist.get_rank() if dist.is_initialized() else 0
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"\n{'='*70}\n  Numerical: Near-zero input values\n{'='*70}")

    from torchtitan.distributed.llep import _llep_swiglu_ffn_forloop

    torch.manual_seed(42)
    dim, hidden_dim, num_local_experts, num_tokens = 64, 128, 4, 32

    w1 = (
        torch.randn(
            num_local_experts, hidden_dim, dim, device=device, dtype=torch.bfloat16
        )
        * 0.02
    )
    w2 = (
        torch.randn(
            num_local_experts, dim, hidden_dim, device=device, dtype=torch.bfloat16
        )
        * 0.02
    )
    w3 = (
        torch.randn(
            num_local_experts, hidden_dim, dim, device=device, dtype=torch.bfloat16
        )
        * 0.02
    )
    x = torch.randn(num_tokens, dim, device=device, dtype=torch.bfloat16) * 1e-6
    expert_ids = torch.arange(num_tokens, device=device) % num_local_experts

    out = _llep_swiglu_ffn_forloop(
        x,
        expert_ids,
        w1,
        w2,
        w3,
        None,
        None,
        None,
        0,
        num_local_experts,
    )

    has_nan = torch.isnan(out).any().item()
    if rank == 0:
        print(f"  NaN: {has_nan}")
    assert not has_nan, "Output contains NaN for near-zero input"
    if rank == 0:
        print(f"  PASSED")


def test_numerical_fp32_precision():
    rank = dist.get_rank() if dist.is_initialized() else 0
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"\n{'='*70}\n  Numerical: fp32 precision\n{'='*70}")

    from torchtitan.distributed.llep import (
        _llep_swiglu_ffn_forloop,
        _llep_swiglu_ffn_grouped_mm,
    )

    torch.manual_seed(42)
    dim, hidden_dim, num_local_experts, num_tokens = 64, 128, 4, 32

    w1 = (
        torch.randn(
            num_local_experts, hidden_dim, dim, device=device, dtype=torch.float32
        )
        * 0.02
    )
    w2 = (
        torch.randn(
            num_local_experts, dim, hidden_dim, device=device, dtype=torch.float32
        )
        * 0.02
    )
    w3 = (
        torch.randn(
            num_local_experts, hidden_dim, dim, device=device, dtype=torch.float32
        )
        * 0.02
    )
    x = torch.randn(num_tokens, dim, device=device, dtype=torch.float32) * 0.1
    expert_ids = torch.arange(num_tokens, device=device) % num_local_experts

    out_forloop = _llep_swiglu_ffn_forloop(
        x.clone(),
        expert_ids,
        w1,
        w2,
        w3,
        None,
        None,
        None,
        0,
        num_local_experts,
    )

    # Manually compute reference
    out_ref = torch.empty_like(x)
    for i in range(num_tokens):
        eid = expert_ids[i].item()
        xi = x[i : i + 1]
        h = F.silu(xi @ w1[eid].T) * (xi @ w3[eid].T)
        out_ref[i : i + 1] = h @ w2[eid].T

    max_diff = (out_forloop - out_ref).abs().max().item()
    if rank == 0:
        print(f"  fp32 forloop vs manual: max_diff={max_diff:.8f}")
    assert max_diff < 1e-5, f"fp32 precision: max_diff={max_diff:.8f}"
    if rank == 0:
        print(f"  PASSED")


# ---------------------------------------------------------------------------
# Category 4: Performance Benchmarks
# ---------------------------------------------------------------------------
def test_benchmark_ffn_grouped_vs_forloop():
    rank = dist.get_rank() if dist.is_initialized() else 0
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"\n{'='*70}\n  Benchmark: Grouped MM vs For-Loop FFN\n{'='*70}")

    from torchtitan.distributed.llep import (
        _llep_swiglu_ffn_forloop,
        _llep_swiglu_ffn_grouped_mm,
    )

    configs = [
        (256, 8, 2048, 64),
        (1024, 16, 2048, 128),
    ]

    for num_tokens, num_experts, hidden_dim, dim in configs:
        torch.manual_seed(42)
        w1 = (
            torch.randn(
                num_experts, hidden_dim, dim, device=device, dtype=torch.bfloat16
            )
            * 0.02
        )
        w2 = (
            torch.randn(
                num_experts, dim, hidden_dim, device=device, dtype=torch.bfloat16
            )
            * 0.02
        )
        w3 = (
            torch.randn(
                num_experts, hidden_dim, dim, device=device, dtype=torch.bfloat16
            )
            * 0.02
        )
        x = torch.randn(num_tokens, dim, device=device, dtype=torch.bfloat16) * 0.1
        expert_ids = torch.arange(num_tokens, device=device) % num_experts

        # Warmup
        for _ in range(3):
            _llep_swiglu_ffn_forloop(
                x, expert_ids, w1, w2, w3, None, None, None, 0, num_experts
            )
            _llep_swiglu_ffn_grouped_mm(
                x, expert_ids, w1, w2, w3, None, None, None, 0, num_experts
            )
        torch.cuda.synchronize()

        # Benchmark forloop
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times_forloop = []
        for _ in range(10):
            start.record()
            _llep_swiglu_ffn_forloop(
                x, expert_ids, w1, w2, w3, None, None, None, 0, num_experts
            )
            end.record()
            torch.cuda.synchronize()
            times_forloop.append(start.elapsed_time(end))

        # Benchmark grouped
        times_grouped = []
        for _ in range(10):
            start.record()
            _llep_swiglu_ffn_grouped_mm(
                x, expert_ids, w1, w2, w3, None, None, None, 0, num_experts
            )
            end.record()
            torch.cuda.synchronize()
            times_grouped.append(start.elapsed_time(end))

        median_forloop = sorted(times_forloop)[5]
        median_grouped = sorted(times_grouped)[5]
        speedup = median_forloop / max(median_grouped, 1e-6)

        if rank == 0:
            print(
                f"  [{num_tokens}tok x {num_experts}exp x {hidden_dim}h] "
                f"forloop={median_forloop:.3f}ms  grouped={median_grouped:.3f}ms  "
                f"speedup={speedup:.2f}x"
            )

    if rank == 0:
        print(f"  DONE (benchmark only, no assertions)")


def test_benchmark_triton_silu_gate():
    rank = dist.get_rank() if dist.is_initialized() else 0
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"\n{'='*70}\n  Benchmark: Triton fused_silu_gate vs F.silu*x3\n{'='*70}")

    from torchtitan.distributed.llep_kernels import fused_silu_gate

    configs = [(256, 2048), (1024, 2048), (4096, 2048)]

    for num_tokens, hidden_size in configs:
        torch.manual_seed(42)
        x1 = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
        x3 = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)

        # Warmup
        for _ in range(3):
            F.silu(x1) * x3
            fused_silu_gate(x1, x3)
        torch.cuda.synchronize()

        # Benchmark unfused
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times_unfused = []
        for _ in range(20):
            start.record()
            _ = F.silu(x1) * x3
            end.record()
            torch.cuda.synchronize()
            times_unfused.append(start.elapsed_time(end))

        # Benchmark fused
        times_fused = []
        for _ in range(20):
            start.record()
            _ = fused_silu_gate(x1, x3)
            end.record()
            torch.cuda.synchronize()
            times_fused.append(start.elapsed_time(end))

        median_unfused = sorted(times_unfused)[10]
        median_fused = sorted(times_fused)[10]
        speedup = median_unfused / max(median_fused, 1e-6)

        if rank == 0:
            print(
                f"  ({num_tokens}, {hidden_size}): "
                f"unfused={median_unfused:.4f}ms  fused={median_fused:.4f}ms  "
                f"speedup={speedup:.2f}x"
            )

    if rank == 0:
        print(f"  DONE (benchmark only, no assertions)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
TEST_REGISTRY = {
    # Category 1: Grouped MM vs For-Loop (single-process)
    "grouped_vs_forloop_native": test_grouped_vs_forloop_native_only,
    "grouped_vs_forloop_foreign": test_grouped_vs_forloop_foreign_stacked,
    "grouped_vs_forloop_balanced": test_grouped_vs_forloop_balanced,
    "grouped_vs_forloop_imbalanced": test_grouped_vs_forloop_imbalanced,
    "grouped_vs_forloop_extreme": test_grouped_vs_forloop_extreme,
    "grouped_vs_forloop_dims": test_grouped_vs_forloop_various_dims,
    "grouped_vs_forloop_kimi_k2": test_grouped_vs_forloop_kimi_k2_dims,
    # Category 2: Triton kernel unit tests
    "fused_silu_gate": test_fused_silu_gate_vs_pytorch,
    "fused_silu_gate_backward": test_fused_silu_gate_backward,
    "fused_silu_gate_shapes": test_fused_silu_gate_various_shapes,
    "fused_silu_gate_determinism": test_fused_silu_gate_determinism,
    # Category 3: Numerical stability
    "numerical_bf16_vs_fp32": test_numerical_bf16_vs_fp32,
    "numerical_large_values": test_numerical_large_values,
    "numerical_near_zero": test_numerical_near_zero,
    "numerical_fp32_precision": test_numerical_fp32_precision,
    # Category 4: Benchmarks
    "benchmark_ffn": test_benchmark_ffn_grouped_vs_forloop,
    "benchmark_triton": test_benchmark_triton_silu_gate,
}


def main():
    parser = argparse.ArgumentParser(description="LLEP Correctness Tests")
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Run a specific test (e.g., 'grouped_vs_forloop_native'). If not specified, run all.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available tests and exit."
    )
    args = parser.parse_args()

    if args.list:
        print("Available tests:")
        for name in TEST_REGISTRY:
            print(f"  {name}")
        sys.exit(0)

    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"\n{'#'*70}")
        print(f"  LLEP Correctness Test Suite")
        print(f"  world_size={world_size}")
        print(f"{'#'*70}")

    if args.test:
        if args.test not in TEST_REGISTRY:
            if rank == 0:
                print(f"Unknown test: {args.test}")
                print(f"Available: {list(TEST_REGISTRY.keys())}")
            dist.destroy_process_group()
            sys.exit(1)
        tests = {args.test: TEST_REGISTRY[args.test]}
    else:
        tests = TEST_REGISTRY

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in tests.items():
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            if rank == 0:
                print(f"\n  FAILED: {name}")
                traceback.print_exc()
            # Continue to next test
            dist.barrier()

    if rank == 0:
        print(f"\n{'#'*70}")
        print(
            f"  Results: {passed} passed, {failed} failed out of {passed + failed} tests"
        )
        if errors:
            print(f"\n  Failed tests:")
            for name, err in errors:
                print(f"    {name}: {err}")
        print(f"{'#'*70}\n")

    dist.destroy_process_group()

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
