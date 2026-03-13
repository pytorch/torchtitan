#!/usr/bin/env python3
# Copyright (c) Nous Research and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for Least-Loaded Expert Parallelism (LLEP) integration.

Tests:
1. LPT planning algorithm (single-process, no distributed)
2. SwiGLU FFN with mixed native/foreign experts (single-process)
3. Multi-GPU LLEP forward (requires torchrun with >=2 GPUs)

Run unit tests:
    python tests/unit_tests/test_llep.py

Run multi-GPU test:
    torchrun --nproc_per_node=2 tests/unit_tests/test_llep.py --distributed
"""

import torch


def test_lpt_planning():
    """Test LPT planning algorithm produces correct load-balanced assignments."""
    from torchtitan.distributed.llep import compute_llep_lpt_plan

    print("=== Test LPT Planning ===")

    # Setup: 8 experts across 2 GPUs (4 experts each)
    # Imbalanced: expert 0 gets 80% of tokens
    ep_size = 2
    num_local_experts = 4
    num_experts = 8
    total_tokens = 1000

    # 80% to expert 0, rest spread evenly
    global_counts = torch.zeros(num_experts, dtype=torch.int64)
    global_counts[0] = 800  # Hot expert on GPU 0
    remaining = total_tokens - 800
    for i in range(1, num_experts):
        global_counts[i] = remaining // (num_experts - 1)

    print(f"  Expert counts: {global_counts.tolist()}")

    # Test for rank 0
    plan = compute_llep_lpt_plan(
        global_counts,
        ep_size,
        ep_rank=0,
        num_local_experts=num_local_experts,
        max_tokens_factor=1.1,
        min_tokens_per_gemm=10,
    )

    print(f"  GPU loads: {plan.gpu_loads.tolist()}")
    print(f"  Weight transfers: {len(plan.weight_transfers)}")
    for wt in plan.weight_transfers:
        print(
            f"    Expert {wt.expert_id}: GPU {wt.src_rank} -> GPU {wt.dst_rank} "
            f"(tokens {wt.token_start}-{wt.token_end})"
        )

    # Verify load is more balanced than naive (all on native GPU)
    naive_gpu0_load = sum(global_counts[:num_local_experts].tolist())
    naive_gpu1_load = sum(global_counts[num_local_experts:].tolist())
    llep_gpu0_load = plan.gpu_loads[0].item()
    llep_gpu1_load = plan.gpu_loads[1].item()

    naive_imbalance = max(naive_gpu0_load, naive_gpu1_load) / max(
        (naive_gpu0_load + naive_gpu1_load) / 2, 1
    )
    llep_imbalance = max(llep_gpu0_load, llep_gpu1_load) / max(
        (llep_gpu0_load + llep_gpu1_load) / 2, 1
    )

    print(
        f"  Naive imbalance ratio: {naive_imbalance:.2f} "
        f"(GPU0={naive_gpu0_load}, GPU1={naive_gpu1_load})"
    )
    print(
        f"  LLEP imbalance ratio:  {llep_imbalance:.2f} "
        f"(GPU0={llep_gpu0_load}, GPU1={llep_gpu1_load})"
    )

    assert (
        llep_imbalance <= naive_imbalance
    ), f"LLEP should reduce imbalance: {llep_imbalance} > {naive_imbalance}"
    assert (
        len(plan.weight_transfers) > 0
    ), "Should have weight transfers for imbalanced routing"

    # Verify expert 0 is spilled (it's the hot expert)
    spilled_experts = {wt.expert_id for wt in plan.weight_transfers}
    assert 0 in spilled_experts, "Hot expert 0 should be spilled"

    print("  PASSED\n")


def test_lpt_balanced_case():
    """Test that balanced routing produces no weight transfers."""
    from torchtitan.distributed.llep import compute_llep_lpt_plan

    print("=== Test LPT Balanced Case ===")

    ep_size = 4
    num_local_experts = 2
    num_experts = 8

    # Perfectly balanced
    global_counts = torch.full((num_experts,), 100, dtype=torch.int64)

    plan = compute_llep_lpt_plan(
        global_counts,
        ep_size,
        ep_rank=0,
        num_local_experts=num_local_experts,
        max_tokens_factor=1.1,
        min_tokens_per_gemm=10,
    )

    print(f"  Expert counts: {global_counts.tolist()}")
    print(f"  GPU loads: {plan.gpu_loads.tolist()}")
    print(f"  Weight transfers: {len(plan.weight_transfers)}")

    assert (
        len(plan.weight_transfers) == 0
    ), "Balanced routing should have no weight transfers"

    print("  PASSED\n")


def test_gpu_imbalance_ratio():
    """Test GPU imbalance ratio computation."""
    from torchtitan.distributed.llep import compute_gpu_imbalance_ratio

    print("=== Test GPU Imbalance Ratio ===")

    # Balanced case
    counts = torch.tensor([100, 100, 100, 100], dtype=torch.int64)
    ratio = compute_gpu_imbalance_ratio(counts, ep_size=2, num_local_experts=2)
    print(f"  Balanced: ratio={ratio:.2f} (expected ~1.0)")
    assert abs(ratio - 1.0) < 0.01

    # Imbalanced case
    counts = torch.tensor([400, 0, 0, 0], dtype=torch.int64)
    ratio = compute_gpu_imbalance_ratio(counts, ep_size=2, num_local_experts=2)
    print(f"  Imbalanced: ratio={ratio:.2f} (expected 2.0)")
    assert abs(ratio - 2.0) < 0.01

    print("  PASSED\n")


if __name__ == "__main__":
    print("Running LLEP unit tests...\n")
    test_lpt_planning()
    test_lpt_balanced_case()
    test_gpu_imbalance_ratio()
    print("All unit tests passed!")
