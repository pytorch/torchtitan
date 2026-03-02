#!/usr/bin/env python3
# Copyright (c) Nous Research and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Profile every function in llep_moe_forward at training sizes.

Training config (mini_kimi_k2_llep_ep8):
  dim=3072, moe_inter_dim=2048, num_experts=256, top_k=8
  EP=8, num_local=32, seq_len=8192, lbs=6
  tokens_per_rank = 49152, N = tokens*top_k = 393216
"""

import time

import numpy as np
import torch

torch.manual_seed(42)

# Training-size constants
DIM = 3072
HIDDEN_DIM = 2048
NUM_EXPERTS = 256
EP_SIZE = 8
NUM_LOCAL = NUM_EXPERTS // EP_SIZE  # 32
TOP_K = 8
TOKENS_PER_RANK = 49152  # 8192 * 6
N = TOKENS_PER_RANK * TOP_K  # 393216
EP_RANK = 0
MAX_TOKENS_FACTOR = 1.1
MIN_TOKENS_PER_GEMM = 1024

DEVICE = "cuda"
WARMUP = 3
ITERS = 10


def timer(fn, warmup=WARMUP, iters=ITERS, label=""):
    """Time a function with warmup and sync."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / iters * 1000  # ms
    print(f"  {label:45s} {elapsed:8.3f} ms")
    return elapsed


def create_imbalanced_counts(num_experts, total_tokens, skew=2.0):
    """Create realistic imbalanced expert counts (zipf-like)."""
    rng = np.random.default_rng(42)
    weights = 1.0 / np.arange(1, num_experts + 1) ** skew
    weights /= weights.sum()
    counts = (weights * total_tokens).astype(np.int64)
    # Fix rounding
    diff = total_tokens - counts.sum()
    counts[0] += diff
    return torch.tensor(counts, dtype=torch.int64, device=DEVICE)


def main():
    import sys

    sys.path.insert(
        0, "/home/phuc/workspace/moe/small_prs/pr008_saleforce_lbs/torchtitan"
    )
    from torchtitan.distributed.llep import (
        _pack_expert_weights,
        _pad_for_grouped_mm,
        _unpad_output,
        assign_tokens_to_gpus,
        compute_gpu_imbalance_ratio,
        compute_llep_lpt_plan,
        llep_swiglu_ffn,
    )
    from torchtitan.distributed.llep_kernels import (
        fused_silu_gate,
        triton_assign_tokens,
        triton_imbalance_ratio,
    )

    print(f"=" * 70)
    print(f"LLEP Component Profiling at Training Sizes")
    print(f"  dim={DIM}, hidden={HIDDEN_DIM}, experts={NUM_EXPERTS}, top_k={TOP_K}")
    print(f"  EP={EP_SIZE}, local={NUM_LOCAL}, tokens={TOKENS_PER_RANK}, N={N}")
    print(f"=" * 70)

    # ---- Create test data ----
    total_global_tokens = N * EP_SIZE  # total across all ranks
    global_expert_counts = create_imbalanced_counts(NUM_EXPERTS, total_global_tokens)
    local_expert_counts = create_imbalanced_counts(NUM_EXPERTS, N)

    # Per-rank counts (simulate 8 ranks)
    all_expert_counts = []
    remaining = global_expert_counts.clone()
    for r in range(EP_SIZE):
        if r < EP_SIZE - 1:
            rank_counts = (remaining / (EP_SIZE - r)).to(torch.int64)
            rank_counts = rank_counts.clamp(min=0)
        else:
            rank_counts = remaining
        all_expert_counts.append(rank_counts.clone())
        remaining -= rank_counts
    # Fix: use actual local counts for rank 0
    all_expert_counts[0] = local_expert_counts.clone()
    # Recompute global
    global_expert_counts = torch.stack(all_expert_counts).sum(0)

    # Token data
    selected_experts = torch.randint(
        0, NUM_EXPERTS, (TOKENS_PER_RANK, TOP_K), device=DEVICE
    )
    top_scores = torch.randn(
        TOKENS_PER_RANK, TOP_K, device=DEVICE, dtype=torch.bfloat16
    ).abs()
    hidden_states = torch.randn(
        TOKENS_PER_RANK, DIM, device=DEVICE, dtype=torch.bfloat16
    )

    # Expert weights
    w1_local = (
        torch.randn(NUM_LOCAL, HIDDEN_DIM, DIM, device=DEVICE, dtype=torch.bfloat16)
        * 0.01
    )
    w2_local = (
        torch.randn(NUM_LOCAL, DIM, HIDDEN_DIM, device=DEVICE, dtype=torch.bfloat16)
        * 0.01
    )
    w3_local = (
        torch.randn(NUM_LOCAL, HIDDEN_DIM, DIM, device=DEVICE, dtype=torch.bfloat16)
        * 0.01
    )

    # ============================================================
    # 1. compute_gpu_imbalance_ratio
    # ============================================================
    print(f"\n--- 1. compute_gpu_imbalance_ratio ---")

    # PyTorch fallback
    def imb_pytorch():
        eff = EP_SIZE * NUM_LOCAL
        c = global_expert_counts[:eff]
        gpu_loads = c.view(EP_SIZE, NUM_LOCAL).sum(dim=1).float()
        mean_load = gpu_loads.mean()
        return (gpu_loads.max() / mean_load).item()

    timer(imb_pytorch, label="PyTorch (view+sum+max/mean)")
    timer(
        lambda: triton_imbalance_ratio(global_expert_counts, EP_SIZE, NUM_LOCAL),
        label="Triton kernel",
    )

    # ============================================================
    # 2. compute_llep_lpt_plan
    # ============================================================
    print(f"\n--- 2. compute_llep_lpt_plan ---")

    def lpt_plan_fn():
        return compute_llep_lpt_plan(
            global_expert_counts,
            EP_SIZE,
            EP_RANK,
            NUM_LOCAL,
            MAX_TOKENS_FACTOR,
            MIN_TOKENS_PER_GEMM,
        )

    plan_time = timer(lpt_plan_fn, label="compute_llep_lpt_plan (full)")

    # Break down: D2H sync vs CPU algorithm
    def lpt_d2h():
        return global_expert_counts.cpu().tolist()

    timer(lpt_d2h, label="  D2H: .cpu().tolist()")

    counts_cpu = global_expert_counts.cpu().tolist()

    def lpt_cpu_only():
        # Just the CPU algorithm part
        total_tokens = sum(counts_cpu)
        balanced = total_tokens // EP_SIZE
        max_per_gpu = int(MAX_TOKENS_FACTOR * balanced)
        native_load = [0] * EP_SIZE
        for eid in range(NUM_EXPERTS):
            native_load[eid // NUM_LOCAL] += counts_cpu[eid]
        pending = list(native_load)
        assigned = [0] * EP_SIZE
        sorted_experts = sorted(enumerate(counts_cpu), key=lambda x: -x[1])
        plan = {}
        for eid, etoks in sorted_experts:
            if etoks == 0:
                continue
            native = eid // NUM_LOCAL
            pending[native] -= etoks
            eff_load = assigned[native] + pending[native]
            avail = max_per_gpu - eff_load
            if avail >= etoks:
                plan[eid] = [(native, 0, etoks)]
                assigned[native] += etoks
            else:
                plan[eid] = [(native, 0, min(avail, etoks))]
                assigned[native] += min(avail, etoks)
        return plan

    timer(lpt_cpu_only, label="  CPU algorithm only (Python)")

    # Get a plan for subsequent tests
    plan = lpt_plan_fn()

    # ============================================================
    # 3. assign_tokens_to_gpus
    # ============================================================
    print(f"\n--- 3. assign_tokens_to_gpus ---")

    def assign_fn():
        return assign_tokens_to_gpus(
            top_scores,
            selected_experts,
            plan.lpt_plan,
            EP_SIZE,
            EP_RANK,
            NUM_LOCAL,
            all_expert_counts,
            local_expert_counts,
        )

    timer(assign_fn, label="assign_tokens_to_gpus (full)")

    # Break down sub-components
    flat_experts = selected_experts.view(-1)
    flat_scores = top_scores.view(-1)

    # D2H batched sync
    def d2h_batch():
        return torch.stack(all_expert_counts).cpu().numpy()

    timer(d2h_batch, label="  D2H: stack+cpu+numpy")

    # Triton assign kernel only
    all_counts_np = torch.stack(all_expert_counts).cpu().numpy()
    cum_counts_np = np.zeros((EP_SIZE + 1, NUM_EXPERTS), dtype=np.int64)
    cum_counts_np[1:] = np.cumsum(all_counts_np, axis=0)
    global_offsets_np = cum_counts_np[EP_RANK]
    global_offsets_gpu = torch.from_numpy(global_offsets_np).to(DEVICE)

    def triton_assign_only():
        return triton_assign_tokens(
            flat_experts,
            local_expert_counts,
            global_offsets_gpu,
            plan.lpt_plan,
            NUM_LOCAL,
            NUM_EXPERTS,
        )

    timer(triton_assign_only, label="  Triton assign kernel only")

    # Sort by target GPU
    result = assign_fn()
    sorted_scores, sorted_experts_out, isp, osp, sorted_idx, undo_idx = result

    def sort_phase():
        target_gpus = flat_experts // NUM_LOCAL
        si = torch.argsort(target_gpus, stable=True)
        ui = torch.argsort(si)
        return flat_scores[si], flat_experts[si]

    timer(sort_phase, label="  argsort (target GPU sort)")

    # send_matrix computation
    def send_matrix_fn():
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
                    overlap_start = max(src_start, dst_start)
                    overlap_end = min(src_end, dst_end)
                    if overlap_start < overlap_end:
                        send_matrix_np[src_rank, dst_gpu] += overlap_end - overlap_start
        return send_matrix_np

    timer(send_matrix_fn, label="  send_matrix (numpy loops)")

    # ============================================================
    # 4. _pack_expert_weights
    # ============================================================
    print(f"\n--- 4. _pack_expert_weights ---")

    # Create realistic expert_ids for received tokens
    recv_expert_ids = torch.randint(
        0, NUM_EXPERTS, (N,), device=DEVICE, dtype=torch.int64
    )
    sorted_eids, sort_perm = recv_expert_ids.sort(stable=True)
    unique_experts, counts = torch.unique_consecutive(sorted_eids, return_counts=True)

    def pack_fn():
        return _pack_expert_weights(
            unique_experts,
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
            EP_RANK,
            NUM_LOCAL,
        )

    timer(pack_fn, label="_pack_expert_weights (native only)")

    # ============================================================
    # 5. _pad_for_grouped_mm
    # ============================================================
    print(f"\n--- 5. _pad_for_grouped_mm ---")

    x_sorted = torch.randn(N, DIM, device=DEVICE, dtype=torch.bfloat16)
    # Realistic counts: ~256 experts, each ~N/256 = 1536 tokens avg
    counts_test = torch.zeros(NUM_EXPERTS, dtype=torch.int64, device=DEVICE)
    for i in range(NUM_EXPERTS):
        counts_test[i] = max(
            1, N // NUM_EXPERTS + torch.randint(-500, 500, (1,)).item()
        )
    # Fix total
    counts_test[-1] += N - counts_test.sum()
    counts_test = counts_test.clamp(min=0)
    actual_total = counts_test.sum().item()
    x_sorted_test = torch.randn(actual_total, DIM, device=DEVICE, dtype=torch.bfloat16)

    def pad_fn():
        return _pad_for_grouped_mm(x_sorted_test, counts_test)

    timer(pad_fn, label="_pad_for_grouped_mm (256 experts)")

    # ============================================================
    # 6. _unpad_output
    # ============================================================
    print(f"\n--- 6. _unpad_output ---")

    x_padded, counts_padded = _pad_for_grouped_mm(x_sorted_test, counts_test)

    def unpad_fn():
        return _unpad_output(x_padded, counts_test, counts_padded)

    timer(unpad_fn, label="_unpad_output (256 experts)")

    # ============================================================
    # 7. fused_silu_gate
    # ============================================================
    print(f"\n--- 7. Activation (silu_gate) ---")

    x1_act = torch.randn(N, HIDDEN_DIM, device=DEVICE, dtype=torch.bfloat16)
    x3_act = torch.randn(N, HIDDEN_DIM, device=DEVICE, dtype=torch.bfloat16)

    def silu_pytorch():
        return torch.nn.functional.silu(x1_act) * x3_act

    def silu_triton():
        return fused_silu_gate(x1_act, x3_act)

    timer(silu_pytorch, label="PyTorch F.silu(x1) * x3")
    timer(silu_triton, label="Triton fused_silu_gate")

    # ============================================================
    # 8. Full SwiGLU FFN (grouped_mm path)
    # ============================================================
    print(f"\n--- 8. Full SwiGLU FFN ---")

    # Use a subset to avoid OOM (grouped_mm needs padded buffers)
    SUBSET = min(N, 50000)
    x_ffn = torch.randn(SUBSET, DIM, device=DEVICE, dtype=torch.bfloat16)
    eid_ffn = torch.randint(
        EP_RANK * NUM_LOCAL, (EP_RANK + 1) * NUM_LOCAL, (SUBSET,), device=DEVICE
    )

    def ffn_grouped():
        return llep_swiglu_ffn(
            x_ffn,
            eid_ffn,
            w1_local,
            w2_local,
            w3_local,
            None,
            None,
            None,
            EP_RANK,
            NUM_LOCAL,
        )

    timer(ffn_grouped, label="llep_swiglu_ffn (grouped_mm, 50K tokens)")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"Profile complete. Focus optimization on the slowest components.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
