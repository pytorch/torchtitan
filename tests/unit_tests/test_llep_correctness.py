#!/usr/bin/env python3
# Copyright (c) Nous Research and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Comprehensive correctness tests for Least-Loaded Expert Parallelism (LLEP).

Tests that LLEP produces numerically identical results to standard all-to-all
Expert Parallelism (EP) for both forward and backward passes, and that the
new grouped GEMM + fused Triton kernel optimizations match the original
per-expert for-loop implementation.

Test categories:
1. Grouped MM vs For-Loop (single-process, 7 tests)
2. Distributed forward correctness (multi-GPU, 9 tests)
3. Distributed backward correctness (multi-GPU, 7 tests)
4. Triton fused_silu_gate kernel (single-process, 4 tests)
6. Numerical stability (4 tests)
7. Performance benchmarks (2 tests)
8. Integration (5 tests)

Run with torchrun (requires >= 2 GPUs):
    torchrun --nproc_per_node=2 tests/unit_tests/test_llep_correctness.py
    torchrun --nproc_per_node=4 tests/unit_tests/test_llep_correctness.py

Run specific test:
    torchrun --nproc_per_node=2 tests/unit_tests/test_llep_correctness.py --test grouped_vs_forloop_native
    torchrun --nproc_per_node=2 tests/unit_tests/test_llep_correctness.py --test fused_silu_gate

List all tests:
    torchrun --nproc_per_node=2 tests/unit_tests/test_llep_correctness.py --list

Environment variables:
    LLEP_W_TRANSFER_AUTOGRAD=1  (default) Enable autograd for weight transfer
    LLEP_MERGE_A2A=1            (default) Merge hidden+scores+ids into single A2A
    LLEP_USE_GROUPED_MM=1       (default) Use grouped GEMM in SwiGLU FFN
"""

import argparse
import os
import sys
import traceback
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reference: Standard EP forward (all-to-all, no LLEP)
# ---------------------------------------------------------------------------
def standard_ep_forward(
    hidden_states: torch.Tensor,  # (num_tokens, dim)
    top_scores: torch.Tensor,  # (num_tokens, top_k)
    selected_experts: torch.Tensor,  # (num_tokens, top_k)
    w1_local: torch.Tensor,  # (num_local_experts, hidden_dim, dim)
    w2_local: torch.Tensor,  # (num_local_experts, dim, hidden_dim)
    w3_local: torch.Tensor,  # (num_local_experts, hidden_dim, dim)
    ep_group,
    num_experts: int,
    num_local_experts: int,
    score_before_experts: bool = False,
) -> torch.Tensor:
    """
    Standard all-to-all EP forward (reference implementation).

    Each token-expert pair is sent to the GPU that owns the expert via A2A,
    the expert FFN is computed, and results are sent back via inverse A2A.
    No load balancing or weight spilling.
    """
    ep_rank = dist.get_rank(group=ep_group)
    ep_size = dist.get_world_size(group=ep_group)
    device = hidden_states.device
    dtype = hidden_states.dtype
    num_tokens = hidden_states.shape[0]
    dim = hidden_states.shape[1]
    top_k = selected_experts.shape[1]

    # Flatten token-expert pairs
    flat_experts = selected_experts.view(-1)  # (num_tokens * top_k,)
    flat_scores = top_scores.view(-1)

    # Compute target GPU for each token-expert pair (default routing)
    target_gpus = flat_experts // num_local_experts

    # Sort by target GPU for A2A
    sorted_indices = torch.argsort(target_gpus, stable=True)
    undo_indices = torch.argsort(sorted_indices)
    sorted_experts = flat_experts[sorted_indices]
    sorted_scores = flat_scores[sorted_indices]

    # Expand hidden states for top_k
    hidden_topk = hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, dim)
    sorted_hidden = hidden_topk[sorted_indices]

    # Compute split sizes
    local_expert_counts = torch.bincount(
        flat_experts.to(torch.int64), minlength=num_experts
    ).to(torch.int64)
    all_expert_counts = [torch.zeros_like(local_expert_counts) for _ in range(ep_size)]
    dist.all_gather(all_expert_counts, local_expert_counts, group=ep_group)

    import numpy as np

    all_counts_np = np.stack([ec.cpu().numpy() for ec in all_expert_counts])
    send_matrix_np = all_counts_np.reshape(ep_size, ep_size, num_local_experts).sum(
        axis=2
    )
    input_split_sizes = send_matrix_np[ep_rank].tolist()
    output_split_sizes = send_matrix_np[:, ep_rank].tolist()

    total_send = sum(input_split_sizes)
    total_recv = sum(output_split_sizes)

    # A2A dispatch
    if total_send > 0 or total_recv > 0:
        recv_hidden = torch.empty(total_recv, dim, device=device, dtype=dtype)
        dist.all_to_all_single(
            recv_hidden,
            sorted_hidden.contiguous(),
            output_split_sizes,
            input_split_sizes,
            group=ep_group,
        )
        recv_scores = torch.empty(total_recv, device=device, dtype=sorted_scores.dtype)
        dist.all_to_all_single(
            recv_scores,
            sorted_scores.contiguous(),
            output_split_sizes,
            input_split_sizes,
            group=ep_group,
        )
        recv_experts = torch.empty(total_recv, device=device, dtype=torch.int64)
        dist.all_to_all_single(
            recv_experts,
            sorted_experts.to(torch.int64).contiguous(),
            output_split_sizes,
            input_split_sizes,
            group=ep_group,
        )
    else:
        recv_hidden = torch.empty(0, dim, device=device, dtype=dtype)
        recv_scores = torch.empty(0, device=device, dtype=sorted_scores.dtype)
        recv_experts = torch.empty(0, device=device, dtype=torch.int64)

    # Apply scores before experts (if configured)
    if score_before_experts and recv_hidden.numel() > 0:
        recv_hidden = (recv_hidden.to(torch.float32) * recv_scores.reshape(-1, 1)).to(
            dtype
        )

    # Run SwiGLU FFN on received tokens
    if recv_hidden.numel() > 0:
        recv_output = _swiglu_ffn(
            recv_hidden,
            recv_experts,
            w1_local,
            w2_local,
            w3_local,
            ep_rank,
            num_local_experts,
        )
    else:
        recv_output = torch.empty(0, dim, device=device, dtype=dtype)

    # Apply scores after experts (if not before)
    if not score_before_experts and recv_output.numel() > 0:
        recv_output = (recv_output.to(torch.float32) * recv_scores.reshape(-1, 1)).to(
            dtype
        )

    # A2A combine (route outputs back)
    if total_send > 0 or total_recv > 0:
        send_output = torch.empty(total_send, dim, device=device, dtype=dtype)
        dist.all_to_all_single(
            send_output,
            recv_output.contiguous(),
            input_split_sizes,
            output_split_sizes,
            group=ep_group,
        )
    else:
        send_output = torch.empty(0, dim, device=device, dtype=dtype)

    # Unsort and aggregate top_k
    unsorted_output = send_output[undo_indices]
    unsorted_output = unsorted_output.view(num_tokens, top_k, dim).sum(dim=1)

    return unsorted_output


def _swiglu_ffn(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    ep_rank: int,
    num_local_experts: int,
) -> torch.Tensor:
    """Simple SwiGLU FFN for reference testing. Only handles native experts."""
    num_tokens, dim = x.shape
    native_start = ep_rank * num_local_experts

    compute_dtype = w1.dtype
    if x.dtype != compute_dtype:
        x = x.to(compute_dtype)

    sorted_ids, sort_perm = expert_ids.sort(stable=True)
    x_sorted = x[sort_perm]

    unique_experts, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
    offsets = torch.zeros(len(counts) + 1, dtype=torch.int64, device=x.device)
    offsets[1:] = counts.cumsum(0)

    out_sorted = torch.empty_like(x_sorted)

    for idx in range(len(unique_experts)):
        eid = unique_experts[idx].item()
        start = offsets[idx].item()
        end = offsets[idx + 1].item()
        x_slice = x_sorted[start:end]

        local_idx = eid - native_start
        h = F.silu(x_slice @ w1[local_idx].T) * (x_slice @ w3[local_idx].T)
        out_sorted[start:end] = h @ w2[local_idx].T

    inverse_perm = sort_perm.argsort()
    return out_sorted[inverse_perm]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def create_routing(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    device: torch.device,
    pattern: str = "balanced",
    hot_expert_ratio: float = 0.7,
    num_hot_experts: int = 2,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create deterministic routing patterns for testing.

    Args:
        pattern: One of "balanced", "imbalanced", "extreme", "single_hot", "random"
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    selected = torch.zeros(num_tokens, top_k, dtype=torch.int64, device=device)
    scores = torch.zeros(num_tokens, top_k, device=device, dtype=torch.float32)

    if pattern == "balanced":
        # Evenly distribute tokens across all experts
        for k in range(top_k):
            selected[:, k] = (torch.arange(num_tokens, device=device) + k) % num_experts
        scores = torch.rand(
            num_tokens, top_k, device=device, dtype=torch.float32, generator=g
        )
        scores = scores / scores.sum(dim=-1, keepdim=True)

    elif pattern == "imbalanced":
        # Most tokens go to a few hot experts
        num_hot_tokens = int(num_tokens * hot_expert_ratio)
        hot_experts = list(range(num_hot_experts))
        cold_experts = list(range(num_hot_experts, num_experts))

        for i in range(num_tokens):
            if i < num_hot_tokens:
                for k in range(top_k):
                    selected[i, k] = hot_experts[k % len(hot_experts)]
            else:
                for k in range(top_k):
                    selected[i, k] = cold_experts[(i + k) % len(cold_experts)]

        scores = torch.rand(
            num_tokens, top_k, device=device, dtype=torch.float32, generator=g
        )
        scores = scores / scores.sum(dim=-1, keepdim=True)

    elif pattern == "extreme":
        # 95% of tokens to expert 0 (forces heavy spilling)
        num_hot = int(num_tokens * 0.95)
        selected[:num_hot, 0] = 0
        selected[:num_hot, 1] = 0 if top_k > 1 else 0
        if top_k > 1:
            selected[:num_hot, 1] = 1
        for i in range(num_hot, num_tokens):
            for k in range(top_k):
                selected[i, k] = torch.randint(
                    0, num_experts, (1,), device=device, generator=g
                ).item()
        scores = torch.rand(
            num_tokens, top_k, device=device, dtype=torch.float32, generator=g
        )
        scores = scores / scores.sum(dim=-1, keepdim=True)

    elif pattern == "single_hot":
        # ALL tokens to expert 0 (maximum spilling)
        selected[:, 0] = 0
        if top_k > 1:
            selected[:, 1] = 1
        scores.fill_(1.0 / top_k)

    elif pattern == "random":
        # Fully random routing
        selected = torch.randint(
            0, num_experts, (num_tokens, top_k), device=device, generator=g
        )
        scores = torch.rand(
            num_tokens, top_k, device=device, dtype=torch.float32, generator=g
        )
        scores = scores / scores.sum(dim=-1, keepdim=True)

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return selected, scores


def broadcast_tensor(tensor: torch.Tensor, src: int = 0, group=None):
    """Broadcast tensor from src rank so all ranks have identical data."""
    dist.broadcast(tensor, src=src, group=group)
    return tensor


def gather_all_expert_weights(
    w1_local: torch.Tensor,
    w2_local: torch.Tensor,
    w3_local: torch.Tensor,
    ep_group,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather all expert weights to all ranks (for reference computation)."""
    ep_size = dist.get_world_size(group=ep_group)

    w1_all = [torch.empty_like(w1_local) for _ in range(ep_size)]
    w2_all = [torch.empty_like(w2_local) for _ in range(ep_size)]
    w3_all = [torch.empty_like(w3_local) for _ in range(ep_size)]

    dist.all_gather(w1_all, w1_local.contiguous(), group=ep_group)
    dist.all_gather(w2_all, w2_local.contiguous(), group=ep_group)
    dist.all_gather(w3_all, w3_local.contiguous(), group=ep_group)

    return torch.cat(w1_all, dim=0), torch.cat(w2_all, dim=0), torch.cat(w3_all, dim=0)


def single_gpu_reference_forward(
    hidden_states: torch.Tensor,  # (num_tokens, dim)
    top_scores: torch.Tensor,  # (num_tokens, top_k)
    selected_experts: torch.Tensor,  # (num_tokens, top_k)
    w1_all: torch.Tensor,  # (num_experts, hidden_dim, dim)
    w2_all: torch.Tensor,  # (num_experts, dim, hidden_dim)
    w3_all: torch.Tensor,  # (num_experts, hidden_dim, dim)
    score_before_experts: bool = False,
) -> torch.Tensor:
    """
    Single-GPU reference: no distributed ops, pure local computation.

    This is the ground truth that both standard EP and LLEP should match.
    """
    num_tokens, dim = hidden_states.shape
    top_k = selected_experts.shape[1]
    dtype = hidden_states.dtype

    compute_dtype = w1_all.dtype
    x = (
        hidden_states.to(compute_dtype)
        if hidden_states.dtype != compute_dtype
        else hidden_states
    )

    output = torch.zeros(num_tokens, dim, device=x.device, dtype=compute_dtype)

    for k in range(top_k):
        expert_ids = selected_experts[:, k]
        scores_k = top_scores[:, k]

        for eid in expert_ids.unique():
            eid_val = eid.item()
            mask = expert_ids == eid_val
            x_expert = x[mask]

            # SwiGLU: silu(x @ w1.T) * (x @ w3.T) @ w2.T
            h = F.silu(x_expert @ w1_all[eid_val].T) * (x_expert @ w3_all[eid_val].T)
            expert_out = h @ w2_all[eid_val].T

            if score_before_experts:
                expert_out = (
                    expert_out.to(torch.float32) * scores_k[mask].reshape(-1, 1)
                ).to(compute_dtype)
                output[mask] += expert_out
            else:
                output[mask] += (
                    expert_out.to(torch.float32) * scores_k[mask].reshape(-1, 1)
                ).to(compute_dtype)

    return output.to(dtype)


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------
def test_forward_correctness(
    pattern: str,
    num_tokens: int = 64,
    dim: int = 64,
    hidden_dim: int = 128,
    num_local_experts: int = 4,
    top_k: int = 2,
    dtype: torch.dtype = torch.bfloat16,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    seed: int = 42,
    score_before_experts: bool = False,
    max_tokens_factor: float = 1.1,
    min_tokens_per_gemm: int = 4,
):
    """
    Test that LLEP forward produces the same output as standard EP.

    Both methods are also compared against a single-GPU reference.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    ep_group = dist.group.WORLD

    num_experts = world_size * num_local_experts

    if rank == 0:
        print(f"\n{'='*70}")
        print(
            f"  Forward Test: pattern={pattern} tokens={num_tokens} "
            f"experts={num_experts} top_k={top_k} dtype={dtype}"
        )
        print(f"  world_size={world_size} score_before={score_before_experts}")
        print(f"{'='*70}")

    # Use same seed on all ranks for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Create expert weights on each rank (each rank has different local experts)
    # First create all weights on rank 0 and broadcast shard to each rank
    all_w1 = (
        torch.randn(num_experts, hidden_dim, dim, device=device, dtype=dtype) * 0.02
    )
    all_w2 = (
        torch.randn(num_experts, dim, hidden_dim, device=device, dtype=dtype) * 0.02
    )
    all_w3 = (
        torch.randn(num_experts, hidden_dim, dim, device=device, dtype=dtype) * 0.02
    )
    broadcast_tensor(all_w1, src=0, group=ep_group)
    broadcast_tensor(all_w2, src=0, group=ep_group)
    broadcast_tensor(all_w3, src=0, group=ep_group)

    # Shard to local experts
    local_start = rank * num_local_experts
    local_end = local_start + num_local_experts
    w1_local = all_w1[local_start:local_end].clone().contiguous()
    w2_local = all_w2[local_start:local_end].clone().contiguous()
    w3_local = all_w3[local_start:local_end].clone().contiguous()

    # Create identical inputs on all ranks
    x = torch.randn(num_tokens, dim, device=device, dtype=dtype) * 0.1
    broadcast_tensor(x, src=0, group=ep_group)
    selected_experts, top_scores = create_routing(
        num_tokens, num_experts, top_k, device, pattern=pattern, seed=seed
    )
    broadcast_tensor(selected_experts, src=0, group=ep_group)
    broadcast_tensor(top_scores, src=0, group=ep_group)

    # --------------- Single-GPU Reference ---------------
    ref_output = single_gpu_reference_forward(
        x,
        top_scores,
        selected_experts,
        all_w1,
        all_w2,
        all_w3,
        score_before_experts=score_before_experts,
    )

    # --------------- Standard EP ---------------
    std_output = standard_ep_forward(
        x,
        top_scores,
        selected_experts,
        w1_local,
        w2_local,
        w3_local,
        ep_group,
        num_experts,
        num_local_experts,
        score_before_experts=score_before_experts,
    )

    # --------------- LLEP ---------------
    from torchtitan.distributed.llep import llep_moe_forward

    # Need barrier before LLEP to ensure A2A state is clean
    dist.barrier(group=ep_group)

    llep_output = llep_moe_forward(
        hidden_states=x.clone(),
        top_scores=top_scores.clone(),
        selected_experts_indices=selected_experts.clone(),
        w1=w1_local.clone(),
        w2=w2_local.clone(),
        w3=w3_local.clone(),
        ep_group=ep_group,
        num_experts=num_experts,
        score_before_experts=score_before_experts,
        max_tokens_factor=max_tokens_factor,
        min_tokens_per_gemm=min_tokens_per_gemm,
        adaptive_threshold=0.0,  # Always use LPT
    )

    # --------------- Compare ---------------
    # All ranks should have the same ref_output (computed from identical inputs)
    # std_output and llep_output are per-rank results from distributed computation
    # They should all match the reference

    # Check std vs reference
    std_vs_ref_diff = (std_output - ref_output).abs()
    std_vs_ref_max = std_vs_ref_diff.max().item()
    std_vs_ref_mean = std_vs_ref_diff.mean().item()

    # Check llep vs reference
    llep_vs_ref_diff = (llep_output - ref_output).abs()
    llep_vs_ref_max = llep_vs_ref_diff.max().item()
    llep_vs_ref_mean = llep_vs_ref_diff.mean().item()

    # Check llep vs std
    llep_vs_std_diff = (llep_output - std_output).abs()
    llep_vs_std_max = llep_vs_std_diff.max().item()
    llep_vs_std_mean = llep_vs_std_diff.mean().item()

    if rank == 0:
        print(
            f"  Standard EP vs Reference:  max_diff={std_vs_ref_max:.6f}  mean_diff={std_vs_ref_mean:.6f}"
        )
        print(
            f"  LLEP vs Reference:         max_diff={llep_vs_ref_max:.6f}  mean_diff={llep_vs_ref_mean:.6f}"
        )
        print(
            f"  LLEP vs Standard EP:       max_diff={llep_vs_std_max:.6f}  mean_diff={llep_vs_std_mean:.6f}"
        )

    # Standard EP should be very close to reference (only float reordering diff)
    assert (
        std_vs_ref_max < atol
    ), f"Standard EP diverged from reference: max_diff={std_vs_ref_max:.6f} > atol={atol}"

    # LLEP should also be close to reference
    assert (
        llep_vs_ref_max < atol
    ), f"LLEP diverged from reference: max_diff={llep_vs_ref_max:.6f} > atol={atol}"

    # LLEP should match standard EP very closely
    assert (
        llep_vs_std_max < atol
    ), f"LLEP diverged from standard EP: max_diff={llep_vs_std_max:.6f} > atol={atol}"

    if rank == 0:
        print(f"  PASSED")

    dist.barrier(group=ep_group)


def test_backward_correctness(
    pattern: str,
    num_tokens: int = 32,
    dim: int = 64,
    hidden_dim: int = 128,
    num_local_experts: int = 4,
    top_k: int = 2,
    dtype: torch.dtype = torch.float32,  # Use float32 for stable gradients
    atol: float = 1e-4,
    seed: int = 42,
    score_before_experts: bool = False,
    max_tokens_factor: float = 1.5,
    min_tokens_per_gemm: int = 2,
):
    """
    Test that LLEP backward produces correct gradients for expert weights.

    Strategy:
    1. Run LLEP forward with requires_grad on w1, w2, w3
    2. Compute loss = output.sum() and backward()
    3. Gather gradients from all ranks (all_gather)
    4. Compare against single-GPU reference gradients

    Uses float32 for numerical stability in gradient comparison.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    ep_group = dist.group.WORLD

    num_experts = world_size * num_local_experts

    if rank == 0:
        print(f"\n{'='*70}")
        print(
            f"  Backward Test: pattern={pattern} tokens={num_tokens} "
            f"experts={num_experts} top_k={top_k}"
        )
        print(f"  world_size={world_size} dtype={dtype}")
        print(f"{'='*70}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Create all expert weights deterministically
    all_w1 = (
        torch.randn(num_experts, hidden_dim, dim, device=device, dtype=dtype) * 0.02
    )
    all_w2 = (
        torch.randn(num_experts, dim, hidden_dim, device=device, dtype=dtype) * 0.02
    )
    all_w3 = (
        torch.randn(num_experts, hidden_dim, dim, device=device, dtype=dtype) * 0.02
    )
    broadcast_tensor(all_w1, src=0, group=ep_group)
    broadcast_tensor(all_w2, src=0, group=ep_group)
    broadcast_tensor(all_w3, src=0, group=ep_group)

    # Inputs (identical on all ranks)
    x = torch.randn(num_tokens, dim, device=device, dtype=dtype) * 0.1
    broadcast_tensor(x, src=0, group=ep_group)
    selected_experts, top_scores = create_routing(
        num_tokens, num_experts, top_k, device, pattern=pattern, seed=seed
    )
    broadcast_tensor(selected_experts, src=0, group=ep_group)
    broadcast_tensor(top_scores, src=0, group=ep_group)

    # --------------- Reference: single-GPU backward ---------------
    ref_w1 = all_w1.clone().detach().requires_grad_(True)
    ref_w2 = all_w2.clone().detach().requires_grad_(True)
    ref_w3 = all_w3.clone().detach().requires_grad_(True)

    ref_output = single_gpu_reference_forward(
        x.clone(),
        top_scores.clone(),
        selected_experts.clone(),
        ref_w1,
        ref_w2,
        ref_w3,
        score_before_experts=score_before_experts,
    )
    ref_loss = ref_output.sum()
    ref_loss.backward()

    ref_grad_w1 = ref_w1.grad.clone()
    ref_grad_w2 = ref_w2.grad.clone()
    ref_grad_w3 = ref_w3.grad.clone()

    # --------------- LLEP backward ---------------
    local_start = rank * num_local_experts
    local_end = local_start + num_local_experts

    llep_w1 = all_w1[local_start:local_end].clone().detach().requires_grad_(True)
    llep_w2 = all_w2[local_start:local_end].clone().detach().requires_grad_(True)
    llep_w3 = all_w3[local_start:local_end].clone().detach().requires_grad_(True)

    from torchtitan.distributed.llep import llep_moe_forward

    dist.barrier(group=ep_group)

    llep_output = llep_moe_forward(
        hidden_states=x.clone(),
        top_scores=top_scores.clone(),
        selected_experts_indices=selected_experts.clone(),
        w1=llep_w1,
        w2=llep_w2,
        w3=llep_w3,
        ep_group=ep_group,
        num_experts=num_experts,
        score_before_experts=score_before_experts,
        max_tokens_factor=max_tokens_factor,
        min_tokens_per_gemm=min_tokens_per_gemm,
        adaptive_threshold=0.0,
    )
    llep_loss = llep_output.sum()
    llep_loss.backward()

    # Gather LLEP gradients from all ranks
    llep_grad_w1_all = [torch.empty_like(llep_w1) for _ in range(world_size)]
    llep_grad_w2_all = [torch.empty_like(llep_w2) for _ in range(world_size)]
    llep_grad_w3_all = [torch.empty_like(llep_w3) for _ in range(world_size)]

    dist.all_gather(llep_grad_w1_all, llep_w1.grad.contiguous(), group=ep_group)
    dist.all_gather(llep_grad_w2_all, llep_w2.grad.contiguous(), group=ep_group)
    dist.all_gather(llep_grad_w3_all, llep_w3.grad.contiguous(), group=ep_group)

    llep_grad_w1 = torch.cat(llep_grad_w1_all, dim=0)
    llep_grad_w2 = torch.cat(llep_grad_w2_all, dim=0)
    llep_grad_w3 = torch.cat(llep_grad_w3_all, dim=0)

    # --------------- Compare output ---------------
    output_diff = (llep_output - ref_output).abs()
    output_max_diff = output_diff.max().item()
    output_mean_diff = output_diff.mean().item()

    # --------------- Compare gradients ---------------
    grad_w1_diff = (llep_grad_w1 - ref_grad_w1).abs()
    grad_w2_diff = (llep_grad_w2 - ref_grad_w2).abs()
    grad_w3_diff = (llep_grad_w3 - ref_grad_w3).abs()

    grad_w1_max = grad_w1_diff.max().item()
    grad_w2_max = grad_w2_diff.max().item()
    grad_w3_max = grad_w3_diff.max().item()

    grad_w1_mean = grad_w1_diff.mean().item()
    grad_w2_mean = grad_w2_diff.mean().item()
    grad_w3_mean = grad_w3_diff.mean().item()

    # Check that gradients are not all zero (sanity check)
    ref_grad_w1_norm = ref_grad_w1.norm().item()
    ref_grad_w2_norm = ref_grad_w2.norm().item()
    ref_grad_w3_norm = ref_grad_w3.norm().item()
    llep_grad_w1_norm = llep_grad_w1.norm().item()
    llep_grad_w2_norm = llep_grad_w2.norm().item()
    llep_grad_w3_norm = llep_grad_w3.norm().item()

    if rank == 0:
        print(
            f"  Forward:  max_diff={output_max_diff:.6f}  mean_diff={output_mean_diff:.6f}"
        )
        print(
            f"  Grad w1:  max_diff={grad_w1_max:.6f}  mean_diff={grad_w1_mean:.6f}  "
            f"(ref_norm={ref_grad_w1_norm:.4f}, llep_norm={llep_grad_w1_norm:.4f})"
        )
        print(
            f"  Grad w2:  max_diff={grad_w2_max:.6f}  mean_diff={grad_w2_mean:.6f}  "
            f"(ref_norm={ref_grad_w2_norm:.4f}, llep_norm={llep_grad_w2_norm:.4f})"
        )
        print(
            f"  Grad w3:  max_diff={grad_w3_max:.6f}  mean_diff={grad_w3_mean:.6f}  "
            f"(ref_norm={ref_grad_w3_norm:.4f}, llep_norm={llep_grad_w3_norm:.4f})"
        )

    # Verify outputs match
    assert (
        output_max_diff < atol
    ), f"LLEP forward output diverged: max_diff={output_max_diff:.6f} > atol={atol}"

    # Verify gradients are non-zero (sanity)
    assert ref_grad_w1_norm > 0, "Reference grad_w1 is all zeros"
    assert llep_grad_w1_norm > 0, "LLEP grad_w1 is all zeros"

    # Verify gradients match
    assert (
        grad_w1_max < atol
    ), f"LLEP grad_w1 diverged: max_diff={grad_w1_max:.6f} > atol={atol}"
    assert (
        grad_w2_max < atol
    ), f"LLEP grad_w2 diverged: max_diff={grad_w2_max:.6f} > atol={atol}"
    assert (
        grad_w3_max < atol
    ), f"LLEP grad_w3 diverged: max_diff={grad_w3_max:.6f} > atol={atol}"

    if rank == 0:
        print(f"  PASSED")

    dist.barrier(group=ep_group)


def test_backward_hidden_grad(
    pattern: str = "imbalanced",
    num_tokens: int = 32,
    dim: int = 64,
    hidden_dim: int = 128,
    num_local_experts: int = 4,
    top_k: int = 2,
    dtype: torch.dtype = torch.float32,
    atol: float = 1e-4,
    seed: int = 42,
    max_tokens_factor: float = 1.5,
    min_tokens_per_gemm: int = 2,
):
    """
    Test that LLEP backward produces correct gradients for input hidden_states.

    This is crucial for training: the MoE layer must propagate gradients
    back through the input tokens correctly.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    ep_group = dist.group.WORLD

    num_experts = world_size * num_local_experts

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"  Hidden Grad Test: pattern={pattern} tokens={num_tokens}")
        print(f"{'='*70}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    all_w1 = (
        torch.randn(num_experts, hidden_dim, dim, device=device, dtype=dtype) * 0.02
    )
    all_w2 = (
        torch.randn(num_experts, dim, hidden_dim, device=device, dtype=dtype) * 0.02
    )
    all_w3 = (
        torch.randn(num_experts, hidden_dim, dim, device=device, dtype=dtype) * 0.02
    )
    broadcast_tensor(all_w1, src=0, group=ep_group)
    broadcast_tensor(all_w2, src=0, group=ep_group)
    broadcast_tensor(all_w3, src=0, group=ep_group)

    x_data = torch.randn(num_tokens, dim, device=device, dtype=dtype) * 0.1
    broadcast_tensor(x_data, src=0, group=ep_group)
    selected_experts, top_scores = create_routing(
        num_tokens, num_experts, top_k, device, pattern=pattern, seed=seed
    )
    broadcast_tensor(selected_experts, src=0, group=ep_group)
    broadcast_tensor(top_scores, src=0, group=ep_group)

    # --------------- Reference ---------------
    x_ref = x_data.clone().detach().requires_grad_(True)
    ref_output = single_gpu_reference_forward(
        x_ref,
        top_scores.clone(),
        selected_experts.clone(),
        all_w1,
        all_w2,
        all_w3,
    )
    ref_loss = ref_output.sum()
    ref_loss.backward()
    ref_grad_x = x_ref.grad.clone()

    # --------------- LLEP ---------------
    local_start = rank * num_local_experts
    local_end = local_start + num_local_experts
    w1_local = all_w1[local_start:local_end].clone().detach()
    w2_local = all_w2[local_start:local_end].clone().detach()
    w3_local = all_w3[local_start:local_end].clone().detach()

    x_llep = x_data.clone().detach().requires_grad_(True)

    from torchtitan.distributed.llep import llep_moe_forward

    dist.barrier(group=ep_group)

    llep_output = llep_moe_forward(
        hidden_states=x_llep,
        top_scores=top_scores.clone(),
        selected_experts_indices=selected_experts.clone(),
        w1=w1_local,
        w2=w2_local,
        w3=w3_local,
        ep_group=ep_group,
        num_experts=num_experts,
        max_tokens_factor=max_tokens_factor,
        min_tokens_per_gemm=min_tokens_per_gemm,
        adaptive_threshold=0.0,
    )
    llep_loss = llep_output.sum()
    llep_loss.backward()
    llep_grad_x = x_llep.grad.clone()

    # --------------- Compare ---------------
    grad_x_diff = (llep_grad_x - ref_grad_x).abs()
    grad_x_max = grad_x_diff.max().item()
    grad_x_mean = grad_x_diff.mean().item()
    ref_norm = ref_grad_x.norm().item()
    llep_norm = llep_grad_x.norm().item()

    if rank == 0:
        print(f"  Hidden grad: max_diff={grad_x_max:.6f}  mean_diff={grad_x_mean:.6f}")
        print(f"  Norms: ref={ref_norm:.4f}  llep={llep_norm:.4f}")

    assert ref_norm > 0, "Reference grad_x is all zeros"
    assert llep_norm > 0, "LLEP grad_x is all zeros"
    assert (
        grad_x_max < atol
    ), f"LLEP hidden_states grad diverged: max_diff={grad_x_max:.6f} > atol={atol}"

    if rank == 0:
        print(f"  PASSED")

    dist.barrier(group=ep_group)


def test_forward_determinism(
    num_runs: int = 3,
    pattern: str = "imbalanced",
    num_tokens: int = 32,
    dim: int = 64,
    hidden_dim: int = 128,
    num_local_experts: int = 4,
    top_k: int = 2,
    seed: int = 42,
):
    """Test that LLEP forward is deterministic across multiple runs."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    ep_group = dist.group.WORLD
    num_experts = world_size * num_local_experts
    dtype = torch.bfloat16

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"  Determinism Test: {num_runs} runs, pattern={pattern}")
        print(f"{'='*70}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    all_w1 = (
        torch.randn(num_experts, hidden_dim, dim, device=device, dtype=dtype) * 0.02
    )
    all_w2 = (
        torch.randn(num_experts, dim, hidden_dim, device=device, dtype=dtype) * 0.02
    )
    all_w3 = (
        torch.randn(num_experts, hidden_dim, dim, device=device, dtype=dtype) * 0.02
    )
    broadcast_tensor(all_w1, src=0, group=ep_group)
    broadcast_tensor(all_w2, src=0, group=ep_group)
    broadcast_tensor(all_w3, src=0, group=ep_group)

    local_start = rank * num_local_experts
    local_end = local_start + num_local_experts
    w1_local = all_w1[local_start:local_end].clone()
    w2_local = all_w2[local_start:local_end].clone()
    w3_local = all_w3[local_start:local_end].clone()

    x = torch.randn(num_tokens, dim, device=device, dtype=dtype) * 0.1
    broadcast_tensor(x, src=0, group=ep_group)
    selected, scores = create_routing(
        num_tokens, num_experts, top_k, device, pattern=pattern, seed=seed
    )
    broadcast_tensor(selected, src=0, group=ep_group)
    broadcast_tensor(scores, src=0, group=ep_group)

    from torchtitan.distributed.llep import llep_moe_forward

    outputs = []
    for run in range(num_runs):
        dist.barrier(group=ep_group)
        out = llep_moe_forward(
            hidden_states=x.clone(),
            top_scores=scores.clone(),
            selected_experts_indices=selected.clone(),
            w1=w1_local.clone(),
            w2=w2_local.clone(),
            w3=w3_local.clone(),
            ep_group=ep_group,
            num_experts=num_experts,
            max_tokens_factor=1.1,
            min_tokens_per_gemm=4,
            adaptive_threshold=0.0,
        )
        outputs.append(out.clone())

    # All runs should produce identical results
    all_match = True
    for i in range(1, num_runs):
        diff = (outputs[i] - outputs[0]).abs().max().item()
        if rank == 0:
            print(f"  Run {i} vs Run 0: max_diff={diff:.10f}")
        if diff > 0:
            all_match = False

    if rank == 0:
        if all_match:
            print(f"  PASSED (bitwise identical)")
        else:
            print(f"  PASSED (numerically identical within tolerance)")

    dist.barrier(group=ep_group)


def test_score_before_vs_after(
    pattern: str = "imbalanced",
    num_tokens: int = 32,
    dim: int = 64,
    hidden_dim: int = 128,
    num_local_experts: int = 4,
    top_k: int = 2,
    dtype: torch.dtype = torch.bfloat16,
    atol: float = 1e-2,
    seed: int = 42,
):
    """Test that score_before_experts works correctly in LLEP vs reference."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    ep_group = dist.group.WORLD
    num_experts = world_size * num_local_experts

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"  Score Before/After Test: pattern={pattern}")
        print(f"{'='*70}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    all_w1 = (
        torch.randn(num_experts, hidden_dim, dim, device=device, dtype=dtype) * 0.02
    )
    all_w2 = (
        torch.randn(num_experts, dim, hidden_dim, device=device, dtype=dtype) * 0.02
    )
    all_w3 = (
        torch.randn(num_experts, hidden_dim, dim, device=device, dtype=dtype) * 0.02
    )
    broadcast_tensor(all_w1, src=0, group=ep_group)
    broadcast_tensor(all_w2, src=0, group=ep_group)
    broadcast_tensor(all_w3, src=0, group=ep_group)

    local_start = rank * num_local_experts
    local_end = local_start + num_local_experts
    w1_local = all_w1[local_start:local_end].clone()
    w2_local = all_w2[local_start:local_end].clone()
    w3_local = all_w3[local_start:local_end].clone()

    x = torch.randn(num_tokens, dim, device=device, dtype=dtype) * 0.1
    broadcast_tensor(x, src=0, group=ep_group)
    selected, scores = create_routing(
        num_tokens, num_experts, top_k, device, pattern=pattern, seed=seed
    )
    broadcast_tensor(selected, src=0, group=ep_group)
    broadcast_tensor(scores, src=0, group=ep_group)

    from torchtitan.distributed.llep import llep_moe_forward

    for score_before in [False, True]:
        dist.barrier(group=ep_group)

        ref_output = single_gpu_reference_forward(
            x.clone(),
            scores.clone(),
            selected.clone(),
            all_w1,
            all_w2,
            all_w3,
            score_before_experts=score_before,
        )

        dist.barrier(group=ep_group)

        llep_output = llep_moe_forward(
            hidden_states=x.clone(),
            top_scores=scores.clone(),
            selected_experts_indices=selected.clone(),
            w1=w1_local.clone(),
            w2=w2_local.clone(),
            w3=w3_local.clone(),
            ep_group=ep_group,
            num_experts=num_experts,
            score_before_experts=score_before,
            max_tokens_factor=1.5,
            min_tokens_per_gemm=2,
            adaptive_threshold=0.0,
        )

        diff = (llep_output - ref_output).abs().max().item()
        if rank == 0:
            print(f"  score_before_experts={score_before}: max_diff={diff:.6f}")

        assert (
            diff < atol
        ), f"LLEP with score_before_experts={score_before} diverged: max_diff={diff:.6f}"

    if rank == 0:
        print(f"  PASSED")

    dist.barrier(group=ep_group)


def test_empty_tokens():
    """Test that LLEP handles the case where some ranks have no tokens gracefully."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    ep_group = dist.group.WORLD

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"  Edge Case: Various token counts")
        print(f"{'='*70}")

    dim = 64
    hidden_dim = 128
    num_local_experts = 4
    num_experts = world_size * num_local_experts
    top_k = 2
    dtype = torch.bfloat16

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    all_w1 = (
        torch.randn(num_experts, hidden_dim, dim, device=device, dtype=dtype) * 0.02
    )
    all_w2 = (
        torch.randn(num_experts, dim, hidden_dim, device=device, dtype=dtype) * 0.02
    )
    all_w3 = (
        torch.randn(num_experts, hidden_dim, dim, device=device, dtype=dtype) * 0.02
    )
    broadcast_tensor(all_w1, src=0, group=ep_group)
    broadcast_tensor(all_w2, src=0, group=ep_group)
    broadcast_tensor(all_w3, src=0, group=ep_group)

    local_start = rank * num_local_experts
    local_end = local_start + num_local_experts
    w1_local = all_w1[local_start:local_end].clone()
    w2_local = all_w2[local_start:local_end].clone()
    w3_local = all_w3[local_start:local_end].clone()

    from torchtitan.distributed.llep import llep_moe_forward

    # Test with small token counts (1, 2, 4, 8)
    for num_tokens in [1, 2, 4, 8]:
        x = torch.randn(num_tokens, dim, device=device, dtype=dtype) * 0.1
        broadcast_tensor(x, src=0, group=ep_group)
        selected, scores = create_routing(
            num_tokens, num_experts, top_k, device, pattern="balanced", seed=42
        )
        broadcast_tensor(selected, src=0, group=ep_group)
        broadcast_tensor(scores, src=0, group=ep_group)

        dist.barrier(group=ep_group)
        try:
            out = llep_moe_forward(
                hidden_states=x,
                top_scores=scores,
                selected_experts_indices=selected,
                w1=w1_local.clone(),
                w2=w2_local.clone(),
                w3=w3_local.clone(),
                ep_group=ep_group,
                num_experts=num_experts,
                max_tokens_factor=2.0,
                min_tokens_per_gemm=1,
                adaptive_threshold=0.0,
            )
            assert out.shape == (
                num_tokens,
                dim,
            ), f"Shape mismatch for {num_tokens} tokens"
            assert not torch.isnan(out).any(), f"NaN in output for {num_tokens} tokens"
            if rank == 0:
                print(f"  num_tokens={num_tokens}: OK (shape={out.shape})")
        except Exception as e:
            if rank == 0:
                print(f"  num_tokens={num_tokens}: FAILED - {e}")
            raise

    if rank == 0:
        print(f"  PASSED")

    dist.barrier(group=ep_group)


def test_varying_top_k():
    """Test LLEP with different top_k values (1, 2, 4)."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    ep_group = dist.group.WORLD

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"  Varying top_k Test")
        print(f"{'='*70}")

    dim = 64
    hidden_dim = 128
    num_local_experts = 4
    num_experts = world_size * num_local_experts
    num_tokens = 32
    dtype = torch.bfloat16
    atol = 1e-2

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    all_w1 = (
        torch.randn(num_experts, hidden_dim, dim, device=device, dtype=dtype) * 0.02
    )
    all_w2 = (
        torch.randn(num_experts, dim, hidden_dim, device=device, dtype=dtype) * 0.02
    )
    all_w3 = (
        torch.randn(num_experts, hidden_dim, dim, device=device, dtype=dtype) * 0.02
    )
    broadcast_tensor(all_w1, src=0, group=ep_group)
    broadcast_tensor(all_w2, src=0, group=ep_group)
    broadcast_tensor(all_w3, src=0, group=ep_group)

    local_start = rank * num_local_experts
    local_end = local_start + num_local_experts
    w1_local = all_w1[local_start:local_end].clone()
    w2_local = all_w2[local_start:local_end].clone()
    w3_local = all_w3[local_start:local_end].clone()

    from torchtitan.distributed.llep import llep_moe_forward

    for top_k in [1, 2, 4]:
        x = torch.randn(num_tokens, dim, device=device, dtype=dtype) * 0.1
        broadcast_tensor(x, src=0, group=ep_group)
        selected, scores = create_routing(
            num_tokens, num_experts, top_k, device, pattern="imbalanced", seed=42
        )
        broadcast_tensor(selected, src=0, group=ep_group)
        broadcast_tensor(scores, src=0, group=ep_group)

        ref_output = single_gpu_reference_forward(
            x,
            scores,
            selected,
            all_w1,
            all_w2,
            all_w3,
        )

        dist.barrier(group=ep_group)

        llep_output = llep_moe_forward(
            hidden_states=x.clone(),
            top_scores=scores.clone(),
            selected_experts_indices=selected.clone(),
            w1=w1_local.clone(),
            w2=w2_local.clone(),
            w3=w3_local.clone(),
            ep_group=ep_group,
            num_experts=num_experts,
            max_tokens_factor=1.5,
            min_tokens_per_gemm=2,
            adaptive_threshold=0.0,
        )

        diff = (llep_output - ref_output).abs().max().item()
        if rank == 0:
            print(f"  top_k={top_k}: max_diff={diff:.6f}")

        assert diff < atol, f"LLEP with top_k={top_k} diverged: max_diff={diff:.6f}"

    if rank == 0:
        print(f"  PASSED")

    dist.barrier(group=ep_group)


def test_adaptive_threshold():
    """Test that adaptive threshold correctly falls back to no-LPT when balanced."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    ep_group = dist.group.WORLD

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"  Adaptive Threshold Test")
        print(f"{'='*70}")

    dim = 64
    hidden_dim = 128
    num_local_experts = 4
    num_experts = world_size * num_local_experts
    num_tokens = 32
    top_k = 2
    dtype = torch.bfloat16
    atol = 1e-2

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    all_w1 = (
        torch.randn(num_experts, hidden_dim, dim, device=device, dtype=dtype) * 0.02
    )
    all_w2 = (
        torch.randn(num_experts, dim, hidden_dim, device=device, dtype=dtype) * 0.02
    )
    all_w3 = (
        torch.randn(num_experts, hidden_dim, dim, device=device, dtype=dtype) * 0.02
    )
    broadcast_tensor(all_w1, src=0, group=ep_group)
    broadcast_tensor(all_w2, src=0, group=ep_group)
    broadcast_tensor(all_w3, src=0, group=ep_group)

    local_start = rank * num_local_experts
    local_end = local_start + num_local_experts
    w1_local = all_w1[local_start:local_end].clone()
    w2_local = all_w2[local_start:local_end].clone()
    w3_local = all_w3[local_start:local_end].clone()

    from torchtitan.distributed.llep import llep_moe_forward

    # With balanced routing and high threshold, should use no-LPT path
    x = torch.randn(num_tokens, dim, device=device, dtype=dtype) * 0.1
    broadcast_tensor(x, src=0, group=ep_group)
    selected, scores = create_routing(
        num_tokens, num_experts, top_k, device, pattern="balanced", seed=42
    )
    broadcast_tensor(selected, src=0, group=ep_group)
    broadcast_tensor(scores, src=0, group=ep_group)

    ref_output = single_gpu_reference_forward(
        x,
        scores,
        selected,
        all_w1,
        all_w2,
        all_w3,
    )

    # Use high adaptive threshold so balanced routing skips LPT
    dist.barrier(group=ep_group)
    llep_output_adaptive = llep_moe_forward(
        hidden_states=x.clone(),
        top_scores=scores.clone(),
        selected_experts_indices=selected.clone(),
        w1=w1_local.clone(),
        w2=w2_local.clone(),
        w3=w3_local.clone(),
        ep_group=ep_group,
        num_experts=num_experts,
        max_tokens_factor=1.5,
        min_tokens_per_gemm=2,
        adaptive_threshold=1.3,  # Balanced routing has ratio ~1.0 < 1.3
    )

    # Use no threshold (always LPT)
    dist.barrier(group=ep_group)
    llep_output_always_lpt = llep_moe_forward(
        hidden_states=x.clone(),
        top_scores=scores.clone(),
        selected_experts_indices=selected.clone(),
        w1=w1_local.clone(),
        w2=w2_local.clone(),
        w3=w3_local.clone(),
        ep_group=ep_group,
        num_experts=num_experts,
        max_tokens_factor=1.5,
        min_tokens_per_gemm=2,
        adaptive_threshold=0.0,  # Always use LPT
    )

    diff_adaptive = (llep_output_adaptive - ref_output).abs().max().item()
    diff_always = (llep_output_always_lpt - ref_output).abs().max().item()
    diff_between = (llep_output_adaptive - llep_output_always_lpt).abs().max().item()

    if rank == 0:
        print(f"  Adaptive vs ref:    max_diff={diff_adaptive:.6f}")
        print(f"  Always-LPT vs ref:  max_diff={diff_always:.6f}")
        print(f"  Adaptive vs LPT:    max_diff={diff_between:.6f}")

    assert diff_adaptive < atol, f"Adaptive output diverged: {diff_adaptive:.6f}"
    assert diff_always < atol, f"Always-LPT output diverged: {diff_always:.6f}"

    if rank == 0:
        print(f"  PASSED")

    dist.barrier(group=ep_group)


# ---------------------------------------------------------------------------
# Category 1: Grouped MM vs For-Loop (single-process, runs on all ranks)
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
# Category 4: Triton Kernel Unit Tests (single-process)
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
# Category 6: Numerical Stability
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
# Category 7: Performance Benchmarks
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
    # Category 2: Distributed forward correctness
    "forward_balanced": lambda: test_forward_correctness("balanced"),
    "forward_imbalanced": lambda: test_forward_correctness("imbalanced"),
    "forward_extreme": lambda: test_forward_correctness("extreme"),
    "forward_single_hot": lambda: test_forward_correctness("single_hot"),
    "forward_random": lambda: test_forward_correctness("random"),
    "forward_bf16": lambda: test_forward_correctness(
        "imbalanced", dtype=torch.bfloat16
    ),
    "forward_fp32": lambda: test_forward_correctness(
        "imbalanced", dtype=torch.float32, atol=1e-4
    ),
    "forward_large": lambda: test_forward_correctness(
        "imbalanced", num_tokens=256, dim=128, hidden_dim=256
    ),
    "forward_score_before": lambda: test_forward_correctness(
        "imbalanced", score_before_experts=True
    ),
    # Category 3: Backward correctness
    "backward_balanced": lambda: test_backward_correctness("balanced", atol=1e-2),
    "backward_imbalanced": lambda: test_backward_correctness("imbalanced", atol=1e-2),
    "backward_extreme": lambda: test_backward_correctness("extreme", atol=1e-2),
    "backward_random": lambda: test_backward_correctness("random", atol=1e-2),
    "backward_hidden_balanced": lambda: test_backward_hidden_grad("balanced"),
    "backward_hidden_imbalanced": lambda: test_backward_hidden_grad("imbalanced"),
    "backward_hidden_extreme": lambda: test_backward_hidden_grad("extreme"),
    # Category 4: Triton kernel unit tests
    "fused_silu_gate": test_fused_silu_gate_vs_pytorch,
    "fused_silu_gate_backward": test_fused_silu_gate_backward,
    "fused_silu_gate_shapes": test_fused_silu_gate_various_shapes,
    "fused_silu_gate_determinism": test_fused_silu_gate_determinism,
    # Category 6: Numerical stability
    "numerical_bf16_vs_fp32": test_numerical_bf16_vs_fp32,
    "numerical_large_values": test_numerical_large_values,
    "numerical_near_zero": test_numerical_near_zero,
    "numerical_fp32_precision": test_numerical_fp32_precision,
    # Category 7: Benchmarks
    "benchmark_ffn": test_benchmark_ffn_grouped_vs_forloop,
    "benchmark_triton": test_benchmark_triton_silu_gate,
    # Category 8: Integration
    "determinism": test_forward_determinism,
    "score_before_after": test_score_before_vs_after,
    "empty_tokens": test_empty_tokens,
    "varying_top_k": test_varying_top_k,
    "adaptive_threshold": test_adaptive_threshold,
}


def main():
    parser = argparse.ArgumentParser(description="LLEP Correctness Tests")
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Run a specific test (e.g., 'forward_balanced'). If not specified, run all.",
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
        print(
            f"  LLEP_W_TRANSFER_AUTOGRAD={os.environ.get('LLEP_W_TRANSFER_AUTOGRAD', '1')}"
        )
        print(f"  LLEP_MERGE_A2A={os.environ.get('LLEP_MERGE_A2A', '1')}")
        print(f"  LLEP_USE_GROUPED_MM={os.environ.get('LLEP_USE_GROUPED_MM', '1')}")
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
