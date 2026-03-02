#!/usr/bin/env python3
# Copyright (c) Nous Research and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Comprehensive correctness tests for the hook-based LLEP flow:
    llep_dispatch_tokens -> llep_compute_with_weights -> llep_combine_output

Tests that the hook-based decomposition produces numerically identical
results to standard EP (all-to-all reference).

Test categories (59 tests total):
 1. top_k sweep         — top_k={1,2,4,8} x 5 routing patterns  (20 tests)
 2. alpha sweep         — max_tokens_factor={1.0,1.1,1.3,2.0} x 3 patterns  (12 tests)
 3. min_tokens sweep    — min_tokens_per_gemm={1,4,64,1024}  (4 tests)
 4. lambda sweep        — adaptive_threshold={0.0,1.3,100.0}  (3 tests)
 5. Expert count sweep  — {8,16,32,64} experts (needs 2-8 GPUs)  (4 tests)
 6. Token edge cases    — num_tokens={1,2,4,16,256}  (5 tests)
 7. Dimension sweep     — (dim,hidden)={(64,128),(128,256),(256,768)}  (3 tests)
 8. Backward            — gradient correctness, float32  (6 tests)
 9. score_before        — score_before_experts={True,False}  (2 tests)

Run (requires >= 2 GPUs):
    torchrun --nproc_per_node=2 tests/unit_tests/test_llep_hooks.py
    torchrun --nproc_per_node=8 tests/unit_tests/test_llep_hooks.py

Run specific category:
    torchrun --nproc_per_node=2 tests/unit_tests/test_llep_hooks.py --category topk

List all tests:
    torchrun --nproc_per_node=2 tests/unit_tests/test_llep_hooks.py --list
"""

import argparse
import sys
import traceback

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def setup():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()


def broadcast_tensor(tensor, src=0, group=None):
    """Broadcast tensor from src rank so all ranks have identical data."""
    dist.broadcast(tensor, src=src, group=group)
    return tensor


# ---------------------------------------------------------------------------
# Routing patterns
# ---------------------------------------------------------------------------
def create_routing(
    num_tokens,
    num_experts,
    top_k,
    device,
    pattern="balanced",
    hot_expert_ratio=0.7,
    num_hot_experts=2,
    seed=42,
):
    """
    Create deterministic routing patterns for testing.

    Patterns:
        balanced    — round-robin across all experts
        imbalanced  — hot_expert_ratio tokens go to num_hot_experts experts
        extreme     — 95% of tokens to expert 0
        single_hot  — ALL tokens to expert 0 (maximum spilling)
        random      — uniform random routing
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    selected = torch.zeros(num_tokens, top_k, dtype=torch.int64, device=device)
    scores = torch.zeros(num_tokens, top_k, device=device, dtype=torch.float32)

    if pattern == "balanced":
        for k in range(top_k):
            selected[:, k] = (torch.arange(num_tokens, device=device) + k) % num_experts
        scores = torch.rand(
            num_tokens, top_k, device=device, dtype=torch.float32, generator=g
        )
        scores = scores / scores.sum(dim=-1, keepdim=True)

    elif pattern == "imbalanced":
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
        num_hot = int(num_tokens * 0.95)
        selected[:num_hot, 0] = 0
        if top_k > 1:
            selected[:num_hot, 1] = 1
        for k in range(2, top_k):
            selected[:num_hot, k] = k % num_experts
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
        selected[:, 0] = 0
        for k in range(1, top_k):
            selected[:, k] = k % num_experts
        scores.fill_(1.0 / top_k)

    elif pattern == "random":
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


# ---------------------------------------------------------------------------
# Reference: Standard EP forward (all-to-all, no LLEP)
# ---------------------------------------------------------------------------
def reference_ep_forward(
    hidden_states,
    top_scores,
    selected_experts,
    w1_local,
    w2_local,
    w3_local,
    ep_group,
    num_experts,
    num_local_experts,
    score_before_experts=True,
):
    """Standard all-to-all EP forward used as ground truth."""
    ep_rank = dist.get_rank(group=ep_group)
    ep_size = dist.get_world_size(group=ep_group)
    device = hidden_states.device
    dtype = hidden_states.dtype
    num_tokens = hidden_states.shape[0]
    dim = hidden_states.shape[1]
    top_k = selected_experts.shape[1]

    flat_experts = selected_experts.view(-1)
    flat_scores = top_scores.view(-1)

    target_gpus = flat_experts // num_local_experts
    sorted_indices = torch.argsort(target_gpus, stable=True)
    undo_indices = torch.argsort(sorted_indices)
    sorted_experts = flat_experts[sorted_indices]
    sorted_scores = flat_scores[sorted_indices]

    hidden_topk = hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, dim)
    sorted_hidden = hidden_topk[sorted_indices]

    local_expert_counts = torch.bincount(
        flat_experts.to(torch.int64), minlength=num_experts
    ).to(torch.int64)
    all_expert_counts = [torch.zeros_like(local_expert_counts) for _ in range(ep_size)]
    dist.all_gather(all_expert_counts, local_expert_counts, group=ep_group)

    all_counts_np = np.stack([ec.cpu().numpy() for ec in all_expert_counts])
    send_matrix_np = all_counts_np.reshape(ep_size, ep_size, num_local_experts).sum(
        axis=2
    )
    input_split_sizes = send_matrix_np[ep_rank].tolist()
    output_split_sizes = send_matrix_np[:, ep_rank].tolist()

    total_send = sum(input_split_sizes)
    total_recv = sum(output_split_sizes)

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

    if score_before_experts and recv_hidden.numel() > 0:
        recv_hidden = (recv_hidden.to(torch.float32) * recv_scores.reshape(-1, 1)).to(
            dtype
        )

    if recv_hidden.numel() > 0:
        native_start = ep_rank * num_local_experts
        compute_dtype = w1_local.dtype
        x = (
            recv_hidden.to(compute_dtype)
            if recv_hidden.dtype != compute_dtype
            else recv_hidden
        )
        sorted_ids, sort_perm = recv_experts.sort(stable=True)
        x_sorted = x[sort_perm]
        unique_experts_t, counts = torch.unique_consecutive(
            sorted_ids, return_counts=True
        )
        offsets = torch.zeros(len(counts) + 1, dtype=torch.int64, device=device)
        offsets[1:] = counts.cumsum(0)
        out_sorted = torch.empty_like(x_sorted)
        for idx in range(len(unique_experts_t)):
            eid = unique_experts_t[idx].item()
            s = offsets[idx].item()
            e = offsets[idx + 1].item()
            local_idx = eid - native_start
            h = F.silu(x_sorted[s:e] @ w1_local[local_idx].T) * (
                x_sorted[s:e] @ w3_local[local_idx].T
            )
            out_sorted[s:e] = h @ w2_local[local_idx].T
        inverse_perm = sort_perm.argsort()
        recv_output = out_sorted[inverse_perm]
    else:
        recv_output = torch.empty(0, dim, device=device, dtype=dtype)

    if not score_before_experts and recv_output.numel() > 0:
        recv_output = (recv_output.to(torch.float32) * recv_scores.reshape(-1, 1)).to(
            dtype
        )

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

    unsorted_output = send_output[undo_indices]
    return unsorted_output.view(num_tokens, top_k, dim).sum(dim=1)


# ---------------------------------------------------------------------------
# Single-GPU reference (for backward comparison)
# ---------------------------------------------------------------------------
def single_gpu_reference_forward(
    hidden_states,
    top_scores,
    selected_experts,
    w1_all,
    w2_all,
    w3_all,
    score_before_experts=True,
):
    """Single-GPU reference: no distributed ops, pure local computation."""
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
# LLEP hook pre/post processing (mirrors MoE.forward)
# ---------------------------------------------------------------------------
def _preprocess_for_hooks(
    x, selected_experts, top_scores, num_experts, score_before_experts
):
    """Flatten, sort by expert, apply scores — what MoE.forward does before hooks."""
    top_k = selected_experts.shape[1]
    dim = x.shape[1]

    flat_experts = selected_experts.view(-1)
    flat_scores = top_scores.view(-1)
    sorted_indices = flat_experts.argsort(stable=True)
    sorted_experts_ids = flat_experts[sorted_indices]
    sorted_scores = flat_scores[sorted_indices]

    hidden_topk = x.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, dim)
    routed_input = hidden_topk[sorted_indices]

    if score_before_experts:
        routed_input = (routed_input.float() * sorted_scores.unsqueeze(1)).to(
            routed_input.dtype
        )

    num_tokens_per_expert = torch.bincount(
        sorted_experts_ids.to(torch.int64), minlength=num_experts
    ).to(torch.int64)

    return routed_input, num_tokens_per_expert, sorted_indices


def _postprocess_hooks_output(combined_output, sorted_indices, num_tokens, top_k, dim):
    """Unsort and aggregate top_k — what MoE.forward does after hooks."""
    undo = sorted_indices.argsort()
    return combined_output[undo].view(num_tokens, top_k, dim).sum(dim=1)


# ---------------------------------------------------------------------------
# Core test runners
# ---------------------------------------------------------------------------
def run_hooks_test(
    num_tokens=64,
    num_experts=16,
    top_k=2,
    dim=64,
    hidden_dim=128,
    pattern="imbalanced",
    dtype=torch.bfloat16,
    atol=1e-2,
    seed=42,
    score_before_experts=True,
    max_tokens_factor=1.1,
    min_tokens_per_gemm=1,
    adaptive_threshold=0.0,
    check_backward=False,
    backward_atol=1e-2,
):
    """
    Core test: compare hook-based LLEP against standard EP reference.

    Returns (max_diff, passed, backward_info).
    """
    from torchtitan.distributed.llep import (
        llep_combine_output,
        llep_compute_with_weights,
        llep_dispatch_tokens,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    ep_group = dist.group.WORLD
    num_local_experts = num_experts // world_size

    assert (
        num_local_experts * world_size == num_experts
    ), f"num_experts={num_experts} not divisible by world_size={world_size}"

    # Shared weights (broadcast from rank 0)
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
    w1_local = all_w1[local_start:local_end].clone().contiguous()
    w2_local = all_w2[local_start:local_end].clone().contiguous()
    w3_local = all_w3[local_start:local_end].clone().contiguous()

    # Shared inputs
    x = torch.randn(num_tokens, dim, device=device, dtype=dtype) * 0.1
    broadcast_tensor(x, src=0, group=ep_group)
    selected_experts, top_scores = create_routing(
        num_tokens,
        num_experts,
        top_k,
        device,
        pattern=pattern,
        seed=seed,
    )
    broadcast_tensor(selected_experts, src=0, group=ep_group)
    broadcast_tensor(top_scores, src=0, group=ep_group)

    # --- Reference: Standard EP ---
    dist.barrier(group=ep_group)
    ref_output = reference_ep_forward(
        x.clone(),
        top_scores.clone(),
        selected_experts.clone(),
        all_w1[local_start:local_end].clone(),
        all_w2[local_start:local_end].clone(),
        all_w3[local_start:local_end].clone(),
        ep_group,
        num_experts,
        num_local_experts,
        score_before_experts=score_before_experts,
    )

    # --- Hooks: dispatch -> compute -> combine ---
    dist.barrier(group=ep_group)

    routed_input, ntpe, sorted_indices = _preprocess_for_hooks(
        x,
        selected_experts,
        top_scores,
        num_experts,
        score_before_experts,
    )

    dispatched, padded_counts, state = llep_dispatch_tokens(
        routed_input,
        ntpe,
        ep_group,
        max_tokens_factor=max_tokens_factor,
        min_tokens_per_gemm=min_tokens_per_gemm,
        adaptive_threshold=adaptive_threshold,
    )
    output = llep_compute_with_weights(
        dispatched,
        padded_counts,
        w1_local,
        w2_local,
        w3_local,
        state,
        use_grouped_mm=True,
    )
    combined = llep_combine_output(output, state)

    hooks_result = _postprocess_hooks_output(
        combined,
        sorted_indices,
        num_tokens,
        top_k,
        dim,
    )

    # --- Compare ---
    max_diff = (ref_output - hooks_result).abs().max().item()
    passed = max_diff < atol

    # --- Backward (optional) ---
    backward_info = None
    if check_backward and passed:
        backward_info = _run_backward_check(
            x,
            top_scores,
            selected_experts,
            all_w1,
            all_w2,
            all_w3,
            num_experts,
            num_local_experts,
            top_k,
            dim,
            ep_group,
            score_before_experts,
            max_tokens_factor,
            min_tokens_per_gemm,
            adaptive_threshold,
            backward_atol,
        )
        passed = passed and backward_info["passed"]

    dist.barrier(group=ep_group)
    return max_diff, passed, backward_info


def _run_backward_check(
    x,
    top_scores,
    selected_experts,
    all_w1,
    all_w2,
    all_w3,
    num_experts,
    num_local_experts,
    top_k,
    dim,
    ep_group,
    score_before_experts,
    max_tokens_factor,
    min_tokens_per_gemm,
    adaptive_threshold,
    backward_atol,
):
    """Run backward pass and compare gradients against single-GPU reference."""
    from torchtitan.distributed.llep import (
        llep_combine_output,
        llep_compute_with_weights,
        llep_dispatch_tokens,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_tokens = x.shape[0]
    local_start = rank * num_local_experts
    local_end = local_start + num_local_experts

    dist.barrier(group=ep_group)

    # Reference: single-GPU backward
    ref_w1 = all_w1.clone().detach().requires_grad_(True)
    ref_w2 = all_w2.clone().detach().requires_grad_(True)
    ref_w3 = all_w3.clone().detach().requires_grad_(True)
    x_ref = x.clone().detach().requires_grad_(True)

    ref_out = single_gpu_reference_forward(
        x_ref,
        top_scores.clone(),
        selected_experts.clone(),
        ref_w1,
        ref_w2,
        ref_w3,
        score_before_experts=score_before_experts,
    )
    ref_out.sum().backward()

    # Hooks backward
    x_hooks = x.clone().detach().requires_grad_(True)
    w1_bwd = all_w1[local_start:local_end].clone().detach().requires_grad_(True)
    w2_bwd = all_w2[local_start:local_end].clone().detach().requires_grad_(True)
    w3_bwd = all_w3[local_start:local_end].clone().detach().requires_grad_(True)

    dist.barrier(group=ep_group)

    routed, ntpe, sorted_idx = _preprocess_for_hooks(
        x_hooks,
        selected_experts,
        top_scores,
        num_experts,
        score_before_experts,
    )
    disp, pc, st = llep_dispatch_tokens(
        routed,
        ntpe,
        ep_group,
        max_tokens_factor=max_tokens_factor,
        min_tokens_per_gemm=min_tokens_per_gemm,
        adaptive_threshold=adaptive_threshold,
    )
    out = llep_compute_with_weights(
        disp, pc, w1_bwd, w2_bwd, w3_bwd, st, use_grouped_mm=True
    )
    comb = llep_combine_output(out, st)
    result = _postprocess_hooks_output(comb, sorted_idx, num_tokens, top_k, dim)
    result.sum().backward()

    # Gather gradients from all ranks
    g_w1_all = [torch.empty_like(w1_bwd) for _ in range(world_size)]
    g_w2_all = [torch.empty_like(w2_bwd) for _ in range(world_size)]
    g_w3_all = [torch.empty_like(w3_bwd) for _ in range(world_size)]
    dist.all_gather(g_w1_all, w1_bwd.grad.contiguous(), group=ep_group)
    dist.all_gather(g_w2_all, w2_bwd.grad.contiguous(), group=ep_group)
    dist.all_gather(g_w3_all, w3_bwd.grad.contiguous(), group=ep_group)

    llep_grad_w1 = torch.cat(g_w1_all, dim=0)
    llep_grad_w2 = torch.cat(g_w2_all, dim=0)
    llep_grad_w3 = torch.cat(g_w3_all, dim=0)

    grad_w1_max = (llep_grad_w1 - ref_w1.grad).abs().max().item()
    grad_w2_max = (llep_grad_w2 - ref_w2.grad).abs().max().item()
    grad_w3_max = (llep_grad_w3 - ref_w3.grad).abs().max().item()
    grad_x_max = (x_hooks.grad - x_ref.grad).abs().max().item()
    grad_max = max(grad_w1_max, grad_w2_max, grad_w3_max, grad_x_max)

    return {
        "grad_w1_max": grad_w1_max,
        "grad_w2_max": grad_w2_max,
        "grad_w3_max": grad_w3_max,
        "grad_x_max": grad_x_max,
        "grad_max": grad_max,
        "passed": grad_max < backward_atol,
    }


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------
def _topk_tests():
    """Category 1: top_k={1,2,4,8} x 5 routing patterns."""
    tests = {}
    for top_k in [1, 2, 4, 8]:
        for pattern in ["balanced", "imbalanced", "extreme", "random", "single_hot"]:
            tests[f"topk_{top_k}_{pattern}"] = {
                "fn": run_hooks_test,
                "kwargs": dict(
                    num_tokens=64,
                    num_experts=16,
                    top_k=top_k,
                    dim=64,
                    hidden_dim=128,
                    pattern=pattern,
                    max_tokens_factor=1.1,
                    min_tokens_per_gemm=1,
                ),
            }
    return tests


def _alpha_tests():
    """Category 2: max_tokens_factor={1.0,1.1,1.3,2.0} x 3 patterns."""
    tests = {}
    for alpha in [1.0, 1.1, 1.3, 2.0]:
        for pattern in ["balanced", "imbalanced", "extreme"]:
            tests[f"alpha_{alpha}_{pattern}"] = {
                "fn": run_hooks_test,
                "kwargs": dict(
                    num_tokens=64,
                    num_experts=16,
                    top_k=4,
                    dim=64,
                    hidden_dim=128,
                    pattern=pattern,
                    max_tokens_factor=alpha,
                    min_tokens_per_gemm=1,
                ),
            }
    return tests


def _min_tokens_tests():
    """Category 3: min_tokens_per_gemm={1,4,64,1024}."""
    tests = {}
    for m in [1, 4, 64, 1024]:
        tests[f"min_tokens_{m}"] = {
            "fn": run_hooks_test,
            "kwargs": dict(
                num_tokens=64,
                num_experts=16,
                top_k=4,
                dim=64,
                hidden_dim=128,
                pattern="imbalanced",
                max_tokens_factor=1.1,
                min_tokens_per_gemm=m,
            ),
        }
    return tests


def _lambda_tests():
    """Category 4: adaptive_threshold={0.0,1.3,100.0}."""
    tests = {}
    for lam in [0.0, 1.3, 100.0]:
        tests[f"lambda_{lam}"] = {
            "fn": run_hooks_test,
            "kwargs": dict(
                num_tokens=64,
                num_experts=16,
                top_k=4,
                dim=64,
                hidden_dim=128,
                pattern="imbalanced",
                max_tokens_factor=1.1,
                min_tokens_per_gemm=1,
                adaptive_threshold=lam,
            ),
        }
    return tests


def _expert_count_tests():
    """Category 5: {8,16,32,64} experts (needs varying GPU counts)."""
    tests = {}
    for num_experts, min_ep in [(8, 2), (16, 2), (32, 4), (64, 8)]:
        tests[f"experts_{num_experts}_ep{min_ep}"] = {
            "fn": run_hooks_test,
            "kwargs": dict(
                num_tokens=64,
                num_experts=num_experts,
                top_k=4,
                dim=64,
                hidden_dim=128,
                pattern="imbalanced",
                max_tokens_factor=1.1,
                min_tokens_per_gemm=1,
            ),
            "min_gpus": min_ep,
        }
    return tests


def _token_count_tests():
    """Category 6: num_tokens={1,2,4,16,256}."""
    tests = {}
    for n in [1, 2, 4, 16, 256]:
        tests[f"tokens_{n}"] = {
            "fn": run_hooks_test,
            "kwargs": dict(
                num_tokens=n,
                num_experts=16,
                top_k=2,
                dim=64,
                hidden_dim=128,
                pattern="balanced" if n <= 4 else "imbalanced",
                max_tokens_factor=2.0,
                min_tokens_per_gemm=1,
            ),
        }
    return tests


def _dim_tests():
    """Category 7: (dim,hidden)={(64,128),(128,256),(256,768)}."""
    tests = {}
    for dim, hidden_dim in [(64, 128), (128, 256), (256, 768)]:
        tests[f"dim_{dim}_h{hidden_dim}"] = {
            "fn": run_hooks_test,
            "kwargs": dict(
                num_tokens=32,
                num_experts=16,
                top_k=4,
                dim=dim,
                hidden_dim=hidden_dim,
                pattern="imbalanced",
                max_tokens_factor=1.1,
                min_tokens_per_gemm=1,
            ),
        }
    return tests


def _backward_tests():
    """Category 8: backward correctness, float32."""
    tests = {}
    for pattern in ["balanced", "imbalanced", "extreme"]:
        for top_k in [2, 4]:
            tests[f"backward_{pattern}_topk{top_k}"] = {
                "fn": run_hooks_test,
                "kwargs": dict(
                    num_tokens=32,
                    num_experts=16,
                    top_k=top_k,
                    dim=64,
                    hidden_dim=128,
                    pattern=pattern,
                    dtype=torch.float32,
                    max_tokens_factor=1.5,
                    min_tokens_per_gemm=1,
                    atol=1e-4,
                    check_backward=True,
                    backward_atol=1e-2,
                ),
            }
    return tests


def _score_before_tests():
    """Category 9: score_before_experts={True,False}."""
    tests = {}
    for val in [True, False]:
        tests[f"score_before_{val}"] = {
            "fn": run_hooks_test,
            "kwargs": dict(
                num_tokens=64,
                num_experts=16,
                top_k=4,
                dim=64,
                hidden_dim=128,
                pattern="imbalanced",
                max_tokens_factor=1.1,
                min_tokens_per_gemm=1,
                score_before_experts=val,
            ),
        }
    return tests


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
CATEGORY_BUILDERS = {
    "topk": ("Category 1: top_k Sweep", _topk_tests),
    "alpha": ("Category 2: max_tokens_factor Sweep", _alpha_tests),
    "min_tokens": ("Category 3: min_tokens_per_gemm Sweep", _min_tokens_tests),
    "lambda": ("Category 4: adaptive_threshold Sweep", _lambda_tests),
    "experts": ("Category 5: Expert Count Sweep", _expert_count_tests),
    "tokens": ("Category 6: Token Count Edge Cases", _token_count_tests),
    "dims": ("Category 7: Dimension Sweep", _dim_tests),
    "backward": ("Category 8: Backward Correctness", _backward_tests),
    "score_before": ("Category 9: score_before_experts", _score_before_tests),
}


def build_all_tests():
    all_tests = {}
    for cat_key, (cat_name, builder) in CATEGORY_BUILDERS.items():
        for name, spec in builder().items():
            spec["category"] = cat_key
            spec["category_name"] = cat_name
            all_tests[name] = spec
    return all_tests


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LLEP Hook Comprehensive Tests")
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Run a specific category (topk, alpha, min_tokens, lambda, experts, "
        "tokens, dims, backward, score_before, parity).",
    )
    parser.add_argument(
        "--test", type=str, default=None, help="Run a specific test by name."
    )
    parser.add_argument(
        "--list", action="store_true", help="List available tests and exit."
    )
    args = parser.parse_args()

    rank, world_size = setup()
    all_tests = build_all_tests()

    if args.list:
        if rank == 0:
            print("Available categories:")
            for cat_key, (cat_name, _) in CATEGORY_BUILDERS.items():
                print(f"  {cat_key}: {cat_name}")
            print(f"\nAvailable tests ({len(all_tests)} total):")
            current_cat = None
            for name, spec in all_tests.items():
                if spec["category"] != current_cat:
                    current_cat = spec["category"]
                    print(f"\n  === {spec['category_name']} ===")
                print(f"    {name} (min_gpus={spec.get('min_gpus', 2)})")
        dist.destroy_process_group()
        sys.exit(0)

    # Filter
    if args.test:
        if args.test not in all_tests:
            if rank == 0:
                print(f"Unknown test: {args.test}")
            dist.destroy_process_group()
            sys.exit(1)
        tests_to_run = {args.test: all_tests[args.test]}
    elif args.category:
        if args.category not in CATEGORY_BUILDERS:
            if rank == 0:
                print(f"Unknown category: {args.category}")
                print(f"Available: {list(CATEGORY_BUILDERS.keys())}")
            dist.destroy_process_group()
            sys.exit(1)
        tests_to_run = {
            k: v for k, v in all_tests.items() if v["category"] == args.category
        }
    else:
        tests_to_run = all_tests

    if rank == 0:
        print(f"\n{'#'*70}")
        print(f"  LLEP Hook Comprehensive Test Suite")
        print(f"  world_size={world_size}")
        print(f"{'#'*70}")

    passed = 0
    failed = 0
    skipped = 0
    errors = []
    current_cat = None

    for name, spec in tests_to_run.items():
        if rank == 0 and spec.get("category") != current_cat:
            current_cat = spec["category"]
            print(f"\n=== {spec['category_name']} ===")

        min_gpus = spec.get("min_gpus", 2)
        if world_size < min_gpus:
            if rank == 0:
                print(f"  [SKIP] {name}: needs {min_gpus} GPUs (have {world_size})")
            skipped += 1
            continue

        try:
            max_diff, test_passed, bwd_info = spec["fn"](**spec["kwargs"])

            if test_passed:
                passed += 1
                if rank == 0:
                    msg = f"  [PASS] {name}: max_diff={max_diff:.6f}"
                    if bwd_info:
                        msg += f" grad_max={bwd_info['grad_max']:.6f}"
                    print(msg)
            else:
                failed += 1
                atol = spec["kwargs"].get("atol", 1e-2)
                err_msg = f"max_diff={max_diff:.6f} > atol={atol}"
                if bwd_info and not bwd_info["passed"]:
                    err_msg += f" | grad_max={bwd_info['grad_max']:.6f}"
                errors.append((name, err_msg))
                if rank == 0:
                    print(f"  [FAIL] {name}: {err_msg}")

        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            if rank == 0:
                print(f"  [FAIL] {name}: EXCEPTION")
                traceback.print_exc()
            dist.barrier()

    if rank == 0:
        total = passed + failed + skipped
        print(f"\n{'='*70}")
        print(
            f"  Results: {passed} passed, {failed} failed, {skipped} skipped out of {total} tests"
        )
        if errors:
            print(f"\n  Failed:")
            for name, err in errors:
                print(f"    {name}: {err}")
        print(f"{'='*70}\n")

    dist.destroy_process_group()
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
