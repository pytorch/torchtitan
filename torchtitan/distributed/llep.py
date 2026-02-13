# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Least-Loaded Expert Parallelism (LLEP) for torchtitan.

Adapted from the Salesforce LLEP implementation for SwiGLU-based MoE
architectures (DeepSeek-V3, Kimi-K2, Qwen3, etc.).

Reference: "Least-Loaded Expert Parallelism: Load Balancing An Imbalanced
Mixture-of-Experts" (Nguyen et al., Salesforce AI Research)

Key differences from the original GPT-OSS implementation:
- SwiGLU activation: silu(x@w1) * (x@w3) @ w2 (no bias)
- Separate w1/w2/w3 weight tensors (not fused gate_up)
- Uses torch._grouped_mm for batched expert computation
- Integrates with torchtitan's DTensor-based Expert Parallelism
"""

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from torchtitan.tools.logging import logger


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class WeightTransferPlan:
    """Describes a single P2P weight transfer between two GPUs."""

    expert_id: int  # Global expert ID
    src_rank: int  # Rank that owns the weight (native)
    dst_rank: int  # Rank that will receive the weight (helper)
    token_start: int  # Start token index for dst_rank to process
    token_end: int  # End token index for dst_rank to process


@dataclass
class LLEPPlan:
    """Complete LPT plan with routing and weight transfer info."""

    # expert_id -> [(gpu_id, token_start, token_end), ...]
    lpt_plan: dict[int, list[tuple[int, int, int]]]
    # Weight transfers needed (globally consistent)
    weight_transfers: list[WeightTransferPlan]
    # gpu_loads[gpu_id] = total tokens assigned to that GPU
    gpu_loads: torch.Tensor
    # For this rank: (expert_id, dst_rank) pairs to send
    weights_to_send: list[tuple[int, int]]
    # For this rank: (expert_id, src_rank) pairs to receive
    weights_to_receive: list[tuple[int, int]]


# ---------------------------------------------------------------------------
# LPT Planning Algorithm (architecture-agnostic)
# ---------------------------------------------------------------------------
def compute_llep_lpt_plan(
    global_expert_counts: torch.Tensor,  # (num_experts,)
    ep_size: int,
    ep_rank: int,
    num_local_experts: int,
    max_tokens_factor: float = 1.1,
    min_tokens_per_gemm: int = 1024,
) -> LLEPPlan:
    """
    Compute LPT (Longest Processing Time) plan for LLEP.

    Implements Algorithm 2 (LLA) from the LLEP paper.
    Key insight: when selecting a helper GPU, account for "effective load"
    which includes both assigned_load + pending_native_load.

    Args:
        global_expert_counts: Global token counts per expert.
        ep_size: Number of EP ranks.
        ep_rank: Current rank.
        num_local_experts: Experts per GPU.
        max_tokens_factor: Capacity factor (alpha in paper).
        min_tokens_per_gemm: Minimum tokens per GEMM to avoid overhead.

    Returns:
        LLEPPlan with routing plan and weight transfer info.
    """
    num_experts = global_expert_counts.size(0)
    device = global_expert_counts.device

    # Single D2H transfer — avoid per-element .item() syncs
    expert_counts_cpu = global_expert_counts.cpu().tolist()

    total_tokens = sum(expert_counts_cpu)
    balanced_tokens = total_tokens // ep_size if ep_size > 0 else total_tokens
    max_tokens_per_gpu = (
        int(max_tokens_factor * balanced_tokens) if balanced_tokens > 0 else total_tokens
    )
    max_tokens_per_gpu = max(max_tokens_per_gpu, 1)

    # Compute native load per GPU
    native_load_per_gpu = [0] * ep_size
    for expert_id in range(num_experts):
        native_gpu = expert_id // num_local_experts
        native_load_per_gpu[native_gpu] += expert_counts_cpu[expert_id]

    pending_native_load = list(native_load_per_gpu)
    assigned_load = [0] * ep_size

    # Sort experts by token count (LPT ordering)
    expert_counts_list = [
        (e, expert_counts_cpu[e]) for e in range(num_experts)
    ]
    expert_counts_sorted = sorted(expert_counts_list, key=lambda x: -x[1])

    lpt_plan: dict[int, list[tuple[int, int, int]]] = {}
    weight_transfers: list[WeightTransferPlan] = []

    for expert_id, expert_tokens in expert_counts_sorted:
        if expert_tokens == 0:
            continue

        native_gpu = expert_id // num_local_experts

        # Remove from pending native load (this expert is now being processed)
        pending_native_load[native_gpu] -= expert_tokens

        def get_effective_load(gpu_id):
            return assigned_load[gpu_id] + pending_native_load[gpu_id]

        native_available = max_tokens_per_gpu - get_effective_load(native_gpu)
        assignments = []

        if native_available >= expert_tokens:
            # Case 1: Native GPU can handle all tokens
            assignments.append((native_gpu, 0, expert_tokens))
            assigned_load[native_gpu] += expert_tokens
        elif native_available > 0:
            # Case 2: Partial native + spill to helpers
            native_chunk = min(native_available, expert_tokens)
            assignments.append((native_gpu, 0, native_chunk))
            assigned_load[native_gpu] += native_chunk

            remaining = expert_tokens - native_chunk
            token_offset = native_chunk

            while remaining > 0:
                other_gpus = []
                for g in range(ep_size):
                    if g == native_gpu:
                        continue
                    eff = get_effective_load(g)
                    avail = max_tokens_per_gpu - eff
                    other_gpus.append((g, eff, avail))
                other_gpus.sort(key=lambda x: x[1])

                if not other_gpus:
                    old_end = assignments[0][2]
                    assignments[0] = (native_gpu, 0, old_end + remaining)
                    assigned_load[native_gpu] += remaining
                    break

                assigned_this_round = False
                for helper_gpu, _, helper_available in other_gpus:
                    if helper_available <= 0:
                        continue
                    chunk = min(remaining, helper_available)
                    if chunk < min_tokens_per_gemm and remaining > chunk:
                        continue

                    assignments.append(
                        (helper_gpu, token_offset, token_offset + chunk)
                    )
                    assigned_load[helper_gpu] += chunk
                    weight_transfers.append(
                        WeightTransferPlan(
                            expert_id=expert_id,
                            src_rank=native_gpu,
                            dst_rank=helper_gpu,
                            token_start=token_offset,
                            token_end=token_offset + chunk,
                        )
                    )
                    token_offset += chunk
                    remaining -= chunk
                    assigned_this_round = True
                    break

                if not assigned_this_round:
                    helper_gpu = other_gpus[0][0]
                    assignments.append(
                        (helper_gpu, token_offset, token_offset + remaining)
                    )
                    assigned_load[helper_gpu] += remaining
                    weight_transfers.append(
                        WeightTransferPlan(
                            expert_id=expert_id,
                            src_rank=native_gpu,
                            dst_rank=helper_gpu,
                            token_start=token_offset,
                            token_end=token_offset + remaining,
                        )
                    )
                    remaining = 0
        else:
            # Case 3: Native GPU at/over capacity, spill everything
            other_gpus = []
            for g in range(ep_size):
                if g == native_gpu:
                    continue
                eff = get_effective_load(g)
                avail = max_tokens_per_gpu - eff
                other_gpus.append((g, eff, avail))
            other_gpus.sort(key=lambda x: x[1])

            remaining = expert_tokens
            token_offset = 0

            for helper_gpu, _, helper_available in other_gpus:
                if remaining <= 0:
                    break
                if helper_available <= 0:
                    continue
                chunk = min(remaining, helper_available)
                if chunk < min_tokens_per_gemm and remaining > chunk:
                    continue

                assignments.append(
                    (helper_gpu, token_offset, token_offset + chunk)
                )
                assigned_load[helper_gpu] += chunk
                weight_transfers.append(
                    WeightTransferPlan(
                        expert_id=expert_id,
                        src_rank=native_gpu,
                        dst_rank=helper_gpu,
                        token_start=token_offset,
                        token_end=token_offset + chunk,
                    )
                )
                token_offset += chunk
                remaining -= chunk

            if remaining > 0:
                if other_gpus:
                    helper_gpu = other_gpus[0][0]
                    assignments.append(
                        (helper_gpu, token_offset, token_offset + remaining)
                    )
                    assigned_load[helper_gpu] += remaining
                    weight_transfers.append(
                        WeightTransferPlan(
                            expert_id=expert_id,
                            src_rank=native_gpu,
                            dst_rank=helper_gpu,
                            token_start=token_offset,
                            token_end=token_offset + remaining,
                        )
                    )
                else:
                    assignments.append((native_gpu, 0, expert_tokens))
                    assigned_load[native_gpu] += expert_tokens

        lpt_plan[expert_id] = assignments

    weights_to_send = []
    weights_to_receive = []
    for wt in weight_transfers:
        if wt.src_rank == ep_rank:
            weights_to_send.append((wt.expert_id, wt.dst_rank))
        if wt.dst_rank == ep_rank:
            weights_to_receive.append((wt.expert_id, wt.src_rank))

    return LLEPPlan(
        lpt_plan=lpt_plan,
        weight_transfers=weight_transfers,
        gpu_loads=torch.tensor(assigned_load, dtype=torch.int64, device=device),
        weights_to_send=weights_to_send,
        weights_to_receive=weights_to_receive,
    )


# ---------------------------------------------------------------------------
# GPU imbalance ratio
# ---------------------------------------------------------------------------
def compute_gpu_imbalance_ratio(
    global_expert_counts: torch.Tensor,
    ep_size: int,
    num_local_experts: int,
) -> float:
    """Compute max_gpu_load / mean_gpu_load. Returns 1.0 for perfect balance."""
    # Reshape and sum to get per-GPU loads in one op (no per-element indexing)
    num_experts = global_expert_counts.size(0)
    gpu_loads = global_expert_counts.view(ep_size, num_local_experts).sum(dim=1).float()
    mean_load = gpu_loads.mean()
    if mean_load == 0:
        return 1.0
    return (gpu_loads.max() / mean_load).item()


# ---------------------------------------------------------------------------
# P2P Weight Transfer (for w1, w2, w3 - no bias)
# ---------------------------------------------------------------------------
def transfer_expert_weights(
    ep_rank: int,
    ep_group,
    plan: LLEPPlan,
    w1_local: torch.Tensor,  # (num_local_experts, hidden_dim, dim)
    w2_local: torch.Tensor,  # (num_local_experts, dim, hidden_dim)
    w3_local: torch.Tensor,  # (num_local_experts, hidden_dim, dim)
    num_local_experts: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """
    Transfer expert weights via P2P for LLEP weight spilling.

    Sends native expert weights to helper GPUs and receives foreign
    expert weights from their native GPUs.

    Returns:
        Tuple of (foreign_w1, foreign_w2, foreign_w3) dicts mapping
        global_expert_id -> weight tensor.
    """
    foreign_w1: dict[int, torch.Tensor] = {}
    foreign_w2: dict[int, torch.Tensor] = {}
    foreign_w3: dict[int, torch.Tensor] = {}

    w_shape = list(w1_local[0].shape)
    bytes_per_weight = w1_local[0].nelement() * w1_local[0].element_size()

    # Build P2P op list for batch_isend_irecv (uses NCCL backend)
    p2p_ops = []

    # Send native expert weights to helpers
    for expert_id, dst_rank in plan.weights_to_send:
        local_idx = expert_id - ep_rank * num_local_experts
        logger.info(
            f"[LLEP-P2P] rank {ep_rank} SENDING expert {expert_id} "
            f"(local_idx={local_idx}) → rank {dst_rank} "
            f"| 3 tensors x {w_shape} = {3 * bytes_per_weight / 1024:.1f} KB"
        )
        p2p_ops.append(dist.P2POp(dist.isend, w1_local[local_idx].contiguous(), group_peer=dst_rank, group=ep_group))
        p2p_ops.append(dist.P2POp(dist.isend, w2_local[local_idx].contiguous(), group_peer=dst_rank, group=ep_group))
        p2p_ops.append(dist.P2POp(dist.isend, w3_local[local_idx].contiguous(), group_peer=dst_rank, group=ep_group))

    # Receive foreign expert weights from owners
    for expert_id, src_rank in plan.weights_to_receive:
        logger.info(
            f"[LLEP-P2P] rank {ep_rank} RECEIVING expert {expert_id} "
            f"← rank {src_rank} "
            f"| 3 tensors x {w_shape} = {3 * bytes_per_weight / 1024:.1f} KB"
        )
        # Allocate receive buffers matching the local expert shape
        recv_w1 = torch.empty_like(w1_local[0])
        recv_w2 = torch.empty_like(w2_local[0])
        recv_w3 = torch.empty_like(w3_local[0])
        p2p_ops.append(dist.P2POp(dist.irecv, recv_w1, group_peer=src_rank, group=ep_group))
        p2p_ops.append(dist.P2POp(dist.irecv, recv_w2, group_peer=src_rank, group=ep_group))
        p2p_ops.append(dist.P2POp(dist.irecv, recv_w3, group_peer=src_rank, group=ep_group))
        foreign_w1[expert_id] = recv_w1
        foreign_w2[expert_id] = recv_w2
        foreign_w3[expert_id] = recv_w3

    # Execute all P2P ops as a batch (uses NCCL backend)
    if p2p_ops:
        handles = dist.batch_isend_irecv(p2p_ops)
        for h in handles:
            h.wait()

    if plan.weights_to_send or plan.weights_to_receive:
        logger.info(
            f"[LLEP-P2P] rank {ep_rank} transfer complete: "
            f"sent {len(plan.weights_to_send)} experts, "
            f"received {len(plan.weights_to_receive)} experts, "
            f"now holding {len(foreign_w1)} foreign experts"
        )

    return foreign_w1, foreign_w2, foreign_w3


# ---------------------------------------------------------------------------
# Token Routing based on LPT Plan
# ---------------------------------------------------------------------------
def assign_tokens_to_gpus(
    top_scores: torch.Tensor,  # (num_tokens, top_k)
    selected_experts: torch.Tensor,  # (num_tokens, top_k)
    lpt_plan: dict[int, list[tuple[int, int, int]]],
    ep_size: int,
    ep_rank: int,
    num_local_experts: int,
    ep_group,
) -> tuple[
    torch.Tensor,  # tokens to process on this rank (num_recv_tokens, dim placeholder)
    torch.Tensor,  # routing weights for received tokens
    torch.Tensor,  # expert ids for received tokens (global)
    list[int],  # input split sizes
    list[int],  # output split sizes
    torch.Tensor,  # undo indices for reverse routing
]:
    """
    Assign token-expert pairs to GPUs based on the LPT plan, then
    perform AllToAll to route tokens to their assigned GPUs.

    When lpt_plan is empty (balanced case), uses default EP routing
    where each token goes to the expert's native GPU.
    """
    device = top_scores.device
    num_tokens, top_k = selected_experts.shape

    # Expand to (num_tokens * top_k) flat view
    flat_experts = selected_experts.view(-1)  # (num_tokens * top_k,)
    flat_scores = top_scores.view(-1)  # (num_tokens * top_k,)
    total_slots = flat_experts.size(0)

    # Assign each token-expert slot to a target GPU
    target_gpus = torch.empty(total_slots, dtype=torch.int64, device=device)

    if lpt_plan:
        # Build cumulative token counters per expert to map token positions
        # to LPT plan assignments
        expert_token_counters: dict[int, int] = {}
        # We need per-rank expert counts to properly index
        # Gather counts from all ranks
        local_counts = torch.bincount(
            flat_experts.to(torch.int64),
            minlength=ep_size * num_local_experts,
        )
        all_counts_list = [torch.zeros_like(local_counts) for _ in range(ep_size)]
        dist.all_gather(all_counts_list, local_counts, group=ep_group)

        # For each token-expert pair, determine which GPU it goes to
        # based on its position in the global token stream for that expert
        # We need global prefix sums per expert across ranks
        global_prefix = torch.zeros_like(local_counts)
        for r in range(ep_rank):
            global_prefix += all_counts_list[r]

        # Count per expert locally to compute local offsets
        local_expert_offset = torch.zeros_like(local_counts)

        for i in range(total_slots):
            eid = flat_experts[i].item()
            # Global position = prefix from earlier ranks + local offset
            local_off = local_expert_offset[eid].item()
            global_pos = int(global_prefix[eid].item()) + local_off
            local_expert_offset[eid] += 1

            # Look up which GPU handles this position in the LPT plan
            if eid in lpt_plan:
                assigned_gpu = -1
                for gpu_id, tok_start, tok_end in lpt_plan[eid]:
                    if tok_start <= global_pos < tok_end:
                        assigned_gpu = gpu_id
                        break
                if assigned_gpu == -1:
                    # Fallback to native GPU
                    assigned_gpu = eid // num_local_experts
                target_gpus[i] = assigned_gpu
            else:
                target_gpus[i] = eid // num_local_experts
    else:
        # Default routing: each token goes to expert's native GPU
        for i in range(total_slots):
            eid = flat_experts[i].item()
            target_gpus[i] = eid // num_local_experts

    # Sort by target GPU for AllToAll
    sorted_indices = torch.argsort(target_gpus, stable=True)
    undo_indices = torch.argsort(sorted_indices)

    sorted_scores = flat_scores[sorted_indices]
    sorted_experts = flat_experts[sorted_indices]
    sorted_targets = target_gpus[sorted_indices]

    # Compute split sizes
    input_split_sizes = []
    for r in range(ep_size):
        count = (sorted_targets == r).sum().item()
        input_split_sizes.append(count)

    # Exchange split sizes
    input_sizes_tensor = torch.tensor(input_split_sizes, dtype=torch.int64, device=device)
    all_input_sizes = [torch.zeros(ep_size, dtype=torch.int64, device=device) for _ in range(ep_size)]
    dist.all_gather(all_input_sizes, input_sizes_tensor, group=ep_group)
    output_split_sizes = [all_input_sizes[r][ep_rank].item() for r in range(ep_size)]

    return (
        sorted_scores,
        sorted_experts,
        input_split_sizes,
        output_split_sizes,
        sorted_indices,
        undo_indices,
    )


# ---------------------------------------------------------------------------
# SwiGLU FFN for native + foreign experts
# ---------------------------------------------------------------------------
def llep_swiglu_ffn(
    x: torch.Tensor,  # (num_tokens, dim)
    expert_ids: torch.Tensor,  # (num_tokens,) global expert IDs
    w1_local: torch.Tensor,  # (num_local_experts, hidden_dim, dim)
    w2_local: torch.Tensor,  # (num_local_experts, dim, hidden_dim)
    w3_local: torch.Tensor,  # (num_local_experts, hidden_dim, dim)
    foreign_w1: dict[int, torch.Tensor],
    foreign_w2: dict[int, torch.Tensor],
    foreign_w3: dict[int, torch.Tensor],
    ep_rank: int,
    num_local_experts: int,
) -> torch.Tensor:
    """
    Run SwiGLU FFN on tokens using both native and foreign expert weights.

    Groups tokens by expert, then for each expert:
      h = silu(x @ w1.T) * (x @ w3.T)
      out = h @ w2.T

    Uses per-expert GEMM (not grouped_mm) to handle mixed native/foreign weights.
    """
    if x.numel() == 0:
        return torch.empty(0, w2_local.shape[1], device=x.device, dtype=x.dtype)

    num_tokens, dim = x.shape
    native_start = ep_rank * num_local_experts
    native_end = native_start + num_local_experts

    # Sort tokens by expert for contiguous access
    sorted_expert_ids, sort_perm = expert_ids.sort(stable=True)
    x_sorted = x[sort_perm]

    # Compute segment boundaries
    unique_experts, counts = torch.unique_consecutive(sorted_expert_ids, return_counts=True)
    offsets = torch.zeros(len(counts) + 1, dtype=torch.int64, device=x.device)
    offsets[1:] = counts.cumsum(0)

    out_sorted = torch.empty_like(x_sorted)

    for idx in range(len(unique_experts)):
        eid = unique_experts[idx].item()
        start = offsets[idx].item()
        end = offsets[idx + 1].item()
        x_slice = x_sorted[start:end]

        if native_start <= eid < native_end:
            local_idx = eid - native_start
            w1 = w1_local[local_idx]
            w2 = w2_local[local_idx]
            w3 = w3_local[local_idx]
        elif eid in foreign_w1:
            w1 = foreign_w1[eid]
            w2 = foreign_w2[eid]
            w3 = foreign_w3[eid]
        else:
            # Should not happen - fill with zeros as safety
            out_sorted[start:end] = 0
            continue

        # SwiGLU: silu(x @ w1.T) * (x @ w3.T) @ w2.T
        h = F.silu(x_slice.to(torch.bfloat16) @ w1.to(torch.bfloat16).T)
        h = h * (x_slice.to(torch.bfloat16) @ w3.to(torch.bfloat16).T)
        out_sorted[start:end] = (h @ w2.to(torch.bfloat16).T).to(x.dtype)

    # Unsort
    inverse_perm = sort_perm.argsort()
    return out_sorted[inverse_perm]


# ---------------------------------------------------------------------------
# Main LLEP forward for torchtitan MoE
# ---------------------------------------------------------------------------
def llep_moe_forward(
    hidden_states: torch.Tensor,  # (num_tokens, dim)
    top_scores: torch.Tensor,  # (num_tokens, top_k) - routing weights
    selected_experts_indices: torch.Tensor,  # (num_tokens, top_k) - expert IDs
    w1: torch.Tensor,  # DTensor or local: (num_local_experts, hidden_dim, dim)
    w2: torch.Tensor,  # DTensor or local: (num_local_experts, dim, hidden_dim)
    w3: torch.Tensor,  # DTensor or local: (num_local_experts, hidden_dim, dim)
    ep_group,
    num_experts: int,
    score_before_experts: bool = False,
    max_tokens_factor: float = 1.1,
    min_tokens_per_gemm: int = 1024,
    adaptive_threshold: float = 0.0,
) -> torch.Tensor:
    """
    LLEP MoE forward pass for SwiGLU-based architectures.

    Implements Algorithm 4 from the LLEP paper, adapted for torchtitan:
    1. Aggregate global expert counts
    2. Compute LPT assignment plan
    3. P2P transfer expert weights to helper GPUs
    4. AllToAll route tokens to assigned GPUs
    5. Run SwiGLU FFN on native + foreign experts
    6. AllToAll route outputs back
    7. Apply routing scores and aggregate

    Args:
        hidden_states: Input tokens (num_tokens, dim).
        top_scores: Routing scores (num_tokens, top_k).
        selected_experts_indices: Expert IDs (num_tokens, top_k).
        w1, w2, w3: Expert weights (possibly DTensors).
        ep_group: EP process group.
        num_experts: Total number of global experts.
        score_before_experts: Apply scores before or after expert computation.
        max_tokens_factor: Alpha parameter for max GPU capacity.
        min_tokens_per_gemm: Minimum tokens per GEMM operation.
        adaptive_threshold: Lambda parameter. If > 0 and imbalance < threshold,
            skip LLEP and use standard EP routing.

    Returns:
        Aggregated routed output (num_tokens, dim).
    """
    from torch.distributed.tensor import DTensor

    # Unwrap DTensors to local
    w1_local = w1.to_local() if isinstance(w1, DTensor) else w1
    w2_local = w2.to_local() if isinstance(w2, DTensor) else w2
    w3_local = w3.to_local() if isinstance(w3, DTensor) else w3

    ep_rank = dist.get_rank(group=ep_group)
    ep_size = dist.get_world_size(group=ep_group)
    num_local_experts = num_experts // ep_size

    device = hidden_states.device

    # Ensure weights are on the same device as hidden_states (handles CPU offload)
    if w1_local.device != device:
        w1_local = w1_local.to(device)
        w2_local = w2_local.to(device)
        w3_local = w3_local.to(device)
    dtype = hidden_states.dtype
    num_tokens, dim = hidden_states.shape
    top_k = selected_experts_indices.shape[1]

    # Override from env
    max_tokens_factor = float(
        os.environ.get("EP_MAX_TOKENS_FACTOR", str(max_tokens_factor))
    )
    min_tokens_per_gemm = int(
        os.environ.get("EP_MIN_TOKENS_PER_GEMM", str(min_tokens_per_gemm))
    )

    # Step 1: Aggregate global expert counts
    local_expert_counts = torch.bincount(
        selected_experts_indices.view(-1).to(torch.int64),
        minlength=num_experts,
    ).to(torch.int64)

    global_expert_counts = local_expert_counts.clone()
    dist.all_reduce(global_expert_counts, op=dist.ReduceOp.SUM, group=ep_group)

    # Step 2: Adaptive threshold check
    imbalance = compute_gpu_imbalance_ratio(
        global_expert_counts, ep_size, num_local_experts
    )
    use_lpt = True
    if adaptive_threshold > 0:
        use_lpt = imbalance >= adaptive_threshold

    # Single D2H for logging — reused by compute_llep_lpt_plan too
    expert_counts_list = global_expert_counts.cpu().tolist()

    if ep_rank == 0:
        logger.info(
            f"[LLEP] tokens={num_tokens} top_k={top_k} "
            f"expert_counts={expert_counts_list} "
            f"imbalance={imbalance:.2f} use_lpt={use_lpt}"
        )

    # Step 3: Compute LPT plan
    if use_lpt:
        plan = compute_llep_lpt_plan(
            global_expert_counts,
            ep_size,
            ep_rank,
            num_local_experts,
            max_tokens_factor=max_tokens_factor,
            min_tokens_per_gemm=min_tokens_per_gemm,
        )
    else:
        plan = LLEPPlan(
            lpt_plan={},
            weight_transfers=[],
            gpu_loads=torch.zeros(ep_size, dtype=torch.int64, device=device),
            weights_to_send=[],
            weights_to_receive=[],
        )

    if ep_rank == 0:
        # Compute naive (before LLEP) GPU loads from CPU list
        naive_loads = [0] * ep_size
        for eid in range(num_experts):
            naive_loads[eid // num_local_experts] += expert_counts_list[eid]
        naive_max = max(naive_loads)
        naive_mean = sum(naive_loads) / len(naive_loads) if naive_loads else 1
        naive_ratio = naive_max / naive_mean if naive_mean > 0 else 1.0

        llep_loads = plan.gpu_loads.tolist()
        llep_max = max(llep_loads) if llep_loads else 0
        llep_mean = sum(llep_loads) / len(llep_loads) if llep_loads else 1
        llep_ratio = llep_max / llep_mean if llep_mean > 0 else 1.0

        logger.info(
            f"[LLEP] BEFORE balance: gpu_loads={naive_loads} "
            f"imbalance={naive_ratio:.2f} (max/mean={naive_max}/{naive_mean:.0f})"
        )
        logger.info(
            f"[LLEP] AFTER  balance: gpu_loads={llep_loads} "
            f"imbalance={llep_ratio:.2f} (max/mean={llep_max}/{llep_mean:.0f}) "
            f"| weight_transfers={len(plan.weight_transfers)} "
            f"send={plan.weights_to_send} recv={plan.weights_to_receive}"
        )

    # Step 4: Barrier before P2P to prevent cross-layer deadlock
    dist.barrier(group=ep_group)

    # Step 5: Transfer expert weights
    foreign_w1, foreign_w2, foreign_w3 = transfer_expert_weights(
        ep_rank, ep_group, plan,
        w1_local, w2_local, w3_local,
        num_local_experts,
    )

    # Step 6: Assign tokens to GPUs and route via AllToAll
    (
        sorted_scores,
        sorted_experts,
        input_split_sizes,
        output_split_sizes,
        sorted_indices,
        undo_indices,
    ) = assign_tokens_to_gpus(
        top_scores, selected_experts_indices,
        plan.lpt_plan, ep_size, ep_rank, num_local_experts, ep_group,
    )

    # Expand hidden states for top_k and sort
    hidden_topk = hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, dim)
    sorted_hidden = hidden_topk[sorted_indices]

    # AllToAll dispatch tokens
    total_send = sum(input_split_sizes)
    total_recv = sum(output_split_sizes)

    if total_send > 0 or total_recv > 0:
        recv_hidden = torch.empty(total_recv, dim, device=device, dtype=dtype)
        recv_scores = torch.empty(total_recv, device=device, dtype=sorted_scores.dtype)
        recv_experts = torch.empty(total_recv, device=device, dtype=sorted_experts.dtype)

        # Use all_to_all_single for each tensor
        dist.all_to_all_single(
            recv_hidden, sorted_hidden,
            output_split_sizes, input_split_sizes,
            group=ep_group,
        )
        dist.all_to_all_single(
            recv_scores, sorted_scores,
            output_split_sizes, input_split_sizes,
            group=ep_group,
        )
        dist.all_to_all_single(
            recv_experts, sorted_experts.to(torch.int64),
            output_split_sizes, input_split_sizes,
            group=ep_group,
        )
    else:
        recv_hidden = torch.empty(0, dim, device=device, dtype=dtype)
        recv_scores = torch.empty(0, device=device, dtype=sorted_scores.dtype)
        recv_experts = torch.empty(0, device=device, dtype=sorted_experts.dtype)

    # Step 7: Apply routing scores before experts (if configured)
    if score_before_experts and recv_hidden.numel() > 0:
        recv_hidden = (
            recv_hidden.to(torch.float32) * recv_scores.reshape(-1, 1)
        ).to(dtype)

    # Step 8: Run SwiGLU FFN on received tokens
    if recv_hidden.numel() > 0:
        recv_output = llep_swiglu_ffn(
            recv_hidden,
            recv_experts.to(torch.int64),
            w1_local, w2_local, w3_local,
            foreign_w1, foreign_w2, foreign_w3,
            ep_rank, num_local_experts,
        )
    else:
        recv_output = torch.empty(0, dim, device=device, dtype=dtype)

    # Step 9: Apply routing scores after experts (if configured)
    if not score_before_experts and recv_output.numel() > 0:
        recv_output = (
            recv_output.to(torch.float32) * recv_scores.reshape(-1, 1)
        ).to(dtype)

    # Step 10: AllToAll combine - route outputs back
    send_output = torch.empty(total_send, dim, device=device, dtype=dtype)
    if total_send > 0 or total_recv > 0:
        dist.all_to_all_single(
            send_output, recv_output,
            input_split_sizes, output_split_sizes,
            group=ep_group,
        )

    # Step 11: Unsort and scatter-add back to original token positions
    unsorted_output = send_output[undo_indices]
    # Reshape from (num_tokens * top_k, dim) back to (num_tokens, dim)
    unsorted_output = unsorted_output.view(num_tokens, top_k, dim).sum(dim=1)

    return unsorted_output
