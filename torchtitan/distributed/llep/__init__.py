# Copyright (c) Nous Research and affiliates.
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
- Integrates with torchtitan's DTensor-based Expert Parallelism

Performance optimizations over the initial port:
- Vectorized token assignment: O(num_experts) not O(num_tokens * top_k)
- Merged AllToAll: single A2A call instead of three separate calls
- Async weight transfer: P2P overlapped with A2A input routing
- No redundant dtype casts in FFN computation
"""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist

from torchtitan.tools.logging import logger

# Debug logging (set LLEP_DEBUG=1 to enable per-step verbose logging)
LLEP_DEBUG = os.environ.get("LLEP_DEBUG", "0") == "1"

# Step counter for LLEP_DEBUG logging (module-level, incremented per dispatch call)
_llep_step_counter = 0

# Merge hidden+scores+expert_ids into single A2A call (set 0 to disable)
LLEP_MERGE_A2A = os.environ.get("LLEP_MERGE_A2A", "1") == "1"


# Token alignment for grouped_mm (must be 8 for bf16)
_TOKEN_ALIGN = 8

# Threshold for vectorized pad/unpad: use for-loop below this, vectorized above.
# Benchmarked on B200: forloop faster for <32 experts, vectorized wins at 64+.
_PAD_VECTORIZE_THRESHOLD = 32

# Lazy-cached Triton pad/unpad functions
_triton_pad_fn = None
_triton_unpad_fn = None

# Lazy-cached vectorized send_matrix function
_send_matrix_fn = None

# Cached env overrides (read once at import, not per call)
_ENV_MAX_TOKENS_FACTOR = os.environ.get("EP_MAX_TOKENS_FACTOR")
_ENV_MIN_TOKENS_PER_GEMM = os.environ.get("EP_MIN_TOKENS_PER_GEMM")
_ENV_ADAPTIVE_THRESHOLD = os.environ.get("EP_ADAPTIVE_THRESHOLD")


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


@dataclass
class LLEPState:
    """State carried between LLEP dispatch (pre-hook) and combine (post-hook).

    This stores the
    intermediate state needed by the combine step and the compute step.
    """

    # Routing state for reverse A2A
    input_split_sizes: list[int]
    output_split_sizes: list[int]
    undo_indices: torch.Tensor  # to unsort tokens back after combine

    # Expert sort permutation applied after A2A receive (must undo before reverse A2A)
    recv_sort_perm: Optional[torch.Tensor]

    # Expert IDs of received tokens (sorted by expert after A2A)
    recv_experts: torch.Tensor  # (num_recv_tokens,) global expert IDs, sorted

    # LPT plan for weight transfer
    plan: LLEPPlan

    # EP group info
    ep_group: object  # dist.ProcessGroup
    ep_rank: int
    ep_size: int
    num_local_experts: int
    num_experts: int

    # Total send/recv counts (for edge case handling)
    total_send: int
    total_recv: int

    # Original input shape for output reconstruction
    num_tokens: int  # original num_tokens (before top_k expansion)
    top_k: int
    dim: int
    dtype: torch.dtype

    # Set during forward (prepare_weights) for use in combine
    padded_counts: Optional[torch.Tensor] = None  # padded token counts per expert
    valid_mask: Optional[
        list
    ] = None  # valid_mask[i] True if expert i has valid weights
    gradient_anchor: Optional[torch.Tensor] = None  # gradient anchor for backward P2P


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

    # Single D2H transfer -- avoid per-element .item() syncs
    expert_counts_cpu = global_expert_counts.cpu().tolist()

    total_tokens = sum(expert_counts_cpu)
    balanced_tokens = total_tokens // ep_size if ep_size > 0 else total_tokens
    max_tokens_per_gpu = (
        int(max_tokens_factor * balanced_tokens)
        if balanced_tokens > 0
        else total_tokens
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
    expert_counts_list = [(e, expert_counts_cpu[e]) for e in range(num_experts)]
    expert_counts_sorted = sorted(expert_counts_list, key=lambda x: -x[1])

    lpt_plan: dict[int, list[tuple[int, int, int]]] = {}
    weight_transfers: list[WeightTransferPlan] = []

    def get_effective_load(gpu_id):
        return assigned_load[gpu_id] + pending_native_load[gpu_id]

    for expert_id, expert_tokens in expert_counts_sorted:
        if expert_tokens == 0:
            continue

        native_gpu = expert_id // num_local_experts

        # Remove from pending native load (this expert is now being processed)
        pending_native_load[native_gpu] -= expert_tokens

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

                    assignments.append((helper_gpu, token_offset, token_offset + chunk))
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

                assignments.append((helper_gpu, token_offset, token_offset + chunk))
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
_triton_imbalance_fn = None


def compute_gpu_imbalance_ratio(
    global_expert_counts: torch.Tensor,
    ep_size: int,
    num_local_experts: int,
) -> float:
    """Compute max_gpu_load / mean_gpu_load. Returns 1.0 for perfect balance."""
    global _triton_imbalance_fn
    if _triton_imbalance_fn is None:
        try:
            from torchtitan.distributed.llep.kernels import triton_imbalance_ratio

            _triton_imbalance_fn = triton_imbalance_ratio
        except (ImportError, RuntimeError):
            _triton_imbalance_fn = False

    if _triton_imbalance_fn and global_expert_counts.is_cuda:
        return _triton_imbalance_fn(global_expert_counts, ep_size, num_local_experts)

    effective = ep_size * num_local_experts
    counts = (
        global_expert_counts[:effective]
        if global_expert_counts.size(0) > effective
        else global_expert_counts
    )
    gpu_loads = counts.view(ep_size, num_local_experts).sum(dim=1).float()
    mean_load = gpu_loads.mean()
    if mean_load == 0:
        return 1.0
    return (gpu_loads.max() / mean_load).item()


# ---------------------------------------------------------------------------
# Differentiable AllToAll (autograd-compatible)
# ---------------------------------------------------------------------------
class AllToAllAutograd(torch.autograd.Function):
    """Differentiable AllToAll wrapper. Backward performs the reverse A2A."""

    @staticmethod
    def forward(ctx, input, output_split_sizes, input_split_sizes, group):
        ctx.output_split_sizes = (
            list(output_split_sizes)
            if not isinstance(output_split_sizes, list)
            else output_split_sizes
        )
        ctx.input_split_sizes = (
            list(input_split_sizes)
            if not isinstance(input_split_sizes, list)
            else input_split_sizes
        )
        ctx.group = group

        total_recv = sum(ctx.output_split_sizes)
        output = torch.empty(
            total_recv, *input.shape[1:], device=input.device, dtype=input.dtype
        )
        dist.all_to_all_single(
            output,
            input.contiguous(),
            ctx.output_split_sizes,
            ctx.input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        total_recv_back = sum(ctx.input_split_sizes)
        grad_input = torch.empty(
            total_recv_back,
            *grad_output.shape[1:],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        # Reverse A2A: swap input/output split sizes
        dist.all_to_all_single(
            grad_input,
            grad_output.contiguous(),
            ctx.input_split_sizes,
            ctx.output_split_sizes,
            group=ctx.group,
        )
        return grad_input, None, None, None


def a2a_autograd(
    input: torch.Tensor,
    output_split_sizes: list[int],
    input_split_sizes: list[int],
    group,
) -> torch.Tensor:
    """Differentiable AllToAll. Wraps all_to_all_single with autograd support."""
    return AllToAllAutograd.apply(input, output_split_sizes, input_split_sizes, group)


# ---------------------------------------------------------------------------
# Differentiable P2P Weight Transfer (autograd-compatible, SwiGLU: w1/w2/w3)
# ---------------------------------------------------------------------------
class WeightTransferAutograd(torch.autograd.Function):
    """
    Differentiable P2P weight transfer for LLEP (SwiGLU: w1, w2, w3, no bias).

    Forward: sends local expert weights to helpers, receives foreign expert weights.
    Backward: sends foreign weight gradients back to owners, receives gradients.

    Returns stacked tensors (not dicts) for autograd compatibility,
    plus a mapping tensor and a gradient anchor scalar.

    The gradient anchor MUST be added to the final MoE output to ensure ALL ranks
    enter backward for this op (even ranks that do no foreign computation).
    """

    @staticmethod
    def forward(
        ctx,
        w1_local: torch.Tensor,
        w2_local: torch.Tensor,
        w3_local: torch.Tensor,
        weights_to_send_tensor: torch.Tensor,
        weights_to_receive_tensor: torch.Tensor,
        num_local_experts_tensor: torch.Tensor,
        num_experts_tensor: torch.Tensor,
        ep_group,
        return_handles,
    ):
        device = w1_local.device
        dtype = w1_local.dtype

        weights_to_send = weights_to_send_tensor.tolist()
        weights_to_receive = weights_to_receive_tensor.tolist()
        num_local_experts = int(num_local_experts_tensor.item())
        num_experts = int(num_experts_tensor.item())

        num_recv = len(weights_to_receive)

        # Mapping: global_expert_id -> stacked index (or -1)
        foreign_expert_id_mapping = torch.full(
            (num_experts,), -1, dtype=torch.long, device=device
        )
        for stacked_idx, (expert_id, _src_rank) in enumerate(weights_to_receive):
            foreign_expert_id_mapping[int(expert_id)] = stacked_idx

        # Allocate stacked receive buffers
        recv_w1 = torch.empty(num_recv, *w1_local.shape[1:], dtype=dtype, device=device)
        recv_w2 = torch.empty(num_recv, *w2_local.shape[1:], dtype=dtype, device=device)
        recv_w3 = torch.empty(num_recv, *w3_local.shape[1:], dtype=dtype, device=device)

        p2p_ops = []

        # Recv first (post recvs before sends for NCCL efficiency)
        # NOTE: use group_peer= (EP-local rank), not peer= (global rank)
        for stacked_idx, (expert_id, src_rank) in enumerate(weights_to_receive):
            p2p_ops.append(
                dist.P2POp(
                    dist.irecv,
                    recv_w1[stacked_idx],
                    group_peer=int(src_rank),
                    group=ep_group,
                )
            )
            p2p_ops.append(
                dist.P2POp(
                    dist.irecv,
                    recv_w2[stacked_idx],
                    group_peer=int(src_rank),
                    group=ep_group,
                )
            )
            p2p_ops.append(
                dist.P2POp(
                    dist.irecv,
                    recv_w3[stacked_idx],
                    group_peer=int(src_rank),
                    group=ep_group,
                )
            )

        # Then send
        for expert_id, dst_rank in weights_to_send:
            local_idx = int(expert_id) % num_local_experts
            p2p_ops.append(
                dist.P2POp(
                    dist.isend,
                    w1_local[local_idx].contiguous(),
                    group_peer=int(dst_rank),
                    group=ep_group,
                )
            )
            p2p_ops.append(
                dist.P2POp(
                    dist.isend,
                    w2_local[local_idx].contiguous(),
                    group_peer=int(dst_rank),
                    group=ep_group,
                )
            )
            p2p_ops.append(
                dist.P2POp(
                    dist.isend,
                    w3_local[local_idx].contiguous(),
                    group_peer=int(dst_rank),
                    group=ep_group,
                )
            )

        handles = []
        if p2p_ops:
            handles = dist.batch_isend_irecv(p2p_ops)
            if not return_handles:
                for h in handles:
                    h.wait()

        # Save context for backward
        ctx.weights_to_send = weights_to_send
        ctx.weights_to_receive = weights_to_receive
        ctx.num_local_experts = num_local_experts
        ctx.ep_group = ep_group
        ctx.w1_shape = tuple(w1_local.shape[1:])
        ctx.w2_shape = tuple(w2_local.shape[1:])
        ctx.w3_shape = tuple(w3_local.shape[1:])
        ctx.dtype = dtype
        ctx.device = device

        # Gradient anchor: ensures all ranks enter backward for this op
        gradient_anchor = w1_local.sum() * 0.0

        if return_handles:
            return (
                recv_w1,
                recv_w2,
                recv_w3,
                foreign_expert_id_mapping,
                handles,
                gradient_anchor,
            )
        return (
            recv_w1,
            recv_w2,
            recv_w3,
            foreign_expert_id_mapping,
            gradient_anchor,
        )

    @staticmethod
    def backward(
        ctx,
        grad_recv_w1,
        grad_recv_w2,
        grad_recv_w3,
        grad_mapping,
        handles_or_anchor=None,
        grad_anchor=None,
    ):
        """
        Backward: P2P gradient transfer (reverse of forward).

        1. Send gradients for foreign weights back to their owners
        2. Receive gradients for weights we sent to others
        3. Accumulate into local weight gradients
        """
        weights_to_send = ctx.weights_to_send
        weights_to_receive = ctx.weights_to_receive
        num_local_experts = ctx.num_local_experts
        ep_group = ctx.ep_group
        dtype = ctx.dtype
        device = ctx.device

        # Initialize gradient accumulators for local weights
        grad_w1_local = torch.zeros(
            num_local_experts, *ctx.w1_shape, dtype=dtype, device=device
        )
        grad_w2_local = torch.zeros(
            num_local_experts, *ctx.w2_shape, dtype=dtype, device=device
        )
        grad_w3_local = torch.zeros(
            num_local_experts, *ctx.w3_shape, dtype=dtype, device=device
        )

        # Prepare recv buffers for gradients of weights we SENT in forward
        recv_grad_buffers = {}
        for expert_id, dst_rank in weights_to_send:
            recv_grad_buffers[(int(expert_id), int(dst_rank))] = (
                torch.empty(*ctx.w1_shape, dtype=dtype, device=device),
                torch.empty(*ctx.w2_shape, dtype=dtype, device=device),
                torch.empty(*ctx.w3_shape, dtype=dtype, device=device),
            )

        p2p_ops = []

        # Recv gradients for weights we sent in forward
        # NOTE: use group_peer= (EP-local rank), not peer= (global rank)
        for (expert_id, src_rank), (buf1, buf2, buf3) in recv_grad_buffers.items():
            p2p_ops.append(
                dist.P2POp(dist.irecv, buf1, group_peer=src_rank, group=ep_group)
            )
            p2p_ops.append(
                dist.P2POp(dist.irecv, buf2, group_peer=src_rank, group=ep_group)
            )
            p2p_ops.append(
                dist.P2POp(dist.irecv, buf3, group_peer=src_rank, group=ep_group)
            )

        # Send gradients for weights we received in forward
        for stacked_idx, (expert_id, src_rank) in enumerate(weights_to_receive):
            p2p_ops.append(
                dist.P2POp(
                    dist.isend,
                    grad_recv_w1[stacked_idx].contiguous(),
                    group_peer=int(src_rank),
                    group=ep_group,
                )
            )
            p2p_ops.append(
                dist.P2POp(
                    dist.isend,
                    grad_recv_w2[stacked_idx].contiguous(),
                    group_peer=int(src_rank),
                    group=ep_group,
                )
            )
            p2p_ops.append(
                dist.P2POp(
                    dist.isend,
                    grad_recv_w3[stacked_idx].contiguous(),
                    group_peer=int(src_rank),
                    group=ep_group,
                )
            )

        if p2p_ops:
            reqs = dist.batch_isend_irecv(p2p_ops)
            for req in reqs:
                req.wait()

        # Accumulate received gradients into local weight gradients
        for (expert_id, _), (g1, g2, g3) in recv_grad_buffers.items():
            local_idx = expert_id % num_local_experts
            grad_w1_local[local_idx] += g1
            grad_w2_local[local_idx] += g2
            grad_w3_local[local_idx] += g3

        # Return grads: 3 for local weights, None for metadata tensors and non-tensor args
        return (
            grad_w1_local,
            grad_w2_local,
            grad_w3_local,
            None,
            None,
            None,
            None,  # metadata tensors
            None,
            None,  # ep_group, return_handles
        )


def transfer_expert_weights_autograd(
    ep_rank: int,
    ep_group,
    plan: LLEPPlan,
    w1_local: torch.Tensor,
    w2_local: torch.Tensor,
    w3_local: torch.Tensor,
    num_local_experts: int,
    num_experts: int,
    return_handles: bool = False,
):
    """
    Differentiable weight transfer wrapper using WeightTransferAutograd.

    Returns stacked tensors (not dicts), a mapping tensor, handles, and a
    gradient anchor. The gradient anchor MUST be added to the final output
    to ensure all ranks enter the backward pass for this operation.
    """
    device = w1_local.device

    weights_to_send = plan.weights_to_send
    weights_to_receive = plan.weights_to_receive

    if weights_to_send:
        wts_tensor = torch.tensor(weights_to_send, dtype=torch.long, device=device)
    else:
        wts_tensor = torch.empty(0, 2, dtype=torch.long, device=device)

    if weights_to_receive:
        wtr_tensor = torch.tensor(weights_to_receive, dtype=torch.long, device=device)
    else:
        wtr_tensor = torch.empty(0, 2, dtype=torch.long, device=device)

    nle_tensor = torch.tensor(num_local_experts, dtype=torch.long, device=device)
    ne_tensor = torch.tensor(num_experts, dtype=torch.long, device=device)

    return WeightTransferAutograd.apply(
        w1_local,
        w2_local,
        w3_local,
        wts_tensor,
        wtr_tensor,
        nle_tensor,
        ne_tensor,
        ep_group,
        return_handles,
    )


# ---------------------------------------------------------------------------
# Grouped GEMM helpers for SwiGLU FFN
# ---------------------------------------------------------------------------
def _pack_expert_weights(
    unique_experts: torch.Tensor,
    w1_local: torch.Tensor,
    w2_local: torch.Tensor,
    w3_local: torch.Tensor,
    foreign_w1_stacked: Optional[torch.Tensor],
    foreign_w2_stacked: Optional[torch.Tensor],
    foreign_w3_stacked: Optional[torch.Tensor],
    foreign_expert_id_mapping: Optional[torch.Tensor],
    ep_rank: int,
    num_local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[bool]]:
    """
    Pack weights for active experts into contiguous (num_active, ...) tensors.

    Uses differentiable indexing to preserve autograd graph.

    Returns:
        (w1_packed, w2_packed, w3_packed, valid_mask)
        valid_mask[i] is True if expert i has valid weights, False if zero-filled.
    """
    native_start = ep_rank * num_local_experts
    native_end = native_start + num_local_experts
    num_active = unique_experts.shape[0]
    device = w1_local.device

    # Classify experts as native vs foreign using tensor ops (no per-element .item())
    is_native = (unique_experts >= native_start) & (unique_experts < native_end)

    # --- Native experts: batch index into w1/w2/w3_local ---
    native_local_indices = (unique_experts - native_start).clamp(
        min=0, max=num_local_experts - 1
    )

    if (
        foreign_expert_id_mapping is not None
        and foreign_w1_stacked is not None
        and foreign_w1_stacked.shape[0] > 0
    ):
        # --- Stacked foreign path (autograd-compatible) ---
        stacked_indices = foreign_expert_id_mapping[unique_experts]  # (num_active,)
        is_valid_foreign = (~is_native) & (stacked_indices >= 0)
        is_invalid = (~is_native) & (stacked_indices < 0)

        # Allocate output
        w1_packed = torch.zeros(
            num_active, *w1_local.shape[1:], device=device, dtype=w1_local.dtype
        )
        w2_packed = torch.zeros(
            num_active, *w2_local.shape[1:], device=device, dtype=w2_local.dtype
        )
        w3_packed = torch.zeros(
            num_active, *w3_local.shape[1:], device=device, dtype=w3_local.dtype
        )

        # Fill native experts
        if is_native.any():
            native_pos = is_native.nonzero(as_tuple=True)[0]
            w1_packed[native_pos] = w1_local[native_local_indices[native_pos]]
            w2_packed[native_pos] = w2_local[native_local_indices[native_pos]]
            w3_packed[native_pos] = w3_local[native_local_indices[native_pos]]

        # Fill valid foreign experts
        if is_valid_foreign.any():
            foreign_pos = is_valid_foreign.nonzero(as_tuple=True)[0]
            safe_stacked_idx = stacked_indices[foreign_pos].clamp(min=0)
            w1_packed[foreign_pos] = foreign_w1_stacked[safe_stacked_idx]
            w2_packed[foreign_pos] = foreign_w2_stacked[safe_stacked_idx]
            w3_packed[foreign_pos] = foreign_w3_stacked[safe_stacked_idx]

        valid_mask = (~is_invalid).tolist()
        return w1_packed, w2_packed, w3_packed, valid_mask

    else:
        # --- No foreign weights: only native experts are valid ---
        w1_packed = torch.zeros(
            num_active, *w1_local.shape[1:], device=device, dtype=w1_local.dtype
        )
        w2_packed = torch.zeros(
            num_active, *w2_local.shape[1:], device=device, dtype=w2_local.dtype
        )
        w3_packed = torch.zeros(
            num_active, *w3_local.shape[1:], device=device, dtype=w3_local.dtype
        )

        if is_native.any():
            native_positions = is_native.nonzero(as_tuple=True)[0]
            w1_packed[native_positions] = w1_local[
                native_local_indices[native_positions]
            ]
            w2_packed[native_positions] = w2_local[
                native_local_indices[native_positions]
            ]
            w3_packed[native_positions] = w3_local[
                native_local_indices[native_positions]
            ]

        valid_mask = is_native.tolist()
        return w1_packed, w2_packed, w3_packed, valid_mask


def _pad_for_grouped_mm(
    x_sorted: torch.Tensor,
    counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad each expert's token group to a multiple of _TOKEN_ALIGN for grouped_mm.

    Also clamps empty experts to _TOKEN_ALIGN minimum tokens (workaround for
    grouped_mm gradient bug with zero-size groups).

    Args:
        x_sorted: (num_tokens, dim) tokens sorted by expert
        counts: (num_experts,) token counts per expert

    Returns:
        (x_padded, counts_padded) where counts_padded are aligned.
    """
    # Compute padded counts: round up to _TOKEN_ALIGN, min _TOKEN_ALIGN
    counts_padded = ((counts + _TOKEN_ALIGN - 1) // _TOKEN_ALIGN * _TOKEN_ALIGN).to(
        torch.int64
    )
    counts_padded = torch.clamp_min(counts_padded, _TOKEN_ALIGN)

    n = counts.shape[0]

    # Try Triton kernel for large expert counts on CUDA (4.6x faster at 256 experts)
    if n > _PAD_VECTORIZE_THRESHOLD and x_sorted.is_cuda:
        global _triton_pad_fn
        if _triton_pad_fn is None:
            try:
                from torchtitan.distributed.llep.kernels import (
                    triton_pad_for_grouped_mm,
                )

                _triton_pad_fn = triton_pad_for_grouped_mm
            except (ImportError, RuntimeError):
                _triton_pad_fn = False

        if _triton_pad_fn:
            x_padded = _triton_pad_fn(x_sorted, counts, counts_padded)
            return x_padded, counts_padded

    total_padded = counts_padded.sum().item()
    dim = x_sorted.shape[1]
    device = x_sorted.device
    dtype = x_sorted.dtype

    x_padded = torch.zeros(total_padded, dim, device=device, dtype=dtype)

    if n <= _PAD_VECTORIZE_THRESHOLD:
        # Small expert count: for-loop is faster (less GPU overhead)
        src_offset = 0
        dst_offset = 0
        counts_list = counts.tolist()
        counts_padded_list = counts_padded.tolist()
        for i in range(n):
            c = int(counts_list[i])
            cp = int(counts_padded_list[i])
            if c > 0:
                x_padded[dst_offset : dst_offset + c] = x_sorted[
                    src_offset : src_offset + c
                ]
            src_offset += c
            dst_offset += cp
    else:
        # Large expert count: vectorized GPU scatter (fallback when Triton unavailable)
        src_offsets = torch.zeros(n + 1, dtype=torch.int64, device=device)
        src_offsets[1:] = counts.cumsum(0)
        dst_offsets = torch.zeros(n + 1, dtype=torch.int64, device=device)
        dst_offsets[1:] = counts_padded.cumsum(0)

        nonempty = counts > 0
        if nonempty.any():
            src_starts = src_offsets[:-1][nonempty]
            dst_starts = dst_offsets[:-1][nonempty]
            lens = counts[nonempty]

            max_len = lens.max().item()
            offsets_range = torch.arange(max_len, device=device).unsqueeze(0)
            mask = offsets_range < lens.unsqueeze(1)
            src_indices = (src_starts.unsqueeze(1) + offsets_range).masked_select(mask)
            dst_indices = (dst_starts.unsqueeze(1) + offsets_range).masked_select(mask)
            x_padded[dst_indices] = x_sorted[src_indices]

    return x_padded, counts_padded


def _unpad_output(
    out_padded: torch.Tensor,
    counts: torch.Tensor,
    counts_padded: torch.Tensor,
) -> torch.Tensor:
    """
    Remove padding from grouped_mm output to recover original token counts.
    """
    n = counts.shape[0]

    # Try Triton kernel for large expert counts on CUDA (6.5x faster at 256 experts)
    if n > _PAD_VECTORIZE_THRESHOLD and out_padded.is_cuda:
        global _triton_unpad_fn
        if _triton_unpad_fn is None:
            try:
                from torchtitan.distributed.llep.kernels import triton_unpad_output

                _triton_unpad_fn = triton_unpad_output
            except (ImportError, RuntimeError):
                _triton_unpad_fn = False

        if _triton_unpad_fn:
            return _triton_unpad_fn(out_padded, counts, counts_padded)

    total_tokens = counts.sum().item()
    dim = out_padded.shape[1]
    device = out_padded.device
    dtype = out_padded.dtype

    out = torch.empty(total_tokens, dim, device=device, dtype=dtype)

    if n <= _PAD_VECTORIZE_THRESHOLD:
        # Small expert count: for-loop is faster
        src_offset = 0
        dst_offset = 0
        counts_list = counts.tolist()
        counts_padded_list = counts_padded.tolist()
        for i in range(n):
            c = int(counts_list[i])
            cp = int(counts_padded_list[i])
            if c > 0:
                out[dst_offset : dst_offset + c] = out_padded[
                    src_offset : src_offset + c
                ]
            src_offset += cp
            dst_offset += c
    else:
        # Large expert count: vectorized GPU gather (fallback when Triton unavailable)
        src_offsets = torch.zeros(n + 1, dtype=torch.int64, device=device)
        src_offsets[1:] = counts_padded.cumsum(0)
        dst_offsets = torch.zeros(n + 1, dtype=torch.int64, device=device)
        dst_offsets[1:] = counts.cumsum(0)

        nonempty = counts > 0
        if nonempty.any():
            src_starts = src_offsets[:-1][nonempty]
            dst_starts = dst_offsets[:-1][nonempty]
            lens = counts[nonempty]

            max_len = lens.max().item()
            offsets_range = torch.arange(max_len, device=device).unsqueeze(0)
            mask = offsets_range < lens.unsqueeze(1)
            src_indices = (src_starts.unsqueeze(1) + offsets_range).masked_select(mask)
            dst_indices = (dst_starts.unsqueeze(1) + offsets_range).masked_select(mask)
            out[dst_indices] = out_padded[src_indices]

    return out


# ---------------------------------------------------------------------------
# Composable LLEP functions (used as EP hooks)
# ---------------------------------------------------------------------------
def llep_dispatch_tokens(
    routed_input: torch.Tensor,  # (total_routed_tokens, dim) sorted by expert
    num_tokens_per_expert: torch.Tensor,  # (num_experts,) counts per expert
    ep_group,
    max_tokens_factor: float = 1.1,
    min_tokens_per_gemm: int = 1024,
    adaptive_threshold: float = 0.0,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, LLEPState]:
    """EP pre-hook: LLEP planning + AllToAll dispatch. Does NOT access weights.

    Receives tokens already sorted by expert (from MoE.forward's reorderer)
    with scores already applied (if score_before_experts). Routes them to
    GPUs via LPT-based AllToAll.

    Returns:
        (dispatched_tokens, padded_counts, llep_state)
        - dispatched_tokens: tokens sorted by expert and padded for grouped_mm
        - padded_counts: aligned token counts per active expert
        - llep_state: state needed by compute and combine steps
    """
    torch.cuda.nvtx.range_push("llep_dispatch_tokens")

    # Override from env (cached at module level, not re-read per call)
    if _ENV_MAX_TOKENS_FACTOR is not None:
        max_tokens_factor = float(_ENV_MAX_TOKENS_FACTOR)
    if _ENV_MIN_TOKENS_PER_GEMM is not None:
        min_tokens_per_gemm = int(_ENV_MIN_TOKENS_PER_GEMM)
    if _ENV_ADAPTIVE_THRESHOLD is not None:
        adaptive_threshold = float(_ENV_ADAPTIVE_THRESHOLD)

    ep_rank = dist.get_rank(group=ep_group)
    ep_size = dist.get_world_size(group=ep_group)
    num_experts = num_tokens_per_expert.shape[0]
    num_local_experts = num_experts // ep_size
    device = routed_input.device
    dtype = routed_input.dtype
    total_tokens = routed_input.shape[0]
    dim = routed_input.shape[1]

    # ------------------------------------------------------------------
    # Step 1: Gather per-rank expert counts via all_gather
    # ------------------------------------------------------------------
    torch.cuda.nvtx.range_push("llep:1_allgather_counts")
    local_expert_counts = num_tokens_per_expert.to(torch.int64)

    all_expert_counts = [torch.zeros_like(local_expert_counts) for _ in range(ep_size)]
    dist.all_gather(all_expert_counts, local_expert_counts, group=ep_group)

    global_expert_counts = torch.stack(all_expert_counts).sum(dim=0)

    # Truncate to actual expert count
    effective_num_experts = num_local_experts * ep_size
    if global_expert_counts.size(0) > effective_num_experts:
        global_expert_counts = global_expert_counts[:effective_num_experts]
        all_expert_counts = [ec[:effective_num_experts] for ec in all_expert_counts]
        local_expert_counts = local_expert_counts[:effective_num_experts]

    torch.cuda.nvtx.range_pop()  # llep:1_allgather_counts

    # ------------------------------------------------------------------
    # Step 2: Adaptive threshold check
    # ------------------------------------------------------------------
    torch.cuda.nvtx.range_push("llep:2a_imbalance_check")
    imbalance = compute_gpu_imbalance_ratio(
        global_expert_counts, ep_size, num_local_experts
    )
    use_lpt = True
    if adaptive_threshold > 0:
        use_lpt = imbalance >= adaptive_threshold
    torch.cuda.nvtx.range_pop()  # llep:2a_imbalance_check

    _log_verbose = verbose or LLEP_DEBUG
    if _log_verbose:
        global _llep_step_counter
        _llep_step_counter += 1
        _step = _llep_step_counter
        # Compute native GPU loads (before LLEP redistribution)
        _native_loads = (
            global_expert_counts[:effective_num_experts]
            .view(ep_size, num_local_experts)
            .sum(dim=1)
            .cpu()
            .tolist()
        )
        logger.info(
            f"[LLEP rank={ep_rank} step={_step}] BEFORE: "
            f"total_tokens={total_tokens} imbalance={imbalance:.2f}\n"
            f"  native_gpu_loads={_native_loads}\n"
            f"  expert_counts={global_expert_counts.cpu().tolist()}"
        )

    # ------------------------------------------------------------------
    # Step 3: Compute LPT plan
    # ------------------------------------------------------------------
    torch.cuda.nvtx.range_push("llep:2b_lpt_plan")
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
    torch.cuda.nvtx.range_pop()  # llep:2b_lpt_plan

    if _log_verbose:
        _after_loads = plan.gpu_loads.cpu()
        _after_loads_f = _after_loads.float()
        _after_mean = _after_loads_f.mean().item()
        _after_imbalance = (
            (_after_loads_f.max().item() / _after_mean) if _after_mean > 0 else 1.0
        )
        _wt_lines = []
        for wt in plan.weight_transfers:
            _wt_lines.append(
                f"    expert {wt.expert_id}: GPU {wt.src_rank}->{wt.dst_rank} "
                f"(tokens {wt.token_start}-{wt.token_end})"
            )
        logger.info(
            f"[LLEP rank={ep_rank} step={_step}] AFTER LPT: use_lpt={use_lpt} "
            f"imbalance={imbalance:.2f}->{_after_imbalance:.2f}\n"
            f"  llep_gpu_loads={_after_loads.tolist()}\n"
            f"  weight_transfers ({len(plan.weight_transfers)}):\n"
            + ("\n".join(_wt_lines) if _wt_lines else "    (none)")
        )

    # ------------------------------------------------------------------
    # Step 4: Barrier before P2P to prevent cross-layer deadlock
    # ------------------------------------------------------------------
    torch.cuda.nvtx.range_push("llep:3_barrier")
    dist.barrier(group=ep_group)
    torch.cuda.nvtx.range_pop()  # llep:3_barrier

    # ------------------------------------------------------------------
    # Step 5: Compute send_matrix and split sizes from LPT plan
    # ------------------------------------------------------------------
    torch.cuda.nvtx.range_push("llep:4_compute_splits")

    # Stack per-rank counts into numpy
    all_counts_np = torch.stack(all_expert_counts).cpu().numpy()
    cum_counts_np = None

    if plan.lpt_plan:
        cum_counts_np = np.zeros((ep_size + 1, effective_num_experts), dtype=np.int64)
        cum_counts_np[1:] = np.cumsum(all_counts_np, axis=0)

    # Compute send_matrix
    global _send_matrix_fn
    if _send_matrix_fn is None:
        try:
            from torchtitan.distributed.llep.kernels import (
                compute_send_matrix_vectorized,
            )

            _send_matrix_fn = compute_send_matrix_vectorized
        except (ImportError, RuntimeError):
            _send_matrix_fn = False

    if _send_matrix_fn and plan.lpt_plan:
        send_matrix_np = _send_matrix_fn(
            all_counts_np,
            cum_counts_np,
            plan.lpt_plan,
            ep_size,
            num_local_experts,
            effective_num_experts,
        )
    elif plan.lpt_plan:
        # Fallback: nested Python loops
        send_matrix_np = np.zeros((ep_size, ep_size), dtype=np.int64)
        lpt_expert_set = set(plan.lpt_plan.keys())
        for eid in range(effective_num_experts):
            if eid not in lpt_expert_set:
                owner = eid // num_local_experts
                send_matrix_np[:, owner] += all_counts_np[:, eid]
        for expert_id, assignments in plan.lpt_plan.items():
            for src_rank in range(ep_size):
                src_start = int(cum_counts_np[src_rank, expert_id])
                src_end = int(cum_counts_np[src_rank + 1, expert_id])
                if src_start == src_end:
                    continue
                for dst_gpu, dst_start, dst_end in assignments:
                    overlap_start = max(src_start, dst_start)
                    overlap_end = min(src_end, dst_end)
                    if overlap_start < overlap_end:
                        send_matrix_np[src_rank, dst_gpu] += overlap_end - overlap_start
    else:
        send_matrix_np = all_counts_np.reshape(ep_size, ep_size, num_local_experts).sum(
            axis=2
        )

    input_split_sizes = send_matrix_np[ep_rank].tolist()
    output_split_sizes = send_matrix_np[:, ep_rank].tolist()

    if _log_verbose:
        _sm_rows = [f"    {send_matrix_np[r].tolist()}" for r in range(ep_size)]
        logger.info(
            f"[LLEP rank={ep_rank} step={_step}] SEND_MATRIX (row=src, col=dst):\n"
            + "\n".join(_sm_rows)
            + f"\n  input_splits={input_split_sizes}"
            + f"\n  output_splits={output_split_sizes}"
        )

    torch.cuda.nvtx.range_pop()  # llep:4_compute_splits

    # ------------------------------------------------------------------
    # Step 6: Assign tokens to target GPUs and sort for A2A
    # ------------------------------------------------------------------
    torch.cuda.nvtx.range_push("llep:5_assign_and_sort")

    # Build expert_ids for each token (reconstructed from counts since
    # routed_input is sorted by expert)
    local_counts_list = local_expert_counts.tolist()
    expert_ids = torch.repeat_interleave(
        torch.arange(effective_num_experts, device=device),
        local_expert_counts[:effective_num_experts],
    )

    # Determine target GPU for each token based on LPT plan
    if plan.lpt_plan:
        # Compute global offsets for this rank
        global_offsets_np = cum_counts_np[ep_rank]

        # Try Triton kernel
        _use_triton = expert_ids.is_cuda
        if _use_triton:
            try:
                from torchtitan.distributed.llep.kernels import triton_assign_tokens

                global_offsets_gpu = torch.from_numpy(global_offsets_np).to(
                    device, non_blocking=True
                )
                target_gpus = triton_assign_tokens(
                    expert_ids,
                    local_expert_counts,
                    global_offsets_gpu,
                    plan.lpt_plan,
                    num_local_experts,
                    effective_num_experts,
                )
            except (ImportError, RuntimeError):
                _use_triton = False

        if not _use_triton:
            # Fallback: vectorized PyTorch path
            target_gpus = (expert_ids // num_local_experts).to(torch.int64)
            _sorted_eids, sort_perm = expert_ids.sort(stable=True)
            local_offsets = torch.zeros(
                effective_num_experts + 1, dtype=torch.int64, device=device
            )
            local_offsets[1:] = local_expert_counts.cumsum(0)
            local_offsets_np = local_offsets.cpu().numpy()

            single_gpu_batches: dict[int, list[torch.Tensor]] = {}
            multi_gpu_work: list[
                tuple[torch.Tensor, torch.Tensor, list[tuple[int, int, int]]]
            ] = []

            for eid, assignments in plan.lpt_plan.items():
                local_start = int(local_offsets_np[eid])
                local_end = int(local_offsets_np[eid + 1])
                local_count = local_end - local_start
                if local_count == 0:
                    continue
                original_positions = sort_perm[local_start:local_end]
                if len(assignments) == 1:
                    gpu_id = assignments[0][0]
                    single_gpu_batches.setdefault(gpu_id, []).append(original_positions)
                else:
                    my_offset = int(global_offsets_np[eid])
                    global_pos = my_offset + torch.arange(local_count, device=device)
                    multi_gpu_work.append((original_positions, global_pos, assignments))

            for gpu_id, pos_list in single_gpu_batches.items():
                all_pos = torch.cat(pos_list) if len(pos_list) > 1 else pos_list[0]
                target_gpus[all_pos] = gpu_id
            for original_positions, global_pos, assignments in multi_gpu_work:
                for gpu_id, start, end in assignments:
                    mask = (global_pos >= start) & (global_pos < end)
                    if mask.any():
                        target_gpus[original_positions[mask]] = gpu_id
    else:
        target_gpus = (expert_ids // num_local_experts).to(torch.int64)

    # Sort by target GPU for AllToAll
    sorted_indices = torch.argsort(target_gpus, stable=True)
    undo_indices = torch.argsort(sorted_indices)
    sorted_hidden = routed_input[sorted_indices]
    sorted_experts = expert_ids[sorted_indices]

    torch.cuda.nvtx.range_pop()  # llep:5_assign_and_sort

    # ------------------------------------------------------------------
    # Step 7: AllToAll dispatch
    # ------------------------------------------------------------------
    torch.cuda.nvtx.range_push("llep:6_alltoall_dispatch")
    total_send = sum(input_split_sizes)
    total_recv = sum(output_split_sizes)

    # Determine if we can merge hidden + expert_ids into one A2A call
    can_merge = LLEP_MERGE_A2A
    if dtype == torch.bfloat16 and effective_num_experts > 256:
        can_merge = False
    elif dtype == torch.float16 and effective_num_experts > 2048:
        can_merge = False

    # NOTE: Always call AllToAll even if total_send=0 and total_recv=0 on this
    # rank.  AllToAll is a collective — all ranks must participate or NCCL hangs.
    if can_merge:
        # Merge hidden (dim cols) + expert_ids (1 col)
        merged_send = torch.cat(
            [sorted_hidden, sorted_experts.unsqueeze(1).to(dtype)],
            dim=1,
        )  # (total_send, dim + 1)

        merged_recv = a2a_autograd(
            merged_send, output_split_sizes, input_split_sizes, ep_group
        )

        recv_hidden = merged_recv[:, :dim]
        recv_experts = merged_recv[:, dim].to(torch.int64)
    else:
        # Separate A2A calls
        recv_hidden = a2a_autograd(
            sorted_hidden, output_split_sizes, input_split_sizes, ep_group
        )
        # Expert indices: always non-differentiable
        recv_experts = torch.empty(total_recv, device=device, dtype=torch.int64)
        dist.all_to_all_single(
            recv_experts,
            sorted_experts.to(torch.int64),
            output_split_sizes,
            input_split_sizes,
            group=ep_group,
        )

    torch.cuda.nvtx.range_pop()  # llep:6_alltoall_dispatch

    # ------------------------------------------------------------------
    # Step 8: Sort received tokens by expert + pad for grouped_mm
    # ------------------------------------------------------------------
    torch.cuda.nvtx.range_push("llep:7_sort_and_pad")

    recv_sort_perm = None
    if recv_hidden.numel() > 0:
        # Sort by expert ID for contiguous access in grouped_mm
        recv_sort_perm = recv_experts.argsort(stable=True)
        recv_hidden = recv_hidden[recv_sort_perm]
        recv_experts = recv_experts[recv_sort_perm]

        # Count tokens per expert
        unique_experts, counts = torch.unique_consecutive(
            recv_experts, return_counts=True
        )
        num_active = unique_experts.shape[0]

        # Pad for grouped_mm alignment
        recv_hidden_padded, counts_padded = _pad_for_grouped_mm(recv_hidden, counts)
    else:
        recv_hidden_padded = recv_hidden
        counts_padded = torch.zeros(0, dtype=torch.int64, device=device)
        unique_experts = torch.zeros(0, dtype=torch.int64, device=device)
        counts = torch.zeros(0, dtype=torch.int64, device=device)

    if _log_verbose:
        logger.info(
            f"[LLEP rank={ep_rank} step={_step}] RECEIVED: "
            f"total_recv={total_recv} "
            f"experts={unique_experts.cpu().tolist()} "
            f"counts={counts.cpu().tolist()}"
        )

    torch.cuda.nvtx.range_pop()  # llep:7_sort_and_pad

    # Build LLEPState
    llep_state = LLEPState(
        input_split_sizes=input_split_sizes,
        output_split_sizes=output_split_sizes,
        undo_indices=undo_indices,
        recv_sort_perm=recv_sort_perm,
        recv_experts=recv_experts,
        plan=plan,
        ep_group=ep_group,
        ep_rank=ep_rank,
        ep_size=ep_size,
        num_local_experts=num_local_experts,
        num_experts=effective_num_experts,
        total_send=total_send,
        total_recv=total_recv,
        num_tokens=total_tokens,
        top_k=1,  # tokens are already expanded by top_k before reaching dispatch
        dim=dim,
        dtype=dtype,
    )

    torch.cuda.nvtx.range_pop()  # llep_dispatch_tokens
    return recv_hidden_padded, counts_padded, llep_state


def llep_prepare_weights(
    w1_local: torch.Tensor,  # (num_local_experts, hidden_dim, dim)
    w2_local: torch.Tensor,  # (num_local_experts, dim, hidden_dim)
    w3_local: torch.Tensor,  # (num_local_experts, hidden_dim, dim)
    state: LLEPState,
) -> tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    list,
    Optional[torch.Tensor],
]:
    """P2P weight transfer + pack weights for active experts.

    Called from GroupedExperts.forward() directly, enabling a unified
    compute path for both standard EP and LLEP.

    Args:
        w1_local, w2_local, w3_local: Unsharded local expert weights.
        state: LLEPState from dispatch (needs recv_experts, plan, etc.).

    Returns:
        (w1_packed, w2_packed, w3_packed, valid_mask, gradient_anchor)
        - Packed weights are None if no tokens were received.
        - gradient_anchor ensures all ranks enter backward for P2P weight transfer.
    """
    torch.cuda.nvtx.range_push("llep_prepare_weights")

    # ------------------------------------------------------------------
    # Step 1: P2P weight transfer
    # ------------------------------------------------------------------
    torch.cuda.nvtx.range_push("llep:8_p2p_weight_transfer")

    (
        foreign_w1_stacked,
        foreign_w2_stacked,
        foreign_w3_stacked,
        foreign_expert_id_mapping,
        gradient_anchor,
    ) = transfer_expert_weights_autograd(
        state.ep_rank,
        state.ep_group,
        state.plan,
        w1_local,
        w2_local,
        w3_local,
        state.num_local_experts,
        state.num_experts,
        return_handles=False,
    )

    torch.cuda.nvtx.range_pop()  # llep:8_p2p_weight_transfer

    # ------------------------------------------------------------------
    # Step 2: Pack weights for active experts
    # ------------------------------------------------------------------
    torch.cuda.nvtx.range_push("llep:9_pack_weights")

    if state.recv_experts.numel() > 0:
        unique_experts = torch.unique_consecutive(state.recv_experts)
    else:
        unique_experts = torch.zeros(0, dtype=torch.int64, device=w1_local.device)

    if unique_experts.numel() > 0:
        w1_packed, w2_packed, w3_packed, valid_mask = _pack_expert_weights(
            unique_experts,
            w1_local,
            w2_local,
            w3_local,
            foreign_w1_stacked,
            foreign_w2_stacked,
            foreign_w3_stacked,
            foreign_expert_id_mapping,
            state.ep_rank,
            state.num_local_experts,
        )
    else:
        w1_packed = w2_packed = w3_packed = None
        valid_mask = []

        # Touch weights for autograd graph when no tokens received
        weight_touch = (w1_local.sum() + w2_local.sum() + w3_local.sum()) * 0.0
        if foreign_w1_stacked is not None and foreign_w1_stacked.numel() > 0:
            weight_touch = (
                weight_touch
                + (
                    foreign_w1_stacked.sum()
                    + foreign_w2_stacked.sum()
                    + foreign_w3_stacked.sum()
                )
                * 0.0
            )
        if gradient_anchor is not None:
            gradient_anchor = gradient_anchor + weight_touch
        else:
            gradient_anchor = weight_touch

    torch.cuda.nvtx.range_pop()  # llep:9_pack_weights
    torch.cuda.nvtx.range_pop()  # llep_prepare_weights

    return w1_packed, w2_packed, w3_packed, valid_mask, gradient_anchor


def _get_original_counts(state: LLEPState) -> torch.Tensor:
    """Get unpadded token counts from recv_experts."""
    if state.recv_experts.numel() == 0:
        return torch.zeros(0, dtype=torch.int64, device=state.recv_experts.device)
    _, counts = torch.unique_consecutive(state.recv_experts, return_counts=True)
    return counts


def llep_combine_output(
    expert_output: torch.Tensor,  # output from GroupedExperts.forward
    state: LLEPState,
) -> torch.Tensor:
    """EP post-hook: zero-invalid + unpad + AllToAll combine to route outputs back.

    When called after the unified forward path, expert_output is still padded
    (aligned to _TOKEN_ALIGN per expert group). This function:
    1. Zeros out invalid expert outputs (experts without weights)
    2. Unpads to recover original token counts
    3. Undoes expert sort (back to A2A-received order)
    4. AllToAll combine to route outputs back
    5. Unsorts to original token order

    Args:
        expert_output: Output from expert computation (may be padded).
        state: LLEPState from dispatch (with padded_counts/valid_mask set by forward).

    Returns:
        Combined output in original token order.
    """
    torch.cuda.nvtx.range_push("llep_combine_output")

    device = expert_output.device
    dim = state.dim
    dtype = state.dtype
    ep_group = state.ep_group
    # ------------------------------------------------------------------
    # Step 0: Zero invalid experts + unpad (moved from compute step)
    # ------------------------------------------------------------------
    if state.padded_counts is not None and state.valid_mask is not None:
        # Zero out invalid experts (experts without weights)
        if state.valid_mask and not all(state.valid_mask):
            offset = 0
            counts_list = state.padded_counts.tolist()
            for i, valid in enumerate(state.valid_mask):
                c = int(counts_list[i])
                if not valid and c > 0:
                    expert_output[offset : offset + c] = 0
                offset += c

        # Unpad to original counts
        if expert_output.numel() > 0:
            original_counts = _get_original_counts(state)
            if original_counts.numel() > 0:
                expert_output = _unpad_output(
                    expert_output, original_counts, state.padded_counts
                )

    if expert_output.dtype != dtype:
        expert_output = expert_output.to(dtype)

    # ------------------------------------------------------------------
    # Step 1: Undo expert sort (back to A2A-received order for reverse A2A)
    # ------------------------------------------------------------------
    if state.recv_sort_perm is not None:
        # Compute inverse permutation: unsorted[sort_perm[i]] = sorted[i]
        inverse_perm = torch.empty_like(state.recv_sort_perm)
        inverse_perm[state.recv_sort_perm] = torch.arange(
            state.recv_sort_perm.shape[0], device=device
        )
        expert_output = expert_output[inverse_perm]

    # ------------------------------------------------------------------
    # Step 2: AllToAll combine - route outputs back
    # ------------------------------------------------------------------
    torch.cuda.nvtx.range_push("llep:10_alltoall_combine")
    # NOTE: Always call AllToAll — it's a collective, all ranks must participate.
    send_output = a2a_autograd(
        expert_output.contiguous(),
        state.input_split_sizes,
        state.output_split_sizes,
        ep_group,
    )
    torch.cuda.nvtx.range_pop()  # llep:10_alltoall_combine

    # ------------------------------------------------------------------
    # Step 3: Unsort back to original token order (undo target-GPU sort from dispatch)
    # ------------------------------------------------------------------
    torch.cuda.nvtx.range_push("llep:11_unsort")
    output = send_output[state.undo_indices]
    torch.cuda.nvtx.range_pop()  # llep:11_unsort

    torch.cuda.nvtx.range_pop()  # llep_combine_output
    return output
