# Copyright (c) Nous Research. All rights reserved.
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
import torch.nn.functional as F

from torchtitan.tools.logging import logger

# Debug logging (set LLEP_DEBUG=1 to enable per-step verbose logging)
LLEP_DEBUG = os.environ.get("LLEP_DEBUG", "0") == "1"

# Merge hidden+scores+expert_ids into single A2A call (set 0 to disable)
LLEP_MERGE_A2A = os.environ.get("LLEP_MERGE_A2A", "1") == "1"

# Enable autograd for weight transfer + A2A (supports backward pass)
# Set LLEP_W_TRANSFER_AUTOGRAD=1 to enable differentiable weight transfer + A2A
LLEP_W_TRANSFER_AUTOGRAD = os.environ.get("LLEP_W_TRANSFER_AUTOGRAD", "1") == "1"

# Use grouped GEMM (torch._grouped_mm) to replace the Python for-loop in SwiGLU FFN.
# Set LLEP_USE_GROUPED_MM=0 to fall back to the original per-expert for-loop.
LLEP_USE_GROUPED_MM = os.environ.get("LLEP_USE_GROUPED_MM", "1") == "1"

# Token alignment for grouped_mm (must be 8 for bf16)
_TOKEN_ALIGN = 8

# Threshold for vectorized pad/unpad: use for-loop below this, vectorized above.
# Benchmarked on B200: forloop faster for <32 experts, vectorized wins at 64+.
_PAD_VECTORIZE_THRESHOLD = 32

# Lazy-cached fused SiLU-gate function (avoids try/import on every forward call)
_fused_silu_gate_fn = None

# Lazy-cached Triton pad/unpad functions
_triton_pad_fn = None
_triton_unpad_fn = None

# Lazy-cached vectorized send_matrix function
_send_matrix_fn = None

# Cached env overrides for llep_moe_forward (read once at import, not per call)
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
            from torchtitan.distributed.llep_kernels import triton_imbalance_ratio

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
    return_handles: bool = False,
) -> tuple[
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    Optional[list],
]:
    """
    Transfer expert weights via P2P for LLEP weight spilling.

    Sends native expert weights to helper GPUs and receives foreign
    expert weights from their native GPUs.

    Args:
        return_handles: If True, return P2P handles without waiting.
            Caller must wait on them before using the received weights.

    Returns:
        Tuple of (foreign_w1, foreign_w2, foreign_w3, handles).
        handles is None when return_handles=False (transfers already completed).
    """
    foreign_w1: dict[int, torch.Tensor] = {}
    foreign_w2: dict[int, torch.Tensor] = {}
    foreign_w3: dict[int, torch.Tensor] = {}

    # Build P2P op list for batch_isend_irecv
    p2p_ops = []

    # Receive first (post recvs before sends for MPI/NCCL efficiency)
    for expert_id, src_rank in plan.weights_to_receive:
        recv_w1 = torch.empty_like(w1_local[0])
        recv_w2 = torch.empty_like(w2_local[0])
        recv_w3 = torch.empty_like(w3_local[0])
        p2p_ops.append(
            dist.P2POp(dist.irecv, recv_w1, group_peer=src_rank, group=ep_group)
        )
        p2p_ops.append(
            dist.P2POp(dist.irecv, recv_w2, group_peer=src_rank, group=ep_group)
        )
        p2p_ops.append(
            dist.P2POp(dist.irecv, recv_w3, group_peer=src_rank, group=ep_group)
        )
        foreign_w1[expert_id] = recv_w1
        foreign_w2[expert_id] = recv_w2
        foreign_w3[expert_id] = recv_w3

    # Then send (use % for safety — matches reference implementation)
    for expert_id, dst_rank in plan.weights_to_send:
        local_idx = expert_id % num_local_experts
        p2p_ops.append(
            dist.P2POp(
                dist.isend,
                w1_local[local_idx].contiguous(),
                group_peer=dst_rank,
                group=ep_group,
            )
        )
        p2p_ops.append(
            dist.P2POp(
                dist.isend,
                w2_local[local_idx].contiguous(),
                group_peer=dst_rank,
                group=ep_group,
            )
        )
        p2p_ops.append(
            dist.P2POp(
                dist.isend,
                w3_local[local_idx].contiguous(),
                group_peer=dst_rank,
                group=ep_group,
            )
        )

    if p2p_ops:
        handles = dist.batch_isend_irecv(p2p_ops)
        if return_handles:
            return foreign_w1, foreign_w2, foreign_w3, handles
        for h in handles:
            h.wait()

    return foreign_w1, foreign_w2, foreign_w3, None


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
# Token Routing based on LPT Plan (vectorized)
# ---------------------------------------------------------------------------
def assign_tokens_to_gpus(
    top_scores: torch.Tensor,  # (num_tokens, top_k)
    selected_experts: torch.Tensor,  # (num_tokens, top_k)
    lpt_plan: dict[int, list[tuple[int, int, int]]],
    ep_size: int,
    ep_rank: int,
    num_local_experts: int,
    all_expert_counts: list[torch.Tensor],  # per-rank expert counts from all_gather
    local_expert_counts: torch.Tensor,  # (num_experts,) this rank's local counts
) -> tuple[
    torch.Tensor,  # sorted scores
    torch.Tensor,  # sorted expert ids
    list[int],  # input split sizes
    list[int],  # output split sizes
    torch.Tensor,  # sorted indices
    torch.Tensor,  # undo indices for reverse routing
]:
    """
    Assign token-expert pairs to GPUs based on the LPT plan.

    Vectorized: O(num_experts) Python iterations, not O(num_tokens * top_k).
    Computes split sizes from the LPT plan's send_matrix, avoiding an extra
    all_gather of split sizes.
    """
    device = top_scores.device
    num_tokens, top_k = selected_experts.shape
    num_experts = ep_size * num_local_experts

    flat_experts = selected_experts.view(-1)  # (num_tokens * top_k,)
    flat_scores = top_scores.view(-1)  # (num_tokens * top_k,)

    # Stack per-rank counts into numpy (single D2H sync instead of N separate ones)
    all_counts_np = torch.stack(all_expert_counts).cpu().numpy()

    if lpt_plan:
        # Cumulative counts: cum[r, e] = total tokens for expert e from ranks 0..r-1
        cum_counts_np = np.zeros((ep_size + 1, num_experts), dtype=np.int64)
        cum_counts_np[1:] = np.cumsum(all_counts_np, axis=0)
        global_offsets_np = cum_counts_np[ep_rank]

        # Try Triton kernel: fused sort + plan lookup + scatter (1 kernel vs ~15 PyTorch ops)
        _use_triton = flat_experts.is_cuda
        if _use_triton:
            try:
                from torchtitan.distributed.llep_kernels import triton_assign_tokens

                global_offsets_gpu = torch.from_numpy(global_offsets_np).to(
                    device, non_blocking=True
                )
                target_gpus = triton_assign_tokens(
                    flat_experts,
                    local_expert_counts,
                    global_offsets_gpu,
                    lpt_plan,
                    num_local_experts,
                    num_experts,
                )
            except (ImportError, RuntimeError):
                _use_triton = False

        if not _use_triton:
            # Fallback: vectorized PyTorch path
            target_gpus = (flat_experts // num_local_experts).to(torch.int64)
            _sorted_expert_ids, sort_perm = flat_experts.sort(stable=True)
            local_offsets = torch.zeros(
                num_experts + 1, dtype=torch.int64, device=device
            )
            local_offsets[1:] = local_expert_counts.cumsum(0)
            local_offsets_np = local_offsets.cpu().numpy()

            single_gpu_batches: dict[int, list[torch.Tensor]] = {}
            multi_gpu_work: list[
                tuple[torch.Tensor, torch.Tensor, list[tuple[int, int, int]]]
            ] = []

            for expert_id, assignments in lpt_plan.items():
                local_start = int(local_offsets_np[expert_id])
                local_end = int(local_offsets_np[expert_id + 1])
                local_count = local_end - local_start
                if local_count == 0:
                    continue
                original_positions = sort_perm[local_start:local_end]
                if len(assignments) == 1:
                    gpu_id = assignments[0][0]
                    single_gpu_batches.setdefault(gpu_id, []).append(original_positions)
                else:
                    my_offset = int(global_offsets_np[expert_id])
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
        target_gpus = (flat_experts // num_local_experts).to(torch.int64)

    # Sort by target GPU for AllToAll
    sorted_indices = torch.argsort(target_gpus, stable=True)
    undo_indices = torch.argsort(sorted_indices)
    sorted_scores = flat_scores[sorted_indices]
    sorted_experts = flat_experts[sorted_indices]

    # Compute split sizes from send_matrix (no extra collective needed)
    # Use vectorized numpy implementation (1.9x faster than nested loops)
    global _send_matrix_fn
    if _send_matrix_fn is None:
        try:
            from torchtitan.distributed.llep_kernels import (
                compute_send_matrix_vectorized,
            )

            _send_matrix_fn = compute_send_matrix_vectorized
        except (ImportError, RuntimeError):
            _send_matrix_fn = False

    if _send_matrix_fn and lpt_plan:
        send_matrix_np = _send_matrix_fn(
            all_counts_np,
            cum_counts_np,
            lpt_plan,
            ep_size,
            num_local_experts,
            num_experts,
        )
    elif lpt_plan:
        # Fallback: nested Python loops
        send_matrix_np = np.zeros((ep_size, ep_size), dtype=np.int64)
        lpt_expert_set = set(lpt_plan.keys())

        for eid in range(num_experts):
            if eid not in lpt_expert_set:
                owner = eid // num_local_experts
                send_matrix_np[:, owner] += all_counts_np[:, eid]

        for expert_id, assignments in lpt_plan.items():
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

    return (
        sorted_scores,
        sorted_experts,
        input_split_sizes,
        output_split_sizes,
        sorted_indices,
        undo_indices,
    )


# ---------------------------------------------------------------------------
# Grouped GEMM helpers for SwiGLU FFN
# ---------------------------------------------------------------------------
def _pack_expert_weights(
    unique_experts: torch.Tensor,
    w1_local: torch.Tensor,
    w2_local: torch.Tensor,
    w3_local: torch.Tensor,
    foreign_w1: dict[int, torch.Tensor] | None,
    foreign_w2: dict[int, torch.Tensor] | None,
    foreign_w3: dict[int, torch.Tensor] | None,
    foreign_w1_stacked: Optional[torch.Tensor],
    foreign_w2_stacked: Optional[torch.Tensor],
    foreign_w3_stacked: Optional[torch.Tensor],
    foreign_expert_id_mapping: Optional[torch.Tensor],
    ep_rank: int,
    num_local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[bool]]:
    """
    Pack weights for active experts into contiguous (num_active, ...) tensors.

    Uses differentiable indexing (torch.stack) to preserve autograd graph.

    Returns:
        (w1_packed, w2_packed, w3_packed, valid_mask)
        valid_mask[i] is True if expert i has valid weights, False if zero-filled.
    """
    native_start = ep_rank * num_local_experts
    native_end = native_start + num_local_experts
    use_stacked = foreign_expert_id_mapping is not None
    num_active = unique_experts.shape[0]
    device = w1_local.device

    # Classify experts as native vs foreign using tensor ops (no per-element .item())
    is_native = (unique_experts >= native_start) & (unique_experts < native_end)

    # --- Native experts: batch index into w1/w2/w3_local ---
    # Clamp to valid range for safe indexing (non-native entries are masked out by torch.where or zeroed)
    native_local_indices = (unique_experts - native_start).clamp(
        min=0, max=num_local_experts - 1
    )

    if (
        use_stacked
        and foreign_w1_stacked is not None
        and foreign_w1_stacked.shape[0] > 0
    ):
        # --- Stacked foreign path (autograd-compatible) ---
        # Selective indexing: only read native weights for native experts and
        # foreign weights for foreign experts. Avoids materializing both full
        # arrays in torch.where (halves memory bandwidth).
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

        # Fill native experts (selective gather from w_local)
        if is_native.any():
            native_pos = is_native.nonzero(as_tuple=True)[0]
            w1_packed[native_pos] = w1_local[native_local_indices[native_pos]]
            w2_packed[native_pos] = w2_local[native_local_indices[native_pos]]
            w3_packed[native_pos] = w3_local[native_local_indices[native_pos]]

        # Fill valid foreign experts (selective gather from w_stacked)
        if is_valid_foreign.any():
            foreign_pos = is_valid_foreign.nonzero(as_tuple=True)[0]
            safe_stacked_idx = stacked_indices[foreign_pos].clamp(min=0)
            w1_packed[foreign_pos] = foreign_w1_stacked[safe_stacked_idx]
            w2_packed[foreign_pos] = foreign_w2_stacked[safe_stacked_idx]
            w3_packed[foreign_pos] = foreign_w3_stacked[safe_stacked_idx]

        # Invalid experts remain zero from initialization
        valid_mask = (~is_invalid).tolist()
        return w1_packed, w2_packed, w3_packed, valid_mask

    elif foreign_w1 is not None:
        # --- Dict-based foreign path (non-autograd, needs Python loop for dict lookups) ---
        # Still vectorize native experts
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

        # Fill foreign experts from dicts (only non-native)
        foreign_positions = (~is_native).nonzero(as_tuple=True)[0]
        for pos in foreign_positions:
            eid = unique_experts[pos].item()
            if eid in foreign_w1:
                w1_packed[pos] = foreign_w1[eid]
                w2_packed[pos] = foreign_w2[eid]
                w3_packed[pos] = foreign_w3[eid]
                valid_mask[pos] = True

        return w1_packed, w2_packed, w3_packed, valid_mask

    else:
        # --- No foreign weights at all: only native experts are valid ---
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
                from torchtitan.distributed.llep_kernels import (
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
                from torchtitan.distributed.llep_kernels import triton_unpad_output

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


def _llep_swiglu_ffn_grouped_mm(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    w1_local: torch.Tensor,
    w2_local: torch.Tensor,
    w3_local: torch.Tensor,
    foreign_w1: dict[int, torch.Tensor] | None,
    foreign_w2: dict[int, torch.Tensor] | None,
    foreign_w3: dict[int, torch.Tensor] | None,
    ep_rank: int,
    num_local_experts: int,
    foreign_w1_stacked: Optional[torch.Tensor] = None,
    foreign_w2_stacked: Optional[torch.Tensor] = None,
    foreign_w3_stacked: Optional[torch.Tensor] = None,
    foreign_expert_id_mapping: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    SwiGLU FFN using torch._grouped_mm — replaces the Python for-loop.

    3 batched kernel launches instead of 3*N per-expert launches.
    """
    if x.numel() == 0:
        return torch.empty(0, w2_local.shape[1], device=x.device, dtype=x.dtype)

    num_tokens, dim = x.shape
    orig_dtype = x.dtype
    compute_dtype = w1_local.dtype
    if x.dtype != compute_dtype:
        x = x.to(compute_dtype)

    # Sort tokens by expert for contiguous access
    sorted_expert_ids, sort_perm = expert_ids.sort(stable=True)
    x_sorted = x[sort_perm]

    # Compute segment boundaries
    unique_experts, counts = torch.unique_consecutive(
        sorted_expert_ids, return_counts=True
    )

    # Pack weights for all active experts
    w1_packed, w2_packed, w3_packed, valid_mask = _pack_expert_weights(
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
    )

    # Pad tokens for grouped_mm alignment
    x_padded, counts_padded = _pad_for_grouped_mm(x_sorted, counts)

    # Offsets for grouped_mm (cumulative sum of padded counts)
    offsets = torch.cumsum(counts_padded, dim=0, dtype=torch.int32)

    # 3 grouped GEMMs
    x1 = torch._grouped_mm(
        x_padded.bfloat16(), w1_packed.bfloat16().transpose(-2, -1), offs=offsets
    )
    x3 = torch._grouped_mm(
        x_padded.bfloat16(), w3_packed.bfloat16().transpose(-2, -1), offs=offsets
    )

    # Fused activation: silu(x1) * x3 (cached import)
    global _fused_silu_gate_fn
    if _fused_silu_gate_fn is None:
        try:
            from torchtitan.distributed.llep_kernels import fused_silu_gate

            _fused_silu_gate_fn = fused_silu_gate
        except (ImportError, RuntimeError):
            _fused_silu_gate_fn = lambda x1, x3: F.silu(x1) * x3
    h = _fused_silu_gate_fn(x1, x3)

    out_padded = torch._grouped_mm(
        h, w2_packed.bfloat16().transpose(-2, -1), offs=offsets
    )

    # Unpad to original counts
    out_sorted = _unpad_output(out_padded, counts, counts_padded)

    # Zero out invalid experts
    if not all(valid_mask):
        offset = 0
        counts_list = counts.tolist()
        for i, valid in enumerate(valid_mask):
            c = int(counts_list[i])
            if not valid and c > 0:
                out_sorted[offset : offset + c] = 0
            offset += c

    # Cast back if needed
    if out_sorted.dtype != orig_dtype:
        out_sorted = out_sorted.to(orig_dtype)

    # Unsort
    inverse_perm = sort_perm.argsort()
    return out_sorted[inverse_perm]


# ---------------------------------------------------------------------------
# SwiGLU FFN for native + foreign experts (original for-loop, used as fallback)
# ---------------------------------------------------------------------------
def _llep_swiglu_ffn_forloop(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    w1_local: torch.Tensor,
    w2_local: torch.Tensor,
    w3_local: torch.Tensor,
    foreign_w1: dict[int, torch.Tensor] | None,
    foreign_w2: dict[int, torch.Tensor] | None,
    foreign_w3: dict[int, torch.Tensor] | None,
    ep_rank: int,
    num_local_experts: int,
    foreign_w1_stacked: Optional[torch.Tensor] = None,
    foreign_w2_stacked: Optional[torch.Tensor] = None,
    foreign_w3_stacked: Optional[torch.Tensor] = None,
    foreign_expert_id_mapping: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Original for-loop SwiGLU FFN (fallback when LLEP_USE_GROUPED_MM=0)."""
    if x.numel() == 0:
        return torch.empty(0, w2_local.shape[1], device=x.device, dtype=x.dtype)

    num_tokens, dim = x.shape
    orig_dtype = x.dtype
    native_start = ep_rank * num_local_experts
    native_end = native_start + num_local_experts

    compute_dtype = w1_local.dtype
    if x.dtype != compute_dtype:
        x = x.to(compute_dtype)

    sorted_expert_ids, sort_perm = expert_ids.sort(stable=True)
    x_sorted = x[sort_perm]

    unique_experts, counts = torch.unique_consecutive(
        sorted_expert_ids, return_counts=True
    )
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
        elif foreign_expert_id_mapping is not None:
            stacked_idx = foreign_expert_id_mapping[eid].item()
            if stacked_idx >= 0:
                w1 = foreign_w1_stacked[stacked_idx]
                w2 = foreign_w2_stacked[stacked_idx]
                w3 = foreign_w3_stacked[stacked_idx]
            else:
                out_sorted[start:end] = 0
                continue
        elif foreign_w1 is not None and eid in foreign_w1:
            w1 = foreign_w1[eid]
            w2 = foreign_w2[eid]
            w3 = foreign_w3[eid]
        else:
            out_sorted[start:end] = 0
            continue

        h = F.silu(x_slice @ w1.T) * (x_slice @ w3.T)
        out_sorted[start:end] = h @ w2.T

    inverse_perm = sort_perm.argsort()
    result = out_sorted[inverse_perm]
    if result.dtype != orig_dtype:
        result = result.to(orig_dtype)
    return result


# ---------------------------------------------------------------------------
# SwiGLU FFN for native + foreign experts
# ---------------------------------------------------------------------------
def llep_swiglu_ffn(
    x: torch.Tensor,  # (num_tokens, dim)
    expert_ids: torch.Tensor,  # (num_tokens,) global expert IDs
    w1_local: torch.Tensor,  # (num_local_experts, hidden_dim, dim)
    w2_local: torch.Tensor,  # (num_local_experts, dim, hidden_dim)
    w3_local: torch.Tensor,  # (num_local_experts, hidden_dim, dim)
    foreign_w1: dict[int, torch.Tensor] | None,
    foreign_w2: dict[int, torch.Tensor] | None,
    foreign_w3: dict[int, torch.Tensor] | None,
    ep_rank: int,
    num_local_experts: int,
    # Stacked tensor interface (for autograd):
    foreign_w1_stacked: Optional[torch.Tensor] = None,
    foreign_w2_stacked: Optional[torch.Tensor] = None,
    foreign_w3_stacked: Optional[torch.Tensor] = None,
    foreign_expert_id_mapping: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Run SwiGLU FFN on tokens using both native and foreign expert weights.

    Groups tokens by expert, then for each expert:
      h = silu(x @ w1.T) * (x @ w3.T)
      out = h @ w2.T

    When LLEP_USE_GROUPED_MM=1 (default), uses torch._grouped_mm to batch
    all expert computations into 3 kernel launches instead of 3*N.
    Set LLEP_USE_GROUPED_MM=0 to fall back to the original per-expert for-loop.
    """
    args = (
        x,
        expert_ids,
        w1_local,
        w2_local,
        w3_local,
        foreign_w1,
        foreign_w2,
        foreign_w3,
        ep_rank,
        num_local_experts,
    )
    kwargs = dict(
        foreign_w1_stacked=foreign_w1_stacked,
        foreign_w2_stacked=foreign_w2_stacked,
        foreign_w3_stacked=foreign_w3_stacked,
        foreign_expert_id_mapping=foreign_expert_id_mapping,
    )

    if LLEP_USE_GROUPED_MM and x.is_cuda:
        return _llep_swiglu_ffn_grouped_mm(*args, **kwargs)
    return _llep_swiglu_ffn_forloop(*args, **kwargs)


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
    1. Gather per-rank expert counts (all_gather, not all_reduce)
    2. Compute LPT assignment plan
    3. Async P2P transfer expert weights to helper GPUs
    4. Vectorized token assignment + merged AllToAll dispatch (overlapped with P2P)
    5. Wait for weight transfer, run SwiGLU FFN
    6. AllToAll route outputs back
    7. Apply routing scores and aggregate
    """
    from torch.distributed.tensor import DTensor

    # Unwrap DTensors to local, properly handling EP vs FSDP sharding.
    #
    # The DTensor may live on a multi-dim mesh (e.g. ["efsdp", "ep"]) where:
    #   - "ep" dim: Shard(0) partitions experts across EP ranks (keep this)
    #   - "efsdp" dim: FSDP sharding for memory savings (undo this)
    # FSDP2 uses _StridedShard (not regular Shard) for its placement, so we
    # use a negative check: any non-Replicate placement on a non-"ep" dim
    # must be redistributed to Replicate.
    if isinstance(w1, DTensor):
        from torch.distributed.tensor import Replicate

        dim_names = w1.device_mesh.mesh_dim_names
        new_placements = list(w1.placements)
        need_redistribute = False

        for i, p in enumerate(w1.placements):
            if not isinstance(p, Replicate) and dim_names[i] != "ep":
                new_placements[i] = Replicate()
                need_redistribute = True

        if need_redistribute:
            new_placements = tuple(new_placements)
            w1_local = w1.redistribute(placements=new_placements).to_local()
            w2_local = w2.redistribute(placements=new_placements).to_local()
            w3_local = w3.redistribute(placements=new_placements).to_local()
        else:
            w1_local = w1.to_local()
            w2_local = w2.to_local()
            w3_local = w3.to_local()
    else:
        w1_local = w1
        w2_local = w2
        w3_local = w3

    ep_rank = dist.get_rank(group=ep_group)
    ep_size = dist.get_world_size(group=ep_group)

    device = hidden_states.device

    # Ensure weights are on the same device as hidden_states (handles CPU offload)
    if w1_local.device != device:
        w1_local = w1_local.to(device)
        w2_local = w2_local.to(device)
        w3_local = w3_local.to(device)

    num_local_experts = num_experts // ep_size
    dtype = hidden_states.dtype
    num_tokens, dim = hidden_states.shape
    top_k = selected_experts_indices.shape[1]

    # Override from env (cached at module level, not re-read per call)
    if _ENV_MAX_TOKENS_FACTOR is not None:
        max_tokens_factor = float(_ENV_MAX_TOKENS_FACTOR)
    if _ENV_MIN_TOKENS_PER_GEMM is not None:
        min_tokens_per_gemm = int(_ENV_MIN_TOKENS_PER_GEMM)
    if _ENV_ADAPTIVE_THRESHOLD is not None:
        adaptive_threshold = float(_ENV_ADAPTIVE_THRESHOLD)

    # ------------------------------------------------------------------
    # Step 1: Gather per-rank expert counts via all_gather
    # (all_gather gives us per-rank counts needed for vectorized routing;
    #  global counts = sum of per-rank counts)
    # ------------------------------------------------------------------
    local_expert_counts = torch.bincount(
        selected_experts_indices.view(-1).to(torch.int64),
        minlength=num_experts,
    ).to(torch.int64)

    all_expert_counts = [torch.zeros_like(local_expert_counts) for _ in range(ep_size)]
    dist.all_gather(all_expert_counts, local_expert_counts, group=ep_group)

    global_expert_counts = torch.stack(all_expert_counts).sum(dim=0)

    # Truncate to actual expert count (num_local_experts * ep_size).
    # bincount may produce a larger tensor if router outputs unexpected IDs.
    effective_num_experts = num_local_experts * ep_size
    if global_expert_counts.size(0) > effective_num_experts:
        global_expert_counts = global_expert_counts[:effective_num_experts]
        all_expert_counts = [ec[:effective_num_experts] for ec in all_expert_counts]
        local_expert_counts = local_expert_counts[:effective_num_experts]

    # ------------------------------------------------------------------
    # Step 2: Adaptive threshold check
    # ------------------------------------------------------------------
    imbalance = compute_gpu_imbalance_ratio(
        global_expert_counts, ep_size, num_local_experts
    )
    use_lpt = True
    if adaptive_threshold > 0:
        use_lpt = imbalance >= adaptive_threshold

    if LLEP_DEBUG and ep_rank == 0:
        expert_counts_list = global_expert_counts.cpu().tolist()
        logger.info(
            f"[LLEP] tokens={num_tokens} top_k={top_k} "
            f"expert_counts={expert_counts_list} "
            f"imbalance={imbalance:.2f} use_lpt={use_lpt}"
        )

    # ------------------------------------------------------------------
    # Step 3: Compute LPT plan
    # ------------------------------------------------------------------
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

    if LLEP_DEBUG and ep_rank == 0:
        logger.info(
            f"[LLEP] gpu_loads={plan.gpu_loads.tolist()} "
            f"weight_transfers={len(plan.weight_transfers)} "
            f"send={plan.weights_to_send} recv={plan.weights_to_receive}"
        )

    # ------------------------------------------------------------------
    # Step 4: Barrier before P2P to prevent cross-layer deadlock
    # ------------------------------------------------------------------
    dist.barrier(group=ep_group)

    # ------------------------------------------------------------------
    # Step 5: Async weight transfer (overlap with token routing + A2A)
    # ------------------------------------------------------------------
    gradient_anchor = None
    foreign_w1_stacked = foreign_w2_stacked = foreign_w3_stacked = None
    foreign_expert_id_mapping = None

    if LLEP_W_TRANSFER_AUTOGRAD:
        (
            foreign_w1_stacked,
            foreign_w2_stacked,
            foreign_w3_stacked,
            foreign_expert_id_mapping,
            p2p_handles,
            gradient_anchor,
        ) = transfer_expert_weights_autograd(
            ep_rank,
            ep_group,
            plan,
            w1_local,
            w2_local,
            w3_local,
            num_local_experts,
            effective_num_experts,
            return_handles=True,
        )
        foreign_w1 = foreign_w2 = foreign_w3 = None
    else:
        foreign_w1, foreign_w2, foreign_w3, p2p_handles = transfer_expert_weights(
            ep_rank,
            ep_group,
            plan,
            w1_local,
            w2_local,
            w3_local,
            num_local_experts,
            return_handles=True,
        )

    # ------------------------------------------------------------------
    # Step 6: Vectorized token assignment (runs while P2P is in flight)
    # ------------------------------------------------------------------
    (
        sorted_scores,
        sorted_experts,
        input_split_sizes,
        output_split_sizes,
        sorted_indices,
        undo_indices,
    ) = assign_tokens_to_gpus(
        top_scores,
        selected_experts_indices,
        plan.lpt_plan,
        ep_size,
        ep_rank,
        num_local_experts,
        all_expert_counts,
        local_expert_counts,
    )

    # Expand hidden states for top_k and sort
    hidden_topk = hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, dim)
    sorted_hidden = hidden_topk[sorted_indices]

    # ------------------------------------------------------------------
    # Step 7: AllToAll dispatch
    # ------------------------------------------------------------------
    total_send = sum(input_split_sizes)
    total_recv = sum(output_split_sizes)

    # Determine if we can merge all tensors into one A2A call
    # bf16 can represent integers 0-255 exactly; float16 up to 2048
    can_merge = LLEP_MERGE_A2A
    if dtype == torch.bfloat16 and num_experts > 256:
        can_merge = False
    elif dtype == torch.float16 and num_experts > 2048:
        can_merge = False

    use_autograd_a2a = LLEP_W_TRANSFER_AUTOGRAD

    if total_send > 0 or total_recv > 0:
        if can_merge:
            # Merge hidden (dim cols) + scores (1 col) + expert_ids (1 col)
            merged_send = torch.cat(
                [
                    sorted_hidden,
                    sorted_scores.unsqueeze(1).to(dtype),
                    sorted_experts.unsqueeze(1).to(dtype),
                ],
                dim=1,
            )  # (total_send, dim + 2)

            if use_autograd_a2a:
                merged_recv = a2a_autograd(
                    merged_send, output_split_sizes, input_split_sizes, ep_group
                )
            else:
                merged_recv = torch.empty(
                    total_recv, dim + 2, device=device, dtype=dtype
                )
                dist.all_to_all_single(
                    merged_recv,
                    merged_send,
                    output_split_sizes,
                    input_split_sizes,
                    group=ep_group,
                )

            recv_hidden = merged_recv[:, :dim]
            recv_scores = merged_recv[:, dim]
            recv_experts = merged_recv[:, dim + 1].to(torch.int64)
        else:
            # Separate A2A calls
            if use_autograd_a2a:
                recv_hidden = a2a_autograd(
                    sorted_hidden, output_split_sizes, input_split_sizes, ep_group
                )
                # A2A scores as 2D for all_to_all_single compatibility
                recv_scores = a2a_autograd(
                    sorted_scores.unsqueeze(1).to(dtype),
                    output_split_sizes,
                    input_split_sizes,
                    ep_group,
                ).squeeze(1)
            else:
                recv_hidden = torch.empty(total_recv, dim, device=device, dtype=dtype)
                recv_scores = torch.empty(
                    total_recv, device=device, dtype=sorted_scores.dtype
                )
                dist.all_to_all_single(
                    recv_hidden,
                    sorted_hidden,
                    output_split_sizes,
                    input_split_sizes,
                    group=ep_group,
                )
                dist.all_to_all_single(
                    recv_scores,
                    sorted_scores,
                    output_split_sizes,
                    input_split_sizes,
                    group=ep_group,
                )
            # Expert indices: always non-differentiable (integer)
            recv_experts = torch.empty(total_recv, device=device, dtype=torch.int64)
            dist.all_to_all_single(
                recv_experts,
                sorted_experts.to(torch.int64),
                output_split_sizes,
                input_split_sizes,
                group=ep_group,
            )
    else:
        recv_hidden = sorted_hidden.new_empty(0, dim)
        recv_scores = sorted_scores.new_empty(0)
        recv_experts = torch.empty(0, device=device, dtype=torch.int64)

    # ------------------------------------------------------------------
    # Step 8: Wait for weight transfer to complete
    # ------------------------------------------------------------------
    if p2p_handles is not None:
        if isinstance(p2p_handles, list):
            for h in p2p_handles:
                h.wait()

    # ------------------------------------------------------------------
    # Step 9: Apply routing scores before experts (if configured)
    # ------------------------------------------------------------------
    if score_before_experts and recv_hidden.numel() > 0:
        recv_hidden = (recv_hidden.to(torch.float32) * recv_scores.reshape(-1, 1)).to(
            dtype
        )

    # ------------------------------------------------------------------
    # Step 10: Run SwiGLU FFN on received tokens
    # ------------------------------------------------------------------
    if recv_hidden.numel() > 0:
        recv_output = llep_swiglu_ffn(
            recv_hidden,
            recv_experts.to(torch.int64),
            w1_local,
            w2_local,
            w3_local,
            foreign_w1,
            foreign_w2,
            foreign_w3,
            ep_rank,
            num_local_experts,
            foreign_w1_stacked=foreign_w1_stacked,
            foreign_w2_stacked=foreign_w2_stacked,
            foreign_w3_stacked=foreign_w3_stacked,
            foreign_expert_id_mapping=foreign_expert_id_mapping,
        )
    else:
        recv_output = torch.empty(0, dim, device=device, dtype=dtype)
        # Touch weights for autograd graph recording (ensures gradients flow
        # even when this rank processes no tokens from a given layer)
        if LLEP_W_TRANSFER_AUTOGRAD:
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
            recv_output = recv_output + weight_touch

    # ------------------------------------------------------------------
    # Step 11: Apply routing scores after experts (if configured)
    # ------------------------------------------------------------------
    if not score_before_experts and recv_output.numel() > 0:
        recv_output = (recv_output.to(torch.float32) * recv_scores.reshape(-1, 1)).to(
            dtype
        )

    # ------------------------------------------------------------------
    # Step 12: AllToAll combine - route outputs back
    # ------------------------------------------------------------------
    if total_send > 0 or total_recv > 0:
        if use_autograd_a2a:
            # Reverse A2A: swap input/output split sizes
            send_output = a2a_autograd(
                recv_output, input_split_sizes, output_split_sizes, ep_group
            )
        else:
            send_output = torch.empty(total_send, dim, device=device, dtype=dtype)
            dist.all_to_all_single(
                send_output,
                recv_output,
                input_split_sizes,
                output_split_sizes,
                group=ep_group,
            )
    else:
        send_output = recv_output.new_empty(0, dim)

    # ------------------------------------------------------------------
    # Step 13: Unsort and scatter-add back to original token positions
    # ------------------------------------------------------------------
    unsorted_output = send_output[undo_indices]
    unsorted_output = unsorted_output.view(num_tokens, top_k, dim).sum(dim=1)

    # Add gradient anchor to ensure all ranks enter backward for P2P
    if gradient_anchor is not None:
        unsorted_output = unsorted_output + gradient_anchor

    return unsorted_output
