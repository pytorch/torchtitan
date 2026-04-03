# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
HybridEP: Expert Parallel Communication for GB200 NVLink72 Systems.

Provides efficient token dispatch/combine for MoE training via TMA-optimized all-to-all.

Configuration (via ParallelismConfig):
    hybridep_non_blocking_expert_capacity_factor: float | None
        None = blocking mode (default).  HybridEP calls cudaStreamSynchronize
        after dispatch and computes the exact num_permuted_tokens on the host.
        float in (0, 1] = non-blocking mode; num_permuted_tokens is estimated as
        num_tokens × ep_size × min(num_local_experts, top_k) × cf, aligned for
        MXFP8.  See _num_permuted_tokens_for_non_blocking().
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch._library.opaque_object import (
    get_opaque_type_name,
    OpaqueBase,
    register_opaque_type,
)
from torch.distributed import ProcessGroup

_buffer: Any = None  # Global buffer instance


class DispatchHandle(OpaqueBase):
    """Opaque wrapper for HybridEP dispatch handle.

    Wraps the deep_ep dispatch handle as an opaque type so it can be returned
    from custom ops and flow through the torch.compile graph, eliminating
    the need for a global handle cache.
    """

    def __init__(self, value=None):
        self.value = value

    def __eq__(self, other):
        if not isinstance(other, DispatchHandle):
            return False
        return self.value is other.value

    def __hash__(self):
        if self.value is None:
            return 0
        try:
            return hash(self.value)
        except TypeError:
            return id(self.value)

    def __fx_repr__(self):
        return "DispatchHandle()", {"DispatchHandle": DispatchHandle}


register_opaque_type(DispatchHandle, typ="reference")


@dataclass
class DispatchState:
    """State from dispatch needed for combine.

    Attributes:
        handle: Opaque dispatch handle wrapping the deep_ep handle.
        permuted_scores: Scores for score_before_experts=False mode.
        num_tokens: Original input token count (for combine fake shape inference).
    """

    handle: DispatchHandle
    permuted_scores: torch.Tensor | None = None
    num_tokens: int = 0


def _apply_scores(
    hidden: torch.Tensor,
    scores: torch.Tensor,
    apply_now: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply routing scores to hidden states if apply_now, else defer."""
    if apply_now and scores is not None and scores.numel() > 0:
        return hidden * scores.to(hidden.dtype).reshape(-1, 1), None
    return hidden, scores


# Custom op registration for torch.compile and SAC compatibility
_handle_type = get_opaque_type_name(DispatchHandle)

torch.library.define(
    "hybridep::dispatch",
    f"(Tensor x, Tensor topk_idx, Tensor topk_weights, int num_experts, "
    f"bool non_blocking, float? moe_expert_capacity_factor, int? pad_multiple) -> (Tensor, Tensor, Tensor, {_handle_type})",
)

torch.library.define(
    "hybridep::combine",
    f"(Tensor x, {_handle_type} handle, int num_tokens, int? pad_multiple) -> Tensor",
)


def _num_permuted_tokens_for_non_blocking(
    num_tokens: int,
    ep_size: int,
    num_local_experts: int,
    top_k: int,
    moe_expert_capacity_factor: float,
    pad_multiple: int | None = None,
) -> int:
    """Pre-allocated output buffer size for non-blocking dispatch.

    Formula: num_tokens × ep_size × min(num_local_experts, top_k) × cf,
    aligned to pad_multiple if set.

    capacity_factor=1.0 sizes for the worst case (every token routed to
    every local expert) — no tokens are dropped but memory usage is highest.
    Values < 1.0 reduce memory at the cost of potentially dropping tokens
    when the permuted offset exceeds the buffer capacity.  With forced load
    balancing (e.g. routing_algo="round_robin"), token distribution across experts
    is roughly uniform, so values < 1.0 are safe in practice.
    """
    n = int(
        num_tokens
        * ep_size
        * min(num_local_experts, top_k)
        * moe_expert_capacity_factor
    )
    if pad_multiple is not None:
        n = ((n + pad_multiple - 1) // pad_multiple) * pad_multiple
    return n


@torch.library.impl("hybridep::dispatch", "CUDA")
def _dispatch_impl(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    non_blocking: bool = False,
    moe_expert_capacity_factor: float | None = None,
    pad_multiple: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, DispatchHandle]:
    """
    DeepEP's dispatch_with_permute needs to know the output buffer size
    (num_permuted_tokens) for the fused permute kernel.

    * **non_blocking=True** — no D2H sync is allowed, so num_permuted_tokens
      must be supplied upfront via moe_expert_capacity_factor.
    * **non_blocking=False (blocking)** — DeepEP does cudaStreamSynchronize,
      then reads tokens_per_expert from pinned CPU memory to compute the
      exact num_permuted_tokens on the host.
    """
    global _buffer
    if _buffer is None:
        raise RuntimeError(
            "HybridEP buffer not initialized. Call dispatch_tokens() first."
        )

    num_local_experts = num_experts // _buffer.group_size
    from deep_ep.hybrid_ep_buffer import (  # pyrefly: ignore [missing-import]
        indices_to_map,
    )

    routing_map, probs = indices_to_map(
        topk_idx, topk_weights.float(), x.shape[0], num_experts
    )

    num_permuted_tokens = None
    if non_blocking:
        assert (
            moe_expert_capacity_factor is not None
        ), "moe_expert_capacity_factor is required for non_blocking dispatch"
        num_permuted_tokens = _num_permuted_tokens_for_non_blocking(
            x.shape[0],
            _buffer.group_size,
            num_local_experts,
            topk_idx.shape[1],
            moe_expert_capacity_factor,  # pyrefly: ignore [bad-argument-type]
            pad_multiple=pad_multiple,
        )

    hidden, scores, _, tokens_per_expert, handle = _buffer.dispatch_with_permute(
        hidden=x,
        routing_map=routing_map,
        probs=probs,
        scaling_factor=None,
        num_of_experts_per_rank=num_local_experts,
        pad_multiple=pad_multiple,
        num_permuted_tokens=num_permuted_tokens,
        non_blocking=non_blocking,
    )

    # NOTE: In non_blocking mode, overflow_flag lives on GPU so checking it
    # (.item()) would force cudaStreamSynchronize, defeating the purpose.
    # Overflow is governed by num_permuted_tokens (the output buffer capacity
    # for the fused permute kernel) — tokens whose permuted offset exceeds
    # that limit are silently dropped.  Correct sizing of num_permuted_tokens
    # via _num_permuted_tokens_for_non_blocking is therefore critical:
    # capacity_factor=1.0 → worst-case sizing, no drops, most memory;
    # capacity_factor<1.0 → less memory, but tokens may be dropped.

    if scores is None:
        scores = torch.empty(0, device=x.device, dtype=torch.float32)
    if tokens_per_expert.device != x.device:
        tokens_per_expert = tokens_per_expert.to(x.device)

    return hidden, scores, tokens_per_expert, DispatchHandle(value=handle)


@torch.library.register_fake("hybridep::dispatch")
def _dispatch_fake(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    non_blocking: bool = False,
    moe_expert_capacity_factor: float | None = None,
    pad_multiple: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, DispatchHandle]:
    """Fake dispatch for torch.compile tracing."""
    num_local_experts = num_experts // _buffer.group_size
    if non_blocking:
        out_tokens = _num_permuted_tokens_for_non_blocking(
            x.shape[0],
            _buffer.group_size,
            num_local_experts,
            topk_idx.shape[1],
            moe_expert_capacity_factor,  # pyrefly: ignore [bad-argument-type]
            pad_multiple=pad_multiple,
        )
    else:
        out_tokens = x.shape[0]
    hidden = x.new_empty(out_tokens, x.shape[1])
    scores = x.new_empty(0, dtype=torch.float32)
    tpe = x.new_empty(num_local_experts, dtype=torch.int64)
    return hidden, scores, tpe, DispatchHandle()


@torch.library.impl("hybridep::combine", "CUDA")
def _combine_impl(
    x: torch.Tensor,
    handle: DispatchHandle,
    num_tokens: int,
    pad_multiple: int | None = None,
) -> torch.Tensor:
    """CUDA combine: reverse dispatch permutation via opaque handle."""
    global _buffer
    if _buffer is None:
        raise RuntimeError("HybridEP buffer not initialized.")

    combined, _ = _buffer.combine_with_unpermute(hidden=x, handle=handle.value)
    return combined


@torch.library.register_fake("hybridep::combine")
def _combine_fake(
    x: torch.Tensor,
    handle: DispatchHandle,
    num_tokens: int,
    pad_multiple: int | None = None,
) -> torch.Tensor:
    """Fake combine for torch.compile tracing."""
    return x.new_empty(num_tokens, x.shape[1])


def _dispatch_backward(ctx, grad_hidden, grad_scores, grad_tpe, grad_handle):
    """Backward: gather gradients via combine."""
    if grad_hidden is None:
        return None, None, None, None, None, None, None

    dispatch_handle = ctx.dispatch_handle
    if dispatch_handle is None or dispatch_handle.value is None:
        raise RuntimeError("DispatchHandle not found in dispatch backward")

    (topk_idx,) = ctx.saved_tensors
    grad_x, grad_probs_dense = _buffer.combine_with_unpermute(
        hidden=grad_hidden,
        probs=(
            grad_scores if grad_scores is not None and grad_scores.numel() > 0 else None
        ),
        handle=dispatch_handle.value,
    )
    grad_x = grad_x.to(ctx.input_dtype)

    # grad_probs_dense is [num_tokens, num_experts]; gather back to sparse [num_tokens, top_k]
    grad_weights = (
        grad_probs_dense.gather(dim=1, index=topk_idx)
        if grad_probs_dense is not None
        else None
    )
    # Gradients for: x, topk_idx, topk_weights, num_experts, non_blocking,
    #                moe_expert_capacity_factor, pad_multiple
    return grad_x, None, grad_weights, None, None, None, None


def _dispatch_setup_context(ctx, inputs, output):
    """Save context for dispatch backward."""
    x, topk_idx, _, _, _, _, _ = inputs
    _, _, _, dispatch_handle = output
    ctx.dispatch_handle = dispatch_handle
    ctx.input_dtype = x.dtype
    ctx.save_for_backward(topk_idx)


def _combine_backward(ctx, grad_combined):
    """Backward: scatter gradients via dispatch."""
    dispatch_handle = ctx.dispatch_handle
    if dispatch_handle is None or dispatch_handle.value is None:
        raise RuntimeError("DispatchHandle not found in combine backward")

    grad_x, _, _, _, _ = _buffer.dispatch_with_permute(
        hidden=grad_combined,
        scaling_factor=None,
        handle=dispatch_handle.value,
        num_permuted_tokens=ctx.num_permuted_tokens,
        pad_multiple=ctx.pad_multiple,
    )
    # Gradients for: x, handle, num_tokens, pad_multiple
    return grad_x, None, None, None


def _combine_setup_context(ctx, inputs, output):
    """Save context for combine backward."""
    x, dispatch_handle, _num_tokens, pad_multiple = inputs
    ctx.dispatch_handle = dispatch_handle
    ctx.num_permuted_tokens = x.shape[0]
    ctx.pad_multiple = pad_multiple


torch.library.register_autograd(
    "hybridep::dispatch", _dispatch_backward, setup_context=_dispatch_setup_context
)
torch.library.register_autograd(
    "hybridep::combine", _combine_backward, setup_context=_combine_setup_context
)


_NUM_SMS_DISPATCH = 16
_NUM_SMS_COMBINE = 16


def get_buffer(
    group: ProcessGroup,
    hidden_dim: int,
    num_tokens: int,
    num_local_experts: int,
    fp8_dispatch: bool = False,
) -> None:
    """Ensure the global HybridEP buffer is initialized, reinitializing if config changed.

    Allocates the all-to-all communication buffers (RDMA inter-node + NVLink IPC
    intra-node), sized by num_tokens as max_num_of_tokens_per_rank.  No capacity
    factor is applied here — these buffers hold the per-rank input tokens, not the
    fan-out permuted output.  HybridEP auto-grows via update_template_config if a
    later dispatch has more tokens than the initial allocation.
    """
    global _buffer

    if fp8_dispatch:
        raise AssertionError("HybridEP FP8 dispatch not yet supported")

    try:
        from deep_ep import HybridEPBuffer  # pyrefly: ignore [missing-import]
    except ImportError as e:
        raise ImportError(
            "HybridEP requires deep_ep library. "
            "Install from: https://github.com/deepseek-ai/DeepEP, branch: hybrid-ep"
        ) from e

    max_tokens_per_rank = num_tokens

    needs_reinit = (
        _buffer is None
        or _buffer.group != group
        or _buffer.config.hidden_dim < hidden_dim
        or _buffer.config.max_num_of_tokens_per_rank < max_tokens_per_rank
        or _buffer.config.num_of_experts_per_rank < num_local_experts
    )

    if needs_reinit:
        _buffer = HybridEPBuffer(
            group=group,
            hidden_dim=hidden_dim,
            max_num_of_tokens_per_rank=max_tokens_per_rank,
            num_local_experts=num_local_experts,
            use_fp8=fp8_dispatch,
            num_sms_dispatch_api=_NUM_SMS_DISPATCH,
            num_sms_combine_api=_NUM_SMS_COMBINE,
            load_cached_kernels=True,
            use_shared_buffer=True,
            enable_custom_allgather=True,
        )


def dispatch_tokens(
    hidden_states: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    top_scores: torch.Tensor,
    num_local_experts: int,
    num_experts: int,
    group: ProcessGroup,
    score_before_experts: bool = True,
    non_blocking_expert_capacity_factor: float | None = None,
    pad_multiple: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, DispatchState]:
    """Dispatch tokens to experts via HybridEP all-to-all.

    Args:
        hidden_states: [num_tokens, hidden_dim]
        selected_experts_indices: [num_tokens, top_k]
        top_scores: [num_tokens, top_k]
        num_local_experts: Experts on this EP rank
        num_experts: Total experts across all ranks
        group: EP ProcessGroup
        score_before_experts: Apply scores before expert computation
        non_blocking_expert_capacity_factor: None = blocking mode (default).
            float in (0, 1] = non-blocking mode; pre-sizes the permute output
            tensor as num_tokens × ep_size × min(num_local_experts, top_k) × cf,
            aligned to pad_multiple.
        pad_multiple: Pad per-expert token groups to this multiple (e.g. 32 for
            MXFP8). None means no padding.

    Returns:
        (permuted_hidden, tokens_per_expert, state)
    """
    non_blocking = non_blocking_expert_capacity_factor is not None

    selected_experts_indices = selected_experts_indices.contiguous()
    top_scores = top_scores.contiguous()

    get_buffer(
        group=group,
        hidden_dim=hidden_states.shape[1],
        num_tokens=hidden_states.shape[0],
        num_local_experts=num_local_experts,
    )

    (
        hidden,
        permuted_scores,
        tokens_per_expert,
        dispatch_handle,
    ) = torch.ops.hybridep.dispatch(
        hidden_states,
        selected_experts_indices,
        top_scores,
        num_experts,
        non_blocking,
        non_blocking_expert_capacity_factor,
        pad_multiple,
    )

    hidden, permuted_scores = _apply_scores(
        hidden, permuted_scores, score_before_experts
    )
    if permuted_scores is not None and permuted_scores.dtype != hidden.dtype:
        permuted_scores = permuted_scores.to(hidden.dtype)

    state = DispatchState(
        handle=dispatch_handle,
        permuted_scores=permuted_scores,
        num_tokens=hidden_states.shape[0],
    )
    return hidden, tokens_per_expert, state


def combine_tokens(
    hidden_states: torch.Tensor,
    state: DispatchState,
    pad_multiple: int | None = None,
) -> torch.Tensor:
    """Combine expert outputs back to original token order.

    Applies deferred scores (if any), then unpermutes via the opaque dispatch handle.
    """
    if state.permuted_scores is not None:
        hidden_states = hidden_states * state.permuted_scores.reshape(-1, 1)

    return torch.ops.hybridep.combine(
        hidden_states, state.handle, state.num_tokens, pad_multiple
    )


__all__ = [
    "dispatch_tokens",
    "combine_tokens",
    "DispatchState",
    "DispatchHandle",
]
