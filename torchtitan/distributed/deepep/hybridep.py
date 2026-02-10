# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
HybridEP: Expert Parallel Communication for GB200 NVLink72 Systems.

Provides efficient token dispatch/combine for MoE training via TMA-optimized all-to-all.

Configuration (via job_config.parallelism.hybridep):
    moe_expert_capacity_factor: Capacity factor per expert in (0, 1]. None means no token dropping.
    enable_non_blocking: Enable CPU-free non-blocking dispatch mode (default: False).
        When True, pre-allocates the output buffer as num_tokens × ep_size × min(num_local_experts, top_k) x moe_expert_capacity_factor.
        If moe_expert_capacity_factor is set, the buffer is further scaled by that factor.
        When False, uses blocking mode with D2H for dynamic sizing.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from torch.distributed import ProcessGroup
from torch._library.opaque_object import (
    OpaqueBase,
    register_opaque_type,
    get_opaque_type_name,
)
from torchtitan.components.quantization import MXFP8_GROUP_ALIGNMENT_SIZE
from torchtitan.models.moe.utils import (
    TOKEN_GROUP_ALIGN_SIZE_M,
    maybe_align_num_tokens_for_mxfp8,
)


_hybrid_ep_cls: Any = None  # Lazily-loaded HybridEPBuffer class
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
    permuted_scores: Optional[torch.Tensor] = None
    num_tokens: int = 0


def _apply_scores(
    hidden: torch.Tensor,
    scores: torch.Tensor,
    apply_now: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Apply routing scores to hidden states if apply_now, else defer."""
    if apply_now and scores is not None and scores.numel() > 0:
        return hidden * scores.to(hidden.dtype).reshape(-1, 1), None
    return hidden, scores


def _require_hybridep() -> Any:
    """Lazily import HybridEPBuffer, raising helpful error if unavailable."""
    global _hybrid_ep_cls
    if _hybrid_ep_cls is not None:
        return _hybrid_ep_cls
    try:
        from deep_ep import HybridEPBuffer
    except ImportError as e:
        raise ImportError(
            "HybridEP requires deep_ep library. "
            "Install from: https://github.com/deepseek-ai/DeepEP, branch: hybrid-ep"
        ) from e
    _hybrid_ep_cls = HybridEPBuffer
    return _hybrid_ep_cls


# Custom op registration for torch.compile and SAC compatibility
_handle_type = get_opaque_type_name(DispatchHandle)

torch.library.define(
    "hybridep::dispatch",
    f"(Tensor x, Tensor topk_idx, Tensor topk_weights, int num_experts, "
    f"int? num_permuted_tokens, float? moe_expert_capacity_factor) -> (Tensor, Tensor, Tensor, {_handle_type})",
)

torch.library.define(
    "hybridep::combine",
    f"(Tensor x, {_handle_type} handle, int num_tokens) -> Tensor",
)


@torch.library.impl("hybridep::dispatch", "CUDA")
def _dispatch_impl(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    num_permuted_tokens: Optional[int] = None,
    moe_expert_capacity_factor: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, DispatchHandle]:
    """CUDA dispatch: convert sparse routing to dense, call TMA-optimized all-to-all."""
    global _buffer
    if _buffer is None:
        raise RuntimeError(
            "HybridEP buffer not initialized. Call dispatch_tokens() first."
        )

    num_local_experts = num_experts // _buffer.group_size
    from deep_ep.hybrid_ep_buffer import indices_to_map

    routing_map, probs = indices_to_map(
        topk_idx, topk_weights.float(), x.shape[0], num_experts
    )

    # MXFP8 requires per-expert-group padding to multiples of 32 (scaling block size).
    # HybridEP's kernel handles this natively via pad_multiple.
    pad_multiple = (
        MXFP8_GROUP_ALIGNMENT_SIZE
        if TOKEN_GROUP_ALIGN_SIZE_M == MXFP8_GROUP_ALIGNMENT_SIZE
        else None
    )

    hidden, scores, _, tokens_per_expert, handle = _buffer.dispatch_with_permute(
        hidden=x,
        routing_map=routing_map,
        probs=probs,
        scaling_factor=None,
        num_of_experts_per_rank=num_local_experts,
        pad_multiple=pad_multiple,
        num_permuted_tokens=num_permuted_tokens,
        non_blocking=num_permuted_tokens is not None,
    )

    # NOTE: No synchronous overflow check here — calling .item() on GPU tensors
    # would trigger cudaStreamSynchronize and defeat the purpose of non_blocking mode.
    # If the buffer is too small, DeepEP silently drops tokens (handle.overflow_flag).
    # Rely on correct buffer sizing in get_buffer() instead.

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
    num_permuted_tokens: Optional[int] = None,
    moe_expert_capacity_factor: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, DispatchHandle]:
    """Fake dispatch for torch.compile tracing."""
    if num_permuted_tokens is not None:
        out_tokens = maybe_align_num_tokens_for_mxfp8(num_permuted_tokens)
    else:
        out_tokens = x.shape[0]
    hidden = x.new_empty(out_tokens, x.shape[1])
    scores = x.new_empty(0, dtype=torch.float32)
    # _buffer is guaranteed initialized: get_buffer() runs before this op is called.
    num_local_experts = num_experts // _buffer.group_size
    tpe = x.new_empty(num_local_experts, dtype=torch.int64)
    return hidden, scores, tpe, DispatchHandle()


@torch.library.impl("hybridep::combine", "CUDA")
def _combine_impl(
    x: torch.Tensor, handle: DispatchHandle, num_tokens: int
) -> torch.Tensor:
    """CUDA combine: reverse dispatch permutation via opaque handle."""
    global _buffer
    if _buffer is None:
        raise RuntimeError("HybridEP buffer not initialized.")

    combined, _ = _buffer.combine_with_unpermute(hidden=x, handle=handle.value)
    return combined


@torch.library.register_fake("hybridep::combine")
def _combine_fake(
    x: torch.Tensor, handle: DispatchHandle, num_tokens: int
) -> torch.Tensor:
    """Fake combine for torch.compile tracing."""
    return x.new_empty(num_tokens, x.shape[1])


def _dispatch_backward(ctx, grad_hidden, grad_scores, grad_tpe, grad_handle):
    """Backward: gather gradients via combine."""
    if grad_hidden is None:
        return None, None, None, None, None

    dispatch_handle = ctx.dispatch_handle
    if dispatch_handle is None or dispatch_handle.value is None:
        raise RuntimeError("DispatchHandle not found in dispatch backward")

    (topk_idx,) = ctx.saved_tensors
    grad_x, grad_probs_dense = _buffer.combine_with_unpermute(
        hidden=grad_hidden,
        probs=grad_scores if grad_scores is not None and grad_scores.numel() > 0 else None,
        handle=dispatch_handle.value,
    )
    grad_x = grad_x.to(ctx.input_dtype)

    # grad_probs_dense is [num_tokens, num_experts]; gather back to sparse [num_tokens, top_k]
    grad_weights = (
        grad_probs_dense.gather(dim=1, index=topk_idx)
        if grad_probs_dense is not None
        else None
    )
    return grad_x, None, grad_weights, None, None, None


def _dispatch_setup_context(ctx, inputs, output):
    """Save context for dispatch backward."""
    x, topk_idx, _, _, _, _ = inputs
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
    )
    # Gradients: x, handle, num_tokens
    return grad_x, None, None


def _combine_setup_context(ctx, inputs, output):
    """Save context for combine backward."""
    x, dispatch_handle, _num_tokens = inputs
    ctx.dispatch_handle = dispatch_handle
    ctx.num_permuted_tokens = x.shape[0]


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
    moe_expert_capacity_factor: Optional[float] = None,
) -> None:
    """Ensure the global HybridEP buffer is initialized, reinitializing if config changed.

    HybridEP internally scales by EP_group_size for worst-case receive sizing.
    If moe_expert_capacity_factor is set, num_tokens is scaled by that factor to limit
    the receive buffer, enabling token dropping (excess tokens are dropped during dispatch).
    """
    global _buffer

    if fp8_dispatch:
        raise AssertionError("HybridEP FP8 dispatch not yet supported")

    HybridEPBuffer = _require_hybridep()
    max_tokens = num_tokens
    if moe_expert_capacity_factor is not None:
        max_tokens = int(num_tokens * moe_expert_capacity_factor)

    needs_reinit = (
        _buffer is None
        or _buffer.group != group
        or _buffer.config.hidden_dim < hidden_dim
        or _buffer.config.max_num_of_tokens_per_rank < max_tokens
        or _buffer.config.num_of_experts_per_rank < num_local_experts
    )

    if needs_reinit:
        _buffer = HybridEPBuffer(
            group=group,
            hidden_dim=hidden_dim,
            max_num_of_tokens_per_rank=max_tokens,
            num_local_experts=num_local_experts,
            use_fp8=fp8_dispatch,
            num_sms_dispatch_api=_NUM_SMS_DISPATCH,
            num_sms_combine_api=_NUM_SMS_COMBINE,
        )


def dispatch_tokens(
    hidden_states: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    top_scores: torch.Tensor,
    num_local_experts: int,
    num_experts: int,
    group: ProcessGroup,
    score_before_experts: bool = True,
    num_permuted_tokens: Optional[int] = None,
    moe_expert_capacity_factor: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, DispatchState]:
    """Dispatch tokens to experts via HybridEP all-to-all.

    Args:
        hidden_states: [num_tokens, hidden_dim]
        selected_experts_indices: [num_tokens, top_k]
        top_scores: [num_tokens, top_k]
        num_local_experts: Experts on this EP rank
        num_experts: Total experts across all ranks
        group: EP ProcessGroup
        score_before_experts: Apply scores before expert computation
        num_permuted_tokens: Pre-allocated output buffer size for grouped_mm.
            When set, enables CPU-free non-blocking mode for CUDA graphs.

            IMPORTANT: This buffer must be large enough to hold all tokens that may
            be received after the all-to-all dispatch. If the buffer is too small:
            - Current HybridEP: Silently drops excess tokens, sets handle.overflow_flag=True
            - Older HybridEP: Causes illegal memory access (IMA) errors
        moe_expert_capacity_factor: Capacity factor per expert in [0, 1].
            None means no token dropping (default).

    Returns:
        (permuted_hidden, tokens_per_expert, state)
    """
    selected_experts_indices = selected_experts_indices.contiguous()
    top_scores = top_scores.contiguous()

    get_buffer(
        group=group,
        hidden_dim=hidden_states.shape[1],
        num_tokens=hidden_states.shape[0],
        num_local_experts=num_local_experts,
        moe_expert_capacity_factor=moe_expert_capacity_factor,
    )

    hidden, permuted_scores, tokens_per_expert, dispatch_handle = (
        torch.ops.hybridep.dispatch(
            hidden_states, selected_experts_indices, top_scores, num_experts,
            num_permuted_tokens, moe_expert_capacity_factor,
        )
    )

    hidden, permuted_scores = _apply_scores(hidden, permuted_scores, score_before_experts)
    if permuted_scores is not None and permuted_scores.dtype != hidden.dtype:
        permuted_scores = permuted_scores.to(hidden.dtype)

    state = DispatchState(
        handle=dispatch_handle,
        permuted_scores=permuted_scores,
        num_tokens=hidden_states.shape[0],
    )
    return hidden, tokens_per_expert, state


def combine_tokens(hidden_states: torch.Tensor, state: DispatchState) -> torch.Tensor:
    """Combine expert outputs back to original token order.

    Applies deferred scores (if any), then unpermutes via the opaque dispatch handle.
    """
    if state.permuted_scores is not None:
        # In-place to reduce peak memory during recompute.
        hidden_states.mul_(state.permuted_scores.reshape(-1, 1))

    return torch.ops.hybridep.combine(hidden_states, state.handle, state.num_tokens)


__all__ = [
    "dispatch_tokens",
    "combine_tokens",
    "DispatchState",
    "DispatchHandle",
]
