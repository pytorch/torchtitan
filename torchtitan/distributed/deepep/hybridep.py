# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
HybridEP: Expert Parallel Communication for GB200 NVLink72 Systems.

Provides efficient token dispatch/combine for MoE training via TMA-optimized all-to-all.

Configuration (via job_config.parallelism.hybridep):
    capacity_factor: Buffer multiplier (must be >= top_k, upper bound is EP group size)
    num_permuted_tokens: Output buffer size for CPU-free non-blocking mode
    pad_multiple: Alignment for MXFP8 (typically 32)

Reference: https://github.com/deepseek-ai/DeepEP
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.distributed import ProcessGroup


# Module state
_hybrid_ep_cls: Any = None  # Lazily-loaded HybridEPBuffer class
_buffer: Any = None  # Global buffer instance
_buffer_config: dict = {}  # Config for reinit detection
_handle_cache: dict = {}  # cache_id → dispatch handle
_cache_counter: int = 0  # Unique ID generator


@dataclass
class DispatchState:
    """State from dispatch needed for combine.
    
    Attributes:
        cache_id: CPU tensor cache key (avoids GPU-CPU sync).
        num_recv_tokens: Actual tokens received (before padding).
        permuted_scores: Scores for score_before_experts=False mode.
        num_permuted_tokens: Buffer size for grouped_mm. When set, enables
            CPU-free non-blocking mode for CUDA graph compatibility.
    """
    cache_id: torch.Tensor
    num_recv_tokens: int
    permuted_scores: Optional[torch.Tensor] = None
    num_permuted_tokens: Optional[int] = None


def _get_next_cache_id() -> torch.Tensor:
    """Generate unique cache_id as CPU tensor (avoids GPU-CPU sync)."""
    global _cache_counter
    _cache_counter += 1
    return torch.tensor([_cache_counter], dtype=torch.int64, device="cpu")


def _cache_handle(cache_id: torch.Tensor, handle: Any) -> None:
    """Store dispatch handle in cache."""
    _handle_cache[cache_id.item()] = handle


def _get_cached_handle(cache_id_int: int) -> Any:
    """Retrieve dispatch handle from cache."""
    return _handle_cache.get(cache_id_int)


def _pop_cached_handle(cache_id_int: int) -> None:
    """Remove dispatch handle from cache (lenient if missing)."""
    _handle_cache.pop(cache_id_int, None)


def _preprocess_inputs(
    indices: torch.Tensor,
    scores: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize inputs: ensure 2D, contiguous, float32 scores, mask invalid."""
    top_k = indices.shape[1] if indices.dim() == 2 else 1
    
    if indices.dim() != 2:
        indices = indices.view(-1, top_k).contiguous()
        scores = scores.view(-1, top_k).contiguous()
    else:
        indices = indices.contiguous()
        scores = scores.contiguous()

    indices = indices.masked_fill(scores == 0, -1)
    scores = scores.float() if scores.dtype != torch.float32 else scores
    return indices, scores


def _pad_to_multiple(tensor: torch.Tensor, multiple: int) -> torch.Tensor:
    """Pad tensor's first dimension to nearest multiple using F.pad (memory efficient)."""
    size = tensor.shape[0]
    if size % multiple == 0:
        return tensor
    pad_size = multiple - (size % multiple)
    # F.pad pads from last dim backwards; for [N, H], pad format is (0, 0, 0, pad_size)
    return F.pad(tensor, (0, 0, 0, pad_size))


def _apply_scores(
    hidden: torch.Tensor,
    scores: torch.Tensor,
    apply_now: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Apply routing scores to hidden states if apply_now, else defer."""
    if apply_now and scores is not None and scores.numel() > 0:
        return hidden * scores.to(hidden.dtype).reshape(-1, 1), None
    return hidden, scores


def clear_handle_cache() -> None:
    """Clear cached dispatch handles. Call at end of each training step."""
    global _handle_cache, _cache_counter
    _handle_cache.clear()
    _cache_counter = 0


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
            "Install from: https://github.com/deepseek-ai/DeepEP"
        ) from e
    _hybrid_ep_cls = HybridEPBuffer
    return _hybrid_ep_cls


# Custom op registration for torch.compile and SAC compatibility
_lib = torch.library.Library("hybridep", "DEF")

_lib.define(
    "dispatch(Tensor x, Tensor topk_idx, Tensor topk_weights, int num_experts, "
    "int? num_permuted_tokens) -> (Tensor, Tensor, Tensor, Tensor)"
)

_lib.define("combine(Tensor x, Tensor cache_id) -> Tensor")


@torch.library.impl(_lib, "dispatch", "CUDA")
def _dispatch_impl(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    num_permuted_tokens: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """CUDA dispatch: convert sparse routing to dense, call TMA-optimized all-to-all."""
    global _buffer
    if _buffer is None:
        raise RuntimeError("HybridEP buffer not initialized. Call dispatch_tokens() first.")

    num_local_experts = num_experts // _buffer.group_size
    from deep_ep.hybrid_ep_buffer import indices_to_map

    routing_map, probs = indices_to_map(
        topk_idx, topk_weights.float(), x.shape[0], num_experts
    )

    hidden, scores, _, tokens_per_expert, handle = _buffer.dispatch_with_permute(
        hidden=x,
        routing_map=routing_map,
        probs=probs,
        scaling_factor=None,
        num_of_experts_per_rank=num_local_experts,
        pad_multiple=None,
        num_permuted_tokens=num_permuted_tokens,
        non_blocking=num_permuted_tokens is not None,
    )

    cache_id = _get_next_cache_id()
    _cache_handle(cache_id, handle)

    if scores is None:
        scores = torch.empty(0, device=x.device, dtype=torch.float32)
    if tokens_per_expert.device != x.device:
        tokens_per_expert = tokens_per_expert.to(x.device)

    return hidden, scores, tokens_per_expert, cache_id


@torch.library.impl(_lib, "combine", "CUDA")
def _combine_impl(x: torch.Tensor, cache_id: torch.Tensor) -> torch.Tensor:
    """CUDA combine: reverse dispatch permutation via cached handle."""
    global _buffer
    if _buffer is None:
        raise RuntimeError("HybridEP buffer not initialized.")

    handle = _get_cached_handle(cache_id.item())
    if handle is None:
        raise RuntimeError(f"Handle not found for cache_id={cache_id.item()}")

    combined, _ = _buffer.combine_with_unpermute(hidden=x, handle=handle)
    return combined


def _dispatch_backward(ctx, grad_hidden, grad_scores, grad_tpe, grad_cid):
    """Backward: gather gradients via combine."""
    if grad_hidden is None:
        return None, None, None, None, None

    handle = ctx.saved_handle
    if handle is None:
        raise RuntimeError(f"Handle not found in dispatch backward")

    (topk_idx,) = ctx.saved_tensors
    grad_x, grad_s = _buffer.combine_with_unpermute(
        hidden=grad_hidden,
        probs=grad_scores if grad_scores is not None and grad_scores.numel() > 0 else None,
        handle=handle,
        pad_multiple=ctx.pad_multiple,
    )
    _pop_cached_handle(ctx.cache_id_int)
    grad_x = grad_x.to(ctx.input_dtype)

    grad_weights = grad_s.gather(dim=1, index=topk_idx) if grad_s is not None else None
    return grad_x, None, grad_weights, None, None


def _dispatch_setup_context(ctx, inputs, output):
    """Save context for dispatch backward."""
    x, topk_idx, _, _, _ = inputs
    recv_x, _, _, cache_id = output
    ctx.cache_id_int = cache_id.item()
    ctx.input_dtype = x.dtype
    ctx.pad_multiple = None
    ctx.num_permuted_tokens = recv_x.shape[0]
    ctx.save_for_backward(topk_idx)
    ctx.saved_handle = _get_cached_handle(ctx.cache_id_int)


def _combine_backward(ctx, grad_combined):
    """Backward: scatter gradients via dispatch."""
    handle = ctx.saved_handle
    if handle is None:
        raise RuntimeError(f"Handle not found in combine backward")

    grad_x, _, _, _, _ = _buffer.dispatch_with_permute(
        hidden=grad_combined,
        scaling_factor=None,
        handle=handle,
        pad_multiple=ctx.pad_multiple,
        num_permuted_tokens=ctx.num_permuted_tokens,
    )
    _pop_cached_handle(ctx.cache_id_int)
    return grad_x, None


def _combine_setup_context(ctx, inputs, output):
    """Save context for combine backward."""
    x, cache_id = inputs
    ctx.cache_id_int = cache_id.item()
    ctx.pad_multiple = None
    ctx.num_permuted_tokens = x.shape[0]
    ctx.saved_handle = _get_cached_handle(ctx.cache_id_int)


torch.library.register_autograd(
    "hybridep::dispatch", _dispatch_backward, setup_context=_dispatch_setup_context
)
torch.library.register_autograd(
    "hybridep::combine", _combine_backward, setup_context=_combine_setup_context
)


def get_buffer(
    group: ProcessGroup,
    hidden_dim: int,
    num_tokens: int,
    num_local_experts: int,
    top_k: int = 1,
    capacity_factor: float = 1.0,
    num_sms_dispatch: int = 16,
    num_sms_combine: int = 16,
    fp8_dispatch: bool = False,
) -> Any:
    """Get or create HybridEP buffer, reinitializing if config changed.
    
    Buffer is sized as: max_tokens = num_tokens × capacity_factor
    
    Args:
        capacity_factor: Must be >= top_k (lower bound) for balanced routing.
            Upper bound is EP group size. Higher values handle load imbalance.
    """
    global _buffer, _buffer_config

    if fp8_dispatch:
        raise AssertionError("HybridEP FP8 dispatch not yet supported")
        
    HybridEPBuffer = _require_hybridep()
    max_tokens = int(num_tokens * capacity_factor)

    needs_reinit = (
        _buffer is None
        or _buffer.group != group
        or _buffer.config.hidden_dim < hidden_dim
        or _buffer.config.max_num_of_tokens_per_rank < max_tokens
        or _buffer.config.num_of_experts_per_rank < num_local_experts
        or _buffer_config.get("num_sms_dispatch") != num_sms_dispatch
        or _buffer_config.get("num_sms_combine") != num_sms_combine
    )

    if needs_reinit:
        _handle_cache.clear()
        _buffer = HybridEPBuffer(
            group=group,
            hidden_dim=hidden_dim,
            max_num_of_tokens_per_rank=max_tokens,
            num_local_experts=num_local_experts,
            use_fp8=fp8_dispatch,
            num_sms_dispatch_api=num_sms_dispatch,
            num_sms_combine_api=num_sms_combine,
        )
        _buffer_config = {
            "num_sms_dispatch": num_sms_dispatch,
            "num_sms_combine": num_sms_combine,
        }

    return _buffer


def dispatch_tokens(
    hidden_states: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    top_scores: torch.Tensor,
    num_local_experts: int,
    num_experts: int,
    group: ProcessGroup,
    score_before_experts: bool = True,
    num_permuted_tokens: Optional[int] = None,
    capacity_factor: float = 1.0,
    num_sms_dispatch: int = 16,
    num_sms_combine: int = 16,
    pad_multiple: Optional[int] = None,
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
        capacity_factor: Buffer multiplier (>= top_k, <= EP group size)
        pad_multiple: Pad output for MXFP8 alignment (e.g., 32)
    
    Returns:
        (permuted_hidden, tokens_per_expert, state)
    """
    indices, scores = _preprocess_inputs(selected_experts_indices, top_scores)
    top_k = indices.shape[1] if indices.dim() == 2 else 1

    get_buffer(
        group=group,
        hidden_dim=hidden_states.shape[1],
        num_tokens=hidden_states.shape[0],
        num_local_experts=num_local_experts,
        top_k=top_k,
        capacity_factor=capacity_factor,
        num_sms_dispatch=num_sms_dispatch,
        num_sms_combine=num_sms_combine,
    )

    hidden, permuted_scores, tokens_per_expert, cache_id = torch.ops.hybridep.dispatch(
        hidden_states, indices, scores, num_experts, num_permuted_tokens
    )

    hidden, permuted_scores = _apply_scores(hidden, permuted_scores, score_before_experts)
    num_recv = hidden.shape[0]
    num_padded = None

    if pad_multiple is not None and pad_multiple > 1:
        hidden = _pad_to_multiple(hidden, pad_multiple)
        num_padded = hidden.shape[0]
        if permuted_scores is not None:
            permuted_scores = _pad_to_multiple(permuted_scores, pad_multiple)

    state = DispatchState(
        cache_id=cache_id,
        num_recv_tokens=num_recv,
        permuted_scores=permuted_scores,
        num_permuted_tokens=num_padded,
    )
    return hidden, tokens_per_expert, state


def combine_tokens(hidden_states: torch.Tensor, state: DispatchState) -> torch.Tensor:
    """Combine expert outputs back to original token order.
    
    Removes padding if applied, applies deferred scores, then unpermutes.
    """
    # Remove padding
    if state.num_permuted_tokens is not None:
        hidden_states = hidden_states[:state.num_recv_tokens]
        scores = state.permuted_scores[:state.num_recv_tokens] if state.permuted_scores is not None else None
    else:
        scores = state.permuted_scores

    # Apply deferred scores
    if scores is not None:
        hidden_states = hidden_states * scores.to(hidden_states.dtype).reshape(-1, 1)

    return torch.ops.hybridep.combine(hidden_states, state.cache_id)


__all__ = ["dispatch_tokens", "combine_tokens", "clear_handle_cache", "get_buffer", "DispatchState"]
