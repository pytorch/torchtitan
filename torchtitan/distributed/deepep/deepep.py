# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
DeepEP primitives for MoE Expert Parallel.

Provides low-level functions and autograd wrappers for DeepEP communication.
Used by DeepEPExpertParallel in expert_parallel.py.

Supports two backends:
- DeepEP: Optimized for H100 with NVLink Switch
- HybridEP: Optimized for GB200 with NVLink
"""

from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple

import torch
from torch.distributed import ProcessGroup

try:
    from deep_ep import Buffer  # pyrefly: ignore[import-error]
    from deep_ep.utils import EventHandle, EventOverlap  # pyrefly: ignore[import-error]
except ImportError as e:
    raise ImportError(
        "DeepEP is required for this module. "
        "Install from: https://github.com/deepseek-ai/deepep"
    ) from e

# We keep HybridEP optional: typed as Any to avoid importing HybridEP unless the
# user selects that backend (import happens lazily in _require_hybridep()).
HybridEPBufferType = Any


# ============================================================================
# Backend Configuration
# ============================================================================

_backend_mode: Literal["deepep", "hybridep"] = "deepep"
_hybridep_num_sms_dispatch_api: int = 16
_hybridep_num_sms_combine_api: int = 16
_hybrid_ep_cls: HybridEPBufferType | None = None


def _reset_backend_state() -> None:
    """Clear buffers/handles when switching backend to avoid stale state."""
    global _buffer, _hybrid_ep_buffer, _handle_cache, _cache_counter
    _buffer = None
    _hybrid_ep_buffer = None
    _handle_cache = {}
    _cache_counter = 0


def _require_hybridep():
    """Import HybridEP lazily and fail fast if the dependency is missing."""
    global _hybrid_ep_cls
    if _hybrid_ep_cls is not None:
        return _hybrid_ep_cls
    try:
        from deep_ep import HybridEPBuffer  # pyrefly: ignore [missing-import]
    except ImportError as e:
        raise ImportError(
            "HybridEP backend selected but HybridEP is not installed. "
            "Install from: https://github.com/deepseek-ai/DeepEP "
            "(checkout to hybrid_ep branch)."
        ) from e
    _hybrid_ep_cls = HybridEPBuffer
    return _hybrid_ep_cls


def configure_deepep_backend(
    backend: Literal["deepep", "hybridep"] = "deepep",
    num_sms_dispatch_api: int = 16,
    num_sms_combine_api: int = 16,
) -> None:
    """Configure backend mode. Call before first forward pass."""
    global _backend_mode, _hybridep_num_sms_dispatch_api, _hybridep_num_sms_combine_api
    if backend != _backend_mode:
        _reset_backend_state()
    if backend == "hybridep":
        _require_hybridep()
    _backend_mode = backend
    _hybridep_num_sms_dispatch_api = num_sms_dispatch_api
    _hybridep_num_sms_combine_api = num_sms_combine_api


# Global buffers (lazily initialized, recreated if group changes)
_buffer: Buffer | None = None
_hybrid_ep_buffer: HybridEPBufferType | None = None

# Handle cache for SAC: maps cache_id -> dispatch handle
_handle_cache: dict = {}
_cache_counter: int = 0


def _get_next_cache_id() -> torch.Tensor:
    """Generate a unique cache_id tensor on CPU to avoid GPU-CPU sync."""
    global _cache_counter
    _cache_counter += 1
    return torch.tensor([_cache_counter], dtype=torch.int64, device="cpu")


# ============================================================================
# Custom Op Registration for SAC Integration
# ============================================================================

_lib = torch.library.Library("deepep", "DEF")

# dispatch returns: (recv_x, recv_indices, recv_scores, num_recv_per_expert, cache_id)
_lib.define(
    "dispatch(Tensor x, Tensor topk_idx, Tensor topk_weights, "
    "Tensor num_tokens_per_rank, Tensor num_tokens_per_rdma_rank, "
    "Tensor is_token_in_rank, Tensor num_tokens_per_expert) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor)"
)

# combine returns: combined_x
_lib.define("combine(Tensor x, Tensor cache_id) -> Tensor")

# HybridEP dispatch returns: (recv_x, recv_scores, tokens_per_expert, cache_id)
# num_permuted_tokens: optional, if provided enables non_blocking mode (avoids D2H sync)
_lib.define(
    "hybridep_dispatch(Tensor x, Tensor topk_idx, Tensor topk_weights, int num_experts, int? num_permuted_tokens) "
    "-> (Tensor, Tensor, Tensor, Tensor)"
)

# HybridEP combine returns: combined_x
_lib.define("hybridep_combine(Tensor x, Tensor cache_id) -> Tensor")


@torch.library.impl(_lib, "dispatch", "CUDA")
def _dispatch_op_impl(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_tokens_per_rank: torch.Tensor,
    num_tokens_per_rdma_rank: torch.Tensor,
    is_token_in_rank: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute DeepEP dispatch."""
    global _buffer

    buffer = _buffer
    assert buffer is not None, "Buffer must be initialized before dispatch"

    previous_event = _create_event_if_async(True)

    (
        recv_x,
        recv_indices,
        recv_scores,
        num_recv_list,
        handle,
        after_event,
    ) = buffer.dispatch(
        x=x,
        topk_idx=topk_idx,
        topk_weights=topk_weights.to(torch.float32),
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        previous_event=previous_event,
        async_finish=True,
        allocate_on_comm_stream=True,
    )

    _sync_stream_if_async(True, after_event)

    cache_id = _get_next_cache_id()
    _handle_cache[cache_id.item()] = handle

    num_recv_tensor = torch.tensor(num_recv_list, dtype=torch.int32, device="cpu")
    return recv_x, recv_indices, recv_scores, num_recv_tensor, cache_id


@torch.library.impl(_lib, "hybridep_dispatch", "CUDA")
def _hybridep_dispatch_op_impl(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    num_permuted_tokens: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute HybridEP dispatch with permute."""
    global _hybrid_ep_buffer

    buffer = _hybrid_ep_buffer
    assert buffer is not None, "HybridEP buffer must be initialized before dispatch"

    num_local_experts = num_experts // buffer.group_size
    num_tokens = x.shape[0]

    from deep_ep.hybrid_ep_buffer import indices_to_map  # pyrefly: ignore [missing-import]

    routing_map, probs = indices_to_map(
        topk_idx,
        topk_weights.float(),
        num_tokens,
        num_experts,
    )

    (
        hidden_states,
        dispatched_expert_scores,
        _,
        tokens_per_expert,
        handle,
    ) = buffer.dispatch_with_permute(
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
    _handle_cache[cache_id.item()] = handle

    if dispatched_expert_scores is None:
        dispatched_expert_scores = torch.empty(0, device=x.device, dtype=torch.float32)

    if tokens_per_expert.device != x.device:
        tokens_per_expert = tokens_per_expert.to(x.device)

    return hidden_states, dispatched_expert_scores, tokens_per_expert, cache_id


@torch.library.impl(_lib, "combine", "CUDA")
def _combine_op_impl(x: torch.Tensor, cache_id: torch.Tensor) -> torch.Tensor:
    """Execute DeepEP combine."""
    global _buffer

    buffer = _buffer
    assert buffer is not None, "Buffer must be initialized before combine"

    handle = _handle_cache.get(cache_id.item())
    assert handle is not None, f"Handle not found for cache_id={cache_id.item()}"

    previous_event = _create_event_if_async(True)

    combined, _, after_event = buffer.combine(
        x=x,
        handle=handle,
        previous_event=previous_event,
        async_finish=True,
        allocate_on_comm_stream=True,
    )

    _sync_stream_if_async(True, after_event)

    return combined


@torch.library.impl(_lib, "hybridep_combine", "CUDA")
def _hybridep_combine_op_impl(x: torch.Tensor, cache_id: torch.Tensor) -> torch.Tensor:
    """Execute HybridEP combine with unpermute."""
    global _hybrid_ep_buffer

    buffer = _hybrid_ep_buffer
    assert buffer is not None, "HybridEP buffer must be initialized before combine"

    handle = _handle_cache.get(cache_id.item())
    assert handle is not None, f"Handle not found for cache_id={cache_id.item()}"

    combined_token, _combined_probs = buffer.combine_with_unpermute(
        hidden=x,
        handle=handle,
    )

    return combined_token


def _dispatch_backward(
    ctx, grad_recv_x, grad_recv_indices, grad_recv_scores, grad_num_recv, grad_cache_id
):
    """Backward for dispatch: performs combine on gradients."""
    global _buffer

    if grad_recv_x is None:
        return None, None, None, None, None, None, None

    # Use saved handle from setup_context (handles activation checkpointing recompute)
    handle = ctx.saved_handle
    assert handle is not None, f"Handle not found for cache_id={ctx.cache_id_int}"

    previous_event = _create_event_if_async(True)

    grad_x, grad_scores, after_event = _buffer.combine(
        x=grad_recv_x,
        handle=handle,
        topk_weights=grad_recv_scores.float() if grad_recv_scores is not None else None,
        previous_event=previous_event,
        async_finish=True,
        allocate_on_comm_stream=True,
    )

    _sync_stream_if_async(True, after_event)
    _handle_cache.pop(ctx.cache_id_int, None)

    grad_x = grad_x.to(ctx.input_dtype)
    grad_topk_weights = (
        grad_scores.to(ctx.input_dtype) if grad_scores is not None else None
    )

    return grad_x, None, grad_topk_weights, None, None, None, None


def _dispatch_setup_context(ctx, inputs, output):
    x, topk_idx, topk_weights, *_ = inputs
    recv_x, recv_indices, recv_scores, num_recv, cache_id = output
    ctx.cache_id_int = cache_id.item()
    ctx.input_dtype = x.dtype
    # Save handle now to survive activation checkpointing recompute
    ctx.saved_handle = _handle_cache.get(ctx.cache_id_int)


def _combine_backward(ctx, grad_combined):
    """Backward for combine: performs dispatch on gradients."""
    handle = ctx.saved_handle
    previous_event = _create_event_if_async(True)

    grad_x, _, _, _, _, after_event = _buffer.dispatch(
        x=grad_combined,
        topk_idx=None,
        topk_weights=None,
        num_tokens_per_rank=None,
        num_tokens_per_rdma_rank=None,
        is_token_in_rank=None,
        num_tokens_per_expert=None,
        handle=handle,
        previous_event=previous_event,
        async_finish=True,
        allocate_on_comm_stream=True,
    )

    _sync_stream_if_async(True, after_event)
    _handle_cache.pop(ctx.cache_id_int, None)

    return grad_x, None


def _combine_setup_context(ctx, inputs, output):
    x, cache_id = inputs
    ctx.cache_id_int = cache_id.item()
    ctx.saved_handle = _handle_cache.get(ctx.cache_id_int)


torch.library.register_autograd(
    "deepep::dispatch", _dispatch_backward, setup_context=_dispatch_setup_context
)
torch.library.register_autograd(
    "deepep::combine", _combine_backward, setup_context=_combine_setup_context
)


# ============================================================================
# HybridEP Autograd Registration
# ============================================================================


def _hybridep_dispatch_backward(ctx, grad_recv_x, grad_recv_scores, grad_tokens_per_expert, grad_cache_id):
    """Backward for HybridEP dispatch: performs combine on gradients."""
    if grad_recv_x is None:
        return None, None, None, None, None

    # Use saved handle from setup_context (handles activation checkpointing recompute)
    handle = ctx.saved_handle
    assert handle is not None, f"Handle not found for cache_id={ctx.cache_id_int}"

    (topk_idx,) = ctx.saved_tensors

    grad_x, grad_scores = _hybrid_ep_buffer.combine_with_unpermute(
        hidden=grad_recv_x,
        probs=grad_recv_scores if grad_recv_scores is not None and grad_recv_scores.numel() > 0 else None,
        handle=handle,
        pad_multiple=ctx.pad_multiple,
    )

    _handle_cache.pop(ctx.cache_id_int, None)

    grad_x = grad_x.to(ctx.input_dtype)

    # Transform grad_scores from [N, num_experts] to [N, top_k] using topk_idx
    # This matches the forward topk_weights shape
    if grad_scores is not None:
        grad_topk_weights = grad_scores.gather(dim=1, index=topk_idx)
    else:
        grad_topk_weights = None

    return grad_x, None, grad_topk_weights, None, None


def _hybridep_dispatch_setup_context(ctx, inputs, output):
    x, topk_idx, _, _, _ = inputs
    recv_x, _, _, cache_id = output
    ctx.cache_id_int = cache_id.item()
    ctx.input_dtype = x.dtype
    ctx.pad_multiple = None
    ctx.num_permuted_tokens = recv_x.shape[0]
    # Save topk_idx for gradient shape transformation in backward
    ctx.save_for_backward(topk_idx)
    # Save handle now to survive activation checkpointing recompute
    ctx.saved_handle = _handle_cache.get(ctx.cache_id_int)


def _hybridep_combine_backward(ctx, grad_combined):
    """Backward for HybridEP combine: performs dispatch on gradients."""
    handle = ctx.saved_handle
    assert handle is not None, f"Handle not found for cache_id={ctx.cache_id_int}"

    grad_x, _, _, _, _ = _hybrid_ep_buffer.dispatch_with_permute(
        hidden=grad_combined,
        scaling_factor=None,
        handle=handle,
        pad_multiple=ctx.pad_multiple,
        num_permuted_tokens=ctx.num_permuted_tokens,
    )

    _handle_cache.pop(ctx.cache_id_int, None)

    return grad_x, None


def _hybridep_combine_setup_context(ctx, inputs, output):
    x, cache_id = inputs
    ctx.cache_id_int = cache_id.item()
    ctx.saved_handle = _handle_cache.get(ctx.cache_id_int)
    ctx.pad_multiple = None
    ctx.num_permuted_tokens = x.shape[0]


torch.library.register_autograd(
    "deepep::hybridep_dispatch",
    _hybridep_dispatch_backward,
    setup_context=_hybridep_dispatch_setup_context,
)
torch.library.register_autograd(
    "deepep::hybridep_combine",
    _hybridep_combine_backward,
    setup_context=_hybridep_combine_setup_context,
)


def _create_event_if_async(async_finish: bool):
    """Create EventOverlap handle if async mode is enabled."""
    return EventOverlap(EventHandle()) if async_finish else None


def _sync_stream_if_async(async_finish: bool, after_event):
    """Synchronize current stream with communication stream if async mode is enabled."""
    if async_finish and after_event is not None:
        after_event.current_stream_wait()


def get_hidden_bytes(x: torch.Tensor) -> int:
    """Calculate hidden bytes for buffer sizing."""
    return x.size(1) * max(x.element_size(), 2)


def get_buffer(group: ProcessGroup, hidden_bytes: int):
    """Get or create a buffer for all-to-all communication."""
    global _buffer

    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes
        )

    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)

    return _buffer


def get_hybrid_ep_buffer(
    group: ProcessGroup,
    hidden_dim: int,
    seq_len: int,
    num_local_experts: int,
    fp8_dispatch: bool = False,
) -> HybridEPBufferType:
    """Get or create HybridEP buffer. Reinitializes if config needs to grow."""
    global _hybrid_ep_buffer

    assert not fp8_dispatch, "HybridEP does not support FP8 dispatch yet"
    HybridEPBuffer = _require_hybridep()

    if (
        _hybrid_ep_buffer is None
        or _hybrid_ep_buffer.group != group
        or _hybrid_ep_buffer.config.hidden_dim < hidden_dim
        or _hybrid_ep_buffer.config.max_num_of_tokens_per_rank < seq_len
        or _hybrid_ep_buffer.config.num_of_experts_per_rank < num_local_experts
    ):
        _hybrid_ep_buffer = HybridEPBuffer(
            group=group,
            hidden_dim=hidden_dim,
            max_num_of_tokens_per_rank=seq_len,
            num_local_experts=num_local_experts,
            use_fp8=fp8_dispatch,
            num_sms_dispatch_api=_hybridep_num_sms_dispatch_api,
            num_sms_combine_api=_hybridep_num_sms_combine_api,
        )

    return _hybrid_ep_buffer


def _indices_to_multihot(
    indices: torch.Tensor, scores: torch.Tensor, num_local_experts: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert topk indices to multihot format for permutation."""
    batch_size = indices.shape[0]
    multihot_routing_map = torch.zeros(
        (batch_size, num_local_experts), dtype=torch.long, device=indices.device
    )
    multihot_scores = torch.zeros(
        (batch_size, num_local_experts), dtype=scores.dtype, device=indices.device
    )

    mask = indices != -1
    valid_indices = indices[mask]
    row_indices = torch.arange(batch_size, device=indices.device).repeat_interleave(
        mask.sum(dim=1)
    )
    multihot_routing_map[row_indices, valid_indices] = 1
    multihot_scores[row_indices, valid_indices] = scores[mask]

    return multihot_routing_map.bool(), multihot_scores


def _permute_tokens(
    tokens: torch.Tensor,
    routing_map: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Permute tokens by expert for grouped_mm.

    Returns:
        (permuted_tokens, permuted_scores, sorted_indices)
    """
    num_tokens = tokens.shape[0]
    num_experts = routing_map.shape[1]

    routing_map_t = routing_map.bool().T.contiguous()
    token_indices = torch.arange(num_tokens, device=routing_map.device)
    token_indices = token_indices.unsqueeze(0).expand(num_experts, -1)
    sorted_indices = token_indices.masked_select(routing_map_t)
    sorted_tokens = tokens.index_select(0, sorted_indices)

    if scores is not None:
        sorted_scores = scores.T.contiguous().masked_select(routing_map_t)
    else:
        sorted_scores = None

    return sorted_tokens, sorted_scores, sorted_indices


def _unpermute_tokens(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Reverse permutation applied by _permute_tokens."""
    hidden = permuted_tokens.shape[1]
    output_tokens = torch.zeros(
        (num_tokens, hidden), dtype=permuted_tokens.dtype, device=permuted_tokens.device
    )
    output_tokens.scatter_add_(
        0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens
    )
    return output_tokens


@dataclass
class DispatchState:
    """State from dispatch needed for combine."""

    cache_id: torch.Tensor  # CPU tensor used to retrieve cached handle
    sorted_indices: Optional[torch.Tensor]  # None for HybridEP
    num_recv_tokens: int
    permuted_scores: Optional[torch.Tensor] = None


def _preprocess_dispatch_inputs(
    selected_experts_indices: torch.Tensor,
    top_scores: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize inputs for dispatch: ensure contiguous, proper shape, and float32 scores."""
    router_topk = (
        selected_experts_indices.shape[1] if selected_experts_indices.dim() == 2 else 1
    )
    if selected_experts_indices.dim() != 2:
        selected_experts_indices = selected_experts_indices.view(-1, router_topk).contiguous()
        top_scores = top_scores.view(-1, router_topk).contiguous()
    else:
        selected_experts_indices = selected_experts_indices.contiguous()
        top_scores = top_scores.contiguous()

    # Mask out zero-score tokens and ensure float32
    selected_experts_indices = selected_experts_indices.masked_fill(top_scores == 0, -1)
    if top_scores.dtype != torch.float32:
        top_scores = top_scores.float()

    return selected_experts_indices, top_scores


def _apply_scores_to_hidden(
    hidden_states: torch.Tensor,
    scores: torch.Tensor,
    score_before_experts: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Apply routing scores to hidden states if score_before_experts is True."""
    if score_before_experts and scores is not None and scores.numel() > 0:
        hidden_states = hidden_states * scores.to(hidden_states.dtype).reshape(-1, 1)
        return hidden_states, None
    return hidden_states, scores


def dispatch_tokens(
    hidden_states: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    top_scores: torch.Tensor,
    num_local_experts: int,
    num_experts: int,
    group: ProcessGroup,
    score_before_experts: bool = True,
    num_permuted_tokens: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, DispatchState]:
    """Dispatch tokens to experts via DeepEP or HybridEP.

    Args:
        hidden_states: Input tokens [num_tokens, hidden_dim]
        selected_experts_indices: Expert indices [num_tokens, top_k]
        top_scores: Routing scores [num_tokens, top_k]
        num_local_experts: Number of experts on this rank
        num_experts: Total number of experts across all ranks
        group: EP process group
        score_before_experts: If True, apply routing scores before expert computation
        num_permuted_tokens: For HybridEP non_blocking mode (avoids D2H sync)

    Returns:
        (hidden_states, tokens_per_expert, dispatch_state)
    """
    global _backend_mode

    selected_experts_indices, top_scores = _preprocess_dispatch_inputs(
        selected_experts_indices, top_scores
    )

    if _backend_mode == "hybridep":
        # HybridEP path (GB200 with NVLink72)
        get_hybrid_ep_buffer(
            group=group,
            hidden_dim=hidden_states.shape[1],
            seq_len=hidden_states.shape[0],
            num_local_experts=num_local_experts,
            fp8_dispatch=False,
        )

        (
            hidden_states,
            dispatched_expert_scores,
            tokens_per_expert,
            cache_id,
        ) = torch.ops.deepep.hybridep_dispatch(
            hidden_states,
            selected_experts_indices,
            top_scores,  # Always send scores for both score_before_experts modes
            num_experts,
            num_permuted_tokens,
        )

        # Apply scores to hidden if score_before_experts=True
        # Otherwise, save scores for application after expert computation (before combine)
        hidden_states, permuted_scores = _apply_scores_to_hidden(
            hidden_states, dispatched_expert_scores, score_before_experts
        )

        state = DispatchState(
            cache_id=cache_id,
            sorted_indices=None,
            num_recv_tokens=hidden_states.shape[0],
            permuted_scores=permuted_scores,  # Save scores for combine when score_before_experts=False
        )
        return hidden_states, tokens_per_expert, state

    # DeepEP path
    buffer = get_buffer(group, get_hidden_bytes(hidden_states))

    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert_dispatch,
        is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(
        topk_idx=selected_experts_indices, num_experts=num_experts
    )

    hidden_states, dispatched_indices, dispatched_expert_scores, _, cache_id = (
        torch.ops.deepep.dispatch(
            hidden_states,
            selected_experts_indices,
            top_scores,
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            is_token_in_rank,
            num_tokens_per_expert_dispatch,
        )
    )

    routing_map, scores_multihot = _indices_to_multihot(
        dispatched_indices, dispatched_expert_scores, num_local_experts
    )
    num_recv_tokens = hidden_states.shape[0]

    hidden_states, permuted_scores, sorted_indices = _permute_tokens(
        hidden_states, routing_map, scores=scores_multihot
    )
    tokens_per_expert = routing_map.sum(dim=0).to(torch.int32).to(hidden_states.device)

    hidden_states, permuted_scores = _apply_scores_to_hidden(
        hidden_states, permuted_scores, score_before_experts
    )

    state = DispatchState(
        cache_id=cache_id,
        sorted_indices=sorted_indices,
        num_recv_tokens=num_recv_tokens,
        permuted_scores=permuted_scores,
    )
    return hidden_states, tokens_per_expert, state


def combine_tokens(
    hidden_states: torch.Tensor,
    state: DispatchState,
) -> torch.Tensor:
    """Combine tokens from experts via DeepEP or HybridEP."""
    global _backend_mode

    if _backend_mode == "hybridep":
        if state.permuted_scores is not None:
            hidden_states = hidden_states * state.permuted_scores.to(
                hidden_states.dtype
            ).reshape(-1, 1)
        return torch.ops.deepep.hybridep_combine(hidden_states, state.cache_id)

    # DeepEP path
    if state.permuted_scores is not None:
        hidden_states = hidden_states * state.permuted_scores.to(
            hidden_states.dtype
        ).reshape(-1, 1)

    if state.sorted_indices is not None:
        hidden_states = _unpermute_tokens(
            hidden_states, state.sorted_indices, state.num_recv_tokens
        )

    return torch.ops.deepep.combine(hidden_states, state.cache_id)
