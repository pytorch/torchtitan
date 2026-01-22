# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
HybridEP: Expert Parallel Communication Backend for GB200 NVLink72 Systems.

This module implements the HybridEP backend for MoE (Mixture of Experts) training,
optimized for NVIDIA GB200 systems with NVLink72 connectivity.

Reference:
    https://github.com/deepseek-ai/DeepEP/blob/hybrid-ep/Hybrid-EP_Implementation.md
"""

import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from torch.distributed import ProcessGroup


_num_sms_dispatch: int = int(os.environ.get("HYBRIDEP_NUM_SMS_DISPATCH", "16"))
"""Number of SMs dedicated to the dispatch kernel. Higher values may improve
throughput but reduce SMs available for compute overlap."""

_num_sms_combine: int = int(os.environ.get("HYBRIDEP_NUM_SMS_COMBINE", "16"))
"""Number of SMs dedicated to the combine kernel."""

_hybrid_ep_cls: Any = None
"""Lazily-loaded HybridEPBuffer class from deep_ep library."""

_buffer: Any = None
"""Global HybridEPBuffer instance. Lazily initialized on first dispatch."""

_handle_cache: dict = {}
"""Cache mapping cache_id (int) → dispatch handle.

The handle contains routing metadata needed by combine_with_unpermute to reverse
the dispatch operation. We cache handles to support:
1. Combining after expert computation (normal forward)
2. Activation checkpointing recompute scenarios

Note: Handles are stored on ctx.saved_handle in setup_context to survive AC recompute.
The cache is a backup and may be cleared on buffer reinit.
"""

_cache_counter: int = 0
"""Monotonically increasing counter for generating unique cache IDs.

We use CPU tensors for cache_id to avoid GPU-CPU synchronization when 
retrieving the ID value in setup_context.
"""


@dataclass
class DispatchState:
    """State from dispatch_tokens needed for combine_tokens.
    
    This dataclass captures all information needed to reverse the dispatch
    operation and combine expert outputs back to the original token ordering.
    
    Attributes:
        cache_id: CPU tensor containing the handle cache key. Using CPU tensor
            avoids GPU-CPU sync when accessing the value.
        num_recv_tokens: Number of tokens received after dispatch. Used for
            buffer sizing validation.
        permuted_scores: Routing scores in permuted order, used when
            score_before_experts=False to apply scores after expert computation.
    """
    cache_id: torch.Tensor
    num_recv_tokens: int
    permuted_scores: Optional[torch.Tensor] = None


def _get_next_cache_id() -> torch.Tensor:
    """Generate a unique cache_id as a CPU tensor.
    
    Using a CPU tensor avoids GPU-CPU synchronization when we need to read
    the cache_id value in autograd's setup_context function.
    
    Returns:
        CPU tensor containing a unique integer ID.
    """
    global _cache_counter
    _cache_counter += 1
    return torch.tensor([_cache_counter], dtype=torch.int64, device="cpu")


def _cache_handle(cache_id: torch.Tensor, handle: Any) -> None:
    """Store a dispatch handle in the cache.
    
    Args:
        cache_id: CPU tensor from _get_next_cache_id()
        handle: Dispatch handle from buffer.dispatch_with_permute()
    """
    _handle_cache[cache_id.item()] = handle


def _get_cached_handle(cache_id_int: int, context: str = "") -> Any:
    """Retrieve a dispatch handle from the cache.
    
    Args:
        cache_id_int: Integer cache key
        context: Description for debugging (unused in production)
        
    Returns:
        The cached handle, or None if not found.
    """
    return _handle_cache.get(cache_id_int)


def _pop_cached_handle(cache_id_int: int, context: str = "") -> None:
    """Remove a dispatch handle from the cache.
    
    This is called after backward to free memory. The removal is lenient
    (no error if key missing) because activation checkpointing may create
    new cache_ids during recompute, and the original may not exist.
    
    Args:
        cache_id_int: Integer cache key
        context: Description for debugging (unused in production)
    """
    _handle_cache.pop(cache_id_int, None)


def _preprocess_dispatch_inputs(
    selected_experts_indices: torch.Tensor,
    top_scores: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize and validate inputs for dispatch.
    
    Ensures inputs are:
    - 2D with shape [num_tokens, top_k]
    - Contiguous in memory
    - Scores are float32 (required by HybridEP kernels)
    - Invalid selections (score=0) are masked to index -1
    
    Args:
        selected_experts_indices: Expert indices, shape [N] or [N, K]
        top_scores: Routing scores, same shape as indices
        
    Returns:
        Tuple of (indices, scores) with shape [N, K] and dtype constraints.
    """
    # Determine top_k and reshape to 2D if needed
    top_k = selected_experts_indices.shape[1] if selected_experts_indices.dim() == 2 else 1
    
    if selected_experts_indices.dim() != 2:
        selected_experts_indices = selected_experts_indices.view(-1, top_k).contiguous()
        top_scores = top_scores.view(-1, top_k).contiguous()
    else:
        selected_experts_indices = selected_experts_indices.contiguous()
        top_scores = top_scores.contiguous()

    # Mask out zero-score tokens (invalid selections) with -1
    # This tells the dispatch kernel to skip these assignments
    selected_experts_indices = selected_experts_indices.masked_fill(top_scores == 0, -1)
    
    # HybridEP kernels require float32 scores
    if top_scores.dtype != torch.float32:
        top_scores = top_scores.float()

    return selected_experts_indices, top_scores


def _apply_scores_to_hidden(
    hidden_states: torch.Tensor,
    scores: torch.Tensor,
    score_before_experts: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Optionally apply routing scores to hidden states.
    
    MoE combines expert outputs weighted by routing scores. This can be done
    either before expert computation (score_before_experts=True) or after
    (score_before_experts=False). Both are mathematically equivalent.
    
    Args:
        hidden_states: Token representations [M, H]
        scores: Routing scores [M] or [M, 1]
        score_before_experts: If True, multiply scores into hidden now
        
    Returns:
        Tuple of (hidden_states, scores_for_later):
        - If score_before_experts=True: (scaled_hidden, None)
        - If score_before_experts=False: (hidden, scores) to apply in combine
    """
    if score_before_experts and scores is not None and scores.numel() > 0:
        hidden_states = hidden_states * scores.to(hidden_states.dtype).reshape(-1, 1)
        return hidden_states, None
    return hidden_states, scores


def configure(
    num_sms_dispatch: Optional[int] = None,
    num_sms_combine: Optional[int] = None,
) -> None:
    """Configure HybridEP kernel parameters.
    
    Call this before the first forward pass to override environment variable
    defaults. These settings affect GPU resource allocation for the dispatch
    and combine kernels.
    
    Args:
        num_sms_dispatch: Number of SMs for dispatch kernel (default: 16).
            Higher values may improve dispatch throughput at the cost of
            SMs available for overlapped compute.
        num_sms_combine: Number of SMs for combine kernel (default: 16).
    
    Note:
        Can also be configured via environment variables:
        - HYBRIDEP_NUM_SMS_DISPATCH
        - HYBRIDEP_NUM_SMS_COMBINE
    """
    global _num_sms_dispatch, _num_sms_combine
    if num_sms_dispatch is not None:
        _num_sms_dispatch = num_sms_dispatch
    if num_sms_combine is not None:
        _num_sms_combine = num_sms_combine


def _require_hybridep() -> Any:
    """Lazily import and return the HybridEPBuffer class.
    
    HybridEP is an optional dependency. This function imports it on first use
    and raises a helpful error if not installed.
    
    Returns:
        The HybridEPBuffer class from deep_ep library.
        
    Raises:
        ImportError: If deep_ep with HybridEP support is not installed.
    """
    global _hybrid_ep_cls
    if _hybrid_ep_cls is not None:
        return _hybrid_ep_cls
        
    try:
        from deep_ep import HybridEPBuffer
    except ImportError as e:
        raise ImportError(
            "HybridEP backend requires the deep_ep library with HybridEP support. "
            "Install from: https://github.com/deepseek-ai/DeepEP "
            "(checkout the hybrid_ep branch)"
        ) from e
        
    _hybrid_ep_cls = HybridEPBuffer
    return _hybrid_ep_cls


# ============================================================================
# Custom Op Registration
# ============================================================================
# We register HybridEP operations as custom ops in the "hybridep" namespace.
# This enables:
# 1. Proper interaction with torch.compile
# 2. Selective Activation Checkpointing (SAC) support
# 3. Clean autograd integration

_lib = torch.library.Library("hybridep", "DEF")

_lib.define(
    "dispatch(Tensor x, Tensor topk_idx, Tensor topk_weights, int num_experts, int? num_permuted_tokens) "
    "-> (Tensor, Tensor, Tensor, Tensor)"
)
"""hybridep::dispatch - Send tokens to their assigned experts.

Args:
    x: Hidden states [num_tokens, hidden_dim]
    topk_idx: Expert indices [num_tokens, top_k]
    topk_weights: Routing scores [num_tokens, top_k]  
    num_experts: Total experts across all EP ranks
    num_permuted_tokens: Pre-computed output size for non-blocking mode (optional)

Returns:
    Tuple of (permuted_hidden, permuted_scores, tokens_per_expert, cache_id)
"""

_lib.define("combine(Tensor x, Tensor cache_id) -> Tensor")
"""hybridep::combine - Gather expert outputs back to original token order.

Args:
    x: Expert output [num_permuted_tokens, hidden_dim]
    cache_id: Handle cache key from dispatch

Returns:
    Combined hidden states [num_tokens, hidden_dim]
"""


@torch.library.impl(_lib, "dispatch", "CUDA")
def _dispatch_impl(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    num_permuted_tokens: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """CUDA implementation of hybridep::dispatch.
    
    Converts sparse routing indices to dense multi-hot format, then calls
    HybridEP's dispatch_with_permute which uses TMA for efficient all-to-all.
    """
    global _buffer

    if _buffer is None:
        raise RuntimeError(
            "HybridEP buffer not initialized. Call get_buffer() or dispatch_tokens() first."
        )

    num_local_experts = num_experts // _buffer.group_size
    num_tokens = x.shape[0]

    # Convert sparse indices [N, K] to dense routing map [N, num_experts]
    # This format is required by HybridEP's TMA-optimized kernels
    from deep_ep.hybrid_ep_buffer import indices_to_map

    routing_map, probs = indices_to_map(
        topk_idx,
        topk_weights.float(),
        num_tokens,
        num_experts,
    )

    # Execute the dispatch with fused permutation
    (
        hidden_states,
        dispatched_expert_scores,
        _,  # recv_counts (unused)
        tokens_per_expert,
        handle,
    ) = _buffer.dispatch_with_permute(
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

    if dispatched_expert_scores is None:
        dispatched_expert_scores = torch.empty(0, device=x.device, dtype=torch.float32)

    if tokens_per_expert.device != x.device:
        tokens_per_expert = tokens_per_expert.to(x.device)

    return hidden_states, dispatched_expert_scores, tokens_per_expert, cache_id


@torch.library.impl(_lib, "combine", "CUDA")
def _combine_impl(x: torch.Tensor, cache_id: torch.Tensor) -> torch.Tensor:
    """CUDA implementation of hybridep::combine.
    
    Uses the cached handle to reverse the dispatch permutation and 
    all-to-all, returning tokens to their original order.
    """
    global _buffer

    if _buffer is None:
        raise RuntimeError("HybridEP buffer not initialized.")

    handle = _get_cached_handle(cache_id.item())
    if handle is None:
        raise RuntimeError(
            f"Dispatch handle not found for cache_id={cache_id.item()}. "
            "Ensure dispatch was called before combine."
        )

    combined_token, _ = _buffer.combine_with_unpermute(
        hidden=x,
        handle=handle,
    )

    return combined_token


def _dispatch_backward(
    ctx: Any,
    grad_recv_x: torch.Tensor,
    grad_recv_scores: torch.Tensor,
    grad_tokens_per_expert: torch.Tensor,
    grad_cache_id: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], ...]:
    """Backward pass for dispatch: route gradients back via combine.
    
    The backward of dispatch is mathematically a combine operation - we need
    to gather gradients from experts back to the original tokens.
    """
    if grad_recv_x is None:
        return None, None, None, None, None

    handle = ctx.saved_handle
    if handle is None:
        raise RuntimeError(f"Handle not found for cache_id={ctx.cache_id_int} in backward")

    (topk_idx,) = ctx.saved_tensors

    # Use combine_with_unpermute to gather gradients
    grad_x, grad_scores = _buffer.combine_with_unpermute(
        hidden=grad_recv_x,
        probs=grad_recv_scores if grad_recv_scores is not None and grad_recv_scores.numel() > 0 else None,
        handle=handle,
        pad_multiple=ctx.pad_multiple,
    )

    _pop_cached_handle(ctx.cache_id_int)

    grad_x = grad_x.to(ctx.input_dtype)

    # Transform grad_scores from dense [N, num_experts] to sparse [N, top_k]
    # by gathering using the original topk_idx
    if grad_scores is not None:
        grad_topk_weights = grad_scores.gather(dim=1, index=topk_idx)
    else:
        grad_topk_weights = None

    return grad_x, None, grad_topk_weights, None, None


def _dispatch_setup_context(ctx: Any, inputs: Tuple, output: Tuple) -> None:
    """Save context for dispatch backward pass."""
    x, topk_idx, _, _, _ = inputs
    recv_x, _, _, cache_id = output
    
    ctx.cache_id_int = cache_id.item()
    ctx.input_dtype = x.dtype
    ctx.pad_multiple = None
    ctx.num_permuted_tokens = recv_x.shape[0]
    
    # Save topk_idx for gradient shape transformation in backward
    ctx.save_for_backward(topk_idx)
    
    # Save handle now to survive activation checkpointing recompute
    ctx.saved_handle = _get_cached_handle(ctx.cache_id_int)


def _combine_backward(ctx: Any, grad_combined: torch.Tensor) -> Tuple[Optional[torch.Tensor], None]:
    """Backward pass for combine: route gradients forward via dispatch.
    
    The backward of combine is mathematically a dispatch operation - we need
    to scatter gradients from tokens to their assigned experts.
    """
    handle = ctx.saved_handle
    if handle is None:
        raise RuntimeError(f"Handle not found for cache_id={ctx.cache_id_int} in combine backward")

    grad_x, _, _, _, _ = _buffer.dispatch_with_permute(
        hidden=grad_combined,
        scaling_factor=None,
        handle=handle,
        pad_multiple=ctx.pad_multiple,
        num_permuted_tokens=ctx.num_permuted_tokens,
    )

    _pop_cached_handle(ctx.cache_id_int)

    return grad_x, None


def _combine_setup_context(ctx: Any, inputs: Tuple, output: torch.Tensor) -> None:
    """Save context for combine backward pass."""
    x, cache_id = inputs
    
    ctx.cache_id_int = cache_id.item()
    ctx.pad_multiple = None
    ctx.num_permuted_tokens = x.shape[0]
    
    # Save handle now to survive activation checkpointing recompute
    ctx.saved_handle = _get_cached_handle(ctx.cache_id_int)


torch.library.register_autograd(
    "hybridep::dispatch",
    _dispatch_backward,
    setup_context=_dispatch_setup_context,
)
torch.library.register_autograd(
    "hybridep::combine",
    _combine_backward,
    setup_context=_combine_setup_context,
)


def get_buffer(
    group: ProcessGroup,
    hidden_dim: int,
    num_tokens: int,
    num_local_experts: int,
    top_k: int = 1,
    capacity_factor: float = 1.0,
    fp8_dispatch: bool = False,
) -> Any:
    """Get or create the HybridEP communication buffer.
    
    The buffer is lazily initialized and reused across forward passes.
    It will be recreated if the configuration needs to grow (e.g., larger
    hidden_dim or more tokens).
    
    Buffer Sizing:
        The buffer must be sized for the RECEIVE side of the all-to-all,
        not the send side. In a balanced MoE:
        
            expected_receive = num_tokens × top_k
        
        This is because each token selects top_k experts, and with balanced
        routing, we expect to receive back top_k times as many token-expert
        pairs as we send tokens.
        
        The capacity_factor provides headroom for load imbalance.
    
    Args:
        group: Expert parallel process group
        hidden_dim: Hidden dimension size (model width)
        num_tokens: Number of input tokens this EP rank will send
        num_local_experts: Number of experts hosted on this EP rank
        top_k: Number of experts each token routes to
        capacity_factor: Multiply expected receive by this for safety margin.
            Use >1.0 if routing may be imbalanced. (default: 1.0)
        fp8_dispatch: Use FP8 quantization for dispatch (not yet supported)
        
    Returns:
        HybridEPBuffer instance ready for dispatch/combine operations.
        
    Raises:
        AssertionError: If fp8_dispatch=True (not yet supported)
    """
    global _buffer

    if fp8_dispatch:
        raise AssertionError("HybridEP FP8 dispatch is not yet supported")
        
    HybridEPBuffer = _require_hybridep()

    # Calculate buffer size for receive side
    # Expected tokens to receive = input_tokens × top_k (balanced case)
    max_tokens_to_receive = int(num_tokens * top_k * capacity_factor)

    # Check if we need to (re)initialize the buffer
    needs_reinit = False
    
    if _buffer is None:
        needs_reinit = True
    else:
        # Reinit if any dimension needs to grow
        if _buffer.group != group:
            needs_reinit = True
        if _buffer.config.hidden_dim < hidden_dim:
            needs_reinit = True
        if _buffer.config.max_num_of_tokens_per_rank < max_tokens_to_receive:
            needs_reinit = True
        if _buffer.config.num_of_experts_per_rank < num_local_experts:
            needs_reinit = True

    if needs_reinit:
        _handle_cache.clear()

        _buffer = HybridEPBuffer(
            group=group,
            hidden_dim=hidden_dim,
            max_num_of_tokens_per_rank=max_tokens_to_receive,
            num_local_experts=num_local_experts,
            use_fp8=fp8_dispatch,
            num_sms_dispatch_api=_num_sms_dispatch,
            num_sms_combine_api=_num_sms_combine,
        )

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
) -> Tuple[torch.Tensor, torch.Tensor, DispatchState]:
    """Dispatch tokens to their assigned experts via HybridEP all-to-all.
    
    This function handles the communication pattern for MoE expert parallelism:
    each token is routed to its top_k selected experts, which may reside on
    different EP ranks. The output tokens are permuted (grouped by expert)
    for efficient batched expert computation.
    
    Args:
        hidden_states: Input token representations.
            Shape: [num_tokens, hidden_dim]
        selected_experts_indices: Expert assignments from router.
            Shape: [num_tokens, top_k], values in [0, num_experts)
        top_scores: Routing probabilities from router.
            Shape: [num_tokens, top_k]
        num_local_experts: Number of experts on this EP rank.
        num_experts: Total number of experts across all EP ranks.
        group: PyTorch ProcessGroup for expert parallel communication.
        score_before_experts: If True (default), multiply routing scores into
            hidden states before dispatch. If False, scores are saved and
            applied in combine_tokens.
        num_permuted_tokens: If provided, enables non-blocking mode which
            avoids a GPU-CPU sync to determine output size. Set this to the
            expected number of output tokens for CUDA graph compatibility.
    
    Returns:
        Tuple containing:
        - hidden_states: Permuted tokens ready for expert computation.
            Shape: [num_permuted_tokens, hidden_dim]
        - tokens_per_expert: Count of tokens assigned to each local expert.
            Shape: [num_local_experts], useful for grouped GEMM.
        - state: DispatchState to pass to combine_tokens.
    """
    # Validate and normalize inputs
    selected_experts_indices, top_scores = _preprocess_dispatch_inputs(
        selected_experts_indices, top_scores
    )

    top_k = selected_experts_indices.shape[1] if selected_experts_indices.dim() == 2 else 1

    get_buffer(
        group=group,
        hidden_dim=hidden_states.shape[1],
        num_tokens=hidden_states.shape[0],
        num_local_experts=num_local_experts,
        top_k=top_k,
        capacity_factor=1.0,
        fp8_dispatch=False,
    )

    (
        hidden_states,
        dispatched_expert_scores,
        tokens_per_expert,
        cache_id,
    ) = torch.ops.hybridep.dispatch(
        hidden_states,
        selected_experts_indices,
        top_scores,
        num_experts,
        num_permuted_tokens,
    )

    hidden_states, permuted_scores = _apply_scores_to_hidden(
        hidden_states, dispatched_expert_scores, score_before_experts
    )

    state = DispatchState(
        cache_id=cache_id,
        num_recv_tokens=hidden_states.shape[0],
        permuted_scores=permuted_scores,
    )
    
    return hidden_states, tokens_per_expert, state


def combine_tokens(
    hidden_states: torch.Tensor,
    state: DispatchState,
) -> torch.Tensor:
    """Combine expert outputs back to original token ordering.
    
    This function reverses the dispatch operation: it gathers expert outputs
    from all EP ranks and unpermutes them to match the original token order.
    If score_before_experts=False was used in dispatch, the routing scores
    are applied here.
    
    Args:
        hidden_states: Expert output representations.
            Shape: [num_permuted_tokens, hidden_dim]
        state: DispatchState returned by dispatch_tokens.
    
    Returns:
        Combined hidden states in original token order.
        Shape: [num_tokens, hidden_dim]
    """
    if state.permuted_scores is not None:
        hidden_states = hidden_states * state.permuted_scores.to(
            hidden_states.dtype
        ).reshape(-1, 1)

    return torch.ops.hybridep.combine(hidden_states, state.cache_id)


__all__ = [
    "dispatch_tokens",
    "combine_tokens",
    "configure",
    "get_buffer",
    "DispatchState",
]
