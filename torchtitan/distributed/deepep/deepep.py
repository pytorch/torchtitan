# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
DeepEP primitives for MoE Expert Parallel.

Provides low-level functions and autograd wrappers for DeepEP communication.
Used by DeepEPExpertParallel in expert_parallel.py.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.distributed import ProcessGroup

try:
    from deep_ep import Buffer  # pyrefly: ignore[missing-import]
    from deep_ep.utils import (  # pyrefly: ignore[missing-import]
        EventHandle,
        EventOverlap,
    )
except ImportError as e:
    raise ImportError(
        "DeepEP is required for this module. "
        "Install from: https://github.com/deepseek-ai/deepep"
    ) from e


# Global buffer (single buffer per process, recreated if group changes)
_buffer: Buffer = None

# Global cache for dispatch handles, keyed by handle_id
# SAC saves the handle_id tensor; we use it to retrieve the non-tensor handle
_handle_cache: dict = {}
_handle_counter: int = 0


def _get_next_handle_id() -> torch.Tensor:
    """Generate a unique handle_id tensor on CPU to avoid GPU-CPU sync."""
    global _handle_counter
    _handle_counter += 1
    return torch.tensor([_handle_counter], dtype=torch.int64, device="cpu")


# ============================================================================
# Custom Op Registration for SAC Integration
# ============================================================================

_lib = torch.library.Library("deepep", "DEF")

# dispatch returns: (recv_x, recv_indices, recv_scores, num_recv_per_expert, handle_id)
_lib.define(
    "dispatch(Tensor x, Tensor topk_idx, Tensor topk_weights, "
    "Tensor num_tokens_per_rank, Tensor num_tokens_per_rdma_rank, "
    "Tensor is_token_in_rank, Tensor num_tokens_per_expert) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor)"
)

# combine returns: combined_x
_lib.define("combine(Tensor x, Tensor handle_id) -> Tensor")


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

    previous_event = EventOverlap(EventHandle())

    (
        recv_x,
        recv_indices,
        recv_scores,
        recv_num_tokens_per_expert_list,
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

    after_event.current_stream_wait()

    handle_id = _get_next_handle_id()
    _handle_cache[handle_id.item()] = handle

    recv_num_tokens_per_expert = torch.tensor(
        recv_num_tokens_per_expert_list, dtype=torch.int32, device="cpu"
    )
    return recv_x, recv_indices, recv_scores, recv_num_tokens_per_expert, handle_id


def _dispatch_setup_context(ctx, inputs, output):
    x, *_ = inputs
    *_, handle_id = output
    ctx.input_dtype = x.dtype
    ctx.saved_handle = _handle_cache.get(handle_id.item())


def _dispatch_backward(
    ctx,
    grad_recv_x,
    grad_recv_indices,
    grad_recv_scores,
    grad_recv_num_tokens_per_expert,
    grad_handle_id,
):
    """Backward for dispatch: performs combine on gradients."""
    global _buffer

    if grad_recv_x is None:
        return None, None, None, None, None, None, None

    # Use handle saved in setup_context instead of from cache
    handle = ctx.saved_handle
    assert handle is not None

    previous_event = EventOverlap(EventHandle())

    grad_x, grad_scores, after_event = _buffer.combine(
        x=grad_recv_x,
        handle=handle,
        topk_weights=grad_recv_scores.float() if grad_recv_scores is not None else None,
        previous_event=previous_event,
        async_finish=True,
        allocate_on_comm_stream=True,
    )

    after_event.current_stream_wait()
    # combine op involves weighted sum on float() so need to convert back
    grad_x = grad_x.to(ctx.input_dtype)
    grad_topk_weights = (
        grad_scores.to(ctx.input_dtype) if grad_scores is not None else None
    )

    return grad_x, None, grad_topk_weights, None, None, None, None


@torch.library.impl(_lib, "combine", "CUDA")
def _combine_op_impl(x: torch.Tensor, handle_id: torch.Tensor) -> torch.Tensor:
    """Execute DeepEP combine."""
    global _buffer

    buffer = _buffer
    assert buffer is not None, "Buffer must be initialized before combine"

    handle = _handle_cache.get(handle_id.item())
    assert handle is not None, f"Handle not found for handle_id={handle_id.item()}"

    previous_event = EventOverlap(EventHandle())

    combined, _, after_event = buffer.combine(
        x=x,
        handle=handle,
        previous_event=previous_event,
        async_finish=True,
        allocate_on_comm_stream=True,
    )

    after_event.current_stream_wait()

    return combined


def _combine_setup_context(ctx, inputs, output):
    _, handle_id = inputs
    # Pop handle from cache and save it for backward
    ctx.saved_handle = _handle_cache.pop(handle_id.item(), None)


def _combine_backward(ctx, grad_combined):
    """Backward for combine: performs dispatch on gradients."""
    global _buffer

    handle = ctx.saved_handle
    previous_event = EventOverlap(EventHandle())

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

    after_event.current_stream_wait()

    return grad_x, None


torch.library.register_autograd(
    "deepep::dispatch", _dispatch_backward, setup_context=_dispatch_setup_context
)
torch.library.register_autograd(
    "deepep::combine", _combine_backward, setup_context=_combine_setup_context
)


def get_hidden_bytes(x: torch.Tensor) -> int:
    """Calculate the number of hidden bytes for one token."""
    # Use at least 2 bytes (bf16 size) so buffer works for both fp8 and bf16 without reallocation
    return x.size(1) * max(x.element_size(), 2)


def get_buffer(group: ProcessGroup, hidden_bytes: int) -> Buffer:
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


def _indices_to_multihot(
    indices: torch.Tensor, scores: torch.Tensor, num_local_experts: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert topk indices to multihot format for permutation.

    Args:
        indices (torch.Tensor): [num_tokens, topk] token indices, where -1 means masked out.
        scores (torch.Tensor): [num_tokens, topk] token scores.
        num_local_experts (int): Number of local experts on current rank.

    Returns:
        A tuple of (routing_map, scores), where routing_map is the multihot vector
        and scores is the multihot probabilities.
    """
    num_tokens = indices.shape[0]
    multihot_routing_map = torch.zeros(
        (num_tokens, num_local_experts), dtype=torch.long, device=indices.device
    )
    multihot_scores = torch.zeros(
        (num_tokens, num_local_experts), dtype=scores.dtype, device=indices.device
    )

    mask = indices != -1
    valid_indices = indices[mask]
    row_indices = torch.arange(num_tokens, device=indices.device).repeat_interleave(
        mask.sum(dim=1)
    )
    # deepep returns local expert indices
    multihot_routing_map[row_indices, valid_indices] = 1
    multihot_scores[row_indices, valid_indices] = scores[mask]

    return multihot_routing_map.bool(), multihot_scores


def _permute_tokens(
    hidden_states: torch.Tensor,
    routing_map: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Permute tokens by expert for grouped_mm.

    Tokens with the same designated expert will be grouped together.

    Args:
        hidden_states (torch.Tensor): The input token tensor, [num_tokens, hidden_dim].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_tokens, num_experts].
        scores (torch.Tensor, optional): The probs tensor, [num_tokens, num_experts].

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        (sorted_hidden_states, permuted_scores, sorted_indices)
    """
    num_tokens = hidden_states.shape[0]  # num of unique tokens
    num_experts = routing_map.shape[1]

    # mask transpose [num_tokens, num_experts] -> [num_experts, num_tokens]
    routing_map_t = routing_map.bool().T.contiguous()
    # create a dense expert-to-token mapping from the sparse token-to-expert mapping
    token_indices = (
        torch.arange(num_tokens, device=routing_map.device)
        .unsqueeze(0)
        .expand(num_experts, -1)
    )
    sorted_indices = token_indices.masked_select(routing_map_t)
    # distribute tokens to experts, num of unique tokens -> num of all tokens to use
    sorted_hidden_states = hidden_states.index_select(0, sorted_indices)

    if scores is not None:
        sorted_scores = scores.T.contiguous().masked_select(routing_map_t)
    else:
        sorted_scores = None

    return sorted_hidden_states, sorted_scores, sorted_indices


def _unpermute_tokens(
    permuted_hidden_states: torch.Tensor,
    sorted_indices: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """
    Reverse permutation applied by _permute_tokens. num_all_tokens is the number of all tokens used in expert computation.

    Args:
        permuted_hidden_states (torch.Tensor): The permuted (duplicated) token tensor, [num_all_tokens, hidden_dim].
        sorted_indices (torch.Tensor): The indices used to sort the tokens, [num_all_tokens, ]
        num_tokens (int): Number of unique tokens received by the current rank.
    Returns:
        torch.Tensor: The tokens are aggregated and restored to their original order.
    """
    hidden_dim = permuted_hidden_states.shape[1]
    output_hidden_states = torch.zeros(
        (num_tokens, hidden_dim),
        dtype=permuted_hidden_states.dtype,
        device=permuted_hidden_states.device,
    )
    output_hidden_states.scatter_add_(
        0, sorted_indices.unsqueeze(1).expand(-1, hidden_dim), permuted_hidden_states
    )
    return output_hidden_states


@dataclass
class DispatchState:
    """State from dispatch needed for combine."""

    handle_id: torch.Tensor  # CPU tensor used to retrieve cached handle
    sorted_indices: torch.Tensor
    num_recv_tokens: int
    permuted_scores: Optional[torch.Tensor] = None


def dispatch_tokens(
    hidden_states: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    top_scores: torch.Tensor,
    num_local_experts: int,
    num_experts: int,
    group: ProcessGroup,
    score_before_experts: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, DispatchState]:
    """Dispatch tokens to experts via DeepEP.

    Args:
        hidden_states: Input tokens [num_tokens, hidden_dim]
        selected_experts_indices: Expert indices for each token [num_tokens, top_k]
        top_scores: Routing scores for each token [num_tokens, top_k]
        num_local_experts: Number of experts on this rank
        num_experts: Total number of experts across all ranks
        group: EP process group
        score_before_experts: If True, apply routing scores before expert computation.

    Returns:
        (permuted_tokens, tokens_per_expert, state_for_combine)
    """
    selected_experts_indices = selected_experts_indices.contiguous()
    top_scores = top_scores.contiguous()

    # Mask out zero-score tokens
    selected_experts_indices = selected_experts_indices.masked_fill(top_scores == 0, -1)

    # Ensure float32 scores (DeepEP requirement)
    if top_scores.dtype != torch.float32:
        top_scores = top_scores.float()

    buffer = get_buffer(group, get_hidden_bytes(hidden_states))

    # Calculate dispatch layout before actual dispatch
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert_dispatch,
        is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(
        topk_idx=selected_experts_indices, num_experts=num_experts
    )

    # Dispatch tokens to experts
    (
        hidden_states,
        dispatched_indices,
        dispatched_expert_scores,
        num_tokens_per_expert,
        handle_id,
    ) = torch.ops.deepep.dispatch(
        hidden_states,
        selected_experts_indices,
        top_scores,
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        is_token_in_rank,
        num_tokens_per_expert_dispatch,
    )

    dispatched_routing_map, dispatched_expert_scores_multihot = _indices_to_multihot(
        dispatched_indices, dispatched_expert_scores, num_local_experts
    )

    num_recv_tokens = hidden_states.shape[0]

    # Sort tokens by expert for grouped_mm
    hidden_states, permuted_scores, sorted_indices = _permute_tokens(
        hidden_states, dispatched_routing_map, scores=dispatched_expert_scores_multihot
    )

    # tokens_per_expert is returned from dispatch as int32 on CPU, move to GPU
    num_tokens_per_expert = num_tokens_per_expert.to(hidden_states.device)

    if score_before_experts and permuted_scores is not None:
        # Avoid float32 conversion to save memory
        hidden_states = hidden_states * permuted_scores.to(hidden_states.dtype).reshape(
            -1, 1
        )
        permuted_scores_for_state = None
    else:
        permuted_scores_for_state = permuted_scores

    state = DispatchState(
        handle_id=handle_id,
        sorted_indices=sorted_indices,
        num_recv_tokens=num_recv_tokens,
        permuted_scores=permuted_scores_for_state,
    )

    return hidden_states, num_tokens_per_expert, state


def combine_tokens(
    hidden_states: torch.Tensor,
    state: DispatchState,
) -> torch.Tensor:
    """Combine tokens from experts via DeepEP."""
    if state.permuted_scores is not None:
        # In-place multiplication to save memory
        hidden_states = hidden_states * state.permuted_scores.to(
            hidden_states.dtype
        ).reshape(-1, 1)

    hidden_states = _unpermute_tokens(
        hidden_states, state.sorted_indices, state.num_recv_tokens
    )

    hidden_states = torch.ops.deepep.combine(hidden_states, state.handle_id)

    return hidden_states
