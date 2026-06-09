# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MinimalAsyncEP primitives for constrained MoE expert parallel dispatch.

This backend is intentionally narrow: it supports the launch shape where the
EP process group is the data-parallel group and TP/CP/PP/SP are disabled.
The symmetric-memory allocation is explicit and must happen before dispatch.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from torchtitan.distributed.minimal_async_ep_kernels import (
    active_swiglu_backward as active_swiglu_backward_kernel,
    active_swiglu_forward,
    copy_full_counts_to_peers,
    copy_rows_to_peers,
    expand_topk_grad,
    fill_combine_metadata,
    fill_dispatch_metadata,
    invert_flat_indices as _invert_flat_indices,
    reduce_topk_slots,
    topk_scores_grad,
)
from torchtitan.tools.logging import logger


_HIDDEN_RECV_BUFFER_COUNT = 2

_hidden_recv_buffers: list[torch.Tensor] | None = None
_hidden_recv_handles: list[Any] | None = None
_hidden_recv_peer_buffers: list[list[torch.Tensor]] | None = None
_hidden_recv_peer_ptrs: list[torch.Tensor] | None = None
_hidden_recv_buffer_index: int = 0
_counts_recv_buffer: torch.Tensor | None = None
_counts_recv_handle: Any = None
_counts_recv_peer_buffers: list[torch.Tensor] | None = None
_counts_recv_peer_ptrs: torch.Tensor | None = None
_group: dist.ProcessGroup | None = None
_tokens_per_rank: int = 0

_HIDDEN_READY_CHANNEL = 0
_COUNTS_READY_CHANNEL = 0


@dataclass
class MinimalAsyncEPDispatchMetadata:
    """MinimalAsyncEP metadata from dispatch needed for combine.

    Shape symbols match ``MinimalAsyncEPTokenDispatcher.dispatch`` and
    ``AllToAllTokenDispatcher``: ``T`` local tokens, ``D`` model dimension,
    ``N = T * K`` T-major routed rows before EP exchange, and ``R`` active rows
    assigned to this rank's local experts. MinimalAsyncEP additionally keeps
    a static receive capacity ``R_max >= R``.

    Field shapes:
        dispatch_dst_ranks, dispatch_dst_rows: ``(N,)``.
        combine_dst_ranks, combine_dst_rows: ``(R_max,)``.
        combine_num_valid_rows: ``(1,)`` active receive rows, where
            ``combine_num_valid_rows[0] == R``.
        E_row_to_T_row,
            T_row_to_E_row,
            routed_scores: ``(N,)``.
        num_tokens: ``T``.
        top_k: ``K``.
    """

    dispatch_dst_ranks: torch.Tensor
    dispatch_dst_rows: torch.Tensor
    combine_dst_ranks: torch.Tensor
    combine_dst_rows: torch.Tensor
    combine_num_valid_rows: torch.Tensor
    E_row_to_T_row: torch.Tensor  # noqa: N815
    T_row_to_E_row: torch.Tensor  # noqa: N815
    routed_scores: torch.Tensor
    num_tokens: int
    top_k: int


def init_buffer(
    group: dist.ProcessGroup,
    hidden_dim: int,
    tokens_per_rank: int,
    num_local_experts: int,
    top_k: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Initialize the process-local MinimalAsyncEP symmetric-memory buffer."""
    global _hidden_recv_buffers, _hidden_recv_handles
    global _hidden_recv_peer_buffers, _hidden_recv_peer_ptrs
    global _hidden_recv_buffer_index
    global _counts_recv_buffer, _counts_recv_handle, _counts_recv_peer_buffers
    global _counts_recv_peer_ptrs
    global _group, _tokens_per_rank

    device = torch.device(device)
    max_routed_tokens = (
        group.size() * tokens_per_rank * min(top_k, num_local_experts)
    )
    num_experts = group.size() * num_local_experts
    needs_init = (
        _hidden_recv_buffers is None
        or _hidden_recv_handles is None
        or _hidden_recv_peer_buffers is None
        or _hidden_recv_peer_ptrs is None
        or _counts_recv_buffer is None
        or _counts_recv_handle is None
        or _counts_recv_peer_buffers is None
        or _counts_recv_peer_ptrs is None
        or _group != group
        or _hidden_recv_buffers[0].shape[1] < hidden_dim
        or _hidden_recv_buffers[0].shape[0] < max_routed_tokens
        or _counts_recv_buffer.shape[1] < num_experts
        or _tokens_per_rank < tokens_per_rank
        or _hidden_recv_buffers[0].dtype != dtype
        or _hidden_recv_buffers[0].device != device
    )
    if not needs_init:
        return

    logger.info(
        "Initializing MinimalAsyncEP buffer: hidden_dim=%d, tokens_per_rank=%d, "
        "top_k=%d, num_local_experts=%d, ep_size=%d, max_routed_tokens=%d",
        hidden_dim,
        tokens_per_rank,
        top_k,
        num_local_experts,
        group.size(),
        max_routed_tokens,
    )
    backend = symm_mem.get_backend(device)
    if backend != "CUDA":
        raise RuntimeError(
            "MinimalAsyncEP custom all-to-allv requires the symmetric-memory CUDA "
            f"backend, got {backend}."
        )

    _hidden_recv_buffers = [
        symm_mem.empty(
            max_routed_tokens,
            hidden_dim,
            dtype=dtype,
            device=device,
        )
        for _ in range(_HIDDEN_RECV_BUFFER_COUNT)
    ]
    _counts_recv_buffer = symm_mem.empty(
        group.size(),
        num_experts,
        dtype=torch.int64,
        device=device,
    )
    _hidden_recv_handles = [
        symm_mem.rendezvous(hidden_recv_buffer, group)
        for hidden_recv_buffer in _hidden_recv_buffers
    ]
    _counts_recv_handle = symm_mem.rendezvous(_counts_recv_buffer, group)
    _hidden_recv_peer_buffers = [
        [
            hidden_recv_handle.get_buffer(
                peer,
                hidden_recv_buffer.shape,
                hidden_recv_buffer.dtype,
            )
            for peer in range(group.size())
        ]
        for hidden_recv_buffer, hidden_recv_handle in zip(
            _hidden_recv_buffers,
            _hidden_recv_handles,
        )
    ]
    _hidden_recv_peer_ptrs = [
        torch.tensor(
            [peer_buffer.data_ptr() for peer_buffer in hidden_recv_peer_buffers],
            dtype=torch.int64,
            device=device,
        )
        for hidden_recv_peer_buffers in _hidden_recv_peer_buffers
    ]
    _counts_recv_peer_buffers = [
        _counts_recv_handle.get_buffer(
            peer,
            _counts_recv_buffer.shape,
            _counts_recv_buffer.dtype,
        )
        for peer in range(group.size())
    ]
    _counts_recv_peer_ptrs = torch.tensor(
        [peer_buffer.data_ptr() for peer_buffer in _counts_recv_peer_buffers],
        dtype=torch.int64,
        device=device,
    )

    _group = group
    _tokens_per_rank = tokens_per_rank
    _hidden_recv_buffer_index = 0


def _copy_rows_to_peers_cuda(
    x: torch.Tensor,
    dst_ranks: torch.Tensor,
    dst_rows: torch.Tensor,
    num_rows: int,
    *,
    block_m: int = 1,
    num_warps: int = 4,
    src_rows: torch.Tensor | None = None,
    num_valid_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    global _hidden_recv_buffer_index
    assert _hidden_recv_buffers is not None
    assert _hidden_recv_handles is not None
    assert _hidden_recv_peer_buffers is not None
    assert _hidden_recv_peer_ptrs is not None
    assert _group is not None

    buffer_index = _hidden_recv_buffer_index
    _hidden_recv_buffer_index = (
        _hidden_recv_buffer_index + 1
    ) % _HIDDEN_RECV_BUFFER_COUNT
    hidden_recv_buffer = _hidden_recv_buffers[buffer_index]
    hidden_recv_handle = _hidden_recv_handles[buffer_index]
    hidden_recv_peer_buffers = _hidden_recv_peer_buffers[buffer_index]
    hidden_recv_peer_ptrs = _hidden_recv_peer_ptrs[buffer_index]

    copy_rows_to_peers(
        x,
        hidden_recv_peer_buffers,
        dst_ranks,
        dst_rows,
        ep_size=_group.size(),
        num_rows=num_rows,
        num_cols=x.shape[1],
        block_m=block_m,
        num_warps=num_warps,
        src_rows=src_rows,
        dst_ptrs=hidden_recv_peer_ptrs,
        num_valid_rows=num_valid_rows,
    )
    _wait_hidden_ready(hidden_recv_handle)
    return hidden_recv_buffer


def _wait_hidden_ready(hidden_recv_handle: Any) -> None:
    _wait_ready(hidden_recv_handle, _HIDDEN_READY_CHANNEL)


def _wait_counts_ready() -> None:
    assert _counts_recv_handle is not None
    _wait_ready(_counts_recv_handle, _COUNTS_READY_CHANNEL)


def _wait_ready(handle: Any, channel: int) -> None:
    """EP-group barrier: ensure every peer has finished writing into this
    rank's symmetric receive buffer before the buffer is read.

    Issues a single fused ``barrier`` kernel that signals and polls all peers
    concurrently. This was previously a Python loop of ``2 * (ep_size - 1)``
    per-peer ``put_signal`` / ``wait_signal`` kernels, all serialized and fully
    exposed on the critical path (each ``wait_signal`` its own spin-wait
    kernel) -- the dominant MinimalAsyncEP comm cost once CPU-launch overhead is removed
    by CUDA graphs / compiled steps.
    """
    assert _group is not None
    handle.barrier(channel=channel)


def _copy_all_counts_to_peers_cuda(
    num_local_tokens_per_expert_E: torch.Tensor,  # noqa: N803
    ep_size: int,
) -> torch.Tensor:
    assert _counts_recv_buffer is not None
    assert _counts_recv_peer_buffers is not None
    assert _counts_recv_peer_ptrs is not None
    assert _group is not None

    num_experts = num_local_tokens_per_expert_E.numel()
    copy_full_counts_to_peers(
        num_local_tokens_per_expert_E,
        _counts_recv_peer_buffers,
        rank=_group.rank(),
        ep_size=ep_size,
        num_experts=num_experts,
        dst_ptrs=_counts_recv_peer_ptrs,
    )
    return _counts_recv_buffer


def _compute_direct_metadata(
    num_local_tokens_per_expert_E: torch.Tensor,  # noqa: N803
    all_tokens_per_expert_RE: torch.Tensor,  # noqa: N803
    num_routed_rows: int,
    receive_capacity: int,
    ep_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    assert _group is not None

    rank = _group.rank()
    num_experts = num_local_tokens_per_expert_E.numel()
    num_local_experts = num_experts // ep_size

    counts_sde = all_tokens_per_expert_RE.view(
        ep_size,
        ep_size,
        num_local_experts,
    )
    source_prefix_sde = counts_sde.cumsum(0) - counts_sde
    total_de = counts_sde.sum(0)
    expert_starts_de = total_de.cumsum(1) - total_de
    tokens_per_expert_e = total_de[rank]

    local_dest_offsets_E = (  # noqa: N806
        expert_starts_de + source_prefix_sde[rank]
    ).reshape(num_experts)
    local_count_ends_E = num_local_tokens_per_expert_E.cumsum(0)  # noqa: N806
    local_count_starts_E = local_count_ends_E - num_local_tokens_per_expert_E  # noqa: N806
    dispatch_dst_ranks_N, dispatch_dst_rows_N = fill_dispatch_metadata(  # noqa: N806
        num_local_tokens_per_expert_E,
        local_dest_offsets_E,
        local_count_starts_E,
        num_routed_tokens=num_routed_rows,
        num_local_experts=num_local_experts,
        max_tokens_per_segment=_tokens_per_rank,
    )

    segment_lens = counts_sde[:, rank, :].t().reshape(-1)
    output_ends = segment_lens.cumsum(0)
    output_starts = output_ends - segment_lens
    source_input_starts_RE = (  # noqa: N806
        all_tokens_per_expert_RE.cumsum(1) - all_tokens_per_expert_RE
    )
    combine_dst_ranks, combine_dst_rows, combine_num_valid_rows = fill_combine_metadata(
        segment_lens,
        output_starts,
        source_input_starts_RE,
        ep_rank=rank,
        ep_size=ep_size,
        num_local_experts=num_local_experts,
        receive_capacity=receive_capacity,
        max_tokens_per_segment=_tokens_per_rank,
    )

    return (
        dispatch_dst_ranks_N,
        dispatch_dst_rows_N,
        combine_dst_ranks,
        combine_dst_rows,
        combine_num_valid_rows,
        tokens_per_expert_e,
    )


def _dispatch_to_experts(
    x_ND: torch.Tensor,  # noqa: N803
    dispatch_dst_ranks: torch.Tensor,
    dispatch_dst_rows: torch.Tensor,
    num_routed_rows: int,
) -> torch.Tensor:
    hidden_recv_buffer = _copy_rows_to_peers_cuda(
        x_ND,
        dispatch_dst_ranks,
        dispatch_dst_rows,
        num_routed_rows,
        block_m=4,
        num_warps=8,
    )
    return hidden_recv_buffer


def _combine_to_origin(
    x_RD: torch.Tensor,  # noqa: N803
    combine_dst_ranks: torch.Tensor,
    combine_dst_rows: torch.Tensor,
    combine_num_valid_rows: torch.Tensor,
    num_routed_rows: int,
) -> torch.Tensor:
    combined = _copy_rows_to_peers_cuda(
        x_RD,
        combine_dst_ranks,
        combine_dst_rows,
        x_RD.shape[0],
        block_m=4,
        num_valid_rows=combine_num_valid_rows,
    )
    return combined[:num_routed_rows, : x_RD.shape[1]]


@torch.library.custom_op(
    "minimal_async_ep::active_swiglu",
    mutates_args=(),
    device_types="cuda",
)
def active_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    active_rows: torch.Tensor,
) -> torch.Tensor:
    """Compute ``silu(gate) * up`` only for active rows in padded expert buffers."""
    return active_swiglu_forward(gate, up, active_rows)


@active_swiglu.register_fake
def active_swiglu_fake(
    gate: torch.Tensor,
    up: torch.Tensor,
    active_rows: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(gate)


@torch.library.custom_op(
    "minimal_async_ep::invert_flat_indices",
    mutates_args=(),
    device_types="cuda",
)
def invert_flat_indices(
    flat_indices: torch.Tensor,
    num_rows: int,
) -> torch.Tensor:
    """Invert an E-major to T-major routed-row mapping."""
    return _invert_flat_indices(flat_indices, num_rows=num_rows)


@invert_flat_indices.register_fake
def invert_flat_indices_fake(
    flat_indices: torch.Tensor,
    num_rows: int,
) -> torch.Tensor:
    return flat_indices.new_empty(num_rows)


@torch.library.custom_op(
    "minimal_async_ep::dispatch_metadata",
    mutates_args=(),
    device_types="cuda",
)
def dispatch_metadata(
    num_local_tokens_per_expert_E: torch.Tensor,  # noqa: N803
    num_routed_rows: int,
    receive_capacity: int,
    ep_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Exchange int64 per-expert local token counts and build routing metadata."""

    # Mirrors AllToAllTokenDispatcher's count exchange: each rank starts with
    # counts for its local tokens over all global experts, then learns how many
    # tokens every peer will send to each of this rank's local experts.
    all_tokens_per_expert_RE = _copy_all_counts_to_peers_cuda(  # noqa: N806
        num_local_tokens_per_expert_E,
        ep_size,
    )
    _wait_counts_ready()

    # Instead of materializing an all-to-all rank-major receive tensor and then
    # calling _permute(), compute the final E-major receive rows directly.
    return _compute_direct_metadata(
        num_local_tokens_per_expert_E,
        all_tokens_per_expert_RE,
        num_routed_rows,
        receive_capacity,
        ep_size,
    )


@dispatch_metadata.register_fake
def dispatch_metadata_fake(
    num_local_tokens_per_expert_E: torch.Tensor,  # noqa: N803
    num_routed_rows: int,
    receive_capacity: int,
    ep_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    num_local_experts = num_local_tokens_per_expert_E.shape[0] // ep_size
    dispatch_dst_ranks = num_local_tokens_per_expert_E.new_empty(
        num_routed_rows,
        dtype=torch.int64,
    )
    dispatch_dst_rows = num_local_tokens_per_expert_E.new_empty(
        num_routed_rows,
        dtype=torch.int64,
    )
    combine_dst_ranks = num_local_tokens_per_expert_E.new_empty(
        receive_capacity,
        dtype=torch.int64,
    )
    combine_dst_rows = num_local_tokens_per_expert_E.new_empty(
        receive_capacity,
        dtype=torch.int64,
    )
    combine_num_valid_rows = num_local_tokens_per_expert_E.new_empty(
        1,
        dtype=torch.int64,
    )
    tokens_per_expert = num_local_tokens_per_expert_E.new_empty(
        num_local_experts,
        dtype=torch.int64,
    )
    return (
        dispatch_dst_ranks,
        dispatch_dst_rows,
        combine_dst_ranks,
        combine_dst_rows,
        combine_num_valid_rows,
        tokens_per_expert,
    )


@torch.library.custom_op(
    "minimal_async_ep::dispatch_forward",
    mutates_args=(),
    device_types="cuda",
)
def dispatch_forward(
    dispatch_input: torch.Tensor,
    dispatch_dst_ranks: torch.Tensor,
    dispatch_dst_rows: torch.Tensor,
    E_row_to_token_N: torch.Tensor,  # noqa: N803
    receive_capacity: int,
) -> torch.Tensor:
    """Dispatch E-major routed rows using source token-row indices."""
    num_routed_rows = E_row_to_token_N.numel()
    # This direct copy corresponds to AllToAllTokenDispatcher's token all-to-all;
    # dispatch_dst_rows already point at the post-_permute E-major layout.
    hidden_recv_buffer = _copy_rows_to_peers_cuda(
        dispatch_input,
        dispatch_dst_ranks,
        dispatch_dst_rows,
        num_routed_rows,
        block_m=4,
        num_warps=8,
        src_rows=E_row_to_token_N,
    )
    return hidden_recv_buffer


@dispatch_forward.register_fake
def dispatch_forward_fake(
    dispatch_input: torch.Tensor,
    dispatch_dst_ranks: torch.Tensor,
    dispatch_dst_rows: torch.Tensor,
    E_row_to_token_N: torch.Tensor,  # noqa: N803
    receive_capacity: int,
) -> torch.Tensor:
    return dispatch_input.new_empty(receive_capacity, dispatch_input.shape[1])


@torch.library.custom_op(
    "minimal_async_ep::combine_forward",
    mutates_args=(),
    device_types="cuda",
)
def combine_forward(
    x: torch.Tensor,
    dispatch_dst_ranks: torch.Tensor,
    dispatch_dst_rows: torch.Tensor,
    combine_dst_ranks: torch.Tensor,
    combine_dst_rows: torch.Tensor,
    combine_num_valid_rows: torch.Tensor,
    T_row_to_E_row_N: torch.Tensor,  # noqa: N803
    E_row_to_T_row_N: torch.Tensor,  # noqa: N803
    routed_scores_N: torch.Tensor,  # noqa: N803
    num_tokens: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move expert outputs to origin ranks and reduce routed top-k rows."""
    routed_output_ND = _combine_to_origin(  # noqa: N806
        x,
        combine_dst_ranks,
        combine_dst_rows,
        combine_num_valid_rows,
        E_row_to_T_row_N.numel(),
    )
    out_TD = reduce_topk_slots(  # noqa: N806
        routed_output_ND,
        T_row_to_E_row_N,
        routed_scores_N,
        num_tokens=num_tokens,
        top_k=top_k,
        scores_are_slot_ordered=True,
    )
    return out_TD, routed_output_ND


@combine_forward.register_fake
def combine_forward_fake(
    x: torch.Tensor,
    dispatch_dst_ranks: torch.Tensor,
    dispatch_dst_rows: torch.Tensor,
    combine_dst_ranks: torch.Tensor,
    combine_dst_rows: torch.Tensor,
    combine_num_valid_rows: torch.Tensor,
    T_row_to_E_row_N: torch.Tensor,  # noqa: N803
    E_row_to_T_row_N: torch.Tensor,  # noqa: N803
    routed_scores_N: torch.Tensor,  # noqa: N803
    num_tokens: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        x.new_empty(num_tokens, x.shape[1]),
        x.new_empty(
            E_row_to_T_row_N.numel(),
            x.shape[1],
        ),
    )


@torch.library.custom_op(
    "minimal_async_ep::active_swiglu_backward",
    mutates_args=(),
    device_types="cuda",
)
def active_swiglu_backward(
    grad_out: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    active_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute active-row gradients for ``minimal_async_ep::active_swiglu``."""
    return active_swiglu_backward_kernel(grad_out, gate, up, active_rows)


@active_swiglu_backward.register_fake
def active_swiglu_backward_fake(
    grad_out: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    active_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(gate), torch.empty_like(up)


@torch.library.custom_op(
    "minimal_async_ep::dispatch_backward",
    mutates_args=(),
    device_types="cuda",
)
def dispatch_backward(
    grad_hidden: torch.Tensor,
    combine_dst_ranks: torch.Tensor,
    combine_dst_rows: torch.Tensor,
    combine_num_valid_rows: torch.Tensor,
    T_row_to_E_row_N: torch.Tensor,  # noqa: N803
    num_routed_rows: int,
    num_tokens: int,
    top_k: int,
) -> torch.Tensor:
    """Move dispatched activation gradients back and reduce routed top-k rows."""
    grad_routed_input = _combine_to_origin(
        grad_hidden,
        combine_dst_ranks,
        combine_dst_rows,
        combine_num_valid_rows,
        num_routed_rows,
    )
    return reduce_topk_slots(
        grad_routed_input,
        T_row_to_E_row_N,
        None,
        num_tokens=num_tokens,
        top_k=top_k,
    )


@dispatch_backward.register_fake
def dispatch_backward_fake(
    grad_hidden: torch.Tensor,
    combine_dst_ranks: torch.Tensor,
    combine_dst_rows: torch.Tensor,
    combine_num_valid_rows: torch.Tensor,
    T_row_to_E_row_N: torch.Tensor,  # noqa: N803
    num_routed_rows: int,
    num_tokens: int,
    top_k: int,
) -> torch.Tensor:
    return grad_hidden.new_empty(num_tokens, grad_hidden.shape[1])


@torch.library.custom_op(
    "minimal_async_ep::combine_backward",
    mutates_args=(),
    device_types="cuda",
)
def combine_backward(
    grad_out: torch.Tensor,
    dispatch_dst_ranks: torch.Tensor,
    dispatch_dst_rows: torch.Tensor,
    E_row_to_T_row_N: torch.Tensor,  # noqa: N803
    routed_scores_N: torch.Tensor,  # noqa: N803
    saved_routed_output_ND: torch.Tensor,  # noqa: N803
    top_k: int,
    receive_capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand combine gradients and dispatch them back to expert-owner ranks."""
    grad_routed_output = expand_topk_grad(
        grad_out,
        E_row_to_T_row_N,
        routed_scores_N,
        top_k=top_k,
        dtype=grad_out.dtype,
        scores_are_slot_ordered=True,
    )
    grad_scores = topk_scores_grad(
        saved_routed_output_ND,
        grad_out,
        E_row_to_T_row_N,
        top_k=top_k,
        dtype=routed_scores_N.dtype,
        scores_are_slot_ordered=True,
    )

    grad_x = _dispatch_to_experts(
        grad_routed_output,
        dispatch_dst_ranks,
        dispatch_dst_rows,
        E_row_to_T_row_N.numel(),
    )
    return grad_x, grad_scores


@combine_backward.register_fake
def combine_backward_fake(
    grad_out: torch.Tensor,
    dispatch_dst_ranks: torch.Tensor,
    dispatch_dst_rows: torch.Tensor,
    E_row_to_T_row_N: torch.Tensor,  # noqa: N803
    routed_scores_N: torch.Tensor,  # noqa: N803
    saved_routed_output_ND: torch.Tensor,  # noqa: N803
    top_k: int,
    receive_capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        grad_out.new_empty(receive_capacity, grad_out.shape[1]),
        routed_scores_N.new_empty(routed_scores_N.shape),
    )


def active_swiglu_autograd_backward(ctx, grad_out):
    gate, up, active_rows = ctx.saved_tensors
    grad_gate, grad_up = active_swiglu_backward(
        grad_out,
        gate,
        up,
        active_rows,
    )
    return grad_gate, grad_up, None


def active_swiglu_setup_context(ctx, inputs, output):
    gate, up, active_rows = inputs
    ctx.save_for_backward(gate, up, active_rows)


class MinimalAsyncEPDispatch(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        dispatch_input: torch.Tensor,
        dispatch_dst_ranks: torch.Tensor,
        dispatch_dst_rows: torch.Tensor,
        combine_dst_ranks: torch.Tensor,
        combine_dst_rows: torch.Tensor,
        combine_num_valid_rows: torch.Tensor,
        E_row_to_token_N: torch.Tensor,  # noqa: N803
        T_row_to_E_row_N: torch.Tensor,  # noqa: N803
        num_tokens: int,
        receive_capacity: int,
    ) -> torch.Tensor:
        hidden = dispatch_forward(
            dispatch_input,
            dispatch_dst_ranks,
            dispatch_dst_rows,
            E_row_to_token_N,
            receive_capacity,
        )

        ctx.num_routed_rows = E_row_to_token_N.numel()
        ctx.num_tokens = num_tokens
        ctx.top_k = E_row_to_token_N.numel() // num_tokens
        ctx.save_for_backward(
            combine_dst_ranks,
            combine_dst_rows,
            combine_num_valid_rows,
            T_row_to_E_row_N,
        )
        return hidden

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(
        ctx,
        grad_hidden,
        *unused_grads,
    ):
        (
            combine_dst_ranks,
            combine_dst_rows,
            combine_num_valid_rows,
            T_row_to_E_row_N,
        ) = ctx.saved_tensors

        grad_input = dispatch_backward(
            grad_hidden,
            combine_dst_ranks,
            combine_dst_rows,
            combine_num_valid_rows,
            T_row_to_E_row_N,
            ctx.num_routed_rows,
            ctx.num_tokens,
            ctx.top_k,
        )

        return grad_input, None, None, None, None, None, None, None, None, None


class MinimalAsyncEPCombine(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        dispatch_dst_ranks: torch.Tensor,
        dispatch_dst_rows: torch.Tensor,
        combine_dst_ranks: torch.Tensor,
        combine_dst_rows: torch.Tensor,
        combine_num_valid_rows: torch.Tensor,
        T_row_to_E_row_N: torch.Tensor,  # noqa: N803
        E_row_to_T_row_N: torch.Tensor,  # noqa: N803
        routed_scores_N: torch.Tensor,  # noqa: N803
        num_tokens: int,
        top_k: int,
    ) -> torch.Tensor:
        combined, routed_output_ND = combine_forward(  # noqa: N806
            hidden_states,
            dispatch_dst_ranks,
            dispatch_dst_rows,
            combine_dst_ranks,
            combine_dst_rows,
            combine_num_valid_rows,
            T_row_to_E_row_N,
            E_row_to_T_row_N,
            routed_scores_N,
            num_tokens,
            top_k,
        )

        ctx.top_k = top_k
        ctx.receive_capacity = hidden_states.shape[0]
        ctx.save_for_backward(
            dispatch_dst_ranks,
            dispatch_dst_rows,
            E_row_to_T_row_N,
            routed_scores_N,
            routed_output_ND,
        )
        return combined

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(
        ctx,
        grad_out,
    ):
        (
            dispatch_dst_ranks,
            dispatch_dst_rows,
            E_row_to_T_row_N,
            routed_scores_N,
            routed_output_ND,
        ) = ctx.saved_tensors
        grad_x, grad_scores = combine_backward(
            grad_out,
            dispatch_dst_ranks,
            dispatch_dst_rows,
            E_row_to_T_row_N,
            routed_scores_N,
            routed_output_ND,
            ctx.top_k,
            ctx.receive_capacity,
        )

        return (
            grad_x,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            grad_scores,
            None,
            None,
        )


active_swiglu.register_autograd(
    active_swiglu_autograd_backward, setup_context=active_swiglu_setup_context
)

__all__ = [
    "MinimalAsyncEPCombine",
    "MinimalAsyncEPDispatch",
    "MinimalAsyncEPDispatchMetadata",
    "active_swiglu",
    "init_buffer",
    "invert_flat_indices",
]
