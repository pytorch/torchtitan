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

_HIDDEN_READY_CHANNEL = 0
_COUNTS_READY_CHANNEL = 0


@dataclass
class _MinimalAsyncEPBufferState:
    """Process-local symmetric-memory state initialized as one unit."""

    group: dist.ProcessGroup
    tokens_per_rank: int
    hidden_recv_buffers: list[torch.Tensor]
    hidden_recv_handles: list[Any]
    hidden_recv_peer_buffers: list[list[torch.Tensor]]
    hidden_recv_peer_ptrs: list[torch.Tensor]
    counts_recv_buffer: torch.Tensor
    counts_recv_handle: Any
    counts_recv_peer_buffers: list[torch.Tensor]
    counts_recv_peer_ptrs: torch.Tensor
    hidden_recv_buffer_index: int = 0


_buffer_state: _MinimalAsyncEPBufferState | None = None


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
    global _buffer_state

    device = torch.device(device)
    max_routed_tokens = (
        group.size() * tokens_per_rank * min(top_k, num_local_experts)
    )
    num_experts = group.size() * num_local_experts
    assert _buffer_state is None

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

    hidden_recv_buffers = [
        symm_mem.empty(
            max_routed_tokens,
            hidden_dim,
            dtype=dtype,
            device=device,
        )
        for _ in range(_HIDDEN_RECV_BUFFER_COUNT)
    ]
    counts_recv_buffer = symm_mem.empty(
        group.size(),
        num_experts,
        dtype=torch.int64,
        device=device,
    )
    hidden_recv_handles = [
        symm_mem.rendezvous(hidden_recv_buffer, group)
        for hidden_recv_buffer in hidden_recv_buffers
    ]
    counts_recv_handle = symm_mem.rendezvous(counts_recv_buffer, group)
    hidden_recv_peer_buffers = [
        [
            hidden_recv_handle.get_buffer(
                peer,
                hidden_recv_buffer.shape,
                hidden_recv_buffer.dtype,
            )
            for peer in range(group.size())
        ]
        for hidden_recv_buffer, hidden_recv_handle in zip(
            hidden_recv_buffers,
            hidden_recv_handles,
        )
    ]
    hidden_recv_peer_ptrs = [
        torch.tensor(
            [peer_buffer.data_ptr() for peer_buffer in hidden_recv_peer_buffers],
            dtype=torch.int64,
            device=device,
        )
        for hidden_recv_peer_buffers in hidden_recv_peer_buffers
    ]
    counts_recv_peer_buffers = [
        counts_recv_handle.get_buffer(
            peer,
            counts_recv_buffer.shape,
            counts_recv_buffer.dtype,
        )
        for peer in range(group.size())
    ]
    counts_recv_peer_ptrs = torch.tensor(
        [peer_buffer.data_ptr() for peer_buffer in counts_recv_peer_buffers],
        dtype=torch.int64,
        device=device,
    )

    _buffer_state = _MinimalAsyncEPBufferState(
        group=group,
        tokens_per_rank=tokens_per_rank,
        hidden_recv_buffers=hidden_recv_buffers,
        hidden_recv_handles=hidden_recv_handles,
        hidden_recv_peer_buffers=hidden_recv_peer_buffers,
        hidden_recv_peer_ptrs=hidden_recv_peer_ptrs,
        counts_recv_buffer=counts_recv_buffer,
        counts_recv_handle=counts_recv_handle,
        counts_recv_peer_buffers=counts_recv_peer_buffers,
        counts_recv_peer_ptrs=counts_recv_peer_ptrs,
    )


def _copy_rows_to_peers_and_wait_cuda(
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
    """Copy rows through symmetric memory and wait before returning.

    MinimalAsyncEP exposes synchronous custom-op boundaries: callers receive a
    readable buffer, not an async handle. This keeps the operator interface
    simple for graph capture, but means this backend does not provide
    microbatch communication overlap.
    """
    assert _buffer_state is not None

    buffer_index = _buffer_state.hidden_recv_buffer_index
    _buffer_state.hidden_recv_buffer_index = (
        _buffer_state.hidden_recv_buffer_index + 1
    ) % _HIDDEN_RECV_BUFFER_COUNT
    hidden_recv_buffer = _buffer_state.hidden_recv_buffers[buffer_index]
    hidden_recv_handle = _buffer_state.hidden_recv_handles[buffer_index]
    hidden_recv_peer_buffers = _buffer_state.hidden_recv_peer_buffers[buffer_index]
    hidden_recv_peer_ptrs = _buffer_state.hidden_recv_peer_ptrs[buffer_index]

    copy_rows_to_peers(
        x,
        hidden_recv_peer_buffers,
        dst_ranks,
        dst_rows,
        ep_size=_buffer_state.group.size(),
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
    assert _buffer_state is not None
    _wait_ready(_buffer_state.counts_recv_handle, _COUNTS_READY_CHANNEL)


def _wait_ready(handle: Any, channel: int) -> None:
    """EP-group barrier: ensure every peer has finished writing into this
    rank's symmetric receive buffer before the buffer is read.

    This wait intentionally happens before custom ops return tensors that are
    consumed by later compute. MinimalAsyncEP is a synchronous, CUDA-graphable
    backend and does not expose a separate async handle for microbatch overlap.

    Issues a single fused ``barrier`` kernel that signals and polls all peers
    concurrently. This was previously a Python loop of ``2 * (ep_size - 1)``
    per-peer ``put_signal`` / ``wait_signal`` kernels, all serialized and fully
    exposed on the critical path (each ``wait_signal`` its own spin-wait
    kernel) -- the dominant MinimalAsyncEP comm cost once CPU-launch overhead
    is removed by CUDA graphs / compiled steps.
    """
    handle.barrier(channel=channel)


def _copy_all_counts_to_peers_and_wait_cuda(
    num_local_tokens_per_expert_E: torch.Tensor,  # noqa: N803
    ep_size: int,
) -> torch.Tensor:
    """Copy this rank's expert counts to all peers and wait for peer counts."""
    assert _buffer_state is not None

    num_experts = num_local_tokens_per_expert_E.numel()
    copy_full_counts_to_peers(
        num_local_tokens_per_expert_E,
        _buffer_state.counts_recv_peer_buffers,
        rank=_buffer_state.group.rank(),
        ep_size=ep_size,
        num_experts=num_experts,
        dst_ptrs=_buffer_state.counts_recv_peer_ptrs,
    )
    _wait_counts_ready()
    return _buffer_state.counts_recv_buffer


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
    assert _buffer_state is not None

    rank = _buffer_state.group.rank()
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
        max_tokens_per_segment=_buffer_state.tokens_per_rank,
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
        max_tokens_per_segment=_buffer_state.tokens_per_rank,
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
    hidden_recv_buffer = _copy_rows_to_peers_and_wait_cuda(
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
    """Copy active local rows back to origin ranks as logical ``(N, D)`` rows."""
    origin_recv_buffer = _copy_rows_to_peers_and_wait_cuda(
        x_RD,
        combine_dst_ranks,
        combine_dst_rows,
        x_RD.shape[0],
        block_m=4,
        num_valid_rows=combine_num_valid_rows,
    )
    # The symmetric buffer is sized for the maximum receive capacity. Combine
    # only writes origin-rank routed rows ``[0:N]`` and downstream reductions
    # should not see capacity padding or unused trailing columns.
    return origin_recv_buffer.narrow(0, 0, num_routed_rows).narrow(1, 0, x_RD.shape[1])


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
    """Compute ``silu(gate) * up`` over active padded expert-buffer rows.

    Args:
        gate, up: ``(R_max, F)`` expert intermediate rows, where only the
            first ``R`` rows are active and ``R_max`` is the static receive
            capacity.
        active_rows: ``(1,)`` device scalar containing ``R``.

    Returns:
        ``(R_max, F)`` output. Rows ``[:R]`` are defined; rows ``[R:]`` are
        unspecified because grouped-mm offsets skip receive-capacity padding.
    """
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
    """Invert the routed-row permutation.

    Args:
        flat_indices: ``(N,)`` mapping E-major routed rows to T-major top-k
            slots.
        num_rows: ``N`` routed rows.

    Returns:
        ``(N,)`` mapping T-major top-k slots to E-major routed rows.
    """
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
    """Exchange per-expert local counts and build dispatch/combine metadata.

    Args:
        num_local_tokens_per_expert_E: ``(E,)`` int64 counts for this rank's
            local token shard over all global experts.
        num_routed_rows: ``N = T * K`` routed rows in local E-major order.
        receive_capacity: ``R_max`` static receive-buffer row capacity.
        ep_size: ``EP`` expert-parallel group size.

    Returns:
        ``dispatch_dst_ranks`` and ``dispatch_dst_rows``: ``(N,)`` maps local
        E-major routed rows to destination EP rank and destination receive row.
        ``combine_dst_ranks`` and ``combine_dst_rows``: ``(R_max,)`` maps
        active received rows back to origin EP rank and origin E-major row.
        ``combine_num_valid_rows``: ``(1,)`` device scalar active row count
        ``R``. ``tokens_per_expert``: ``(E / EP,)`` active rows per local
        expert on this rank.
    """

    # Mirrors AllToAllTokenDispatcher's count exchange: each rank starts with
    # counts for its local tokens over all global experts, then learns how many
    # tokens every peer will send to each of this rank's local experts.
    all_tokens_per_expert_RE = _copy_all_counts_to_peers_and_wait_cuda(  # noqa: N806
        num_local_tokens_per_expert_E,
        ep_size,
    )

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
    """Dispatch local token rows to expert-owner ranks.

    Args:
        dispatch_input: ``(T, D)`` local token rows.
        dispatch_dst_ranks, dispatch_dst_rows: ``(N,)`` destination rank and
            receive row for each E-major routed row.
        E_row_to_token_N: ``(N,)`` maps E-major routed rows to source token
            rows in ``dispatch_input``.
        receive_capacity: ``R_max`` static receive-buffer row capacity.

    Returns:
        ``(R_max, D)`` local expert-major receive buffer. Rows beyond the
        active received row count are padding.
    """
    num_routed_rows = E_row_to_token_N.numel()
    # This direct copy corresponds to AllToAllTokenDispatcher's token all-to-all;
    # dispatch_dst_rows already point at the post-_permute E-major layout.
    hidden_recv_buffer = _copy_rows_to_peers_and_wait_cuda(
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
    """Move expert outputs to origin ranks and reduce routed top-k rows.

    Args:
        x: ``(R_max, D)`` local expert output rows.
        dispatch_dst_ranks, dispatch_dst_rows: ``(N,)`` forward dispatch
            destinations, saved for backward.
        combine_dst_ranks, combine_dst_rows: ``(R_max,)`` origin rank and
            origin E-major row for each active received row.
        combine_num_valid_rows: ``(1,)`` device scalar active row count ``R``.
        T_row_to_E_row_N: ``(N,)`` maps T-major top-k slots to E-major routed
            rows.
        E_row_to_T_row_N: ``(N,)`` maps E-major routed rows to T-major top-k
            slots.
        routed_scores_N: ``(N,)`` T-major routed scores.
        num_tokens: ``T`` local tokens.
        top_k: ``K`` routed rows per token.

    Returns:
        ``out_TD``: ``(T, D)`` reduced token rows.
        ``routed_output_ND``: ``(N, D)`` origin-rank E-major routed rows, saved
        for routing-score gradients.
    """
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
    """Compute active-row gradients for ``minimal_async_ep::active_swiglu``.

    Args:
        grad_out, gate, up: ``(R_max, F)`` tensors from the forward op.
        active_rows: ``(1,)`` device scalar containing active row count ``R``.

    Returns:
        ``grad_gate`` and ``grad_up`` with shape ``(R_max, F)``. Rows ``[R:]``
        are unspecified because receive-capacity padding is not consumed.
    """
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
    """Move dispatched activation gradients back and reduce routed top-k rows.

    Args:
        grad_hidden: ``(R_max, D)`` gradient for expert-owner received rows.
        combine_dst_ranks, combine_dst_rows: ``(R_max,)`` origin rank and
            origin E-major row for each active received row.
        combine_num_valid_rows: ``(1,)`` device scalar active row count ``R``.
        T_row_to_E_row_N: ``(N,)`` maps T-major top-k slots to E-major routed
            rows.
        num_routed_rows: ``N`` routed rows.
        num_tokens: ``T`` local tokens.
        top_k: ``K`` routed rows per token.

    Returns:
        ``(T, D)`` gradient for the dispatch input token rows.
    """
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
    """Expand combine gradients and dispatch them back to expert-owner ranks.

    Args:
        grad_out: ``(T, D)`` gradient for reduced token rows.
        dispatch_dst_ranks, dispatch_dst_rows: ``(N,)`` destination rank and
            receive row for each E-major routed row.
        E_row_to_T_row_N: ``(N,)`` maps E-major routed rows to T-major top-k
            slots.
        routed_scores_N: ``(N,)`` T-major routed scores.
        saved_routed_output_ND: ``(N, D)`` origin-rank E-major routed output
            rows from ``combine_forward``.
        top_k: ``K`` routed rows per token.
        receive_capacity: ``R_max`` static receive-buffer row capacity.

    Returns:
        ``grad_x``: ``(R_max, D)`` gradient for expert output rows.
        ``grad_scores``: ``(N,)`` gradient for routed scores.
    """
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
