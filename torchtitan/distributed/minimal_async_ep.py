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
from torch.utils._python_dispatch import _disable_current_modes

from torchtitan.distributed.minimal_async_ep_kernels import (
    active_swiglu_backward,
    active_swiglu_forward,
    copy_full_counts_to_peers,
    copy_rows_to_peers,
    expand_topk_grad,
    expert_counting_sort as _expert_counting_sort,
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
_rendezvous_handle: list[Any] | None = None
_group: dist.ProcessGroup | None = None
_group_name: str | None = None
_hidden_dim: int = 0
_max_tokens_per_rank: int = 0
_max_routed_tokens: int = 0
_num_local_experts: int = 0
_top_k: int = 0

_HIDDEN_READY_CHANNEL = 0
_COUNTS_READY_CHANNEL = 0


@dataclass
class MinimalAsyncEPDispatchMetadata:
    """MinimalAsyncEP metadata from dispatch needed for combine."""

    dispatch_dst_ranks: torch.Tensor
    dispatch_dst_rows: torch.Tensor
    combine_dst_ranks: torch.Tensor
    combine_dst_rows: torch.Tensor
    combine_num_valid_rows: torch.Tensor
    flat_token_indices: torch.Tensor
    slot_to_row: torch.Tensor
    routed_scores: torch.Tensor | None = None
    routed_scores_are_slot_ordered: bool = False
    num_tokens: int = 0
    top_k: int = 0


def init_buffer(
    group: dist.ProcessGroup,
    hidden_dim: int,
    max_tokens_per_rank: int,
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
    global _rendezvous_handle, _group, _group_name
    global _hidden_dim, _max_tokens_per_rank, _max_routed_tokens
    global _num_local_experts, _top_k

    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
    if max_tokens_per_rank <= 0:
        raise ValueError(
            f"max_tokens_per_rank must be positive, got {max_tokens_per_rank}."
        )
    if num_local_experts <= 0:
        raise ValueError(
            f"num_local_experts must be positive, got {num_local_experts}."
        )
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}.")

    device = torch.device(device)
    max_routed_tokens = (
        group.size() * max_tokens_per_rank * min(top_k, num_local_experts)
    )
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
        or _hidden_dim < hidden_dim
        or _max_tokens_per_rank < max_tokens_per_rank
        or _max_routed_tokens < max_routed_tokens
        or _num_local_experts < num_local_experts
        or _top_k < top_k
        or _hidden_recv_buffers[0].dtype != dtype
        or _hidden_recv_buffers[0].device != device
    )
    if not needs_init:
        return

    logger.info(
        "Initializing MinimalAsyncEP buffer: hidden_dim=%d, max_tokens_per_rank=%d, "
        "top_k=%d, num_local_experts=%d, ep_size=%d, max_routed_tokens=%d",
        hidden_dim,
        max_tokens_per_rank,
        top_k,
        num_local_experts,
        group.size(),
        max_routed_tokens,
    )
    with _disable_current_modes():
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
            group.size() * num_local_experts,
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
        _rendezvous_handle = [
            *_hidden_recv_handles,
            _counts_recv_handle,
        ]

    _group = group
    _group_name = group.group_name
    _hidden_dim = hidden_dim
    _max_tokens_per_rank = max_tokens_per_rank
    _max_routed_tokens = max_routed_tokens
    _num_local_experts = num_local_experts
    _top_k = top_k
    _hidden_recv_buffer_index = 0


def _require_initialized(
    group_name: str,
    x: torch.Tensor,
) -> None:
    if (
        _hidden_recv_buffers is None
        or _hidden_recv_handles is None
        or _hidden_recv_peer_buffers is None
        or _hidden_recv_peer_ptrs is None
        or _counts_recv_buffer is None
        or _counts_recv_handle is None
        or _counts_recv_peer_buffers is None
        or _counts_recv_peer_ptrs is None
        or _rendezvous_handle is None
    ):
        raise RuntimeError("MinimalAsyncEP buffer not initialized.")
    if _group_name != group_name:
        raise RuntimeError(
            f"MinimalAsyncEP buffer initialized for group {_group_name!r}, "
            f"but dispatch used group {group_name!r}."
        )
    if x.device != _hidden_recv_buffers[0].device:
        raise RuntimeError(
            "MinimalAsyncEP buffer initialized on device "
            f"{_hidden_recv_buffers[0].device}, but dispatch used device {x.device}."
        )
    if x.shape[1] > _hidden_dim:
        raise RuntimeError(
            f"MinimalAsyncEP buffer hidden_dim ({_hidden_dim}) is smaller than input "
            f"hidden_dim ({x.shape[1]})."
        )


def _copy_rows_to_peers_cuda(
    x: torch.Tensor,
    dst_ranks: torch.Tensor,
    dst_rows: torch.Tensor,
    valid_rows: torch.Tensor | None,
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

    if x.shape[0] > _max_routed_tokens:
        raise RuntimeError(
            f"MinimalAsyncEP send buffer capacity ({_max_routed_tokens}) is smaller "
            f"than input rows ({x.shape[0]})."
        )

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
        valid_rows,
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

    num_experts = ep_size * _num_local_experts
    if num_local_tokens_per_expert_E.numel() != num_experts:
        raise RuntimeError(
            "MinimalAsyncEP count exchange expected "
            f"{num_experts} counts, got {num_local_tokens_per_expert_E.numel()}."
        )

    copy_full_counts_to_peers(
        num_local_tokens_per_expert_E,
        _counts_recv_peer_buffers,
        rank=_group.rank(),
        ep_size=ep_size,
        num_experts=num_experts,
        dst_ptrs=_counts_recv_peer_ptrs,
    )
    return _counts_recv_buffer[:ep_size, :num_experts]


def _compute_direct_metadata(
    num_local_tokens_per_expert_E: torch.Tensor,  # noqa: N803
    all_tokens_per_expert_RE: torch.Tensor,  # noqa: N803
    num_routed_tokens: int,
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
        num_routed_tokens=num_routed_tokens,
        num_local_experts=num_local_experts,
        max_tokens_per_segment=_max_tokens_per_rank,
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
        max_tokens_per_segment=_max_tokens_per_rank,
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
    num_routed_tokens: int,
    receive_capacity: int,
) -> torch.Tensor:
    hidden_recv_buffer = _copy_rows_to_peers_cuda(
        x_ND,
        dispatch_dst_ranks,
        dispatch_dst_rows,
        None,
        num_routed_tokens,
        block_m=4,
        num_warps=8,
    )
    return hidden_recv_buffer[:receive_capacity, : x_ND.shape[1]]


def _combine_to_origin(
    x_RD: torch.Tensor,  # noqa: N803
    combine_dst_ranks: torch.Tensor,
    combine_dst_rows: torch.Tensor,
    combine_num_valid_rows: torch.Tensor,
    num_routed_tokens: int,
) -> torch.Tensor:
    combined = _copy_rows_to_peers_cuda(
        x_RD,
        combine_dst_ranks,
        combine_dst_rows,
        None,
        x_RD.shape[0],
        block_m=4,
        num_valid_rows=combine_num_valid_rows,
    )
    return combined[:num_routed_tokens, : x_RD.shape[1]]


@torch.library.custom_op(
    "minimal_async_ep::active_swiglu",
    mutates_args=(),
    device_types="cuda",
)
def _active_swiglu_impl(
    gate: torch.Tensor,
    up: torch.Tensor,
    active_rows: torch.Tensor,
) -> torch.Tensor:
    """Compute ``silu(gate) * up`` only for active rows in padded expert buffers."""
    return active_swiglu_forward(gate, up, active_rows)


@_active_swiglu_impl.register_fake
def _active_swiglu_fake(
    gate: torch.Tensor,
    up: torch.Tensor,
    active_rows: torch.Tensor,
) -> torch.Tensor:
    del up, active_rows
    return gate.new_empty(gate.shape)


@torch.library.custom_op(
    "minimal_async_ep::expert_counting_sort",
    mutates_args=(),
    device_types="cuda",
)
def _expert_counting_sort_impl(
    topk_expert_ids: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group flattened top-k slots by expert id."""
    return _expert_counting_sort(topk_expert_ids, num_experts=num_experts)


@_expert_counting_sort_impl.register_fake
def _expert_counting_sort_fake(
    topk_expert_ids: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_ids = topk_expert_ids.numel()
    counts = topk_expert_ids.new_empty(num_experts, dtype=torch.int64)
    token_indices = topk_expert_ids.new_empty(num_ids, dtype=torch.int64)
    flat_indices = topk_expert_ids.new_empty(num_ids, dtype=torch.int64)
    return counts, token_indices, flat_indices


@torch.library.custom_op(
    "minimal_async_ep::invert_flat_indices",
    mutates_args=(),
    device_types="cuda",
)
def _invert_flat_indices_impl(
    flat_indices: torch.Tensor,
    num_rows: int,
) -> torch.Tensor:
    """Invert expert-sorted row indices back to original top-k slot order."""
    return _invert_flat_indices(flat_indices, num_rows=num_rows)


@_invert_flat_indices_impl.register_fake
def _invert_flat_indices_fake(
    flat_indices: torch.Tensor,
    num_rows: int,
) -> torch.Tensor:
    return flat_indices.new_empty(num_rows)


@torch.library.custom_op(
    "minimal_async_ep::dispatch",
    mutates_args=(),
    device_types="cuda",
)
def _dispatch_impl(
    dispatch_input: torch.Tensor,
    num_local_tokens_per_expert_E: torch.Tensor,  # noqa: N803
    source_token_indices_N: torch.Tensor,  # noqa: N803
    slot_to_row_N: torch.Tensor,  # noqa: N803
    num_tokens: int,
    ep_size: int,
    group_name: str,
    input_is_token_ordered: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Dispatch tokens to local experts through the EP process group."""
    _require_initialized(group_name, dispatch_input)
    num_experts = num_local_tokens_per_expert_E.numel()
    if num_experts % ep_size != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})."
        )

    num_local_tokens_per_expert_E = num_local_tokens_per_expert_E.to(
        torch.int64
    ).contiguous()
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}.")
    num_routed_tokens = source_token_indices_N.numel()
    if num_routed_tokens % num_tokens != 0:
        raise ValueError(
            "routed input rows must be divisible by num_tokens. Got "
            f"{num_routed_tokens} rows and {num_tokens} tokens."
        )
    if slot_to_row_N.numel() != num_routed_tokens:
        raise ValueError(
            "slot_to_row length must match routed rows. Got "
            f"{slot_to_row_N.numel()} and {num_routed_tokens}."
        )
    if source_token_indices_N.dtype != torch.int64:
        raise ValueError("source_token_indices_N must be int64.")
    if source_token_indices_N.device != dispatch_input.device:
        raise ValueError("source_token_indices_N must match dispatch input device.")
    if slot_to_row_N.device != dispatch_input.device:
        raise ValueError("slot_to_row_N must match dispatch input device.")
    top_k = num_routed_tokens // num_tokens
    if input_is_token_ordered:
        if dispatch_input.shape[0] != num_tokens:
            raise ValueError(
                "token-ordered dispatch input rows must match num_tokens. Got "
                f"{dispatch_input.shape[0]} and {num_tokens}."
            )
    elif dispatch_input.shape[0] != num_routed_tokens:
        raise ValueError(
            "routed dispatch input rows must match routed rows. Got "
            f"{dispatch_input.shape[0]} and {num_routed_tokens}."
        )
    slot_size = num_tokens * min(top_k, num_experts // ep_size)
    receive_capacity = ep_size * slot_size
    if receive_capacity > _max_routed_tokens:
        raise RuntimeError(
            f"MinimalAsyncEP receive buffer capacity ({_max_routed_tokens}) is smaller "
            f"than required receive capacity ({receive_capacity})."
        )

    all_tokens_per_expert_RE = _copy_all_counts_to_peers_cuda(  # noqa: N806
        num_local_tokens_per_expert_E,
        ep_size,
    )
    _wait_counts_ready()
    (
        dispatch_dst_ranks,
        dispatch_dst_rows,
        combine_dst_ranks,
        combine_dst_rows,
        combine_num_valid_rows,
        tokens_per_expert_e,
    ) = _compute_direct_metadata(
        num_local_tokens_per_expert_E,
        all_tokens_per_expert_RE,
        num_routed_tokens,
        receive_capacity,
        ep_size,
    )
    hidden_recv_buffer = _copy_rows_to_peers_cuda(
        dispatch_input,
        dispatch_dst_ranks,
        dispatch_dst_rows,
        None,
        num_routed_tokens,
        block_m=4,
        num_warps=8,
        src_rows=source_token_indices_N if input_is_token_ordered else None,
    )
    hidden_RD = hidden_recv_buffer[:receive_capacity, : dispatch_input.shape[1]]
    return (
        hidden_RD,
        tokens_per_expert_e,
        dispatch_dst_ranks,
        dispatch_dst_rows,
        combine_dst_ranks,
        combine_dst_rows,
        combine_num_valid_rows,
    )


@_dispatch_impl.register_fake
def _dispatch_fake(
    dispatch_input: torch.Tensor,
    num_local_tokens_per_expert_E: torch.Tensor,  # noqa: N803
    source_token_indices_N: torch.Tensor,  # noqa: N803
    slot_to_row_N: torch.Tensor,  # noqa: N803
    num_tokens: int,
    ep_size: int,
    group_name: str,
    input_is_token_ordered: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    del group_name, slot_to_row_N, input_is_token_ordered
    num_local_experts = num_local_tokens_per_expert_E.shape[0] // ep_size
    top_k = source_token_indices_N.numel() // num_tokens
    out_tokens = num_tokens * ep_size * torch.sym_min(top_k, num_local_experts)
    hidden = dispatch_input.new_empty(out_tokens, dispatch_input.shape[1])
    tokens_per_expert = dispatch_input.new_empty(num_local_experts, dtype=torch.int64)
    num_routed_tokens = source_token_indices_N.numel()
    dispatch_dst_ranks = source_token_indices_N.new_empty(num_routed_tokens)
    dispatch_dst_rows = source_token_indices_N.new_empty(num_routed_tokens)
    combine_dst_ranks = source_token_indices_N.new_empty(out_tokens)
    combine_dst_rows = source_token_indices_N.new_empty(out_tokens)
    combine_num_valid_rows = source_token_indices_N.new_empty(1)
    return (
        hidden,
        tokens_per_expert,
        dispatch_dst_ranks,
        dispatch_dst_rows,
        combine_dst_ranks,
        combine_dst_rows,
        combine_num_valid_rows,
    )


@torch.library.custom_op(
    "minimal_async_ep::combine",
    mutates_args=(),
    device_types="cuda",
)
def _combine_impl(
    x: torch.Tensor,
    dispatch_dst_ranks: torch.Tensor,
    dispatch_dst_rows: torch.Tensor,
    combine_dst_ranks: torch.Tensor,
    combine_dst_rows: torch.Tensor,
    combine_num_valid_rows: torch.Tensor,
    slot_to_row_N: torch.Tensor,  # noqa: N803
    flat_token_indices_N: torch.Tensor,  # noqa: N803
    routed_scores_N: torch.Tensor,  # noqa: N803
    has_scores: bool,
    scores_are_slot_ordered: bool,
    num_tokens: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move expert outputs to origin ranks and reduce routed top-k rows."""
    if _hidden_recv_buffers is None:
        raise RuntimeError("MinimalAsyncEP buffer not initialized.")
    del dispatch_dst_ranks, dispatch_dst_rows

    routed_output_ND = _combine_to_origin(  # noqa: N806
        x,
        combine_dst_ranks,
        combine_dst_rows,
        combine_num_valid_rows,
        flat_token_indices_N.numel(),
    )
    out_TD = reduce_topk_slots(  # noqa: N806
        routed_output_ND,
        slot_to_row_N,
        routed_scores_N if has_scores else None,
        num_tokens=num_tokens,
        top_k=top_k,
        scores_are_slot_ordered=scores_are_slot_ordered,
    )
    return out_TD, routed_output_ND


@_combine_impl.register_fake
def _combine_fake(
    x: torch.Tensor,
    dispatch_dst_ranks: torch.Tensor,
    dispatch_dst_rows: torch.Tensor,
    combine_dst_ranks: torch.Tensor,
    combine_dst_rows: torch.Tensor,
    combine_num_valid_rows: torch.Tensor,
    slot_to_row_N: torch.Tensor,  # noqa: N803
    flat_token_indices_N: torch.Tensor,  # noqa: N803
    routed_scores_N: torch.Tensor,  # noqa: N803
    has_scores: bool,
    scores_are_slot_ordered: bool,
    num_tokens: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del (
        dispatch_dst_ranks,
        dispatch_dst_rows,
        combine_dst_ranks,
        combine_dst_rows,
        combine_num_valid_rows,
        slot_to_row_N,
        routed_scores_N,
        has_scores,
        scores_are_slot_ordered,
        top_k,
    )
    return (
        x.new_empty(num_tokens, x.shape[1]),
        x.new_empty(flat_token_indices_N.numel(), x.shape[1]),
    )


def _active_swiglu_backward(ctx, grad_out):
    gate, up, active_rows = ctx.saved_tensors
    grad_gate, grad_up = active_swiglu_backward(grad_out, gate, up, active_rows)
    return grad_gate, grad_up, None


def _active_swiglu_setup_context(ctx, inputs, output):
    del output
    gate, up, active_rows = inputs
    ctx.save_for_backward(gate, up, active_rows)


def _dispatch_backward(
    ctx,
    grad_hidden,
    grad_tpe,
    grad_dispatch_dst_ranks,
    grad_dispatch_dst_rows,
    grad_combine_dst_ranks,
    grad_combine_dst_rows,
    grad_combine_num_valid_rows,
):
    del (
        grad_tpe,
        grad_dispatch_dst_ranks,
        grad_dispatch_dst_rows,
        grad_combine_dst_ranks,
        grad_combine_dst_rows,
        grad_combine_num_valid_rows,
    )
    (
        combine_dst_ranks,
        combine_dst_rows,
        combine_num_valid_rows,
        slot_to_row_N,
    ) = ctx.saved_tensors

    grad_input = None
    if grad_hidden is not None:
        grad_routed_input = _combine_to_origin(
            grad_hidden,
            combine_dst_ranks,
            combine_dst_rows,
            combine_num_valid_rows,
            ctx.num_routed_tokens,
        )
        if ctx.input_is_token_ordered:
            grad_input = reduce_topk_slots(
                grad_routed_input,
                slot_to_row_N,
                None,
                num_tokens=ctx.num_tokens,
                top_k=ctx.top_k,
            )
        else:
            grad_input = grad_routed_input

    return grad_input, None, None, None, None, None, None, None


def _dispatch_setup_context(ctx, inputs, output):
    (
        _dispatch_input,
        _num_local_tokens_per_expert_E,
        source_token_indices_N,
        slot_to_row_N,
        num_tokens,
        _ep_size,
        _group_name,
        input_is_token_ordered,
    ) = inputs
    (
        _hidden,
        _tokens_per_expert,
        _dispatch_dst_ranks,
        _dispatch_dst_rows,
        combine_dst_ranks,
        combine_dst_rows,
        combine_num_valid_rows,
    ) = output
    ctx.num_routed_tokens = source_token_indices_N.numel()
    ctx.num_tokens = num_tokens
    ctx.top_k = source_token_indices_N.numel() // num_tokens
    ctx.input_is_token_ordered = input_is_token_ordered
    ctx.save_for_backward(
        combine_dst_ranks,
        combine_dst_rows,
        combine_num_valid_rows,
        slot_to_row_N,
    )


def _combine_backward(ctx, grad_out, grad_routed_output_extra):
    (
        dispatch_dst_ranks,
        dispatch_dst_rows,
        flat_token_indices_N,
        routed_scores_N,
        saved_routed_output_ND,
    ) = ctx.saved_tensors
    scores = routed_scores_N if ctx.has_scores else None
    grad_routed_output = None
    grad_scores = None

    if grad_out is not None:
        grad_routed_output = expand_topk_grad(
            grad_out,
            flat_token_indices_N,
            scores,
            top_k=ctx.top_k,
            dtype=ctx.hidden_states_dtype,
            scores_are_slot_ordered=ctx.scores_are_slot_ordered,
        )
        if ctx.has_scores and ctx.routed_scores_requires_grad:
            grad_scores = topk_scores_grad(
                saved_routed_output_ND,
                grad_out,
                flat_token_indices_N,
                top_k=ctx.top_k,
                dtype=routed_scores_N.dtype,
                scores_are_slot_ordered=ctx.scores_are_slot_ordered,
            )

    if grad_routed_output_extra is not None:
        grad_routed_output = (
            grad_routed_output_extra
            if grad_routed_output is None
            else grad_routed_output + grad_routed_output_extra
        )

    grad_x = None
    if grad_routed_output is not None:
        grad_x = _dispatch_to_experts(
            grad_routed_output,
            dispatch_dst_ranks,
            dispatch_dst_rows,
            flat_token_indices_N.numel(),
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
        None,
        None,
    )


def _combine_setup_context(ctx, inputs, output):
    (
        x,
        dispatch_dst_ranks,
        dispatch_dst_rows,
        _combine_dst_ranks,
        _combine_dst_rows,
        _combine_num_valid_rows,
        _slot_to_row_N,
        flat_token_indices_N,
        routed_scores_N,
        has_scores,
        scores_are_slot_ordered,
        _num_tokens,
        top_k,
    ) = inputs
    _, routed_output_ND = output
    ctx.has_scores = has_scores
    ctx.scores_are_slot_ordered = scores_are_slot_ordered
    ctx.top_k = top_k
    ctx.hidden_states_dtype = x.dtype
    ctx.receive_capacity = x.shape[0]
    ctx.routed_scores_requires_grad = routed_scores_N.requires_grad
    saved_routed_output_ND = (  # noqa: N806
        routed_output_ND
        if has_scores and routed_scores_N.requires_grad
        else routed_output_ND.new_empty(0)
    )
    ctx.save_for_backward(
        dispatch_dst_ranks,
        dispatch_dst_rows,
        flat_token_indices_N,
        routed_scores_N,
        saved_routed_output_ND,
    )


_active_swiglu_impl.register_autograd(
    _active_swiglu_backward, setup_context=_active_swiglu_setup_context
)
_dispatch_impl.register_autograd(
    _dispatch_backward, setup_context=_dispatch_setup_context
)
_combine_impl.register_autograd(_combine_backward, setup_context=_combine_setup_context)


def active_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    active_rows: torch.Tensor,
) -> torch.Tensor:
    """Compute ``silu(gate) * up`` while skipping padded MinimalAsyncEP rows."""
    if not gate.is_cuda:
        return torch.nn.functional.silu(gate) * up
    return torch.ops.minimal_async_ep.active_swiglu(gate, up, active_rows)


def expert_counting_sort(
    topk_expert_ids: torch.Tensor,
    *,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group flattened top-k slots by expert id without exposing raw Triton calls."""
    if not topk_expert_ids.is_cuda:
        return _expert_counting_sort(topk_expert_ids, num_experts=num_experts)
    return torch.ops.minimal_async_ep.expert_counting_sort(
        topk_expert_ids,
        num_experts,
    )


def invert_flat_indices(
    flat_indices: torch.Tensor,
    *,
    num_rows: int,
) -> torch.Tensor:
    """Invert flat top-k slot indices without exposing raw Triton calls."""
    if not flat_indices.is_cuda:
        return _invert_flat_indices(flat_indices, num_rows=num_rows)
    return torch.ops.minimal_async_ep.invert_flat_indices(flat_indices, num_rows)


def dispatch_tokens(
    dispatch_input: torch.Tensor,
    num_local_tokens_per_expert_E: torch.Tensor,  # noqa: N803
    source_token_indices_N: torch.Tensor,  # noqa: N803
    flat_token_indices_N: torch.Tensor,  # noqa: N803
    routed_scores_N: torch.Tensor | None,  # noqa: N803
    num_tokens: int,
    num_local_experts: int,
    group: dist.ProcessGroup,
    *,
    input_is_token_ordered: bool = False,
    routed_scores_are_slot_ordered: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, MinimalAsyncEPDispatchMetadata]:
    """Dispatch tokens to experts via MinimalAsyncEP."""
    if _hidden_recv_buffers is None or _rendezvous_handle is None:
        raise RuntimeError("MinimalAsyncEP buffer not initialized.")
    _require_initialized(group.group_name, dispatch_input)
    if num_tokens > _max_tokens_per_rank:
        raise RuntimeError(
            "MinimalAsyncEP buffer max_tokens_per_rank "
            f"({_max_tokens_per_rank}) is smaller than input tokens ({num_tokens})."
        )
    if num_local_experts != _num_local_experts:
        raise RuntimeError(
            "MinimalAsyncEP buffer initialized for "
            f"{_num_local_experts} local experts, "
            f"but dispatch used {num_local_experts}."
        )
    ep_size = group.size()
    num_experts = num_local_tokens_per_expert_E.numel()
    if num_experts % ep_size != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})."
        )
    if flat_token_indices_N.numel() > _max_routed_tokens:
        raise RuntimeError(
            f"MinimalAsyncEP buffer max_routed_tokens ({_max_routed_tokens}) "
            "is smaller "
            f"than routed input rows ({flat_token_indices_N.numel()})."
        )
    if source_token_indices_N.numel() != flat_token_indices_N.numel():
        raise ValueError(
            "source_token_indices_N and flat_token_indices_N must have the same "
            f"length, got {source_token_indices_N.numel()} and "
            f"{flat_token_indices_N.numel()}."
        )
    if routed_scores_N is not None and routed_scores_N.dtype != dispatch_input.dtype:
        routed_scores_N = routed_scores_N.to(dispatch_input.dtype)

    slot_to_row_N = invert_flat_indices(  # noqa: N806
        flat_token_indices_N,
        num_rows=flat_token_indices_N.numel(),
    )

    (
        hidden,
        tokens_per_expert,
        dispatch_dst_ranks,
        dispatch_dst_rows,
        combine_dst_ranks,
        combine_dst_rows,
        combine_num_valid_rows,
    ) = torch.ops.minimal_async_ep.dispatch(
        dispatch_input,
        num_local_tokens_per_expert_E,
        source_token_indices_N,
        slot_to_row_N,
        num_tokens,
        ep_size,
        group.group_name,
        input_is_token_ordered,
    )

    metadata = MinimalAsyncEPDispatchMetadata(
        dispatch_dst_ranks=dispatch_dst_ranks,
        dispatch_dst_rows=dispatch_dst_rows,
        combine_dst_ranks=combine_dst_ranks,
        combine_dst_rows=combine_dst_rows,
        combine_num_valid_rows=combine_num_valid_rows,
        flat_token_indices=flat_token_indices_N,
        slot_to_row=slot_to_row_N,
        routed_scores=routed_scores_N,
        routed_scores_are_slot_ordered=routed_scores_are_slot_ordered,
        num_tokens=num_tokens,
        top_k=flat_token_indices_N.numel() // num_tokens,
    )
    return hidden, tokens_per_expert, metadata


def combine_tokens(
    hidden_states: torch.Tensor,
    metadata: MinimalAsyncEPDispatchMetadata,
) -> torch.Tensor:
    """Combine expert outputs back to original token order."""
    if _hidden_recv_buffers is None or _rendezvous_handle is None:
        raise RuntimeError("MinimalAsyncEP buffer not initialized.")
    has_scores = metadata.routed_scores is not None
    routed_scores_N = (  # noqa: N806
        metadata.routed_scores
        if metadata.routed_scores is not None
        else hidden_states.new_empty(0)
    )
    out_TD, _routed_output_ND = torch.ops.minimal_async_ep.combine(  # noqa: N806
        hidden_states,
        metadata.dispatch_dst_ranks,
        metadata.dispatch_dst_rows,
        metadata.combine_dst_ranks,
        metadata.combine_dst_rows,
        metadata.combine_num_valid_rows,
        metadata.slot_to_row,
        metadata.flat_token_indices,
        routed_scores_N,
        has_scores,
        metadata.routed_scores_are_slot_ordered,
        metadata.num_tokens,
        metadata.top_k,
    )
    return out_TD


__all__ = [
    "MinimalAsyncEPDispatchMetadata",
    "active_swiglu",
    "combine_tokens",
    "dispatch_tokens",
    "expert_counting_sort",
    "init_buffer",
    "invert_flat_indices",
]
