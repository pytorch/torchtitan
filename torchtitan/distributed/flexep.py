# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FlexEP primitives for constrained MoE expert parallel dispatch.

This backend is intentionally narrow: it supports the launch shape where the
EP process group is the data-parallel group and TP/CP/PP/SP are disabled.
The symmetric-memory allocation is explicit and must happen before dispatch.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch._library.opaque_object import OpaqueBase, register_opaque_type
from torch.utils._python_dispatch import _disable_current_modes

from torchtitan.distributed.flexep_kernels import (
    copy_full_counts_to_peers,
    copy_rows_to_peers,
    expand_topk_grad,
    fill_combine_metadata,
    fill_dispatch_metadata,
    invert_flat_indices,
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


class DispatchHandle(OpaqueBase):
    """Opaque wrapper for FlexEP dispatch metadata."""

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


@dataclass(frozen=True)
class _DispatchRuntimeState:
    num_routed_tokens: int
    receive_capacity: int
    dispatch_dst_ranks: torch.Tensor
    dispatch_dst_rows: torch.Tensor
    combine_dst_ranks: torch.Tensor
    combine_dst_rows: torch.Tensor
    combine_num_valid_rows: torch.Tensor
    slot_to_row: torch.Tensor
    input_is_token_ordered: bool
    num_tokens: int
    top_k: int


@dataclass
class FlexEPDispatchState:
    """FlexEP state from dispatch needed for combine."""

    handle: DispatchHandle
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
    """Initialize the process-local FlexEP symmetric-memory buffer."""
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
        "Initializing FlexEP buffer: hidden_dim=%d, max_tokens_per_rank=%d, "
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
                "FlexEP custom all-to-allv requires the symmetric-memory CUDA "
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
        raise RuntimeError("FlexEP buffer not initialized.")
    if _group_name != group_name:
        raise RuntimeError(
            f"FlexEP buffer initialized for group {_group_name!r}, "
            f"but dispatch used group {group_name!r}."
        )
    if x.device != _hidden_recv_buffers[0].device:
        raise RuntimeError(
            f"FlexEP buffer initialized on device {_hidden_recv_buffers[0].device}, "
            f"but dispatch used device {x.device}."
        )
    if x.shape[1] > _hidden_dim:
        raise RuntimeError(
            f"FlexEP buffer hidden_dim ({_hidden_dim}) is smaller than input "
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
) -> tuple[torch.Tensor, Any]:
    global _hidden_recv_buffer_index
    assert _hidden_recv_buffers is not None
    assert _hidden_recv_handles is not None
    assert _hidden_recv_peer_buffers is not None
    assert _hidden_recv_peer_ptrs is not None
    assert _group is not None

    if x.shape[0] > _max_routed_tokens:
        raise RuntimeError(
            f"FlexEP send buffer capacity ({_max_routed_tokens}) is smaller "
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
    return hidden_recv_buffer, hidden_recv_handle


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
    kernel) -- the dominant FlexEP comm cost once CPU-launch overhead is removed
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
            "FlexEP count exchange expected "
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
    state: _DispatchRuntimeState,
) -> torch.Tensor:
    hidden_recv_buffer, _ = _copy_rows_to_peers_cuda(
        x_ND,
        state.dispatch_dst_ranks,
        state.dispatch_dst_rows,
        None,
        state.num_routed_tokens,
        block_m=4,
        num_warps=8,
    )
    return hidden_recv_buffer[: state.receive_capacity, : x_ND.shape[1]]


def _combine_to_origin(
    x_RD: torch.Tensor,  # noqa: N803
    state: _DispatchRuntimeState,
) -> torch.Tensor:
    combined, _ = _copy_rows_to_peers_cuda(
        x_RD,
        state.combine_dst_ranks,
        state.combine_dst_rows,
        None,
        state.receive_capacity,
        block_m=4,
        num_valid_rows=state.combine_num_valid_rows,
    )
    return combined[: state.num_routed_tokens, : x_RD.shape[1]]


@torch.library.custom_op("flexep::dispatch", mutates_args=(), device_types="cuda")
def _dispatch_impl(
    dispatch_input: torch.Tensor,
    num_local_tokens_per_expert_E: torch.Tensor,  # noqa: N803
    source_token_indices_N: torch.Tensor,  # noqa: N803
    slot_to_row_N: torch.Tensor,  # noqa: N803
    num_tokens: int,
    ep_size: int,
    group_name: str,
    input_is_token_ordered: bool,
) -> tuple[torch.Tensor, torch.Tensor, DispatchHandle]:
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
            f"FlexEP receive buffer capacity ({_max_routed_tokens}) is smaller "
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
    hidden_recv_buffer, _ = _copy_rows_to_peers_cuda(
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
    state = _DispatchRuntimeState(
        num_routed_tokens=num_routed_tokens,
        receive_capacity=receive_capacity,
        dispatch_dst_ranks=dispatch_dst_ranks,
        dispatch_dst_rows=dispatch_dst_rows,
        combine_dst_ranks=combine_dst_ranks,
        combine_dst_rows=combine_dst_rows,
        combine_num_valid_rows=combine_num_valid_rows,
        slot_to_row=slot_to_row_N,
        input_is_token_ordered=input_is_token_ordered,
        num_tokens=num_tokens,
        top_k=top_k,
    )
    return hidden_RD, tokens_per_expert_e, DispatchHandle(value=state)


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
) -> tuple[torch.Tensor, torch.Tensor, DispatchHandle]:
    del group_name, slot_to_row_N, input_is_token_ordered
    num_local_experts = num_local_tokens_per_expert_E.shape[0] // ep_size
    top_k = source_token_indices_N.numel() // num_tokens
    out_tokens = num_tokens * ep_size * torch.sym_min(top_k, num_local_experts)
    hidden = dispatch_input.new_empty(out_tokens, dispatch_input.shape[1])
    tokens_per_expert = dispatch_input.new_empty(num_local_experts, dtype=torch.int64)
    return hidden, tokens_per_expert, DispatchHandle()


@torch.library.custom_op("flexep::combine", mutates_args=(), device_types="cuda")
def _combine_impl(
    x: torch.Tensor,
    handle: DispatchHandle,
    num_routed_tokens: int,
) -> torch.Tensor:
    """Move expert outputs back to the origin rank's routed-token order."""
    if _hidden_recv_buffers is None:
        raise RuntimeError("FlexEP buffer not initialized.")
    state = handle.value
    assert isinstance(state, _DispatchRuntimeState)
    assert state.num_routed_tokens == num_routed_tokens

    return _combine_to_origin(x, state)


@_combine_impl.register_fake
def _combine_fake(
    x: torch.Tensor,
    handle: DispatchHandle,
    num_routed_tokens: int,
) -> torch.Tensor:
    del handle
    return x.new_empty(num_routed_tokens, x.shape[1])


def _dispatch_backward(ctx, grad_hidden, grad_tpe, grad_handle):
    del grad_tpe, grad_handle
    state = ctx.dispatch_handle.value
    assert isinstance(state, _DispatchRuntimeState)

    grad_input = None
    if grad_hidden is not None:
        grad_routed_input = _combine_to_origin(grad_hidden, state)
        if state.input_is_token_ordered:
            grad_input = reduce_topk_slots(
                grad_routed_input,
                state.slot_to_row,
                None,
                num_tokens=state.num_tokens,
                top_k=state.top_k,
            )
        else:
            grad_input = grad_routed_input

    return grad_input, None, None, None, None, None, None, None


def _dispatch_setup_context(ctx, inputs, output):
    del inputs
    *_, dispatch_handle = output
    ctx.dispatch_handle = dispatch_handle


def _combine_backward(ctx, grad_combined):
    state = ctx.dispatch_handle.value
    assert isinstance(state, _DispatchRuntimeState)
    grad_x = _dispatch_to_experts(grad_combined, state)
    return grad_x, None, None


def _combine_setup_context(ctx, inputs, output):
    del output
    _, dispatch_handle, _ = inputs
    ctx.dispatch_handle = dispatch_handle


_dispatch_impl.register_autograd(
    _dispatch_backward, setup_context=_dispatch_setup_context
)
_combine_impl.register_autograd(_combine_backward, setup_context=_combine_setup_context)


@torch.library.custom_op("flexep::reduce_topk", mutates_args=(), device_types="cuda")
def _reduce_topk_impl(
    routed_output_ND: torch.Tensor,  # noqa: N803
    slot_to_row_N: torch.Tensor,  # noqa: N803
    flat_token_indices_N: torch.Tensor,  # noqa: N803
    routed_scores_N: torch.Tensor,  # noqa: N803
    has_scores: bool,
    scores_are_slot_ordered: bool,
    num_tokens: int,
    top_k: int,
) -> torch.Tensor:
    """Reduce routed top-k rows back to token order."""
    del flat_token_indices_N
    return reduce_topk_slots(
        routed_output_ND,
        slot_to_row_N,
        routed_scores_N if has_scores else None,
        num_tokens=num_tokens,
        top_k=top_k,
        scores_are_slot_ordered=scores_are_slot_ordered,
    )


@_reduce_topk_impl.register_fake
def _reduce_topk_fake(
    routed_output_ND: torch.Tensor,  # noqa: N803
    slot_to_row_N: torch.Tensor,  # noqa: N803
    flat_token_indices_N: torch.Tensor,  # noqa: N803
    routed_scores_N: torch.Tensor,  # noqa: N803
    has_scores: bool,
    scores_are_slot_ordered: bool,
    num_tokens: int,
    top_k: int,
) -> torch.Tensor:
    del (
        slot_to_row_N,
        flat_token_indices_N,
        routed_scores_N,
        has_scores,
        scores_are_slot_ordered,
        top_k,
    )
    return routed_output_ND.new_empty(num_tokens, routed_output_ND.shape[1])


def _reduce_topk_backward(ctx, grad_out):
    flat_token_indices_N, routed_scores_N, routed_output_ND = ctx.saved_tensors  # noqa: N806
    scores = routed_scores_N if ctx.has_scores else None
    grad_routed_output = expand_topk_grad(
        grad_out,
        flat_token_indices_N,
        scores,
        top_k=ctx.top_k,
        dtype=ctx.hidden_states_dtype,
        scores_are_slot_ordered=ctx.scores_are_slot_ordered,
    )
    grad_scores = None
    if ctx.has_scores and ctx.routed_scores_requires_grad:
        grad_scores = topk_scores_grad(
            routed_output_ND,
            grad_out,
            flat_token_indices_N,
            top_k=ctx.top_k,
            dtype=routed_scores_N.dtype,
            scores_are_slot_ordered=ctx.scores_are_slot_ordered,
        )

    return grad_routed_output, None, None, grad_scores, None, None, None, None


def _reduce_topk_setup_context(ctx, inputs, output):
    del output
    (
        routed_output_ND,
        _slot_to_row_N,
        flat_token_indices_N,
        routed_scores_N,
        has_scores,
        scores_are_slot_ordered,
        _num_tokens,
        top_k,
    ) = inputs
    ctx.has_scores = has_scores
    ctx.scores_are_slot_ordered = scores_are_slot_ordered
    ctx.top_k = top_k
    ctx.hidden_states_dtype = routed_output_ND.dtype
    ctx.routed_scores_requires_grad = routed_scores_N.requires_grad
    saved_routed_output_ND = (  # noqa: N806
        routed_output_ND
        if has_scores and routed_scores_N.requires_grad
        else routed_output_ND.new_empty(0)
    )
    ctx.save_for_backward(
        flat_token_indices_N,
        routed_scores_N,
        saved_routed_output_ND,
    )


_reduce_topk_impl.register_autograd(
    _reduce_topk_backward, setup_context=_reduce_topk_setup_context
)


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
) -> tuple[torch.Tensor, torch.Tensor, FlexEPDispatchState]:
    """Dispatch tokens to experts via FlexEP."""
    if _hidden_recv_buffers is None or _rendezvous_handle is None:
        raise RuntimeError("FlexEP buffer not initialized.")
    _require_initialized(group.group_name, dispatch_input)
    if num_tokens > _max_tokens_per_rank:
        raise RuntimeError(
            "FlexEP buffer max_tokens_per_rank "
            f"({_max_tokens_per_rank}) is smaller than input tokens ({num_tokens})."
        )
    if num_local_experts != _num_local_experts:
        raise RuntimeError(
            f"FlexEP buffer initialized for {_num_local_experts} local experts, "
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
            f"FlexEP buffer max_routed_tokens ({_max_routed_tokens}) is smaller "
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

    hidden, tokens_per_expert, dispatch_handle = torch.ops.flexep.dispatch(
        dispatch_input,
        num_local_tokens_per_expert_E,
        source_token_indices_N,
        slot_to_row_N,
        num_tokens,
        ep_size,
        group.group_name,
        input_is_token_ordered,
    )

    state = FlexEPDispatchState(
        handle=dispatch_handle,
        flat_token_indices=flat_token_indices_N,
        slot_to_row=slot_to_row_N,
        routed_scores=routed_scores_N,
        routed_scores_are_slot_ordered=routed_scores_are_slot_ordered,
        num_tokens=num_tokens,
        top_k=flat_token_indices_N.numel() // num_tokens,
    )
    return hidden, tokens_per_expert, state


def combine_tokens(
    hidden_states: torch.Tensor,
    state: FlexEPDispatchState,
) -> torch.Tensor:
    """Combine expert outputs back to original token order."""
    if _hidden_recv_buffers is None or _rendezvous_handle is None:
        raise RuntimeError("FlexEP buffer not initialized.")
    has_scores = state.routed_scores is not None
    routed_scores_N = (  # noqa: N806
        state.routed_scores
        if state.routed_scores is not None
        else hidden_states.new_empty(0)
    )
    routed_output_ND = torch.ops.flexep.combine(  # noqa: N806
        hidden_states,
        state.handle,
        state.flat_token_indices.numel(),
    )
    return torch.ops.flexep.reduce_topk(
        routed_output_ND,
        state.slot_to_row,
        state.flat_token_indices,
        routed_scores_N,
        has_scores,
        state.routed_scores_are_slot_ordered,
        state.num_tokens,
        state.top_k,
    )


__all__ = [
    "DispatchHandle",
    "FlexEPDispatchState",
    "combine_tokens",
    "dispatch_tokens",
    "init_buffer",
]
