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
from torch.distributed._functional_collectives import all_to_all_single
from torch.utils._python_dispatch import _disable_current_modes

from torchtitan.distributed.flexep_kernels import copy_peer
from torchtitan.ops.scatter_add import deterministic_scatter_add
from torchtitan.tools.logging import logger


_buffer: torch.Tensor | None = None
_hidden_recv_buffer: torch.Tensor | None = None
_input_splits_buffer: torch.Tensor | None = None
_all_input_splits_buffer: torch.Tensor | None = None
_hidden_recv_handle: Any = None
_input_splits_handle: Any = None
_rendezvous_handle: list[Any] | None = None
_group: dist.ProcessGroup | None = None
_group_name: str | None = None
_hidden_dim: int = 0
_max_tokens_per_rank: int = 0
_max_routed_tokens: int = 0
_num_local_experts: int = 0
_top_k: int = 0


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
    input_splits: torch.Tensor
    output_splits: torch.Tensor
    permuted_indices: torch.Tensor


@dataclass
class FlexEPDispatchState:
    """FlexEP state from dispatch needed for combine."""

    handle: DispatchHandle
    token_indices: torch.Tensor
    routed_scores: torch.Tensor | None = None
    num_tokens: int = 0


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
    global _buffer, _hidden_recv_buffer
    global _input_splits_buffer, _all_input_splits_buffer
    global _hidden_recv_handle, _input_splits_handle
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
        _buffer is None
        or _hidden_recv_buffer is None
        or _input_splits_buffer is None
        or _all_input_splits_buffer is None
        or _group != group
        or _hidden_dim < hidden_dim
        or _max_tokens_per_rank < max_tokens_per_rank
        or _max_routed_tokens < max_routed_tokens
        or _num_local_experts < num_local_experts
        or _top_k < top_k
        or _buffer.dtype != dtype
        or _buffer.device != device
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

        _buffer = symm_mem.empty(
            max_routed_tokens,
            hidden_dim,
            dtype=dtype,
            device=device,
        )
        _hidden_recv_buffer = symm_mem.empty(
            max_routed_tokens,
            hidden_dim,
            dtype=dtype,
            device=device,
        )
        _input_splits_buffer = symm_mem.empty(
            group.size(),
            dtype=torch.int64,
            device=device,
        )
        _all_input_splits_buffer = torch.empty(
            group.size(),
            group.size(),
            dtype=torch.int64,
            device=device,
        )
        buffer_handle = symm_mem.rendezvous(_buffer, group)
        _hidden_recv_handle = symm_mem.rendezvous(_hidden_recv_buffer, group)
        _input_splits_handle = symm_mem.rendezvous(_input_splits_buffer, group)
        _rendezvous_handle = [
            buffer_handle,
            _hidden_recv_handle,
            _input_splits_handle,
        ]

    _group = group
    _group_name = group.group_name
    _hidden_dim = hidden_dim
    _max_tokens_per_rank = max_tokens_per_rank
    _max_routed_tokens = max_routed_tokens
    _num_local_experts = num_local_experts
    _top_k = top_k


def _require_initialized(
    group_name: str,
    x: torch.Tensor,
) -> None:
    if (
        _buffer is None
        or _hidden_recv_buffer is None
        or _input_splits_buffer is None
        or _all_input_splits_buffer is None
        or _hidden_recv_handle is None
        or _input_splits_handle is None
        or _rendezvous_handle is None
    ):
        raise RuntimeError("FlexEP buffer not initialized.")
    if _group_name != group_name:
        raise RuntimeError(
            f"FlexEP buffer initialized for group {_group_name!r}, "
            f"but dispatch used group {group_name!r}."
        )
    if x.device != _buffer.device:
        raise RuntimeError(
            f"FlexEP buffer initialized on device {_buffer.device}, "
            f"but dispatch used device {x.device}."
        )
    if x.shape[1] > _hidden_dim:
        raise RuntimeError(
            f"FlexEP buffer hidden_dim ({_hidden_dim}) is smaller than input "
            f"hidden_dim ({x.shape[1]})."
        )


def _wait_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.ops._c10d_functional.wait_tensor(x)


def _exchange_input_splits(input_splits: torch.Tensor) -> torch.Tensor:
    assert _input_splits_buffer is not None
    assert _all_input_splits_buffer is not None
    assert _input_splits_handle is not None
    assert _group is not None

    group_size = _group.size()
    rank = _group.rank()

    # Ensure no rank overwrites its split vector while a peer may still read it
    # from the previous all-to-allv.
    _input_splits_handle.barrier()
    _input_splits_buffer.copy_(input_splits.to(torch.int64))
    _input_splits_handle.barrier()

    for peer in range(group_size):
        peer_splits = _input_splits_handle.get_buffer(
            peer,
            _input_splits_buffer.shape,
            _input_splits_buffer.dtype,
        )
        _all_input_splits_buffer[peer].copy_(peer_splits)

    _input_splits_handle.barrier()
    return _all_input_splits_buffer[:, rank].contiguous()


def _all_to_allv_cuda(
    x: torch.Tensor,
    input_splits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert _buffer is not None
    assert _hidden_recv_buffer is not None
    assert _all_input_splits_buffer is not None
    assert _hidden_recv_handle is not None
    assert _group is not None

    if x.shape[0] > _max_routed_tokens:
        raise RuntimeError(
            f"FlexEP send buffer capacity ({_max_routed_tokens}) is smaller "
            f"than input rows ({x.shape[0]})."
        )

    _buffer[: x.shape[0], : x.shape[1]].copy_(x)
    output_splits = _exchange_input_splits(input_splits)

    rank = _group.rank()
    group_size = _group.size()
    for peer in range(group_size):
        peer_recv_buffer = _hidden_recv_handle.get_buffer(
            peer,
            _hidden_recv_buffer.shape,
            _hidden_recv_buffer.dtype,
        )
        copy_peer(
            _buffer,
            peer_recv_buffer,
            _all_input_splits_buffer,
            peer=peer,
            rank=rank,
            ep_size=group_size,
            num_rows=x.shape[0],
            num_cols=x.shape[1],
        )

    _hidden_recv_handle.barrier()
    return _hidden_recv_buffer, output_splits


def _permute_to_expert_major(
    x: torch.Tensor,
    num_global_tokens_per_local_expert_E: torch.Tensor,  # noqa: N803
    ep_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_local_experts = num_global_tokens_per_local_expert_E.shape[0] // ep_size
    device = num_global_tokens_per_local_expert_E.device
    capacity = x.shape[0]

    token_count_matrix = num_global_tokens_per_local_expert_E.view(
        ep_size, num_local_experts
    )
    input_starts = (
        num_global_tokens_per_local_expert_E.cumsum(0)
        - num_global_tokens_per_local_expert_E
    ).view(ep_size, num_local_experts)

    segment_lens = token_count_matrix.t().reshape(-1)
    input_starts = input_starts.t().reshape(-1)

    output_ends = segment_lens.cumsum(0)
    output_starts = output_ends - segment_lens
    positions = torch.arange(capacity, device=device)
    # Keep the dispatch output fixed-size so the CUDA token-count total never
    # has to become a host-side tensor dimension. Rows at positions >= total
    # are padding; grouped_mm ignores them because expert offsets end at total.
    segment_ids = torch.searchsorted(output_ends, positions, right=True)
    segment_ids = segment_ids.clamp(max=segment_lens.shape[0] - 1)
    permuted_indices = (
        input_starts[segment_ids]
        + positions
        - output_starts[segment_ids]
    )
    total = output_ends[-1]
    permuted_indices = torch.where(positions < total, permuted_indices, positions)

    num_global_tokens_per_local_expert_e = token_count_matrix.sum(0)
    return (
        x[permuted_indices, ...],
        permuted_indices,
        num_global_tokens_per_local_expert_e,
    )


def _unpermute_from_expert_major(
    x: torch.Tensor,
    shape: tuple[int, ...],
    permuted_indices: torch.Tensor,
) -> torch.Tensor:
    out = x.new_empty(shape)
    out[permuted_indices, ...] = x
    return out


def _dispatch_to_experts(
    x_ND: torch.Tensor,  # noqa: N803
    state: _DispatchRuntimeState,
) -> torch.Tensor:
    rank_major, _ = _all_to_allv_cuda(
        x_ND,
        state.input_splits,
    )
    return rank_major[state.permuted_indices, ...]


def _combine_to_origin(
    x_RD: torch.Tensor,  # noqa: N803
    state: _DispatchRuntimeState,
) -> torch.Tensor:
    rank_major = _unpermute_from_expert_major(x_RD, x_RD.shape, state.permuted_indices)
    combined, _ = _all_to_allv_cuda(
        rank_major,
        state.output_splits,
    )
    return combined[: state.num_routed_tokens, : rank_major.shape[1]]


@torch.library.custom_op("flexep::dispatch", mutates_args=(), device_types="cuda")
def _dispatch_impl(
    routed_input_ND: torch.Tensor,  # noqa: N803
    num_local_tokens_per_expert_E: torch.Tensor,  # noqa: N803
    num_tokens: int,
    ep_size: int,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor, DispatchHandle]:
    """Dispatch tokens to local experts through the EP process group."""
    _require_initialized(group_name, routed_input_ND)
    num_experts = num_local_tokens_per_expert_E.numel()
    if num_experts % ep_size != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})."
        )

    group = dist.distributed_c10d._resolve_process_group(group_name)
    num_local_tokens_per_expert_E = num_local_tokens_per_expert_E.to(
        torch.int64
    ).contiguous()
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}.")
    if routed_input_ND.shape[0] % num_tokens != 0:
        raise ValueError(
            "routed input rows must be divisible by num_tokens. Got "
            f"{routed_input_ND.shape[0]} rows and {num_tokens} tokens."
        )
    top_k = routed_input_ND.shape[0] // num_tokens
    receive_capacity = ep_size * num_tokens * min(top_k, num_experts // ep_size)
    if receive_capacity > _max_routed_tokens:
        raise RuntimeError(
            f"FlexEP receive buffer capacity ({_max_routed_tokens}) is smaller "
            f"than required receive capacity ({receive_capacity})."
        )

    num_global_tokens_per_local_expert_E = all_to_all_single(
        num_local_tokens_per_expert_E,
        None,
        None,
        group,
    )
    num_global_tokens_per_local_expert_E = _wait_tensor(
        num_global_tokens_per_local_expert_E
    )

    input_splits = num_local_tokens_per_expert_E.view(ep_size, -1).sum(dim=1)

    hidden_rank_major, output_splits = _all_to_allv_cuda(
        routed_input_ND,
        input_splits,
    )
    hidden_rank_major = hidden_rank_major[
        :receive_capacity, : routed_input_ND.shape[1]
    ]

    hidden_RD, permuted_indices, tokens_per_expert_e = _permute_to_expert_major(
        hidden_rank_major,
        num_global_tokens_per_local_expert_E,
        ep_size,
    )

    state = _DispatchRuntimeState(
        num_routed_tokens=routed_input_ND.shape[0],
        input_splits=input_splits.clone(),
        output_splits=output_splits,
        permuted_indices=permuted_indices,
    )
    return hidden_RD, tokens_per_expert_e, DispatchHandle(value=state)


@_dispatch_impl.register_fake
def _dispatch_fake(
    routed_input_ND: torch.Tensor,  # noqa: N803
    num_local_tokens_per_expert_E: torch.Tensor,  # noqa: N803
    num_tokens: int,
    ep_size: int,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor, DispatchHandle]:
    del group_name
    num_local_experts = num_local_tokens_per_expert_E.shape[0] // ep_size
    top_k = routed_input_ND.shape[0] // num_tokens
    out_tokens = num_tokens * ep_size * torch.sym_min(top_k, num_local_experts)
    hidden = routed_input_ND.new_empty(out_tokens, routed_input_ND.shape[1])
    tokens_per_expert = routed_input_ND.new_empty(num_local_experts, dtype=torch.int64)
    return hidden, tokens_per_expert, DispatchHandle()


@torch.library.custom_op("flexep::combine", mutates_args=(), device_types="cuda")
def _combine_impl(
    x: torch.Tensor,
    handle: DispatchHandle,
    num_routed_tokens: int,
) -> torch.Tensor:
    """Move expert outputs back to the origin rank's routed-token order."""
    if _buffer is None:
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

    grad_routed_input = None
    if grad_hidden is not None:
        grad_routed_input = _combine_to_origin(grad_hidden, state)

    return grad_routed_input, None, None, None, None


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


def dispatch_tokens(
    routed_input_ND: torch.Tensor,  # noqa: N803
    num_local_tokens_per_expert_E: torch.Tensor,  # noqa: N803
    token_indices_N: torch.Tensor,  # noqa: N803
    routed_scores_N: torch.Tensor | None,  # noqa: N803
    num_tokens: int,
    num_local_experts: int,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor, FlexEPDispatchState]:
    """Dispatch tokens to experts via FlexEP."""
    if _buffer is None or _rendezvous_handle is None:
        raise RuntimeError("FlexEP buffer not initialized.")
    _require_initialized(group.group_name, routed_input_ND)
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
    if routed_input_ND.shape[0] > _max_routed_tokens:
        raise RuntimeError(
            f"FlexEP buffer max_routed_tokens ({_max_routed_tokens}) is smaller "
            f"than routed input rows ({routed_input_ND.shape[0]})."
        )
    if routed_scores_N is not None and routed_scores_N.dtype != routed_input_ND.dtype:
        routed_scores_N = routed_scores_N.to(routed_input_ND.dtype)

    hidden, tokens_per_expert, dispatch_handle = torch.ops.flexep.dispatch(
        routed_input_ND,
        num_local_tokens_per_expert_E,
        num_tokens,
        ep_size,
        group.group_name,
    )

    state = FlexEPDispatchState(
        handle=dispatch_handle,
        token_indices=token_indices_N,
        routed_scores=routed_scores_N,
        num_tokens=num_tokens,
    )
    return hidden, tokens_per_expert, state


def combine_tokens(
    hidden_states: torch.Tensor,
    state: FlexEPDispatchState,
) -> torch.Tensor:
    """Combine expert outputs back to original token order."""
    if _buffer is None or _rendezvous_handle is None:
        raise RuntimeError("FlexEP buffer not initialized.")
    routed_output_ND = torch.ops.flexep.combine(  # noqa: N806
        hidden_states,
        state.handle,
        state.token_indices.shape[0],
    )
    if state.routed_scores is not None:
        routed_output_ND = routed_output_ND * state.routed_scores.reshape(-1, 1)
    out_TD = torch.zeros(  # noqa: N806
        state.num_tokens,
        hidden_states.shape[1],
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    return deterministic_scatter_add(
        out_TD,
        state.token_indices.reshape(-1, 1).expand(-1, hidden_states.shape[1]),
        routed_output_ND,
    )


__all__ = [
    "DispatchHandle",
    "FlexEPDispatchState",
    "combine_tokens",
    "dispatch_tokens",
    "init_buffer",
]
