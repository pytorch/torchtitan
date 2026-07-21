# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MinimalAsyncEP primitives for constrained MoE expert parallel dispatch.

This backend is intentionally narrow: it supports the launch shape where the
EP process group is the data-parallel group and TP/CP/PP/SP are disabled.
The symmetric-memory allocation is explicit and must happen before dispatch.

Shape symbols used by the API entrypoints:
    ``T``: local token rows.
    ``D``: model dimension.
    ``K``: routed experts per token.
    ``N = T * K``: local routed rows before EP exchange.
    ``R``: active rows assigned to this rank's local experts.
    ``R_max >= R``: static receive-buffer row capacity.
    ``E``: global experts.
    ``EP``: expert-parallel group size.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from torchtitan.distributed.minimal_async_ep.kernels import (
    copy_full_counts_to_peers_kernel,
    copy_rows_to_peers_kernel,
    expand_topk_grad_kernel,
    fill_combine_metadata_kernel,
    fill_dispatch_metadata_kernel,
    invert_flat_indices_kernel,
    reduce_topk_slots_kernel,
    topk_scores_grad_kernel,
)
from torchtitan.tools.logging import logger


_HIDDEN_RECV_BUFFER_COUNT = 2

_HIDDEN_READY_CHANNEL = 0
_COUNTS_READY_CHANNEL = 0


class _DispatchMetadataOutputs(NamedTuple):
    dispatch_dst_ranks: torch.Tensor
    dispatch_dst_rows: torch.Tensor
    combine_dst_ranks: torch.Tensor
    combine_dst_rows: torch.Tensor
    combine_num_valid_rows: torch.Tensor
    tokens_per_expert: torch.Tensor


class _DispatchMetadataScratch(NamedTuple):
    source_prefix: torch.Tensor
    totals: torch.Tensor
    expert_starts: torch.Tensor
    local_dest_offsets: torch.Tensor
    local_count_ends: torch.Tensor
    local_count_starts: torch.Tensor
    segment_lens: torch.Tensor
    output_ends: torch.Tensor
    output_starts: torch.Tensor
    source_input_starts: torch.Tensor


@dataclass
class _PendingEvent:
    event: torch.cuda.Event
    # Released after the main stream has joined the event.
    retained: tuple[torch.Tensor, ...] = ()


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
    comm_stream: torch.cuda.Stream
    hidden_recv_buffer_index: int = 0
    pending_events: dict[tuple[str, int], deque[_PendingEvent]] = field(
        default_factory=dict
    )


_buffer_state: _MinimalAsyncEPBufferState | None = None


def maybe_update_minimal_async_ep_config(model_config: Any, config: Any) -> None:
    """Validate and fill MinimalAsyncEP dispatcher configs from runtime config."""
    from torchtitan.config import ParallelismConfig, TORCH_DTYPE_MAP
    from torchtitan.distributed.activation_checkpoint import FullAC
    from torchtitan.models.common.token_dispatcher import MinimalAsyncEPTokenDispatcher
    from torchtitan.trainer import Trainer

    assert hasattr(
        config, "parallelism"
    ), "config passed to update_from_config must provide a parallelism field."
    parallelism = config.parallelism
    assert isinstance(parallelism, ParallelismConfig), (
        "config.parallelism must be a ParallelismConfig, got "
        f"{type(parallelism).__name__}."
    )

    dispatcher_cfgs = []
    for layer_cfg in model_config.layers:
        moe_cfg = getattr(layer_cfg, "moe", None)
        if moe_cfg is None:
            continue
        token_dispatcher_cfg = moe_cfg.routed_experts.token_dispatcher
        if isinstance(token_dispatcher_cfg, MinimalAsyncEPTokenDispatcher.Config):
            dispatcher_cfgs.append(token_dispatcher_cfg)

    if not dispatcher_cfgs:
        return

    if parallelism.spmd_backend == "full_dtensor":
        raise ValueError("MinimalAsyncEP does not support full_dtensor SPMD.")
    if parallelism.expert_parallel_degree == 1:
        raise ValueError(
            "MinimalAsyncEPTokenDispatcher.Config requires expert parallelism "
            "(expert_parallel_degree > 1)."
        )
    if parallelism.tensor_parallel_degree != 1:
        raise ValueError(
            "MinimalAsyncEP does not support tensor or sequence parallelism."
        )
    if parallelism.context_parallel_degree != 1:
        raise ValueError("MinimalAsyncEP does not support context parallelism.")
    if parallelism.pipeline_parallel_degree != 1:
        raise ValueError("MinimalAsyncEP does not support pipeline parallelism.")
    for num_experts in {cfg.num_experts for cfg in dispatcher_cfgs}:
        if num_experts % parallelism.expert_parallel_degree != 0:
            raise ValueError(
                f"MinimalAsyncEP num_experts ({num_experts}) must be "
                "divisible by expert_parallel_degree "
                f"({parallelism.expert_parallel_degree})."
            )

    if not isinstance(config, Trainer.Config):
        raise ValueError(
            "MinimalAsyncEP requires a Trainer.Config-compatible runtime config "
            "to set hidden_dim, tokens_per_rank, and dtype."
        )

    memory_policy = getattr(config.compile, "memory_policy", None)
    if (
        not isinstance(config.activation_checkpoint, FullAC.Config)
        and memory_policy != "full"
    ):
        raise ValueError(
            "MinimalAsyncEP requires full recompute: set "
            "activation-checkpoint:full for eager training or "
            "--compile.memory_policy full for graph_trainer."
        )

    for token_dispatcher_cfg in dispatcher_cfgs:
        token_dispatcher_cfg.hidden_dim = model_config.dim
        token_dispatcher_cfg.tokens_per_rank = (
            config.training.local_batch_size * config.training.seq_len
        )
        token_dispatcher_cfg.dtype = TORCH_DTYPE_MAP[
            config.training.mixed_precision_param
        ]


@dataclass
class MinimalAsyncEPDispatchMetadata:
    """MinimalAsyncEP metadata from dispatch needed for combine.

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
    max_routed_tokens = group.size() * tokens_per_rank * min(top_k, num_local_experts)
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
        group.size(), num_experts, dtype=torch.int64, device=device
    )
    hidden_recv_handles = [
        symm_mem.rendezvous(buffer, group) for buffer in hidden_recv_buffers
    ]
    counts_recv_handle = symm_mem.rendezvous(counts_recv_buffer, group)
    hidden_recv_peer_buffers = [
        [
            handle.get_buffer(peer, buffer.shape, buffer.dtype)
            for peer in range(group.size())
        ]
        for buffer, handle in zip(hidden_recv_buffers, hidden_recv_handles)
    ]
    hidden_recv_peer_ptrs = [
        torch.tensor(
            [peer_buffer.data_ptr() for peer_buffer in peer_buffers],
            dtype=torch.int64,
            device=device,
        )
        for peer_buffers in hidden_recv_peer_buffers
    ]
    counts_recv_peer_buffers = [
        counts_recv_handle.get_buffer(
            peer, counts_recv_buffer.shape, counts_recv_buffer.dtype
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
        comm_stream=torch.cuda.Stream(device=device),
    )


def _event_key(exchange: str, tensor: torch.Tensor) -> tuple[str, int]:
    return (exchange, tensor.data_ptr())


def _record_pending_event(
    exchange: str,
    tensor: torch.Tensor,
    stream: torch.cuda.Stream,
    retained: tuple[torch.Tensor, ...] = (),
) -> None:
    assert _buffer_state is not None
    event = torch.cuda.Event()
    event.record(stream)
    key = _event_key(exchange, tensor)
    _buffer_state.pending_events.setdefault(key, deque()).append(
        _PendingEvent(event, retained)
    )


def _wait_pending_event(exchange: str, tensor: torch.Tensor) -> None:
    assert _buffer_state is not None
    key = _event_key(exchange, tensor)
    events = _buffer_state.pending_events.get(key)
    if not events:
        raise RuntimeError(
            f"MinimalAsyncEP wait_{exchange} found no pending launch event for "
            f"tensor data_ptr={tensor.data_ptr()}."
        )
    pending = events.popleft()
    if not events:
        del _buffer_state.pending_events[key]
    torch.cuda.current_stream(tensor.device).wait_event(pending.event)


def _copy_rows_to_peers_async_cuda(
    x: torch.Tensor,
    dst_ranks: torch.Tensor,
    dst_rows: torch.Tensor,
    num_rows: int,
    *,
    receive_capacity: int,
    exchange: str,
    block_m: int = 1,
    num_warps: int = 4,
    src_rows: torch.Tensor | None = None,
    src_row_divisor: int = 1,
    num_valid_rows: torch.Tensor | None = None,
    retained: tuple[torch.Tensor, ...] = (),
) -> torch.Tensor:
    """Launch a row copy through symmetric memory on the EP comm stream."""
    assert _buffer_state is not None

    buffer_rows = _buffer_state.hidden_recv_buffers[0].shape[0]
    if receive_capacity > buffer_rows:
        raise RuntimeError(
            "MinimalAsyncEP receive capacity exceeds the initialized symmetric "
            f"buffer: receive_capacity={receive_capacity}, "
            f"buffer_rows={buffer_rows}."
        )

    buffer_index = _buffer_state.hidden_recv_buffer_index
    _buffer_state.hidden_recv_buffer_index = (
        buffer_index + 1
    ) % _HIDDEN_RECV_BUFFER_COUNT
    hidden_recv_buffer = _buffer_state.hidden_recv_buffers[buffer_index]
    hidden_recv_handle = _buffer_state.hidden_recv_handles[buffer_index]
    hidden_recv_peer_buffers = _buffer_state.hidden_recv_peer_buffers[buffer_index]
    hidden_recv_peer_ptrs = _buffer_state.hidden_recv_peer_ptrs[buffer_index]
    hidden_recv_view = hidden_recv_buffer.narrow(0, 0, receive_capacity).narrow(
        1,
        0,
        x.shape[1],
    )

    launch_stream = torch.cuda.current_stream(x.device)
    comm_stream = _buffer_state.comm_stream
    comm_stream.wait_stream(launch_stream)
    with torch.cuda.stream(comm_stream):
        copy_rows_to_peers_kernel(
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
            src_row_divisor=src_row_divisor,
            dst_ptrs=hidden_recv_peer_ptrs,
            num_valid_rows=num_valid_rows,
        )
        _wait_ready(hidden_recv_handle, _HIDDEN_READY_CHANNEL)
        _record_pending_event(exchange, hidden_recv_view, comm_stream, retained)
    return hidden_recv_view


def _wait_ready(handle: Any, channel: int) -> None:
    """EP-group barrier: ensure every peer has finished writing into this
    rank's symmetric receive buffer before the buffer is read.

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
    copy_full_counts_to_peers_kernel(
        num_local_tokens_per_expert_E,
        _buffer_state.counts_recv_peer_buffers,
        rank=_buffer_state.group.rank(),
        ep_size=ep_size,
        num_experts=num_experts,
        dst_ptrs=_buffer_state.counts_recv_peer_ptrs,
    )
    _wait_ready(_buffer_state.counts_recv_handle, _COUNTS_READY_CHANNEL)
    return _buffer_state.counts_recv_buffer


def _allocate_direct_metadata(
    num_local_tokens_per_expert_E: torch.Tensor,  # noqa: N803
    num_routed_rows: int,
    receive_capacity: int,
    ep_size: int,
) -> tuple[_DispatchMetadataOutputs, _DispatchMetadataScratch]:
    assert _buffer_state is not None

    rank = _buffer_state.group.rank()
    num_experts = num_local_tokens_per_expert_E.numel()
    num_local_experts = num_experts // ep_size
    new_empty = num_local_tokens_per_expert_E.new_empty

    total_de = new_empty(ep_size, num_local_experts)
    outputs = _DispatchMetadataOutputs(
        new_empty(num_routed_rows),
        new_empty(num_routed_rows),
        new_empty(receive_capacity),
        new_empty(receive_capacity),
        new_empty(1),
        total_de[rank],
    )
    scratch = _DispatchMetadataScratch(
        new_empty(ep_size, ep_size, num_local_experts),
        total_de,
        new_empty(ep_size, num_local_experts),
        new_empty(num_experts),
        new_empty(num_experts),
        new_empty(num_experts),
        new_empty(num_experts),
        new_empty(num_experts),
        new_empty(num_experts),
        new_empty(ep_size, num_experts),
    )
    return outputs, scratch


def _compute_direct_metadata(
    num_local_tokens_per_expert_E: torch.Tensor,  # noqa: N803
    all_tokens_per_expert_RE: torch.Tensor,  # noqa: N803
    num_routed_rows: int,
    receive_capacity: int,
    ep_size: int,
    outputs: _DispatchMetadataOutputs,
    scratch: _DispatchMetadataScratch,
) -> _DispatchMetadataOutputs:
    assert _buffer_state is not None

    rank = _buffer_state.group.rank()
    num_experts = num_local_tokens_per_expert_E.numel()
    num_local_experts = num_experts // ep_size
    counts_sde = all_tokens_per_expert_RE.view(
        ep_size,
        ep_size,
        num_local_experts,
    )
    torch.cumsum(counts_sde, 0, out=scratch.source_prefix)
    torch.sub(scratch.source_prefix, counts_sde, out=scratch.source_prefix)
    torch.sum(counts_sde, 0, out=scratch.totals)
    torch.cumsum(scratch.totals, 1, out=scratch.expert_starts)
    torch.sub(
        scratch.expert_starts,
        scratch.totals,
        out=scratch.expert_starts,
    )
    torch.add(
        scratch.expert_starts,
        scratch.source_prefix[rank],
        out=scratch.local_dest_offsets.view(ep_size, num_local_experts),
    )
    torch.cumsum(
        num_local_tokens_per_expert_E,
        0,
        out=scratch.local_count_ends,
    )
    torch.sub(
        scratch.local_count_ends,
        num_local_tokens_per_expert_E,
        out=scratch.local_count_starts,
    )
    fill_dispatch_metadata_kernel(
        num_local_tokens_per_expert_E,
        scratch.local_dest_offsets,
        scratch.local_count_starts,
        num_routed_tokens=num_routed_rows,
        num_local_experts=num_local_experts,
        max_tokens_per_segment=_buffer_state.tokens_per_rank,
        dst_ranks=outputs.dispatch_dst_ranks,
        dst_rows=outputs.dispatch_dst_rows,
    )

    scratch.segment_lens.view(num_local_experts, ep_size).copy_(
        counts_sde[:, rank, :].t()
    )
    torch.cumsum(scratch.segment_lens, 0, out=scratch.output_ends)
    torch.sub(
        scratch.output_ends,
        scratch.segment_lens,
        out=scratch.output_starts,
    )
    torch.cumsum(
        all_tokens_per_expert_RE,
        1,
        out=scratch.source_input_starts,
    )
    torch.sub(
        scratch.source_input_starts,
        all_tokens_per_expert_RE,
        out=scratch.source_input_starts,
    )
    fill_combine_metadata_kernel(
        scratch.segment_lens,
        scratch.output_starts,
        scratch.source_input_starts,
        ep_rank=rank,
        ep_size=ep_size,
        num_local_experts=num_local_experts,
        receive_capacity=receive_capacity,
        max_tokens_per_segment=_buffer_state.tokens_per_rank,
        dst_ranks=outputs.combine_dst_ranks,
        dst_rows=outputs.combine_dst_rows,
        num_valid_rows=outputs.combine_num_valid_rows,
    )

    return outputs


def _dispatch_metadata(
    num_local_tokens_per_expert_E: torch.Tensor,  # noqa: N803
    num_routed_rows: int,
    receive_capacity: int,
    ep_size: int,
    outputs: _DispatchMetadataOutputs,
    scratch: _DispatchMetadataScratch,
) -> _DispatchMetadataOutputs:
    """Exchange per-expert local counts and build dispatch/combine metadata.

    Args:
        num_local_tokens_per_expert_E: ``(E,)`` int64 counts for this rank's
            local token shard over all global experts.
        num_routed_rows: ``N`` routed rows in local E-major order.
        receive_capacity: ``R_max``.
        ep_size: ``EP``.
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
        outputs,
        scratch,
    )


@torch.library.custom_op(
    "minimal_async_ep::dispatch",
    mutates_args=(),
    device_types="cuda",
)
def dispatch_op(
    dispatch_input: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,  # noqa: N803
    num_local_tokens_per_expert_E: torch.Tensor,  # noqa: N803
    receive_capacity: int,
    ep_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build MinimalAsyncEP metadata and dispatch rows to expert-owner ranks.

    Args:
        dispatch_input: ``(T, D)`` local token rows.
        topk_expert_ids_TK: ``(T, K)`` global expert ids.
        num_local_tokens_per_expert_E: ``(E,)`` counts for this rank's token
            shard over all global experts.
        receive_capacity: ``R_max``.
        ep_size: ``EP``.
    Returns:
        Pending ``hidden_states`` and local-expert counts, plus routing metadata
        used after the dispatch wait by combine and backward.
    """
    assert _buffer_state is not None

    T_row_to_expert_N = topk_expert_ids_TK.reshape(-1)  # noqa: N806
    num_routed_rows = T_row_to_expert_N.numel()
    E_row_to_T_row_N = topk_expert_ids_TK.new_empty(  # noqa: N806
        num_routed_rows,
        dtype=torch.int64,
    )
    T_row_to_E_row_N = torch.empty_like(E_row_to_T_row_N)  # noqa: N806
    top_k = topk_expert_ids_TK.shape[1]
    metadata_outputs, scratch = _allocate_direct_metadata(
        num_local_tokens_per_expert_E,
        num_routed_rows,
        receive_capacity,
        ep_size,
    )

    launch_stream = torch.cuda.current_stream(dispatch_input.device)
    comm_stream = _buffer_state.comm_stream
    comm_stream.wait_stream(launch_stream)
    with torch.cuda.stream(comm_stream):
        torch.argsort(
            T_row_to_expert_N,
            stable=True,
            out=E_row_to_T_row_N,
        )
        (
            dispatch_dst_ranks,
            dispatch_dst_rows,
            combine_dst_ranks,
            combine_dst_rows,
            combine_num_valid_rows,
            num_tokens_per_local_expert_e,
        ) = _dispatch_metadata(
            num_local_tokens_per_expert_E,
            num_routed_rows,
            receive_capacity,
            ep_size,
            metadata_outputs,
            scratch,
        )

        # E_row_to_T_row_N=[2, 0, 3, 1] means E-major row 0 came
        # from T-major row 2, so T_row_to_E_row_N=[1, 3, 0, 2].
        invert_flat_indices_kernel(
            E_row_to_T_row_N,
            num_rows=num_routed_rows,
            out=T_row_to_E_row_N,
        )

        hidden_states = _copy_rows_to_peers_async_cuda(
            dispatch_input,
            dispatch_dst_ranks,
            dispatch_dst_rows,
            num_routed_rows,
            receive_capacity=receive_capacity,
            exchange="dispatch",
            block_m=4,
            num_warps=8,
            src_rows=E_row_to_T_row_N,
            src_row_divisor=top_k,
            retained=scratch,
        )
    return (
        hidden_states,
        dispatch_dst_ranks,
        dispatch_dst_rows,
        combine_dst_ranks,
        combine_dst_rows,
        combine_num_valid_rows,
        E_row_to_T_row_N,
        T_row_to_E_row_N,
        num_tokens_per_local_expert_e,
    )


@dispatch_op.register_fake
def dispatch_op_fake(
    dispatch_input: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,  # noqa: N803
    num_local_tokens_per_expert_E: torch.Tensor,  # noqa: N803
    receive_capacity: int,
    ep_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    num_routed_rows = topk_expert_ids_TK.numel()
    num_local_experts = num_local_tokens_per_expert_E.shape[0] // ep_size
    return (
        dispatch_input.new_empty(receive_capacity, dispatch_input.shape[1]),
        topk_expert_ids_TK.new_empty(num_routed_rows, dtype=torch.int64),
        topk_expert_ids_TK.new_empty(num_routed_rows, dtype=torch.int64),
        topk_expert_ids_TK.new_empty(receive_capacity, dtype=torch.int64),
        topk_expert_ids_TK.new_empty(receive_capacity, dtype=torch.int64),
        topk_expert_ids_TK.new_empty(1, dtype=torch.int64),
        topk_expert_ids_TK.new_empty(num_routed_rows, dtype=torch.int64),
        topk_expert_ids_TK.new_empty(num_routed_rows, dtype=torch.int64),
        num_local_tokens_per_expert_E.new_empty(
            num_local_experts,
            dtype=torch.int64,
        ),
    )


_WAIT_OPS_LIB = torch.library.Library("minimal_async_ep", "FRAGMENT")
_WAIT_OPS_LIB.define(
    "wait_dispatch(Tensor(a) pending, Tensor(b) pending_counts, "
    "Tensor[] keepalives) -> (Tensor(a), Tensor(b))"
)
_WAIT_OPS_LIB.define(
    "wait_dispatch_data(Tensor(a) pending, Tensor[] keepalives) -> Tensor(a)"
)
_WAIT_OPS_LIB.define(
    "wait_combine(Tensor(a) pending, Tensor[] keepalives) -> Tensor(a)"
)


def _wait_dispatch_impl(
    pending: torch.Tensor,
    pending_counts: torch.Tensor,
    keepalives: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    del keepalives
    _wait_pending_event("dispatch", pending)
    return pending, pending_counts


def _wait_dispatch_data_impl(
    pending: torch.Tensor,
    keepalives: list[torch.Tensor],
) -> torch.Tensor:
    del keepalives
    _wait_pending_event("dispatch", pending)
    return pending


def _wait_combine_impl(
    pending: torch.Tensor,
    keepalives: list[torch.Tensor],
) -> torch.Tensor:
    del keepalives
    _wait_pending_event("combine", pending)
    return pending


def _wait_dispatch_meta(
    pending: torch.Tensor,
    pending_counts: torch.Tensor,
    keepalives: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    del keepalives
    return pending, pending_counts


def _wait_combine_meta(
    pending: torch.Tensor,
    keepalives: list[torch.Tensor],
) -> torch.Tensor:
    del keepalives
    return pending


_WAIT_OPS_LIB.impl("wait_dispatch", _wait_dispatch_impl, "CompositeExplicitAutograd")
_WAIT_OPS_LIB.impl(
    "wait_dispatch_data", _wait_dispatch_data_impl, "CompositeExplicitAutograd"
)
_WAIT_OPS_LIB.impl("wait_combine", _wait_combine_impl, "CompositeExplicitAutograd")
wait_dispatch_op = torch.ops.minimal_async_ep.wait_dispatch.default
_wait_dispatch_data_op = torch.ops.minimal_async_ep.wait_dispatch_data.default
wait_combine_op = torch.ops.minimal_async_ep.wait_combine.default


class _WaitTensor(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, pending, keepalives, op):
        del ctx
        with torch._C._AutoDispatchBelowAutograd():
            return op(pending, keepalives)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad):
        del ctx
        return grad, None, None


class _WaitDispatchTensors(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, pending, pending_counts, keepalives):
        del ctx
        with torch._C._AutoDispatchBelowAutograd():
            return wait_dispatch_op(pending, pending_counts, keepalives)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_pending, grad_counts):
        del ctx
        return grad_pending, grad_counts, None


def _wait_dispatch_autograd(pending, pending_counts, keepalives):
    return _WaitDispatchTensors.apply(pending, pending_counts, keepalives)


def _wait_dispatch_data_autograd(pending, keepalives):
    return _WaitTensor.apply(pending, keepalives, _wait_dispatch_data_op)


def _wait_combine_autograd(pending, keepalives):
    return _WaitTensor.apply(pending, keepalives, wait_combine_op)


_WAIT_OPS_LIB.impl("wait_dispatch", _wait_dispatch_autograd, "Autograd")
_WAIT_OPS_LIB.impl("wait_dispatch_data", _wait_dispatch_data_autograd, "Autograd")
_WAIT_OPS_LIB.impl("wait_combine", _wait_combine_autograd, "Autograd")


def _functionalize_wait(op, pending, keepalives):
    torch._sync(pending)
    for tensor in keepalives:
        torch._sync(tensor)
    pending_inner = torch._from_functional_tensor(pending)
    keepalives_inner = [torch._from_functional_tensor(t) for t in keepalives]
    with torch._C._ExcludeDispatchKeyGuard(
        torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
    ):
        op(pending_inner, keepalives_inner)
    return pending


@torch.library.impl(_WAIT_OPS_LIB, "wait_dispatch", "Functionalize")
def _wait_dispatch_functionalize(pending, pending_counts, keepalives):
    torch._sync(pending)
    torch._sync(pending_counts)
    for tensor in keepalives:
        torch._sync(tensor)
    pending_inner = torch._from_functional_tensor(pending)
    counts_inner = torch._from_functional_tensor(pending_counts)
    keepalives_inner = [torch._from_functional_tensor(t) for t in keepalives]
    with torch._C._ExcludeDispatchKeyGuard(
        torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
    ):
        wait_dispatch_op(pending_inner, counts_inner, keepalives_inner)
    return pending, pending_counts


@torch.library.impl(_WAIT_OPS_LIB, "wait_dispatch_data", "Functionalize")
def _wait_dispatch_data_functionalize(pending, keepalives):
    return _functionalize_wait(_wait_dispatch_data_op, pending, keepalives)


@torch.library.impl(_WAIT_OPS_LIB, "wait_combine", "Functionalize")
def _wait_combine_functionalize(pending, keepalives):
    return _functionalize_wait(wait_combine_op, pending, keepalives)


torch.library.register_fake("minimal_async_ep::wait_dispatch")(_wait_dispatch_meta)
torch.library.register_fake("minimal_async_ep::wait_dispatch_data")(_wait_combine_meta)
torch.library.register_fake("minimal_async_ep::wait_combine")(_wait_combine_meta)


def dispatch(
    dispatch_input: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,  # noqa: N803
    num_local_tokens_per_expert_E: torch.Tensor,  # noqa: N803
    receive_capacity: int,
    ep_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Launch dispatch, then wait before returning a readable hidden buffer."""
    (
        hidden_states,
        dispatch_dst_ranks,
        dispatch_dst_rows,
        combine_dst_ranks,
        combine_dst_rows,
        combine_num_valid_rows,
        E_row_to_T_row_N,
        T_row_to_E_row_N,
        num_tokens_per_local_expert_e,
    ) = dispatch_op(
        dispatch_input,
        topk_expert_ids_TK,
        num_local_tokens_per_expert_E,
        receive_capacity,
        ep_size,
    )
    hidden_states, num_tokens_per_local_expert_e = wait_dispatch_op(
        hidden_states,
        num_tokens_per_local_expert_e,
        [
            dispatch_input,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
        ],
    )
    return (
        hidden_states,
        dispatch_dst_ranks,
        dispatch_dst_rows,
        combine_dst_ranks,
        combine_dst_rows,
        combine_num_valid_rows,
        E_row_to_T_row_N,
        T_row_to_E_row_N,
        num_tokens_per_local_expert_e,
    )


@torch.library.custom_op(
    "minimal_async_ep::dispatch_data",
    mutates_args=(),
    device_types="cuda",
)
def dispatch_data_op(
    x_ND: torch.Tensor,  # noqa: N803
    dispatch_dst_ranks: torch.Tensor,
    dispatch_dst_rows: torch.Tensor,
    receive_capacity: int,
    num_routed_rows: int,
) -> torch.Tensor:
    """Launch a precomputed dispatch row copy to expert-owner ranks."""
    return _copy_rows_to_peers_async_cuda(
        x_ND,
        dispatch_dst_ranks,
        dispatch_dst_rows,
        num_routed_rows,
        receive_capacity=receive_capacity,
        exchange="dispatch",
        block_m=4,
        num_warps=8,
    )


@dispatch_data_op.register_fake
def dispatch_data_op_fake(
    x_ND: torch.Tensor,  # noqa: N803
    dispatch_dst_ranks: torch.Tensor,
    dispatch_dst_rows: torch.Tensor,
    receive_capacity: int,
    num_routed_rows: int,
) -> torch.Tensor:
    del dispatch_dst_ranks, dispatch_dst_rows, num_routed_rows
    return x_ND.new_empty(receive_capacity, x_ND.shape[1])


def dispatch_data(
    x_ND: torch.Tensor,  # noqa: N803
    dispatch_dst_ranks: torch.Tensor,
    dispatch_dst_rows: torch.Tensor,
    receive_capacity: int,
    num_routed_rows: int,
) -> torch.Tensor:
    """Launch and wait for a precomputed dispatch row copy."""
    hidden_states = dispatch_data_op(
        x_ND,
        dispatch_dst_ranks,
        dispatch_dst_rows,
        receive_capacity,
        num_routed_rows,
    )
    return _wait_dispatch_data_op(
        hidden_states,
        [x_ND, dispatch_dst_ranks, dispatch_dst_rows],
    )


@torch.library.custom_op(
    "minimal_async_ep::combine",
    mutates_args=(),
    device_types="cuda",
)
def combine_op(
    x: torch.Tensor,
    dispatch_dst_ranks: torch.Tensor,
    dispatch_dst_rows: torch.Tensor,
    combine_dst_ranks: torch.Tensor,
    combine_dst_rows: torch.Tensor,
    combine_num_valid_rows: torch.Tensor,
    num_routed_rows: int,
) -> torch.Tensor:
    """Launch expert-output rows back to origin ranks.

    Args:
        x: ``(R_max, D)`` local expert output rows.
        dispatch_dst_ranks, dispatch_dst_rows: ``(N,)`` forward dispatch
            destinations, saved for backward.
        combine_dst_ranks, combine_dst_rows: ``(R_max,)`` origin rank and
            origin E-major row for each active received row.
        combine_num_valid_rows: ``(1,)`` device scalar active row count ``R``.
        num_routed_rows: ``N`` local E-major routed rows.
    Returns:
        ``routed_output_ND``: pending ``(N, D)`` origin-rank E-major routed
        rows. The tensor is not readable until ``wait_combine``.
    """
    del dispatch_dst_ranks, dispatch_dst_rows
    origin_recv_buffer = _copy_rows_to_peers_async_cuda(
        x,
        combine_dst_ranks,
        combine_dst_rows,
        x.shape[0],
        receive_capacity=num_routed_rows,
        exchange="combine",
        block_m=4,
        num_valid_rows=combine_num_valid_rows,
    )
    return origin_recv_buffer.narrow(0, 0, num_routed_rows).narrow(1, 0, x.shape[1])


@combine_op.register_fake
def combine_op_fake(
    x: torch.Tensor,
    dispatch_dst_ranks: torch.Tensor,
    dispatch_dst_rows: torch.Tensor,
    combine_dst_ranks: torch.Tensor,
    combine_dst_rows: torch.Tensor,
    combine_num_valid_rows: torch.Tensor,
    num_routed_rows: int,
) -> torch.Tensor:
    del dispatch_dst_ranks
    del dispatch_dst_rows
    del combine_dst_ranks
    del combine_dst_rows
    del combine_num_valid_rows
    return x.new_empty(num_routed_rows, x.shape[1])


@torch.library.custom_op(
    "minimal_async_ep::combine_data",
    mutates_args=(),
    device_types="cuda",
)
def combine_data_op(
    x: torch.Tensor,
    combine_dst_ranks: torch.Tensor,
    combine_dst_rows: torch.Tensor,
    combine_num_valid_rows: torch.Tensor,
    num_routed_rows: int,
) -> torch.Tensor:
    """Launch a precomputed combine row copy back to origin ranks."""
    origin_recv_buffer = _copy_rows_to_peers_async_cuda(
        x,
        combine_dst_ranks,
        combine_dst_rows,
        x.shape[0],
        receive_capacity=num_routed_rows,
        exchange="combine",
        block_m=4,
        num_valid_rows=combine_num_valid_rows,
    )
    return origin_recv_buffer.narrow(0, 0, num_routed_rows).narrow(1, 0, x.shape[1])


@combine_data_op.register_fake
def combine_data_op_fake(
    x: torch.Tensor,
    combine_dst_ranks: torch.Tensor,
    combine_dst_rows: torch.Tensor,
    combine_num_valid_rows: torch.Tensor,
    num_routed_rows: int,
) -> torch.Tensor:
    del combine_dst_ranks, combine_dst_rows, combine_num_valid_rows
    return x.new_empty(num_routed_rows, x.shape[1])


def combine_data(
    x: torch.Tensor,
    combine_dst_ranks: torch.Tensor,
    combine_dst_rows: torch.Tensor,
    combine_num_valid_rows: torch.Tensor,
    num_routed_rows: int,
) -> torch.Tensor:
    """Launch and wait for a precomputed combine row copy."""
    routed_output = combine_data_op(
        x,
        combine_dst_ranks,
        combine_dst_rows,
        combine_num_valid_rows,
        num_routed_rows,
    )
    return wait_combine_op(
        routed_output,
        [x, combine_dst_ranks, combine_dst_rows, combine_num_valid_rows],
    )


@torch.library.custom_op(
    "minimal_async_ep::reduce_topk",
    mutates_args=(),
    device_types="cuda",
)
def reduce_topk_op(
    routed_output_ND: torch.Tensor,  # noqa: N803
    T_row_to_E_row_N: torch.Tensor,  # noqa: N803
    E_row_to_T_row_N: torch.Tensor,  # noqa: N803
    routed_scores_N: torch.Tensor,  # noqa: N803
    num_tokens: int,
    top_k: int,
) -> torch.Tensor:
    del E_row_to_T_row_N
    return reduce_topk_slots_kernel(
        routed_output_ND,
        T_row_to_E_row_N,
        routed_scores_N,
        num_tokens=num_tokens,
        top_k=top_k,
        scores_are_slot_ordered=True,
    )


@reduce_topk_op.register_fake
def reduce_topk_op_fake(
    routed_output_ND: torch.Tensor,  # noqa: N803
    T_row_to_E_row_N: torch.Tensor,  # noqa: N803
    E_row_to_T_row_N: torch.Tensor,  # noqa: N803
    routed_scores_N: torch.Tensor,  # noqa: N803
    num_tokens: int,
    top_k: int,
) -> torch.Tensor:
    del T_row_to_E_row_N, E_row_to_T_row_N, routed_scores_N, top_k
    return routed_output_ND.new_empty(num_tokens, routed_output_ND.shape[1])


@torch.library.custom_op(
    "minimal_async_ep::expand_topk_grad",
    mutates_args=(),
    device_types="cuda",
)
def _expand_topk_grad_op(
    grad_out_TD: torch.Tensor,  # noqa: N803
    E_row_to_T_row_N: torch.Tensor,  # noqa: N803
    routed_scores_N: torch.Tensor,  # noqa: N803
    grad_scores_N: torch.Tensor,  # noqa: N803
    top_k: int,
) -> torch.Tensor:
    # Dependency-only input: ``grad_scores_N`` is computed from
    # ``routed_output_ND``, a view into the combine receive pool. It must run
    # before this branch can launch downstream combine-pool reuse.
    del grad_scores_N
    return expand_topk_grad_kernel(
        grad_out_TD,
        E_row_to_T_row_N,
        routed_scores_N,
        top_k=top_k,
        dtype=grad_out_TD.dtype,
        scores_are_slot_ordered=True,
    )


@_expand_topk_grad_op.register_fake
def _expand_topk_grad_op_fake(
    grad_out_TD: torch.Tensor,  # noqa: N803
    E_row_to_T_row_N: torch.Tensor,  # noqa: N803
    routed_scores_N: torch.Tensor,  # noqa: N803
    grad_scores_N: torch.Tensor,  # noqa: N803
    top_k: int,
) -> torch.Tensor:
    del routed_scores_N, grad_scores_N, top_k
    return grad_out_TD.new_empty(E_row_to_T_row_N.numel(), grad_out_TD.shape[1])


@torch.library.custom_op(
    "minimal_async_ep::topk_scores_grad",
    mutates_args=(),
    device_types="cuda",
)
def _topk_scores_grad_op(
    routed_output_ND: torch.Tensor,  # noqa: N803
    grad_out_TD: torch.Tensor,  # noqa: N803
    E_row_to_T_row_N: torch.Tensor,  # noqa: N803
    routed_scores_N: torch.Tensor,  # noqa: N803
    top_k: int,
) -> torch.Tensor:
    return topk_scores_grad_kernel(
        routed_output_ND,
        grad_out_TD,
        E_row_to_T_row_N,
        top_k=top_k,
        dtype=routed_scores_N.dtype,
        scores_are_slot_ordered=True,
    )


@_topk_scores_grad_op.register_fake
def _topk_scores_grad_op_fake(
    routed_output_ND: torch.Tensor,  # noqa: N803
    grad_out_TD: torch.Tensor,  # noqa: N803
    E_row_to_T_row_N: torch.Tensor,  # noqa: N803
    routed_scores_N: torch.Tensor,  # noqa: N803
    top_k: int,
) -> torch.Tensor:
    del routed_output_ND, grad_out_TD, E_row_to_T_row_N, top_k
    return torch.empty_like(routed_scores_N)


def reduce_topk_setup_context(ctx, inputs, output):
    (
        routed_output_ND,
        _T_row_to_E_row_N,
        E_row_to_T_row_N,
        routed_scores_N,
        _num_tokens,
        top_k,
    ) = inputs
    del output
    ctx.top_k = top_k
    ctx.save_for_backward(
        routed_output_ND,
        E_row_to_T_row_N,
        routed_scores_N,
    )


def reduce_topk_autograd_backward(ctx, grad_out):
    (
        routed_output_ND,
        E_row_to_T_row_N,
        routed_scores_N,
    ) = ctx.saved_tensors
    # ``routed_output_ND`` is a view into the MinimalAsyncEP symmetric receive
    # buffer. Consume it before producing ``grad_routed_output``; downstream
    # combine backward may launch another exchange that reuses that buffer.
    grad_scores = _topk_scores_grad_op(
        routed_output_ND,
        grad_out,
        E_row_to_T_row_N,
        routed_scores_N,
        ctx.top_k,
    )
    grad_routed_output = _expand_topk_grad_op(
        grad_out,
        E_row_to_T_row_N,
        routed_scores_N,
        grad_scores,
        ctx.top_k,
    )
    return grad_routed_output, None, None, grad_scores, None, None


reduce_topk_op.register_autograd(
    reduce_topk_autograd_backward,
    setup_context=reduce_topk_setup_context,
)


@torch.library.custom_op(
    "minimal_async_ep::reduce_topk_no_scores",
    mutates_args=(),
    device_types="cuda",
)
def reduce_topk_no_scores_op(
    routed_output_ND: torch.Tensor,  # noqa: N803
    T_row_to_E_row_N: torch.Tensor,  # noqa: N803
    num_tokens: int,
    top_k: int,
) -> torch.Tensor:
    return reduce_topk_slots_kernel(
        routed_output_ND,
        T_row_to_E_row_N,
        None,
        num_tokens=num_tokens,
        top_k=top_k,
    )


@reduce_topk_no_scores_op.register_fake
def reduce_topk_no_scores_op_fake(
    routed_output_ND: torch.Tensor,  # noqa: N803
    T_row_to_E_row_N: torch.Tensor,  # noqa: N803
    num_tokens: int,
    top_k: int,
) -> torch.Tensor:
    del T_row_to_E_row_N, top_k
    return routed_output_ND.new_empty(num_tokens, routed_output_ND.shape[1])


def combine(
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
    """Launch combine, wait, then reduce routed top-k rows."""
    routed_output_ND = combine_op(  # noqa: N806
        x,
        dispatch_dst_ranks,
        dispatch_dst_rows,
        combine_dst_ranks,
        combine_dst_rows,
        combine_num_valid_rows,
        num_tokens * top_k,
    )
    routed_output_ND = wait_combine_op(
        routed_output_ND,
        [x, combine_dst_ranks, combine_dst_rows, combine_num_valid_rows],
    )
    out_TD = reduce_topk_op(  # noqa: N806
        routed_output_ND,
        T_row_to_E_row_N,
        E_row_to_T_row_N,
        routed_scores_N,
        num_tokens,
        top_k,
    )
    return out_TD, routed_output_ND


def dispatch_setup_context(ctx, inputs, output):
    (
        dispatch_input,
        topk_expert_ids_TK,  # noqa: N806
        _num_local_tokens_per_expert_E,
        _receive_capacity,
        _ep_size,
    ) = inputs
    (
        _hidden_states,
        _dispatch_dst_ranks,
        _dispatch_dst_rows,
        combine_dst_ranks,
        combine_dst_rows,
        combine_num_valid_rows,
        _E_row_to_T_row_N,
        T_row_to_E_row_N,
        _num_tokens_per_local_expert_e,
    ) = output
    ctx.num_routed_rows = topk_expert_ids_TK.numel()
    ctx.num_tokens = dispatch_input.shape[0]
    ctx.top_k = topk_expert_ids_TK.shape[1]
    ctx.save_for_backward(
        combine_dst_ranks,
        combine_dst_rows,
        combine_num_valid_rows,
        T_row_to_E_row_N,
    )


def dispatch_autograd_backward(ctx, grad_hidden, *unused_grads):
    del unused_grads
    (
        combine_dst_ranks,
        combine_dst_rows,
        combine_num_valid_rows,
        T_row_to_E_row_N,
    ) = ctx.saved_tensors

    grad_routed_input = combine_data(
        grad_hidden,
        combine_dst_ranks,
        combine_dst_rows,
        combine_num_valid_rows,
        ctx.num_routed_rows,
    )
    grad_input = reduce_topk_no_scores_op(
        grad_routed_input,
        T_row_to_E_row_N,
        ctx.num_tokens,
        ctx.top_k,
    )

    # Grads for dispatch_input, topk_expert_ids, counts, capacity, and ep_size.
    return grad_input, None, None, None, None


def combine_setup_context(ctx, inputs, output):
    (
        hidden_states,
        dispatch_dst_ranks,
        dispatch_dst_rows,
        _combine_dst_ranks,
        _combine_dst_rows,
        _combine_num_valid_rows,
        num_routed_rows,
    ) = inputs
    del output
    ctx.receive_capacity = hidden_states.shape[0]
    ctx.num_routed_rows = num_routed_rows
    ctx.save_for_backward(
        dispatch_dst_ranks,
        dispatch_dst_rows,
    )


def combine_autograd_backward(ctx, grad_routed_output):
    (
        dispatch_dst_ranks,
        dispatch_dst_rows,
    ) = ctx.saved_tensors
    grad_x = dispatch_data(
        grad_routed_output,
        dispatch_dst_ranks,
        dispatch_dst_rows,
        ctx.receive_capacity,
        ctx.num_routed_rows,
    )

    # Grads for x, dispatch/combine metadata, and num_routed_rows.
    return (
        grad_x,
        None,
        None,
        None,
        None,
        None,
        None,
    )


dispatch_op.register_autograd(
    dispatch_autograd_backward,
    setup_context=dispatch_setup_context,
)
combine_op.register_autograd(
    combine_autograd_backward,
    setup_context=combine_setup_context,
)

# Preserve the asynchronous launches and stream joins through FX cleanup.
for _side_effect_op in (
    torch.ops.minimal_async_ep.dispatch.default,
    torch.ops.minimal_async_ep.dispatch_data.default,
    torch.ops.minimal_async_ep.wait_dispatch.default,
    torch.ops.minimal_async_ep.wait_dispatch_data.default,
    torch.ops.minimal_async_ep.combine.default,
    torch.ops.minimal_async_ep.combine_data.default,
    torch.ops.minimal_async_ep.wait_combine.default,
):
    torch.fx.node.has_side_effect(_side_effect_op)

__all__ = [
    "MinimalAsyncEPDispatchMetadata",
    "combine",
    "combine_data",
    "combine_data_op",
    "combine_op",
    "dispatch",
    "dispatch_data",
    "dispatch_data_op",
    "dispatch_op",
    "init_buffer",
    "maybe_update_minimal_async_ep_config",
    "reduce_topk_no_scores_op",
    "reduce_topk_op",
    "wait_combine_op",
    "wait_dispatch_op",
]
