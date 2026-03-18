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

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.distributed import ProcessGroup

try:
    from deep_ep import Buffer, Config  # pyrefly: ignore[missing-import]
    from deep_ep.utils import (  # pyrefly: ignore[missing-import]
        EventHandle,
        EventOverlap,
    )
except ImportError as e:
    raise ImportError(
        "DeepEP is required for this module. "
        "Install from: https://github.com/deepseek-ai/deepep"
    ) from e

from torchtitan.tools.logging import logger


# Global buffer (single buffer per process, recreated if group changes)
# pyrefly: ignore [bad-assignment]
_buffer: Buffer = None

# Global cache for dispatch handles, keyed by handle_id
# SAC saves the handle_id tensor; we use it to retrieve the non-tensor handle
_handle_cache: dict = {}
_handle_counter: int = 0

# Pending combine event for deferred synchronization.
# Stores the EventOverlap from buffer.combine() to allow overlapping
# shared_experts computation with combine communication.
# This is process-local state (each GPU process has its own Python interpreter),
# and execution is single-threaded, so a simple module variable suffices.
_pending_combine_event: Optional[EventOverlap] = None

# Global tuned configs (set by autotune or manually)
_tuned_dispatch_config: Optional[Config] = None
_tuned_combine_config: Optional[Config] = None


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
        config=_tuned_dispatch_config,
    )

    after_event.current_stream_wait()

    handle_id = _get_next_handle_id()
    _handle_cache[handle_id.item()] = handle

    recv_num_tokens_per_expert = torch.tensor(
        recv_num_tokens_per_expert_list, dtype=torch.int32, device="cpu"
    )
    # pyrefly: ignore [bad-return]
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
        config=_tuned_combine_config,
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
    global _buffer, _pending_combine_event

    buffer = _buffer
    assert buffer is not None, "Buffer must be initialized before combine"

    # In inference mode, setup_context doesn't run, so we clean up handle_cache here.
    # NOTE: For inference, use torch.inference_mode() instead of torch.no_grad()
    if torch.is_inference_mode_enabled():
        handle = _handle_cache.pop(handle_id.item(), None)
    else:
        handle = _handle_cache.get(handle_id.item())
    assert handle is not None, f"Handle not found for handle_id={handle_id.item()}"

    previous_event = EventOverlap(EventHandle())

    combined, _, after_event = buffer.combine(
        x=x,
        handle=handle,
        previous_event=previous_event,
        async_finish=True,
        allocate_on_comm_stream=True,
        config=_tuned_combine_config,
    )

    # Store event for deferred sync instead of syncing immediately.
    # This enables overlapping shared_experts computation with combine communication.
    # The caller MUST call sync_combine() before using the returned tensor.
    _pending_combine_event = after_event

    return combined


def _combine_setup_context(ctx, inputs, output):
    _, handle_id = inputs
    # Pop handle from cache and save it for backward
    ctx.saved_handle = _handle_cache.pop(handle_id.item(), None)


def _combine_backward(ctx, grad_combined):
    """Backward for combine: performs dispatch on gradients."""
    global _buffer

    handle = ctx.saved_handle
    assert handle is not None, "Handle not found in combine backward"
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
        config=_tuned_dispatch_config,
    )

    after_event.current_stream_wait()

    return grad_x, None


torch.library.register_autograd(
    "deepep::dispatch", _dispatch_backward, setup_context=_dispatch_setup_context
)
torch.library.register_autograd(
    "deepep::combine", _combine_backward, setup_context=_combine_setup_context
)


@torch.compiler.disable()
def sync_combine() -> None:
    """Synchronize the current CUDA stream with the pending combine operation.

    This function MUST be called before using the result of combine_tokens()
    to ensure the async combine has completed. It inserts a wait operation
    on the current CUDA stream, making subsequent CUDA kernels wait for
    the combine to finish.

    torch.compile Compatibility:
        Decorated with @torch.compiler.disable() to always run in eager mode.
        This avoids issues with CUDA event operations not being traceable.

    Process Isolation:
        Each GPU process has its own Python interpreter, so this module-level
        variable is inherently process-local. No cross-process interference.

    Single-Threaded Execution:
        PyTorch training is single-threaded per process, so no thread safety
        concerns. Sequential execution guarantees correct event ordering.

    Activation Checkpointing Compatibility:
        - During forward: combine stores event, sync_combine() waits on it
        - During AC recomputation: combine runs again, stores NEW event,
          sync_combine() waits on the new event
        - Sequential execution ensures each forward/recompute uses its own event

    Multiple MoE Layers:
        Each layer's combine overwrites the pending event. Since sync_combine()
        is called before using each layer's output (and before the next layer's
        combine), this is safe. The sync clears the event to prevent double-sync.

    Safe to call multiple times - subsequent calls are no-ops if the event
    was already synced or if no combine operation is pending.
    """
    global _pending_combine_event

    if _pending_combine_event is not None:
        _pending_combine_event.current_stream_wait()
        _pending_combine_event = None


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


def _permute_tokens(
    hidden_states: torch.Tensor,
    dispatched_indices: torch.Tensor,
    dispatched_scores: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert dispatch output to grouped_mm format with permutation and token expansion.

    Each token may be routed to multiple experts (top-k), so tokens are expanded and sorted
    by expert ID for efficient grouped matrix multiplication. num_recv_tokens is the number
    of unique tokens received, while num_all_tokens is the expanded count after replication.

    Args:
        hidden_states: Received tokens [num_recv_tokens, hidden_dim]
        dispatched_indices: Expert indices for each token [num_recv_tokens, topk], -1 means masked
        dispatched_scores: Routing scores [num_recv_tokens, topk]

    Returns:
        permuted_hidden_states: Tokens sorted by expert [num_all_tokens, hidden_dim]
        permuted_scores: Routing scores in same order [num_all_tokens]
        permuted_indices: Original token indices for unpermutation [num_all_tokens]
    """
    mask = dispatched_indices != -1
    valid_expert_ids = dispatched_indices[mask]  # 1d tensor
    valid_scores = dispatched_scores[mask]

    # Repeat each token by its valid count and select tokens in expert order
    sort_order = torch.argsort(valid_expert_ids, stable=True)
    permuted_indices = torch.arange(
        len(hidden_states), device=hidden_states.device
    ).repeat_interleave(mask.sum(dim=1))[sort_order]
    permuted_hidden_states = hidden_states.index_select(0, permuted_indices)
    permuted_scores = valid_scores[sort_order]

    return permuted_hidden_states, permuted_scores, permuted_indices


def _unpermute_tokens(
    permuted_hidden_states: torch.Tensor,
    permuted_indices: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Reverse permutation applied by _permute_tokens.

    Args:
        permuted_hidden_states: The permuted token tensor [num_all_tokens, hidden_dim]
        permuted_indices: The indices used to permute the tokens [num_all_tokens]
        num_tokens: Number of unique tokens received by the current rank

    Returns:
        Tokens aggregated and restored to their original order [num_tokens, hidden_dim]
    """
    hidden_dim = permuted_hidden_states.shape[1]
    output_hidden_states = torch.zeros(
        (num_tokens, hidden_dim),
        dtype=permuted_hidden_states.dtype,
        device=permuted_hidden_states.device,
    )
    output_hidden_states.scatter_add_(
        0, permuted_indices.unsqueeze(1).expand(-1, hidden_dim), permuted_hidden_states
    )
    return output_hidden_states


@dataclass
class DispatchState:
    """State from dispatch needed for combine."""

    handle_id: torch.Tensor  # CPU tensor used to retrieve cached handle
    permuted_indices: torch.Tensor
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

    num_recv_tokens = hidden_states.shape[0]

    hidden_states, permuted_scores, permuted_indices = _permute_tokens(
        hidden_states, dispatched_indices, dispatched_expert_scores
    )

    # num_tokens_per_expert is returned from dispatch as int32 on CPU, move to GPU
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
        permuted_indices=permuted_indices,
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
        hidden_states, state.permuted_indices, state.num_recv_tokens
    )

    hidden_states = torch.ops.deepep.combine(hidden_states, state.handle_id)

    return hidden_states


# ============================================================================
# Tuned Config Management
# ============================================================================


def set_tuned_configs(
    dispatch_config: Optional[Config] = None,
    combine_config: Optional[Config] = None,
) -> None:
    """Set the tuned configs for dispatch and combine operations."""
    global _tuned_dispatch_config, _tuned_combine_config
    _tuned_dispatch_config = dispatch_config
    _tuned_combine_config = combine_config


def get_tuned_configs() -> Tuple[Optional[Config], Optional[Config]]:
    """Get the current tuned configs."""
    return _tuned_dispatch_config, _tuned_combine_config


# ============================================================================
# Auto-tuning
# ============================================================================


@dataclass
class AutotuneResult:
    """Result from autotuning."""

    dispatch_config: Tuple[int, ...]
    combine_config: Tuple[int, ...]
    dispatch_bandwidth_gbps: float
    combine_bandwidth_gbps: float
    is_internode: bool = False
    best_dispatch_sms: int = 24
    best_combine_sms: int = 24
    worst_dispatch_config: Optional[Tuple[int, ...]] = None
    worst_dispatch_bandwidth_gbps: float = 0.0
    worst_combine_config: Optional[Tuple[int, ...]] = None
    worst_combine_bandwidth_gbps: float = 0.0


def _bench_fn(fn, warmup: int = 3, repeat: int = 5) -> float:
    """Benchmark a function and return average time in seconds.

    Raises Exception on CUDA errors so callers can skip bad configs.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()

    elapsed = end - start
    return elapsed / repeat if elapsed > 0 else float("inf")


def _detect_internode(buffer: Buffer) -> Tuple[bool, int, int]:
    """
    Detect if communication requires internode (RDMA) or is intranode only (NVLink).

    Returns:
        (is_internode, num_local_ranks, num_nodes)
    """
    import os

    num_ranks = buffer.group_size
    num_rdma_ranks = buffer.runtime.get_num_rdma_ranks()
    is_internode = num_rdma_ranks > 1

    if is_internode:
        num_nodes = num_rdma_ranks
        local_world_size = num_ranks // num_nodes
    else:
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", num_ranks))
        num_nodes = 1

    return is_internode, local_world_size, num_nodes


def _get_gpu_sm_range(default_sms: int = 24) -> list:
    """Auto-detect GPU type and return appropriate SM search range."""
    try:
        gpu_name = torch.cuda.get_device_name(0).lower()
        if "b200" in gpu_name or "b100" in gpu_name:
            return [24, 32, 48, 64]
        elif "h200" in gpu_name or "h100" in gpu_name:
            return [16, 20, 24, 28, 32]
        elif "a100" in gpu_name:
            return [16, 20, 24, 28, 32]
        else:
            return [default_sms]
    except Exception:
        return [default_sms]


def autotune_deepep(
    group: ProcessGroup,
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_topk: int,
    nvl_buffer_size: int = 512,
    rdma_buffer_size: int = 128,
    warmup: int = 5,
    repeat: int = 10,
    verbose: bool = False,
) -> AutotuneResult:
    """
    Autotune DeepEP dispatch/combine configs before training.

    Searches over all tunable parameters:
        - num_sms: Auto-detected range based on GPU type
        - nvl_chunk: NVLink chunk size
        - rdma_chunk: RDMA chunk size (for internode only)

    Args:
        group: Process group for EP communication
        num_tokens: Number of tokens per batch (batch_size * seq_len)
        hidden: Hidden dimension
        num_experts: Total number of experts
        num_topk: Top-k experts per token
        nvl_buffer_size: NVLink buffer size (default: 512)
        rdma_buffer_size: RDMA buffer size for internode (default: 128)
        warmup: Warmup iterations (default: 5)
        repeat: Benchmark iterations (default: 10)
        verbose: Print every config result (default: False)

    Returns:
        AutotuneResult with optimal configs
    """
    rank = torch.distributed.get_rank(group)
    num_ranks = group.size()

    sms_range = _get_gpu_sm_range()

    # Create synthetic test data
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs()
        + 1
    )
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]

    buffer = get_buffer(group, get_hidden_bytes(x))

    is_internode, num_local_ranks, num_nodes = _detect_internode(buffer)

    # Get dispatch layout
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(topk_idx, num_experts)

    # Search space
    if is_internode:
        # Use conservative ranges for internode to avoid DeepEP dispatch
        # timeouts that fatally corrupt CUDA state. Small nvl/rdma chunks
        # can cause NVL receiver timeouts on internode.
        nvl_dispatch_range = list(range(16, 48, 4))
        rdma_dispatch_range = list(range(8, 36, 4))
        nvl_combine_range = list(range(4, 16, 2))
        rdma_combine_range = list(range(8, 36, 4))
    else:
        nvl_dispatch_range = list(range(4, 34, 2))
        rdma_dispatch_range = [16]  # dummy for intranode
        nvl_combine_range = list(range(1, 17, 1))
        rdma_combine_range = [16]  # dummy for intranode

    num_dispatch_configs = (
        len(sms_range) * len(nvl_dispatch_range) * len(rdma_dispatch_range)
    )
    num_combine_configs = (
        len(sms_range) * len(nvl_combine_range) * len(rdma_combine_range)
    )
    total_configs = num_dispatch_configs + num_combine_configs

    if rank == 0:
        gpu_name = torch.cuda.get_device_name(0)
        mode_str = (
            f"internode ({num_nodes} nodes)"
            if is_internode
            else f"intranode ({num_ranks} GPUs)"
        )
        logger.info(f"[DeepEP Autotune] {mode_str} on {gpu_name}")
        logger.info(
            f"[DeepEP Autotune] tokens={num_tokens}, hidden={hidden}, "
            f"experts={num_experts}, topk={num_topk}"
        )
        logger.info(f"[DeepEP Autotune] Search space: sms={sms_range}")
        if is_internode:
            logger.info(
                f"  dispatch: nvl={nvl_dispatch_range[0]}-{nvl_dispatch_range[-1]}"
                f" ({len(nvl_dispatch_range)}), rdma={rdma_dispatch_range[0]}"
                f"-{rdma_dispatch_range[-1]} ({len(rdma_dispatch_range)})"
            )
            logger.info(
                f"  combine:  nvl={nvl_combine_range[0]}-{nvl_combine_range[-1]}"
                f" ({len(nvl_combine_range)}), rdma={rdma_combine_range[0]}"
                f"-{rdma_combine_range[-1]} ({len(rdma_combine_range)})"
            )
        else:
            logger.info(
                f"  dispatch: nvl={nvl_dispatch_range[0]}-{nvl_dispatch_range[-1]}"
                f" ({len(nvl_dispatch_range)})"
            )
            logger.info(
                f"  combine:  nvl={nvl_combine_range[0]}-{nvl_combine_range[-1]}"
                f" ({len(nvl_combine_range)})"
            )
        logger.info(
            f"[DeepEP Autotune] Total: {num_dispatch_configs} dispatch + "
            f"{num_combine_configs} combine = {total_configs} configs"
        )
        logger.info(f"[DeepEP Autotune] warmup={warmup}, repeat={repeat}")

    def make_config(sms: int, nvl_chunk: int, rdma_chunk: int = 16) -> Config:
        if is_internode:
            return Config(sms, nvl_chunk, nvl_buffer_size, rdma_chunk, rdma_buffer_size)
        else:
            return Config(sms, nvl_chunk, nvl_buffer_size)

    # Initial dispatch to get handle and calculate bytes.
    # Use middle-of-range values for initial dispatch to avoid timeouts
    # on internode (small nvl/rdma chunks can timeout).
    init_nvl = nvl_dispatch_range[len(nvl_dispatch_range) // 2]
    init_rdma = rdma_dispatch_range[len(rdma_dispatch_range) // 2]
    initial_config = make_config(sms_range[0], init_nvl, init_rdma)
    recv_x, _, _, _, handle, _ = buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=scores.gather(1, topk_idx).float(),
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        config=initial_config,
    )
    torch.cuda.synchronize()

    dispatch_recv_bytes = recv_x.numel() * 2  # bfloat16
    combine_send_bytes = recv_x.numel() * 2

    autotune_start = time.time()

    if is_internode:
        # ====================================================================
        # INTERNODE: Joint tuning of num_sms for best combined dispatch+combine
        # ====================================================================
        # For internode, only use the sms value that the initial dispatch
        # validated (sms_range[0]). Sweeping sms on internode can cause
        # DeepEP dispatch timeouts that fatally corrupt CUDA state
        # (cudaErrorLaunchFailure). Only tune nvl/rdma chunk sizes.
        sms_range = [sms_range[0]]

        if rank == 0:
            logger.info(
                "[DeepEP Autotune] Internode: tuning nvl/rdma chunks "
                f"at num_sms={sms_range[0]}..."
            )

        best_combined_time = float("inf")
        best_sms = sms_range[0]
        best_dispatch_nvl = nvl_dispatch_range[0]
        best_dispatch_rdma = rdma_dispatch_range[0]
        best_combine_nvl = nvl_combine_range[0]
        best_combine_rdma = rdma_combine_range[0]
        worst_dispatch_time = 0.0
        worst_combine_time = 0.0
        worst_dispatch_nvl = nvl_dispatch_range[0]
        worst_dispatch_rdma = rdma_dispatch_range[0]
        worst_combine_nvl = nvl_combine_range[0]
        worst_combine_rdma = rdma_combine_range[0]
        best_dispatch_time = float("inf")
        best_combine_time = float("inf")

        for sms in sms_range:
            if rank == 0:
                logger.info(f"[DeepEP Autotune] Testing sms={sms}...")

            # Find best dispatch config for this sms
            sms_best_dispatch_time = float("inf")
            sms_best_dispatch_nvl = nvl_dispatch_range[0]
            sms_best_dispatch_rdma = rdma_dispatch_range[0]

            for nvl_chunk in nvl_dispatch_range:
                for rdma_chunk in rdma_dispatch_range:
                    config = make_config(sms, nvl_chunk, rdma_chunk)

                    def dispatch_fn():
                        buffer.dispatch(
                            x,
                            topk_idx=topk_idx,
                            topk_weights=scores.gather(1, topk_idx).float(),
                            num_tokens_per_rank=num_tokens_per_rank,
                            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                            is_token_in_rank=is_token_in_rank,
                            num_tokens_per_expert=num_tokens_per_expert,
                            config=config,
                        )

                    try:
                        t = _bench_fn(dispatch_fn, warmup, repeat)
                    except RuntimeError:
                        continue

                    if t < sms_best_dispatch_time:
                        sms_best_dispatch_time = t
                        sms_best_dispatch_nvl = nvl_chunk
                        sms_best_dispatch_rdma = rdma_chunk
                    if t > worst_dispatch_time:
                        worst_dispatch_time = t
                        worst_dispatch_nvl = nvl_chunk
                        worst_dispatch_rdma = rdma_chunk

            if sms_best_dispatch_time == float("inf"):
                continue

            # Re-dispatch with best config to get handle for combine
            dispatch_cfg = make_config(
                sms, sms_best_dispatch_nvl, sms_best_dispatch_rdma
            )
            try:
                recv_x, _, _, _, handle, _ = buffer.dispatch(
                    x,
                    topk_idx=topk_idx,
                    topk_weights=scores.gather(1, topk_idx).float(),
                    num_tokens_per_rank=num_tokens_per_rank,
                    num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                    is_token_in_rank=is_token_in_rank,
                    num_tokens_per_expert=num_tokens_per_expert,
                    config=dispatch_cfg,
                )
            except RuntimeError:
                continue

            # Find best combine config for this sms
            sms_best_combine_time = float("inf")
            sms_best_combine_nvl = nvl_combine_range[0]
            sms_best_combine_rdma = rdma_combine_range[0]

            for nvl_chunk in nvl_combine_range:
                for rdma_chunk in rdma_combine_range:
                    config = make_config(sms, nvl_chunk, rdma_chunk)

                    def combine_fn():
                        buffer.combine(recv_x, handle=handle, config=config)

                    try:
                        t = _bench_fn(combine_fn, warmup, repeat)
                    except RuntimeError:
                        continue

                    if t < sms_best_combine_time:
                        sms_best_combine_time = t
                        sms_best_combine_nvl = nvl_chunk
                        sms_best_combine_rdma = rdma_chunk
                    if t > worst_combine_time:
                        worst_combine_time = t
                        worst_combine_nvl = nvl_chunk
                        worst_combine_rdma = rdma_chunk

            if sms_best_combine_time == float("inf"):
                continue

            combined_time = sms_best_dispatch_time + sms_best_combine_time
            dispatch_bw = dispatch_recv_bytes / 1e9 / sms_best_dispatch_time
            combine_bw = combine_send_bytes / 1e9 / sms_best_combine_time

            if rank == 0:
                logger.info(
                    f"  sms={sms}: dispatch={dispatch_bw:.1f} GB/s,"
                    f" combine={combine_bw:.1f} GB/s,"
                    f" total={combined_time*1000:.2f}ms"
                )

            if combined_time < best_combined_time:
                best_combined_time = combined_time
                best_sms = sms
                best_dispatch_nvl = sms_best_dispatch_nvl
                best_dispatch_rdma = sms_best_dispatch_rdma
                best_dispatch_time = sms_best_dispatch_time
                best_combine_nvl = sms_best_combine_nvl
                best_combine_rdma = sms_best_combine_rdma
                best_combine_time = sms_best_combine_time

        best_dispatch_sms = best_sms
        best_combine_sms = best_sms

        total_elapsed = time.time() - autotune_start
        if rank == 0:
            best_dispatch_bw = (
                dispatch_recv_bytes / 1e9 / best_dispatch_time
                if best_dispatch_time > 0
                else 0
            )
            best_combine_bw = (
                combine_send_bytes / 1e9 / best_combine_time
                if best_combine_time > 0
                else 0
            )
            logger.info(
                f"[DeepEP Autotune] Internode tuning done in {total_elapsed:.1f}s"
            )
            logger.info(
                f"  Best sms={best_sms}: dispatch nvl={best_dispatch_nvl},"
                f" rdma={best_dispatch_rdma} -> {best_dispatch_bw:.1f} GB/s"
            )
            logger.info(
                f"  Best sms={best_sms}: combine nvl={best_combine_nvl},"
                f" rdma={best_combine_rdma} -> {best_combine_bw:.1f} GB/s"
            )

    else:
        # ====================================================================
        # INTRANODE: Independent tuning of dispatch and combine
        # ====================================================================

        # --- Tune dispatch ---
        best_dispatch_time = float("inf")
        worst_dispatch_time = 0.0
        best_dispatch_sms = sms_range[0]
        best_dispatch_nvl = nvl_dispatch_range[0]
        best_dispatch_rdma = rdma_dispatch_range[0]
        worst_dispatch_nvl = nvl_dispatch_range[0]
        worst_dispatch_rdma = rdma_dispatch_range[0]

        if rank == 0:
            logger.info(
                f"[DeepEP Autotune] Tuning dispatch ({num_dispatch_configs} configs)..."
            )

        dispatch_start = time.time()
        tested = 0
        skipped = 0

        for sms in sms_range:
            for nvl_chunk in nvl_dispatch_range:
                for rdma_chunk in rdma_dispatch_range:
                    config = make_config(sms, nvl_chunk, rdma_chunk)

                    def dispatch_fn():
                        buffer.dispatch(
                            x,
                            topk_idx=topk_idx,
                            topk_weights=scores.gather(1, topk_idx).float(),
                            num_tokens_per_rank=num_tokens_per_rank,
                            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                            is_token_in_rank=is_token_in_rank,
                            num_tokens_per_expert=num_tokens_per_expert,
                            config=config,
                        )

                    try:
                        t = _bench_fn(dispatch_fn, warmup, repeat)
                        tested += 1
                    except RuntimeError:
                        skipped += 1
                        continue

                    bw = dispatch_recv_bytes / 1e9 / t if t > 0 else 0

                    if verbose and rank == 0:
                        logger.info(
                            f"  [{tested}/{num_dispatch_configs}] sms={sms:2d},"
                            f" nvl={nvl_chunk:2d}: {bw:6.1f} GB/s"
                        )

                    if t < best_dispatch_time:
                        best_dispatch_time = t
                        best_dispatch_sms = sms
                        best_dispatch_nvl = nvl_chunk
                        best_dispatch_rdma = rdma_chunk
                    if t > worst_dispatch_time:
                        worst_dispatch_time = t
                        worst_dispatch_nvl = nvl_chunk
                        worst_dispatch_rdma = rdma_chunk

        dispatch_elapsed = time.time() - dispatch_start
        if rank == 0:
            best_bw = (
                dispatch_recv_bytes / 1e9 / best_dispatch_time
                if best_dispatch_time > 0
                else 0
            )
            logger.info(
                f"[DeepEP Autotune] Dispatch done: {tested} tested, "
                f"{skipped} skipped in {dispatch_elapsed:.1f}s"
            )
            logger.info(
                f"  Best: sms={best_dispatch_sms}, nvl={best_dispatch_nvl}"
                f" -> {best_bw:.1f} GB/s"
            )

        # Re-dispatch with best config to get handle for combine tuning
        best_dispatch_cfg = make_config(
            best_dispatch_sms, best_dispatch_nvl, best_dispatch_rdma
        )
        recv_x, _, _, _, handle, _ = buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=scores.gather(1, topk_idx).float(),
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            config=best_dispatch_cfg,
        )

        # --- Tune combine ---
        best_combine_time = float("inf")
        worst_combine_time = 0.0
        best_combine_sms = sms_range[0]
        best_combine_nvl = nvl_combine_range[0]
        best_combine_rdma = rdma_combine_range[0]
        worst_combine_nvl = nvl_combine_range[0]
        worst_combine_rdma = rdma_combine_range[0]

        num_combine_configs = (
            len(sms_range) * len(nvl_combine_range) * len(rdma_combine_range)
        )

        if rank == 0:
            logger.info(
                f"[DeepEP Autotune] Tuning combine ({num_combine_configs} configs)..."
            )

        combine_start = time.time()
        tested = 0
        skipped = 0
        current_sms = best_dispatch_sms

        for sms in sms_range:
            if sms != current_sms:
                try:
                    dispatch_cfg = make_config(
                        sms, best_dispatch_nvl, best_dispatch_rdma
                    )
                    recv_x, _, _, _, handle, _ = buffer.dispatch(
                        x,
                        topk_idx=topk_idx,
                        topk_weights=scores.gather(1, topk_idx).float(),
                        num_tokens_per_rank=num_tokens_per_rank,
                        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                        is_token_in_rank=is_token_in_rank,
                        num_tokens_per_expert=num_tokens_per_expert,
                        config=dispatch_cfg,
                    )
                    current_sms = sms
                except RuntimeError:
                    skipped += len(nvl_combine_range) * len(rdma_combine_range)
                    continue

            for nvl_chunk in nvl_combine_range:
                for rdma_chunk in rdma_combine_range:
                    config = make_config(sms, nvl_chunk, rdma_chunk)

                    def combine_fn():
                        buffer.combine(recv_x, handle=handle, config=config)

                    try:
                        t = _bench_fn(combine_fn, warmup, repeat)
                        tested += 1
                    except RuntimeError:
                        skipped += 1
                        continue

                    bw = combine_send_bytes / 1e9 / t if t > 0 else 0

                    if verbose and rank == 0:
                        logger.info(
                            f"  [{tested}/{num_combine_configs}] sms={sms:2d},"
                            f" nvl={nvl_chunk:2d}: {bw:6.1f} GB/s"
                        )

                    if t < best_combine_time:
                        best_combine_time = t
                        best_combine_sms = sms
                        best_combine_nvl = nvl_chunk
                        best_combine_rdma = rdma_chunk
                    if t > worst_combine_time:
                        worst_combine_time = t
                        worst_combine_nvl = nvl_chunk
                        worst_combine_rdma = rdma_chunk

        combine_elapsed = time.time() - combine_start
        total_elapsed = time.time() - autotune_start
        if rank == 0:
            best_bw = (
                combine_send_bytes / 1e9 / best_combine_time
                if best_combine_time > 0
                else 0
            )
            logger.info(
                f"[DeepEP Autotune] Combine done: {tested} tested, "
                f"{skipped} skipped in {combine_elapsed:.1f}s"
            )
            logger.info(
                f"  Best: sms={best_combine_sms}, nvl={best_combine_nvl}"
                f" -> {best_bw:.1f} GB/s"
            )

    # Calculate bandwidths
    best_dispatch_bw = (
        dispatch_recv_bytes / 1e9 / best_dispatch_time if best_dispatch_time > 0 else 0
    )
    worst_dispatch_bw = (
        dispatch_recv_bytes / 1e9 / worst_dispatch_time
        if worst_dispatch_time > 0
        else 0
    )
    best_combine_bw = (
        combine_send_bytes / 1e9 / best_combine_time if best_combine_time > 0 else 0
    )
    worst_combine_bw = (
        combine_send_bytes / 1e9 / worst_combine_time if worst_combine_time > 0 else 0
    )

    # Build result configs
    if is_internode:
        dispatch_cfg_tuple = (
            best_dispatch_nvl,
            nvl_buffer_size,
            best_dispatch_rdma,
            rdma_buffer_size,
        )
        combine_cfg_tuple = (
            best_combine_nvl,
            nvl_buffer_size,
            best_combine_rdma,
            rdma_buffer_size,
        )
        worst_dispatch_cfg = (
            worst_dispatch_nvl,
            nvl_buffer_size,
            worst_dispatch_rdma,
            rdma_buffer_size,
        )
        worst_combine_cfg = (
            worst_combine_nvl,
            nvl_buffer_size,
            worst_combine_rdma,
            rdma_buffer_size,
        )
    else:
        dispatch_cfg_tuple = (best_dispatch_nvl, nvl_buffer_size)
        combine_cfg_tuple = (best_combine_nvl, nvl_buffer_size)
        worst_dispatch_cfg = (worst_dispatch_nvl, nvl_buffer_size)
        worst_combine_cfg = (worst_combine_nvl, nvl_buffer_size)

    result = AutotuneResult(
        dispatch_config=dispatch_cfg_tuple,
        combine_config=combine_cfg_tuple,
        dispatch_bandwidth_gbps=best_dispatch_bw,
        combine_bandwidth_gbps=best_combine_bw,
        is_internode=is_internode,
        best_dispatch_sms=best_dispatch_sms,
        best_combine_sms=best_combine_sms,
        worst_dispatch_config=worst_dispatch_cfg,
        worst_dispatch_bandwidth_gbps=worst_dispatch_bw,
        worst_combine_config=worst_combine_cfg,
        worst_combine_bandwidth_gbps=worst_combine_bw,
    )

    # Set global configs with best num_sms values
    best_dispatch_config = make_config(
        best_dispatch_sms, best_dispatch_nvl, best_dispatch_rdma
    )
    best_combine_config = make_config(
        best_combine_sms, best_combine_nvl, best_combine_rdma
    )
    set_tuned_configs(
        dispatch_config=best_dispatch_config,
        combine_config=best_combine_config,
    )

    # Print summary on rank 0
    if rank == 0:
        dispatch_speedup = (
            best_dispatch_bw / worst_dispatch_bw if worst_dispatch_bw > 0 else 1.0
        )
        combine_speedup = (
            best_combine_bw / worst_combine_bw if worst_combine_bw > 0 else 1.0
        )

        logger.info("=" * 70)
        logger.info(f"[DeepEP Autotune] RESULTS (total time: {total_elapsed:.1f}s)")
        logger.info("=" * 70)

        if is_internode:
            logger.info(f"  DISPATCH (num_sms={best_dispatch_sms}):")
            logger.info(
                f"    Best:  nvl={best_dispatch_nvl:2d}, rdma={best_dispatch_rdma:2d}"
                f" -> {best_dispatch_bw:6.1f} GB/s"
            )
            logger.info(
                f"    Worst: nvl={worst_dispatch_nvl:2d}, rdma={worst_dispatch_rdma:2d}"
                f" -> {worst_dispatch_bw:6.1f} GB/s"
            )
            logger.info(f"    Speedup: {dispatch_speedup:.2f}x over worst config")
            logger.info(f"  COMBINE (num_sms={best_combine_sms}):")
            logger.info(
                f"    Best:  nvl={best_combine_nvl:2d}, rdma={best_combine_rdma:2d}"
                f" -> {best_combine_bw:6.1f} GB/s"
            )
            logger.info(
                f"    Worst: nvl={worst_combine_nvl:2d}, rdma={worst_combine_rdma:2d}"
                f" -> {worst_combine_bw:6.1f} GB/s"
            )
            logger.info(f"    Speedup: {combine_speedup:.2f}x over worst config")
        else:
            logger.info(f"  DISPATCH (num_sms={best_dispatch_sms}):")
            logger.info(
                f"    Best:  nvl={best_dispatch_nvl:2d}"
                f" -> {best_dispatch_bw:6.1f} GB/s"
            )
            logger.info(
                f"    Worst: nvl={worst_dispatch_nvl:2d}"
                f" -> {worst_dispatch_bw:6.1f} GB/s"
            )
            logger.info(f"    Speedup: {dispatch_speedup:.2f}x over worst config")
            logger.info(f"  COMBINE (num_sms={best_combine_sms}):")
            logger.info(
                f"    Best:  nvl={best_combine_nvl:2d}"
                f" -> {best_combine_bw:6.1f} GB/s"
            )
            logger.info(
                f"    Worst: nvl={worst_combine_nvl:2d}"
                f" -> {worst_combine_bw:6.1f} GB/s"
            )
            logger.info(f"    Speedup: {combine_speedup:.2f}x over worst config")

        logger.info("=" * 70)

    return result


def run_deepep_autotune_if_enabled(
    deepep_config,
    ep_group: ProcessGroup,
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_topk: int,
) -> Optional[AutotuneResult]:
    """
    Run DeepEP autotune if enabled in config.

    Should be called after EP process group is created and before training begins.

    Args:
        deepep_config: The deepep config from job_config.deepep
        ep_group: Expert parallelism process group
        num_tokens: Number of tokens per micro-batch (batch_size * seq_len)
        hidden: Model hidden dimension
        num_experts: Total number of MoE experts
        num_topk: Top-k experts per token
    """
    if deepep_config is None or not getattr(deepep_config, "autotune", False):
        # Autotune not enabled, set manual configs from deepep_config
        num_sms = getattr(deepep_config, "num_sms", 24) if deepep_config else 24
        nvl_buffer = (
            getattr(deepep_config, "nvl_buffer_size", 256) if deepep_config else 256
        )
        rdma_buffer = (
            getattr(deepep_config, "rdma_buffer_size", 128) if deepep_config else 128
        )

        # Detect internode vs intranode to set correct Config format
        # Create a temporary buffer to probe topology
        hidden_bytes = hidden * 2  # bfloat16
        buffer = get_buffer(ep_group, hidden_bytes)
        is_internode, _, num_nodes = _detect_internode(buffer)

        if is_internode:
            # Internode: Config(sms, nvl_chunk, nvl_buffer, rdma_chunk, rdma_buffer)
            default_dispatch = (6, nvl_buffer, 8, rdma_buffer)
            default_combine = (4, nvl_buffer, 8, rdma_buffer)
        else:
            # Intranode: Config(sms, nvl_chunk, nvl_buffer)
            default_dispatch = (6, nvl_buffer)
            default_combine = (4, nvl_buffer)

        Buffer.set_num_sms(num_sms)
        set_tuned_configs(
            dispatch_config=Config(num_sms, *default_dispatch),
            combine_config=Config(num_sms, *default_combine),
        )

        rank = torch.distributed.get_rank(ep_group) if ep_group else 0
        if rank == 0:
            mode_str = (
                f"internode ({num_nodes} nodes)"
                if is_internode
                else "intranode"
            )
            logger.info(
                f"DeepEP using default configs ({mode_str}): num_sms={num_sms}, "
                f"dispatch={default_dispatch}, combine={default_combine}"
            )
        return None

    # Run autotune
    nvl_buffer = getattr(deepep_config, "nvl_buffer_size", 512)
    rdma_buffer = getattr(deepep_config, "rdma_buffer_size", 128)
    warmup = getattr(deepep_config, "autotune_warmup", 5)
    repeat = getattr(deepep_config, "autotune_repeat", 10)
    verbose = getattr(deepep_config, "autotune_verbose", False)

    rank = torch.distributed.get_rank(ep_group)
    if rank == 0:
        logger.info(
            f"Running DeepEP autotune: tokens={num_tokens}, hidden={hidden}, "
            f"experts={num_experts}, topk={num_topk}"
        )

    try:
        result = autotune_deepep(
            group=ep_group,
            num_tokens=num_tokens,
            hidden=hidden,
            num_experts=num_experts,
            num_topk=num_topk,
            nvl_buffer_size=nvl_buffer,
            rdma_buffer_size=rdma_buffer,
            warmup=warmup,
            repeat=repeat,
            verbose=verbose,
        )

        # Barrier to ensure all ranks complete autotune before training
        torch.distributed.barrier(ep_group)

        return result
    except Exception as e:
        # DeepEP internode dispatch timeouts can fatally corrupt CUDA state
        # (cudaErrorLaunchFailure). This is unrecoverable — the process must
        # be restarted. Log the error so the user knows to disable autotune
        # or narrow the search ranges for internode.
        if rank == 0:
            logger.error(
                f"[DeepEP Autotune] Fatal error: {type(e).__name__}: {e}. "
                f"CUDA state may be corrupted. For internode, consider running "
                f"with autotune=false and manually setting num_sms/nvl_buffer_size."
            )
        raise
