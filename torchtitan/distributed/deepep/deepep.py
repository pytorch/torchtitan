# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
DeepEP primitives for MoE Expert Parallel, in two modes (selected by
``DeepEPTokenDispatcher.Config.low_latency`` in token_dispatcher.py):

``dispatch_tokens``/``combine_tokens`` are the single entry points; their ``low_latency``
argument selects the mode:

- High-throughput (HT, ``low_latency=False``): dynamic-shape ``buffer.dispatch``/``combine``
  with full autograd (training) and SAC-integrated custom ops. Has a CPU-waits-for-GPU
  handoff, so it is NOT cudagraph-able.
- Low-latency (LL, ``low_latency=True``): masked, static-shape
  ``buffer.low_latency_dispatch``/``combine`` (slab
  ``[num_local_experts, num_max_tokens * num_ranks, hidden]``) with a device-side receive
  hook, so the whole MoE forward can be captured in a CUDA graph (vLLM FULL_DECODE_ONLY).
  Inference-only (no autograd). Requires IBGDA; on grandteton RoCE set
  ``NVSHMEM_IB_GID_INDEX=3``.
"""

from dataclasses import dataclass

import torch
from torch.distributed import ProcessGroup
from torch.utils._python_dispatch import _disable_current_modes

try:
    from deep_ep import Buffer
except ImportError as e:
    raise ImportError(
        "DeepEP is required for this module. "
        "Install from: https://github.com/deepseek-ai/deepep"
    ) from e

# EventHandle/EventOverlap are used ONLY by the high-throughput (HT) custom-op autograd path,
# not by low-latency (LL). Some deep_ep builds omit them (e.g. EventHandle is absent on the
# inference build), so import them tolerantly -- this keeps the LL path (which needs only
# Buffer) working. HT functions that use them run only on builds that provide them.
try:
    from deep_ep import EventHandle, EventOverlap
except ImportError:
    EventHandle = EventOverlap = None


# Global buffer (single buffer per process, recreated if group changes)
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
_pending_combine_event: "EventOverlap | None" = None


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    )

    after_event.current_stream_wait()

    return grad_x, None


torch.library.register_autograd(
    "deepep::dispatch", _dispatch_backward, setup_context=_dispatch_setup_context
)
torch.library.register_autograd(
    "deepep::combine", _combine_backward, setup_context=_combine_setup_context
)


def sync_combine() -> None:
    """Synchronize the current CUDA stream with the pending combine operation.

    This function MUST be called before using the result of combine_tokens()
    to ensure the async combine has completed. It inserts a wait operation
    on the current CUDA stream, making subsequent CUDA kernels wait for
    the combine to finish.

    torch.compile Compatibility:
        Guarded with is_compiling() to skip in compile mode.
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
    # is_compiling() is True under Dynamo (aot mode) — skip to avoid tracing
    # through global state Dynamo can't handle. Under make_fx (aot_fx_trace),
    # is_compiling() is False so this guard doesn't fire, but the function is
    # still safe: _pending_combine_event is None during tracing (no real
    # combine ran), so the body below is a no-op.
    if torch.compiler.is_compiling():
        return

    if _pending_combine_event is not None:
        _pending_combine_event.current_stream_wait()
        _pending_combine_event = None


def get_hidden_bytes(x: torch.Tensor) -> int:
    """Calculate the number of hidden bytes for one token."""
    # Use at least 2 bytes (bf16 size) so buffer works for both fp8 and bf16 without reallocation
    return x.size(1) * max(x.element_size(), 2)


def get_buffer(
    group: ProcessGroup,
    *,
    low_latency: bool = False,
    hidden_bytes: int = 0,
    num_max_dispatch_tokens_per_rank: int = 0,
    ll_buffer_hidden_size: int = 0,
    num_experts: int = 0,
) -> Buffer:
    """Get or create the process-global DeepEP buffer, in HT or LL mode.

    HT and LL use DIFFERENT deep_ep ``Buffer`` modes, fixed at construction, so a process uses
    one or the other; this single global holds whichever the model's dispatcher requested.

    HT (``low_latency=False``): normal-mode buffer (NVL+RDMA) sized by the dispatch/combine
    config hints from ``hidden_bytes``; recreated if the group changes or a larger size is
    needed.

    LL (``low_latency=True``): low-latency-mode buffer (RDMA-only, ``num_qps_per_rank ==
    num_local_experts``) sized by ``get_low_latency_rdma_size_hint`` from
    ``num_max_dispatch_tokens_per_rank``/``ll_buffer_hidden_size``/``num_experts``. CREATE-ONCE with
    ``explicitly_destroy=True`` so ``deep_ep::Buffer::~Buffer()`` does NOT auto-run
    ``destroy()`` (-> ``cudaDeviceSynchronize`` + host ``nvshmem_barrier_all``) on GC: that
    barrier inside a CUDA graph capture aborts with "team_internal.cpp ... cuda failed with
    invalid argument". We never call ``destroy()`` (leaking at process exit is fine). Matches
    vLLM's DeepEP buffer usage.
    """
    global _buffer
    if low_latency:
        # LL buffer is fixed-size for the whole run -> create once and reuse.
        if _buffer is not None:
            return _buffer
        num_nvl_bytes = 0
        num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank,
            ll_buffer_hidden_size,
            group.size(),
            num_experts,
        )
        extra_kwargs = dict(
            low_latency_mode=True,
            num_qps_per_rank=num_experts // group.size(),
            explicitly_destroy=True,
        )
    else:
        # HT -> recreate only if the group changed or a larger buffer is needed.
        num_nvl_bytes, num_rdma_bytes = 0, 0
        for config in (
            Buffer.get_dispatch_config(group.size()),
            Buffer.get_combine_config(group.size()),
        ):
            num_nvl_bytes = max(
                config.get_nvl_buffer_size_hint(hidden_bytes, group.size()),
                num_nvl_bytes,
            )
            num_rdma_bytes = max(
                config.get_rdma_buffer_size_hint(hidden_bytes, group.size()),
                num_rdma_bytes,
            )
        if (
            _buffer is not None
            and _buffer.group == group
            and _buffer.num_nvl_bytes >= num_nvl_bytes
            and _buffer.num_rdma_bytes >= num_rdma_bytes
        ):
            return _buffer
        extra_kwargs = {}

    _buffer = Buffer(
        group,
        num_nvl_bytes=num_nvl_bytes,
        num_rdma_bytes=num_rdma_bytes,
        **extra_kwargs,
    )
    return _buffer


def _permute_tokens(
    hidden_states: torch.Tensor,
    dispatched_indices: torch.Tensor,
    dispatched_scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    permuted_scores: torch.Tensor | None = None


@dataclass
class LLDispatchState:
    """State from low-latency (LL) dispatch needed for combine."""

    handle: object  # opaque DeepEP LL handle (encodes routing for combine)
    topk_idx: torch.Tensor  # [T, K] expert ids per token
    topk_weights: torch.Tensor  # [T, K] routing scores (applied in combine)
    num_local_experts: int  # E (dim 0 of the masked slab)
    tokens_per_slab: int  # M = num_max_dispatch_tokens_per_rank * num_ranks


def dispatch_tokens(
    hidden_states: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    top_scores: torch.Tensor,
    num_local_experts: int,
    num_experts: int,
    group: ProcessGroup,
    *,
    low_latency: bool = False,
    num_max_dispatch_tokens_per_rank: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, "DispatchState | LLDispatchState"]:
    """Dispatch tokens to experts via DeepEP (HT by default; LL when ``low_latency=True``).

    Routing scores are applied to the expert outputs in ``combine_tokens``, after expert
    computation. The returned ``state`` is opaque: pass it back to ``combine_tokens`` with the
    same ``low_latency``. The LL branch ignores ``num_local_experts``; the HT branch ignores
    ``num_max_dispatch_tokens_per_rank``.

    Args:
        hidden_states: Input tokens [num_tokens, hidden_dim]
        selected_experts_indices: Expert indices for each token [num_tokens, top_k]
        top_scores: Routing scores for each token [num_tokens, top_k]
        num_local_experts: Number of experts on this rank (HT only)
        num_experts: Total number of experts across all ranks
        group: EP process group
        low_latency: Use the masked, cudagraph-able LL kernels (inference-only)
        num_max_dispatch_tokens_per_rank: LL static per-rank dispatch capacity

    Returns:
        (routed_tokens, tokens_per_expert, state_for_combine)
    """
    if low_latency:
        # --- Low-latency (LL): masked, static-shape, cudagraph-able (inference-only). ---
        # The masked slab is flattened to [E*M, hidden] with a uniform num_tokens_per_expert
        # = [M, M, ...] so the grouped-mm expert path computes every slot statically; padding
        # slots produce garbage that low_latency_combine discards via the handle.
        with _disable_current_modes():
            buffer = get_buffer(
                group,
                low_latency=True,
                num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
                ll_buffer_hidden_size=hidden_states.shape[1],
                num_experts=num_experts,
            )
        # return_recv_hook=True only ISSUES the RDMA (device-side, capturable); hook() does the
        # device-side receive. Synchronous mode would "ensure arrival" via a host NVSHMEM team
        # barrier (cudaStreamSynchronize) that is illegal mid-capture. recv_x: [E, M, H];
        # recv_count [E] unused (all M slots computed statically; combine masks via the handle).
        recv_x, _recv_count, handle, _event, hook = buffer.low_latency_dispatch(
            hidden_states,
            selected_experts_indices,
            num_max_dispatch_tokens_per_rank,
            num_experts,
            use_fp8=False,
            async_finish=False,
            return_recv_hook=True,
        )
        hook()  # device-side receive; keeps the whole comm inside the CUDA graph
        # recv_x is a masked, static slab [num_local_experts, num_max_tokens * num_ranks,
        # hidden] = [E, M, H]. Dim 0 IS the local-expert axis -- the kernel writes each
        # expert's received tokens directly into that expert's own fixed [M, H] slot, so the
        # output is ALREADY expert-grouped by construction. That is why LL needs no permute
        # (unlike the HT path): flattening to [E*M, H] with a uniform num_tokens_per_expert
        # = [M, M, ...] feeds the grouped-mm expert path directly. Unused padding slots hold
        # garbage that low_latency_combine discards via the handle.
        E, M, H = recv_x.shape
        routed_input_RD = recv_x.reshape(E * M, H)
        num_tokens_per_expert_E = torch.full(
            (E,), M, dtype=torch.int32, device=hidden_states.device
        )
        state = LLDispatchState(
            handle=handle,
            topk_idx=selected_experts_indices,
            topk_weights=top_scores,
            num_local_experts=E,
            tokens_per_slab=M,
        )
        return routed_input_RD, num_tokens_per_expert_E, state

    selected_experts_indices = selected_experts_indices.contiguous()
    top_scores = top_scores.contiguous()

    # Mask out zero-score tokens
    selected_experts_indices = selected_experts_indices.masked_fill(top_scores == 0, -1)

    # Ensure float32 scores (DeepEP requirement)
    if top_scores.dtype != torch.float32:
        top_scores = top_scores.float()

    # Hide buffer setup from SAC's __torch_dispatch__ via _disable_current_modes().
    # Buffer.__init__ calls all_gather_object() which triggers aten._to_copy
    # (CUDA→CPU), a MUST_SAVE op in our SAC policy. These are infrastructure
    # ops, not model compute, and must not enter SAC's FIFO cache.
    with _disable_current_modes():
        buffer = get_buffer(group, hidden_bytes=get_hidden_bytes(hidden_states))

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

    # Routing scores are applied to expert outputs in combine_tokens.
    state = DispatchState(
        handle_id=handle_id,
        permuted_indices=permuted_indices,
        num_recv_tokens=num_recv_tokens,
        permuted_scores=permuted_scores,
    )

    return hidden_states, num_tokens_per_expert, state


def combine_tokens(
    hidden_states: torch.Tensor,
    state: "DispatchState | LLDispatchState",
    *,
    low_latency: bool = False,
    num_experts: int = 0,
    num_max_dispatch_tokens_per_rank: int = 0,
    group: ProcessGroup | None = None,
) -> torch.Tensor:
    """Combine expert outputs back to tokens via DeepEP (HT by default; LL when low_latency).

    HT combine is async (the caller syncs via ``sync_combine()``); LL combine is synchronous.
    ``state`` must be the object returned by ``dispatch_tokens`` with the same ``low_latency``.
    The LL path needs ``num_experts``/``num_max_dispatch_tokens_per_rank``/``group``; HT ignores
    them.

    Args:
        hidden_states: Expert outputs to combine.
        state: Dispatch state from ``dispatch_tokens``.
        low_latency: Use the LL combine kernel (inference-only).
        num_experts: Total experts across ranks (LL only).
        num_max_dispatch_tokens_per_rank: LL static per-rank dispatch capacity.
        group: EP process group (LL only).

    Returns:
        Combined tokens [num_tokens, hidden_dim].
    """
    if low_latency:
        assert group is not None, "group is required for low-latency combine"
        # --- Low-latency (LL) combine: scatter expert outputs back to tokens. ---
        # low_latency_combine applies topk_weights and gathers only the valid slots (via the
        # handle), so the expert output passed in must be the RAW (unweighted) expert(x).
        E, M = state.num_local_experts, state.tokens_per_slab
        H = hidden_states.shape[-1]
        expert_out = hidden_states.reshape(E, M, H)
        with _disable_current_modes():
            buffer = get_buffer(
                group,
                low_latency=True,
                num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
                ll_buffer_hidden_size=H,
                num_experts=num_experts,
            )
        combined, _event, _hook = buffer.low_latency_combine(
            expert_out, state.topk_idx, state.topk_weights.float(), state.handle
        )
        return combined

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
