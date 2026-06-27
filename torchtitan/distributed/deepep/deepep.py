# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
DeepEP v2 primitives for MoE Expert Parallel, on the unified ``ElasticBuffer`` API.

DeepEP v2 (>= 2.0.0) collapses the v1 two-path design -- high-throughput (HT,
``buffer.dispatch``/``combine``) and low-latency (LL,
``buffer.low_latency_dispatch``/``combine``) -- into a SINGLE ``dispatch``/``combine``
on ``deep_ep.ElasticBuffer``. There is one buffer, one pair of custom ops, and one
``DispatchState`` for both modes; only ``dispatch`` branches, selected by the
``cudagraph`` flag (combine is handle-driven and mode-agnostic):

- training / prefill (``cudagraph=False``, default): ``do_expand=False`` +
  ``do_cpu_sync=True`` -- a compact layout ALREADY grouped by local expert
  (``handle.num_recv_tokens_per_expert_list`` gives the per-expert token counts for the
  grouped GEMM), so no manual permute is needed. Full autograd via the custom ops below
  (dispatch backward is a combine, combine backward is a dispatch). The total received
  count is data-dependent and needs a host sync, so this path is NOT cudagraph-able.
- inference / decode (``cudagraph=True``): ``do_expand=True`` + ``do_cpu_sync=False`` --
  the static "one-token-per-expert-slot" expanding layout, routing-independent (correct
  even as gating changes between captured replays) and with no host sync, so the MoE
  forward is cudagraph-capturable. Per-expert offsets come from the device-side
  ``handle.psum_num_recv_tokens_per_expert`` (no CPU sync). Inference-only: the expanding
  layout "must not be backward" per the DeepEP kernels.

Routing scores are applied to expert outputs in plain PyTorch (in ``combine_tokens``,
before the pure-reduction combine op), so autograd handles the score gradient, the custom
ops stay pure communication, and combine works unchanged in both modes (``combine``
ignores ``topk_weights`` in expand mode anyway).
"""

from dataclasses import dataclass

import torch
from torch.distributed import ProcessGroup
from torch.utils._python_dispatch import _disable_current_modes

try:
    from deep_ep import ElasticBuffer
except ImportError as e:
    raise ImportError(
        "DeepEP v2 (>= 2.0.0, ElasticBuffer) is required for this module. "
        "Install from: https://github.com/deepseek-ai/DeepEP"
    ) from e


# Global buffer (single buffer per process, recreated if the group changes or a
# larger size is needed). v2 uses ONE ElasticBuffer for both training and inference.
_buffer: ElasticBuffer | None = None

# Global cache for dispatch handles (EPHandle objects), keyed by an int handle_id.
# The torch.library custom ops can only pass tensors across the op boundary, so we
# smuggle the opaque EPHandle through a CPU int64 handle_id tensor + this cache.
# SAC saves the handle_id tensor; we use it to retrieve the non-tensor handle.
_handle_cache: dict = {}
_handle_counter: int = 0

# Pending combine event for deferred synchronization, so shared_experts compute can
# overlap with the combine communication (the caller MUST call sync_combine() before
# using the combine result). Process-local + single-threaded, so a module var suffices.
_pending_combine_event = None


def _get_next_handle_id() -> torch.Tensor:
    """Generate a unique handle_id tensor on CPU to avoid a GPU-CPU sync."""
    global _handle_counter
    _handle_counter += 1
    return torch.tensor([_handle_counter], dtype=torch.int64, device="cpu")


# ============================================================================
# Custom Op Registration for SAC Integration + autograd
# ============================================================================
#
# ElasticBuffer.dispatch/combine are not autograd-aware. We wrap them in
# torch.library custom ops so (a) SAC saves the comm outputs instead of recomputing
# them and (b) we attach manual backward: dispatch backward is a combine and combine
# backward is a dispatch (the DeepEP forward/backward duality). The opaque EPHandle
# is passed across the op boundary via a CPU handle_id + _handle_cache.

_lib = torch.library.Library("deepep", "DEF")

# dispatch returns: (recv_x, recv_topk_idx, recv_scores, num_recv_per_expert, handle_id).
# recv_topk_idx is the per-received-token local-expert assignment, used by the compact
# path to gather tokens into expert-major order (it is an empty placeholder in expand mode,
# whose static layout is already expert-grouped).
_lib.define(
    "dispatch(Tensor x, Tensor topk_idx, Tensor topk_weights, "
    "int num_experts, int num_max_tokens_per_rank, bool cudagraph) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor)"
)
# combine returns: combined_x. ``will_backward`` is the caller's outer grad state
# (torch.is_grad_enabled() evaluated before the op): it is the only reliable signal for
# whether a backward will consume the cached handle, since inside a custom-op forward
# autograd disables grad regardless of the outer context. When False (generator no_grad /
# inference), the op frees the handle itself (setup_context never runs).
_lib.define("combine(Tensor x, Tensor handle_id, bool will_backward) -> Tensor")


@torch.library.impl(_lib, "dispatch", "CUDA")
def _dispatch_op_impl(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    num_max_tokens_per_rank: int,
    cudagraph: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute DeepEP v2 dispatch.

    ``cudagraph=False`` (training/prefill): the COMPACT (non-expand) layout. ``recv_x`` is
    DEDUPLICATED -- one row per unique received token -- while ``recv_topk_idx`` gives each
    token's local-expert assignments (-1 for picks not on this rank). ``dispatch_tokens``
    gathers this into expert-major order for the grouped GEMM (matching the v1 path). A host
    sync gives exact per-expert counts, so it is NOT cudagraph-able.
    ``cudagraph=True`` (inference/decode): the static ``do_expand`` layout, already
    expert-grouped (tokens packed in ``[0:sum(counts)]``, tail unused) with no host sync, so
    the forward is cudagraph-capturable; per-expert counts come from the device-side
    ``psum_num_recv_tokens_per_expert``.
    """
    global _buffer
    buffer = _buffer
    assert buffer is not None, "Buffer must be initialized before dispatch"

    recv_x, recv_topk_idx, recv_scores, handle, _event = buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        do_expand=cudagraph,
        do_cpu_sync=not cudagraph,
    )

    handle_id = _get_next_handle_id()
    _handle_cache[handle_id.item()] = handle

    # Per-local-expert received-token counts for the grouped GEMM.
    if cudagraph:
        # Expand mode: no host sync allowed. Recover per-expert counts from the
        # device-side inclusive prefix sum (expert_alignment defaults to 1, so this is
        # a plain prefix sum). _experts_forward cumsums these back into grouped-mm offs.
        psum = handle.psum_num_recv_tokens_per_expert
        num_recv_per_expert = torch.diff(psum, prepend=psum.new_zeros(1)).to(
            torch.int32
        )
        # Expand layout is already expert-grouped; no gather needed -> empty placeholder
        # (a torch.library op must return a Tensor, not None).
        recv_topk_idx = recv_x.new_empty(0, dtype=torch.long)
    else:
        # Compact mode: exact counts are a CPU list (available after the host sync).
        num_recv_per_expert = torch.tensor(
            handle.num_recv_tokens_per_expert_list, dtype=torch.int32, device="cpu"
        )
    return recv_x, recv_topk_idx, recv_scores, num_recv_per_expert, handle_id


def _dispatch_setup_context(ctx, inputs, output):
    x, *_ = inputs
    *_, handle_id = output
    ctx.input_dtype = x.dtype
    ctx.saved_handle = _handle_cache.get(handle_id.item())


def _dispatch_backward(
    ctx,
    grad_recv_x,
    grad_recv_topk_idx,
    grad_recv_scores,
    grad_num_recv,
    grad_handle_id,
):
    """Backward for dispatch: a combine of the gradients.

    The combine reduces grad_recv_x back to the original tokens (grad for x); passing
    grad_recv_scores as the combine's topk_weights yields the gradient for the
    dispatched routing scores (DeepEP combine returns the reduced weights too).
    recv_topk_idx is non-differentiable, so grad_recv_topk_idx is ignored.
    """
    global _buffer
    if grad_recv_x is None:
        return None, None, None, None, None, None

    handle = ctx.saved_handle
    assert handle is not None

    grad_x, grad_scores, _event = _buffer.combine(
        grad_recv_x,
        handle=handle,
        topk_weights=grad_recv_scores.float() if grad_recv_scores is not None else None,
    )
    grad_x = grad_x.to(ctx.input_dtype)
    grad_topk_weights = (
        grad_scores.to(ctx.input_dtype) if grad_scores is not None else None
    )
    # Order matches op inputs:
    # x, topk_idx, topk_weights, num_experts, num_max_tokens_per_rank, cudagraph.
    # Backward only runs on the compact (cudagraph=False) path; the expand layout is
    # inference-only ("must not be backward").
    return grad_x, None, grad_topk_weights, None, None, None


@torch.library.impl(_lib, "combine", "CUDA")
def _combine_op_impl(
    x: torch.Tensor, handle_id: torch.Tensor, will_backward: bool
) -> torch.Tensor:
    """Execute DeepEP v2 combine (pure reduction; scores already applied upstream)."""
    global _buffer, _pending_combine_event
    buffer = _buffer
    assert buffer is not None, "Buffer must be initialized before combine"

    # When no backward will run (generator forward under no_grad/inference_mode), the
    # dispatch setup_context never fires to free the handle, so pop it here. When a backward
    # will run (training), keep it: _combine_setup_context pops it for combine-backward.
    # ``will_backward`` is the caller's OUTER grad state -- inside this forward impl
    # torch.is_grad_enabled() is always False (autograd disables grad during forward), so it
    # cannot tell training from inference here.
    if not will_backward:
        handle = _handle_cache.pop(handle_id.item(), None)
    else:
        handle = _handle_cache.get(handle_id.item())
    assert handle is not None, f"Handle not found for handle_id={handle_id.item()}"

    combined, _combined_weights, after_event = buffer.combine(
        x,
        handle=handle,
        topk_weights=None,
        async_with_compute_stream=True,
    )
    # Defer the sync so shared_experts compute can overlap the combine communication.
    _pending_combine_event = after_event
    return combined


def _combine_setup_context(ctx, inputs, output):
    _, handle_id, _will_backward = inputs
    ctx.saved_handle = _handle_cache.pop(handle_id.item(), None)


def _combine_backward(ctx, grad_combined):
    """Backward for combine: a dispatch of the gradient (reuses the cached handle).

    Returns grads for op inputs (x, handle_id, will_backward); only x is differentiable.
    """
    global _buffer
    handle = ctx.saved_handle
    assert handle is not None, "Handle not found in combine backward"

    # Reuse the dispatch layout via the cached handle (no CPU sync, topk_idx/weights None).
    # Pass num_sms from the handle: with a cached handle, dispatch's automatic
    # get_theoretical_num_sms(num_experts, ...) runs BEFORE num_experts is inferred from the
    # handle, so it would hit num_experts=None. handle.num_sms reuses the dispatch SM count.
    grad_x, _idx, _scores, _handle, _event = _buffer.dispatch(
        grad_combined,
        handle=handle,
        num_sms=handle.num_sms,
        do_cpu_sync=False,
    )
    return grad_x, None, None


torch.library.register_autograd(
    "deepep::dispatch", _dispatch_backward, setup_context=_dispatch_setup_context
)
torch.library.register_autograd(
    "deepep::combine", _combine_backward, setup_context=_combine_setup_context
)


def sync_combine() -> None:
    """Wait the current CUDA stream on the pending async combine.

    MUST be called before using a combine result. Guarded under compile (CUDA event
    ops are not traceable); during make_fx tracing _pending_combine_event is None
    (no real combine ran), so the body is a no-op. Safe to call multiple times.
    """
    global _pending_combine_event
    if torch.compiler.is_compiling():
        return
    if _pending_combine_event is not None:
        _pending_combine_event.current_stream_wait()
        _pending_combine_event = None


def get_hidden_bytes(x: torch.Tensor) -> int:
    """Bytes for one token's hidden vector (>= 2 so fp8 and bf16 share a buffer)."""
    return x.size(1) * max(x.element_size(), 2)


def get_buffer(
    group: ProcessGroup,
    *,
    hidden: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    use_fp8_dispatch: bool = False,
) -> ElasticBuffer:
    """Get or create the process-global DeepEP v2 ``ElasticBuffer``.

    A single buffer serves both training and inference (v2 unified the HT/LL buffers).
    It is recreated only if the group changes or a larger buffer is needed. The size
    is computed analytically by ``get_buffer_size_hint`` from the MoE settings; v2
    needs ``num_max_tokens_per_rank`` (the max tokens any rank may dispatch in one
    forward) up front because the buffer is sized statically.

    Created with ``explicitly_destroy=True`` so the C++ destructor does NOT auto-run
    ``destroy()`` (-> ``cudaDeviceSynchronize`` + host barrier) on GC: that barrier
    inside a CUDA-graph capture aborts the capture. We never call ``destroy()`` (the
    buffer lives for the process; leaking the comm buffer at exit is fine). Matches
    vLLM's DeepEP buffer usage and the validated v1 low-latency cudagraph path.
    """
    global _buffer
    needed_bytes = ElasticBuffer.get_buffer_size_hint(
        group,
        num_max_tokens_per_rank,
        hidden,
        num_topk=num_topk,
        use_fp8_dispatch=use_fp8_dispatch,
    )
    if (
        _buffer is not None
        and _buffer.group == group
        and _buffer.num_bytes >= needed_bytes
    ):
        return _buffer
    _buffer = ElasticBuffer(
        group,
        num_bytes=needed_bytes,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=num_topk,
        use_fp8_dispatch=use_fp8_dispatch,
        explicitly_destroy=True,
    )
    return _buffer


def _permute_tokens(
    hidden_states: torch.Tensor,
    dispatched_indices: torch.Tensor,
    dispatched_scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather the compact (deduplicated) dispatch output into expert-major order.

    v2 non-expand ``recv_x`` has one row per unique received token, but a token routed to
    several local experts must appear once per expert for the grouped GEMM. This expands and
    sorts by expert id (a token's valid count comes from ``dispatched_indices != -1``),
    matching the validated v1 high-throughput path so numerics are identical.

    Args:
        hidden_states: Received tokens ``[num_recv_tokens, hidden]`` (deduplicated).
        dispatched_indices: Local expert ids per received token ``[num_recv_tokens, topk]``
            (-1 means a pick not assigned to a local expert).
        dispatched_scores: Routing scores ``[num_recv_tokens, topk]``.

    Returns:
        permuted_hidden_states: ``[num_all_tokens, hidden]`` sorted by expert.
        permuted_scores: ``[num_all_tokens]`` scores in the same order.
        permuted_indices: ``[num_all_tokens]`` original token index for un-permute.
    """
    mask = dispatched_indices != -1
    valid_expert_ids = dispatched_indices[mask]  # 1d tensor
    valid_scores = dispatched_scores[mask]

    # Repeat each token by its valid count and select tokens in expert order.
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
    """Reverse ``_permute_tokens``: scatter-add expert outputs back to unique tokens."""
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

    handle_id: torch.Tensor  # CPU tensor used to retrieve the cached EPHandle
    num_recv_tokens: int
    cudagraph: bool = False
    # Compact path (cudagraph=False): gather/scatter mapping for the grouped GEMM.
    permuted_indices: torch.Tensor | None = None
    permuted_scores: torch.Tensor | None = None
    # Expand path (cudagraph=True): per-received-row routing scores.
    recv_scores: torch.Tensor | None = None


def dispatch_tokens(
    hidden_states: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    top_scores: torch.Tensor,
    num_local_experts: int,
    num_experts: int,
    group: ProcessGroup,
    *,
    num_max_tokens_per_rank: int,
    cudagraph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, DispatchState]:
    """Dispatch tokens to experts via DeepEP v2 ``ElasticBuffer``.

    Returns tokens in expert-major order for the grouped-GEMM expert path. In compact mode
    (``cudagraph=False``) the deduplicated dispatch output is gathered by ``_permute_tokens``
    (matching the v1 path for identical numerics); in expand mode (``cudagraph=True``) the
    static layout is already expert-grouped. Routing scores are applied to the expert outputs
    in ``combine_tokens``.

    Args:
        hidden_states: Input tokens [num_tokens, hidden_dim]
        selected_experts_indices: Expert indices per token [num_tokens, top_k]
        top_scores: Routing scores per token [num_tokens, top_k]
        num_local_experts: Number of experts on this rank
        num_experts: Total number of experts across all ranks
        group: EP process group
        num_max_tokens_per_rank: Max tokens any rank may dispatch (buffer sizing)
        cudagraph: If True, use the static, no-host-sync expand layout so the forward is
            cudagraph-capturable (inference/decode only, no backward). If False, use the
            compact layout with a host sync and full autograd (training/prefill).

    Returns:
        (routed_tokens [num_recv, hidden], tokens_per_expert [num_local_experts], state)
    """
    del num_local_experts  # counts come from the handle, not this hint

    # The expand layout is inference-only ("must not be backward"), so gate it on a
    # no-grad context. With a single model_spec shared by trainer and generator, this
    # auto-selects: the trainer (autograd enabled) takes the compact path, while the
    # generator -- which runs the forward under torch.no_grad()/inference_mode -- takes
    # the cudagraph-able expand path. A cudagraph=True spec used in a grad context
    # safely falls back to compact rather than hitting the no-backward kernel error.
    cudagraph = cudagraph and not torch.is_grad_enabled()

    selected_experts_indices = selected_experts_indices.contiguous()
    top_scores = top_scores.contiguous()
    # Mask out zero-score selections (DeepEP uses -1 for "no selection").
    selected_experts_indices = selected_experts_indices.masked_fill(top_scores == 0, -1)
    if top_scores.dtype != torch.float32:
        top_scores = top_scores.float()

    # Hide buffer setup (all_gather_object -> aten._to_copy, a MUST_SAVE op in our SAC
    # policy) from SAC's __torch_dispatch__: it is infrastructure, not model compute.
    with _disable_current_modes():
        get_buffer(
            group,
            hidden=hidden_states.shape[1],
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_topk=selected_experts_indices.shape[1],
        )

    (
        recv_x,
        recv_topk_idx,
        recv_scores,
        num_recv_per_expert,
        handle_id,
    ) = torch.ops.deepep.dispatch(
        hidden_states,
        selected_experts_indices,
        top_scores,
        num_experts,
        num_max_tokens_per_rank,
        cudagraph,
    )

    num_tokens_per_expert = num_recv_per_expert.to(recv_x.device)

    if cudagraph:
        # Expand layout is already expert-grouped; feed it straight to the grouped GEMM.
        state = DispatchState(
            handle_id=handle_id,
            num_recv_tokens=recv_x.shape[0],
            cudagraph=True,
            recv_scores=recv_scores,
        )
        return recv_x, num_tokens_per_expert, state

    # Compact layout is deduplicated; gather into expert-major order (plain autograd, so the
    # gather's gradient flows back through the dispatch custom op's combine-backward).
    num_recv_tokens = recv_x.shape[0]
    routed_input, permuted_scores, permuted_indices = _permute_tokens(
        recv_x, recv_topk_idx, recv_scores
    )
    state = DispatchState(
        handle_id=handle_id,
        num_recv_tokens=num_recv_tokens,
        cudagraph=False,
        permuted_indices=permuted_indices,
        permuted_scores=permuted_scores,
    )
    return routed_input, num_tokens_per_expert, state


def combine_tokens(
    hidden_states: torch.Tensor,
    state: DispatchState,
) -> torch.Tensor:
    """Combine expert outputs back to tokens via DeepEP v2.

    Routing scores are applied here (in plain PyTorch) before the pure-reduction combine
    op, so autograd handles the score gradient. Combine is async; the caller MUST call
    ``sync_combine()`` before using the result.

    Compact (``cudagraph=False``): weight each expert-major row, scatter-add back to the
    deduplicated tokens (``_unpermute_tokens``), then combine -- matching the v1 path.
    Expand (``cudagraph=True``): weight per received row, then combine over the static
    layout (the handle drives the expand reduction).

    Args:
        hidden_states: Raw (unweighted) expert outputs [num_recv, hidden].
        state: Dispatch state from ``dispatch_tokens``.

    Returns:
        Combined tokens [num_tokens, hidden_dim].
    """
    # Outer grad state decides whether the combine op frees the handle itself (no
    # backward) or leaves it for combine-backward. Evaluated here, before the op.
    will_backward = torch.is_grad_enabled()

    if not state.cudagraph:
        if state.permuted_scores is not None:
            hidden_states = hidden_states * state.permuted_scores.to(
                hidden_states.dtype
            ).reshape(-1, 1)
        hidden_states = _unpermute_tokens(
            hidden_states, state.permuted_indices, state.num_recv_tokens
        )
        return torch.ops.deepep.combine(hidden_states, state.handle_id, will_backward)

    if state.recv_scores is not None:
        # One routing score per received row (each row is one token->expert assignment).
        # Collapse the trailing dim with sum so this is correct whether recv_scores is
        # [num_recv], [num_recv, 1], or [num_recv, topk] with a single valid entry/row.
        per_row_score = state.recv_scores.reshape(hidden_states.shape[0], -1).sum(
            dim=-1, keepdim=True
        )
        hidden_states = hidden_states * per_row_score.to(hidden_states.dtype)
    return torch.ops.deepep.combine(hidden_states, state.handle_id, will_backward)
