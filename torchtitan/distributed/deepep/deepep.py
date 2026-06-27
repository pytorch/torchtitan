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
on ``deep_ep.ElasticBuffer``. The mode is chosen by kwargs, not by a separate buffer:

- training / prefill (default): ``do_expand=False`` -- a dynamic, compact layout that
  is ALREADY grouped by local expert (``handle.num_recv_tokens_per_expert_list`` gives
  the per-expert token counts for grouped GEMM), so no manual permute is needed. Full
  autograd via the custom ops below (dispatch backward is a combine, combine backward
  is a dispatch). One CPU sync for exact counts, so this path is NOT cudagraph-able.
- decode / cudagraph: a cached ``EPHandle`` + ``do_cpu_sync=False`` makes ``dispatch``
  reuse the prior layout without a host sync, so the MoE forward is cudagraph-capturable.
  ``do_expand=True`` (static one-slot-per-expert layout) is inference-only -- the
  expanding layout "must not be backward" per the DeepEP kernels.

Routing scores are applied to expert outputs in plain PyTorch (in ``combine_tokens``,
before the pure-reduction combine op), so autograd handles the score gradient and the
custom ops stay pure communication -- matching the v1 structure.
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
    )
    return _buffer


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

# dispatch returns: (recv_x, recv_scores, num_recv_per_expert, handle_id)
_lib.define(
    "dispatch(Tensor x, Tensor topk_idx, Tensor topk_weights, "
    "int num_experts, int num_max_tokens_per_rank) "
    "-> (Tensor, Tensor, Tensor, Tensor)"
)
# combine returns: combined_x
_lib.define("combine(Tensor x, Tensor handle_id) -> Tensor")


@torch.library.impl(_lib, "dispatch", "CUDA")
def _dispatch_op_impl(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    num_max_tokens_per_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute DeepEP v2 dispatch (compact, expert-grouped layout)."""
    global _buffer
    buffer = _buffer
    assert buffer is not None, "Buffer must be initialized before dispatch"

    recv_x, _recv_topk_idx, recv_scores, handle, _event = buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        do_expand=False,
    )

    handle_id = _get_next_handle_id()
    _handle_cache[handle_id.item()] = handle

    # Per-local-expert received-token counts for the grouped GEMM (CPU list -> tensor).
    num_recv_per_expert = torch.tensor(
        handle.num_recv_tokens_per_expert_list, dtype=torch.int32, device="cpu"
    )
    return recv_x, recv_scores, num_recv_per_expert, handle_id


def _dispatch_setup_context(ctx, inputs, output):
    x, *_ = inputs
    *_, handle_id = output
    ctx.input_dtype = x.dtype
    ctx.saved_handle = _handle_cache.get(handle_id.item())


def _dispatch_backward(
    ctx, grad_recv_x, grad_recv_scores, grad_num_recv, grad_handle_id
):
    """Backward for dispatch: a combine of the gradients.

    The combine reduces grad_recv_x back to the original tokens (grad for x); passing
    grad_recv_scores as the combine's topk_weights yields the gradient for the
    dispatched routing scores (DeepEP combine returns the reduced weights too).
    """
    global _buffer
    if grad_recv_x is None:
        return None, None, None, None, None

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
    # Order matches op inputs: x, topk_idx, topk_weights, num_experts, num_max_tokens_per_rank
    return grad_x, None, grad_topk_weights, None, None


@torch.library.impl(_lib, "combine", "CUDA")
def _combine_op_impl(x: torch.Tensor, handle_id: torch.Tensor) -> torch.Tensor:
    """Execute DeepEP v2 combine (pure reduction; scores already applied upstream)."""
    global _buffer, _pending_combine_event
    buffer = _buffer
    assert buffer is not None, "Buffer must be initialized before combine"

    # In inference, setup_context does not run, so clean up here.
    # NOTE: for inference, use torch.inference_mode() (not torch.no_grad()).
    if torch.is_inference_mode_enabled():
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
    _, handle_id = inputs
    ctx.saved_handle = _handle_cache.pop(handle_id.item(), None)


def _combine_backward(ctx, grad_combined):
    """Backward for combine: a dispatch of the gradient (reuses the cached handle)."""
    global _buffer
    handle = ctx.saved_handle
    assert handle is not None, "Handle not found in combine backward"

    # Reuse the dispatch layout via the cached handle (no CPU sync, topk_idx/weights None).
    grad_x, _idx, _scores, _handle, _event = _buffer.dispatch(
        grad_combined,
        handle=handle,
        do_cpu_sync=False,
    )
    return grad_x, None


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


@dataclass
class DispatchState:
    """State from dispatch needed for combine."""

    handle_id: torch.Tensor  # CPU tensor used to retrieve the cached EPHandle
    num_recv_tokens: int
    recv_scores: torch.Tensor | None = None  # per-received-token routing scores


def dispatch_tokens(
    hidden_states: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    top_scores: torch.Tensor,
    num_local_experts: int,
    num_experts: int,
    group: ProcessGroup,
    *,
    num_max_tokens_per_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, DispatchState]:
    """Dispatch tokens to experts via DeepEP v2 ``ElasticBuffer``.

    The compact layout is already grouped by local expert, so the returned tokens feed
    the grouped-GEMM expert path directly (no permute). Routing scores are applied to
    the expert outputs in ``combine_tokens``.

    Args:
        hidden_states: Input tokens [num_tokens, hidden_dim]
        selected_experts_indices: Expert indices per token [num_tokens, top_k]
        top_scores: Routing scores per token [num_tokens, top_k]
        num_local_experts: Number of experts on this rank
        num_experts: Total number of experts across all ranks
        group: EP process group
        num_max_tokens_per_rank: Max tokens any rank may dispatch (buffer sizing)

    Returns:
        (routed_tokens [num_recv, hidden], tokens_per_expert [num_local_experts], state)
    """
    del num_local_experts  # counts come from the handle, not this hint

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

    recv_x, recv_scores, num_recv_per_expert, handle_id = torch.ops.deepep.dispatch(
        hidden_states,
        selected_experts_indices,
        top_scores,
        num_experts,
        num_max_tokens_per_rank,
    )

    num_tokens_per_expert = num_recv_per_expert.to(recv_x.device)
    state = DispatchState(
        handle_id=handle_id,
        num_recv_tokens=recv_x.shape[0],
        recv_scores=recv_scores,
    )
    return recv_x, num_tokens_per_expert, state


def combine_tokens(
    hidden_states: torch.Tensor,
    state: DispatchState,
) -> torch.Tensor:
    """Combine expert outputs back to tokens via DeepEP v2.

    Routing scores are applied here (in plain PyTorch) before the pure-reduction combine
    op, so autograd handles the score gradient. Combine is async; the caller MUST call
    ``sync_combine()`` before using the result.

    Args:
        hidden_states: Raw (unweighted) expert outputs [num_recv, hidden].
        state: Dispatch state from ``dispatch_tokens``.

    Returns:
        Combined tokens [num_tokens, hidden_dim].
    """
    if state.recv_scores is not None:
        # One routing score per received row (each row is one token->expert assignment).
        # Collapse the trailing dim with sum so this is correct whether recv_scores is
        # [num_recv], [num_recv, 1], or [num_recv, topk] with a single valid entry/row.
        per_row_score = state.recv_scores.reshape(hidden_states.shape[0], -1).sum(
            dim=-1, keepdim=True
        )
        hidden_states = hidden_states * per_row_score.to(hidden_states.dtype)
    return torch.ops.deepep.combine(hidden_states, state.handle_id)
