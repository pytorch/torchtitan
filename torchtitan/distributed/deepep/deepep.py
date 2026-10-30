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
on ``deep_ep.ElasticBuffer``.

Dispatch uses DeepEP's expand layout: each received token is copied once per local expert
it picked, grouped by expert, so ``recv_x`` feeds the grouped GEMM directly. Combine sums a
token's expert rows on the expert rank, then across ranks. ``cuda_graph_compatible`` picks
whether dispatch syncs with the host:

- False (training, the default): ``do_cpu_sync=True``, so ``recv_x`` has exactly one row
  per (token, local expert) pick.
- True (inference, under no_grad): ``do_cpu_sync=False``, so the MoE forward is
  CUDA-graph-capturable. ``recv_x`` reserves about
  ``num_tokens_per_rank * ep_size * min(top_k, num_local_experts)`` rows; the rows past
  ``sum(num_recv_per_expert)`` are unused.

Routing scores are applied outside the custom ops (by the routed experts' SwiGLU, or
else in ``combine_tokens``), so autograd handles the score gradient and the custom ops
stay pure communication.
"""

import weakref
from dataclasses import dataclass

import torch
import torch_remat as remat
from torch.distributed import ProcessGroup

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
# Combine removes the entry it uses. If a dispatch never reaches its combine (a recomputed
# dispatch whose combine region is saved), a finalizer in _dispatch_op_impl removes it.
# TODO: return an opaque handle from the ops (like hybridep.DispatchHandle) and delete this cache.
_handle_cache: dict = {}
_handle_counter: int = 0

# Pending combine event for deferred synchronization. The caller MUST call
# sync_combine() before using the result. Process-local + single-threaded, so a
# module var suffices.
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

# dispatch returns: (recv_x, recv_scores, num_recv_per_expert, handle_id).
_lib.define(
    "dispatch(Tensor x, Tensor topk_idx, Tensor topk_weights, "
    "int num_experts, int num_tokens_per_rank, bool cuda_graph_compatible) "
    "-> (Tensor, Tensor, Tensor, Tensor)"
)
# combine returns: combined_x. ``will_backward`` is the caller's outer grad state
# (torch.is_grad_enabled() evaluated before the op): it is the only reliable signal for
# whether a backward will consume the cached handle, since inside a custom-op forward
# autograd disables grad regardless of the outer context. When False (generator no_grad /
# inference), the op frees the handle itself (setup_context never runs).
_lib.define("combine(Tensor x, Tensor handle_id, bool will_backward) -> Tensor")
# Ordered effects, so graph-level selective checkpointing (the graph trainer's memory
# policy) saves both ops instead of replaying them in backward. DeepEP assigns receive
# slots with atomics, so a replayed dispatch can receive rows in another order, while
# backward routes the gradients with the forward's handle. torch_remat ignores effects and
# follows the ``recompute`` argument of dispatch_tokens/combine_tokens; dispatch_tokens
# only replays with a deterministic buffer.
_lib._register_effectful_op("deepep::dispatch", torch.library.EffectType.ORDERED)
_lib._register_effectful_op("deepep::combine", torch.library.EffectType.ORDERED)


# Fallback dispatch/combine SM count when deep_ep's bandwidth heuristic cannot run
# (see _resolve_dispatch_num_sms). num_sms only affects performance, not correctness;
# 20 matches vLLM's deep_ep integration (all2all.py uses num_sms=20).
_DEEPEP_MULTINODE_NUM_SMS = 20


def _resolve_dispatch_num_sms(buffer, num_experts: int, num_topk: int) -> int:
    """SM count for the dispatch kernel (also reused by combine via the handle).

    deep_ep's get_theoretical_num_sms() derives the count from link bandwidths,
    but its RDMA-bandwidth auto-detect can report 0 GB/s on some multi-node
    topologies, making the heuristic divide by zero. On that failure, fall back
    to a fixed count (_DEEPEP_MULTINODE_NUM_SMS); num_sms only affects
    performance, not correctness. dispatch() stores the value on the returned
    handle and combine() reuses it.
    """
    try:
        return buffer.get_theoretical_num_sms(num_experts, num_topk)
    except ZeroDivisionError:
        num_device_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        return min(_DEEPEP_MULTINODE_NUM_SMS, num_device_sms)


@torch.library.impl(_lib, "dispatch", "CUDA")
def _dispatch_op_impl(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    num_tokens_per_rank: int,
    cuda_graph_compatible: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute DeepEP v2 dispatch in the expand layout (see the module docstring).

    Example, this rank holds experts e0 and e1:

        token A picks e0 and e1, token B picks e1, token C picks e0
        recv_x              = [A, C, A, B]   # e0's rows, then e1's
        num_recv_per_expert = [2, 2]
        recv_scores         = [s(A,e0), s(C,e0), s(A,e1), s(B,e1)]

    Order within an expert varies between calls unless the buffer is deterministic.
    """
    global _buffer
    buffer = _buffer
    assert buffer is not None, "Buffer must be initialized before dispatch"

    # Resolve num_sms ourselves and pass it explicitly: the resolver calls deep_ep's
    # bandwidth heuristic but catches its multi-node RDMA-bandwidth divide-by-zero, so
    # dispatch() never falls into that heuristic internally with the default num_sms=0.
    # See _resolve_dispatch_num_sms.
    num_sms = _resolve_dispatch_num_sms(buffer, num_experts, topk_idx.shape[1])
    recv_x, _recv_topk_idx, recv_scores, handle, _event = buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=num_experts,
        # Denote C = num_tokens_per_rank. DeepEP uses C as the per-rank layout stride
        # (global token ID = rank * C + local token ID) and, without a host sync,
        # to size recv_x. MoE physically pads x so C is identical across ranks.
        num_max_tokens_per_rank=num_tokens_per_rank,
        num_sms=num_sms,
        # Backward-safe because get_buffer sets allow_multiple_reduction=True.
        do_expand=True,
        do_cpu_sync=not cuda_graph_compatible,
    )

    handle_id = _get_next_handle_id()
    handle_key = handle_id.item()
    _handle_cache[handle_key] = handle
    # weakref.finalize(obj, fn) calls fn() once obj is garbage-collected.
    weakref.finalize(handle_id, lambda: _handle_cache.pop(handle_key, None))

    # Per-local-expert received-token counts for the grouped GEMM, from the device-side
    # inclusive prefix sum (expert_alignment defaults to 1, so this is a plain prefix
    # sum). RoutedExperts.forward cumsums these into grouped-mm offs.
    psum = handle.psum_num_recv_tokens_per_expert
    num_recv_per_expert = torch.diff(psum, prepend=psum.new_zeros(1)).to(torch.int32)
    return recv_x, recv_scores, num_recv_per_expert, handle_id


def _dispatch_setup_context(ctx, inputs, output):
    x, *_ = inputs
    *_, handle_id = output
    ctx.input_dtype = x.dtype
    ctx.saved_handle = _handle_cache.get(handle_id.item())


def _dispatch_backward(
    ctx,
    grad_recv_x,
    grad_recv_scores,
    grad_num_recv,
    grad_handle_id,
):
    """Backward for dispatch: a combine of the gradients.

    The combine reduces grad_recv_x back to the original tokens (grad for x); passing
    grad_recv_scores (one per row) as the combine's topk_weights returns them as
    ``[num_tokens, top_k]``, the gradient for the dispatched routing scores.
    """
    global _buffer
    if grad_recv_x is None:
        return None, None, None, None, None, None

    buffer = _buffer
    assert buffer is not None, "Buffer must be initialized before combine"

    handle = ctx.saved_handle
    assert handle is not None

    grad_x, grad_scores, _event = buffer.combine(
        grad_recv_x,
        handle=handle,
        topk_weights=grad_recv_scores.float() if grad_recv_scores is not None else None,
    )
    grad_x = grad_x.to(ctx.input_dtype)
    # Order matches op inputs: x, topk_idx, topk_weights, num_experts,
    # num_tokens_per_rank, cuda_graph_compatible. grad_scores stays fp32, like the
    # topk_weights input (dispatch_tokens casts it).
    return grad_x, None, grad_scores, None, None, None


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
    # Record completion so the dispatcher can synchronize before returning.
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
    buffer = _buffer
    assert buffer is not None, "Buffer must be initialized before dispatch"

    handle = ctx.saved_handle
    assert handle is not None, "Handle not found in combine backward"

    # Reuse the dispatch layout via the cached handle (no CPU sync, topk_idx/weights None).
    # Pass num_sms from the handle: with a cached handle, dispatch's automatic
    # get_theoretical_num_sms(num_experts, ...) runs BEFORE num_experts is inferred from the
    # handle, so it would hit num_experts=None. handle.num_sms reuses the dispatch SM count.
    grad_x, _idx, _scores, _handle, _event = buffer.dispatch(
        grad_combined,
        handle=handle,
        num_sms=handle.num_sms,
        do_cpu_sync=False,
        # A cached dispatch takes the layout from this argument, not the handle.
        do_expand=True,
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
    deterministic: bool = False,
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
    vLLM's DeepEP buffer usage and the validated v1 low-latency CUDA graph path.

    ``deterministic=True`` makes two identical dispatches receive rows in the same
    order, at the cost of a sort per dispatch (see ``dispatch_tokens``). A
    deterministic buffer also serves callers that do not need it, so the shared
    buffer is recreated as deterministic once any caller asks for it, and kept so.
    """
    global _buffer
    needed_bytes = ElasticBuffer.get_buffer_size_hint(
        group,
        num_max_tokens_per_rank,
        hidden,
        num_topk=num_topk,
        use_fp8_dispatch=use_fp8_dispatch,
    )
    if _buffer is not None and _buffer.group == group:
        deterministic = deterministic or _buffer.deterministic
        if _buffer.num_bytes >= needed_bytes and _buffer.deterministic == deterministic:
            return _buffer
    _buffer = ElasticBuffer(
        group,
        num_bytes=needed_bytes,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=num_topk,
        use_fp8_dispatch=use_fp8_dispatch,
        deterministic=deterministic,
        # Dispatch backward passes topk_weights to an expand-layout combine, which DeepEP
        # allows only when combine sums a token's rows on the expert rank first.
        allow_multiple_reduction=True,
        explicitly_destroy=True,
    )
    return _buffer


@dataclass
class DispatchState:
    """State from dispatch needed for combine."""

    handle_id: torch.Tensor  # CPU tensor used to retrieve the cached EPHandle
    # fp32 routing score per received row; None once the experts took it
    recv_scores: torch.Tensor | None


def dispatch_tokens(
    hidden_states: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    top_scores: torch.Tensor,
    num_local_experts: int,
    num_experts: int,
    *,
    num_tokens_per_rank: int,
    remat_region_name: str,
    recompute: bool,
    cuda_graph_compatible: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, DispatchState]:
    """Dispatch tokens to experts via DeepEP v2 ``ElasticBuffer``.

    Returns the received rows grouped by local expert, ready for the grouped GEMM. The
    routing scores come back in ``state``; the experts or ``combine_tokens`` apply them.

    Args:
        hidden_states: Input tokens [num_tokens, hidden_dim]
        selected_experts_indices: Expert indices per token [num_tokens, top_k]
        top_scores: Routing scores per token [num_tokens, top_k]
        num_local_experts: Number of experts on this rank
        num_experts: Total number of experts across all ranks
        num_tokens_per_rank: Current number of local tokens. With uneven
            sharding, this must be the maximum local token count across EP
            ranks. DeepEP uses it as the per-rank layout stride and, without a
            host sync, to size the current ``recv_x``. This is distinct from the
            lifetime maximum used to initialize the communication buffer and
            must not exceed that maximum.
        remat_region_name: Name for the dispatch communication region.
        recompute: Whether activation remat replays the dispatch communication during
            backward. Requires a deterministic buffer (see below), and must match
            the ``recompute`` passed to ``combine_tokens``.
        cuda_graph_compatible: If True, dispatch without a host sync so the forward is
            CUDA-graph-capturable (inference only; forced False whenever grad is enabled).
            If False, a host sync sizes the output exactly (training).

    Returns:
        (routed_tokens [num_recv, hidden], tokens_per_expert [num_local_experts], state)
    """
    del num_local_experts  # counts come from the handle, not this hint

    # Without a host sync, recv_x reserves about num_tokens_per_rank * ep_size *
    # min(top_k, num_local_experts) rows, up to ep_size times the routed rows when routing
    # is balanced: too many to keep for backward. So a grad-enabled forward always syncs, even
    # if the config asks for True.
    cuda_graph_compatible = cuda_graph_compatible and not torch.is_grad_enabled()

    buffer = _buffer
    assert buffer is not None, "Buffer must be initialized before dispatch"
    # The dispatch autograd nodes keep the forward's handle, which records the receive
    # order. A replay must receive rows in that same order, or each token's gradient
    # meets another token's activations. DeepEP assigns receive slots with atomics, so
    # only a deterministic buffer guarantees this.
    if recompute and not buffer.deterministic:
        raise RuntimeError(
            "DeepEP dispatch is configured to be replayed by activation checkpointing "
            "(ep_communication not saved by the remat policy), but the DeepEP buffer "
            "is not deterministic, which would silently corrupt the MoE gradients. "
            "DeepEPTokenDispatcher.init_buffer() creates a deterministic buffer when "
            "it runs after activation checkpointing has been applied."
        )
    assert num_tokens_per_rank <= buffer.num_max_tokens_per_rank, (
        "DeepEP current token count "
        f"{num_tokens_per_rank} exceeds the "
        f"preallocated capacity of {buffer.num_max_tokens_per_rank}."
    )

    selected_experts_indices = selected_experts_indices.contiguous()
    top_scores = top_scores.contiguous()
    # Mask out zero-score selections (DeepEP uses -1 for "no selection").
    selected_experts_indices = selected_experts_indices.masked_fill(top_scores == 0, -1)
    if top_scores.dtype != torch.float32:
        top_scores = top_scores.float()

    dispatch_region = remat.region(
        torch.ops.deepep.dispatch,
        remat_region_name,
        recompute=recompute,
    )
    recv_x, recv_scores, num_recv_per_expert, handle_id = dispatch_region(
        hidden_states,
        selected_experts_indices,
        top_scores,
        num_experts=num_experts,
        num_tokens_per_rank=num_tokens_per_rank,
        cuda_graph_compatible=cuda_graph_compatible,
    )
    # The saved region is skipped during replay, while the caller (grouped GEMM, score
    # multiply) still runs and needs its original outputs.
    remat.recompute_needs_tensor(recv_x, recv_scores, num_recv_per_expert, handle_id)

    state = DispatchState(handle_id=handle_id, recv_scores=recv_scores)
    return recv_x, num_recv_per_expert, state


def combine_tokens(
    hidden_states: torch.Tensor,
    state: DispatchState,
    *,
    remat_region_name: str,
    recompute: bool,
) -> torch.Tensor:
    """Combine expert outputs back to tokens via DeepEP v2.

    Unless the experts took them, routing scores are applied here (in plain PyTorch)
    before the pure-reduction combine op, so autograd handles the score gradient. Combine
    is async; the caller MUST call ``sync_combine()`` before using the result.

    Args:
        hidden_states: Expert outputs [num_recv, hidden]; already weighted if the
            experts took the routing scores.
        state: Dispatch state from ``dispatch_tokens``.
        remat_region_name: Name for the combine communication region.
        recompute: Whether activation remat replays the combine communication during
            backward. Must match the ``recompute`` passed to ``dispatch_tokens``.

    Returns:
        Combined tokens [num_tokens, hidden_dim].
    """
    # Outer grad state decides whether the combine op frees the handle itself (no
    # backward) or leaves it for combine-backward. Evaluated here, before the op.
    will_backward = torch.is_grad_enabled()

    if state.recv_scores is not None:
        # The multiply reads the expert outputs with bare ops.
        remat.recompute_needs_tensor(hidden_states)
        recv_scores = state.recv_scores.to(hidden_states.dtype)
        hidden_states = hidden_states * recv_scores.unsqueeze(1)

    combined = remat.region(
        torch.ops.deepep.combine,
        remat_region_name,
        recompute=recompute,
    )(hidden_states, state.handle_id, will_backward)
    # The caller consumes this output outside another remat region.
    remat.recompute_needs_tensor(combined)
    return combined
