# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HybridEP dispatch/combine with explicit asynchronous completion.

Native handles describe routing and survive through backward. Exchange requests
own one launch's completion event. Waits take explicit input keepalives so
compiler buffer reuse cannot overwrite a launch's inputs before completion.
Launches and waits accept arbitrary compiler layouts: native contiguous copies
are made inside the launch, never by independent input-layout constraints that
could give the launch and its keepalive different storage.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.distributed as dist
from torch._library.opaque_object import CustomClassBase, register_custom_class
from torch.fx import has_side_effect
from torch.fx.traceback import annotate, annotate_fn

logger = logging.getLogger(__name__)

# Tensor axes: T = input tokens, R = routed rows, D = hidden width, K = top-k.
_EP_TOKEN_EXCHANGE = "EP_token_exchange"
_EP_TOKEN_EXCHANGE_WAIT = "EP_token_exchange_wait"
_buffer: Any = None
_communication_stream: torch.cuda.Stream | None = None
_over_budget: torch.Tensor | None = None


class DispatchHandle(CustomClassBase):
    """Opaque native routing state shared by dispatch, combine, and backward."""

    def __init__(self, value=None, *, non_blocking: bool = False):
        self.value = value
        self.non_blocking = non_blocking

    def __eq__(self, other):
        return isinstance(other, DispatchHandle) and self.value is other.value

    def __hash__(self):
        return id(self.value)

    def __fx_repr__(self):
        return "DispatchHandle()", {"DispatchHandle": DispatchHandle}


class _Exchange(CustomClassBase):
    """One launch's stream completion, distinct from its routing handle."""

    def __init__(self):
        self.event: torch.cuda.Event | None = None

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

    def __fx_repr__(self):
        return "_Exchange()", {"_Exchange": _Exchange}


register_custom_class(DispatchHandle, typ="symbolic")
register_custom_class(_Exchange, typ="symbolic")


@contextmanager
def _launch(*, async_op: bool, inputs: tuple[torch.Tensor, ...]) -> Iterator[_Exchange]:
    """Launch on the communication stream after the input-producing stream."""
    current = torch.cuda.current_stream()
    stream = current
    request = _Exchange()
    if async_op:
        assert _communication_stream is not None, "Initialize HybridEP before launch"
        stream = _communication_stream
        stream.wait_stream(current)
        for tensor in inputs:
            tensor.record_stream(stream)
    with torch.cuda.stream(stream):
        yield request
        if async_op:
            request.event = torch.cuda.Event()
            request.event.record(stream)


@dataclass
class DispatchState:
    """Native routing state and scores needed to combine expert outputs."""

    handle: DispatchHandle
    permuted_scores: torch.Tensor | None = None
    num_tokens: int = 0
    async_op: bool = False


def _num_permuted_tokens_for_non_blocking(
    num_tokens: int,
    ep_size: int,
    num_local_experts: int,
    top_k: int,
    moe_expert_capacity_factor: float,
    pad_multiple: int | None = None,
) -> int:
    """Derive a static expert-output capacity from the input-token bound."""
    if not math.isfinite(moe_expert_capacity_factor) or moe_expert_capacity_factor <= 0:
        raise ValueError("HybridEP capacity factor must be finite and positive")
    if pad_multiple is not None and pad_multiple <= 0:
        raise ValueError("HybridEP pad_multiple must be positive")
    num_rows = int(
        num_tokens
        * ep_size
        * min(num_local_experts, top_k)
        * moe_expert_capacity_factor
    )
    if pad_multiple is not None:
        num_rows = ((num_rows + pad_multiple - 1) // pad_multiple) * pad_multiple
    return num_rows


@torch.library.custom_op(
    "hybridep::dispatch",
    mutates_args=(),
    device_types="cuda",
    tags=(torch.Tag.flexible_layout,),
)
def _dispatch_impl(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    ep_size: int,
    group_name: str,
    num_output_tokens: int | None = None,
    pad_multiple: int | None = None,
    async_op: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, DispatchHandle, _Exchange]:
    """Dispatch through native HybridEP and return a separate completion request."""
    assert _buffer is not None, "Initialize HybridEP before dispatch"
    group = dist.distributed_c10d._resolve_process_group(
        cast(dist.GroupName, group_name)
    )
    assert _buffer.group == group
    assert x.shape[0] <= _buffer.configurer.buffer_config.max_num_of_tokens_per_rank
    non_blocking = num_output_tokens is not None
    if async_op and not non_blocking:
        raise ValueError("Asynchronous HybridEP requires a non-blocking capacity")
    num_local_experts = num_experts // ep_size
    with _launch(async_op=async_op, inputs=(x, topk_idx, topk_weights)) as request:
        hidden, scores, _, counts, native_handle = _buffer.dispatch_with_permute(
            hidden=x.contiguous(),
            topk_idx=topk_idx.contiguous(),
            topk_weights=topk_weights.float().contiguous(),
            num_of_experts_per_rank=num_local_experts,
            num_of_experts=num_experts,
            pad_multiple=pad_multiple,
            num_permuted_tokens=num_output_tokens,
            non_blocking=non_blocking,
        )
        if non_blocking:
            _record_over_budget(native_handle[-1])
        if scores is None:
            scores = x.new_empty(0, dtype=torch.float32)
        counts = counts.to(x.device)
    return (
        hidden,
        scores,
        counts,
        DispatchHandle(
            native_handle,
            non_blocking=non_blocking,
        ),
        request,
    )


@_dispatch_impl.register_fake
def _dispatch_fake(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    ep_size: int,
    group_name: str,
    num_output_tokens: int | None = None,
    pad_multiple: int | None = None,
    async_op: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, DispatchHandle, _Exchange]:
    """Describe the native dispatch outputs without reading routing data."""
    non_blocking = num_output_tokens is not None
    num_rows = num_output_tokens if non_blocking else x.shape[0]
    return (
        x.new_empty(num_rows, x.shape[1]),
        x.new_empty(num_rows, dtype=torch.float32),
        x.new_empty(num_experts // ep_size, dtype=torch.int64),
        DispatchHandle(non_blocking=non_blocking),
        _Exchange(),
    )


@torch.library.custom_op(
    "hybridep::combine",
    mutates_args=(),
    device_types="cuda",
    tags=(torch.Tag.flexible_layout,),
)
def _combine_impl(
    x: torch.Tensor,
    handle: DispatchHandle,
    num_tokens: int,
    pad_multiple: int | None = None,
    async_op: bool = False,
) -> tuple[torch.Tensor, _Exchange]:
    """Combine through the saved native routing handle."""
    assert _buffer is not None, "Initialize HybridEP before combine"
    with _launch(async_op=async_op, inputs=(x,)) as request:
        combined, _ = _buffer.combine_with_unpermute(
            hidden=x.contiguous(),
            handle=handle.value,
            pad_multiple=pad_multiple,
            non_blocking=handle.non_blocking,
        )
    return combined, request


@_combine_impl.register_fake
def _combine_fake(
    x: torch.Tensor,
    handle: DispatchHandle,
    num_tokens: int,
    pad_multiple: int | None = None,
    async_op: bool = False,
) -> tuple[torch.Tensor, _Exchange]:
    """Describe the combined token tensor and its completion request."""
    return x.new_empty(num_tokens, x.shape[1]), _Exchange()


@torch.library.custom_op(
    "hybridep::dispatch_bwd",
    mutates_args=(),
    device_types="cuda",
    tags=(torch.Tag.flexible_layout,),
)
def _dispatch_bwd_impl(
    grad_combined: torch.Tensor,
    handle: DispatchHandle,
    num_permuted_tokens: int,
    pad_multiple: int | None = None,
    async_op: bool = False,
) -> tuple[torch.Tensor, _Exchange]:
    """Scatter combine's gradient using its cached routing layout."""
    assert _buffer is not None, "Initialize HybridEP before backward dispatch"
    with _launch(async_op=async_op, inputs=(grad_combined,)) as request:
        grad_x, _, _, _, _ = _buffer.dispatch_with_permute(
            hidden=grad_combined.contiguous(),
            handle=handle.value,
            num_permuted_tokens=num_permuted_tokens,
            pad_multiple=pad_multiple,
            non_blocking=handle.non_blocking,
        )
    return grad_x, request


@_dispatch_bwd_impl.register_fake
def _dispatch_bwd_fake(
    grad_combined: torch.Tensor,
    handle: DispatchHandle,
    num_permuted_tokens: int,
    pad_multiple: int | None = None,
    async_op: bool = False,
) -> tuple[torch.Tensor, _Exchange]:
    """Describe the cached dispatch gradient without reading its handle."""
    return (
        grad_combined.new_empty(num_permuted_tokens, grad_combined.shape[1]),
        _Exchange(),
    )


@torch.library.custom_op(
    "hybridep::combine_bwd",
    mutates_args=(),
    device_types="cuda",
    tags=(torch.Tag.flexible_layout,),
)
def _combine_bwd_impl(
    grad_hidden: torch.Tensor,
    grad_scores: torch.Tensor,
    handle: DispatchHandle,
    num_tokens: int,
    num_experts: int,
    async_op: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, _Exchange]:
    """Gather dispatch's input and routing-score gradients."""
    assert _buffer is not None, "Initialize HybridEP before backward combine"
    with _launch(async_op=async_op, inputs=(grad_hidden, grad_scores)) as request:
        grad_x, grad_probs = _buffer.combine_with_unpermute(
            hidden=grad_hidden.contiguous(),
            probs=grad_scores.float().contiguous() if grad_scores.numel() else None,
            handle=handle.value,
            non_blocking=handle.non_blocking,
        )
        if grad_probs is None:
            grad_probs = grad_hidden.new_empty(0, dtype=torch.float32)
    return grad_x, grad_probs, request


@_combine_bwd_impl.register_fake
def _combine_bwd_fake(
    grad_hidden: torch.Tensor,
    grad_scores: torch.Tensor,
    handle: DispatchHandle,
    num_tokens: int,
    num_experts: int,
    async_op: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, _Exchange]:
    """Describe input and dense routing-score gradients."""
    scores = (
        grad_hidden.new_empty(num_tokens, num_experts, dtype=torch.float32)
        if grad_scores.numel()
        else grad_hidden.new_empty(0, dtype=torch.float32)
    )
    return grad_hidden.new_empty(num_tokens, grad_hidden.shape[1]), scores, _Exchange()


_lib = torch.library.Library("hybridep", "FRAGMENT")
_lib.define(
    "wait(Tensor(a)[] tensors, "
    "torchtitan.distributed.deepep.hybridep._Exchange request, "
    "Tensor[] keepalives) -> Tensor(a)[]",
    tags=(torch.Tag.flexible_layout,),
)


@torch.library.impl("hybridep::wait", "CompositeExplicitAutograd")
def _wait_impl(
    tensors: list[torch.Tensor], request: _Exchange, keepalives: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Join one exchange and transfer its output lifetime to compute."""
    stream = torch.cuda.current_stream()
    assert request.event is not None, "HybridEP wait requires an asynchronous launch"
    stream.wait_event(request.event)
    for tensor in tensors:
        tensor.record_stream(stream)
    del keepalives
    return tensors


@torch.library.register_fake("hybridep::wait")
def _wait_fake(
    tensors: list[torch.Tensor], request: _Exchange, keepalives: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Preserve aliases while describing the completion boundary."""
    return tensors


class _WaitTensors(torch.autograd.Function):
    """Preserve each pending tensor's gradient through a stream join."""

    @staticmethod
    # Autograd calls forward with the FunctionCtx, unlike its variadic base stub.
    # pyrefly: ignore [bad-override]
    def forward(ctx, request, keepalives, *tensors):
        """Join completion below autograd without copying any output."""
        del ctx
        with torch._C._AutoDispatchBelowAutograd():
            return tuple(
                torch.ops.hybridep.wait.default(list(tensors), request, keepalives)
            )

    @staticmethod
    def backward(ctx, *gradients):
        """Return identity gradients for tensor inputs, none for the request."""
        del ctx
        return None, None, *gradients


@torch.library.impl("hybridep::wait", "Autograd")
def _wait_autograd(
    tensors: list[torch.Tensor], request: _Exchange, keepalives: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Expose the wait's identity gradient contract for every tensor input."""
    return list(_WaitTensors.apply(request, keepalives, *tensors))


@torch.library.impl("hybridep::wait", "Functionalize")
def _wait_functionalize(
    tensors: list[torch.Tensor], request: _Exchange, keepalives: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Join the underlying tensors while preserving their functional wrappers."""
    # _sync commits functional mutations; it is not a CPU/GPU stream sync.
    for tensor in (*tensors, *keepalives):
        torch._sync(tensor)
    underlying = [torch._from_functional_tensor(tensor) for tensor in tensors]
    keepalives_inner = [torch._from_functional_tensor(tensor) for tensor in keepalives]
    with torch._C._ExcludeDispatchKeyGuard(
        torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
    ):
        torch.ops.hybridep.wait.default(underlying, request, keepalives_inner)
    return tensors


has_side_effect(torch.ops.hybridep.wait.default)


def _wait(
    tensors: list[torch.Tensor],
    request: _Exchange,
    *,
    phase: str,
    keepalives: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Annotate one completion wait for all outputs of an exchange."""
    with annotate({_EP_TOKEN_EXCHANGE: None, _EP_TOKEN_EXCHANGE_WAIT: phase}):
        return torch.ops.hybridep.wait.default(tensors, request, keepalives)


@annotate_fn({_EP_TOKEN_EXCHANGE: None, _EP_TOKEN_EXCHANGE_WAIT: None})
def _dispatch_backward(
    ctx, grad_hidden, grad_scores, grad_tpe, grad_handle, grad_request
):
    """Combine the input and score gradients with the forward routing layout."""
    if grad_hidden is None:
        return (None,) * 9
    (topk_idx,) = ctx.saved_tensors
    with annotate({_EP_TOKEN_EXCHANGE: "combine"}):
        grad_x, grad_probs, request = torch.ops.hybridep.combine_bwd(
            grad_hidden,
            grad_scores,
            ctx.dispatch_handle,
            topk_idx.shape[0],
            ctx.num_experts,
            ctx.async_op,
        )
    if ctx.async_op:
        grad_x, grad_probs = _wait(
            [grad_x, grad_probs],
            request,
            phase="combine",
            keepalives=[grad_hidden, grad_scores],
        )
    grad_weights = grad_probs.gather(1, topk_idx) if grad_probs.numel() else None
    return grad_x.to(ctx.input_dtype), None, grad_weights, *([None] * 6)


def _dispatch_setup_context(ctx, inputs, output):
    """Save routing state and the execution contract for backward."""
    x, topk_idx, _, num_experts, _, _, _, _, async_op = inputs
    _, _, _, handle, _ = output
    ctx.dispatch_handle = handle
    ctx.input_dtype = x.dtype
    ctx.num_experts = num_experts
    ctx.async_op = async_op
    ctx.save_for_backward(topk_idx)


@annotate_fn({_EP_TOKEN_EXCHANGE: None, _EP_TOKEN_EXCHANGE_WAIT: None})
def _combine_backward(ctx, grad_combined, grad_request):
    """Dispatch the combined gradient with the forward's cached layout."""
    with annotate({_EP_TOKEN_EXCHANGE: "dispatch"}):
        grad_x, request = torch.ops.hybridep.dispatch_bwd(
            grad_combined,
            ctx.dispatch_handle,
            ctx.num_permuted_tokens,
            ctx.pad_multiple,
            ctx.async_op,
        )
    if ctx.async_op:
        (grad_x,) = _wait(
            [grad_x], request, phase="dispatch", keepalives=[grad_combined]
        )
    return grad_x, None, None, None, None


def _combine_setup_context(ctx, inputs, output):
    """Keep the input capacity and routing layout for combine backward."""
    x, handle, _, pad_multiple, async_op = inputs
    ctx.dispatch_handle = handle
    ctx.num_permuted_tokens = x.shape[0]
    ctx.pad_multiple = pad_multiple
    ctx.async_op = async_op


_dispatch_impl.register_autograd(
    _dispatch_backward, setup_context=_dispatch_setup_context
)
_combine_impl.register_autograd(_combine_backward, setup_context=_combine_setup_context)


def get_buffer(
    group: dist.ProcessGroup,
    hidden_dim: int,
    num_max_tokens_per_rank: int,
    num_local_experts: int,
    fp8_dispatch: bool = False,
) -> None:
    """Initialize shared native transport storage and its launch stream.

    Args:
        group: Expert-parallel process group owning the transport.
        hidden_dim: Maximum input hidden width for this buffer.
        num_max_tokens_per_rank: Input-token bound used for transport storage,
            independent of each dispatch's expert-output capacity.
        num_local_experts: Number of experts resident on each rank.
        fp8_dispatch: Unsupported; only unquantized transport is available.

    The buffer grows to cover later requests; routing handles and completion
    requests are separate per-call objects, not shared transport state.
    Set TORCHTITAN_HYBRIDEP_DISPATCH_SMS and TORCHTITAN_HYBRIDEP_COMBINE_SMS
    before initialization to override the native communication SM targets
    (16 each). Compute kernels retain their normal resource policy.
    """
    global _buffer, _communication_stream
    if fp8_dispatch:
        raise ValueError("HybridEP FP8 dispatch is not supported")
    from deep_ep import HybridEPBuffer

    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.configurer.buffer_config.hidden_dim < hidden_dim
        or _buffer.configurer.buffer_config.max_num_of_tokens_per_rank
        < num_max_tokens_per_rank
        or _buffer.configurer.buffer_config.num_of_experts_per_rank < num_local_experts
    ):
        num_sms_dispatch = int(os.getenv("TORCHTITAN_HYBRIDEP_DISPATCH_SMS", "16"))
        num_sms_combine = int(os.getenv("TORCHTITAN_HYBRIDEP_COMBINE_SMS", "16"))
        num_device_sms = torch.cuda.get_device_properties().multi_processor_count
        if not (
            0 < num_sms_dispatch <= num_device_sms
            and 0 < num_sms_combine <= num_device_sms
        ):
            raise ValueError(
                "TORCHTITAN_HYBRIDEP_DISPATCH_SMS and TORCHTITAN_HYBRIDEP_COMBINE_SMS "
                f"must be in [1, {num_device_sms}], "
                f"got {num_sms_dispatch} and {num_sms_combine}"
            )
        _buffer = HybridEPBuffer(
            group=group,
            hidden_dim=hidden_dim,
            max_num_of_tokens_per_rank=num_max_tokens_per_rank,
            num_local_experts=num_local_experts,
            use_fp8=False,
            num_sms_dispatch_api=num_sms_dispatch,
            num_sms_combine_api=num_sms_combine,
            load_cached_kernels=True,
            use_shared_buffer=True,
            enable_custom_allgather=True,
        )
        _communication_stream = torch.cuda.Stream(priority=-1)


def dispatch_tokens(
    hidden_states: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    top_scores: torch.Tensor,
    num_local_experts: int,
    num_experts: int,
    group: dist.ProcessGroup,
    *,
    non_blocking_expert_capacity_factor: float | None = None,
    pad_multiple: int | None = None,
    async_op: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, DispatchState]:
    """Dispatch tokens and join completion before exposing expert inputs.

    Args:
        hidden_states: Input tokens with shape (T, D).
        selected_experts_indices: Expert indices with shape (T, K).
        top_scores: Routing scores with shape (T, K).
        num_local_experts: Number of experts on this rank.
        num_experts: Number of experts across the group.
        group: Expert-parallel process group.
        non_blocking_expert_capacity_factor: Static output-capacity multiplier;
            None selects native blocking dispatch.
        pad_multiple: Native per-expert alignment.
        async_op: Launch on the communication stream with a graph-visible wait.

    Returns:
        Expert-major inputs, device counts, and state for combine.
    """
    capacity = None
    if non_blocking_expert_capacity_factor is not None:
        capacity = _num_permuted_tokens_for_non_blocking(
            hidden_states.shape[0],
            group.size(),
            num_local_experts,
            selected_experts_indices.shape[1],
            non_blocking_expert_capacity_factor,
            pad_multiple,
        )
    with annotate({_EP_TOKEN_EXCHANGE: "dispatch"}):
        hidden, scores, counts, handle, request = torch.ops.hybridep.dispatch(
            hidden_states,
            selected_experts_indices,
            top_scores,
            num_experts,
            group.size(),
            group.group_name,
            capacity,
            pad_multiple,
            async_op,
        )
    if async_op:
        hidden, scores, counts = _wait(
            [hidden, scores, counts],
            request,
            phase="dispatch",
            keepalives=[hidden_states, selected_experts_indices, top_scores],
        )
    scores = scores.to(hidden.dtype)
    return (
        hidden,
        counts,
        DispatchState(
            handle,
            scores,
            hidden_states.shape[0],
            async_op,
        ),
    )


def combine_tokens(
    hidden_states: torch.Tensor,
    state: DispatchState,
    pad_multiple: int | None = None,
) -> torch.Tensor:
    """Apply routing scores and combine through the saved native handle."""
    if state.permuted_scores is not None:
        hidden_states = hidden_states * state.permuted_scores.reshape(-1, 1)
    with annotate({_EP_TOKEN_EXCHANGE: "combine"}):
        combined, request = torch.ops.hybridep.combine(
            hidden_states,
            state.handle,
            state.num_tokens,
            pad_multiple,
            state.async_op,
        )
    if state.async_op:
        (combined,) = _wait(
            [combined], request, phase="combine", keepalives=[hidden_states]
        )
    return combined


def _record_over_budget(overflow_flag: torch.Tensor) -> None:
    """Accumulate native token-dropping flags on device without a host read."""
    global _over_budget
    if _over_budget is None:
        _over_budget = torch.zeros(1, dtype=torch.bool, device=overflow_flag.device)
    _over_budget.logical_or_(overflow_flag)


def check_hybridep_over_budget() -> torch.Tensor | None:
    """Return the sticky device-side token-dropping flag."""
    return _over_budget


def reset_hybridep_over_budget() -> None:
    """Clear the device-side flag at a training-step boundary."""
    if _over_budget is not None:
        _over_budget.zero_()


__all__ = [
    "DispatchHandle",
    "DispatchState",
    "dispatch_tokens",
    "combine_tokens",
    "check_hybridep_over_budget",
    "reset_hybridep_over_budget",
]
