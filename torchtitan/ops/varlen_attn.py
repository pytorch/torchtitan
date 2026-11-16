# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Varlen FlashAttention whose backward is deterministic by itself.

FlashAttention (FA3/FA4) runs a deterministic backward only when
``torch.are_deterministic_algorithms_enabled()`` is true at backward time. The
backward op here turns the flag on locally, so the result does not depend on
the global deterministic mode. The forward and backward call torch's varlen ops
directly, so the backward does not recompute the forward.

Hooks on the ``grad_fn`` of the public ``varlen_attn`` do not work here: Dynamo
cannot trace ``grad_fn``, and a compiled backward raises when the flag differs
from the forward.

TODO: The ``torch_attn::_varlen_attn`` ops are private; switch to a public
deterministic-backward option once torch varlen_attn has one.
"""

import torch
from torch.nn.attention import SDPBackend

_FLASH = SDPBackend.FLASH_ATTENTION.value


def _varlen_attn_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    scale: float | None,
    window_size: list[int],
    num_splits: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(out, lse, rng_state)`` of Flash varlen attention."""
    return torch.ops.torch_attn._varlen_attn(
        query,
        key,
        value,
        cu_seq_q,
        cu_seq_k,
        max_q,
        max_k,
        is_causal=window_size == [-1, 0],
        scale=scale,
        window_size=window_size,
        num_splits=num_splits,
        backend=_FLASH,
    )


deterministic_varlen_attn = torch.library.custom_op(
    "torchtitan::deterministic_varlen_attn", mutates_args=()
)(_varlen_attn_forward)
# The torch op has a fake kernel, so the same function computes the fake outputs.
deterministic_varlen_attn.register_fake(_varlen_attn_forward)


@torch.library.custom_op(
    "torchtitan::deterministic_varlen_attn_backward", mutates_args=()
)
def _deterministic_varlen_attn_backward(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    rng_state: torch.Tensor,
    scale: float | None,
    window_size: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prev = torch.are_deterministic_algorithms_enabled()
    prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    # pyrefly: ignore [missing-attribute]
    prev_fill = torch.utils.deterministic.fill_uninitialized_memory
    # The deterministic flag is process-wide. Ops that other threads dispatch
    # during this backward also run in deterministic mode, and raise if they
    # have no deterministic kernel.
    # TODO(fegin): Share this toggle with scatter_add.py if a third op needs it.
    torch.use_deterministic_algorithms(True, warn_only=False)
    # Deterministic mode also NaN-fills every new tensor. Turn that off, as
    # set_determinism does: the fill kernel races with DeepEP comm streams.
    # pyrefly: ignore [missing-attribute]
    torch.utils.deterministic.fill_uninitialized_memory = False
    try:
        return torch.ops.torch_attn._varlen_attn_backward(
            grad_out,
            query,
            key,
            value,
            out,
            lse,
            cu_seq_q,
            cu_seq_k,
            max_q,
            max_k,
            is_causal=window_size == [-1, 0],
            rng_state=rng_state,
            scale=scale,
            window_size=window_size,
            backend=_FLASH,
        )
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn_only)
        # pyrefly: ignore [missing-attribute]
        torch.utils.deterministic.fill_uninitialized_memory = prev_fill


@_deterministic_varlen_attn_backward.register_fake
def _(grad_out, query, key, value, *args):
    return torch.empty_like(query), torch.empty_like(key), torch.empty_like(value)


def _backward(ctx, grad_out, grad_lse, grad_rng_state):
    query, key, value, cu_seq_q, cu_seq_k, out, lse, rng_state = ctx.saved_tensors
    dq, dk, dv = _deterministic_varlen_attn_backward(
        grad_out,
        query,
        key,
        value,
        out,
        lse,
        cu_seq_q,
        cu_seq_k,
        ctx.max_q,
        ctx.max_k,
        rng_state,
        ctx.scale,
        ctx.window_size,
    )
    return dq, dk, dv, None, None, None, None, None, None, None


def _setup_context(ctx, inputs, output):
    query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, scale, window_size, _ = inputs
    out, lse, rng_state = output
    ctx.mark_non_differentiable(lse, rng_state)
    ctx.save_for_backward(query, key, value, cu_seq_q, cu_seq_k, out, lse, rng_state)
    ctx.max_q = max_q
    ctx.max_k = max_k
    ctx.scale = scale
    ctx.window_size = window_size


deterministic_varlen_attn.register_autograd(_backward, setup_context=_setup_context)
