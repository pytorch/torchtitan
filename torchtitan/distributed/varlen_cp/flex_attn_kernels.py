# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Range-based attention using PyTorch's ``flex_attention``.

Provides ``flex_attn_forward`` and ``flex_attn_backward`` that accept
range descriptors (q_ranges, k_ranges, attn_type_map) and compute
attention using ``torch.nn.attention.flex_attention`` with a
dynamically created ``BlockMask``.

This is the PyTorch-native replacement for magi-attention's CUTLASS FFA
kernel. ``flex_attention`` generates optimized fused kernels via
``torch.compile``, supports GQA, returns LSE, and handles block
sparsity (skipping masked-out blocks).

Shape convention:
    Q/K/V: (seqlen, n_heads, head_dim)  — packed token format
    LSE:   (seqlen, n_heads) float32
    Ranges: (N, 2) int32  — [start, end) pairs
    attn_type_map: (N,) int32  — 0=FULL, 1=CAUSAL
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

# Raise dynamo cache limit to avoid fallback to unfused O(n²) backward
# when many different BlockMask shapes are encountered across layers.
torch._dynamo.config.cache_size_limit = 256


@dataclass
class FlexAttnMeta:
    """Metadata returned from flex_attn_forward."""

    lse: torch.Tensor  # (seqlen, n_heads) float32
    block_mask: BlockMask | None = field(default=None, repr=False)


# Compile flex_attention for fused kernel generation.
_compiled_flex_attention = torch.compile(flex_attention)


def _build_range_mask_mod(
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: torch.Tensor,
):
    """Build a mask_mod function for flex_attention from range descriptors.

    The returned function checks whether (q_idx, kv_idx) falls within any
    of the given ranges. For CAUSAL ranges, it additionally enforces
    relative causal masking: (q_idx - q_start) >= (kv_idx - k_start).

    Args:
        q_ranges: (N, 2) int32 tensor, [q_start, q_end) per range.
        k_ranges: (N, 2) int32 tensor, [k_start, k_end) per range.
        attn_type_map: (N,) int32 tensor. 0=FULL, 1=CAUSAL.
    """
    # Keep tensors for closure capture (they must be on the right device)
    qr = q_ranges  # (N, 2)
    kr = k_ranges  # (N, 2)
    at = attn_type_map  # (N,)

    def mask_mod(b, h, q_idx, kv_idx):
        allowed = False
        for i in range(qr.shape[0]):
            q_in = (q_idx >= qr[i, 0]) & (q_idx < qr[i, 1])
            k_in = (kv_idx >= kr[i, 0]) & (kv_idx < kr[i, 1])
            in_range = q_in & k_in
            # Causal: relative position check
            causal_ok = (q_idx - qr[i, 0]) >= (kv_idx - kr[i, 0])
            range_ok = torch.where(at[i] == 1, in_range & causal_ok, in_range)
            allowed = allowed | range_ok
        return allowed

    return mask_mod


def build_flex_block_mask(
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: torch.Tensor,
    Q_LEN: int,
    KV_LEN: int,
    device: torch.device,
) -> BlockMask:
    """Build a BlockMask from range descriptors.

    Call this once and pass the result to ``flex_attn_forward(block_mask_arg=...)``
    to avoid rebuilding the mask every call (e.g. cross-layer caching).
    """
    qr = q_ranges.contiguous().to(dtype=torch.int32, device=device)
    kr = k_ranges.contiguous().to(dtype=torch.int32, device=device)
    at = attn_type_map.contiguous().to(dtype=torch.int32, device=device)
    mask_mod = _build_range_mask_mod(qr, kr, at)
    return create_block_mask(
        mask_mod,
        B=1,
        H=None,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        device=device,
    )


def flex_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: torch.Tensor,
    block_mask_arg: "BlockMask | None" = None,
) -> tuple[torch.Tensor, FlexAttnMeta]:
    """Range-based attention forward using ``flex_attention``.

    Drop-in replacement for magi-attention's ``flex_flash_attn_func``.

    Args:
        q: (Q_LEN, n_heads_q, head_dim), bf16/fp16.
        k: (KV_LEN, n_kv_heads, head_dim), bf16/fp16.
        v: (KV_LEN, n_kv_heads, head_dim), bf16/fp16.
        q_ranges: (N, 2) int32 tensor of [start, end) Q index ranges.
        k_ranges: (N, 2) int32 tensor of [start, end) K index ranges.
        attn_type_map: (N,) int32 tensor. 0=FULL, 1=CAUSAL.
        block_mask_arg: Pre-built BlockMask (for caching across layers).
            If provided, q_ranges/k_ranges/attn_type_map are ignored.

    Returns:
        (output, meta) where:
            output: (Q_LEN, n_heads_q, head_dim) same dtype as q.
            meta.lse: (Q_LEN, n_heads_q) float32 — log-sum-exp.
    """
    Q_LEN, n_heads_q, head_dim = q.shape
    KV_LEN = k.shape[0]
    n_kv_heads = k.shape[1]
    device = q.device
    dtype = q.dtype

    # Ensure ranges are contiguous int32 on device
    qr = q_ranges.contiguous().to(dtype=torch.int32, device=device)
    kr = k_ranges.contiguous().to(dtype=torch.int32, device=device)
    at = attn_type_map.contiguous().to(dtype=torch.int32, device=device)
    num_ranges = qr.shape[0]

    # Allocate output for empty case
    if num_ranges == 0 or Q_LEN == 0:
        out = torch.zeros(Q_LEN, n_heads_q, head_dim, device=device, dtype=dtype)
        lse = torch.full(
            (Q_LEN, n_heads_q), float("-inf"), device=device, dtype=torch.float32
        )
        return out, FlexAttnMeta(lse=lse, block_mask=None)

    # flex_attention expects (B, H, S, D) layout
    q_bshd = q.permute(1, 0, 2).unsqueeze(0)  # (1, n_heads_q, Q_LEN, head_dim)
    k_bshd = k.permute(1, 0, 2).unsqueeze(0)  # (1, n_kv_heads, KV_LEN, head_dim)
    v_bshd = v.permute(1, 0, 2).unsqueeze(0)

    # Use pre-built block mask if provided (for cross-layer caching)
    if block_mask_arg is not None:
        block_mask = block_mask_arg
    else:
        mask_mod = _build_range_mask_mod(qr, kr, at)
        block_mask = create_block_mask(
            mask_mod,
            B=1,
            H=None,
            Q_LEN=Q_LEN,
            KV_LEN=KV_LEN,
            device=device,
        )

    # Run flex_attention with compiled kernel
    out_bshd, lse_bhs = _compiled_flex_attention(
        q_bshd,
        k_bshd,
        v_bshd,
        block_mask=block_mask,
        enable_gqa=(n_heads_q != n_kv_heads),
        return_lse=True,
    )

    # Convert back to (seqlen, n_heads, head_dim) layout
    out = out_bshd.squeeze(0).permute(1, 0, 2).to(dtype)  # (Q_LEN, n_heads_q, head_dim)
    lse = lse_bhs.squeeze(0).permute(1, 0).to(torch.float32)  # (Q_LEN, n_heads_q)

    return out, FlexAttnMeta(lse=lse, block_mask=block_mask)


def flex_attn_backward(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: torch.Tensor,
    block_mask: BlockMask | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Range-based attention backward using ``flex_attention`` recompute.

    Recomputes the forward pass through ``_compiled_flex_attention`` and
    uses ``torch.autograd.grad`` to obtain gradients. When a ``block_mask``
    is provided (cached from forward), the expensive ``create_block_mask``
    call is skipped.

    Args:
        grad_output: (Q_LEN, n_heads_q, head_dim).
        q, k, v: same shapes as forward.
        out: forward output (unused, kept for API compatibility).
        lse: forward LSE (unused, kept for API compatibility).
        q_ranges, k_ranges, attn_type_map: same as forward.
        block_mask: Cached BlockMask from forward (avoids rebuild).

    Returns:
        (dq, dk, dv) all in float32.
            dq: (Q_LEN, n_heads_q, head_dim)
            dk: (KV_LEN, n_kv_heads, head_dim)
            dv: (KV_LEN, n_kv_heads, head_dim)
    """
    Q_LEN = q.shape[0]
    KV_LEN = k.shape[0]
    n_heads_q = q.shape[1]
    head_dim = q.shape[2]
    n_kv_heads = k.shape[1]
    device = q.device

    # Empty case
    if q_ranges.shape[0] == 0 or Q_LEN == 0:
        dq = torch.zeros(Q_LEN, n_heads_q, head_dim, device=device, dtype=torch.float32)
        dk = torch.zeros(
            KV_LEN, n_kv_heads, head_dim, device=device, dtype=torch.float32
        )
        dv = torch.zeros(
            KV_LEN, n_kv_heads, head_dim, device=device, dtype=torch.float32
        )
        return dq, dk, dv

    # Build or reuse BlockMask
    qr = q_ranges.contiguous().to(dtype=torch.int32, device=device)
    kr = k_ranges.contiguous().to(dtype=torch.int32, device=device)
    at = attn_type_map.contiguous().to(dtype=torch.int32, device=device)

    if block_mask is not None:
        bm = block_mask
    else:
        mask_mod = _build_range_mask_mod(qr, kr, at)
        bm = create_block_mask(
            mask_mod,
            B=1,
            H=None,
            Q_LEN=Q_LEN,
            KV_LEN=KV_LEN,
            device=device,
        )

    # Detach and re-enable grad for recompute
    q_det = q.detach().requires_grad_(True)
    k_det = k.detach().requires_grad_(True)
    v_det = v.detach().requires_grad_(True)

    # Reshape to (B, H, S, D) layout
    q_bshd = q_det.permute(1, 0, 2).unsqueeze(0)
    k_bshd = k_det.permute(1, 0, 2).unsqueeze(0)
    v_bshd = v_det.permute(1, 0, 2).unsqueeze(0)
    grad_bshd = grad_output.permute(1, 0, 2).unsqueeze(0)

    with torch.enable_grad():
        out_bshd = _compiled_flex_attention(
            q_bshd,
            k_bshd,
            v_bshd,
            block_mask=bm,
            enable_gqa=(n_heads_q != n_kv_heads),
        )
        grads = torch.autograd.grad(
            out_bshd,
            (q_det, k_det, v_det),
            grad_bshd,
        )

    return (
        grads[0].to(torch.float32),
        grads[1].to(torch.float32),
        grads[2].to(torch.float32),
    )
