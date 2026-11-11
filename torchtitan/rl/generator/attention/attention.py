# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Paged-KV varlen attention through PyTorch's FlashAttention, the kernel the trainer's
``VarlenInnerAttention`` runs. Every engine's TorchTitan attention backend calls it, so
the generator computes attention with the trainer's kernel whatever the engine."""

from typing import Any

import torch
from torch.nn.attention import current_flash_attention_impl, sdpa_kernel, SDPBackend
from torch.nn.attention.varlen import AuxRequest

from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode


def paged_varlen_attention_out(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
    scale: float,
    window_size: tuple[int, int],
    enable_gqa: bool,
    return_lse: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Causal varlen attention of new query tokens over a paged KV cache, written into ``out``.

    Args:
        out: ``(num_tokens, num_heads, value_head_dim)`` output buffer.
        query: ``(num_tokens, num_heads, head_dim)``.
        key_cache: ``(num_pages, page_size, num_kv_heads, head_dim)``.
        value_cache: ``(num_pages, page_size, num_kv_heads, value_head_dim)``.
        cu_seqlens_q: ``(batch + 1,)`` int32 offsets of each sequence's query tokens.
        max_seqlen_q: Upper bound on any sequence's query tokens.
        max_seqlen_k: Upper bound on any sequence's KV tokens.
        seqused_k: ``(batch,)`` int32 KV tokens per sequence, its new tokens included.
        block_table: ``(batch, max_pages)`` int32 page ids of each sequence.
        scale: Softmax scale.
        window_size: ``(left, right)`` attention window; ``(-1, 0)`` is full causal.
        enable_gqa: Whether ``num_heads`` is a multiple of ``num_kv_heads`` > 1.
        return_lse: Also return the ``(num_heads, num_tokens)`` logsumexp.

    Returns:
        The written ``out[:num_tokens]``, and the logsumexp with ``return_lse``.
    """
    # FA3 and FA4 infer key lengths from block_table + seqused_k for paged
    # KV. FA4 explicitly rejects page_table together with cu_seqlens_k.
    # FA2 requires cu_seqlens_k to be explicitly set.
    fa_impl = current_flash_attention_impl()
    if fa_impl in ("FA3", "FA4"):
        cu_seqlens_k = None
    else:
        num_seqs = seqused_k.shape[0]
        cu_seqlens_k = torch.zeros(num_seqs + 1, dtype=torch.int32, device=query.device)
        cu_seqlens_k[1:] = torch.cumsum(seqused_k, dim=0)
    extra_kwargs: dict[str, Any] = {}

    # TODO(pytorch/pytorch#179760): FA2's auto num_splits heuristic
    # produces NaN intermittently with paged KV (block_table). Force
    # num_splits=1 as a workaround until the root cause is fixed
    # upstream. current_flash_attention_impl() returns None when FA2
    # is the implicit default (SM < 9.0). For FA3, only force
    # num_splits=1 in batch-invariant mode (determinism).
    if fa_impl in (None, "FA2") or is_in_batch_invariant_mode():
        extra_kwargs["num_splits"] = 1

    if enable_gqa:
        extra_kwargs["enable_gqa"] = True

    if return_lse:
        extra_kwargs["return_aux"] = AuxRequest(lse=True)

    # varlen_attn otherwise prefers cuDNN where it accepts the call (paged KV with
    # power-of-two pages, outside batch-invariant mode, whose num_splits=1 it rejects).
    # On the generator's paged decode and mixed batches, cuDNN is 1.5-15x slower than
    # FA4 (e.g. 1.70 vs 0.51 ms for 100 decode rows at head dim 256 on GB300).
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        return torch.nn.attention.varlen.varlen_attn_out(
            out,
            query,
            key_cache,
            value_cache,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            scale=scale,
            window_size=window_size,
            block_table=block_table,
            seqused_k=seqused_k,
            **extra_kwargs,
        )
