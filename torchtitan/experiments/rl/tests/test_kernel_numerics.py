#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Kernel-level numerics test: compare varlen_attn (contiguous KV) vs
varlen_attn_out (paged KV cache) to isolate the source of logprob divergence
between the trainer and generator attention paths.

The trainer uses ``varlen_attn`` with contiguous Q/K/V tensors, while the
generator (vLLM CUSTOM backend) uses ``varlen_attn_out`` with a paged KV cache
(block_table). Both use FA3 under the hood, but the paged access pattern can
change floating-point accumulation order.

Tests:
  1. varlen_attn vs varlen_attn_out with contiguous KV (no paging) — should
     be bitwise identical.
  2. varlen_attn (contiguous) vs varlen_attn_out (paged KV cache) — measures
     the numerical impact of paging.
  3. varlen_attn_out paged with different page sizes — measures sensitivity
     to page size.
  4. Multiple sequence lengths — checks if divergence scales with seq_len.

Run (single GPU):
    python torchtitan/experiments/rl/tests/test_kernel_numerics.py
"""

import logging
import sys

import torch
from torch.nn.attention import (
    activate_flash_attention_impl,
    current_flash_attention_impl,
)
from torch.nn.attention.varlen import varlen_attn, varlen_attn_out

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


def _activate_fa3():
    """Activate FA3 if not already active."""
    if current_flash_attention_impl() != "FA3":
        activate_flash_attention_impl("FA3")
        logger.info(f"Activated FA3 (was: {current_flash_attention_impl()})")


def _make_qkv(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random Q, K, V in packed layout (total_tokens, heads, head_dim)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    total_tokens = batch_size * seq_len
    q = torch.randn(
        total_tokens, num_heads, head_dim, device=device, dtype=dtype, generator=gen
    )
    k = torch.randn(
        total_tokens, num_kv_heads, head_dim, device=device, dtype=dtype, generator=gen
    )
    v = torch.randn(
        total_tokens, num_kv_heads, head_dim, device=device, dtype=dtype, generator=gen
    )
    return q, k, v


def _make_cu_seqlens(
    batch_size: int, seq_len: int, device: torch.device
) -> torch.Tensor:
    """Cumulative sequence lengths for uniform-length sequences."""
    return torch.arange(
        0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=device
    )


def _kv_to_paged(
    k: torch.Tensor,
    v: torch.Tensor,
    batch_size: int,
    seq_len: int,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert contiguous K/V into a paged KV cache + block_table + seqused_k.

    Returns:
        key_cache: (num_blocks, page_size, num_kv_heads, head_dim)
        value_cache: (num_blocks, page_size, num_kv_heads, head_dim)
        block_table: (batch_size, blocks_per_seq) int32
        seqused_k: (batch_size,) int32
    """
    num_kv_heads = k.shape[1]
    head_dim = k.shape[2]
    device = k.device
    dtype = k.dtype

    blocks_per_seq = (seq_len + page_size - 1) // page_size
    num_blocks = batch_size * blocks_per_seq

    key_cache = torch.zeros(
        num_blocks, page_size, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    value_cache = torch.zeros(
        num_blocks, page_size, num_kv_heads, head_dim, device=device, dtype=dtype
    )

    # Reshape K/V to (batch, seq_len, heads, head_dim) for easier slicing
    k_batched = k.view(batch_size, seq_len, num_kv_heads, head_dim)
    v_batched = v.view(batch_size, seq_len, num_kv_heads, head_dim)

    block_table = torch.zeros(
        batch_size, blocks_per_seq, dtype=torch.int32, device=device
    )
    seqused_k = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

    for b in range(batch_size):
        for blk_idx in range(blocks_per_seq):
            global_blk = b * blocks_per_seq + blk_idx
            block_table[b, blk_idx] = global_blk

            start = blk_idx * page_size
            end = min(start + page_size, seq_len)
            length = end - start

            key_cache[global_blk, :length] = k_batched[b, start:end]
            value_cache[global_blk, :length] = v_batched[b, start:end]

    return key_cache, value_cache, block_table, seqused_k


def _compare(label: str, out_a: torch.Tensor, out_b: torch.Tensor) -> dict:
    """Compare two attention outputs and log results."""
    identical = torch.equal(out_a, out_b)
    delta = (out_a.float() - out_b.float()).abs()
    max_delta = delta.max().item()
    mean_delta = delta.mean().item()

    # Relative error (avoid div-by-zero)
    denom = out_a.float().abs().clamp(min=1e-12)
    rel_error = (delta / denom).max().item()

    status = "PASS (bitwise identical)" if identical else "FAIL (NOT identical)"
    logger.info(f"  [{label}] {status}")
    logger.info(f"    max_abs_delta  = {max_delta:.6e}")
    logger.info(f"    mean_abs_delta = {mean_delta:.6e}")
    logger.info(f"    max_rel_error  = {rel_error:.6e}")

    return {
        "label": label,
        "identical": identical,
        "max_delta": max_delta,
        "mean_delta": mean_delta,
        "max_rel_error": rel_error,
    }


def test_varlen_attn_vs_varlen_attn_out_contiguous(
    batch_size: int = 1,
    seq_len: int = 128,
    num_heads: int = 16,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    device: torch.device = torch.device("cuda"),
) -> dict:
    """Test 1: varlen_attn vs varlen_attn_out with contiguous KV (no paging).

    Both should produce bitwise-identical output since they call the same
    underlying kernel with the same data layout.
    """
    q, k, v = _make_qkv(batch_size, seq_len, num_heads, num_kv_heads, head_dim, device)
    cu_seqlens = _make_cu_seqlens(batch_size, seq_len, device)
    scale = head_dim**-0.5

    out_attn = varlen_attn(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        seq_len,
        seq_len,
        scale=scale,
        window_size=(-1, 0),
    )

    out_buf = torch.empty_like(q)
    varlen_attn_out(
        out_buf,
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        seq_len,
        seq_len,
        scale=scale,
        window_size=(-1, 0),
    )

    return _compare("varlen_attn vs varlen_attn_out (contiguous)", out_attn, out_buf)


def test_contiguous_vs_paged(
    batch_size: int = 1,
    seq_len: int = 128,
    num_heads: int = 16,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 16,
    device: torch.device = torch.device("cuda"),
) -> dict:
    """Test 2: varlen_attn (contiguous KV) vs varlen_attn_out (paged KV cache).

    This isolates the numerical impact of paged KV cache access patterns.
    Page size 16 is vLLM's default.
    """
    q, k, v = _make_qkv(batch_size, seq_len, num_heads, num_kv_heads, head_dim, device)
    cu_seqlens = _make_cu_seqlens(batch_size, seq_len, device)
    scale = head_dim**-0.5

    # Contiguous path (trainer)
    out_contiguous = varlen_attn(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        seq_len,
        seq_len,
        scale=scale,
        window_size=(-1, 0),
    )

    # Paged path (generator)
    key_cache, value_cache, block_table, seqused_k = _kv_to_paged(
        k, v, batch_size, seq_len, page_size
    )
    out_paged = torch.empty_like(q)
    varlen_attn_out(
        out_paged,
        q,
        key_cache,
        value_cache,
        cu_seqlens,
        None,
        seq_len,
        seq_len,
        scale=scale,
        window_size=(-1, 0),
        block_table=block_table,
        seqused_k=seqused_k,
    )

    return _compare(
        f"contiguous vs paged (page_size={page_size})", out_contiguous, out_paged
    )


def test_paged_page_sizes(
    batch_size: int = 1,
    seq_len: int = 256,
    num_heads: int = 16,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    page_sizes: tuple[int, ...] = (16, 64, 128, 256),
    device: torch.device = torch.device("cuda"),
) -> list[dict]:
    """Test 3: Compare paged outputs across different page sizes.

    When page_size >= seq_len, paging degenerates to contiguous and should
    match the contiguous baseline.
    """
    q, k, v = _make_qkv(batch_size, seq_len, num_heads, num_kv_heads, head_dim, device)
    cu_seqlens = _make_cu_seqlens(batch_size, seq_len, device)
    scale = head_dim**-0.5

    out_contiguous = varlen_attn(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        seq_len,
        seq_len,
        scale=scale,
        window_size=(-1, 0),
    )

    results = []
    for ps in page_sizes:
        key_cache, value_cache, block_table, seqused_k = _kv_to_paged(
            k, v, batch_size, seq_len, ps
        )
        out_paged = torch.empty_like(q)
        varlen_attn_out(
            out_paged,
            q,
            key_cache,
            value_cache,
            cu_seqlens,
            None,
            seq_len,
            seq_len,
            scale=scale,
            window_size=(-1, 0),
            block_table=block_table,
            seqused_k=seqused_k,
        )
        results.append(
            _compare(f"contiguous vs paged (page_size={ps})", out_contiguous, out_paged)
        )

    return results


def test_seq_len_scaling(
    batch_size: int = 1,
    num_heads: int = 16,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 16,
    seq_lens: tuple[int, ...] = (32, 64, 128, 256, 512),
    device: torch.device = torch.device("cuda"),
) -> list[dict]:
    """Test 4: Check how contiguous-vs-paged divergence scales with seq_len."""
    results = []
    for sl in seq_lens:
        q, k, v = _make_qkv(batch_size, sl, num_heads, num_kv_heads, head_dim, device)
        cu_seqlens = _make_cu_seqlens(batch_size, sl, device)
        scale = head_dim**-0.5

        out_contiguous = varlen_attn(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            sl,
            sl,
            scale=scale,
            window_size=(-1, 0),
        )

        key_cache, value_cache, block_table, seqused_k = _kv_to_paged(
            k, v, batch_size, sl, page_size
        )
        out_paged = torch.empty_like(q)
        varlen_attn_out(
            out_paged,
            q,
            key_cache,
            value_cache,
            cu_seqlens,
            None,
            sl,
            sl,
            scale=scale,
            window_size=(-1, 0),
            block_table=block_table,
            seqused_k=seqused_k,
        )
        results.append(
            _compare(f"seq_len={sl}, page_size={page_size}", out_contiguous, out_paged)
        )

    return results


def test_multi_sequence_batch(
    batch_size: int = 4,
    seq_len: int = 128,
    num_heads: int = 16,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 16,
    device: torch.device = torch.device("cuda"),
) -> dict:
    """Test 5: Multi-sequence batch to check batch-dimension effects."""
    return test_contiguous_vs_paged(
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        device=device,
    )


def main():
    if not torch.cuda.is_available():
        logger.error("CUDA not available — this test requires a GPU.")
        sys.exit(1)

    device = torch.device("cuda")
    _activate_fa3()

    # Use Qwen3-0.6B dimensions: num_heads=16, num_kv_heads=8, head_dim=64
    qwen3_kwargs = dict(num_heads=16, num_kv_heads=8, head_dim=64)

    all_pass = True

    logger.info("=" * 70)
    logger.info("Test 1: varlen_attn vs varlen_attn_out (contiguous, no paging)")
    logger.info("=" * 70)
    r = test_varlen_attn_vs_varlen_attn_out_contiguous(device=device, **qwen3_kwargs)
    all_pass &= r["identical"]

    logger.info("")
    logger.info("=" * 70)
    logger.info("Test 2: contiguous vs paged KV cache (page_size=16, vLLM default)")
    logger.info("=" * 70)
    r = test_contiguous_vs_paged(page_size=16, device=device, **qwen3_kwargs)
    all_pass &= r["identical"]

    logger.info("")
    logger.info("=" * 70)
    logger.info("Test 3: contiguous vs paged across page sizes")
    logger.info("=" * 70)
    results = test_paged_page_sizes(device=device, **qwen3_kwargs)

    logger.info("")
    logger.info("=" * 70)
    logger.info("Test 4: contiguous vs paged across sequence lengths")
    logger.info("=" * 70)
    results = test_seq_len_scaling(device=device, **qwen3_kwargs)

    logger.info("")
    logger.info("=" * 70)
    logger.info("Test 5: multi-sequence batch (batch_size=4)")
    logger.info("=" * 70)
    r = test_multi_sequence_batch(device=device, **qwen3_kwargs)

    logger.info("")
    logger.info("=" * 70)
    if all_pass:
        logger.info(
            "ALL TESTS PASSED: contiguous and paged paths are bitwise identical"
        )
    else:
        logger.info("SOME TESTS FAILED: contiguous and paged paths diverge numerically")
        logger.info("This explains the logprob mismatch between trainer and generator.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
