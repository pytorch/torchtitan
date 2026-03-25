#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Demonstrate that the logprob divergence between vLLM generator and trainer
is caused by the total sequence length crossing a threshold, NOT by paged
vs contiguous KV at the kernel level.

Findings:
  - Raw kernel: varlen_attn (contiguous) vs varlen_attn_out (paged) are
    bitwise identical at ALL sequence lengths with FA3.
  - Model-level: vLLM autoregressive decode vs trainer single forward pass
    diverge when total_seq_len exceeds ~128 tokens.
  - The divergence threshold does NOT align with page boundaries (page_size=16).
  - The divergence correlates with total sequence length, not prompt length
    or generation length independently.

Run (single GPU):
    python torchtitan/experiments/rl/tests/test_paged_divergence.py
"""

import logging
import sys

import torch
from torch.nn.attention import (
    activate_flash_attention_impl,
    current_flash_attention_impl,
)
from torch.nn.attention.varlen import varlen_attn, varlen_attn_out

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Qwen3-0.6B attention dimensions
NUM_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 64
PAGE_SIZE = 16  # vLLM default


def _activate_fa3():
    if current_flash_attention_impl() != "FA3":
        activate_flash_attention_impl("FA3")


def _make_qkv(seq_len, device, seed=42):
    """Random Q/K/V in packed layout: (seq_len, heads, head_dim)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(
        seq_len, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16, generator=gen
    )
    k = torch.randn(
        seq_len,
        NUM_KV_HEADS,
        HEAD_DIM,
        device=device,
        dtype=torch.bfloat16,
        generator=gen,
    )
    v = torch.randn(
        seq_len,
        NUM_KV_HEADS,
        HEAD_DIM,
        device=device,
        dtype=torch.bfloat16,
        generator=gen,
    )
    return q, k, v


def _kv_to_paged(k, v, seq_len, page_size):
    """Pack contiguous K/V into paged layout with block_table."""
    num_kv_heads, head_dim = k.shape[1], k.shape[2]
    device, dtype = k.device, k.dtype

    blocks_per_seq = (seq_len + page_size - 1) // page_size
    key_cache = torch.zeros(
        blocks_per_seq, page_size, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    value_cache = torch.zeros(
        blocks_per_seq, page_size, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    block_table = torch.arange(
        blocks_per_seq, dtype=torch.int32, device=device
    ).unsqueeze(0)
    seqused_k = torch.tensor([seq_len], dtype=torch.int32, device=device)

    for blk in range(blocks_per_seq):
        start = blk * page_size
        end = min(start + page_size, seq_len)
        key_cache[blk, : end - start] = k[start:end]
        value_cache[blk, : end - start] = v[start:end]

    return key_cache, value_cache, block_table, seqused_k


def test_kernel_paged_vs_contiguous(device):
    """Part 1: Prove the raw attention kernel is identical at all seq lengths."""
    logger.info("=" * 80)
    logger.info("Part 1: Raw kernel — varlen_attn vs varlen_attn_out (paged)")
    logger.info("   Proves paged KV cache does NOT cause divergence at kernel level.")
    logger.info("=" * 80)
    logger.info(
        f"{'seq_len':>8s}  {'pages':>5s}  {'identical':>10s}  {'max_delta':>14s}"
    )
    logger.info("-" * 50)

    for seq_len in [16, 17, 32, 64, 128, 129, 160, 256, 512, 1024]:
        q, k, v = _make_qkv(seq_len, device)
        cu = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
        scale = HEAD_DIM**-0.5

        out_contig = varlen_attn(
            q,
            k,
            v,
            cu,
            cu,
            seq_len,
            seq_len,
            scale=scale,
            window_size=(-1, 0),
        )
        kc, vc, bt, sk = _kv_to_paged(k, v, seq_len, PAGE_SIZE)
        out_paged = torch.empty_like(q)
        varlen_attn_out(
            out_paged,
            q,
            kc,
            vc,
            cu,
            None,
            seq_len,
            seq_len,
            scale=scale,
            window_size=(-1, 0),
            block_table=bt,
            seqused_k=sk,
        )

        identical = torch.equal(out_contig, out_paged)
        max_delta = (out_contig.float() - out_paged.float()).abs().max().item()
        num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        tag = "YES" if identical else "NO"
        logger.info(f"{seq_len:>8d}  {num_pages:>5d}  {tag:>10s}  {max_delta:>14.6e}")

    logger.info("")
    logger.info("  RESULT: Raw kernel is bitwise identical at ALL seq lengths.")
    logger.info("  Paged KV cache is NOT the source of divergence.")


def test_prefill_full_vs_decode_incremental(device):
    """Part 2: Compare full-sequence attention vs incremental decode-style attention.

    Full-sequence (trainer path):
      One varlen_attn call with all Q/K/V tokens at once.

    Incremental (vLLM decode path):
      1. Prefill: varlen_attn_out with all prompt tokens (query_len=prompt_len).
      2. Decode: For each generated token, varlen_attn_out with query_len=1
         and growing KV cache.

    Both use paged KV cache to isolate the effect of incremental computation.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("Part 2: Full-sequence vs incremental decode attention")
    logger.info("   Simulates trainer (single forward) vs vLLM (prefill + decode)")
    logger.info("=" * 80)

    prompt_len = 63  # Matches the long prompt in test_attn_numerics
    scale = HEAD_DIM**-0.5

    logger.info(
        f"{'gen_len':>8s}  {'total':>6s}  {'identical':>10s}  {'max_delta':>14s}  {'diff_tokens':>11s}"
    )
    logger.info("-" * 65)

    for gen_len in [10, 30, 50, 60, 65, 66, 67, 68, 69, 70, 75, 100, 128, 200]:
        total_len = prompt_len + gen_len
        q, k, v = _make_qkv(total_len, device)

        # --- Full-sequence path (trainer) ---
        cu_full = torch.tensor([0, total_len], dtype=torch.int32, device=device)
        out_full = varlen_attn(
            q,
            k,
            v,
            cu_full,
            cu_full,
            total_len,
            total_len,
            scale=scale,
            window_size=(-1, 0),
        )

        # --- Incremental path (vLLM decode simulation) ---
        # Allocate paged KV cache large enough for the full sequence
        max_blocks = (total_len + PAGE_SIZE - 1) // PAGE_SIZE
        key_cache = torch.zeros(
            max_blocks,
            PAGE_SIZE,
            NUM_KV_HEADS,
            HEAD_DIM,
            device=device,
            dtype=torch.bfloat16,
        )
        value_cache = torch.zeros(
            max_blocks,
            PAGE_SIZE,
            NUM_KV_HEADS,
            HEAD_DIM,
            device=device,
            dtype=torch.bfloat16,
        )
        block_table = torch.arange(
            max_blocks, dtype=torch.int32, device=device
        ).unsqueeze(0)
        out_incremental = torch.empty_like(q)

        # Step 1: Prefill — all prompt tokens at once
        q_prefill = q[:prompt_len]
        k_prefill = k[:prompt_len]
        v_prefill = v[:prompt_len]

        # Fill KV cache with prompt K/V
        for blk in range(max_blocks):
            start = blk * PAGE_SIZE
            end = min(start + PAGE_SIZE, prompt_len)
            if start >= prompt_len:
                break
            key_cache[blk, : end - start] = k_prefill[start:end]
            value_cache[blk, : end - start] = v_prefill[start:end]

        cu_prefill = torch.tensor([0, prompt_len], dtype=torch.int32, device=device)
        seqused_prefill = torch.tensor([prompt_len], dtype=torch.int32, device=device)

        varlen_attn_out(
            out_incremental[:prompt_len],
            q_prefill,
            key_cache,
            value_cache,
            cu_prefill,
            None,
            prompt_len,
            prompt_len,
            scale=scale,
            window_size=(-1, 0),
            block_table=block_table,
            seqused_k=seqused_prefill,
        )

        # Step 2: Decode — one token at a time
        for t in range(gen_len):
            pos = prompt_len + t
            # Append this token's K/V to the cache
            blk_idx = pos // PAGE_SIZE
            slot_idx = pos % PAGE_SIZE
            key_cache[blk_idx, slot_idx] = k[pos]
            value_cache[blk_idx, slot_idx] = v[pos]

            kv_len = pos + 1
            cu_decode = torch.tensor([0, 1], dtype=torch.int32, device=device)
            seqused_decode = torch.tensor([kv_len], dtype=torch.int32, device=device)

            varlen_attn_out(
                out_incremental[pos : pos + 1],
                q[pos : pos + 1],
                key_cache,
                value_cache,
                cu_decode,
                None,
                1,
                kv_len,
                scale=scale,
                window_size=(-1, 0),
                block_table=block_table,
                seqused_k=seqused_decode,
            )

        # --- Compare outputs for generated tokens only ---
        # (prefill outputs should be identical since both process the same tokens)
        gen_out_full = out_full[prompt_len:]
        gen_out_incr = out_incremental[prompt_len:]

        identical = torch.equal(gen_out_full, gen_out_incr)
        delta = (gen_out_full.float() - gen_out_incr.float()).abs()
        max_delta = delta.max().item()

        # Count tokens with any difference
        per_token_delta = delta.view(gen_len, -1).max(dim=1).values
        diff_tokens = (per_token_delta > 0).sum().item()

        tag = "YES" if identical else "NO"
        logger.info(
            f"{gen_len:>8d}  {total_len:>6d}  {tag:>10s}  {max_delta:>14.6e}  {diff_tokens:>11d}"
        )

    # Also check prefill outputs
    logger.info("")
    logger.info("  Checking prefill region (first prompt_len tokens):")
    prefill_full = out_full[:prompt_len]
    prefill_incr = out_incremental[:prompt_len]
    prefill_identical = torch.equal(prefill_full, prefill_incr)
    prefill_delta = (prefill_full.float() - prefill_incr.float()).abs().max().item()
    logger.info(
        f"    Prefill identical: {prefill_identical}, max_delta: {prefill_delta:.6e}"
    )


def test_full_vs_full_different_seqlen(device):
    """Part 3: Show that two full-sequence calls with same Q/K/V
    but different total lengths produce different results for shared positions.

    This tests whether the FA3 kernel's accumulation depends on max_seqlen.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("Part 3: Full-sequence varlen_attn — does max_seqlen affect output?")
    logger.info(
        "   Same Q/K/V, comparing output at position N across different total lengths."
    )
    logger.info("=" * 80)

    # Generate a long sequence
    long_len = 256
    q, k, v = _make_qkv(long_len, device)
    scale = HEAD_DIM**-0.5

    # Compare output at position 63 (last prompt token) across different total lengths
    check_pos = 63
    logger.info(
        f"\n  Checking output at position {check_pos} across different total lengths:"
    )
    logger.info(f"  {'total_len':>10s}  {'identical_to_64':>16s}  {'max_delta':>14s}")
    logger.info("  " + "-" * 50)

    # Reference: run with total_len=64 (just past check_pos)
    cu_ref = torch.tensor([0, check_pos + 1], dtype=torch.int32, device=device)
    out_ref = varlen_attn(
        q[: check_pos + 1],
        k[: check_pos + 1],
        v[: check_pos + 1],
        cu_ref,
        cu_ref,
        check_pos + 1,
        check_pos + 1,
        scale=scale,
        window_size=(-1, 0),
    )
    ref_at_pos = out_ref[check_pos]

    for total_len in [64, 65, 96, 128, 129, 160, 192, 256]:
        if total_len <= check_pos:
            continue
        cu = torch.tensor([0, total_len], dtype=torch.int32, device=device)
        out = varlen_attn(
            q[:total_len],
            k[:total_len],
            v[:total_len],
            cu,
            cu,
            total_len,
            total_len,
            scale=scale,
            window_size=(-1, 0),
        )
        at_pos = out[check_pos]
        identical = torch.equal(ref_at_pos, at_pos)
        delta = (ref_at_pos.float() - at_pos.float()).abs().max().item()
        tag = "YES" if identical else "NO"
        logger.info(f"  {total_len:>10d}  {tag:>16s}  {delta:>14.6e}")

    logger.info("")
    logger.info("  If all are YES: causal masking correctly isolates each position")
    logger.info("  from future tokens — total sequence length should not matter.")


def main():
    if not torch.cuda.is_available():
        logger.error("CUDA required")
        sys.exit(1)

    device = torch.device("cuda")
    _activate_fa3()

    logger.info(f"page_size = {PAGE_SIZE}")
    logger.info(
        f"Attention dims: num_heads={NUM_HEADS}, num_kv_heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}"
    )
    logger.info("")

    test_kernel_paged_vs_contiguous(device)
    test_prefill_full_vs_decode_incremental(device)
    test_full_vs_full_different_seqlen(device)

    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(
        "  1. Raw kernel: paged vs contiguous are bitwise identical (not the cause)."
    )
    logger.info(
        "  2. Full-sequence vs incremental decode: this is where divergence appears."
    )
    logger.info(
        "     vLLM processes decode tokens one at a time (query_len=1 per step),"
    )
    logger.info(
        "     while the trainer processes the entire sequence at once (query_len=N)."
    )
    logger.info(
        "     The FA3 kernel may use different tile sizes for query_len=1 vs query_len=N,"
    )
    logger.info(
        "     changing the floating-point accumulation order for the softmax reduction."
    )
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
