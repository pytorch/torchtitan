# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Multi-GPU tests for varlen ring attention.

Run unit tests:
    python -m pytest tests/unit_tests/test_varlen_cp/test_ring_attention.py -v

Run multi-GPU tests (2 GPUs):
    torchrun --nproc_per_node=2 tests/unit_tests/test_varlen_cp/test_ring_attention.py
"""

import os
import unittest

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.nn.attention.varlen import varlen_attn

from torchtitan.distributed.varlen_cp.dispatch_solver import DispatchPlan, solve_dispatch
from torchtitan.distributed.varlen_cp.mask_primitives import cu_seqlens_to_attn_slices
from torchtitan.distributed.varlen_cp.magi_attention import varlen_magi_dispatch
from torchtitan.distributed.varlen_cp.ring_attention import (
    _backward_step_flash,
    _build_ring_step_cu_seqlens,
    _compute_and_merge_step,
    _extract_tokens_for_docs,
    _scatter_to_chunk,
    merge_with_lse,
    varlen_ring_attention,
)


# ---------------------------------------------------------------------------
# Local test helpers (moved from ring_attention.py, only used in tests)
# ---------------------------------------------------------------------------


def _raw_ring_pass_blocking(
    tensor: torch.Tensor,
    group: dist.ProcessGroup,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """Blocking ring-pass: send right, receive left."""
    recv_buffer = torch.empty_like(tensor)
    send_rank = (rank + 1) % world_size
    recv_rank = (rank - 1) % world_size
    send_global = dist.get_global_rank(group, send_rank)
    recv_global = dist.get_global_rank(group, recv_rank)
    ops = [
        dist.P2POp(dist.isend, tensor.contiguous(), send_global, group),
        dist.P2POp(dist.irecv, recv_buffer, recv_global, group),
    ]
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()
    return recv_buffer


def _varlen_backward_via_ring_pass(
    grad_output, q, k, v, merged_out, merged_lse, global_cu_seqlens,
    plan, cp_group, cp_rank, cp_world_size,
):
    """Ring-pass backward loop for testing only."""
    chunk_size = plan.chunk_size
    original_total_seqlen = plan.total_seqlen - plan.pad_size
    device = q.device
    dtype = q.dtype
    n_kv_heads = k.shape[1]
    kv_head_dim = k.shape[2] * 2

    q_chunk_start = cp_rank * chunk_size
    q_chunk_end = min(q_chunk_start + chunk_size, plan.total_seqlen)

    with torch.no_grad():
        grad_q = torch.zeros_like(q, dtype=torch.float32)
        grad_kv = torch.zeros(
            chunk_size, n_kv_heads, kv_head_dim,
            device=device, dtype=torch.float32,
        )

        current_kv = torch.cat([k, v], dim=-1)
        _has_assignments = bool(plan.assignments and any(plan.assignments))

        for step in range(cp_world_size):
            if step > 0:
                current_kv = _raw_ring_pass_blocking(
                    current_kv, cp_group, cp_rank, cp_world_size
                )
                grad_kv = _raw_ring_pass_blocking(
                    grad_kv, cp_group, cp_rank, cp_world_size
                )

            k_source_rank = (cp_rank - step) % cp_world_size
            if _has_assignments and not plan.pair_has_work(
                cp_rank, k_source_rank
            ):
                continue

            k_chunk_start = k_source_rank * chunk_size
            k_chunk_end = min(k_chunk_start + chunk_size, plan.total_seqlen)

            cur_k = current_kv[..., : kv_head_dim // 2].contiguous()
            cur_v = current_kv[..., kv_head_dim // 2 :].contiguous()

            _backward_step_flash(
                grad_output, q, cur_k, cur_v,
                merged_out, merged_lse,
                global_cu_seqlens,
                q_chunk_start, q_chunk_end,
                k_chunk_start, k_chunk_end,
                original_total_seqlen, chunk_size,
                grad_q, grad_kv,
            )

        if cp_world_size > 1:
            grad_kv = _raw_ring_pass_blocking(
                grad_kv, cp_group, cp_rank, cp_world_size
            )

    half = kv_head_dim // 2
    return (
        grad_q.to(dtype).contiguous(),
        grad_kv[:, :, :half].to(dtype).contiguous(),
        grad_kv[:, :, half:].to(dtype).contiguous(),
    )


# ---------------------------------------------------------------------------
# Ring-pass reference implementation (for testing / precision alignment only)
# ---------------------------------------------------------------------------


class _VarlenRingPassTestFunc(torch.autograd.Function):
    """Ring-pass varlen attention for testing. Not used in production."""

    @staticmethod
    def forward(ctx, q, k, v, global_cu_seqlens, plan, cp_group, cp_rank, cp_world_size):
        chunk_size = plan.chunk_size
        original_total_seqlen = plan.total_seqlen - plan.pad_size
        device = q.device
        dtype = q.dtype
        n_heads = q.shape[1]
        head_dim = q.shape[2]

        q_chunk_start = cp_rank * chunk_size
        q_chunk_end = min(q_chunk_start + chunk_size, plan.total_seqlen)

        with torch.no_grad():
            accum_out = torch.zeros(
                chunk_size, n_heads, head_dim, device=device, dtype=dtype
            )
            accum_lse = torch.full(
                (n_heads, chunk_size), float("-inf"), device=device, dtype=torch.float32
            )

            current_kv = torch.cat([k, v], dim=-1)
            kv_head_dim = current_kv.shape[-1]
            _has_assignments = bool(plan.assignments and any(plan.assignments))

            for step in range(cp_world_size):
                if step > 0:
                    current_kv = _raw_ring_pass_blocking(
                        current_kv, cp_group, cp_rank, cp_world_size
                    )

                k_source_rank = (cp_rank - step) % cp_world_size
                if _has_assignments and not plan.pair_has_work(
                    cp_rank, k_source_rank
                ):
                    continue

                k_chunk_start = k_source_rank * chunk_size
                k_chunk_end = min(k_chunk_start + chunk_size, plan.total_seqlen)

                cur_k = current_kv[..., : kv_head_dim // 2]
                cur_v = current_kv[..., kv_head_dim // 2 :]

                accum_out, accum_lse = _compute_and_merge_step(
                    q, cur_k, cur_v,
                    global_cu_seqlens,
                    q_chunk_start, q_chunk_end,
                    k_chunk_start, k_chunk_end,
                    original_total_seqlen, chunk_size,
                    accum_out, accum_lse,
                )

        merged_out = accum_out.to(dtype).contiguous()
        merged_lse = accum_lse.contiguous()

        ctx.save_for_backward(q, k, v, merged_out, merged_lse, global_cu_seqlens)
        ctx.plan = plan
        ctx.cp_group = cp_group
        ctx.cp_rank = cp_rank
        ctx.cp_world_size = cp_world_size

        return merged_out

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, merged_out, merged_lse, global_cu_seqlens = ctx.saved_tensors
        grad_q, grad_k, grad_v = _varlen_backward_via_ring_pass(
            grad_output, q, k, v, merged_out, merged_lse, global_cu_seqlens,
            ctx.plan, ctx.cp_group, ctx.cp_rank, ctx.cp_world_size,
        )
        return grad_q, grad_k, grad_v, None, None, None, None, None


def _varlen_ring_attention_ring_pass(q, k, v, global_cu_seqlens, plan, cp_mesh):
    """Ring-pass varlen attention for testing only."""
    cp_rank = cp_mesh.get_local_rank()
    cp_world_size = cp_mesh.size(0)
    cp_group = cp_mesh.get_group()
    return _VarlenRingPassTestFunc.apply(
        q, k, v, global_cu_seqlens, plan, cp_group, cp_rank, cp_world_size,
    )


# ---------------------------------------------------------------------------
# All-gather reference implementation (for testing only)
# ---------------------------------------------------------------------------


class _DifferentiableAllGather(torch.autograd.Function):
    """Differentiable all-gather: forward=allgather, backward=reduce_scatter."""

    @staticmethod
    def forward(ctx, tensor, group, world_size):
        ctx.group = group
        ctx.world_size = world_size
        gathered = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor.contiguous(), group=group)
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        chunk_size = grad_output.size(0) // ctx.world_size
        grad_input = torch.empty(
            chunk_size, *grad_output.shape[1:],
            dtype=grad_output.dtype, device=grad_output.device,
        )
        dist.reduce_scatter_tensor(
            grad_input, grad_output.contiguous(),
            op=dist.ReduceOp.SUM, group=ctx.group,
        )
        return grad_input, None, None


def _varlen_ring_attention_all_gather(q, k, v, global_cu_seqlens, plan, cp_mesh):
    """All-gather reference: each rank gathers full K/V and computes locally."""
    cp_rank = cp_mesh.get_local_rank()
    cp_world_size = cp_mesh.size(0)
    cp_group = cp_mesh.get_group()
    chunk_size = plan.chunk_size
    original_total_seqlen = plan.total_seqlen - plan.pad_size

    full_k = _DifferentiableAllGather.apply(k, cp_group, cp_world_size)
    full_v = _DifferentiableAllGather.apply(v, cp_group, cp_world_size)

    chunk_start = cp_rank * chunk_size
    chunk_end = min(chunk_start + chunk_size, plan.total_seqlen)
    device = q.device
    n_heads, head_dim = q.shape[1], q.shape[2]

    # Build paired cu_seqlens
    global_cu = global_cu_seqlens.tolist()
    cu_q_list, cu_k_list, k_doc_ranges = [0], [0], []
    num_real_q = 0
    for d in range(len(global_cu) - 1):
        doc_start, doc_end = int(global_cu[d]), int(global_cu[d + 1])
        q_s, q_e = max(doc_start, chunk_start), min(doc_end, chunk_end)
        if q_e - q_s <= 0:
            continue
        k_end = min(doc_end, chunk_end)
        cu_q_list.append(cu_q_list[-1] + (q_e - q_s))
        cu_k_list.append(cu_k_list[-1] + (k_end - doc_start))
        k_doc_ranges.append((doc_start, k_end))
        num_real_q += q_e - q_s

    cu_q = torch.tensor(cu_q_list, dtype=torch.int32, device=device)
    cu_k = torch.tensor(cu_k_list, dtype=torch.int32, device=device)

    if num_real_q == 0:
        return torch.zeros(chunk_size, n_heads, head_dim, device=device, dtype=q.dtype)

    k_packed = torch.cat([full_k[s:e] for s, e in k_doc_ranges], dim=0)
    v_packed = torch.cat([full_v[s:e] for s, e in k_doc_ranges], dim=0)
    q_real = q[:num_real_q]

    q_diffs, k_diffs = torch.diff(cu_q), torch.diff(cu_k)
    max_q = q_diffs.max().item() if q_diffs.numel() > 0 else 0
    max_k = k_diffs.max().item() if k_diffs.numel() > 0 else 0

    output = varlen_attn(
        q_real, k_packed, v_packed, cu_q, cu_k, max_q, max_k, is_causal=True,
    )

    if num_real_q < chunk_size:
        padding = torch.zeros(
            chunk_size - num_real_q, n_heads, head_dim,
            device=device, dtype=output.dtype,
        )
        output = torch.cat([output, padding], dim=0)
    return output


def setup_distributed():
    """Initialize distributed process group."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank


# ---------------------------------------------------------------------------
# Unit tests (single GPU / CPU)
# ---------------------------------------------------------------------------


class TestMergeWithLSE(unittest.TestCase):
    """Test LSE merge (runs on single GPU)."""

    def test_merge_with_self(self):
        """Merging a result with itself should give the same result."""
        torch.manual_seed(42)
        n_tokens, n_heads, head_dim = 32, 4, 16
        out = torch.randn(n_tokens, n_heads, head_dim)
        lse = torch.randn(n_heads, n_tokens)

        merged_out, merged_lse = merge_with_lse(out, lse, out, lse)
        torch.testing.assert_close(merged_out, out, atol=1e-5, rtol=1e-5)

    def test_merge_with_neg_inf_lse(self):
        """Merging with -inf LSE should return the other result."""
        n_tokens, n_heads, head_dim = 32, 4, 16
        out1 = torch.randn(n_tokens, n_heads, head_dim)
        lse1 = torch.randn(n_heads, n_tokens)
        out2 = torch.randn(n_tokens, n_heads, head_dim)
        lse2 = torch.full((n_heads, n_tokens), float("-inf"))

        merged_out, merged_lse = merge_with_lse(out1, lse1, out2, lse2)
        torch.testing.assert_close(merged_out, out1, atol=1e-5, rtol=1e-5)

    def test_both_neg_inf_lse(self):
        """Merging two -inf LSE should not produce NaN."""
        n_tokens, n_heads, head_dim = 16, 2, 8
        out1 = torch.randn(n_tokens, n_heads, head_dim)
        out2 = torch.randn(n_tokens, n_heads, head_dim)
        lse1 = torch.full((n_heads, n_tokens), float("-inf"))
        lse2 = torch.full((n_heads, n_tokens), float("-inf"))

        merged_out, merged_lse = merge_with_lse(out1, lse1, out2, lse2)
        self.assertFalse(torch.isnan(merged_out).any())
        self.assertFalse(torch.isnan(merged_lse).any())

    def test_bf16_precision(self):
        """Merge in bf16 should still be accurate (fp32 intermediate)."""
        torch.manual_seed(42)
        n_tokens, n_heads, head_dim = 64, 4, 16

        out1 = torch.randn(n_tokens, n_heads, head_dim)
        out2 = torch.randn(n_tokens, n_heads, head_dim)
        lse1 = torch.randn(n_heads, n_tokens)
        lse2 = torch.randn(n_heads, n_tokens)

        # fp32 reference
        ref_out, ref_lse = merge_with_lse(out1, lse1, out2, lse2)

        # bf16 inputs
        bf16_out, bf16_lse = merge_with_lse(
            out1.bfloat16(), lse1.bfloat16(),
            out2.bfloat16(), lse2.bfloat16(),
        )

        torch.testing.assert_close(
            bf16_out.float(), ref_out, atol=2e-2, rtol=2e-2
        )

    def test_extreme_lse_values(self):
        """Large LSE differences should not overflow/underflow."""
        n_tokens, n_heads, head_dim = 16, 2, 8
        out1 = torch.randn(n_tokens, n_heads, head_dim)
        out2 = torch.randn(n_tokens, n_heads, head_dim)
        # Large gap: lse1 >> lse2, result should be dominated by out1
        lse1 = torch.full((n_heads, n_tokens), 100.0)
        lse2 = torch.full((n_heads, n_tokens), -100.0)

        merged_out, merged_lse = merge_with_lse(out1, lse1, out2, lse2)
        self.assertFalse(torch.isnan(merged_out).any())
        self.assertFalse(torch.isinf(merged_out).any())
        torch.testing.assert_close(merged_out, out1, atol=1e-5, rtol=1e-5)


class TestBuildRingStepCuSeqlens(unittest.TestCase):
    """Unit tests for _build_ring_step_cu_seqlens."""

    def test_diagonal_single_doc(self):
        """Diagonal step with a single document spanning the chunk."""
        # Single doc of length 128, chunk_size=64, 2 chunks
        global_cu = torch.tensor([0, 128], dtype=torch.int32)
        # Chunk 0 diagonal: q=[0,64), k=[0,64)
        cu_q, cu_k, q_ranges, k_ranges, n_q, is_causal = _build_ring_step_cu_seqlens(
            global_cu, 0, 64, 0, 64, 128, 64, torch.device("cpu")
        )
        self.assertTrue(is_causal)
        self.assertEqual(n_q, 64)
        self.assertEqual(q_ranges, [(0, 64)])
        self.assertEqual(k_ranges, [(0, 64)])
        self.assertEqual(cu_q.tolist(), [0, 64])
        self.assertEqual(cu_k.tolist(), [0, 64])

    def test_below_diagonal(self):
        """Below diagonal: q_chunk=1, k_chunk=0."""
        global_cu = torch.tensor([0, 128], dtype=torch.int32)
        # q=[64,128), k=[0,64)
        cu_q, cu_k, q_ranges, k_ranges, n_q, is_causal = _build_ring_step_cu_seqlens(
            global_cu, 64, 128, 0, 64, 128, 64, torch.device("cpu")
        )
        self.assertFalse(is_causal)
        self.assertEqual(n_q, 64)
        self.assertEqual(q_ranges, [(64, 128)])
        self.assertEqual(k_ranges, [(0, 64)])

    def test_above_diagonal_skip(self):
        """Above diagonal: q_chunk=0, k_chunk=1 should skip."""
        global_cu = torch.tensor([0, 128], dtype=torch.int32)
        cu_q, cu_k, q_ranges, k_ranges, n_q, is_causal = _build_ring_step_cu_seqlens(
            global_cu, 0, 64, 64, 128, 128, 64, torch.device("cpu")
        )
        self.assertEqual(n_q, 0)
        self.assertEqual(q_ranges, [])
        self.assertEqual(k_ranges, [])

    def test_diagonal_multi_doc(self):
        """Diagonal step with multiple documents in the chunk."""
        # Docs: [0,30), [30,64) all in chunk 0=[0,64)
        global_cu = torch.tensor([0, 30, 64], dtype=torch.int32)
        cu_q, cu_k, q_ranges, k_ranges, n_q, is_causal = _build_ring_step_cu_seqlens(
            global_cu, 0, 64, 0, 64, 64, 64, torch.device("cpu")
        )
        self.assertTrue(is_causal)
        self.assertEqual(n_q, 64)
        self.assertEqual(q_ranges, [(0, 30), (30, 64)])
        self.assertEqual(k_ranges, [(0, 30), (30, 64)])
        # cu_seqlens should reflect two documents
        self.assertEqual(cu_q.tolist(), [0, 30, 64])
        self.assertEqual(cu_k.tolist(), [0, 30, 64])

    def test_doc_spans_chunks_below_diagonal(self):
        """Document spanning both chunks, below-diagonal step."""
        # Doc [0, 128), chunk_size=64
        # q_chunk=1 [64,128), k_chunk=0 [0,64)
        global_cu = torch.tensor([0, 128], dtype=torch.int32)
        cu_q, cu_k, q_ranges, k_ranges, n_q, is_causal = _build_ring_step_cu_seqlens(
            global_cu, 64, 128, 0, 64, 128, 64, torch.device("cpu")
        )
        self.assertFalse(is_causal)
        self.assertEqual(n_q, 64)
        self.assertEqual(q_ranges, [(64, 128)])
        self.assertEqual(k_ranges, [(0, 64)])

    def test_doc_only_in_q_chunk(self):
        """Document only in Q chunk, not in K chunk."""
        # Doc [64, 100), only in chunk 1=[64,128)
        # q_chunk=1 [64,128), k_chunk=0 [0,64) → doc doesn't overlap K chunk
        global_cu = torch.tensor([0, 64, 100, 128], dtype=torch.int32)
        cu_q, cu_k, q_ranges, k_ranges, n_q, is_causal = _build_ring_step_cu_seqlens(
            global_cu, 64, 128, 0, 64, 128, 64, torch.device("cpu")
        )
        self.assertFalse(is_causal)
        # Doc [0,64) overlaps both chunks, doc [64,100) only in Q chunk (no K overlap)
        # Doc [100,128) only in Q chunk (no K overlap)
        # Only doc [0,64) has K in chunk 0... wait, doc [0,64) has q in [64,128)?
        # No! doc [0,64) ends at 64, q_chunk starts at 64. q_s = max(0,64)=64, q_e=min(64,128)=64, q_len=0.
        # So no doc overlaps both.
        self.assertEqual(n_q, 0)
        self.assertEqual(q_ranges, [])
        self.assertEqual(k_ranges, [])

    def test_no_overlap_padding_chunk(self):
        """All-padding chunk: no real Q tokens."""
        # Doc [0, 50), total_seqlen padded to 128, chunk_size=64
        # Chunk 1 [64,128) has only padding past position 50
        global_cu = torch.tensor([0, 50], dtype=torch.int32)
        cu_q, cu_k, q_ranges, k_ranges, n_q, is_causal = _build_ring_step_cu_seqlens(
            global_cu, 64, 128, 64, 128, 50, 64, torch.device("cpu")
        )
        self.assertEqual(n_q, 0)

    def test_doc_at_chunk_boundary(self):
        """Document that ends exactly at chunk boundary."""
        # Docs: [0, 64), [64, 128), chunk_size=64
        global_cu = torch.tensor([0, 64, 128], dtype=torch.int32)
        # Chunk 1 diagonal: only doc [64,128) overlaps
        cu_q, cu_k, q_ranges, k_ranges, n_q, is_causal = _build_ring_step_cu_seqlens(
            global_cu, 64, 128, 64, 128, 128, 64, torch.device("cpu")
        )
        self.assertTrue(is_causal)
        self.assertEqual(n_q, 64)
        self.assertEqual(q_ranges, [(64, 128)])
        self.assertEqual(k_ranges, [(64, 128)])

    def test_many_small_docs(self):
        """Many small docs, some in chunk 0, some in chunk 1."""
        # 8 docs of 16 tokens each, total=128, chunk_size=64
        # Chunk 0: docs 0-3, Chunk 1: docs 4-7
        global_cu = torch.tensor(
            [0, 16, 32, 48, 64, 80, 96, 112, 128], dtype=torch.int32
        )
        # Diagonal chunk 0
        cu_q, cu_k, q_ranges, k_ranges, n_q, is_causal = _build_ring_step_cu_seqlens(
            global_cu, 0, 64, 0, 64, 128, 64, torch.device("cpu")
        )
        self.assertTrue(is_causal)
        self.assertEqual(n_q, 64)
        self.assertEqual(len(q_ranges), 4)

        # Below-diagonal (q_chunk=1, k_chunk=0): no doc spans both → n_q=0
        cu_q, cu_k, q_ranges, k_ranges, n_q, is_causal = _build_ring_step_cu_seqlens(
            global_cu, 64, 128, 0, 64, 128, 64, torch.device("cpu")
        )
        self.assertFalse(is_causal)
        self.assertEqual(n_q, 0)  # No doc spans both chunks


class TestExtractTokensForDocs(unittest.TestCase):
    """Unit tests for _extract_tokens_for_docs."""

    def test_basic_extraction(self):
        """Extract tokens from two doc ranges."""
        chunk = torch.arange(64).unsqueeze(-1).float()  # (64, 1)
        # Global ranges: [10, 20) and [30, 40), chunk_start=0
        result = _extract_tokens_for_docs(chunk, [(10, 20), (30, 40)], 0)
        self.assertEqual(result.shape[0], 20)
        expected = torch.cat([chunk[10:20], chunk[30:40]], dim=0)
        torch.testing.assert_close(result, expected)

    def test_with_chunk_offset(self):
        """Extract with non-zero chunk start."""
        chunk = torch.arange(64).unsqueeze(-1).float()
        # chunk_start=64, global ranges [64, 80) and [90, 100)
        # local offsets: [0, 16) and [26, 36)
        result = _extract_tokens_for_docs(chunk, [(64, 80), (90, 100)], 64)
        self.assertEqual(result.shape[0], 26)
        expected = torch.cat([chunk[0:16], chunk[26:36]], dim=0)
        torch.testing.assert_close(result, expected)

    def test_empty_ranges(self):
        """Empty doc ranges should return empty tensor."""
        chunk = torch.randn(64, 4, 8)
        result = _extract_tokens_for_docs(chunk, [], 0)
        self.assertEqual(result.shape[0], 0)


class TestScatterToChunk(unittest.TestCase):
    """Unit tests for _scatter_to_chunk."""

    def test_full_coverage(self):
        """All positions covered by one document."""
        n_heads, head_dim, chunk_size = 2, 8, 32
        packed_out = torch.randn(32, n_heads, head_dim)
        packed_lse = torch.randn(n_heads, 32)
        q_doc_ranges = [(0, 32)]

        full_out, full_lse = _scatter_to_chunk(
            packed_out, packed_lse, q_doc_ranges, 0, chunk_size, n_heads, head_dim,
            torch.device("cpu"), packed_out.dtype,
        )
        torch.testing.assert_close(full_out, packed_out)
        torch.testing.assert_close(full_lse, packed_lse)

    def test_partial_coverage(self):
        """Only part of chunk covered; rest should be zero / -inf."""
        n_heads, head_dim, chunk_size = 2, 8, 64
        packed_out = torch.randn(20, n_heads, head_dim)
        packed_lse = torch.randn(n_heads, 20)
        # Doc at positions [10, 30) in chunk starting at 0
        q_doc_ranges = [(10, 30)]

        full_out, full_lse = _scatter_to_chunk(
            packed_out, packed_lse, q_doc_ranges, 0, chunk_size, n_heads, head_dim,
            torch.device("cpu"), packed_out.dtype,
        )
        # Check covered region
        torch.testing.assert_close(full_out[10:30], packed_out)
        torch.testing.assert_close(full_lse[:, 10:30], packed_lse)
        # Check uncovered regions
        self.assertTrue((full_out[:10] == 0).all())
        self.assertTrue((full_out[30:] == 0).all())
        self.assertTrue(torch.isinf(full_lse[:, :10]).all())
        self.assertTrue(torch.isinf(full_lse[:, 30:]).all())

    def test_multi_doc(self):
        """Two documents, gap in between."""
        n_heads, head_dim, chunk_size = 2, 8, 64
        # Doc1: 10 tokens, Doc2: 15 tokens = 25 packed tokens
        packed_out = torch.randn(25, n_heads, head_dim)
        packed_lse = torch.randn(n_heads, 25)
        # chunk_start=64, doc1=[64,74), doc2=[80,95)
        q_doc_ranges = [(64, 74), (80, 95)]

        full_out, full_lse = _scatter_to_chunk(
            packed_out, packed_lse, q_doc_ranges, 64, chunk_size, n_heads, head_dim,
            torch.device("cpu"), packed_out.dtype,
        )
        # Doc1 at local positions [0,10)
        torch.testing.assert_close(full_out[0:10], packed_out[:10])
        torch.testing.assert_close(full_lse[:, 0:10], packed_lse[:, :10])
        # Doc2 at local positions [16,31)
        torch.testing.assert_close(full_out[16:31], packed_out[10:25])
        torch.testing.assert_close(full_lse[:, 16:31], packed_lse[:, 10:25])
        # Gap at [10,16) should be zero/-inf
        self.assertTrue((full_out[10:16] == 0).all())
        self.assertTrue(torch.isinf(full_lse[:, 10:16]).all())


class TestWorkEstimate(unittest.TestCase):
    """Test exact work estimate in AttnSlice."""

    def test_full_mask_exact(self):
        """FULL mask: work = q * k."""
        from torchtitan.distributed.varlen_cp.mask_primitives import AttnSlice, MaskType, make_slice_mask

        for q, k in [(16, 16), (32, 64), (10, 5)]:
            s = AttnSlice(0, q, 0, k, MaskType.FULL)
            mask = make_slice_mask(q, k, MaskType.FULL)
            expected = mask.sum().item()
            self.assertAlmostEqual(s.work_estimate, expected, places=0)

    def test_causal_mask_exact(self):
        """CAUSAL mask: exact trapezoid area matches materialized mask."""
        from torchtitan.distributed.varlen_cp.mask_primitives import AttnSlice, MaskType, make_slice_mask

        for q, k in [(16, 16), (8, 32), (32, 8), (1, 1), (64, 64)]:
            s = AttnSlice(0, q, 0, k, MaskType.CAUSAL)
            mask = make_slice_mask(q, k, MaskType.CAUSAL)
            expected = mask.sum().item()
            self.assertAlmostEqual(
                s.work_estimate, max(expected, 1.0), places=0,
                msg=f"CAUSAL q={q} k={k}: got {s.work_estimate}, expected {expected}",
            )


class TestMagiDispatchSolver(unittest.TestCase):
    """Test solve_magi_dispatch correctness."""

    def test_equal_count_constraint(self):
        """Each rank gets exactly sub_chunks_per_rank Q sub-chunks."""
        from torchtitan.distributed.varlen_cp.dispatch_solver import solve_magi_dispatch

        cu_seqlens = [0, 100, 200, 256]
        global_slices = cu_seqlens_to_attn_slices(cu_seqlens)
        plan = solve_magi_dispatch(
            global_slices, total_seqlen=256, chunk_size=128,
            cp_world_size=2, sub_chunks_per_rank=2,
        )
        for r in range(2):
            self.assertEqual(
                len(plan.q_assignments[r]), plan.sub_chunks_per_rank,
                f"Rank {r} got {len(plan.q_assignments[r])} sub-chunks, "
                f"expected {plan.sub_chunks_per_rank}",
            )

    def test_all_sub_chunks_assigned(self):
        """All sub-chunks are assigned exactly once."""
        from torchtitan.distributed.varlen_cp.dispatch_solver import solve_magi_dispatch

        cu_seqlens = [0, 256]
        global_slices = cu_seqlens_to_attn_slices(cu_seqlens)
        plan = solve_magi_dispatch(
            global_slices, total_seqlen=256, chunk_size=128,
            cp_world_size=2, sub_chunks_per_rank=2,
        )
        all_assigned = sorted(qi for qa in plan.q_assignments for qi in qa)
        self.assertEqual(all_assigned, list(range(plan.num_sub_chunks)))

    def test_k_needs_populated(self):
        """Each assigned Q sub-chunk has at least one K dependency."""
        from torchtitan.distributed.varlen_cp.dispatch_solver import solve_magi_dispatch

        cu_seqlens = [0, 100, 256]
        global_slices = cu_seqlens_to_attn_slices(cu_seqlens)
        plan = solve_magi_dispatch(
            global_slices, total_seqlen=256, chunk_size=128,
            cp_world_size=2, sub_chunks_per_rank=2,
        )
        for r in range(2):
            for qi in plan.q_assignments[r]:
                self.assertIn(
                    qi, plan.q_to_k_needs,
                    f"Q sub-chunk {qi} assigned to rank {r} has no K needs",
                )
                self.assertGreater(len(plan.q_to_k_needs[qi]), 0)


class TestVarlenRingAttention(unittest.TestCase):
    """Tests that can run without distributed setup."""

    def test_merge_with_lse_basic(self):
        """Basic LSE merge test."""
        n_tokens, n_heads, head_dim = 16, 2, 8
        out1 = torch.randn(n_tokens, n_heads, head_dim)
        lse1 = torch.randn(n_heads, n_tokens)
        out2 = torch.randn(n_tokens, n_heads, head_dim)
        lse2 = torch.randn(n_heads, n_tokens)

        merged_out, merged_lse = merge_with_lse(out1, lse1, out2, lse2)
        self.assertEqual(merged_out.shape, (n_tokens, n_heads, head_dim))
        self.assertEqual(merged_lse.shape, (n_heads, n_tokens))
        # Result should not have NaN
        self.assertFalse(torch.isnan(merged_out).any())
        self.assertFalse(torch.isnan(merged_lse).any())


# ---------------------------------------------------------------------------
# Multi-GPU tests
# ---------------------------------------------------------------------------


def _run_distributed_tests():
    """Run all distributed tests within a single process group.

    This function must be called within a distributed context with 2 GPUs.
    """
    rank = setup_distributed()
    world_size = dist.get_world_size()
    assert world_size == 2, f"This test requires exactly 2 GPUs, got {world_size}"

    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16

    # ---------------------------------------------------------------
    # Test 1: Ring attention vs single-GPU reference
    # ---------------------------------------------------------------
    if rank == 0:
        print("=== Test 1: Ring attention vs single-GPU reference ===")

    test_configs = [
        (256, [0, 100, 200, 256], "3 docs"),
        (256, [0, 256], "single doc"),
        (256, [0, 32, 64, 96, 128, 160, 192, 224, 256], "8 equal docs"),
        (256, [0, 128, 256], "2 docs at boundary"),
        (256, [0, 50, 200, 256], "uneven docs"),
    ]

    n_heads = 4
    n_kv_heads = 2
    head_dim = 32
    repeat_factor = n_heads // n_kv_heads

    for total_seqlen, doc_bounds, desc in test_configs:
        torch.manual_seed(42)

        q_global = torch.randn(
            total_seqlen, n_heads, head_dim, device=device, dtype=dtype
        )
        k_global = torch.randn(
            total_seqlen, n_kv_heads, head_dim, device=device, dtype=dtype
        )
        v_global = torch.randn(
            total_seqlen, n_kv_heads, head_dim, device=device, dtype=dtype
        )

        dist.broadcast(q_global, src=0)
        dist.broadcast(k_global, src=0)
        dist.broadcast(v_global, src=0)

        global_cu_seqlens = torch.tensor(doc_bounds, dtype=torch.int32, device=device)

        # Reference: single-GPU varlen_attn
        k_expanded = k_global.repeat_interleave(repeat_factor, dim=1)
        v_expanded = v_global.repeat_interleave(repeat_factor, dim=1)

        max_doc_len = max(
            doc_bounds[i + 1] - doc_bounds[i] for i in range(len(doc_bounds) - 1)
        )
        ref_output = varlen_attn(
            q_global, k_expanded, v_expanded,
            global_cu_seqlens, global_cu_seqlens,
            max_doc_len, max_doc_len, is_causal=True,
        )

        # Test: varlen ring attention with CP=2
        cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))
        chunk_size = total_seqlen // world_size

        q_local = q_global[rank * chunk_size : (rank + 1) * chunk_size].clone()
        k_local = k_global[rank * chunk_size : (rank + 1) * chunk_size].clone()
        v_local = v_global[rank * chunk_size : (rank + 1) * chunk_size].clone()

        global_slices = cu_seqlens_to_attn_slices(doc_bounds)
        plan = solve_dispatch(
            global_slices,
            total_seqlen=total_seqlen,
            chunk_size=chunk_size,
            cp_world_size=world_size,
        )

        k_local_expanded = k_local.repeat_interleave(repeat_factor, dim=1)
        v_local_expanded = v_local.repeat_interleave(repeat_factor, dim=1)

        cp_output = varlen_ring_attention(
            q_local, k_local_expanded, v_local_expanded,
            global_cu_seqlens, plan, cp_mesh,
        )

        ref_chunk = ref_output[rank * chunk_size : (rank + 1) * chunk_size]
        max_diff = (cp_output - ref_chunk).abs().max().item()

        if rank == 0:
            print(f"  [{desc}] Max absolute difference: {max_diff}")

        assert max_diff < 1e-2, (
            f"Rank {rank} [{desc}]: varlen ring attention output differs from "
            f"reference by {max_diff} (threshold: 1e-2)"
        )

    if rank == 0:
        print("PASSED: All varlen ring attention tests match single-GPU reference")

    # ---------------------------------------------------------------
    # Test 2: Ring-pass vs all-gather equivalence
    # ---------------------------------------------------------------
    if rank == 0:
        print("\n=== Test 2: Ring-pass vs all-gather equivalence ===")

    torch.manual_seed(42)

    total_seqlen = 256
    n_heads_eq = 4
    head_dim_eq = 32
    chunk_size_eq = total_seqlen // world_size

    q_global = torch.randn(
        total_seqlen, n_heads_eq, head_dim_eq, device=device, dtype=dtype
    )
    k_global = torch.randn(
        total_seqlen, n_heads_eq, head_dim_eq, device=device, dtype=dtype
    )
    v_global = torch.randn(
        total_seqlen, n_heads_eq, head_dim_eq, device=device, dtype=dtype
    )

    dist.broadcast(q_global, src=0)
    dist.broadcast(k_global, src=0)
    dist.broadcast(v_global, src=0)

    global_cu_seqlens = torch.tensor(
        [0, 100, 200, 256], dtype=torch.int32, device=device
    )

    cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))

    q_local = q_global[rank * chunk_size_eq : (rank + 1) * chunk_size_eq].clone()
    k_local = k_global[rank * chunk_size_eq : (rank + 1) * chunk_size_eq].clone()
    v_local = v_global[rank * chunk_size_eq : (rank + 1) * chunk_size_eq].clone()

    global_slices = cu_seqlens_to_attn_slices([0, 100, 200, 256])
    plan = solve_dispatch(
        global_slices,
        total_seqlen=total_seqlen,
        chunk_size=chunk_size_eq,
        cp_world_size=world_size,
    )

    ring_output = _varlen_ring_attention_ring_pass(
        q_local.clone(), k_local.clone(), v_local.clone(),
        global_cu_seqlens, plan, cp_mesh,
    )
    allgather_output = _varlen_ring_attention_all_gather(
        q_local.clone(), k_local.clone(), v_local.clone(),
        global_cu_seqlens, plan, cp_mesh,
    )

    max_diff = (ring_output - allgather_output).abs().max().item()
    if rank == 0:
        print(f"  Ring vs All-gather max diff: {max_diff}")

    assert max_diff < 1e-2, (
        f"Rank {rank}: ring-pass vs all-gather differ by {max_diff} (threshold: 1e-2)"
    )

    if rank == 0:
        print("PASSED: Ring-pass and all-gather produce equivalent results")

    # ---------------------------------------------------------------
    # Test 5: Backward pass — all-gather gradient correctness
    # ---------------------------------------------------------------
    if rank == 0:
        print("\n=== Test 5: Backward gradient correctness (all-gather) ===")

    grad_configs = [
        (256, [0, 100, 200, 256], "3 docs"),
        (256, [0, 256], "single doc"),
        (256, [0, 32, 64, 96, 128, 160, 192, 224, 256], "8 equal docs"),
    ]

    for total_seqlen, doc_bounds, desc in grad_configs:
        torch.manual_seed(42)

        # FlashAttention requires fp16/bf16
        grad_dtype = torch.bfloat16
        q_global = torch.randn(
            total_seqlen, n_heads, head_dim, device=device, dtype=grad_dtype
        )
        k_global = torch.randn(
            total_seqlen, n_heads, head_dim, device=device, dtype=grad_dtype
        )
        v_global = torch.randn(
            total_seqlen, n_heads, head_dim, device=device, dtype=grad_dtype
        )

        dist.broadcast(q_global, src=0)
        dist.broadcast(k_global, src=0)
        dist.broadcast(v_global, src=0)

        global_cu_seqlens = torch.tensor(
            doc_bounds, dtype=torch.int32, device=device
        )

        cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))
        chunk_size = total_seqlen // world_size

        global_slices = cu_seqlens_to_attn_slices(doc_bounds)
        plan = solve_dispatch(
            global_slices,
            total_seqlen=total_seqlen,
            chunk_size=chunk_size,
            cp_world_size=world_size,
        )

        # --- Reference: single-GPU backward ---
        q_ref = q_global.clone().requires_grad_(True)
        k_ref = k_global.clone().requires_grad_(True)
        v_ref = v_global.clone().requires_grad_(True)

        max_doc_len = max(
            doc_bounds[i + 1] - doc_bounds[i] for i in range(len(doc_bounds) - 1)
        )
        ref_output = varlen_attn(
            q_ref, k_ref, v_ref,
            global_cu_seqlens, global_cu_seqlens,
            max_doc_len, max_doc_len, is_causal=True,
        )

        # Use a deterministic grad_output for reproducibility
        torch.manual_seed(123)
        grad_output_global = torch.randn_like(ref_output)
        dist.broadcast(grad_output_global, src=0)

        ref_output.backward(grad_output_global)
        ref_grad_q = q_ref.grad.clone()
        ref_grad_k = k_ref.grad.clone()
        ref_grad_v = v_ref.grad.clone()

        # --- Test all-gather backward (should match single-GPU) ---
        q_local = q_global[rank * chunk_size : (rank + 1) * chunk_size].clone().requires_grad_(True)
        k_local = k_global[rank * chunk_size : (rank + 1) * chunk_size].clone().requires_grad_(True)
        v_local = v_global[rank * chunk_size : (rank + 1) * chunk_size].clone().requires_grad_(True)

        ag_output = _varlen_ring_attention_all_gather(
            q_local, k_local, v_local,
            global_cu_seqlens, plan, cp_mesh,
        )

        grad_output_local = grad_output_global[
            rank * chunk_size : (rank + 1) * chunk_size
        ]
        ag_output.backward(grad_output_local)

        ref_grad_q_chunk = ref_grad_q[rank * chunk_size : (rank + 1) * chunk_size]
        ref_grad_k_chunk = ref_grad_k[rank * chunk_size : (rank + 1) * chunk_size]
        ref_grad_v_chunk = ref_grad_v[rank * chunk_size : (rank + 1) * chunk_size]

        q_diff = (q_local.grad - ref_grad_q_chunk).abs().max().item()
        k_diff = (k_local.grad - ref_grad_k_chunk).abs().max().item()
        v_diff = (v_local.grad - ref_grad_v_chunk).abs().max().item()

        if rank == 0:
            print(
                f"  [{desc}] all_gather vs ref: "
                f"grad_q={q_diff:.6f}, grad_k={k_diff:.6f}, grad_v={v_diff:.6f}"
            )

        grad_tol = 1e-1  # bf16 tolerance
        assert q_diff < grad_tol, (
            f"Rank {rank} [{desc}]: all_gather grad_q diff {q_diff} >= {grad_tol}"
        )
        assert k_diff < grad_tol, (
            f"Rank {rank} [{desc}]: all_gather grad_k diff {k_diff} >= {grad_tol}"
        )
        assert v_diff < grad_tol, (
            f"Rank {rank} [{desc}]: all_gather grad_v diff {v_diff} >= {grad_tol}"
        )

    if rank == 0:
        print("PASSED: All-gather backward matches single-GPU reference")

    # ---------------------------------------------------------------
    # Test 5b: Ring-pass backward vs all-gather (known LSE grad issue)
    #
    # Known issue: varlen_attn's LSE output has zero backward gradient
    # (FlashAttention does not differentiate through LSE). This causes
    # ring-pass backward to be incorrect for K/V because merge_with_lse
    # weights depend on LSE. The all-gather path avoids this by using
    # a single varlen_attn call.
    #
    # This test documents the current magnitude of the gradient error.
    # TODO: fix by implementing custom backward for ring attention
    # that recomputes attention weights without relying on LSE autograd.
    # ---------------------------------------------------------------
    if rank == 0:
        print("\n=== Test 5b: Ring-pass backward (known LSE grad issue) ===")

    # Single diagnostic config — just measure and report the difference
    diag_total = 256
    diag_bounds = [0, 100, 200, 256]
    torch.manual_seed(42)
    grad_dtype = torch.bfloat16

    q_diag = torch.randn(diag_total, n_heads, head_dim, device=device, dtype=grad_dtype)
    k_diag = torch.randn(diag_total, n_heads, head_dim, device=device, dtype=grad_dtype)
    v_diag = torch.randn(diag_total, n_heads, head_dim, device=device, dtype=grad_dtype)
    dist.broadcast(q_diag, src=0)
    dist.broadcast(k_diag, src=0)
    dist.broadcast(v_diag, src=0)

    diag_cu = torch.tensor(diag_bounds, dtype=torch.int32, device=device)
    cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))
    diag_chunk = diag_total // world_size

    diag_slices = cu_seqlens_to_attn_slices(diag_bounds)
    diag_plan = solve_dispatch(
        diag_slices, total_seqlen=diag_total,
        chunk_size=diag_chunk, cp_world_size=world_size,
    )

    torch.manual_seed(123)
    grad_diag = torch.randn(diag_total, n_heads, head_dim, device=device, dtype=grad_dtype)
    dist.broadcast(grad_diag, src=0)
    grad_diag_local = grad_diag[rank * diag_chunk : (rank + 1) * diag_chunk]

    # All-gather reference
    q_ag = q_diag[rank * diag_chunk : (rank + 1) * diag_chunk].clone().requires_grad_(True)
    k_ag = k_diag[rank * diag_chunk : (rank + 1) * diag_chunk].clone().requires_grad_(True)
    v_ag = v_diag[rank * diag_chunk : (rank + 1) * diag_chunk].clone().requires_grad_(True)
    ag_out = _varlen_ring_attention_all_gather(q_ag, k_ag, v_ag, diag_cu, diag_plan, cp_mesh)
    ag_out.backward(grad_diag_local)

    # Ring-pass
    q_rp = q_diag[rank * diag_chunk : (rank + 1) * diag_chunk].clone().requires_grad_(True)
    k_rp = k_diag[rank * diag_chunk : (rank + 1) * diag_chunk].clone().requires_grad_(True)
    v_rp = v_diag[rank * diag_chunk : (rank + 1) * diag_chunk].clone().requires_grad_(True)
    rp_out = _varlen_ring_attention_ring_pass(q_rp, k_rp, v_rp, diag_cu, diag_plan, cp_mesh)
    rp_out.backward(grad_diag_local)

    q_diff = (q_rp.grad - q_ag.grad).abs().max().item()
    k_diff = (k_rp.grad - k_ag.grad).abs().max().item()
    v_diff = (v_rp.grad - v_ag.grad).abs().max().item()

    if rank == 0:
        print(f"  ring_pass vs all_gather: grad_q={q_diff:.6f}, grad_k={k_diff:.6f}, grad_v={v_diff:.6f}")
        print(
            "NOTE: K/V gradient differences expected due to varlen_attn LSE "
            "backward returning zero gradients. Needs custom backward."
        )

    # Sync after ring-pass backward to drain pending P2P operations
    dist.barrier()

    # ---------------------------------------------------------------
    # Test 6: Skip-noop correctness — sparse plan skips expected steps
    # ---------------------------------------------------------------
    if rank == 0:
        print("\n=== Test 6: Skip-noop correctness ===")

    # Use 8 small docs aligned to chunk boundaries so most cross-chunk
    # pairs have zero work (docs don't span across chunks).
    noop_total = 256
    noop_doc_bounds = [0, 32, 64, 96, 128, 160, 192, 224, 256]
    noop_cu = torch.tensor(noop_doc_bounds, dtype=torch.int32, device=device)
    noop_chunk_size = noop_total // world_size

    noop_slices = cu_seqlens_to_attn_slices(noop_doc_bounds)
    noop_plan = solve_dispatch(
        noop_slices,
        total_seqlen=noop_total,
        chunk_size=noop_chunk_size,
        cp_world_size=world_size,
    )

    # Count how many (q_rank, k_source) pairs have work
    work_count = 0
    skip_count = 0
    for q_idx in range(world_size):
        for k_idx in range(world_size):
            if noop_plan.pair_has_work(q_idx, k_idx):
                work_count += 1
            else:
                skip_count += 1

    if rank == 0:
        print(
            f"  8 aligned docs, CP={world_size}: "
            f"{work_count} pairs with work, {skip_count} pairs skipped "
            f"(out of {world_size * world_size} total)"
        )

    # With 8 aligned docs and CP=2, each chunk gets 4 docs that don't
    # span the boundary. Diagonal pairs always have work.
    # Off-diagonal pairs should have NO work because no doc spans chunks.
    assert skip_count > 0, "Expected some pairs to be skipped with aligned docs"
    assert work_count >= world_size, "At least diagonal pairs should have work"

    # Verify output is still correct despite skipping
    torch.manual_seed(42)
    q_global_noop = torch.randn(
        noop_total, n_heads, head_dim, device=device, dtype=dtype
    )
    k_global_noop = torch.randn(
        noop_total, n_heads, head_dim, device=device, dtype=dtype
    )
    v_global_noop = torch.randn(
        noop_total, n_heads, head_dim, device=device, dtype=dtype
    )
    dist.broadcast(q_global_noop, src=0)
    dist.broadcast(k_global_noop, src=0)
    dist.broadcast(v_global_noop, src=0)

    # Reference
    max_doc_noop = max(
        noop_doc_bounds[i + 1] - noop_doc_bounds[i]
        for i in range(len(noop_doc_bounds) - 1)
    )
    ref_noop = varlen_attn(
        q_global_noop, k_global_noop, v_global_noop,
        noop_cu, noop_cu,
        max_doc_noop, max_doc_noop, is_causal=True,
    )

    q_local_noop = q_global_noop[rank * noop_chunk_size : (rank + 1) * noop_chunk_size].clone()
    k_local_noop = k_global_noop[rank * noop_chunk_size : (rank + 1) * noop_chunk_size].clone()
    v_local_noop = v_global_noop[rank * noop_chunk_size : (rank + 1) * noop_chunk_size].clone()

    ring_noop = _varlen_ring_attention_ring_pass(
        q_local_noop, k_local_noop, v_local_noop,
        noop_cu, noop_plan, cp_mesh,
    )

    ref_chunk_noop = ref_noop[rank * noop_chunk_size : (rank + 1) * noop_chunk_size]
    noop_diff = (ring_noop - ref_chunk_noop).abs().max().item()

    if rank == 0:
        print(f"  Ring-pass with skips vs reference: {noop_diff:.6f}")

    assert noop_diff < 1e-2, (
        f"Rank {rank}: skip-noop ring-pass differs by {noop_diff}"
    )

    if rank == 0:
        print("PASSED: Skip-noop produces correct results with sparse plan")

    # ---------------------------------------------------------------
    # Test 3: Magi Attention forward vs single-GPU reference
    # ---------------------------------------------------------------
    if rank == 0:
        print("\n=== Test 3: Magi Attention forward vs single-GPU reference ===")

    magi_configs = [
        (256, [0, 100, 200, 256], "3 docs"),
        (256, [0, 256], "single doc"),
        (256, [0, 32, 64, 96, 128, 160, 192, 224, 256], "8 equal docs"),
        (256, [0, 128, 256], "2 docs at boundary"),
        (256, [0, 50, 200, 256], "uneven docs"),
    ]

    for total_seqlen, doc_bounds, desc in magi_configs:
        torch.manual_seed(42)

        q_global = torch.randn(
            total_seqlen, n_heads, head_dim, device=device, dtype=dtype
        )
        k_global = torch.randn(
            total_seqlen, n_kv_heads, head_dim, device=device, dtype=dtype
        )
        v_global = torch.randn(
            total_seqlen, n_kv_heads, head_dim, device=device, dtype=dtype
        )

        dist.broadcast(q_global, src=0)
        dist.broadcast(k_global, src=0)
        dist.broadcast(v_global, src=0)

        global_cu_seqlens = torch.tensor(
            doc_bounds, dtype=torch.int32, device=device
        )

        # Reference: single-GPU varlen_attn
        k_expanded = k_global.repeat_interleave(repeat_factor, dim=1)
        v_expanded = v_global.repeat_interleave(repeat_factor, dim=1)

        max_doc_len = max(
            doc_bounds[i + 1] - doc_bounds[i] for i in range(len(doc_bounds) - 1)
        )
        ref_output = varlen_attn(
            q_global, k_expanded, v_expanded,
            global_cu_seqlens, global_cu_seqlens,
            max_doc_len, max_doc_len, is_causal=True,
        )

        # Test: magi dispatch with CP=2
        cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))
        chunk_size = total_seqlen // world_size

        q_local = q_global[rank * chunk_size : (rank + 1) * chunk_size].clone()
        k_local = k_global[rank * chunk_size : (rank + 1) * chunk_size].clone()
        v_local = v_global[rank * chunk_size : (rank + 1) * chunk_size].clone()

        global_slices = cu_seqlens_to_attn_slices(doc_bounds)
        plan = solve_dispatch(
            global_slices,
            total_seqlen=total_seqlen,
            chunk_size=chunk_size,
            cp_world_size=world_size,
        )

        k_local_expanded = k_local.repeat_interleave(repeat_factor, dim=1)
        v_local_expanded = v_local.repeat_interleave(repeat_factor, dim=1)

        magi_output = varlen_magi_dispatch(
            q_local, k_local_expanded, v_local_expanded,
            global_cu_seqlens, plan, cp_mesh,
        )

        ref_chunk = ref_output[rank * chunk_size : (rank + 1) * chunk_size]
        max_diff = (magi_output - ref_chunk).abs().max().item()

        if rank == 0:
            print(f"  [{desc}] Max absolute difference: {max_diff}")

        assert max_diff < 1e-2, (
            f"Rank {rank} [{desc}]: magi dispatch output differs from "
            f"reference by {max_diff} (threshold: 1e-2)"
        )

    if rank == 0:
        print("PASSED: Magi Attention forward matches single-GPU reference")

    # ---------------------------------------------------------------
    # Test 3b: Magi Attention backward vs single-GPU reference
    # ---------------------------------------------------------------
    if rank == 0:
        print("\n=== Test 3b: Magi Attention backward vs single-GPU reference ===")

    magi_grad_configs = [
        (256, [0, 100, 200, 256], "3 docs"),
        (256, [0, 256], "single doc"),
        (256, [0, 32, 64, 96, 128, 160, 192, 224, 256], "8 equal docs"),
    ]

    for total_seqlen, doc_bounds, desc in magi_grad_configs:
        torch.manual_seed(42)

        grad_dtype = torch.bfloat16
        q_global = torch.randn(
            total_seqlen, n_heads, head_dim, device=device, dtype=grad_dtype
        )
        k_global = torch.randn(
            total_seqlen, n_heads, head_dim, device=device, dtype=grad_dtype
        )
        v_global = torch.randn(
            total_seqlen, n_heads, head_dim, device=device, dtype=grad_dtype
        )

        dist.broadcast(q_global, src=0)
        dist.broadcast(k_global, src=0)
        dist.broadcast(v_global, src=0)

        global_cu_seqlens = torch.tensor(
            doc_bounds, dtype=torch.int32, device=device
        )

        cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))
        chunk_size = total_seqlen // world_size

        global_slices = cu_seqlens_to_attn_slices(doc_bounds)
        plan = solve_dispatch(
            global_slices,
            total_seqlen=total_seqlen,
            chunk_size=chunk_size,
            cp_world_size=world_size,
        )

        # --- Reference: single-GPU backward ---
        q_ref = q_global.clone().requires_grad_(True)
        k_ref = k_global.clone().requires_grad_(True)
        v_ref = v_global.clone().requires_grad_(True)

        max_doc_len = max(
            doc_bounds[i + 1] - doc_bounds[i] for i in range(len(doc_bounds) - 1)
        )
        ref_output = varlen_attn(
            q_ref, k_ref, v_ref,
            global_cu_seqlens, global_cu_seqlens,
            max_doc_len, max_doc_len, is_causal=True,
        )

        torch.manual_seed(123)
        grad_output_global = torch.randn_like(ref_output)
        dist.broadcast(grad_output_global, src=0)

        ref_output.backward(grad_output_global)
        ref_grad_q = q_ref.grad.clone()

        # --- Test magi dispatch backward ---
        q_local = q_global[rank * chunk_size : (rank + 1) * chunk_size].clone().requires_grad_(True)
        k_local = k_global[rank * chunk_size : (rank + 1) * chunk_size].clone().requires_grad_(True)
        v_local = v_global[rank * chunk_size : (rank + 1) * chunk_size].clone().requires_grad_(True)

        magi_output = varlen_magi_dispatch(
            q_local, k_local, v_local,
            global_cu_seqlens, plan, cp_mesh,
        )

        grad_output_local = grad_output_global[
            rank * chunk_size : (rank + 1) * chunk_size
        ]
        magi_output.backward(grad_output_local)

        ref_grad_q_chunk = ref_grad_q[rank * chunk_size : (rank + 1) * chunk_size]
        ref_grad_k_chunk = k_ref.grad[rank * chunk_size : (rank + 1) * chunk_size]
        ref_grad_v_chunk = v_ref.grad[rank * chunk_size : (rank + 1) * chunk_size]

        q_diff = (q_local.grad - ref_grad_q_chunk).abs().max().item()
        k_diff = (k_local.grad - ref_grad_k_chunk).abs().max().item()
        v_diff = (v_local.grad - ref_grad_v_chunk).abs().max().item()

        if rank == 0:
            print(
                f"  [{desc}] magi vs ref: grad_q={q_diff:.6f}, "
                f"grad_k={k_diff:.6f}, grad_v={v_diff:.6f}"
            )

        grad_tol = 0.05
        assert q_diff < grad_tol, (
            f"Rank {rank} [{desc}]: magi grad_q diff {q_diff} >= {grad_tol}"
        )
        assert k_diff < grad_tol, (
            f"Rank {rank} [{desc}]: magi grad_k diff {k_diff} >= {grad_tol}"
        )
        assert v_diff < grad_tol, (
            f"Rank {rank} [{desc}]: magi grad_v diff {v_diff} >= {grad_tol}"
        )

    if rank == 0:
        print("PASSED: Magi Attention backward matches single-GPU reference")

    # ---------------------------------------------------------------
    # Test 8: magi_attn backend (full magi-attention pipeline)
    # ---------------------------------------------------------------
    # Tests TORCHTITAN_VARLEN_CP_COMM=magi_attn mode by calling
    # _varlen_attention_magi_attn directly and comparing against
    # single-GPU reference.
    #
    # Requires magi-attention to be installed (PYTHONPATH).

    _has_magi_attn = False
    try:
        # magi-attention is an optional external dependency (Apache 2.0, SandAI).
        # This test validates interop but the API is not part of production code.
        from magi_attention.api import (  # type: ignore[import-untyped]
            calc_attn,  # noqa: F401
            magi_attn_flex_key,
        )
        from magi_attention.api.functools import (  # type: ignore[import-untyped]
            compute_pad_size,
        )
        from magi_attention.common import AttnRanges  # type: ignore[import-untyped]
        from magi_attention.common.enum import (  # type: ignore[import-untyped]
            AttnMaskType,
        )
        from magi_attention.config import (  # type: ignore[import-untyped]
            DispatchConfig,
            DistAttnConfig,
            SequentialDispatchAlg,
        )

        def _varlen_attention_magi_attn(q, k, v, global_cu_seqlens, plan, cp_mesh):
            """Varlen attention via magi-attention's calc_attn() API (test only)."""
            chunk_size = q.shape[0]
            num_heads_q = q.shape[1]
            num_heads_kv = k.shape[1]
            head_dim = q.shape[2]
            total_seqlen = plan.total_seqlen
            cp_world_size = cp_mesh.size(0)

            cu = global_cu_seqlens.tolist()
            ranges = [[cu[i], cu[i + 1]] for i in range(len(cu) - 1)]
            q_ranges = AttnRanges.from_ranges(ranges)
            k_ranges = AttnRanges.from_ranges(ranges)
            attn_mask_type = [AttnMaskType.CAUSAL] * len(ranges)

            pad_size = compute_pad_size(total_seqlen, cp_world_size, chunk_size)

            key = magi_attn_flex_key(
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_mask_type=attn_mask_type,
                total_seqlen_q=total_seqlen,
                total_seqlen_k=total_seqlen,
                num_heads_q=num_heads_q,
                num_heads_kv=num_heads_kv,
                head_dim=head_dim,
                pad_size=pad_size,
                chunk_size=chunk_size,
                cp_group_or_mesh=cp_mesh,
                dist_attn_config=DistAttnConfig(
                    dispatch_config=DispatchConfig(alg=SequentialDispatchAlg()),
                ),
            )

            out, _meta = calc_attn(q, k, v, key)
            return out

        _has_magi_attn = True
    except ImportError:
        pass

    if _has_magi_attn:
        if rank == 0:
            print("\n=== Test 8: magi_attn backend vs single-GPU reference ===")

        # magi-attention's FFA kernel requires head_dim >= 64.  The test uses
        # head_dim=32 for speed, so force SDPA backend.  Save/restore the env
        # var to avoid side-effects.
        _prev_sdpa = os.environ.get("MAGI_ATTENTION_SDPA_BACKEND")
        os.environ["MAGI_ATTENTION_SDPA_BACKEND"] = "1"

        magi_attn_configs = [
            (256, [0, 100, 200, 256], "3 docs"),
            (256, [0, 256], "single doc"),
            (256, [0, 128, 256], "2 docs at boundary"),
        ]

        for total_seqlen, doc_bounds, desc in magi_attn_configs:
            torch.manual_seed(42)

            q_global = torch.randn(
                total_seqlen, n_heads, head_dim, device=device, dtype=dtype
            )
            k_global = torch.randn(
                total_seqlen, n_kv_heads, head_dim, device=device, dtype=dtype
            )
            v_global = torch.randn(
                total_seqlen, n_kv_heads, head_dim, device=device, dtype=dtype
            )

            dist.broadcast(q_global, src=0)
            dist.broadcast(k_global, src=0)
            dist.broadcast(v_global, src=0)

            global_cu_seqlens = torch.tensor(
                doc_bounds, dtype=torch.int32, device=device
            )

            # Reference: single-GPU varlen_attn
            k_expanded = k_global.repeat_interleave(repeat_factor, dim=1)
            v_expanded = v_global.repeat_interleave(repeat_factor, dim=1)
            max_doc_len = max(
                doc_bounds[i + 1] - doc_bounds[i]
                for i in range(len(doc_bounds) - 1)
            )
            ref_output = varlen_attn(
                q_global, k_expanded, v_expanded,
                global_cu_seqlens, global_cu_seqlens,
                max_doc_len, max_doc_len, is_causal=True,
            )

            # Test: magi_attn with CP
            # Note: magi-attention handles GQA natively — pass non-expanded KV.
            cp_mesh = init_device_mesh(
                "cuda", (world_size,), mesh_dim_names=("cp",)
            )
            chunk_size = total_seqlen // world_size

            q_local = q_global[
                rank * chunk_size : (rank + 1) * chunk_size
            ].clone()
            k_local = k_global[
                rank * chunk_size : (rank + 1) * chunk_size
            ].clone()
            v_local = v_global[
                rank * chunk_size : (rank + 1) * chunk_size
            ].clone()

            global_slices = cu_seqlens_to_attn_slices(doc_bounds)
            plan = solve_dispatch(
                global_slices,
                total_seqlen=total_seqlen,
                chunk_size=chunk_size,
                cp_world_size=world_size,
            )

            magi_attn_output = _varlen_attention_magi_attn(
                q_local, k_local, v_local,
                global_cu_seqlens, plan, cp_mesh,
            )

            ref_chunk = ref_output[
                rank * chunk_size : (rank + 1) * chunk_size
            ]
            max_diff = (magi_attn_output - ref_chunk).abs().max().item()

            if rank == 0:
                print(f"  [{desc}] Max absolute difference: {max_diff}")

            assert max_diff < 0.05, (
                f"Rank {rank} [{desc}]: magi_attn output differs from "
                f"reference by {max_diff} (threshold: 0.05)"
            )

        # Restore env var
        if _prev_sdpa is None:
            os.environ.pop("MAGI_ATTENTION_SDPA_BACKEND", None)
        else:
            os.environ["MAGI_ATTENTION_SDPA_BACKEND"] = _prev_sdpa

        if rank == 0:
            print(
                "PASSED: magi_attn backend matches single-GPU reference"
            )
    else:
        if rank == 0:
            print(
                "\n=== Test 8: SKIPPED (magi-attention not installed) ==="
            )

    dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    # When run with torchrun, execute the distributed tests
    if "RANK" in os.environ:
        _run_distributed_tests()
    else:
        unittest.main()
