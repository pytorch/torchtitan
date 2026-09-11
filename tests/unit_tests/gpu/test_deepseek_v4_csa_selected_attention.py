# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPU numerics test for CSA's Attention Gym ``selected_attention`` core.

``CompressedSparseAttention`` (compress_ratio == 4) was switched from
building a FlexAttention ``BlockMask`` over a concatenated
[sliding-window KV, compressed KV, sink] buffer to calling
``attn_gym.sparse.selected_attention.selected_attention`` directly. This test
verifies the two formulations compute the same attention math by
reimplementing the prior ``BlockMask``-based forward inline (the class itself
no longer has this code path) and comparing outputs and gradients against
the current ``CompressedSparseAttention``.

CPU is not used here because ``flex_attention`` (the prior formulation)
requires CUDA for backward.
"""

import unittest

import torch
from torch.nn.attention.flex_attention import flex_attention

from torchtitan.models.deepseek_v4.attention import (
    CompressedSparseAttention,
    DSV4FlexAttention,
)
from torchtitan.models.deepseek_v4.compressor import Indexer


def _prior_block_mask_forward(
    dsa: DSV4FlexAttention,
    q: torch.Tensor,
    swa_k: torch.Tensor,
    cmp_k: torch.Tensor,
    idx_q: torch.Tensor,
    idx_k: torch.Tensor,
    idx_w: torch.Tensor,
    attn_sink: torch.Tensor,
) -> torch.Tensor:
    """Inline copy of the pre-selected_attention CSA forward (ratio == 4).

    Mirrors what ``DSV4FlexAttention._forward_impl`` did for compress_ratio
    == 4 before ``CompressedSparseAttention`` was switched to call
    ``selected_attention`` directly: concatenate [swa_k, cmp_k, sink] into
    one KV buffer, build a ``BlockMask`` from the top-k indices (offset into
    that concatenated space), and run ``flex_attention`` with a sink
    ``score_mod``.
    """
    seqlen, _, head_dim = q.size()
    n_cmp = cmp_k.size(0)
    sink_idx = seqlen + n_cmp

    kv = swa_k.unsqueeze(1)
    kv = torch.cat([kv, cmp_k.unsqueeze(1)], dim=0)
    sink_kv = kv.new_zeros((1, 1, head_dim))
    kv = torch.cat([kv, sink_kv], dim=0)
    kv = kv.expand(-1, q.size(1), -1)

    win = dsa.get_window_topk_idxs(bsz=1, seqlen=seqlen, device=q.device)
    cmp_topk = Indexer.select(
        idx_q,
        idx_k,
        idx_w,
        seqlen=seqlen,
        ratio=dsa.compress_ratio,
        topk=dsa.index_topk,
    ).unsqueeze(0)
    causal_limit = (
        torch.arange(1, seqlen + 1, device=q.device).unsqueeze(1) // dsa.compress_ratio
    )
    cmp_topk_offset = torch.where(
        cmp_topk < causal_limit.unsqueeze(0), seqlen + cmp_topk, -1
    )
    sink_indices = torch.full(
        (1, seqlen, 1), sink_idx, dtype=torch.int64, device=q.device
    )
    selected_indices = torch.cat([win, cmp_topk_offset, sink_indices], dim=-1)

    block_mask = dsa._build_block_mask(
        1, seqlen, kv.size(0), selected_indices, q.device
    )

    def v4_sink_score_mod(score, b, h, q_idx, kv_idx):
        return torch.where(kv_idx == sink_idx, attn_sink[h], score)

    q_1htk = q.transpose(0, 1).unsqueeze(0)
    kv_1htk = kv.transpose(0, 1).unsqueeze(0)
    out_1htk = flex_attention(
        q_1htk,
        kv_1htk,
        kv_1htk,
        score_mod=v4_sink_score_mod,
        block_mask=block_mask,
        scale=dsa.softmax_scale,
    )
    return out_1htk.squeeze(0).transpose(0, 1)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class TestCSASelectedAttentionMatchesBlockMask(unittest.TestCase):
    def _make_inputs(
        self, *, seqlen, n_heads, head_dim, n_index_heads, index_head_dim, dtype, device
    ):
        torch.manual_seed(0)
        ratio = 4
        n_cmp = seqlen // ratio
        q = torch.randn(
            seqlen, n_heads, head_dim, dtype=dtype, device=device, requires_grad=True
        )
        swa_k = torch.randn(
            seqlen, head_dim, dtype=dtype, device=device, requires_grad=True
        )
        cmp_k = torch.randn(
            n_cmp, head_dim, dtype=dtype, device=device, requires_grad=True
        )
        idx_q = torch.randn(
            seqlen,
            n_index_heads,
            index_head_dim,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        idx_k = torch.randn(
            n_cmp, index_head_dim, dtype=dtype, device=device, requires_grad=True
        )
        idx_w = torch.randn(
            seqlen, n_index_heads, dtype=dtype, device=device, requires_grad=True
        )
        attn_sink = torch.randn(n_heads, dtype=dtype, device=device, requires_grad=True)
        return q, swa_k, cmp_k, idx_q, idx_k, idx_w, attn_sink

    def test_forward_and_grad_match_fp32(self):
        """fp32 comparison isolates the math from bf16 rounding.

        fp64 is not used here because CompressedSparseAttention dispatches to
        the Triton backend on CUDA, which only supports fp16/bf16/fp32 (its
        Triton kernels infer types from the input dtype and do not compile
        for fp64). The prior formulation's top-k selection (Indexer.select)
        is computed once and reused (detached, as it would be in either
        formulation -- topk indices are not differentiable) by both the prior
        BlockMask path and the current CompressedSparseAttention forward.
        """
        device = "cuda"
        seqlen, n_heads, head_dim = 32, 4, 8
        n_index_heads, index_head_dim = 3, 6
        window_size, index_topk = 6, 5
        ratio = 4
        softmax_scale = head_dim**-0.5

        inputs = self._make_inputs(
            seqlen=seqlen,
            n_heads=n_heads,
            head_dim=head_dim,
            n_index_heads=n_index_heads,
            index_head_dim=index_head_dim,
            dtype=torch.float32,
            device=device,
        )
        q, swa_k, cmp_k, idx_q, idx_k, idx_w, attn_sink = inputs

        with torch.no_grad():
            cmp_topk = Indexer.select(
                idx_q, idx_k, idx_w, seqlen=seqlen, ratio=ratio, topk=index_topk
            )

        cfg = CompressedSparseAttention.Config(
            block_size=8,
            window_size=window_size,
            compress_ratio=ratio,
            softmax_scale=softmax_scale,
            index_topk=index_topk,
        )
        dsa = DSV4FlexAttention(cfg)
        csa = CompressedSparseAttention(cfg)

        # Prior formulation: patch Indexer.select to return the precomputed
        # topk so both formulations select identical positions.
        real_select = Indexer.select
        Indexer.select = staticmethod(lambda *a, **k: cmp_topk)
        try:
            out_old = _prior_block_mask_forward(
                dsa, q, swa_k, cmp_k, idx_q, idx_k, idx_w, attn_sink
            )
            out_old.sum().backward()

            q2, swa_k2, cmp_k2, attn_sink2 = (
                t.detach().clone().requires_grad_(True)
                for t in (q, swa_k, cmp_k, attn_sink)
            )
            out_new = csa(q2, swa_k2, cmp_k2, idx_q, idx_k, idx_w, attn_sink2)
            out_new.sum().backward()
        finally:
            Indexer.select = real_select

        torch.testing.assert_close(out_old, out_new, atol=1e-3, rtol=1e-2)
        for name, a, b in (
            ("q", q, q2),
            ("swa_k", swa_k, swa_k2),
            ("cmp_k", cmp_k, cmp_k2),
            ("attn_sink", attn_sink, attn_sink2),
        ):
            torch.testing.assert_close(
                a.grad, b.grad, atol=1e-3, rtol=1e-2, msg=f"grad[{name}] mismatch"
            )

    def test_rejects_softmax_scale_mismatch(self):
        cfg = CompressedSparseAttention.Config(
            block_size=8,
            window_size=4,
            compress_ratio=4,
            softmax_scale=0.1234,  # deliberately wrong vs. head_dim**-0.5
            index_topk=2,
        )
        csa = CompressedSparseAttention(cfg)
        device = "cuda"
        seqlen, n_heads, head_dim = 8, 2, 8
        q = torch.randn(seqlen, n_heads, head_dim, device=device)
        swa_k = torch.randn(seqlen, head_dim, device=device)
        cmp_k = torch.randn(seqlen // 4, head_dim, device=device)
        idx_q = torch.randn(seqlen, 2, 4, device=device)
        idx_k = torch.randn(seqlen // 4, 4, device=device)
        idx_w = torch.randn(seqlen, 2, device=device)
        attn_sink = torch.randn(n_heads, device=device)
        with self.assertRaises(ValueError):
            csa(q, swa_k, cmp_k, idx_q, idx_k, idx_w, attn_sink)


if __name__ == "__main__":
    unittest.main()
