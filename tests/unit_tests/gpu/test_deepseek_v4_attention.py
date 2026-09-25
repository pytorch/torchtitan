# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checks the DeepSeek-V4 attention cores are wired to ``gather_attn``
correctly (layouts, window, compressed positions, sink, scale) against dense
masked attention, on CPU (reference impl) and CUDA (fused impl). Kernel
numerics are tested in Attention Gym.
"""

import pytest
import torch

from torchtitan.models.deepseek_v4.attention import (
    CompressedSparseAttention,
    HeavilyCompressedAttention,
    SlidingWindowAttention,
)
from torchtitan.models.deepseek_v4.compressor import Indexer

SEQLEN, N_HEADS, HEAD_DIM, WINDOW, SCALE, TOPK = 32, 4, 8, 6, 0.1234, 5


def dense_attention(q, swa_k, cmp_k, cmp_allowed, attn_sink):
    """Softmax over [swa_k; cmp_k; sink] with a causal window over swa_k and
    ``cmp_allowed`` [T, S] over cmp_k."""
    t = torch.arange(SEQLEN).unsqueeze(1)
    window = (torch.arange(SEQLEN) <= t) & (t - torch.arange(SEQLEN) < WINDOW)
    sink = torch.ones(SEQLEN, 1, dtype=torch.bool)
    allowed = torch.cat([window, cmp_allowed, sink], dim=-1)

    kv = torch.cat([swa_k, cmp_k, swa_k.new_zeros(1, HEAD_DIM)])
    scores_HTS = torch.einsum("thd,sd->hts", q, kv) * SCALE
    scores_HTS[:, :, -1] = attn_sink.unsqueeze(-1)
    scores_HTS = scores_HTS.masked_fill(~allowed, float("-inf"))
    return torch.einsum("hts,sd->thd", scores_HTS.softmax(dim=-1), kv)


def causal_cmp_allowed(ratio):
    """Compressed block s summarizes tokens [s * ratio, (s + 1) * ratio)."""
    t = torch.arange(SEQLEN).unsqueeze(1)
    block_end = (torch.arange(SEQLEN // ratio) + 1) * ratio - 1
    return block_end <= t


# HCA runs at ratio 8 instead of 128 so SEQLEN 32 has compressed blocks; the
# all-causal-blocks pattern is the same.
@pytest.mark.parametrize(
    "cls, ratio",
    [
        (SlidingWindowAttention, 1),
        (HeavilyCompressedAttention, 8),
        (CompressedSparseAttention, 4),
    ],
    ids=["swa", "hca", "csa"],
)
@pytest.mark.parametrize(
    "device, dtype, tol",
    [
        pytest.param("cpu", torch.float64, {}, id="cpu"),
        pytest.param(
            "cuda",
            torch.float32,
            {"atol": 1e-3, "rtol": 1e-2},
            id="cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
def test_matches_dense_masked_attention(cls, ratio, device, dtype, tol):
    torch.manual_seed(0)
    module = cls(
        cls.Config(
            window_size=WINDOW,
            compress_ratio=ratio,
            softmax_scale=SCALE,
            index_topk=TOPK,
        )
    )
    n_cmp = 0 if ratio == 1 else SEQLEN // ratio
    # fp64 CPU copies feed the oracle; the module runs on `device` in `dtype`.
    ref_inputs = [
        torch.randn(*shape, dtype=torch.float64, requires_grad=True)
        for shape in (
            (SEQLEN, N_HEADS, HEAD_DIM),
            (SEQLEN, HEAD_DIM),
            (n_cmp, HEAD_DIM),
            (N_HEADS,),
        )
    ]
    if ratio == 1:
        ref_inputs[2].requires_grad_(False)
    inputs = [
        t.detach().to(device, dtype).requires_grad_(t.requires_grad) for t in ref_inputs
    ]
    q, swa_k, cmp_k, attn_sink = inputs

    if cls is CompressedSparseAttention:
        idx_q, idx_k, idx_w = (
            torch.randn(*shape) for shape in ((SEQLEN, 3, 6), (n_cmp, 6), (SEQLEN, 3))
        )
        out = module(
            q, swa_k, cmp_k, *(t.to(device) for t in (idx_q, idx_k, idx_w)), attn_sink
        )
        cmp_topk = Indexer.select(
            idx_q, idx_k, idx_w, seqlen=SEQLEN, ratio=ratio, topk=TOPK
        ).long()
        cmp_allowed = torch.zeros(SEQLEN, n_cmp, dtype=torch.bool)
        rows = torch.arange(SEQLEN).unsqueeze(1).expand_as(cmp_topk)
        cmp_allowed[rows[cmp_topk >= 0], cmp_topk[cmp_topk >= 0]] = True
        cmp_allowed &= causal_cmp_allowed(ratio)
    elif ratio == 1:  # SWA
        out = module(q, swa_k, attn_sink)
        cmp_allowed = torch.zeros(SEQLEN, 0, dtype=torch.bool)
    else:  # HCA
        out = module(q, swa_k, cmp_k, attn_sink)
        cmp_allowed = causal_cmp_allowed(ratio)

    ref = dense_attention(*ref_inputs[:3], cmp_allowed, ref_inputs[3])
    diff_inputs = [t for t in inputs if t.requires_grad]
    diff_ref_inputs = [t for t in ref_inputs if t.requires_grad]
    grads = torch.autograd.grad(out.sum(), diff_inputs)
    ref_grads = torch.autograd.grad(ref.sum(), diff_ref_inputs)

    torch.testing.assert_close(out.cpu().double(), ref, **tol)
    for grad, ref_grad in zip(grads, ref_grads, strict=True):
        torch.testing.assert_close(grad.cpu().double(), ref_grad, **tol)
