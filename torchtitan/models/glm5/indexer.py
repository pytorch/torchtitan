# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GLM local-query selection, independent of the attention kernel.

Q/K/N/H/S denote local queries, global keys, heads, head width and selected keys.
Bounds contain absolute half-open intervals in the supplied packed K pool.
Outputs are absolute pool IDs with -1 padding, not document-relative indices.
"""

import math

import torch


torch.library.define(
    "torchtitan::glm5_local_topk",
    "(Tensor q, Tensor k, Tensor weights, Tensor bounds, int topk) -> Tensor",
)


def _local_topk_cuda(q, k, weights, bounds, topk):
    from ._indexer_triton import launch

    return launch(q, k, weights, bounds, topk)


torch.library.impl("torchtitan::glm5_local_topk", "CUDA", _local_topk_cuda)


@torch.library.register_fake("torchtitan::glm5_local_topk")
def _local_topk_fake(q, k, weights, bounds, topk):
    return q.new_empty((q.shape[0], topk), dtype=torch.int32)


_local_topk_op = torch.ops.torchtitan.glm5_local_topk.default


@torch.no_grad()
def select_local_topk(q_QNH, k_KH, weights_QN, bounds_Q2, topk: int):
    """Select unique keys within each local query's explicit candidate interval.

    The caller guarantees 0 <= begin <= end <= K, including empty intervals for
    invalid queries. No tensor values are read on the host. CUDA FP16/BF16 uses
    a fused streaming selector; other floating inputs use bounded PyTorch tiles.
    Selection is nondifferentiable. NaN/Inf scores are outside the contract.
    Output order and the choice among cutoff ties are not a cross-backend API.
    """
    Q, N, H = q_QNH.shape
    if k_KH.shape[1] != H or weights_QN.shape != (Q, N):
        raise ValueError("GLM selector Q/K/weights shapes do not agree")
    if k_KH.dtype != q_QNH.dtype or not q_QNH.is_floating_point():
        raise ValueError("GLM selector Q/K must have the same floating dtype")
    if (
        bounds_Q2.shape != (Q, 2)
        or bounds_Q2.dtype != torch.int32
        or not bounds_Q2.is_contiguous()
    ):
        raise ValueError("GLM selector bounds must be contiguous INT32 [Q, 2]")
    if any(t.device != q_QNH.device for t in (k_KH, weights_QN, bounds_Q2)):
        raise ValueError("GLM selector inputs must be on the same device")
    if N <= 0 or H <= 0:
        raise ValueError("GLM selector requires positive head count and width")
    if topk < 0 or k_KH.shape[0] >= 2**31:
        raise ValueError("GLM selector requires nonnegative topk and INT32 key IDs")
    if q_QNH.is_cuda and q_QNH.dtype in (torch.float16, torch.bfloat16):
        return _local_topk_op(q_QNH, k_KH, weights_QN, bounds_Q2, topk)

    # Bound reference intermediates by 16 queries and 1024 keys, not Q * K.
    dtype = torch.float64 if q_QNH.dtype == torch.float64 else torch.float32
    result_QS = torch.full((Q, topk), -1, dtype=torch.int32, device=q_QNH.device)
    if topk == 0:
        return result_QS
    scale = 1.0 / math.sqrt(N * H)
    for start in range(0, Q, 16):
        stop = min(start + 16, Q)
        q_CNH = q_QNH[start:stop].to(dtype)
        weights_CN = weights_QN[start:stop].to(dtype)
        best_CS = q_CNH.new_empty((stop - start, 0))
        ids_CS = result_QS.new_empty((stop - start, 0))
        bounds_C2 = bounds_Q2[start:stop]
        for key_start in range(0, k_KH.shape[0], 1024):
            key_stop = min(key_start + 1024, k_KH.shape[0])
            dots_CNB = torch.einsum(
                "cnh,bh->cnb", q_CNH, k_KH[key_start:key_stop].to(dtype)
            )
            scores_CB = (dots_CNB.relu() * weights_CN[:, :, None]).sum(1) * scale
            keys_B = torch.arange(
                key_start, key_stop, dtype=torch.int32, device=q_QNH.device
            )
            valid_CB = (keys_B >= bounds_C2[:, :1]) & (keys_B < bounds_C2[:, 1:])
            scores_CB = scores_CB.masked_fill(~valid_CB, -torch.inf)
            candidates_CS = torch.cat((best_CS, scores_CB), dim=-1)
            ids_CS = torch.cat((ids_CS, keys_B.expand(stop - start, -1)), dim=-1)
            best_CS, order_CS = candidates_CS.topk(
                min(topk, candidates_CS.shape[1]), dim=-1, sorted=False
            )
            ids_CS = ids_CS.gather(1, order_CS)
        result_QS[start:stop, : ids_CS.shape[1]] = ids_CS.masked_fill(
            ~torch.isfinite(best_CS), -1
        )
    return result_QS
