# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The MoE router after its gate GEMM: softmax, top-8 by masked argmax rounds, each pick's score, normalize, and the
one-hot routing map. Every step reads only the token's own row of 256 scores, yet Inductor emits 3 kernels:
softmax + argmaxes, then the pick gather + map scatter (plus a zeros kernel), with the normalization. Rewriting the
gather and scatter as row ops (a masked row sum per pick, an OR of pick masks) still gives 3 kernels and recomputes
the softmax. FlashInfer's routing kernel does softmax, top-k, normalization and the expert sort in one.

Expected on 1x GB300 (see ../README.md, ask 11): 3 kernels in both forms; graph ~15 us vs ~8 us of kernels at T=34.
"""
import torch
from _common import graph_kernels, graph_us, header, NUM_EXPERTS, TOP_K


def topk_ids_by_argmax(scores_TE, k):
    ids_E = torch.arange(scores_TE.shape[-1], device=scores_TE.device)
    picks = []
    for _ in range(k):
        pick_T1 = scores_TE.argmax(dim=-1, keepdim=True)
        picks.append(pick_T1)
        scores_TE = scores_TE.masked_fill(ids_E == pick_T1, float("-inf"))
    return torch.cat(picks, dim=-1)


def router_gather_scatter(gate_TE):
    """TorchTitan's router region today."""
    scores_TE = gate_TE.softmax(dim=-1)
    ids_TK = topk_ids_by_argmax(scores_TE, TOP_K)
    picks_TK = scores_TE.gather(-1, ids_TK)
    picks_TK = picks_TK / picks_TK.sum(-1, keepdim=True)
    map_TE = torch.zeros_like(scores_TE, dtype=torch.bool).scatter_(-1, ids_TK, True)
    return picks_TK, ids_TK, map_TE


def router_row_ops(gate_TE):
    """The same outputs with the gather and scatter written as row ops."""
    scores_TE = gate_TE.softmax(dim=-1)
    ids_TK = topk_ids_by_argmax(scores_TE, TOP_K)
    ids_E = torch.arange(NUM_EXPERTS, device=gate_TE.device)
    masks = [ids_TK[:, k : k + 1] == ids_E for k in range(TOP_K)]
    picks_TK = torch.cat([torch.where(m, scores_TE, 0).sum(-1, keepdim=True) for m in masks], -1)
    picks_TK = picks_TK / picks_TK.sum(-1, keepdim=True)
    map_TE = masks[0]
    for m in masks[1:]:
        map_TE = map_TE | m
    return picks_TK, ids_TK, map_TE


header()
with torch.no_grad():
    for name, fn in (("gather/scatter", router_gather_scatter), ("row ops", router_row_ops)):
        compiled = torch.compile(fn, fullgraph=True)
        for num_tokens in (16384, 14336):  # largest first, as in vLLM's generator: regions turn dynamic
            compiled(torch.randn(num_tokens, NUM_EXPERTS, device="cuda"))
        print(f"--- {name}")
        for num_tokens in (1, 34, 256, 4096):
            gate_TE = torch.randn(num_tokens, NUM_EXPERTS, device="cuda")
            wall = graph_us(lambda: compiled(gate_TE))
            kernels = graph_kernels(lambda: compiled(gate_TE))
            print(
                f"T={num_tokens:5d}: graph {wall:5.1f} us, {len(kernels)} kernels, kernel sum "
                f"{sum(us for us, _ in kernels):5.1f} us: "
                + "  ".join(f"{us:.1f}" for us, _ in kernels)
            )
        gate_TE = torch.randn(64, NUM_EXPERTS, device="cuda")
        picks_TK, ids_TK, map_TE = compiled(gate_TE)
        eager_picks_TK, eager_ids_TK, eager_map_TE = router_gather_scatter(gate_TE)
        print(
            f"vs eager: ids and map equal {torch.equal(ids_TK, eager_ids_TK) and torch.equal(map_TE, eager_map_TE)}, "
            f"picks max abs diff {(picks_TK - eager_picks_TK).abs().max().item():.1e}"
        )
