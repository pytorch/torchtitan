# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tokens per expert: the column sum of the (T, E) routing map. Compiled dynamic (large sizes first, as in vLLM's
generator), Inductor picks a split reduction from the size hint: 2 kernels at every T, though at decode one kernel is
3x faster. With split reductions off it is one kernel, but 6x slower at T=16384. The split choice is made once, at
the hint, and never at runtime.

Expected on 1x GB300 (see ../README.md, ask 12): T=34 default 2 kernels ~4.6 us, no split 1 kernel ~1.4 us;
T=16384 default ~5.8 us, no split ~34 us.
"""
import torch
from _common import graph_kernels, graph_us, header, NUM_EXPERTS, TOP_K


def tokens_per_expert(routing_map_TE):
    return routing_map_TE.sum(dim=0)


def routing_map(num_tokens):
    ids_TK = torch.rand(num_tokens, NUM_EXPERTS, device="cuda").topk(TOP_K, dim=-1).indices
    return torch.zeros(num_tokens, NUM_EXPERTS, device="cuda", dtype=torch.bool).scatter_(1, ids_TK, True)


header()
with torch.no_grad():
    for name, options in (("default", None), ("split_reductions=False", {"split_reductions": False})):
        compiled = torch.compile(tokens_per_expert, fullgraph=True, options=options)
        for num_tokens in (16384, 14336):
            compiled(routing_map(num_tokens))
        print(f"--- {name}")
        for num_tokens in (1, 34, 256, 4096, 16384):
            map_TE = routing_map(num_tokens)
            wall = graph_us(lambda: compiled(map_TE))
            kernels = graph_kernels(lambda: compiled(map_TE))
            print(
                f"T={num_tokens:5d}: graph {wall:5.1f} us, {len(kernels)} kernels, kernel sum "
                f"{sum(us for us, _ in kernels):5.1f} us: " + "  ".join(f"{us:.1f}" for us, _ in kernels)
            )
