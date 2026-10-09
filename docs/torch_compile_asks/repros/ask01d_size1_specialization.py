# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Size-1 dims always specialize: a region whose stack width varies 1..8 keeps a separate N=1
graph even after N goes dynamic, and maybe_mark_dynamic does not change that. mark_unbacked is the
existing escape hatch (it also disables other specializations, so it is not free).

Expected on 1x GB300 (torch 2.15.0.dev20260926+cu130, triton 3.8.0; see ../README.md, ask 1):
default 2 graphs; maybe_mark_dynamic 2 graphs; mark_unbacked 1 graph.
"""
import torch
from _common import num_graphs


def weighted(stack, probs):
    return (probs[:, :, None] * stack).sum(1)


f = torch.compile(weighted, fullgraph=True)
for n in (1, 2, 3, 5, 8, 1):
    f(torch.randn(64, n, 32, device="cuda"), torch.rand(64, n, device="cuda"))
print("default, widths 1,2,3,5,8,1:", num_graphs(weighted), "graphs")

torch._dynamo.reset()
f = torch.compile(weighted, fullgraph=True)
s = torch.randn(64, 1, 32, device="cuda")
torch._dynamo.maybe_mark_dynamic(s, 1)
f(s, torch.rand(64, 1, device="cuda"))
f(torch.randn(64, 4, 32, device="cuda"), torch.rand(64, 4, device="cuda"))
print(
    "maybe_mark_dynamic on the size-1 dim, widths 1 then 4:",
    num_graphs(weighted),
    "graphs",
)

torch._dynamo.reset()
f = torch.compile(weighted, fullgraph=True)
try:
    for n in (1, 4):
        s, p = torch.randn(64, n, 32, device="cuda"), torch.rand(64, n, device="cuda")
        torch._dynamo.decorators.mark_unbacked(s, 1)
        torch._dynamo.decorators.mark_unbacked(p, 1)
        f(s, p)
    print(
        "mark_unbacked on the width, widths 1 then 4:", num_graphs(weighted), "graphs"
    )
except Exception as e:
    print("mark_unbacked:", type(e).__name__, str(e).splitlines()[0][:100])
