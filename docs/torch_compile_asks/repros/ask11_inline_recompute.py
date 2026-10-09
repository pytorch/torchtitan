# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inductor inlines a cheap-looking producer (softmax + zero pad of [T, N+1] scores) into a big
consumer (an 8-way unrolled weighted sum over [T, 8, D]), so every output element recomputes 8
exp/div and masked loads. When the padded probs are needed for backward they are realized and the
kernel is fast; in a no_grad forward they are not. realize_reads_threshold=1 shows the cause.

Kimi K3 residual, stack width N=7 (dynamic, read with clamped indices up to 8), D=7168, T=16384.

Expected on 1x GB300 (torch 2.15.0.dev20260926+cu130, triton 3.8.0; see ../README.md, ask 11):
default ~1225 us, realize_reads_threshold=1 ~278 us.
"""
import torch
import torch.nn.functional as F
from _common import kernel_us

T, N, D, MAX_N = 16384, 7, 7168, 8


def weighted_sum(scores, stack):
    n = stack.size(1)
    probs = F.pad(torch.softmax(scores, dim=-1), (0, MAX_N - n))
    out = probs[:, 0:1] * stack[:, 0].float()
    for i in range(1, MAX_N):
        out = (
            out + probs[:, i : i + 1] * stack.select(1, torch.sym_min(i, n - 1)).float()
        )
    return out.to(stack.dtype)


for name, opts in (
    ("default", None),
    ("realize_reads_threshold=1", {"realize_reads_threshold": 1}),
):
    torch._dynamo.reset()
    torch.manual_seed(0)
    stack = torch.randn(T, N, D, device="cuda", dtype=torch.bfloat16)
    scores = torch.randn(T, N, device="cuda")
    torch._dynamo.mark_dynamic(stack, 1)
    torch._dynamo.mark_dynamic(scores, 1)
    f = (
        torch.compile(weighted_sum, fullgraph=True, options=opts)
        if opts
        else torch.compile(weighted_sum, fullgraph=True)
    )
    with torch.no_grad():
        us, n, top = kernel_us(lambda: f(scores, stack))
    print(f"{name:28s} no_grad fwd {us:7.1f} us  {n} kernels  top {top[0]}")
print(
    f"ideal (read stack, write out, bf16): {(2 * T * N * D + 2 * T * D) / 7.1e12 * 1e6:.0f} us"
)
