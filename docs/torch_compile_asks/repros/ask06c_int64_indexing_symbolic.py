# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""With a symbolic dim, Inductor keeps int64 index math whenever the index expression could
overflow int32 under the symbol's (unbounded) range; torch._check(N <= 8) does not narrow it.
assume_32bit_indexing=True speeds up the residual's backward reduction.

Kimi K3 residual weighted sum, stack [16384, N=7 (dynamic), 7168] + partial, fwd+bwd.

Expected on 1x GB300 (torch 2.15.0.dev20260926+cu130, triton 3.8.0; see ../README.md, ask 6(c)):
default ~1918 us, torch._check ~1914 us, assume_32bit_indexing ~1816 us.
"""
import torch
from _common import kernel_us

T, N, D = 16384, 7, 7168


def weighted_sum(scores, stack, partial):
    probs = torch.softmax(scores, dim=-1)
    out = (probs[:, :-1, None] * stack.float()).sum(dim=1) + probs[
        :, -1:
    ] * partial.float()
    return out.to(stack.dtype)


def weighted_sum_checked(scores, stack, partial):
    torch._check(stack.size(1) <= 8)
    return weighted_sum(scores, stack, partial)


for name, fn, opts in (
    ("default", weighted_sum, None),
    ("torch._check(N <= 8)", weighted_sum_checked, None),
    ("assume_32bit_indexing=True", weighted_sum, {"assume_32bit_indexing": True}),
):
    torch._dynamo.reset()
    torch.manual_seed(0)
    stack = torch.randn(
        T, N, D, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    partial = torch.randn(T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    scores = torch.randn(T, N + 1, device="cuda", requires_grad=True)
    torch._dynamo.mark_dynamic(stack, 1)
    torch._dynamo.mark_dynamic(scores, 1)
    f = (
        torch.compile(fn, fullgraph=True, options=opts)
        if opts
        else torch.compile(fn, fullgraph=True)
    )
    grad = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)

    def step(f=f):
        stack.grad = partial.grad = scores.grad = None
        f(scores, stack, partial).backward(grad)

    us, n, _ = kernel_us(step)
    print(f"{name:28s} fwd+bwd {us:7.1f} us  {n} kernels")
