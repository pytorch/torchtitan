# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A reduction over a small static dim K that is not innermost (stride D) is lowered as a persistent
reduction running far below bandwidth; written as an unrolled Python loop (or with
unroll_reductions_threshold > K, strict <) it becomes a pointwise kernel at roofline.

Case: MoE dispatch-gather backward, grad rows [T*K, D] gathered per token and summed over K.
Kimi K3: T=16384, K=16, D=3584. Also the same sum over a contiguous [T, K, D] (no gather).

Expected on 1x GB300 (torch 2.15.0.dev20260926+cu130, triton 3.8.0; see ../README.md, ask 6):
gather .sum ~1354 us (threshold 8 or 16), ~290 us (threshold 17) = unrolled loop ~290 us; contiguous 999 -> 278 us.
"""
import torch
from _common import kernel_us

T, K, D = 16384, 16, 3584
g = torch.Generator("cuda").manual_seed(0)
grad_rows = torch.randn(T * K, D, device="cuda", generator=g).bfloat16()
order_TK = torch.randperm(T * K, device="cuda", generator=g).view(T, K)  # noqa: N816
contig = torch.randn(T, K, D, device="cuda", generator=g).bfloat16()


def gather_sum(grad_rows, order_TK):
    return (
        grad_rows[order_TK.flatten()]
        .unflatten(0, (T, K))
        .float()
        .sum(1)
        .to(grad_rows.dtype)
    )


def gather_unrolled(grad_rows, order_TK):
    acc = grad_rows[order_TK[:, 0]].float()
    for k in range(1, K):
        acc = acc + grad_rows[order_TK[:, k]].float()
    return acc.to(grad_rows.dtype)


def contig_sum(x):
    return x.float().sum(1).to(x.dtype)


cases = (
    (
        "gather, .sum over K (default threshold 8)",
        gather_sum,
        (grad_rows, order_TK),
        None,
    ),
    (
        "gather, .sum over K, threshold 16",
        gather_sum,
        (grad_rows, order_TK),
        {"unroll_reductions_threshold": 16},
    ),
    (
        "gather, .sum over K, threshold 17",
        gather_sum,
        (grad_rows, order_TK),
        {"unroll_reductions_threshold": 17},
    ),
    ("gather, unrolled Python loop", gather_unrolled, (grad_rows, order_TK), None),
    ("contiguous [T,K,D], .sum over K", contig_sum, (contig,), None),
    (
        "contiguous [T,K,D], threshold 17",
        contig_sum,
        (contig,),
        {"unroll_reductions_threshold": 17},
    ),
)
for name, fn, args, opts in cases:
    torch._dynamo.reset()
    f = (
        torch.compile(fn, fullgraph=True, options=opts)
        if opts
        else torch.compile(fn, fullgraph=True)
    )
    us, n, top = kernel_us(lambda: f(*args))
    print(f"{name:44s} {us:7.1f} us  {n} kernels  {top[0][0][:40]}")
print(
    f"ideal (read T*K*D + write T*D, bf16): {(2 * T * K * D + 2 * T * D) / 7.1e12 * 1e6:.0f} us"
)
