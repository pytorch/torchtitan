# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A Sinkhorn loop on [T, 4, 4] compiles to one kernel per step because each column normalization
reads the previous step at transposed indices, so Inductor realizes every intermediate. The same
math on 16 per-entry [T] tensors (every read same-index) fuses into a handful of kernels.

DeepSeek-V4 mHC: 20 iterations of row then column normalization, T=16384, fp32, fwd+bwd.

Expected on 1x GB300 (torch 2.15.0.dev20260926+cu130, triton 3.8.0; see ../README.md, ask 9):
[T,4,4] form ~289-316 us / 159 kernels, per-entry form ~86 us / 32 kernels, both rel vs fp64 8.8e-8.
"""
import torch
from _common import kernel_us

T, H, ITERS, EPS = 16384, 4, 20, 1e-6


def sinkhorn_tensor(comb):
    comb = comb / (comb.sum(-1, keepdim=True) + EPS)
    comb = comb / (comb.sum(-2, keepdim=True) + EPS)
    for _ in range(ITERS - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + EPS)
    return comb


def sinkhorn_entries(comb):
    c = [[comb[:, i, j] for j in range(H)] for i in range(H)]
    for _ in range(ITERS):
        for i in range(H):
            r = sum(c[i]) + EPS
            c[i] = [v / r for v in c[i]]
        for j in range(H):
            s = sum(c[i][j] for i in range(H)) + EPS
            for i in range(H):
                c[i][j] = c[i][j] / s
    return torch.stack([torch.stack(row, -1) for row in c], -2)


torch.manual_seed(0)
comb = torch.rand(T, H, H, device="cuda").requires_grad_()
grad = torch.randn(T, H, H, device="cuda")
ref = sinkhorn_tensor(comb.double())
for name, fn in (
    ("[T,4,4] tensor form", sinkhorn_tensor),
    ("16 per-entry [T] tensors", sinkhorn_entries),
):
    f = torch.compile(fn, fullgraph=True)
    out = f(comb)
    err = ((out.double() - ref).norm() / ref.norm()).item()

    def step(f=f):
        comb.grad = None
        f(comb).backward(grad)

    us, n, _ = kernel_us(step)
    print(f"{name:26s} fwd+bwd {us:7.1f} us  {n:3d} kernels  rel vs fp64 {err:.1e}")
