# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inductor's default pointwise-cat lowering (one masked kernel with per-element index math) is
~2-3x slower than per-input copies (ConcatKernel) for some shapes on GB300 and faster for others,
so the right choice is per region (we scope `max_pointwise_cat_inputs=0` to one region).

(a) inverse partial RoPE on the DeepSeek-V4 attention output: split o [T, 64, 512] into 448 + 64,
    rotate the 64, cat (contiguous input; in the model o is a transposed view, which only adds an
    eager gradient layout copy outside the region).
(b) per-head RMS norm + partial RoPE + cat on q (DeepSeek-V4 q_norm_rope): the opposite; with the
    default lowering the cat is fused into the norm's persistent reduction, ConcatKernel splits them.

Expected on 1x GB300 (torch 2.15.0.dev20260926+cu130, triton 3.8.0; see ../README.md, ask 7):
(a) default ~2155 us vs ConcatKernel ~781 us; (b) default ~1026 us vs ConcatKernel ~1502 us.
"""
import torch
from _common import fwd_bwd, kernel_us

T, H, D, R = 16384, 64, 512, 64
torch.manual_seed(0)
angles = torch.randn(T, 1, R // 2, device="cuda")
cos, sin = angles.cos(), angles.sin()
OPTS = {"max_pointwise_cat_inputs": 0, "max_complex_pointwise_cat_inputs": 0}


def rotate(x, inverse=False):
    s = -sin if inverse else sin
    p = x.float().unflatten(-1, (-1, 2))
    a, b = p[..., 0], p[..., 1]
    return torch.stack((a * cos - b * s, a * s + b * cos), -1).flatten(-2).type_as(x)


def inverse_rope(o):
    nope, rope = torch.split(o, [D - R, R], dim=-1)
    return torch.cat([nope, rotate(rope, inverse=True)], dim=-1)


def q_norm_rope(q):
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6)
    nope, rope = torch.split(q, [D - R, R], dim=-1)
    return torch.cat([nope, rotate(rope)], dim=-1)


o = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
q = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
for name, fn, x in (
    ("(a) inverse rope", inverse_rope, o),
    ("(b) q norm + rope", q_norm_rope, q),
):
    row = []
    for label, opts in (("default cat", None), ("ConcatKernel", OPTS)):
        torch._dynamo.reset()
        f = (
            torch.compile(fn, fullgraph=True, options=opts)
            if opts
            else torch.compile(fn, fullgraph=True)
        )
        us, n, _ = kernel_us(fwd_bwd(f, x))
        row.append(f"{label} {us:7.1f} us ({n} kernels)")
    print(f"{name:18s} fwd+bwd: " + " | ".join(row))
print(
    f"ideal (a): {4 * T * H * D * 2 / 7.1e12 * 1e6:.0f} us, (b): {5 * T * H * D * 2 / 7.1e12 * 1e6:.0f} us"
)
