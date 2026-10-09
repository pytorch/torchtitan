# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compiled and eager results differ by 1 ulp on some elements because Triton contracts
`a*c - b*d` into FMAs, while eager rounds each multiply and the subtraction to fp32 as separate
ops (no contraction across ops). Two cases from torchtitan:
(a) complex RoPE: eager complex mul vs compiled real arithmetic;
(b) cos/sin rotate-half RoPE: eager vs compiled (upstream test expects bitwise; fails on GB300).
Inductor sets Triton's enable_fp_fusion = not config.emulate_precision_casts (codegen/triton.py), so
TRITON_DEFAULT_FP_FUSION is ignored; run with EMULATE_PRECISION_CASTS=1 to disable contraction.

Expected on 1x GB300 (torch 2.15.0.dev20260926+cu130, triton 3.8.0; see ../README.md, Known behaviors):
(a) ~44, (a') ~92, (b) ~81 of 4194304 elements differ by 1 bf16 ulp; with emulate_precision_casts (a') and (b) 0.
"""
import os

import torch

torch.manual_seed(0)
T, H, R = 4096, 8, 128
x = torch.randn(T, H, R, device="cuda", dtype=torch.bfloat16)
angles = torch.randn(T, 1, R // 2, device="cuda")
cache = torch.polar(torch.ones_like(angles), angles)


def complex_eager(x):
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * cache).flatten(-2).type_as(x)


def complex_real(x):
    c = torch.view_as_real(cache)
    cs, sn = c[..., 0], c[..., 1]
    p = x.float().unflatten(-1, (-1, 2))
    a, b = p[..., 0], p[..., 1]
    return torch.stack((a * cs - b * sn, a * sn + b * cs), -1).flatten(-2).type_as(x)


cos_sin = torch.cat([angles, angles], -1)
cos, sin = cos_sin.cos(), cos_sin.sin()


def rotate_half(x):
    xf = x.float()
    x1, x2 = xf[..., : R // 2], xf[..., R // 2 :]
    return (xf * cos + torch.cat((-x2, x1), -1) * sin).type_as(x)


def report(name, eager, compiled):
    diff = (eager.float() - compiled.float()).abs()
    print(
        f"{name:32s} mismatched {int((diff > 0).sum()):6d} / {diff.numel()}  max abs diff {diff.max().item():.2e}"
    )


torch._inductor.config.emulate_precision_casts = (
    os.environ.get("EMULATE_PRECISION_CASTS") == "1"
)
print("emulate_precision_casts =", torch._inductor.config.emulate_precision_casts)
report(
    "(a) complex eager vs real compiled",
    complex_eager(x),
    torch.compile(complex_real, fullgraph=True)(x),
)
report(
    "(a') real eager vs real compiled",
    complex_real(x),
    torch.compile(complex_real, fullgraph=True)(x),
)
report(
    "(b) rotate-half eager vs compiled",
    rotate_half(x),
    torch.compile(rotate_half, fullgraph=True)(x),
)
