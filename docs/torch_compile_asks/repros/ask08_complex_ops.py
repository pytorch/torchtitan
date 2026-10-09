# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Complex ops have no Inductor codegen: a compiled complex RoPE keeps eager fallbacks.

The same rotation spelled in real arithmetic fuses into one kernel per tensor.
Shape: DeepSeek-V4 attention rope channels, q [16384, 64, 64] bf16, complex cache [16384, 1, 32].

Expected on 1x GB300 (torch 2.15.0.dev20260926+cu130, triton 3.8.0; see ../README.md, ask 8):
eager ~446 us / 7 kernels, compiled complex ~444 us / 7 kernels, compiled real arithmetic ~199 us / 2 kernels.
"""
import warnings

import torch
from _common import fwd_bwd, kernel_us

T, H, R = 16384, 64, 64
torch.manual_seed(0)
x = torch.randn(T, H, R, device="cuda", dtype=torch.bfloat16, requires_grad=True)
angles = torch.randn(T, 1, R // 2, device="cuda")
cache = torch.polar(torch.ones_like(angles), angles)  # complex64


def complex_rotate(x):
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * cache).flatten(-2).type_as(x)


def real_rotate(x):
    c = torch.view_as_real(cache)
    cos, sin = c[..., 0], c[..., 1]
    pairs = x.float().unflatten(-1, (-1, 2))
    a, b = pairs[..., 0], pairs[..., 1]
    return (
        torch.stack((a * cos - b * sin, a * sin + b * cos), -1).flatten(-2).type_as(x)
    )


with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    variants = {
        "eager complex": complex_rotate,
        "compiled complex": torch.compile(complex_rotate, fullgraph=True),
        "compiled real arithmetic": torch.compile(real_rotate, fullgraph=True),
    }
    for name, fn in variants.items():
        us, n, top = kernel_us(fwd_bwd(fn, x))
        print(f"{name:26s} fwd+bwd {us:8.1f} us  {n:3d} kernels  top {top}")
print(
    "inductor warnings:",
    sorted({str(w.message)[:90] for w in caught if "complex" in str(w.message)}),
)
