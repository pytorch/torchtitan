# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""When one region sees two hidden sizes (Kimi K3 SiTUGLU: routed F=3072, then shared F=6144),
automatic dynamic shapes make F symbolic. gate/up are views of a packed [T, 2, F] projection, so
their stride (2F) and up's storage offset (F) become symbolic too, and the pointwise kernels index
with a runtime divisor (`xindex % ks0`) instead of a constant one. mark_static(F) restores it at one
graph per F.

SwiGLU on gate/up = unbind of [16384, 2, 6144] bf16, fwd+bwd; static = compiled only at F=6144,
dynamic = compiled at F=3072 first, then called at F=6144 (automatic dynamic).

Expected on 1x GB300 (torch 2.15.0.dev20260926+cu130, triton 3.8.0; see ../README.md, ask 5):
F static ~361 us, F dynamic (3072 then 6144) ~601 us.
"""
import torch
import torch.nn.functional as F
from _common import kernel_us

T, HID, OTHER = 16384, 6144, 3072


def swiglu(gate, up):
    return F.silu(gate) * up


def packed(hid):
    gate_up = torch.randn(
        T, 2, hid, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    gate, up = gate_up.unbind(1)
    return gate_up, gate, up


for name, warm in (
    ("F static (compiled at 6144 only)", False),
    ("F dynamic (3072 first, then 6144)", True),
):
    torch._dynamo.reset()
    torch.manual_seed(0)
    f = torch.compile(swiglu, fullgraph=True)
    if warm:
        _, g0, u0 = packed(OTHER)
        f(g0, u0).float().sum().backward()
    gate_up, gate, up = packed(HID)
    grad = torch.randn(T, HID, device="cuda", dtype=torch.bfloat16)

    def step(f=f):
        gate_up.grad = None
        f(gate, up).backward(grad, retain_graph=True)

    us, n, _ = kernel_us(step)
    print(f"{name:36s} fwd+bwd {us:7.1f} us  {n} kernels")
