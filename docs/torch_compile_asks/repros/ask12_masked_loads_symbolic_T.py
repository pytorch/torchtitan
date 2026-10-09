# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""With a symbolic token count, Inductor emits the same pointwise kernel plus an `xindex < xnumel`
mask on every load; for this many-broadcast-load kernel that alone costs ~2x on SM103, even
though xnumel = 7168 * T keeps its divisibility hint. A plain SwiGLU is unaffected (control).

Expected on 1x GB300 (torch 2.15.0.dev20260926+cu130, triton 3.8.0; see ../README.md, ask 12):
residual sum T static ~314 us vs T dynamic ~663 us; swiglu control ~134 us either way.
"""
import torch
import torch.nn.functional as F
from _common import kernel_us

T, N, D = 16384, 7, 7168
torch.manual_seed(0)


def weighted_sum(scores, stack, partial):
    probs = torch.softmax(scores, dim=-1)
    out = (probs[:, :-1, None] * stack.float()).sum(dim=1) + probs[
        :, -1:
    ] * partial.float()
    return out.to(stack.dtype)


def swiglu(g, u):
    return F.silu(g) * u


cases = {
    "residual weighted sum": (
        weighted_sum,
        lambda: (
            torch.randn(T, N + 1, device="cuda"),
            torch.randn(T, N, D, device="cuda", dtype=torch.bfloat16),
            torch.randn(T, D, device="cuda", dtype=torch.bfloat16),
        ),
    ),
    "swiglu (control)": (
        swiglu,
        lambda: (
            torch.randn(T, 9216, device="cuda", dtype=torch.bfloat16),
            torch.randn(T, 9216, device="cuda", dtype=torch.bfloat16),
        ),
    ),
}
for name, (fn, make) in cases.items():
    row = []
    for dyn in (False, True):
        torch._dynamo.reset()
        args = make()
        if dyn:
            for a in args:
                torch._dynamo.mark_dynamic(a, 0)
        f = torch.compile(fn, fullgraph=True)
        with torch.no_grad():
            f(*args)
            us, n, top = kernel_us(lambda: f(*args))
        row.append(f"{'T dynamic' if dyn else 'T static'} {us:7.1f} us")
    print(f"{name:22s} fwd: " + " | ".join(row))
