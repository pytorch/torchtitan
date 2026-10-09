# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Small reductions are unrolled into a pointwise kernel only if the reduction size is a static
integer below config.unroll_reductions_threshold (default 8, strict <; ir.py:1903-1908). A symbolic
size is never unrolled, whatever its value range. So the Kimi residual's weighted sum over its stack
width N is fast for static N <= 7, slow for static N = 8 (Kimi's widest stack), and slow for any
dynamic N.

Op: stack [T, N, D] + partial [T, D], softmax weights recomputed per element, forward only.

Expected on 1x GB300 (torch 2.15.0.dev20260926+cu130, triton 3.8.0; see ../README.md, ask 6):
N=7 static ~314 us, N=8 static ~1574 us, N=8 threshold 9 ~353 us, N=7 dynamic ~1245 us (also with threshold 64).
"""
import torch
from _common import kernel_us
from torch._inductor.utils import run_and_get_code

T, D = 16384, 7168


def weighted_sum(scores, stack, partial):
    probs = torch.softmax(scores, dim=-1)
    out = (probs[:, :-1, None] * stack.float()).sum(dim=1) + probs[
        :, -1:
    ] * partial.float()
    return out.to(stack.dtype)


partial = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)
for name, n, dyn, opts in (
    ("N=7 static", 7, False, None),
    ("N=8 static", 8, False, None),
    ("N=8 static, threshold 9", 8, False, {"unroll_reductions_threshold": 9}),
    ("N=7 dynamic", 7, True, None),
    ("N=7 dynamic, threshold 64", 7, True, {"unroll_reductions_threshold": 64}),
):
    torch._dynamo.reset()
    torch.manual_seed(0)
    stack = torch.randn(T, n, D, device="cuda", dtype=torch.bfloat16)
    scores = torch.randn(T, n + 1, device="cuda")
    if dyn:
        torch._dynamo.mark_dynamic(stack, 1)
        torch._dynamo.mark_dynamic(scores, 1)
    f = (
        torch.compile(weighted_sum, fullgraph=True, options=opts)
        if opts
        else torch.compile(weighted_sum, fullgraph=True)
    )
    with torch.no_grad():
        _, (code,) = run_and_get_code(f, scores, stack, partial)
        kinds = sorted(
            {
                w.split("(")[0].split("_fused")[0]
                for w in code.split()
                if w.startswith(("triton_poi", "triton_red", "triton_per"))
            }
        )
        us, _, top = kernel_us(lambda: f(scores, stack, partial))
    ideal = (2 * T * n * D + 4 * T * D) / 7.1e12 * 1e6
    print(f"{name:28s} fwd {us:7.1f} us (ideal {ideal:.0f})  kernel kinds {kinds}")
