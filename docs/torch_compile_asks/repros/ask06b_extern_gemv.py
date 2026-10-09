# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A batched matmul with a tiny K goes to extern cuBLAS, forcing a large fp32 operand to be
materialized; the explicit weighted sum fuses with the upcast.

Kimi K3 attention residual: probs [T, 1, W] @ values [T, W, D] in fp32, W=8 stack entries, D=7168.
Inductor's decompose_mm_pass (fx_passes/decompose_mem_bound_mm.py:62-90: mat1.shape[0] >= 10240 and
2 of m, k, n < 32) does handle this shape, but it is opt-in
(post_grad_fusion_options={"decompose_mm_pass": {}}) and its first-dim floor excludes small T.

Expected on 1x GB300 (torch 2.15.0.dev20260926+cu130, triton 3.8.0; see ../README.md, ask 6(b)):
T=16384: bmm ~3475 us, bmm + decompose_mm_pass ~2357 us, explicit sum ~2155 us; T=2048: 459 / 459 / 276 us.
"""
import torch
from _common import fwd_bwd, kernel_us
from torch._inductor.utils import run_and_get_code

W, D = 8, 7168
DECOMPOSE = {"post_grad_fusion_options": {"decompose_mm_pass": {}}}


def as_written(values, scores):
    probs = torch.softmax(scores, dim=-1).unsqueeze(1)
    return torch.matmul(probs, values.float()).squeeze(1).to(values.dtype)


def explicit_sum(values, scores):
    probs = torch.softmax(scores, dim=-1)
    return (probs.unsqueeze(-1) * values.float()).sum(dim=1).to(values.dtype)


for T in (16384, 2048):
    torch.manual_seed(0)
    values = torch.randn(
        T, W, D, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    scores = torch.randn(T, W, device="cuda", requires_grad=True)
    for name, fn, opts in (
        ("bmm as written", as_written, None),
        ("bmm + decompose_mm_pass", as_written, DECOMPOSE),
        ("explicit sum", explicit_sum, None),
    ):
        torch._dynamo.reset()
        compiled = (
            torch.compile(fn, fullgraph=True, options=opts)
            if opts
            else torch.compile(fn, fullgraph=True)
        )
        _, (code,) = run_and_get_code(compiled, values.detach(), scores.detach())
        externs = sorted(
            {
                line.split("(")[0].strip()
                for line in code.splitlines()
                if "extern_kernels." in line
            }
        )
        us, n, top = kernel_us(fwd_bwd(compiled, values, scores))
        print(
            f"T={T:5d} {name:24s} fwd+bwd {us:8.1f} us  {n:2d} kernels  forward externs {externs}  top {top[:2]}"
        )
    print(
        f"T={T:5d} ideal fwd+bwd (read values 2x + write grad, bf16): {(2 * 3 * T * W * D) / 7.1e12 * 1e6:.0f} us"
    )
