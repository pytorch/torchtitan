# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inductor's mix-order-reduction heuristic evaluates `nrow >= 4096` (and `nrow * ncol >= 5M`) on a
symbolic token count and installs them as guards (torch/_inductor/scheduler.py:470-494). After the
dynamic graph is built at T >= 4096, a short batch recompiles. Under fullgraph local compile the
extra graphs count against recompile_limit=8 with no eager fallback.
`triton.mix_order_reduction_non_strict_mode=True` skips the check.

Function: the Kimi K3 attention residual (RMS-normalized scores, softmax, weighted sum), fwd+bwd.

Expected on 1x GB300 (torch 2.15.0.dev20260926+cu130, triton 3.8.0; see ../README.md, ask 1):
default 3 graphs with a '4096 <= stack.size()[0]' guard failure; non_strict_mode 2 graphs.
"""
import torch
from _common import num_graphs
from torch._dynamo.utils import guard_failures

D = 7168


def residual(partial, stack, proj, norm_w, eps: float = 1e-6):
    stack_f, partial_f = stack.float(), partial.float()
    w = norm_w.float() * proj.float()

    def score(v):
        return (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps) * w).sum(-1)

    probs = torch.softmax(
        torch.cat((score(stack_f), score(partial_f).unsqueeze(1)), 1), -1
    )
    return ((probs[:, :-1, None] * stack_f).sum(1) + probs[:, -1:] * partial_f).to(
        partial.dtype
    )


for non_strict in (False, True):
    torch._dynamo.reset()
    torch._inductor.config.triton.mix_order_reduction_non_strict_mode = non_strict
    f = torch.compile(residual, fullgraph=True)
    proj = torch.randn(D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    norm_w = torch.ones(D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    for T in (16384, 8192, 2048):
        stack = torch.randn(
            T, 7, D, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        partial = torch.randn(
            T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        f(partial, stack, proj, norm_w).float().sum().backward()
    reasons = [
        str(r).split("\n")[0][:110] for r in guard_failures.get(residual.__code__, [])
    ]
    print(
        f"non_strict={non_strict}: graphs={num_graphs(residual)}; guard failures: {reasons}"
    )
