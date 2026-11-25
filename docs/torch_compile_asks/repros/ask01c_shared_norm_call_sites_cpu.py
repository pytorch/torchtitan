# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU, aot_eager variant of ask01c_shared_norm_call_sites.py: Dynamo-level graph counts only (no
Inductor, so no mix-order guards; the non_strict variants equal their strict twins).

One compiled norm shared by call sites of different rank and layout runs out of graphs.

Qwen3.5's OffsetRMSNorm.forward is one fullgraph region used for the 2D layer norms [T, D] and
the 3D q/k norms [T, H, K]. q is a strided view (a `chunk` of the 2x-wide wq output), k is
contiguous with fewer heads. Dynamo keys graphs on rank, strides and sizes, Inductor's
mix-order-reduction heuristic adds `nrow >= 4096` / `nrow * ncol >= 5M` guards on the symbolic
T, and a no_grad pass doubles everything: 9+ graphs for one code object, a hard
FailOnRecompileLimitHit under fullgraph=True (recompile_limit=8).

Call pattern: one Qwen3.5-35B-A3B attention block + one GDN block per step, training at
T = 16k, 8k, 2k, then no_grad at 16k, 2k. The limit is raised here to count graphs.

Expected on CPU (torch 2.15.0.dev20260926+cu130; see ../README.md, ask 1), Dynamo-level only:
7 / 7 / 5 / 5 / 6 / 6 / 6 graphs for the seven configurations in order. The over-limit count
needs Inductor's mix-order guards on GPU (ask01c_shared_norm_call_sites.py).
"""
import torch
from _common import num_graphs
from torch._dynamo.utils import guard_failures

D, H_Q, H_KV, K = 2048, 16, 2, 256


def offset_rmsnorm(x, weight, eps: float = 1e-6):
    input_dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return ((1.0 + weight.float()) * x).to(input_dtype)


def step(norm, T, grad, w_d, w_k, contiguous_q=False):
    with torch.set_grad_enabled(grad):
        x = torch.randn(T, D, device="cpu", dtype=torch.bfloat16, requires_grad=grad)
        q_gate = torch.randn(
            T, H_Q, 2 * K, device="cpu", dtype=torch.bfloat16, requires_grad=grad
        )
        q = q_gate.chunk(2, dim=-1)[0]
        q = q.contiguous() if contiguous_q else q
        k = torch.randn(
            T, H_KV, K, device="cpu", dtype=torch.bfloat16, requires_grad=grad
        )
        outs = [
            norm(x, w_d),
            norm(q, w_k),
            norm(k, w_k),
            norm(x, w_d),
        ]  # attention block
        outs += [norm(x, w_d), norm(x, w_d)]  # GDN block
        if grad:
            sum(o.float().sum() for o in outs).backward()


def count(non_strict=False, contiguous_q=False, mark=None):
    """mark: None, or "x"/"view": mark dim 0 dynamic and the normalized dim static before the
    compiled call, on the input itself or on a fresh view of it."""
    torch._dynamo.reset()
    torch._inductor.config.triton.mix_order_reduction_non_strict_mode = non_strict
    compiled = torch.compile(offset_rmsnorm, fullgraph=True, backend="aot_eager")

    def norm(x, weight):
        if mark is not None:
            x = x.view(x.shape) if mark == "view" else x
            torch._dynamo.maybe_mark_dynamic(x, 0)
            torch._dynamo.mark_static(x, x.ndim - 1)
        return compiled(x, weight)

    w_d = torch.zeros(D, device="cpu", dtype=torch.bfloat16, requires_grad=True)
    w_k = torch.zeros(K, device="cpu", dtype=torch.bfloat16, requires_grad=True)
    for T in (16384, 8192, 2048):
        step(norm, T, True, w_d, w_k, contiguous_q)
    for T in (16384, 2048):
        step(norm, T, False, w_d, w_k, contiguous_q)
    reasons = [
        str(r).split("\n")[0][:90]
        for r in guard_failures.get(offset_rmsnorm.__code__, [])
    ]
    guard_failures.clear()
    return num_graphs(offset_rmsnorm), reasons


if __name__ == "__main__":
    torch._dynamo.config.recompile_limit = 64
    for kwargs in (
        {},
        {"non_strict": True},
        {"contiguous_q": True},
        {"non_strict": True, "contiguous_q": True},
        {"mark": "x"},
        {"mark": "x", "non_strict": True},
        {"mark": "view", "non_strict": True},
    ):
        graphs, reasons = count(**kwargs)
        print(f"{kwargs}: graphs={graphs}")
        for r in reasons:
            if "GLOBAL_STATE" not in r:
                print("   ", r)
