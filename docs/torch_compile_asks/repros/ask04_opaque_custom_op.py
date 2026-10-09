# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""An opaque custom op blocks fusion: compiling a MoE combine that ends in a custom-op
deterministic scatter_add only fuses the scaling; the op's deterministic index_put stays.

Rewritten as per-token sums (each token owns exactly K rows) it fuses and stays deterministic.
Shape: Kimi K3 rank-local combine, T=16384 tokens, K=16, D=3584 (N = 262144 rows) bf16.

Expected on 1x GB300 (torch 2.15.0.dev20260926+cu130, triton 3.8.0; see ../README.md, ask 4):
eager ~13540 us, compiled as written ~4248 us, per-token Function ~1102 us.
"""
import torch
from _common import kernel_us

T, K, D, E = 16384, 16, 3584, 112


@torch.library.custom_op("repro::deterministic_scatter_add", mutates_args=())
def deterministic_scatter_add(
    out: torch.Tensor, index: torch.Tensor, src: torch.Tensor
) -> torch.Tensor:
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        return out.scatter_add(0, index, src)
    finally:
        torch.use_deterministic_algorithms(prev)


@deterministic_scatter_add.register_fake
def _(out, index, src):
    return torch.empty_like(out)


def _bwd(ctx, grad):
    (index,) = ctx.saved_tensors
    return grad, None, torch.gather(grad, 0, index)


def _setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[1])


deterministic_scatter_add.register_autograd(_bwd, setup_context=_setup)


class CombineRows(torch.autograd.Function):
    """out[t] = sum_k score * row over token t's K rows (unrolled, fp32); backward gathers by token."""

    @staticmethod
    def forward(ctx, routed, scores, token, order_TK):
        ctx.save_for_backward(routed, scores, token)
        out = None
        for k in range(order_TK.shape[1]):
            idx = order_TK[:, k]
            term = routed[idx].float() * scores[idx].unsqueeze(-1)
            out = term if out is None else out + term
        return out.to(routed.dtype)

    @staticmethod
    def backward(ctx, grad):
        routed, scores, token = ctx.saved_tensors
        g = grad[token].float()
        return (
            (g * scores[:, None]).to(routed.dtype),
            (g * routed.float()).sum(-1),
            None,
            None,
        )


g = torch.Generator("cuda").manual_seed(0)
ids = torch.randint(0, E, (T, K), device="cuda", generator=g)
token = torch.argsort(ids.view(-1), stable=True) // K
order_TK = torch.argsort(token, stable=True).view(T, K)  # noqa: N816
routed = torch.randn(T * K, D, device="cuda", generator=g).bfloat16().requires_grad_()
scores = torch.rand(T * K, device="cuda", generator=g).requires_grad_()
x = torch.empty(T, D, device="cuda", dtype=torch.bfloat16)


def combine_as_written(routed, scores):
    scaled = (routed.float() * scores[:, None]).to(routed.dtype)
    return deterministic_scatter_add(
        torch.zeros_like(x), token[:, None].expand(-1, D), scaled
    )


def combine_per_token(routed, scores):
    return CombineRows.apply(routed, scores, token, order_TK)


grad = torch.randn(T, D, device="cuda", generator=g).bfloat16()
for name, fn in {
    "eager as written": combine_as_written,
    "compiled as written": torch.compile(combine_as_written, fullgraph=True),
    "compiled per-token rewrite": torch.compile(combine_per_token, fullgraph=True),
}.items():

    def step(fn=fn):
        routed.grad = scores.grad = None
        fn(routed, scores).backward(grad)

    us, n, top = kernel_us(step)
    print(f"{name:28s} fwd+bwd {us:8.1f} us  {n:3d} kernels  top {top}")
print(f"ideal (bytes / 7.1 TB/s): {(6 * T * K * D + 4 * T * D) / 7.1e12 * 1e6:.0f} us")
