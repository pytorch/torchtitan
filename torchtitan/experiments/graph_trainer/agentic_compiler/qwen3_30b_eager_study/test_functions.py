# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bitwise check of the eager Functions against the eager modules they replace (one layer).

    PYTHONPATH=<this dir>:. python test_functions.py [attention_input|attention_output|...]

Builds the modules of layer 0 of the eager recipe's model config on one GPU with random
weights, runs the module path and the Function path on the same inputs and upstream gradients,
and compares every output and gradient as bytes.
"""

import sys
import time

import graph_kernels as gk

import torch
from torchtitan.quantization.mxfp8.tensor import _LinearShardedTensorWithMXFP8Compute

from torchtitan_recipes.graph_trainer.megatron_bridge_benchmarks import (
    qwen3_30b_a3b_dist_moe_mxfp8_8gpu,
)

T = 32768


def bits_equal(a, b):
    if a is None or b is None:
        return a is None and b is None
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(
        a.detach().reshape(-1).contiguous().view(torch.uint8),
        b.detach().reshape(-1).contiguous().view(torch.uint8),
    )


def report(name, pairs):
    ok = True
    for label, a, b in pairs:
        eq = bits_equal(a, b)
        ok &= eq
        extra = (
            ""
            if eq
            else f"  max|diff|={(a.float() - b.float()).abs().max().item():.3e}"
        )
        print(f"  {name} {label:24s} {'EXACT' if eq else 'DIFF'}{extra}")
    return ok


def layer_modules():
    cfg = qwen3_30b_a3b_dist_moe_mxfp8_8gpu().model.layers[0]
    torch.manual_seed(0)
    with torch.device("cuda"):
        attn_norm = cfg.attention_norm.build().to(torch.bfloat16)
        attn = cfg.attention.build()
        attn.q_norm.to(torch.bfloat16)
        attn.k_norm.to(torch.bfloat16)
        ffn_norm = cfg.ffn_norm.build().to(torch.bfloat16)
    for m in (attn_norm, ffn_norm, attn.q_norm, attn.k_norm):
        with torch.no_grad():
            m.weight.copy_(1 + 0.1 * torch.randn_like(m.weight))
    for lin in (attn.qkv_linear.wqkv, attn.wo):
        data = 0.02 * torch.randn(lin.weight.shape, device="cuda", dtype=torch.bfloat16)
        lin.weight = torch.nn.Parameter(_LinearShardedTensorWithMXFP8Compute(data))
    return attn_norm, attn, ffn_norm


def grads(params):
    out = [p.grad for p in params]
    for p in params:
        p.grad = None
    return out


def test_attention_input(attn_norm, attn):
    x = torch.randn(T, 2048, device="cuda", dtype=torch.bfloat16)
    positions = torch.randint(0, 4096, (T,), device="cuda")
    up = [
        torch.randn(T, 32, 128, device="cuda", dtype=torch.bfloat16),
        torch.randn(T, 4, 128, device="cuda", dtype=torch.bfloat16),
        torch.randn(T, 4, 128, device="cuda", dtype=torch.bfloat16),
        torch.randn(T, 2048, device="cuda", dtype=torch.bfloat16),
    ]
    params = [
        attn_norm.weight,
        attn.qkv_linear.wqkv.weight,
        attn.q_norm.weight,
        attn.k_norm.weight,
    ]

    # Module path (the eager block's ops, local_compile off).
    xr = x.clone().requires_grad_()
    xq, xk, xv = attn.qkv_linear(attn_norm(xr))
    xq, xk = attn.rope(attn.q_norm(xq), attn.k_norm(xk), positions)
    torch.autograd.backward([xq, xk, xv, xr], up)
    ref_out, ref_gx, ref_g = (xq, xk, xv), xr.grad, grads(params)

    # Function path.
    xf = x.clone().requires_grad_()
    wqkv = attn.qkv_linear.wqkv
    weight = wqkv.weight
    ops = gk.mxfp8_operands(weight)
    t0 = time.time()
    q, k, v, x_res = gk.AttentionInput.apply(
        xf,
        attn_norm.weight,
        weight,
        ops.weight_qdata_fprop_KN,
        ops.weight_scale_fprop_swizzled,
        ops.weight_qdata_dgrad_NK,
        ops.weight_scale_dgrad_swizzled,
        attn.q_norm.weight,
        attn.k_norm.weight,
        attn.rope.cache,
        positions,
        gk.linear_meta(wqkv, weight),
    )
    torch.autograd.backward([q, k, v, x_res], up)
    torch.cuda.synchronize()
    print(f"  (first call incl. JIT: {time.time() - t0:.1f} s)")
    got_g = grads(params)
    return report(
        "F1",
        [
            ("q", ref_out[0], q),
            ("k", ref_out[1], k),
            ("v", ref_out[2], v),
            ("grad x", ref_gx, xf.grad),
            ("grad attention_norm.w", ref_g[0], got_g[0]),
            ("grad wqkv.w", ref_g[1], got_g[1]),
            ("grad q_norm.w", ref_g[2], got_g[2]),
            ("grad k_norm.w", ref_g[3], got_g[3]),
        ],
    )


def test_attention_output(attn, ffn_norm):
    a = torch.randn(T, 4096, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(T, 2048, device="cuda", dtype=torch.bfloat16)
    up = [torch.randn(T, 2048, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    params = [attn.wo.weight, ffn_norm.weight]

    ar, xr = a.clone().requires_grad_(), x.clone().requires_grad_()
    h = xr + attn.wo(ar)
    n = ffn_norm(h)
    # The norm output has two consumers (router gate and routed experts).
    torch.autograd.backward([h, n, n], up)
    ref = (h, n, ar.grad, xr.grad, *grads(params))

    af, xf = a.clone().requires_grad_(), x.clone().requires_grad_()
    wo = attn.wo
    ops = gk.mxfp8_operands(wo.weight)
    h2, n_r, n_e = gk.AttentionOutput.apply(
        af,
        xf,
        wo.weight,
        ops.weight_qdata_fprop_KN,
        ops.weight_scale_fprop_swizzled,
        ops.weight_qdata_dgrad_NK,
        ops.weight_scale_dgrad_swizzled,
        ffn_norm.weight,
        gk.linear_meta(wo, wo.weight),
    )
    torch.autograd.backward([h2, n_r, n_e], up)
    got = (h2, n_r, af.grad, xf.grad, *grads(params))
    return report(
        "F2",
        list(
            zip(
                ["h", "normed", "grad attn", "grad x", "grad wo.w", "grad ffn_norm.w"],
                ref,
                got,
            )
        ),
    )


def test_router():
    cfg = qwen3_30b_a3b_dist_moe_mxfp8_8gpu().model.layers[0].feed_forward.router
    with torch.device("cuda"):
        router = cfg.build()
    router.gate.weight.data = (0.02 * torch.randn_like(router.gate.weight)).to(
        torch.bfloat16
    )
    router.train()
    aux = router.aux_loss
    x = torch.randn(T, 2048, device="cuda", dtype=torch.bfloat16)
    pad = torch.zeros(T, dtype=torch.bool, device="cuda")
    pad[-1000:] = True
    denom = torch.tensor(T * 8 - 8000, device="cuda")
    g = torch.randn(T, 8, device="cuda", dtype=torch.float32)

    def reset():
        router.tokens_per_expert_E.zero_()
        aux._cumulative_expert_counts_E.zero_()
        aux.instance_acc.zero_()

    def run(fn):
        reset()
        xr = x.clone().requires_grad_()
        topk, ids, _ = fn(xr)
        # Dist-MoE masks padded routes' scores before dispatch.
        topk.masked_fill(pad.unsqueeze(-1), 0.0).backward(g)
        gw = router.gate.weight.grad
        router.gate.weight.grad = None
        return (
            topk,
            ids,
            xr.grad,
            gw,
            router.tokens_per_expert_E.clone(),
            aux._cumulative_expert_counts_E.clone(),
            aux.instance_acc.clone(),
        )

    ref = run(
        lambda xr: router(xr, None, padding_mask_T=pad, aux_loss_denominator=denom)
    )
    got = run(lambda xr: gk.router_forward(router, xr, None, pad, denom))
    return report(
        "F3",
        list(
            zip(
                [
                    "topk scores",
                    "expert ids",
                    "grad x",
                    "grad gate.w",
                    "tokens_per_expert",
                    "cumulative counts",
                    "aux metric",
                ],
                ref,
                got,
            )
        ),
    )


def test_loss():
    from torchtitan.components.loss import cross_entropy_loss

    pred = torch.randn(4096, 151936, device="cuda", dtype=torch.bfloat16)
    labels = torch.randint(0, 151936, (4096,), device="cuda")
    labels[::7] = -100
    denom = torch.tensor(30000, device="cuda")
    g = torch.tensor(0.75, device="cuda")
    pr = pred.clone().requires_grad_()
    ref = cross_entropy_loss(pr, labels) / denom
    ref.backward(g)
    pf = pred.clone().requires_grad_()
    got = gk.CrossEntropyChunk.apply(pf, labels, denom)
    got.backward(g)
    return report("F4", [("loss", ref, got), ("grad pred", pr.grad, pf.grad)])


def test_expert_weight_prep():
    from torchtitan.models.common.dist_moe.mxfp8 import (
        _DistMoeW13ShardedTensor,
        _DistMoeW2ShardedTensor,
    )

    stock = {
        c: c._build_operands
        for c in (_DistMoeW13ShardedTensor, _DistMoeW2ShardedTensor)
    }
    w13 = torch.randn(16, 2, 768, 2048, device="cuda") * 0.02
    w2 = torch.randn(16, 2048, 768, device="cuda") * 0.02
    refs = {
        c: stock[c](c(w), w.to(torch.bfloat16))
        for c, w in ((_DistMoeW13ShardedTensor, w13), (_DistMoeW2ShardedTensor, w2))
    }
    gk.install_expert_weight_prep()
    pairs = []
    for c, w, name in (
        (_DistMoeW13ShardedTensor, w13, "w13"),
        (_DistMoeW2ShardedTensor, w2, "w2"),
    ):
        bf16 = w.to(torch.bfloat16)
        first = c._build_operands(c(w), bf16)
        out = stock[c](c(w * 3), (w * 3).to(torch.bfloat16))  # stale operands to refill
        refilled = c._build_operands(c(w), bf16, out=out)
        assert refilled is out
        for field in ("qdata", "fprop_scale", "dgrad_scale"):
            pairs.append(
                (f"{name} {field}", getattr(refs[c], field), getattr(first, field))
            )
            pairs.append(
                (
                    f"{name} {field} refill",
                    getattr(refs[c], field),
                    getattr(refilled, field),
                )
            )
    return report("F5", pairs)


def main():
    which = sys.argv[1:] or [
        "attention_input",
        "attention_output",
        "router",
        "loss",
        "expert_weight_prep",
    ]
    attn_norm, attn, ffn_norm = layer_modules()
    ok = True
    if "attention_input" in which:
        ok &= test_attention_input(attn_norm, attn)
    if "attention_output" in which:
        ok &= test_attention_output(attn, ffn_norm)
    if "router" in which:
        ok &= test_router()
    if "loss" in which:
        ok &= test_loss()
    if "expert_weight_prep" in which:
        ok &= test_expert_weight_prep()
    print("ALL EXACT" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
