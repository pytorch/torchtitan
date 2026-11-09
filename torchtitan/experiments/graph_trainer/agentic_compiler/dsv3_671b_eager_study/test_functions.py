# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bitwise check of the eager Functions against the eager modules they replace.

    PYTHONPATH=<this dir>:<agentic_compiler dir>:. python test_functions.py [attention_input|...]

Builds modules of the eager recipe's model config (layer 3, the first MoE layer, and layer 0,
dense) on one GPU with random weights, runs the module path and the Function path on the same
inputs and upstream gradients, and compares every output and gradient as bytes.
"""

import os
import sys

os.environ.setdefault("RUN", "/tmp/dsv3_eager_study_test")

import eager_recipe
import graph_kernels as gk

import torch
from override_dsv3 import graph_kernel_block

from torchtitan_recipes.overrides.fused_mla import (
    _materialize_fused_mla_kv,
    _resolve_positions,
)

T = 4096


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
        extra = ""
        if not eq and a is not None and b is not None and a.shape == b.shape:
            extra = f"  max|diff|={(a.float() - b.float()).abs().max().item():.3e}"
        elif not eq:
            extra = f"  {getattr(a, 'shape', a)} {getattr(a, 'dtype', '')} vs {getattr(b, 'shape', b)} {getattr(b, 'dtype', '')}"
        print(f"  {name} {label:28s} {'EXACT' if eq else 'DIFF'}{extra}")
    return ok


def randn(*shape, dtype=torch.bfloat16, scale=1.0):
    return (scale * torch.randn(*shape, device="cuda")).to(dtype)


def block_config(layer):
    config = eager_recipe._base(local_compile=False)
    return graph_kernel_block(config.model.layers[layer])


def init_(module):
    """Random weights: norms near 1, linears small, kept in the module's parameter types."""
    from torchtitan.quantization.mxfp8.tensor import (
        _LinearShardedTensorWithMXFP8Compute,
    )

    for name, p in list(module.named_parameters()):
        owner = module.get_submodule(name.rsplit(".", 1)[0]) if "." in name else module
        attr = name.rsplit(".", 1)[-1]
        if isinstance(p.data, _LinearShardedTensorWithMXFP8Compute):
            data = randn(*p.shape, scale=0.02)
            setattr(
                owner,
                attr,
                torch.nn.Parameter(_LinearShardedTensorWithMXFP8Compute(data)),
            )
        elif p.dim() == 1:
            setattr(owner, attr, torch.nn.Parameter(1 + randn(*p.shape, scale=0.1)))
        else:
            setattr(owner, attr, torch.nn.Parameter(randn(*p.shape, scale=0.02)))


def grads(params):
    out = [p.grad for p in params]
    for p in params:
        p.grad = None
    return out


def build_attention():
    cfg = block_config(3)
    torch.manual_seed(0)
    with torch.device("cuda"):
        attn_norm = cfg.attention_norm.build()
        attn = cfg.attention.build()
        ffn_norm = cfg.ffn_norm.build()
    for m in (attn_norm, attn, ffn_norm):
        init_(m)
    attn.rope._init_self_buffers(buffer_device=torch.device("cuda"))
    return attn_norm, attn, ffn_norm


def test_attention_input(attn_norm, attn):
    x = randn(T, 7168)
    up = [randn(T, 1536), randn(T, 576), randn(T, 7168)]
    params = [attn_norm.weight, attn.wq_a.weight, attn.wkv_a.weight]

    xr = x.clone().requires_grad_()
    y = attn_norm(xr)
    q_a, kv_a = attn.wq_a(y), attn.wkv_a(y)
    torch.autograd.backward([q_a, kv_a, xr], up)
    ref = (q_a, kv_a, xr.grad, *grads(params))
    ref_ops = gk._grad_operands(xr.grad)

    # A non-leaf input, as in the model: its tensor hook sees the gradient object the engine
    # passes on to layer L-1 (AccumulateGrad of a leaf may store a copy).
    x_leaf = x.clone().requires_grad_()
    xf = x_leaf.view_as(x_leaf)
    passed = []
    xf.register_hook(passed.append)
    q_a2, kv_a2, x_res = gk.AttentionInput.apply(
        xf,
        attn_norm.weight,
        attn.wq_a.weight,
        attn.wkv_a.weight,
        gk.linear_meta(attn.wq_a, attn.wq_a.weight),
        gk.linear_meta(attn.wkv_a, attn.wkv_a.weight),
    )
    torch.autograd.backward([q_a2, kv_a2, x_res], up)
    got = (q_a2, kv_a2, x_leaf.grad, *grads(params))
    got_ops = gk._take_grad_quant(passed[0])
    names = ["wq_a out", "wkv_a out", "grad x", "grad attention_norm.w"]
    names += ["grad wq_a.w", "grad wkv_a.w"]
    pairs = list(zip(names, ref, got))
    pairs += list(
        zip(
            [
                "grad x row qdata",
                "grad x row scale",
                "grad x col qdata.t",
                "grad x col scale",
            ],
            ref_ops,
            got_ops or (None,) * 4,
        )
    )
    return report("attention_input", pairs)


def test_wq_b(attn):
    x = randn(T, 1536)
    up = randn(T, 24576)
    xr = x.clone().requires_grad_()
    out = attn.wq_b(xr)
    out.backward(up)
    ref = (out, xr.grad, *grads([attn.wq_b.weight]))
    xf = x.clone().requires_grad_()
    out2 = gk.linear(attn.wq_b, xf)
    out2.backward(up)
    got = (out2, xf.grad, *grads([attn.wq_b.weight]))
    return report("wq_b", list(zip(["out", "grad x", "grad wq_b.w"], ref, got)))


def test_mla_kv(attn):
    mla = attn.inner_attention
    kv_c = randn(T, 512)
    k_pe = randn(T, 64)
    positions = (torch.arange(T, device="cuda") % 1024).to(torch.int64)
    q = randn(T, 128, 192)
    up = [randn(T, 128, 192), randn(T, 128, 128)]
    params = [mla.wkv_b.weight]

    kr, pr = kv_c.clone().requires_grad_(), k_pe.clone().requires_grad_()
    k, v = _materialize_fused_mla_kv(
        q, mla.project_mla_kv(kr), pr, rope_cache=attn.rope.cache, positions_T=positions
    )
    torch.autograd.backward([k, v], up)
    ref = (k, v, kr.grad, pr.grad, *grads(params))

    kf, pf = kv_c.clone().requires_grad_(), k_pe.clone().requires_grad_()
    k2, v2 = gk.MLAKeyValue.apply(
        kf,
        pf,
        mla.wkv_b.weight,
        torch.view_as_real(attn.rope.cache).contiguous(),
        _resolve_positions(positions, kf.unsqueeze(0)),
        attn.qk_nope_head_dim,
        mla.packed_kv_head_dim,
        gk.linear_meta(mla.wkv_b, mla.wkv_b.weight),
    )
    torch.autograd.backward([k2, v2], up)
    got = (k2, v2, kf.grad, pf.grad, *grads(params))
    names = ["k", "v", "grad kv_c", "grad k_pe", "grad wkv_b.w"]
    return report("mla_kv", list(zip(names, ref, got)))


def test_attention_output(attn, ffn_norm):
    a = randn(T, 16384)
    x = randn(T, 7168)
    up = [randn(T, 7168) for _ in range(2)]
    params = [attn.wo.weight, ffn_norm.weight]

    ar, xr = a.clone().requires_grad_(), x.clone().requires_grad_()
    h = xr + attn.wo(ar)
    y = ffn_norm(h)
    torch.autograd.backward([h, y], up)
    ref = (h, y, *gk.quantize_rows(y), ar.grad, xr.grad, *grads(params))

    af, xf = a.clone().requires_grad_(), x.clone().requires_grad_()
    h2, y2, y_q, y_s = gk.AttentionOutput.apply(
        af, xf, attn.wo.weight, ffn_norm.weight, gk.linear_meta(attn.wo, attn.wo.weight)
    )
    torch.autograd.backward([h2, y2], up)
    got = (h2, y2, y_q, y_s, af.grad, xf.grad, *grads(params))
    names = ["h", "y", "y row qdata", "y row scale", "grad attn", "grad x"]
    names += ["grad wo.w", "grad ffn_norm.w"]
    return report("attention_output", list(zip(names, ref, got)))


def test_router():
    cfg = block_config(3).feed_forward.router
    torch.manual_seed(1)
    with torch.device("cuda"):
        router = cfg.build()
    router.gate.weight = torch.nn.Parameter(
        randn(*router.gate.weight.shape, scale=0.02)
    )
    router.train()
    aux = router.aux_loss
    x = randn(T, 7168)
    expert_bias = randn(256, dtype=torch.float32, scale=0.01)
    pad = torch.zeros(T, dtype=torch.bool, device="cuda")
    pad[-300:] = True
    denom = torch.tensor(T * 8 - 2400, device="cuda")
    g = randn(T, 8, dtype=torch.float32)

    def reset():
        router.tokens_per_expert_E.zero_()
        aux.instance_acc.zero_()

    def run(fn):
        reset()
        xr = x.clone().requires_grad_()
        scores, ids = fn(xr)
        scores.backward(g)
        gw = router.gate.weight.grad
        router.gate.weight.grad = None
        return (
            scores,
            ids,
            xr.grad,
            gw,
            router.tokens_per_expert_E.clone(),
            aux.instance_acc.clone(),
        )

    def stock(xr):
        # The router, then Dist-MoE's masking of padded routes (routed_experts.forward).
        scores, ids, _ = router(
            xr, expert_bias, padding_mask_T=pad, aux_loss_denominator=denom
        )
        pad_TK = pad.unsqueeze(-1)
        return scores.masked_fill(pad_TK, 0.0), ids.masked_fill(pad_TK, -1)

    ref = run(stock)
    got = run(lambda xr: gk.router_forward(router, xr, expert_bias, pad, denom))
    names = ["scores", "expert ids", "grad x", "grad gate.w", "tokens_per_expert"]
    names += ["aux metric"]
    return report("router", list(zip(names, ref, got)))


def test_feed_forward(layer):
    cfg = block_config(layer).feed_forward
    cfg = cfg.shared_experts if layer >= 3 else cfg
    torch.manual_seed(2)
    with torch.device("cuda"):
        ffn = cfg.build()
    init_(ffn)
    y = randn(T, 7168)
    g = randn(T, 7168)
    params = [ffn.w13.weight, ffn.w2.weight]

    yr = y.clone().requires_grad_()
    out = ffn(yr)
    out.backward(g)
    ref = (out, yr.grad, *grads(params))

    ok = True
    for stash in (False, True):
        yf = y.clone().requires_grad_()
        y_q, y_s = gk.quantize_rows(y)
        out2 = gk.SwiGLUFeedForward.apply(
            yf,
            y_q,
            y_s,
            ffn.w13.weight,
            ffn.w2.weight,
            gk.linear_meta(ffn.w13, ffn.w13.weight),
            gk.linear_meta(ffn.w2, ffn.w2.weight),
            False,
        )
        if stash:
            gk._stash_grad_quant(g, *gk._grad_operands(g))
        lookups = dict(gk.GRAD_QUANT_LOOKUPS)
        out2.backward(g)
        hit = gk.GRAD_QUANT_LOOKUPS["hit"] - lookups.get("hit", 0)
        got = (out2, yf.grad, *grads(params))
        print(
            f"  ffn layer {layer} stash={stash}: quantized gradient taken from stash: {hit}"
        )
        ok &= hit == int(stash)
        names = ["out", "grad y", "grad w13.w", "grad w2.w"]
        ok &= report(
            f"feed_forward[{layer}{', stash' if stash else ''}]",
            list(zip(names, ref, got)),
        )
    return ok


def test_fsdp_copy_in():
    """G-007 (w13 WGRAD + row-block permutation) and G-003 (pack) against FSDP2's stock
    reduce-scatter input for the attention / shared-expert group of one MoE block."""
    from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
        _default_reduce_scatter_input_fn,
    )

    world = 256
    module = gk._pack_module("ponk")
    shapes = [shape for _, shape in module.SHAPES]
    shapes[10] = (2, 2048, 7168)  # the stacked shared-expert w13, sharded on dim 1
    shard_dims = [1 if i == 10 else 0 for i in range(len(shapes))]
    grads = [randn(*shape, scale=0.01) for shape in shapes]
    # w13's WGRAD operands, as SwiGLUFeedForward's backward builds them.
    g_TN = randn(T, 4096)
    y = randn(T, 7168)
    q_row, s_row, q_col_t, s_col = gk._grad_operands(g_TN)
    _, y_col, _, y_scol = gk.mxfp8_quantize_cuda(
        y, rowwise=False, colwise=True, scaling_mode="rceil"
    )
    y_scol = gk.triton_mx_block_rearrange(y_scol)
    grads[10] = gk.scaled_mm(q_col_t, y_col, s_col, y_scol).view(2, 2048, 7168)

    ref_list = list(grads)
    out_ref = torch.empty(module.NOUT, dtype=torch.bfloat16, device="cuda")
    _default_reduce_scatter_input_fn(ref_list, shard_dims=shard_dims, world_size=world)(
        out_ref
    )

    pairs = []
    for permute in (False, True):
        got_list = list(grads)
        if permute:
            permuted = gk.kernel("5nj2", q_col_t, y_col, s_col, y_scol)(
                q_col_t, y_col, s_col, y_scol
            )
            gk._PREPERMUTED[permuted.data_ptr()] = permuted
            got_list[10] = permuted.view(2, 2048, 7168)
        out = torch.full(
            (module.NOUT,), float("nan"), dtype=torch.bfloat16, device="cuda"
        )
        gk.reduce_scatter_input_fn(got_list, shard_dims=shard_dims, world_size=world)(
            out
        )
        pairs.append(
            (f"reduce-scatter input{' (G-007)' if permute else ''}", out_ref, out)
        )
    return report("fsdp_copy_in", pairs)


def test_expert_weight_prep():
    from torchtitan.models.common.dist_moe.mxfp8 import (
        _DistMoeW13ShardedTensor,
        _DistMoeW2ShardedTensor,
    )

    stock = {
        c: c._build_operands
        for c in (_DistMoeW13ShardedTensor, _DistMoeW2ShardedTensor)
    }
    # The all-gathered BF16 expert weights of one EP rank (4 local experts).
    w13 = randn(4, 2, 2048, 7168, scale=0.02)
    w2 = randn(4, 7168, 2048, scale=0.02)
    refs = {
        c: stock[c](c(w.float()), w)
        for c, w in ((_DistMoeW13ShardedTensor, w13), (_DistMoeW2ShardedTensor, w2))
    }
    gk.install_expert_weight_prep()
    pairs = []
    try:
        for c, w, name in (
            (_DistMoeW13ShardedTensor, w13, "w13"),
            (_DistMoeW2ShardedTensor, w2, "w2"),
        ):
            first = c._build_operands(c(w.float()), w)
            stale = stock[c](c(w.float()), w * 3)  # operands to refill
            refilled = c._build_operands(c(w.float()), w, out=stale)
            assert refilled is stale
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
    finally:
        for c, fn in stock.items():
            c._build_operands = fn
    return report("expert_weight_prep", pairs)


def main():
    which = sys.argv[1:] or [
        "attention_input",
        "wq_b",
        "mla_kv",
        "attention_output",
        "router",
        "feed_forward",
        "fsdp_copy_in",
        "expert_weight_prep",
    ]
    ok = True
    if {"attention_input", "wq_b", "mla_kv", "attention_output"} & set(which):
        attn_norm, attn, ffn_norm = build_attention()
        if "attention_input" in which:
            ok &= test_attention_input(attn_norm, attn)
        if "wq_b" in which:
            ok &= test_wq_b(attn)
        if "mla_kv" in which:
            ok &= test_mla_kv(attn)
        if "attention_output" in which:
            ok &= test_attention_output(attn, ffn_norm)
    if "router" in which:
        ok &= test_router()
    if "feed_forward" in which:
        ok &= test_feed_forward(3)
        ok &= test_feed_forward(0)
    if "fsdp_copy_in" in which:
        ok &= test_fsdp_copy_in()
    if "expert_weight_prep" in which:
        ok &= test_expert_weight_prep()
    print(f"kernel calls: {dict(gk.CALLS)}")
    print("ALL EXACT" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
