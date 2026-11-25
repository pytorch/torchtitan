# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Minimal pure-torch repros of two places where whole-graph Inductor is slower than a region boundary.

Found in a DeepSeek-V3 GraphTrainer study (see ../README.md, observations), DSv3 671B
shapes at one EP8 rank (R = 4096 tokens x top-8 routed rows, F = 2048, D = 7168, 32 local experts).

1. stack_bwd: ``gate, up = gate_up_R2F.unbind(-2); silu(gate) * up`` (GroupedExperts). The backward
   of unbind is a 2-way stack that Inductor lowers as a masked pointwise cat fused with the SwiGLU
   backward: every output element evaluates both halves with predicated loads, so grad and gate are
   read twice. Variants: default; ConcatKernel lowering (max_pointwise_cat_inputs=0); the
   activation compiled alone with the unbind/stack left to eager (what a local region does).
2. combine_inline: per-token weighted sum of K gathered rows whose weights come from the router
   (sigmoid -> gather -> normalize -> scale). Compiled as one graph, Inductor inlines the score chain
   into the [T, D] loop and recomputes it per element (K sigmoids, divides and dependent index loads
   per element). Variant: scores computed in a separate compiled graph and passed in (the region
   boundary of the moe_dispatch_combine region in
   pytorch/torchtitan#5157).

Example:
    python obs_whole_graph_vs_region.py
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import kernel_stats  # noqa: E402


def timed(name, fn, ideal_gb=None):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    _, ms, per, _ = kernel_stats(fn, steps=5)
    top = max(per.items(), key=lambda kv: kv[1])
    ideal = (
        f"  ideal {ideal_gb / 7.1:.3f} ms ({ideal_gb:.2f} GB at 7.1 TB/s)"
        if ideal_gb
        else ""
    )
    print(
        f"{name:52s} {ms:8.3f} ms   top kernel {top[1]:.3f} ms {top[0][:70]}{ideal}",
        flush=True,
    )


def stack_bwd():
    R, Fh = 32768, 2048
    torch.manual_seed(0)
    gate_up = torch.randn(
        R, 2, Fh, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    grad = torch.randn(R, Fh, device="cuda", dtype=torch.bfloat16)

    def swiglu_of_packed(gate_up_R2F):
        gate, up = gate_up_R2F.unbind(dim=-2)
        return F.silu(gate) * up

    def silu_mul(gate, up):
        return F.silu(gate) * up

    # Each variant compiles after torch._dynamo.reset(): Dynamo caches compiled code on the function's
    # code object, so a second torch.compile of the same function would reuse the first variant.
    # The SwiGLU-backward inputs to the stack are multi-op pointwise, so the knob that matters is
    # max_complex_pointwise_cat_inputs (the GraphTrainer runs set both).
    concat_options = {
        "max_pointwise_cat_inputs": 0,
        "max_complex_pointwise_cat_inputs": 0,
    }

    def step(fn):
        def run():
            gate_up.grad = None
            fn(gate_up).backward(grad)

        return run

    def region_style():
        gate_up.grad = None
        gate, up = gate_up.unbind(dim=-2)
        compiled_act(gate, up).backward(grad)

    ref_out = swiglu_of_packed(gate_up)
    ref_out.backward(grad)
    ref_grad = gate_up.grad.clone()
    # fwd: read R*2F, write R*F; bwd: read grad + gate_up, write grad_gate_up (bf16)
    gb = (
        R * 2 * Fh * 2 + R * Fh * 2 + R * Fh * 2 + R * 2 * Fh * 2 + R * 2 * Fh * 2
    ) / 1e9
    timed("stack_bwd eager (fwd+bwd)", step(swiglu_of_packed), gb)
    for name, options in (
        ("default cat lowering", None),
        ("ConcatKernel (both cat knobs = 0)", concat_options),
    ):
        torch._dynamo.reset()
        fn = step(torch.compile(swiglu_of_packed, fullgraph=True, options=options))
        fn()
        assert torch.allclose(
            gate_up.grad.float(), ref_grad.float(), rtol=2e-2, atol=2e-2
        ), name
        timed(f"stack_bwd compiled, {name}", fn, gb)
    torch._dynamo.reset()
    compiled_act = torch.compile(silu_mul, fullgraph=True)
    region_style()
    assert torch.allclose(
        gate_up.grad.float(), ref_grad.float(), rtol=2e-2, atol=2e-2
    ), "region"
    timed("stack_bwd activation compiled, unbind/stack eager", region_style, gb)


def combine_inline():
    T, K, D, E = 4096, 8, 7168, 32
    torch.manual_seed(0)
    logits = torch.randn(T, E, device="cuda", dtype=torch.float32)
    expert_ids = torch.topk(logits, K, dim=-1).indices
    # Routed rows in expert order; token_order[t*K + k] = row of token t's k-th choice.
    order = torch.argsort(expert_ids.flatten(), stable=True)
    token_order = torch.empty_like(order)
    token_order[order] = torch.arange(T * K, device="cuda")
    routed = torch.randn(T * K, D, device="cuda", dtype=torch.bfloat16)

    def scores_of(logits, expert_ids):
        scores = torch.sigmoid(logits).gather(-1, expert_ids)
        return scores / (scores.sum(-1, keepdim=True) + 1e-20) * 2.5

    def weighted_sum(routed, token_order, scores):
        rows = routed[token_order].view(T, K, D)
        out = rows[:, 0].float() * scores[:, :1]
        for k in range(1, K):
            out = out + rows[:, k].float() * scores[:, k : k + 1]
        return out.to(torch.bfloat16)

    def fused(logits, expert_ids, routed, token_order):
        return weighted_sum(routed, token_order, scores_of(logits, expert_ids))

    compiled_fused = torch.compile(fused, fullgraph=True)
    compiled_scores = torch.compile(scores_of, fullgraph=True)
    compiled_sum = torch.compile(weighted_sum, fullgraph=True)

    def boundary():
        return compiled_sum(routed, token_order, compiled_scores(logits, expert_ids))

    ref = fused(logits, expert_ids, routed, token_order).float()
    for name, out in (
        ("fused", compiled_fused(logits, expert_ids, routed, token_order)),
        ("boundary", boundary()),
    ):
        assert torch.allclose(out.float(), ref, rtol=2e-2, atol=2e-2), name
    gb = (T * K * D * 2 + T * D * 2) / 1e9
    timed(
        "combine eager (fwd)",
        lambda: fused(logits, expert_ids, routed, token_order),
        gb,
    )
    timed(
        "combine compiled as one graph (scores inlined)",
        lambda: compiled_fused(logits, expert_ids, routed, token_order),
        gb,
    )
    timed("combine scores compiled separately, passed in", boundary, gb)


def kernel_names(fn):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    _, _, per, _ = kernel_stats(fn, steps=1)
    return sorted(k for k in per if k.startswith("triton_"))


def options_reach_backward():
    """Does torch.compile(options=...) reach the lazily compiled backward graph?"""
    import torch._inductor.config as inductor_config

    R, Fh = 32768, 2048
    gate_up = torch.randn(
        R, 2, Fh, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    grad = torch.randn(R, Fh, device="cuda", dtype=torch.bfloat16)

    def swiglu_of_packed(gate_up_R2F):
        gate, up = gate_up_R2F.unbind(dim=-2)
        return F.silu(gate) * up

    def run_with(fn):
        def run():
            gate_up.grad = None
            fn(gate_up).backward(grad)

        return run

    torch._dynamo.reset()
    with_options = torch.compile(
        swiglu_of_packed,
        fullgraph=True,
        options={"max_pointwise_cat_inputs": 0, "max_complex_pointwise_cat_inputs": 0},
    )
    print("options= (per compile):  ", kernel_names(run_with(with_options)), flush=True)
    torch._dynamo.reset()
    with inductor_config.patch(
        max_pointwise_cat_inputs=0, max_complex_pointwise_cat_inputs=0
    ):
        patched = torch.compile(swiglu_of_packed, fullgraph=True)
        print("global config patch:     ", kernel_names(run_with(patched)), flush=True)
    torch._dynamo.reset()
    print(
        "default:                 ",
        kernel_names(run_with(torch.compile(swiglu_of_packed, fullgraph=True))),
        flush=True,
    )


def token_sum_chain():
    """The dispatch backward as in the DSv3 graph: per-token sum of K gathered gradient rows times a
    per-row score that is itself computed from the router (row -> original (t, k) -> expert -> logit)."""
    T, K, D, E = 4096, 8, 7168, 32
    torch.manual_seed(0)
    logits = torch.randn(T, E, device="cuda", dtype=torch.float32)
    expert_ids = torch.topk(logits, K, dim=-1).indices
    orig_of_row = torch.argsort(
        expert_ids.flatten(), stable=True
    )  # row r holds flat (t, k) = orig_of_row[r]
    row_of_orig = torch.empty_like(orig_of_row)
    row_of_orig[orig_of_row] = torch.arange(T * K, device="cuda")
    grad_rows = torch.randn(T * K, D, device="cuda", dtype=torch.bfloat16)

    def row_scores(logits, expert_ids, orig_of_row):
        scores = torch.sigmoid(logits).gather(-1, expert_ids)
        scores = scores / (scores.sum(-1, keepdim=True) + 1e-20) * 2.5
        return scores.flatten()[
            orig_of_row
        ]  # [T*K] in row (expert) order, as the dispatch saves it

    def token_sum(grad_rows, row_of_orig, scores_rows):
        idx = row_of_orig.view(T, K)
        out = grad_rows[idx[:, 0]].float() * scores_rows[idx[:, 0]].unsqueeze(-1)
        for k in range(1, K):
            out = out + grad_rows[idx[:, k]].float() * scores_rows[idx[:, k]].unsqueeze(
                -1
            )
        return out.to(torch.bfloat16)

    def fused(logits, expert_ids, orig_of_row, grad_rows, row_of_orig):
        return token_sum(
            grad_rows, row_of_orig, row_scores(logits, expert_ids, orig_of_row)
        )

    torch._dynamo.reset()
    compiled_fused = torch.compile(fused, fullgraph=True)
    compiled_scores = torch.compile(row_scores, fullgraph=True)
    compiled_sum = torch.compile(token_sum, fullgraph=True)
    import torch._inductor.config as inductor_config

    with inductor_config.patch(assert_indirect_indexing=False):
        compiled_fused_noassert = torch.compile(fused, fullgraph=True, dynamic=False)
        args = (logits, expert_ids, orig_of_row, grad_rows, row_of_orig)
        compiled_fused_noassert(*args)  # compile inside the patch

    def boundary():
        return compiled_sum(
            grad_rows, row_of_orig, compiled_scores(logits, expert_ids, orig_of_row)
        )

    ref = fused(*args).float()
    for name, out in (
        ("fused", compiled_fused(*args)),
        ("boundary", boundary()),
        ("noassert", compiled_fused_noassert(*args)),
    ):
        assert torch.allclose(out.float(), ref, rtol=2e-2, atol=2e-2), name
    gb = (T * K * D * 2 + T * D * 2) / 1e9
    timed(
        "token_sum compiled as one graph (score chain inlined)",
        lambda: compiled_fused(*args),
        gb,
    )
    timed(
        "token_sum, one graph, assert_indirect_indexing=False",
        lambda: compiled_fused_noassert(*args),
        gb,
    )
    timed("token_sum, scores compiled separately, passed in", boundary, gb)


if __name__ == "__main__":
    print(torch.__version__, torch.cuda.get_device_name())
    which = sys.argv[1:] or [
        "stack_bwd",
        "combine_inline",
        "options_reach_backward",
        "token_sum_chain",
    ]
    for name in which:
        globals()[name]()
