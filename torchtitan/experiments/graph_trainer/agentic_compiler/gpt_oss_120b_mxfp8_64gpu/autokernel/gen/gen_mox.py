# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs for mox's MoE gather/scatter problems (G-011, G-014, G-023, G-025, G-036).

make_inputs(spec, generator, sizes) detects the problem from the spec signature/inputs and
returns inputs in run() order, laid out exactly as the spec (expanded int64 indices keep
stride [1, 0]).

Index structure follows the model code (torchtitan/models/common/moe/token_dispatcher.py):
  perm = argsort(topk_expert_ids_TK.view(-1), stable=True)   (a permutation of T*K)
  tok  = perm // top_k                                          (each token exactly K times)
  permuted_indices (_permute) = a permutation of [0, R) built from (rank, expert) segments.

Seed families (seed % 100 // 10):
  0: training-like routing (T=16384, K=4, E=128, EP=16), randn values
  1: arbitrary valid indices: duplicates and empty destinations (randint), randn values
  2: training-like routing, special values (NaN payloads, +-inf, +-0, denormals, huge)
  3: heavy duplicates (few destinations, >= 32 duplicates each), wide-exponent values
  4: negative indices where the op wraps them (index_put / scatter_add), randn values
R defaults to 65536; --sym N0=.. (sum of N0..N15) sets R, N16..N31 set M (G-036).
"""

import torch

T, K, E, EP, D = 16384, 4, 128, 16, 2880


def _routing(g, dev):
    scores = torch.rand(T, E, generator=g, device=dev)
    ids = scores.topk(K, dim=-1).indices  # distinct experts per token
    perm = torch.argsort(ids.reshape(-1), stable=True)
    return perm


def _special(shape, g, dev, dtype):
    x = torch.randn(shape, generator=g, device=dev)
    u = torch.rand(shape, generator=g, device=dev)
    bits = torch.randint(0, 1 << 16, shape, generator=g, device=dev, dtype=torch.int32)
    if dtype == torch.bfloat16:
        rnd = (
            (bits.to(torch.int16)).view(torch.bfloat16).float()
        )  # random bit patterns incl NaN payloads
    else:
        rnd = (
            torch.randint(
                -(1 << 31),
                (1 << 31) - 1,
                shape,
                generator=g,
                device=dev,
                dtype=torch.int64,
            )
            .to(torch.int32)
            .view(torch.float32)
        )
    x = torch.where(u < 0.02, rnd, x)
    x = torch.where((u >= 0.02) & (u < 0.03), torch.full_like(x, float("inf")), x)
    x = torch.where((u >= 0.03) & (u < 0.04), torch.full_like(x, float("-inf")), x)
    x = torch.where((u >= 0.04) & (u < 0.06), torch.zeros_like(x), x)
    x = torch.where((u >= 0.06) & (u < 0.08), -torch.zeros_like(x), x)
    x = torch.where((u >= 0.08) & (u < 0.10), x * 1e-39, x)  # fp32 denormals
    x = torch.where((u >= 0.10) & (u < 0.12), x * 1e30, x)
    x = torch.where((u >= 0.12) & (u < 0.13), x * 3e38, x)
    if dtype == torch.bfloat16:
        x = x.to(torch.bfloat16)
        # bf16 denormals and NaN payload bit patterns directly
        b = x.view(torch.int16)
        b = torch.where((u >= 0.13) & (u < 0.15), bits.to(torch.int16) & 0x807F, b)
        b = torch.where((u >= 0.15) & (u < 0.16), (bits.to(torch.int16) | 0x7F81), b)
        return b.view(torch.bfloat16)
    return x.to(dtype)


def _vals(shape, g, dev, dtype, fam):
    if fam == 2:
        return _special(shape, g, dev, dtype)
    x = torch.randn(shape, generator=g, device=dev)
    if fam == 3:
        e = torch.randint(-60, 60, shape, generator=g, device=dev).float()
        x = x * torch.exp2(e)
    return x.to(dtype)


def _scores(shape, g, dev, fam):
    if fam == 2:
        return _special(shape, g, dev, torch.float32)
    s = torch.rand(shape, generator=g, device=dev)
    if fam == 3:
        s = s * torch.exp2(
            torch.randint(-30, 30, shape, generator=g, device=dev).float()
        )
    return s


def _expand_rows(idx, d):
    return idx.reshape(-1, 1).expand(-1, d)


def _segment_perm(r, g, dev):
    """permuted_indices as _permute builds it, for R rows over EP ranks x 8 local experts."""
    e = 8
    if r == 0:
        return torch.zeros(0, dtype=torch.int64, device=dev)
    w = torch.rand(EP * e, generator=g, device=dev)
    counts = (w / w.sum() * r).floor().long()
    counts[0] += r - counts.sum()
    t_mat = counts.view(EP, e)
    input_starts = (counts.cumsum(0) - counts).view(EP, e)
    seg_lens = t_mat.t().reshape(-1)
    input_starts = input_starts.t().reshape(-1)
    seg_ids = torch.arange(seg_lens.shape[0], device=dev).repeat_interleave(
        seg_lens, output_size=r
    )
    out_starts = seg_lens.cumsum(0) - seg_lens
    return input_starts[seg_ids] + torch.arange(r, device=dev) - out_starts[seg_ids]


def _size(expr, sizes, default=65536):
    if not isinstance(expr, str):
        return int(expr)
    if expr in sizes:
        return int(sizes[expr])
    if "__default__" not in sizes and not any(k.isidentifier() for k in sizes):
        return default
    import re

    names = re.findall(r"[A-Za-z_]\w*", expr)
    given = {k: int(v) for k, v in sizes.items() if k.isidentifier()}
    if any(
        n in given for n in names
    ):  # unspecified symbols of a partly given sum count as 0
        return int(eval(expr, {}, {n: given.get(n, 0) for n in names}))
    return int(sizes.get("__default__", default))


def _rows(spec, sizes):
    """R = the routed-row symbol (N0 + ... + N15); M = the G-036 output rows (N16 + ...)."""
    rexpr = next(
        x for d in spec["inputs"] for x in d.get("shape", []) if isinstance(x, str)
    )
    r = _size(rexpr, sizes)
    m = r
    v = spec["inputs"][1].get("value") if len(spec["inputs"]) > 1 else None
    if isinstance(v, str) and v != rexpr:
        m = _size(v, sizes, default=r)
    return r, m


def make_inputs(spec, g, sizes):
    dev = "cuda"
    seed = g.initial_seed()
    fam = (seed % 100) // 10
    sig = spec["signature"]
    ins = spec["inputs"]
    r, m = _rows(spec, sizes)
    n = 65536
    perm = _routing(g, dev)
    tok = perm // K
    if fam == 1:
        perm = torch.randint(0, n, (n,), generator=g, device=dev)
        tok = torch.randint(0, T, (n,), generator=g, device=dev)
    elif fam == 3:
        perm = torch.randint(0, 64, (n,), generator=g, device=dev) * 997
        tok = torch.randint(0, 16, (n,), generator=g, device=dev) * 1021
    elif fam == 4:
        perm = torch.where(torch.rand(n, generator=g, device=dev) < 0.5, perm - n, perm)
        tok_neg = torch.where(
            torch.rand(n, generator=g, device=dev) < 0.5, tok - T, tok
        )

    shapes = [d.get("shape") for d in ins]
    if shapes[0] == [16384, 2880] and len(ins) == 5:
        # G-011 combine backward: run(in0 [T,D] bf16, in1 [N,D] i64 (stride 1,0), in2 [R,D] bf16,
        #                            in3 [N,1] f32, in4 [N] i64)
        in0 = _vals((T, D), g, dev, torch.bfloat16, fam)
        in1 = _expand_rows(tok.clone(), D)  # gather indices: must be valid (no wrap)
        in2 = _vals((r, D), g, dev, torch.bfloat16, fam)
        in3 = _scores((n, 1), g, dev, fam)
        in4 = perm.clone()
        return [in0, in1, in2, in3, in4]
    if shapes[0] == [16384, 4]:
        # G-014 combine forward: run(in0 [T,K] f32, in1 [N] i64, in2 [R,D] bf16, in3 [T,D] bf16,
        #                           in4 [N,D] i64 (stride 1,0))
        in0 = _scores((T, K), g, dev, fam)
        in1 = perm.clone()
        in2 = _vals((r, D), g, dev, torch.bfloat16, fam)
        in3 = _vals((T, D), g, dev, torch.bfloat16, 0)
        in4 = _expand_rows((tok_neg if fam == 4 else tok).clone(), D)
        return [in0, in1, in2, in3, in4]
    if len(ins) == 3 and ins[1].get("shape") == [65536]:
        # G-023: run(in0 [R,D] bf16, in1 [N] i64, in2 [T,D] bf16)
        in0 = _vals((r, D), g, dev, torch.bfloat16, fam)
        in1 = (tok_neg if fam == 4 else tok).clone()
        in2 = _vals((T, D), g, dev, torch.bfloat16, fam)
        return [in0, in1, in2]
    if len(ins) == 3 and "value" in ins[1]:
        accumulate = "N16" not in str(ins[1]["value"])
        in0 = _vals((r, D), g, dev, torch.bfloat16, fam)
        if accumulate:
            # G-025: run(in0 [R,D] bf16, R, in2 [R] i64) -> zeros [R,D] + index_put(accumulate)
            if fam == 1:
                idx = torch.randint(0, max(r, 1), (r,), generator=g, device=dev)
            elif fam == 3:
                idx = torch.randint(
                    0, max(min(r, 16), 1), (r,), generator=g, device=dev
                )
            else:
                idx = (
                    _segment_perm(r, g, dev)
                    if seed % 2 == 0
                    else torch.randperm(r, generator=g, device=dev)
                )
                if fam == 4:
                    idx = torch.where(
                        torch.rand(r, generator=g, device=dev) < 0.5, idx - r, idx
                    )
            return [in0, r, idx]
        # G-036: run(in0 [R,D] bf16, M, in2 [R] i64) -> new_empty [M,D] + index_put_ (no accumulate).
        # Every destination row must be written (new_empty), so M <= R with full coverage.
        if fam in (1, 3) and r > 1:
            mm = max(r // 2, 1) if fam == 1 else max(min(r, 16), 1)
            idx = torch.cat(
                [
                    torch.randperm(mm, generator=g, device=dev),
                    torch.randint(0, mm, (r - mm,), generator=g, device=dev),
                ]
            )
            idx = idx[torch.randperm(r, generator=g, device=dev)]
            return [in0, mm, idx]
        idx = (
            _segment_perm(r, g, dev)
            if seed % 2 == 0
            else torch.randperm(r, generator=g, device=dev)
        )
        if fam == 4:
            idx = torch.where(
                torch.rand(r, generator=g, device=dev) < 0.5, idx - r, idx
            )
        return [in0, r, idx]
    raise ValueError(f"unknown problem {sig} {shapes}")
