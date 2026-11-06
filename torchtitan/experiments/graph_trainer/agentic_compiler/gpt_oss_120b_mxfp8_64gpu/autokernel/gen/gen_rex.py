# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs for rex's problems: G-031 / G-039 (expert-bias expand), G-048 (mul), G-041 (router
backward), G-046 (aux-loss forward). make_inputs(spec, generator, sizes) detects the problem
from the spec and returns inputs in run() order, laid out as the spec.

R (routed rows, the sum N0 + ... + N15) is the sum of the --sym values given (missing symbols
count 0), else 65536 (training).

Seed families (seed % 100 // 10):
  0: training-like (seed 0 is what harness --time measures)
  1: edge structures (empty experts, all rows in one expert / in the tail, ...), randn values
  2: training-like structure, special values (NaN payloads, +-inf, +-0, denormals, huge)
  3: training-like structure, uniformly random bit patterns
  4: G-041/G-046: heavy routing skew; G-031/G-039: offsets at tile/row boundaries
  5: G-031 (GA capture): training-like, bias at element offset seed % 8 inside a larger buffer
"""

import torch

E = 8  # local experts (G-031 / G-039)
T, K, NE = 16384, 4, 128  # tokens, top-k, routed experts (G-041 / G-046)


def _rows(sizes):
    ns = {k: v for k, v in sizes.items() if k.startswith("N") and k[1:].isdigit()}
    if not ns:
        return 65536
    return sum(ns.values())


def _special(shape, g, dev, dtype):
    x = torch.randn(shape, generator=g, device=dev)
    u = torch.rand(shape, generator=g, device=dev)
    x = torch.where(u < 0.02, torch.full_like(x, float("inf")), x)
    x = torch.where((u >= 0.02) & (u < 0.04), torch.full_like(x, float("-inf")), x)
    x = torch.where((u >= 0.04) & (u < 0.07), torch.zeros_like(x), x)
    x = torch.where((u >= 0.07) & (u < 0.10), -torch.zeros_like(x), x)
    x = torch.where((u >= 0.10) & (u < 0.12), x * 1e-39, x)
    x = torch.where((u >= 0.12) & (u < 0.14), x * 3e38, x)
    x = torch.where((u >= 0.14) & (u < 0.16), x * 1e-30, x)
    x = x.to(dtype)
    bits_dtype = {torch.bfloat16: torch.int16, torch.float32: torch.int32}[dtype]
    nb = 16 if dtype == torch.bfloat16 else 32
    bits = torch.randint(
        -(1 << (nb - 1)),
        1 << (nb - 1),
        shape,
        generator=g,
        device=dev,
        dtype=torch.int64,
    ).to(bits_dtype)
    b = x.view(bits_dtype)
    sign = torch.tensor(1 << (nb - 1), dtype=torch.int64).to(bits_dtype).item()
    if dtype == torch.bfloat16:
        den, nanbits = 0x007F, 0x7F81
    else:
        den, nanbits = 0x007FFFFF, 0x7F800001
    b = torch.where((u >= 0.16) & (u < 0.19), bits & (den | sign), b)  # denormals
    b = torch.where(
        (u >= 0.19) & (u < 0.21), bits | nanbits, b
    )  # NaN payloads, both signs
    b = torch.where((u >= 0.21) & (u < 0.23), bits, b)  # random patterns
    return b.view(dtype)


def _bits(shape, g, dev, dtype):
    nb = 16 if dtype == torch.bfloat16 else 32
    bits_dtype = torch.int16 if nb == 16 else torch.int32
    return (
        torch.randint(
            -(1 << (nb - 1)),
            1 << (nb - 1),
            shape,
            generator=g,
            device=dev,
            dtype=torch.int64,
        )
        .to(bits_dtype)
        .view(dtype)
    )


def _vals(shape, g, dev, dtype, fam):
    if fam == 2:
        return _special(shape, g, dev, dtype)
    if fam == 3:
        return _bits(shape, g, dev, dtype)
    return torch.randn(shape, generator=g, device=dev).to(dtype)


def _offsets(r, g, dev, fam, seed):
    """int32 [E] cumulative row offsets with 0 <= o[0] <= ... <= o[7] <= r (valid repeat counts)."""
    if fam == 1:
        k = seed % 10
        if k == 0:
            o = torch.zeros(E)  # every row in the tail
        elif k == 1:
            o = torch.full((E,), float(r))  # all rows in expert 0
        elif k == 2:
            o = torch.full((E,), float(r))
            o[:7] = 0  # all rows in expert 7
        elif k == 3:
            o = torch.tensor(
                [0, 0, r // 3, r // 3, r // 3, r // 2, r // 2, r // 2],
                dtype=torch.float,
            )  # empties
        elif k == 4:
            o = torch.full((E,), float(r // 2))  # half expert 0, half tail
        else:
            o = torch.sort(
                torch.randint(0, r + 1, (E,), generator=g, device=dev).float().cpu()
            ).values
        return o.to(torch.int32).to(dev)
    if fam == 4:
        # boundaries at multiples / neighbours of the 1024-vector tile and of 8
        cand = torch.tensor(
            [0, 1, 7, 8, 9, 31, 32, 33, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025]
        )
        pick = cand[torch.randint(0, len(cand), (E,), generator=g, device=dev).cpu()]
        o = torch.sort(
            torch.clamp(
                pick
                + torch.randint(0, max(r, 1), (1,), generator=g, device=dev).cpu()
                * (seed % 2),
                max=r,
            )
        ).values
        return o.to(torch.int32).to(dev)
    # training-like: tokens per expert from a skewed distribution, some tail padding on odd seeds
    tail = (
        int(torch.randint(0, min(r, 256) + 1, (), generator=g, device=dev))
        if seed % 2
        else 0
    )
    w = torch.rand(E, generator=g, device=dev) + 0.2
    c = torch.floor(w / w.sum() * (r - tail)).long()
    c[0] += (r - tail) - c.sum()
    return torch.cumsum(c, 0).to(torch.int32)


def _topk(g, dev, fam):
    scores = torch.rand(T, NE, generator=g, device=dev)
    if fam == 4:
        scores = scores * torch.exp2(
            torch.linspace(0, 30, NE, device=dev)
        )  # most tokens -> few experts
    return scores.topk(K, dim=-1)


def make_inputs(spec, g, sizes):
    dev = "cuda"
    seed = g.initial_seed()
    fam = seed % 100 // 10
    sig = spec["signature"]
    ins = spec["inputs"]
    if len(ins) == 4 and ins[1].get("shape", [None, None])[-1] == 5760:  # G-031
        r = _rows(sizes)
        bias = _vals((E * 5760,), g, dev, torch.bfloat16, fam)
        x = _vals((r, 5760), g, dev, torch.bfloat16, fam)
        return [bias, x, r, _offsets(r, g, dev, fam, seed)]
    if len(ins) == 4 and ins[2].get("shape") == [
        8,
        2,
        2880,
    ]:  # G-031 (GA capture: bias [8, 2, 2880] third)
        r = _rows(sizes)
        bias = _vals((E, 2, 2880), g, dev, torch.bfloat16, fam if fam != 5 else 0)
        if fam == 5:
            # bias as a view into a bucket buffer at element offset seed % 8 (2-byte aligned at worst)
            off = seed % 8
            buf = torch.randn(E * 5760 + 64, generator=g, device=dev).to(torch.bfloat16)
            view = buf[off : off + E * 5760].view(E, 2, 2880)
            view.copy_(bias)
            bias = view
        x = _vals((r, 5760), g, dev, torch.bfloat16, fam if fam != 5 else 0)
        return [x, r, bias, _offsets(r, g, dev, fam if fam != 5 else 0, seed)]
    if len(ins) == 4 and ins[2].get("shape") == [8, 2880]:  # G-039
        r = _rows(sizes)
        bias = _vals((E, 2880), g, dev, torch.bfloat16, fam)
        x = _vals((r, 2880), g, dev, torch.bfloat16, fam)
        return [_offsets(r, g, dev, fam, seed), r, bias, x]
    if len(ins) == 2:  # G-048
        r = _rows(sizes)
        return [
            _vals((r, 2880), g, dev, torch.bfloat16, fam),
            _vals((r, 2880), g, dev, torch.bfloat16, fam),
        ]
    if len(ins) == 6:  # G-041 router backward
        vals, ids = _topk(g, dev, fam)
        logits = torch.randn(T, NE, generator=g, device=dev) * 2
        sm = torch.softmax(logits, -1)
        top = sm.gather(1, ids)
        denom = top.sum(-1, keepdim=True)
        if fam == 2:
            in0 = _special((T, K), g, dev, torch.float32)
            in1 = _special((T, K), g, dev, torch.float32)
            in2 = _special((T, 1), g, dev, torch.float32)
            in4 = _special((T, NE), g, dev, torch.float32)
            in5 = _special((T, NE), g, dev, torch.float32)
        elif fam == 3:
            in0, in1, in2, in4, in5 = (
                _bits(s, g, dev, torch.float32)
                for s in ((T, K), (T, K), (T, 1), (T, NE), (T, NE))
            )
        else:
            in0 = torch.randn(T, K, generator=g, device=dev) * 1e-3
            in1 = top.clone()
            in2 = denom
            in4 = torch.randn(T, NE, generator=g, device=dev) * 1e-4
            in5 = sm
        return [
            in0.contiguous(),
            in1.contiguous(),
            in2.contiguous(),
            ids.contiguous(),
            in4.contiguous(),
            in5.contiguous(),
        ]
    if len(ins) == 3:  # G-046 aux-loss forward
        vals, ids = _topk(g, dev, fam)
        in0 = torch.randn(T, NE, generator=g, device=dev)
        if fam == 1:
            k = seed % 10
            p = [0.0, 1.0, 0.5, 0.01, 0.99][k % 5]
        else:
            p = 0.1
        in2 = torch.rand(T, generator=g, device=dev) < p
        if fam == 1 and seed % 10 >= 5:
            # duplicate experts within a token (scatter_ of True is idempotent)
            ids = torch.randint(0, NE, (T, K), generator=g, device=dev)
        return [in0, ids.contiguous(), in2]
    raise ValueError(f"unknown problem: {sig}")
