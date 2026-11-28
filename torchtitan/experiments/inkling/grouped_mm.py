# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ragged grouped GEMM over Inkling's frozen checkpoint-format expert matrices.

Native grouped_mm requires the output dtype to match its inputs, but Inkling's
outer scales must be applied to the FP32 accumulator, before BF16 rounding.
Decode only the current MMA tile, including in the transposed (dX) operation.
R: routed rows, E: experts, I/O: projection input/output, P: packed columns.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_tile(W, S, row, col, mask, I: tl.constexpr, NVFP4: tl.constexpr):
    if NVFP4:
        packed = tl.load(W + row * (I // 2) + col // 2, mask, other=0)
        code = (packed >> ((col % 2) * 4)) & 15
        mag = code & 7
        bits = tl.where(mag < 2, mag * 48, 48 + mag * 4) | ((code & 8) << 4)
        value = bits.to(tl.uint8).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale = tl.load(S + row * (I // 16) + col // 16, mask, other=0.0).to(tl.float32)
    else:
        value = tl.load(W + row * I + col, mask, other=0.0).to(tl.float32)
        exp = tl.load(S + row * (I // 32) + col // 32, mask, other=0).to(tl.int32)
        scale = tl.maximum((exp << 23).to(tl.float32, bitcast=True), 2.0**-127)
    return (value * scale).to(tl.bfloat16)


@triton.jit
def _packed_grouped_mm(
    X,
    W,
    S,
    Edges,
    Multiplier,
    Y,
    O: tl.constexpr,
    I: tl.constexpr,
    NVFP4: tl.constexpr,
    DX: tl.constexpr,
    STRIDE_XR: tl.constexpr,
    STRIDE_XK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    expert = tl.program_id(1)
    start = tl.load(Edges + expert)
    end = tl.load(Edges + expert + 1)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    N: tl.constexpr = I if DX else O
    K: tl.constexpr = O if DX else I
    W += expert.to(tl.int64) * O * (I // (2 if NVFP4 else 1))
    S += expert.to(tl.int64) * O * (I // (16 if NVFP4 else 32))
    for first in range(start + BM * tl.program_id(2), end, BM * tl.num_programs(2)):
        m = first + tl.arange(0, BM)
        acc = tl.full((BM, BN), 0, tl.float32)
        for ki in range(tl.cdiv(K, BK)):
            k = ki * BK + tl.arange(0, BK)
            a = tl.load(
                X + m[:, None].to(tl.int64) * STRIDE_XR + k[None, :] * STRIDE_XK,
                (m[:, None] < end) & (k[None, :] < K),
                other=0,
            )
            if DX:
                b = _decode_tile(
                    W,
                    S,
                    k[:, None],
                    n[None, :],
                    (k[:, None] < K) & (n[None, :] < N),
                    I,
                    NVFP4,
                )
            else:
                b = tl.trans(
                    _decode_tile(
                        W,
                        S,
                        n[:, None],
                        k[None, :],
                        (n[:, None] < N) & (k[None, :] < K),
                        I,
                        NVFP4,
                    )
                )
            acc = tl.dot(a, b, acc)
        if NVFP4:
            acc *= tl.load(Multiplier + m, m < end, other=0)[:, None]
        tl.store(
            Y + m[:, None].to(tl.int64) * N + n[None, :],
            acc,
            (m[:, None] < end) & (n[None, :] < N),
        )


def packed_grouped_mm(
    x_RI: torch.Tensor,
    weight_EOP: torch.Tensor,
    scale_EOP: torch.Tensor,
    edges: torch.Tensor,
    multiplier_R: torch.Tensor | None,
    *,
    dx: bool,
) -> torch.Tensor:
    experts, out_dim, packed_dim = weight_EOP.shape
    nvfp4 = weight_EOP.dtype == torch.uint8
    in_dim = packed_dim * (2 if nvfp4 else 1)
    columns = in_dim if dx else out_dim
    result = x_RI.new_empty((x_RI.size(0), columns))
    # The third grid axis distributes long token runs without padding short
    # runs to the capacity of the busiest expert (or reading counts on CPU).
    grid = (
        triton.cdiv(columns, 64),
        experts,
        max(1, triton.cdiv(len(x_RI), 16 * experts)),
    )
    _packed_grouped_mm.run(
        x_RI,
        weight_EOP,
        scale_EOP,
        edges,
        multiplier_R,
        result,
        out_dim,
        in_dim,
        nvfp4,
        dx,
        *x_RI.stride(),
        16,
        64,
        32,
        grid=grid,
        warmup=False,
    )
    return result
