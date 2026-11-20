# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused GLM selector: one CTA per local query, bounded running Top-K.

Q/K/N/H/S denote local queries, global keys, heads, head width and selected keys.
Key scores are streamed through registers, never stored as a global Q-by-K map.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _select(
    Q, K, W, Bounds, Out,
    Q0: tl.constexpr, Q1: tl.constexpr, Q2: tl.constexpr,
    K0: tl.constexpr, K1: tl.constexpr,
    W0: tl.constexpr, W1: tl.constexpr,
    N: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
    BN: tl.constexpr, BH: tl.constexpr, KEEP: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    begin = tl.load(Bounds + row * 2).to(tl.int64)
    end = tl.load(Bounds + row * 2 + 1).to(tl.int64)
    slots = tl.arange(0, KEEP)
    if end - begin <= S:
        ids = tl.where(slots < end - begin, begin + slots, -1)
        tl.store(Out + row * S + slots, ids.to(tl.int32), slots < S)
        return
    heads = tl.arange(0, BN)
    channels = tl.arange(0, BH)
    q_NH = tl.load(
        Q + row * Q0 + heads[:, None].to(tl.int64) * Q1
        + channels[None, :].to(tl.int64) * Q2,
        (heads[:, None] < N) & (channels[None, :] < H), 0,
    )
    weights_N = tl.load(W + row * W0 + heads.to(tl.int64) * W1, heads < N, 0)
    best = tl.full((KEEP,), -9223372036854775808, tl.int64)
    tile_B = tl.arange(0, 128)
    for start in tl.range(begin, end, 128):
        ids_B = start + tile_B
        k_HB = tl.load(
            K + ids_B[None, :] * K0 + channels[:, None].to(tl.int64) * K1,
            (ids_B[None, :] < end) & (channels[:, None] < H), 0,
        )
        dots_NB = tl.dot(q_NH, k_HB)
        scores_B = tl.sum(tl.maximum(dots_NB, 0) * weights_N[:, None], axis=0)
        scores_B = scores_B * (N * H) ** -0.5
        scores = tl.gather(scores_B, slots % 128, axis=0)
        # Ordered FP32 high bits rank scores; low bits resolve ties by key ID.
        bits = tl.where(scores == 0, 0.0, scores).to(tl.int32, bitcast=True)
        ordinal = bits ^ ((bits >> 31) & 0x7FFFFFFF)
        ids = start + slots
        encoded = (ordinal.to(tl.int64) << 32) | (0xFFFFFFFF - ids)
        encoded = tl.where(
            (slots < 128) & (ids < end), encoded, -9223372036854775808
        )
        best = tl.topk(tl.join(best, encoded).reshape(2 * KEEP), KEEP)
    ids = (0xFFFFFFFF - (best & 0xFFFFFFFF)).to(tl.int32)
    ids = tl.where(best == -9223372036854775808, -1, ids)
    tl.store(Out + row * S + slots, ids, slots < S)


def launch(q, k, weights, bounds, topk):
    queries, heads, width = q.shape
    if torch.version.hip:
        raise ValueError("GLM fused local selector requires NVIDIA CUDA")
    if torch.cuda.get_device_capability(q.device)[0] < 8:
        raise ValueError("GLM fused local selector requires SM80 or newer")
    if not 1 <= heads <= 64 or width < 16 or width > 256 or width % 8:
        raise ValueError(
            "GLM fused selector requires 1<=heads<=64, width 16..256 divisible by 8"
        )
    if topk > 4096:
        raise ValueError("GLM fused selector supports topk <= 4096")
    output = torch.empty((queries, topk), dtype=torch.int32, device=q.device)
    if queries == 0 or topk == 0:
        return output
    with torch.cuda.device(q.device):
        _select[(queries,)](
            q, k, weights, bounds, output,
            *q.stride(), *k.stride(), *weights.stride(),
            heads, width, topk,
            max(16, triton.next_power_of_2(heads)),
            triton.next_power_of_2(width),
            max(128, triton.next_power_of_2(topk)),
            num_warps=8,
        )
    return output
