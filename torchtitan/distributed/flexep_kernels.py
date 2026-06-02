# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_peer_kernel(
    src,
    dst,
    splits,
    PEER: tl.constexpr,
    RANK: tl.constexpr,
    EP_SIZE: tl.constexpr,
    NUM_COLS: tl.constexpr,
    SRC_STRIDE: tl.constexpr,
    DST_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    src_start = tl.full((), 0, tl.int64)
    for dst_rank in tl.static_range(0, EP_SIZE):
        if dst_rank < PEER:
            src_start += tl.load(splits + RANK * EP_SIZE + dst_rank)

    dst_start = tl.full((), 0, tl.int64)
    for src_rank in tl.static_range(0, EP_SIZE):
        if src_rank < RANK:
            dst_start += tl.load(splits + src_rank * EP_SIZE + PEER)

    rows = tl.load(splits + RANK * EP_SIZE + PEER)
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < rows * NUM_COLS
    row = offsets // NUM_COLS
    col = offsets - row * NUM_COLS

    values = tl.load(
        src + (src_start + row) * SRC_STRIDE + col,
        mask=mask,
    )
    tl.store(
        dst + (dst_start + row) * DST_STRIDE + col,
        values,
        mask=mask,
    )


def copy_peer(
    src: torch.Tensor,
    dst: torch.Tensor,
    splits: torch.Tensor,
    *,
    peer: int,
    rank: int,
    ep_size: int,
    num_rows: int,
    num_cols: int,
) -> None:
    if src.ndim != 2 or dst.ndim != 2:
        raise ValueError("copy_peer expects 2D source and destination tensors.")
    if num_rows < 0:
        raise ValueError(f"num_rows must be non-negative, got {num_rows}.")
    if num_cols <= 0:
        raise ValueError(f"num_cols must be positive, got {num_cols}.")

    block_size = 1024
    grid = (triton.cdiv(num_rows * num_cols, block_size),)
    _copy_peer_kernel[grid](
        src,
        dst,
        splits,
        PEER=peer,
        RANK=rank,
        EP_SIZE=ep_size,
        NUM_COLS=num_cols,
        SRC_STRIDE=src.stride(0),
        DST_STRIDE=dst.stride(0),
        BLOCK_SIZE=block_size,
    )
