# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import triton
import triton.language as tl

from ..math import get_powers_of_2


def get_elementwise_2d_configs() -> list[triton.Config]:
    configs = []
    for BLOCK_SIZE_B in get_powers_of_2(16, 64):
        for BLOCK_SIZE_H in get_powers_of_2(16, 64):
            for num_warps in get_powers_of_2(4, 8):
                configs.append(
                    triton.Config({"BLOCK_SIZE_B": BLOCK_SIZE_B, "BLOCK_SIZE_H": BLOCK_SIZE_H}, num_warps=num_warps)
                )
    return configs


@triton.autotune(configs=get_elementwise_2d_configs(), key=[])
@triton.jit
def elementwise_2d_kernel(
    x0_ptr,
    x0_stride,
    x1_ptr,
    x1_stride,
    x2_ptr,
    x2_stride,
    y0_ptr,
    y0_stride,
    y1_ptr,
    y1_stride,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    COMPUTE_FN: tl.constexpr,
):
    BLOCK_ID_B = tl.program_id(0)
    BLOCK_ID_H = tl.program_id(1)

    BLOCK_B = BLOCK_ID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    BLOCK_H = BLOCK_ID_H * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    MASK = (BLOCK_B < B)[:, None] & (BLOCK_H < H)[None, :]

    x0 = tl.load(x0_ptr + BLOCK_B[:, None] * x0_stride[0] + BLOCK_H[None, :] * x0_stride[1], mask=MASK)

    if x1_ptr is not None:
        x1 = tl.load(x1_ptr + BLOCK_B[:, None] * x1_stride[0] + BLOCK_H[None, :] * x1_stride[1], mask=MASK)

    if x2_ptr is not None:
        assert x1_ptr is not None
        x2 = tl.load(x2_ptr + BLOCK_B[:, None] * x2_stride[0] + BLOCK_H[None, :] * x2_stride[1], mask=MASK)

    if x1_ptr is None:
        y = COMPUTE_FN(x0)
    elif x2_ptr is None:
        y = COMPUTE_FN(x0, x1)
    else:
        y = COMPUTE_FN(x0, x1, x2)

    if y1_ptr is None:
        y0 = y
    else:
        y0, y1 = y

    tl.store(y0_ptr + BLOCK_B[:, None] * y0_stride[0] + BLOCK_H[None, :] * y0_stride[1], y0, mask=MASK)

    if y1_ptr is not None:
        tl.store(y1_ptr + BLOCK_B[:, None] * y1_stride[0] + BLOCK_H[None, :] * y1_stride[1], y1, mask=MASK)
