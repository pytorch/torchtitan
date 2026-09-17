# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....math import ceil_divide
from ....custom_op import xma_op
from ....triton_utils import elementwise_2d_kernel, sigmoid


@triton.jit
def _compute(g, u, dy):
    g = g.to(tl.float32)
    g_sigmoid = sigmoid(g)
    g_silu = g * g_sigmoid

    dg = dy * u * (g_sigmoid + g_silu * (1 - g_sigmoid))
    du = dy * g_silu

    return dg, du


@xma_op(mutates_args={"dg", "du"})
def _swiglu_backward_triton(
    g: torch.Tensor, u: torch.Tensor, dy: torch.Tensor, dg: torch.Tensor, du: torch.Tensor
) -> None:
    B, H = g.size()
    GRID = lambda meta: (ceil_divide(B, meta["BLOCK_SIZE_B"]), ceil_divide(H, meta["BLOCK_SIZE_H"]))

    elementwise_2d_kernel[GRID](
        x0_ptr=g,
        x0_stride=g.stride(),
        x1_ptr=u,
        x1_stride=u.stride(),
        x2_ptr=dy,
        x2_stride=dy.stride(),
        y0_ptr=dg,
        y0_stride=dg.stride(),
        y1_ptr=du,
        y1_stride=du.stride(),
        B=B,
        H=H,
        COMPUTE_FN=_compute,
    )
