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
def _compute(g, u):
    g = g.to(tl.float32)
    return u * g * sigmoid(g)


@xma_op(mutates_args={"y"})
def _swiglu_forward_triton(g: torch.Tensor, u: torch.Tensor, y: torch.Tensor) -> None:
    B, H = g.size()
    GRID = lambda meta: (ceil_divide(B, meta["BLOCK_SIZE_B"]), ceil_divide(H, meta["BLOCK_SIZE_H"]))

    elementwise_2d_kernel[GRID](
        x0_ptr=g,
        x0_stride=g.stride(),
        x1_ptr=u,
        x1_stride=u.stride(),
        x2_ptr=None,
        x2_stride=None,
        y0_ptr=y,
        y0_stride=y.stride(),
        y1_ptr=None,
        y1_stride=None,
        B=B,
        H=H,
        COMPUTE_FN=_compute,
    )
