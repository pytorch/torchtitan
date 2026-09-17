# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ....custom_op import ctx_save_for_backward
from .backward import _swiglu_backward_triton
from .forward import _swiglu_forward_triton


class _SwigluTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        ctx_save_for_backward(ctx, g, u)

        y = torch.empty_like(g, memory_format=torch.contiguous_format)
        _swiglu_forward_triton(g=g, u=u, y=y)

        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g, u = ctx.saved_tensors

        dg = torch.empty_like(g, memory_format=torch.contiguous_format)
        du = torch.empty_like(u, memory_format=torch.contiguous_format)
        _swiglu_backward_triton(g=g, u=u, dy=dy, dg=dg, du=du)

        return dg, du
