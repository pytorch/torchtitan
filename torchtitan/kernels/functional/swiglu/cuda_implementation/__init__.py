# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ....math import divide_if_divisible
from ....custom_op import ctx_save_for_backward
from .backward import _swiglu_backward_cuda, _swiglu_packed_backward_cuda
from .forward import _swiglu_forward_cuda, _swiglu_packed_forward_cuda


class _SwigluCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        g = g.contiguous()
        u = u.contiguous()
        y = torch.empty_like(g, memory_format=torch.contiguous_format)

        ctx_save_for_backward(ctx, g, u)
        _swiglu_forward_cuda(g=g, u=u, y=y)

        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g, u = ctx.saved_tensors

        dy = dy.contiguous()
        dg = torch.empty_like(g, memory_format=torch.contiguous_format)
        du = torch.empty_like(u, memory_format=torch.contiguous_format)

        _swiglu_backward_cuda(g=g, u=u, dy=dy, dg=dg, du=du)

        return dg, du


class _SwigluPackedCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        y = torch.empty(*x.size()[:-1], divide_if_divisible(x.size(-1), 2), device=x.device, dtype=x.dtype)

        ctx_save_for_backward(ctx, x)
        _swiglu_packed_forward_cuda(x=x, y=y)

        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> torch.Tensor:
        x = ctx.saved_tensors[0]
        dy = dy.contiguous()
        dx = torch.empty_like(x, memory_format=torch.contiguous_format)

        _swiglu_packed_backward_cuda(x=x, dy=dy, dx=dx)

        return dx
