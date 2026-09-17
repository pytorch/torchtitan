# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ....custom_op import ctx_save_for_backward, xma_op
from ....jit import cpp_jit


@xma_op(mutates_args={"y"})
@cpp_jit(is_mps=True)
def _swiglu_forward_mps(g: torch.Tensor, u: torch.Tensor, y: torch.Tensor) -> None: ...


@xma_op(mutates_args={"dg", "du"})
@cpp_jit(is_mps=True)
def _swiglu_backward_mps(
    g: torch.Tensor, u: torch.Tensor, dy: torch.Tensor, dg: torch.Tensor, du: torch.Tensor
) -> None: ...


class _SwigluMPS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        ctx_save_for_backward(ctx, g, u)

        y = torch.empty_like(g, memory_format=torch.contiguous_format)
        _swiglu_forward_mps(g=g, u=u, y=y)

        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g, u = ctx.saved_tensors

        dg = torch.empty_like(g, memory_format=torch.contiguous_format)
        du = torch.empty_like(u, memory_format=torch.contiguous_format)
        _swiglu_backward_mps(g=g, u=u, dy=dy, dg=dg, du=du)

        return dg, du
