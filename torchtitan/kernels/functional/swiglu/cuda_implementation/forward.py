# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import math
from functools import partial

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import Float32, const_expr, range_constexpr

from ....math import get_powers_of_2
from ....autotuner import AutotuneConfig, autotune
from ....custom_op import xma_op
from ....cute_dsl_utils import (
    ElementwiseCUDAKernel,
    ElementwisePackedCUDAKernel,
    get_compiled_elementwise_cuda_kernel,
    sigmoid,
)


class _SwiGLUForwardCUDAKernel(ElementwiseCUDAKernel):
    @cute.jit
    def compute(self, xs: list[cute.TensorSSA]) -> list[cute.TensorSSA]:
        g, u = xs

        dtype = g.dtype
        g = g.to(Float32)
        y = u * g * sigmoid(g)

        return (y.to(dtype),)


class _SwigluPackedForwardCUDAKernel(ElementwisePackedCUDAKernel):
    @cute.jit
    def compute(self, xs_1: list[cute.Tensor], xs_2: list[cute.Tensor]) -> tuple[list[cute.Tensor], list[cute.Tensor]]:
        assert const_expr(len(xs_1) == 0)
        assert const_expr(len(xs_2) == 1)

        x = xs_2[0]
        dtype = x.dtype

        N = cute.size(x.shape)
        H = N >> 1

        y = cute.make_rmem_tensor(H, Float32)

        for j in range_constexpr(H):
            h = j << 1
            g = x[h].to(Float32)
            u = x[h + 1]

            y[j] = u * g * sigmoid(g)

        return [y.load().to(dtype)], []


def _get_autotune_configs() -> list[AutotuneConfig]:
    configs = []
    for BLOCK_SIZE in get_powers_of_2(128, 1024):
        for M in get_powers_of_2(1, 16):
            configs.append(AutotuneConfig({"BLOCK_SIZE": BLOCK_SIZE, "M": M}))

    return configs


@xma_op(mutates_args={"y"})
@autotune(configs=_get_autotune_configs(), triggers={"g.size(1)", "g.dtype"})
def _swiglu_forward_cuda(g: torch.Tensor, u: torch.Tensor, y: torch.Tensor, BLOCK_SIZE: int, M: int) -> None:
    N = g.size(1)
    div = math.gcd(16 // g.dtype.itemsize, N)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    kernel = get_compiled_elementwise_cuda_kernel(
        caller_op=_swiglu_forward_cuda,
        key=(g.dtype, div, BLOCK_SIZE, M),
        kernel_class=partial(_SwiGLUForwardCUDAKernel, BLOCK_SIZE=BLOCK_SIZE, M=M),
        example_tensors_list=([g, u], [y]),
        divisibility_list_list=([div, div], [div]),
        stream=stream,
    )

    kernel([g, u], [y], stream)


@xma_op(mutates_args={"y"})
@autotune(configs=_get_autotune_configs(), triggers={"x.size(1)", "x.dtype"})
def _swiglu_packed_forward_cuda(x: torch.Tensor, y: torch.Tensor, BLOCK_SIZE: int, M: int) -> None:
    N = x.size(1)
    div_x = math.gcd(16 // x.dtype.itemsize, N)
    div_y = math.gcd(16 // x.dtype.itemsize, N >> 1)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    kernel = get_compiled_elementwise_cuda_kernel(
        caller_op=_swiglu_packed_forward_cuda,
        key=(x.dtype, div_x, div_y, BLOCK_SIZE, M),
        kernel_class=partial(_SwigluPackedForwardCUDAKernel, BLOCK_SIZE=BLOCK_SIZE, M=M),
        example_tensors_list=([], [x], [y], []),
        divisibility_list_list=([], [div_x], [div_y], []),
        stream=stream,
    )

    kernel([], [x], [y], [], stream)
