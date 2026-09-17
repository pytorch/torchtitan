# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import cutlass.cute as cute
from cutlass import Float32, Numeric, const_expr, range_constexpr
from cutlass.cute import TensorSSA


@cute.jit
def tanh(x: Numeric | TensorSSA, output_dtype: Numeric | None = None) -> Numeric | TensorSSA:
    if const_expr(output_dtype is None):
        output_dtype = x.dtype

    if const_expr(isinstance(x, TensorSSA)):
        y = cute.make_rmem_tensor(x.shape, Float32)
        y.store(x.to(Float32))

        for i in range_constexpr(cute.size(y.shape)):
            y[i] = cute.math.tanh(y[i], fastmath=True)

        y = y.load()
    else:
        y = cute.math.tanh(x.to(Float32), fastmath=True)
        y = Float32(y).to(output_dtype)

    return y


@cute.jit
def sigmoid(x: Numeric | TensorSSA, output_dtype: Numeric | None = None) -> Numeric | TensorSSA:
    if const_expr(output_dtype is None):
        output_dtype = x.dtype

    x = x.to(Float32)
    x = 0.5 * tanh(0.5 * x, output_dtype=Float32) + 0.5
    x = x.to(output_dtype)

    return x
