# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import triton
import triton.language as tl


@triton.jit
def clamp(x, min_value, max_value):
    dtype = x.dtype

    x = max(min_value, x)
    x = min(max_value, x)
    x = x.to(dtype)

    return x


@triton.jit
def tanh(x, output_dtype: tl.constexpr = None):
    if output_dtype is None:
        output_dtype = x.dtype

    x = x.to(tl.float32)
    x = tl.inline_asm_elementwise("tanh.approx.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1)
    x = x.to(output_dtype)

    return x


@triton.jit
def tanh_backward(y):
    dtype = y.dtype

    y = y.to(tl.float32)
    y = 1 - y * y
    y = y.to(dtype)

    return y


@triton.jit
def sigmoid(x, output_dtype: tl.constexpr = None):
    if output_dtype is None:
        output_dtype = x.dtype

    x = x.to(tl.float32)
    x = tanh(0.5 * x, output_dtype=tl.float32)
    x = 0.5 * x + 0.5
    x = x.to(output_dtype)

    return x


@triton.jit
def sigmoid_backward(y):
    dtype = y.dtype

    y = y.to(tl.float32)
    y = y * (1 - y)
    y = y.to(dtype)

    return y


@triton.jit
def leaky_relu(x, negative_slope):
    return max(0, x) + negative_slope * min(0, x)


@triton.jit
def leaky_relu_backward(y, relu_negative_slope):
    return tl.where(y >= 0, 1, relu_negative_slope)


@triton.jit
def silu(x, output_dtype: tl.constexpr = None):
    if output_dtype is None:
        output_dtype = x.dtype

    x = x.to(tl.float32)
    x *= sigmoid(x, output_dtype=tl.float32)
    x = x.to(output_dtype)

    return x


@triton.jit
def silu_backward(x):
    dtype = x.dtype

    x = x.to(tl.float32)
    s = sigmoid(x, output_dtype=tl.float32)
    x = s + s * (x - x * s)
    x = x.to(dtype)

    return x
