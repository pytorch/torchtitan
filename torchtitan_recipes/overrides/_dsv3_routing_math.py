# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Scalar CuTeDSL building blocks preserving the native FP32 reduction trees."""

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda
from packaging.version import Version

if Version(cutlass.__version__) < Version("4.8.0"):
    raise ImportError("DSv3 routing kernels require nvidia-cutlass-dsl>=4.8.0.")
from cutlass import Float32, Uint32, Uint8
from cutlass.cute import arch
from cutlass.cute.runtime import make_ptr


def check_tensor(tensor, shape, dtype, device, name, *, alignment=1):
    if (
        tensor.shape != shape
        or tensor.dtype != dtype
        or tensor.device != device
        or not tensor.is_cuda
        or not tensor.is_contiguous()
        or tensor.is_neg()
        or tensor.is_conj()
    ):
        raise ValueError(f"{name} requires a CUDA {dtype} tensor shaped {shape}")
    if tensor.data_ptr() % alignment:
        raise ValueError(f"{name} requires {alignment}-byte alignment")


@cute.jit
def add(a, b):
    return arch.inline_ptx(
        "add.rn.f32 {$w0}, {$r0}, {$r1};",
        write_only_types=[Float32],
        read_only_args=[a, b],
    )


@cute.jit
def mul(a, b):
    return arch.inline_ptx(
        "mul.rn.f32 {$w0}, {$r0}, {$r1};",
        write_only_types=[Float32],
        read_only_args=[a, b],
    )


@cute.jit
def div(a, b):
    return arch.inline_ptx(
        "div.rn.f32 {$w0}, {$r0}, {$r1};",
        write_only_types=[Float32],
        read_only_args=[a, b],
    )


@cute.jit
def fma(a, b, c):
    return arch.inline_ptx(
        "fma.rn.f32 {$w0}, {$r0}, {$r1}, {$r2};",
        write_only_types=[Float32],
        read_only_args=[a, b, c],
    )


@cute.jit
def absf(a):
    return arch.inline_ptx(
        "abs.f32 {$w0}, {$r0};", write_only_types=[Float32], read_only_args=[a]
    )


@cute.jit
def clamp_norm(a):
    return arch.inline_ptx(
        "max.NaN.f32 {$w0}, {$r0}, {$r1};",
        write_only_types=[Float32],
        read_only_args=[a, Float32(1.0e-12)],
    )


@cute.jit
def choose(predicate, yes, no):
    return arch.inline_ptx(
        "selp.b32 {$w0}, {$r0}, {$r1}, {$r2};",
        write_only_types=[type(yes)],
        read_only_args=[yes, no, predicate],
    )


@cute.jit
def load4(ptr, shared: cutlass.Constexpr = False):
    if cutlass.const_expr(shared):
        values = arch.inline_ptx(
            "ld.shared.v4.f32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
            write_only_types=[Float32] * 4,
            read_only_args=[ptr],
        )
    else:
        values = arch.inline_ptx(
            "ld.global.v4.f32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
            write_only_types=[Float32] * 4,
            read_only_args=[ptr],
        )
    return values


@cute.jit
def store4(ptr, values, shared: cutlass.Constexpr = False):
    if cutlass.const_expr(shared):
        arch.inline_ptx(
            "st.shared.v4.f32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
            read_only_args=[ptr, *values],
        )
    else:
        arch.inline_ptx(
            "st.global.v4.f32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
            read_only_args=[ptr, *values],
        )


@cute.jit
def row_load(ptr, aligned: cutlass.Constexpr = True):
    values = cute.make_rmem_tensor(8, Float32)
    if cutlass.const_expr(aligned):
        first = load4(ptr)
        second = load4(ptr + 128)
        for i in cutlass.range_constexpr(4):
            values[i] = first[i]
            values[i + 4] = second[i]
    else:
        row = cute.make_tensor(ptr, cute.make_layout(132))
        for i in cutlass.range_constexpr(8):
            values[i] = row[(i & 3) + (i // 4) * 128]
    return values


@cute.jit
def row_store(ptr, values, shared: cutlass.Constexpr = False):
    store4(ptr, (values[0], values[1], values[2], values[3]), shared)
    store4(ptr + 128, (values[4], values[5], values[6], values[7]), shared)


@cute.jit
def warp_sum(value):
    for offset in cutlass.range_constexpr(5):
        value = add(value, arch.shuffle_sync_down(value, 16 >> offset))
    return arch.shuffle_sync(value, 0)


@cute.jit
def row_sum(values, initial_zero: cutlass.Constexpr = False):
    # Native row256: pair values 128 apart, fold four slots, then shuffle down.
    first = values[0]
    if cutlass.const_expr(initial_zero):
        first = add(Float32(0), first)
    result = add(first, values[4])
    for i in cutlass.range_constexpr(1, 4):
        first = values[i]
        if cutlass.const_expr(initial_zero):
            first = add(Float32(0), first)
        result = add(result, add(first, values[i + 4]))
    return warp_sum(result)


@cute.jit
def row_reciprocal(denominator):
    approximate = arch.rcp_approx(denominator)
    return fma(approximate, fma(-denominator, approximate, Float32(1)), approximate)


@cute.jit
def fast_range(value):
    magnitude = absf(value)
    return (magnitude >= Float32(2.0**-60)) & (magnitude < Float32(2.0**60))


@cute.jit
def row_divide(values, denominator, reciprocal):
    result = cute.make_rmem_tensor(8, Float32)
    fast = fast_range(denominator)
    for i in cutlass.range_constexpr(8):
        value = values[i]
        fast = fast & (fast_range(value) | (value == Float32(0)))
        quotient = mul(value, reciprocal)
        corrected = fma(reciprocal, fma(-denominator, quotient, value), quotient)
        result[i] = choose(value == Float32(0), quotient, corrected)
    if not arch.vote_all_sync(fast):
        for i in cutlass.range_constexpr(8):
            result[i] = div(values[i], denominator)
    return result


@cute.jit
def cluster_sync():
    arch.cluster_arrive(aligned=True)
    arch.cluster_wait()


@cute.jit
def remote_load(tensor, index, rank):
    ptr = arch.map_dsmem_ptr(tensor.iterator, rank)
    return arch.inline_ptx(
        "ld.shared::cluster.b32 {$w0}, [{$r0}];",
        write_only_types=[tensor.element_type],
        read_only_args=[ptr + index],
    )


@cute.jit
def remote_store(tensor, index, rank, value):
    ptr = arch.map_dsmem_ptr(tensor.iterator, rank)
    arch.inline_ptx(
        "st.shared::cluster.b32 [{$r0}], {$r1};", read_only_args=[ptr + index, value]
    )


def launch(compiled_kernels, name, function, tensors, *constants):
    types = {
        torch.float32: Float32,
        torch.int32: Uint32,
        torch.int64: cutlass.Int64,
        torch.bool: Uint8,
        torch.uint8: Uint8,
    }
    alignments = tuple(
        16 if tensor.data_ptr() % 16 == 0 else tensor.element_size()
        for tensor in tensors
    )
    with torch.cuda.device(tensors[0].device):
        args = tuple(
            make_ptr(
                types[tensor.dtype],
                tensor.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=alignment,
            )
            for tensor, alignment in zip(tensors, alignments)
        )
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        key = (name, tensors[0].device.index, alignments, *constants)
        if key not in compiled_kernels:
            compiled_kernels[key] = cute.compile(function, args, stream, *constants)
        compiled_kernels[key](args, stream)
