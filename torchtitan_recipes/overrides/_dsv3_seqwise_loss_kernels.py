# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""CuTeDSL kernels for the FP32 [4096, 256] DSv3 load-balance loss."""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Uint32
from cutlass.cute import arch
from cutlass.memory import SmemAllocator

from ._dsv3_routing_math import (
    absf,
    add,
    check_tensor,
    choose,
    clamp_norm,
    finish_loss,
    launch,
    mul,
    row_divide,
    row_load,
    row_reciprocal,
    row_store,
    row_sum,
)


@cute.kernel
def forward_kernel(
    scores_ptr,
    routing_map_ptr,
    arrivals_ptr,
    partition_sums_ptr,
    partition_counts_ptr,
    raw_ptr,
    frequencies_ptr,
):
    scores = cute.make_tensor(scores_ptr, cute.make_layout(4096 * 256))
    routing_map = cute.make_tensor(routing_map_ptr, cute.make_layout(4096 * 256))
    arrivals = cute.make_tensor(arrivals_ptr, cute.make_layout(1))
    partition_sums = cute.make_tensor(partition_sums_ptr, cute.make_layout(64 * 256))
    partition_counts = cute.make_tensor(
        partition_counts_ptr, cute.make_layout(64 * 256)
    )
    raw = cute.make_tensor(raw_ptr, cute.make_layout(1))
    frequencies = cute.make_tensor(frequencies_ptr, cute.make_layout(256))
    thread = arch.thread_idx()[0]
    rank = arch.block_idx_in_cluster()
    partition = arch.block_idx()[0] // 8
    lane = thread & 31
    warp = thread >> 5
    slot = 2 * (rank & 1) + (warp >> 2)
    token = partition * 4 + (rank >> 1) + (slot + (warp & 3) * 4) * 256
    smem = SmemAllocator()
    normalized_rows = smem.allocate_tensor(Float32, 8 * 256, byte_alignment=16)
    selected_masks = smem.allocate_tensor(Uint32, 8 * 32, byte_alignment=16)
    items = row_load(scores.iterator + token * 256 + lane * 4)
    magnitudes = cute.make_rmem_tensor(8, Float32)
    for i in cutlass.range_constexpr(8):
        magnitudes[i] = absf(items[i])
    denominator = clamp_norm(row_sum(magnitudes))
    normalized = row_divide(items, denominator, row_reciprocal(denominator))
    row_store(normalized_rows.iterator + warp * 256 + lane * 4, normalized, True)
    selected = Uint32(0)
    for i in cutlass.range_constexpr(8):
        expert = lane * 4 + (i & 3) + (i // 4) * 128
        selected |= Uint32(routing_map[token * 256 + expert] != 0) << i
    selected_masks[warp * 32 + lane] = selected
    finish_loss(
        normalized_rows,
        selected_masks,
        partition_sums,
        partition_counts,
        arrivals,
        raw,
        frequencies,
        raw,
        smem,
        False,
    )


@cute.kernel
def backward_kernel(grad_raw_ptr, scores_ptr, frequencies_ptr, grad_scores_ptr):
    grad_raw = cute.make_tensor(grad_raw_ptr, cute.make_layout(1))
    frequencies = cute.make_tensor(frequencies_ptr, cute.make_layout(256))
    thread = arch.thread_idx()[0]
    lane = thread & 31
    token = arch.block_idx()[0] * 8 + (thread >> 5)
    smem = SmemAllocator()
    coefficients = smem.allocate_tensor(Float32, 256, byte_alignment=16)
    items = row_load(scores_ptr + token * 256 + lane * 4)
    coefficients[thread] = mul(grad_raw[0], frequencies[thread])
    magnitudes = cute.make_rmem_tensor(8, Float32)
    for i in cutlass.range_constexpr(8):
        magnitudes[i] = absf(items[i])
    norm = row_sum(magnitudes)
    denominator = clamp_norm(norm)
    reciprocal = row_reciprocal(denominator)
    normalized = row_divide(items, denominator, reciprocal)
    quotients = row_divide(normalized, denominator, reciprocal)
    arch.sync_threads()
    coefficient = cute.make_rmem_tensor(8, Float32)
    for i in cutlass.range_constexpr(8):
        coefficient[i] = coefficients[lane * 4 + (i & 3) + (i // 4) * 128]
    direct = row_divide(coefficient, denominator, reciprocal)
    terms = cute.make_rmem_tensor(8, Float32)
    for i in cutlass.range_constexpr(8):
        terms[i] = mul(-coefficient[i], quotients[i])
    norm_gradient = choose(norm > Float32(1.0e-12), row_sum(terms), Float32(0))
    output = cute.make_rmem_tensor(8, Float32)
    for i in cutlass.range_constexpr(8):
        sign = Float32(items[i] > Float32(0)) - Float32(items[i] < Float32(0))
        output[i] = add(direct[i], mul(sign, norm_gradient))
    row_store(grad_scores_ptr + token * 256 + lane * 4, output)


@cute.jit
def launch_forward(args, stream):
    forward_kernel(*args).launch(
        grid=(512, 1, 1), block=(256, 1, 1), cluster=(8, 1, 1), stream=stream
    )


@cute.jit
def launch_backward(args, stream):
    backward_kernel(*args).launch(grid=(512, 1, 1), block=(256, 1, 1), stream=stream)


_compiled_kernels = {}


def forward(scores_TE, routing_map_TE, arrival_counter):
    check_tensor(
        scores_TE, (4096, 256), torch.float32, scores_TE.device, "scores", alignment=16
    )
    check_tensor(
        routing_map_TE, scores_TE.shape, torch.bool, scores_TE.device, "routing map"
    )
    check_tensor(
        arrival_counter, (1,), torch.int32, scores_TE.device, "arrival counter"
    )
    raw_sum = scores_TE.new_empty(())
    frequencies_E = scores_TE.new_empty((256,))
    partition_sums = scores_TE.new_empty((64, 256))
    partition_counts = torch.empty(
        (64, 256), device=scores_TE.device, dtype=torch.uint8
    )
    launch(
        _compiled_kernels,
        "forward",
        launch_forward,
        (
            scores_TE,
            routing_map_TE,
            arrival_counter,
            partition_sums,
            partition_counts,
            raw_sum,
            frequencies_E,
        ),
    )
    return raw_sum, frequencies_E


def backward(grad_raw_sum, scores_TE, frequencies_E):
    grad_raw_sum = grad_raw_sum.resolve_neg()
    check_tensor(
        scores_TE, (4096, 256), torch.float32, scores_TE.device, "scores", alignment=16
    )
    check_tensor(frequencies_E, (256,), torch.float32, scores_TE.device, "frequencies")
    check_tensor(grad_raw_sum, (), torch.float32, scores_TE.device, "loss gradient")
    grad_scores_TE = torch.empty_like(scores_TE)
    launch(
        _compiled_kernels,
        "backward",
        launch_backward,
        (grad_raw_sum, scores_TE, frequencies_E, grad_scores_TE),
    )
    return grad_scores_TE
