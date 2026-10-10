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
import torch.nn.functional as F
from cutlass import Float32, Int32, Uint32, Uint8
from cutlass.cute import arch
from cutlass.memory import SmemAllocator

from ._dsv3_routing_math import (
    absf,
    add,
    check_tensor,
    choose,
    clamp_norm,
    cluster_sync,
    div,
    launch,
    mul,
    remote_load,
    remote_store,
    row_divide,
    row_load,
    row_reciprocal,
    row_store,
    row_sum,
    warp_sum,
)


@cute.jit
def finish_loss(
    normalized_rows,
    selected_masks,
    partition_sums,
    partition_counts,
    arrivals,
    raw,
    frequencies,
    smem,
):
    thread = arch.thread_idx()[0]
    rank = arch.block_idx_in_cluster()
    partition = arch.block_idx()[0] // 8
    lane = thread & 31
    slot_sums = smem.allocate_tensor(Float32, 512, byte_alignment=16)
    slot_counts = smem.allocate_tensor(Uint32, 128, byte_alignment=16)
    row_values = smem.allocate_tensor(Float32, 128, byte_alignment=16)
    row_counts = smem.allocate_tensor(Int32, 128, byte_alignment=16)
    products = smem.allocate_tensor(Float32, 256, byte_alignment=16)
    counts = smem.allocate_tensor(Float32, 256, byte_alignment=16)
    count_denominator = smem.allocate_tensor(Float32, 1)
    partial = smem.allocate_tensor(Float32, 64)
    is_last = smem.allocate_tensor(Int32, 1)
    arch.sync_threads()
    # Preserve ATen's partition/slot/y reduction order without FP atomics.
    for local in cutlass.range_constexpr(2):
        value = Float32(0)
        for step in cutlass.range_constexpr(4):
            value = add(value, normalized_rows[(local * 4 + step) * 256 + thread])
        slot_sums[local * 256 + thread] = value
    if thread < 128:
        local = thread >> 6
        word_index = thread & 63
        owner = word_index & 31
        shift = (word_index >> 5) * 4
        word = Uint32(0)
        for step in cutlass.range_constexpr(4):
            bits = selected_masks[(local * 4 + step) * 32 + owner] >> shift
            word += (
                (bits & 1)
                | ((bits >> 1 & 1) << 8)
                | ((bits >> 2 & 1) << 16)
                | ((bits >> 3 & 1) << 24)
            )
        slot_counts[local * 64 + word_index] = word
    cluster_sync()
    expert = rank * 32 + lane
    if thread < 128:
        y = thread >> 5
        value = Float32(0)
        count = Int32(0)
        for slot in cutlass.range_constexpr(4):
            source = y * 2 + (slot >> 1)
            term = remote_load(slot_sums, (slot & 1) * 256 + expert, source)
            if cutlass.const_expr(slot == 0):
                value = term
            else:
                value = add(value, term)
            word = remote_load(slot_counts, (slot & 1) * 64 + (expert >> 2), source)
            count += Int32((word >> ((expert & 3) * 8)) & 255)
        row_values[thread] = value
        row_counts[thread] = count
    arch.sync_threads()
    if thread < 32:
        partition_sums[partition * 256 + expert] = add(
            add(row_values[lane], row_values[lane + 64]),
            add(row_values[lane + 32], row_values[lane + 96]),
        )
        partition_counts[partition * 256 + expert] = Uint8(
            row_counts[lane]
            + row_counts[lane + 32]
            + row_counts[lane + 64]
            + row_counts[lane + 96]
        )
    arch.sync_threads()
    if thread == 0:
        arch.fence_acq_rel_gpu()
    cluster_sync()
    # Each call owns scratch; the module's stream-ordered counter wraps mod 64.
    if (rank == 0) & (thread == 0):
        ticket = arch.atomic_add(
            arrivals.iterator, Uint32(1), sem="acq_rel", scope="gpu"
        )
        last = Int32((ticket & 63) == 63)
        for target in cutlass.range_constexpr(8):
            remote_store(is_last, 0, target, last)
    cluster_sync()
    if is_last[0] != 0:
        arch.fence_acq_rel_gpu()
        if thread < 128:
            y = thread >> 5
            values = cute.make_rmem_tensor(16, Float32)
            bytes_ = cute.make_rmem_tensor(16, Uint8)
            for index in cutlass.range_constexpr(16):
                source = (y + index * 4) * 256 + expert
                values[index] = arch.load(
                    partition_sums.iterator + source, Float32, cop="cg"
                )
                bytes_[index] = arch.load(
                    partition_counts.iterator + source, Uint8, cop="cg"
                )
            value = Float32(0)
            count = Int32(0)
            for index in cutlass.range_constexpr(16):
                value = add(value, values[index])
                count += Int32(bytes_[index])
            row_values[thread] = value
            row_counts[thread] = count
        arch.sync_threads()
        if thread < 32:
            value = add(
                add(row_values[lane], row_values[lane + 64]),
                add(row_values[lane + 32], row_values[lane + 96]),
            )
            count = (
                row_counts[lane]
                + row_counts[lane + 32]
                + row_counts[lane + 64]
                + row_counts[lane + 96]
            )
            remote_store(counts, expert, 0, Float32(count))
            remote_store(products, expert, 0, value)
        cluster_sync()
        if rank == 0:
            if thread < 32:
                items = cute.make_rmem_tensor(8, Float32)
                for i in cutlass.range_constexpr(8):
                    items[i] = counts[lane * 4 + (i & 3) + (i // 4) * 128]
                denominator = clamp_norm(row_sum(items))
                if lane == 0:
                    count_denominator[0] = denominator
            arch.sync_threads()
            frequency = mul(div(counts[thread], count_denominator[0]), Float32(256))
            frequencies[thread] = frequency
            products[thread] = mul(frequency, products[thread])
            arch.sync_threads()
            if thread < 64:
                value = products[thread * 4]
                for i in cutlass.range_constexpr(1, 4):
                    value = add(value, products[thread * 4 + i])
                partial[thread] = value
            arch.sync_threads()
            if thread < 32:
                value = warp_sum(add(partial[lane], partial[lane + 32]))
                if lane == 0:
                    raw[0] = value


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
        smem,
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
    check_tensor(scores_TE, (4096, 256), torch.float32, scores_TE.device, "scores")
    check_tensor(
        routing_map_TE, scores_TE.shape, torch.bool, scores_TE.device, "routing map"
    )
    check_tensor(
        arrival_counter, (1,), torch.int32, scores_TE.device, "arrival counter"
    )
    if scores_TE.data_ptr() % 16:
        frequencies_E = F.normalize(routing_map_TE.float().sum(0), p=1, dim=0) * 256
        raw_sum = (frequencies_E * F.normalize(scores_TE, p=1, dim=-1).sum(0)).sum()
        return raw_sum, frequencies_E
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
    check_tensor(scores_TE, (4096, 256), torch.float32, scores_TE.device, "scores")
    check_tensor(frequencies_E, (256,), torch.float32, scores_TE.device, "frequencies")
    check_tensor(grad_raw_sum, (), torch.float32, scores_TE.device, "loss gradient")
    if scores_TE.data_ptr() % 16:
        # Unaligned ATen norms use a different reduction order.
        norm_T1 = scores_TE.norm(p=1, dim=-1, keepdim=True)
        denominator_TE = norm_T1.clamp_min(1e-12).expand_as(scores_TE)
        coefficients_TE = (
            (grad_raw_sum.expand_as(frequencies_E) * frequencies_E)
            .unsqueeze(0)
            .expand_as(scores_TE)
        )
        quotients_TE = (scores_TE / denominator_TE) / denominator_TE
        terms_TE = -coefficients_TE * quotients_TE
        direct_TE = coefficients_TE / denominator_TE
        norm_grad_T1 = torch.where(norm_T1 > 1e-12, terms_TE.sum(-1, keepdim=True), 0.0)
        return direct_TE + scores_TE.sgn() * norm_grad_T1
    grad_scores_TE = torch.empty_like(scores_TE)
    launch(
        _compiled_kernels,
        "backward",
        launch_backward,
        (grad_raw_sum, scores_TE, frequencies_E, grad_scores_TE),
    )
    return grad_scores_TE
