# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from Yidi Wu (@ydwu4), pytorch/torchtitan#5259.
# Source revision: 25dafa9a6db1e7d5327377a0e28913485e704a1f.
# pyrefly: ignore-errors

"""Experimental DeepSeek V3 sequence-wise loss with fused forward/backward.

Current TorchTitan calls this objective ``MicrobatchWiseLoadBalanceLoss``.
The routing map, loss scaling, metric accumulation, and collectives keep their
native contracts. General fused kernels accept FP32 scores [T,E]; the tuned
CuTe specialization requires nvidia-cutlass-dsl >= 4.8.0."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import spmd_types as spmd
import torch
import triton
import triton.language as tl
from packaging.version import Version
from torch.autograd.function import once_differentiable
from torchtitan.config import derive, override
from torchtitan.distributed.spmd_types import spmd_local_context, spmd_mesh_size
from torchtitan.models.common.moe import (
    MicrobatchWiseLoadBalanceLoss,
    RoundRobinTokenChoiceTopKRouter,
)
from torchtitan.models.deepseek_v3.moe import DeepSeekV3Router

from torchtitan_recipes.overrides.dsv3_mlperf import (
    _unbiased_routing_map,
    UnbiasedTopKLoadBalanceLoss,
)
from torchtitan_recipes.overrides.fused_dsv3_router import (
    _native_row_sum,
    _topk_mask,
    _with_local_counter_sharding,
    check_tensor,
)


def _column_reduction_plan(tokens, experts, device):
    """Derive ATen's FP32 output-vector, warp and CTA reduction geometry."""
    vector = 4 if experts % 4 == 0 else (2 if experts % 2 == 0 else 1)
    threads = 512 // vector
    output_power = min(threads, 1 << ((experts // vector).bit_length() - 1))
    initial_width = min(output_power, 32)
    height = min(1 << (tokens.bit_length() - 1), threads // initial_width)
    width = min(output_power, threads // height)
    split_warps = tokens >= min(height * 16, 256)
    reduction_threads = height if split_warps else 1
    per_thread = triton.cdiv(tokens, reduction_threads)
    properties = torch.cuda.get_device_properties(device)
    target_grid = properties.multi_processor_count * (
        properties.max_threads_per_multi_processor // (width * height)
    )
    grid = triton.cdiv(experts, vector * width * (1 if split_warps else height))
    partitions = 1
    if split_warps and per_thread >= 256 and grid <= target_grid:
        partitions = max(
            triton.cdiv(per_thread, 256),
            min(triton.cdiv(target_grid, grid), triton.cdiv(per_thread, 16)),
        )
    return reduction_threads, partitions


@triton.jit
def _generic_seqwise_rows(
    Scores,
    Map,
    Padding,
    Normalized,
    SelectedMap,
    T: tl.constexpr,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
    UNBIASED: tl.constexpr,
    HAS_PADDING: tl.constexpr,
    TOP_K: tl.constexpr,
    ALIGNMENT_OFFSET: tl.constexpr,
):
    """Fuse native row normalization and optional unbiased selection/masking."""
    token = tl.program_id(0).to(tl.int64)
    experts = tl.arange(0, BLOCK_E)
    valid = experts < E
    scores = tl.load(Scores + token * E + experts, valid, other=0)
    values = scores if UNBIASED else tl.abs(scores)
    norm = _native_row_sum(values, token, T, E, False, ALIGNMENT_OFFSET)
    denominator = norm + 1e-20 if UNBIASED else tl.where(norm < 1e-12, 1e-12, norm)
    normalized = tl.div_rn(scores, denominator)
    if UNBIASED:
        selected, _ = _topk_mask(normalized, experts, valid, TOP_K)
    else:
        selected = tl.load(Map + token * E + experts, valid, other=False)
    active = tl.full((), True, tl.int1)
    if HAS_PADDING:
        active = ~tl.load(Padding + token)
    tl.store(Normalized + token * E + experts, tl.where(active, normalized, 0.0), valid)
    tl.store(SelectedMap + token * E + experts, selected & active, valid)


@triton.jit
def _native_y_sum(values, Y: tl.constexpr):
    """Fold virtual block.y lanes high-to-low independently of Triton layout."""
    rows = tl.arange(0, Y)[:, None]
    for level in tl.static_range(Y.bit_length() - 1):
        distance = Y >> (level + 1)
        indices = ((rows + distance) % Y) + tl.full((1, values.shape[1]), 0, tl.int32)
        peer = tl.gather(values, indices, 0)
        values = tl.where(rows < distance, values + peer, values)
    return tl.sum(tl.where(rows == 0, values, 0.0), 0)


@triton.jit
def _generic_seqwise_columns(
    Normalized,
    Map,
    Sums,
    Counts,
    T: tl.constexpr,
    E: tl.constexpr,
    Y: tl.constexpr,
    PARTITIONS: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Reduce token columns using native four-accumulator CTA partition order."""
    experts = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
    rows = tl.arange(0, Y)[:, None] + tl.program_id(1) * Y
    first = tl.full((Y, BLOCK_E), 0.0, tl.float32)
    second, third, fourth = first, first, first
    count_first, count_second, count_third, count_fourth = first, first, first, first
    for part in range((T + 4 * Y * PARTITIONS - 1) // (4 * Y * PARTITIONS)):
        base = rows + part * 4 * Y * PARTITIONS
        index0 = base * E + experts[None, :]
        index1 = (base + Y * PARTITIONS) * E + experts[None, :]
        index2 = (base + 2 * Y * PARTITIONS) * E + experts[None, :]
        index3 = (base + 3 * Y * PARTITIONS) * E + experts[None, :]
        valid0 = (base < T) & (experts[None, :] < E)
        valid1 = (base + Y * PARTITIONS < T) & (experts[None, :] < E)
        valid2 = (base + 2 * Y * PARTITIONS < T) & (experts[None, :] < E)
        valid3 = (base + 3 * Y * PARTITIONS < T) & (experts[None, :] < E)
        first += tl.load(Normalized + index0, valid0, other=0)
        second += tl.load(Normalized + index1, valid1, other=0)
        third += tl.load(Normalized + index2, valid2, other=0)
        fourth += tl.load(Normalized + index3, valid3, other=0)
        count_first += tl.load(Map + index0, valid0, other=False).to(tl.float32)
        count_second += tl.load(Map + index1, valid1, other=False).to(tl.float32)
        count_third += tl.load(Map + index2, valid2, other=False).to(tl.float32)
        count_fourth += tl.load(Map + index3, valid3, other=False).to(tl.float32)
    sums = _native_y_sum(((first + second) + third) + fourth, Y)
    counts = _native_y_sum(
        ((count_first + count_second) + count_third) + count_fourth, Y
    )
    tl.store(Sums + tl.program_id(1) * E + experts, sums, experts < E)
    tl.store(Counts + tl.program_id(1) * E + experts, counts, experts < E)


@triton.jit
def _generic_seqwise_finish(
    Sums,
    Counts,
    Frequencies,
    Loss,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
    Y: tl.constexpr,
    PARTITIONS: tl.constexpr,
):
    """Fuse native cross-CTA sums, usage frequencies and the final dot product."""
    experts = tl.arange(0, BLOCK_E)
    rows = tl.arange(0, Y)
    sums = tl.full((Y, BLOCK_E), 0.0, tl.float32)
    counts = sums
    for part in range((PARTITIONS + Y - 1) // Y):
        partitions = rows + part * Y
        valid = (partitions[:, None] < PARTITIONS) & (experts[None, :] < E)
        offsets = partitions[:, None] * E + experts[None, :]
        sums += tl.load(Sums + offsets, valid, other=0)
        counts += tl.load(Counts + offsets, valid, other=0)
    sum_E = _native_y_sum(sums, Y)
    count_E = _native_y_sum(counts, Y)
    count_norm = _native_row_sum(count_E, 0, 1, E)
    denominator = tl.where(count_norm < 1e-12, 1e-12, count_norm)
    frequency = tl.div_rn(count_E, denominator) * E
    tl.store(Frequencies + experts, frequency, experts < E)
    raw = _native_row_sum(frequency * sum_E, 0, 1, E)
    tl.store(Loss, raw)


@triton.jit
def _generic_seqwise_backward(
    Grad,
    Scores,
    Frequency,
    Padding,
    Output,
    T: tl.constexpr,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
    UNBIASED: tl.constexpr,
    HAS_PADDING: tl.constexpr,
    ALIGNMENT_OFFSET: tl.constexpr,
):
    """Differentiate the raw local loss with native FP32 normalization order."""
    token = tl.program_id(0).to(tl.int64)
    experts = tl.arange(0, BLOCK_E)
    valid = experts < E
    scores = tl.load(Scores + token * E + experts, valid, other=0)
    frequencies = tl.load(Frequency + experts, valid, other=0)
    coefficients = tl.load(Grad) * frequencies
    norm = _native_row_sum(
        scores if UNBIASED else tl.abs(scores), token, T, E, False, ALIGNMENT_OFFSET
    )
    denominator = norm + 1e-20 if UNBIASED else tl.where(norm < 1e-12, 1e-12, norm)
    normalized = tl.div_rn(scores, denominator)
    terms = -coefficients * tl.div_rn(normalized, denominator)
    norm_gradient = _native_row_sum(terms, token, T, E)
    if not UNBIASED:
        norm_gradient = tl.where(norm > 1e-12, norm_gradient, 0.0)
    sign = (
        tl.full((BLOCK_E,), 1.0, tl.float32)
        if UNBIASED
        else (scores > 0).to(tl.float32) - (scores < 0).to(tl.float32)
    )
    gradient = tl.div_rn(coefficients, denominator) + sign * norm_gradient
    if HAS_PADDING:
        gradient = tl.where(tl.load(Padding + token), 0.0, gradient)
    tl.store(Output + token * E + experts, gradient, valid)


def _generic_forward(scores, routing_map, unbiased, padding, top_k):
    """Use three fused launches with fresh shape-derived reduction scratch."""
    tokens, experts = scores.shape
    y, partitions = _column_reduction_plan(tokens, experts, scores.device)
    normalized = torch.empty_like(scores)
    selected = torch.empty_like(routing_map)
    sums = scores.new_empty((partitions, experts))
    counts = torch.empty_like(sums)
    frequencies = scores.new_empty((experts,))
    raw = scores.new_empty(())
    _generic_seqwise_rows[(tokens,)](
        scores,
        routing_map,
        padding if padding is not None else routing_map,
        normalized,
        selected,
        tokens,
        experts,
        triton.next_power_of_2(experts),
        unbiased,
        padding is not None,
        top_k,
        (scores.data_ptr() % 16) // scores.element_size(),
        num_warps=4,
        enable_fp_fusion=False,
    )
    _generic_seqwise_columns[(triton.cdiv(experts, 32), partitions)](
        normalized,
        selected,
        sums,
        counts,
        tokens,
        experts,
        y,
        partitions,
        32,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _generic_seqwise_finish[(1,)](
        sums,
        counts,
        frequencies,
        raw,
        experts,
        triton.next_power_of_2(experts),
        y if partitions > 1 else 1,
        partitions,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return raw, frequencies


def _generic_backward(grad, scores, frequencies, unbiased, padding):
    """Launch the general-shape derivative without saving normalized score buffers."""
    tokens, experts = scores.shape
    output = torch.empty_like(scores)
    _generic_seqwise_backward[(tokens,)](
        grad.resolve_neg(),
        scores,
        frequencies,
        padding if padding is not None else scores,
        output,
        tokens,
        experts,
        triton.next_power_of_2(experts),
        unbiased,
        padding is not None,
        (scores.data_ptr() % 16) // scores.element_size(),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


try:
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda
    from cutlass import Float32, Int32, Uint32, Uint8
    from cutlass.cute import arch
    from cutlass.cute.runtime import make_ptr
    from cutlass.memory import SmemAllocator

    if Version(cutlass.__version__) < Version("4.8.0"):
        raise ImportError("DSv3 routing kernels require nvidia-cutlass-dsl>=4.8.0.")
    _CUTEDSL_IMPORT_ERROR: ImportError | None = None
except ImportError as error:
    _CUTEDSL_IMPORT_ERROR = error


if TYPE_CHECKING or _CUTEDSL_IMPORT_ERROR is None:

    from torchtitan_recipes.overrides.fused_dsv3_router import select_eight

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
            "st.shared::cluster.b32 [{$r0}], {$r1};",
            read_only_args=[ptr + index, value],
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
            key = (
                name,
                tensors[0].device.index,
                tuple(tensor.dtype for tensor in tensors),
                alignments,
                *constants,
            )
            if key not in compiled_kernels:
                compiled_kernels[key] = cute.compile(function, args, stream, *constants)
            compiled_kernels[key](args, stream)

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
        padding_ptr,
        unbiased: cutlass.Constexpr,
        has_padding: cutlass.Constexpr,
    ):
        scores = cute.make_tensor(scores_ptr, cute.make_layout(4096 * 256))
        routing_map = cute.make_tensor(routing_map_ptr, cute.make_layout(4096 * 256))
        arrivals = cute.make_tensor(arrivals_ptr, cute.make_layout(1))
        partition_sums = cute.make_tensor(
            partition_sums_ptr, cute.make_layout(64 * 256)
        )
        partition_counts = cute.make_tensor(
            partition_counts_ptr, cute.make_layout(64 * 256)
        )
        raw = cute.make_tensor(raw_ptr, cute.make_layout(1))
        frequencies = cute.make_tensor(frequencies_ptr, cute.make_layout(256))
        padding = cute.make_tensor(padding_ptr, cute.make_layout(4096))
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
        if cutlass.const_expr(unbiased):
            denominator = add(row_sum(items), Float32(1e-20))
        else:
            denominator = clamp_norm(row_sum(magnitudes))
        normalized = row_divide(items, denominator, row_reciprocal(denominator))
        if cutlass.const_expr(unbiased):
            unbiased_selected = select_eight(normalized, lane)[0]
        if cutlass.const_expr(has_padding):
            is_padding = padding[token] != 0
            for i in cutlass.range_constexpr(8):
                normalized[i] = choose(is_padding, Float32(0), normalized[i])
        row_store(normalized_rows.iterator + warp * 256 + lane * 4, normalized, True)
        selected = Uint32(0)
        for i in cutlass.range_constexpr(8):
            expert = lane * 4 + (i & 3) + (i // 4) * 128
            if cutlass.const_expr(unbiased):
                selected |= Uint32(unbiased_selected[i]) << i
            else:
                selected |= Uint32(routing_map[token * 256 + expert] != 0) << i
        if cutlass.const_expr(has_padding):
            selected = choose(is_padding, Uint32(0), selected)
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
    def backward_kernel(
        grad_raw_ptr,
        scores_ptr,
        frequencies_ptr,
        grad_scores_ptr,
        padding_ptr,
        unbiased: cutlass.Constexpr,
        has_padding: cutlass.Constexpr,
    ):
        grad_raw = cute.make_tensor(grad_raw_ptr, cute.make_layout(1))
        frequencies = cute.make_tensor(frequencies_ptr, cute.make_layout(256))
        thread = arch.thread_idx()[0]
        lane = thread & 31
        token = arch.block_idx()[0] * 8 + (thread >> 5)
        padding = cute.make_tensor(padding_ptr, cute.make_layout(4096))
        smem = SmemAllocator()
        coefficients = smem.allocate_tensor(Float32, 256, byte_alignment=16)
        items = row_load(scores_ptr + token * 256 + lane * 4)
        coefficients[thread] = mul(grad_raw[0], frequencies[thread])
        magnitudes = cute.make_rmem_tensor(8, Float32)
        for i in cutlass.range_constexpr(8):
            magnitudes[i] = absf(items[i])
        norm = row_sum(magnitudes)
        if cutlass.const_expr(unbiased):
            denominator = add(norm, Float32(1e-20))
        else:
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
        if cutlass.const_expr(unbiased):
            norm_gradient = row_sum(terms)
        else:
            norm_gradient = choose(norm > Float32(1.0e-12), row_sum(terms), Float32(0))
        output = cute.make_rmem_tensor(8, Float32)
        for i in cutlass.range_constexpr(8):
            if cutlass.const_expr(unbiased):
                sign = Float32(1)
            else:
                sign = Float32(items[i] > Float32(0)) - Float32(items[i] < Float32(0))
            output[i] = add(direct[i], mul(sign, norm_gradient))
            if cutlass.const_expr(has_padding):
                output[i] = choose(padding[token] != 0, Float32(0), output[i])
        row_store(grad_scores_ptr + token * 256 + lane * 4, output)

    @cute.jit
    def launch_forward(
        args, stream, unbiased: cutlass.Constexpr, has_padding: cutlass.Constexpr
    ):
        forward_kernel(*args, unbiased, has_padding).launch(
            grid=(512, 1, 1), block=(256, 1, 1), cluster=(8, 1, 1), stream=stream
        )

    @cute.jit
    def launch_backward(
        args, stream, unbiased: cutlass.Constexpr, has_padding: cutlass.Constexpr
    ):
        backward_kernel(*args, unbiased, has_padding).launch(
            grid=(512, 1, 1), block=(256, 1, 1), stream=stream
        )

    _compiled_kernels = {}

    def _kernel_forward(
        scores_TE, routing_map_TE, arrival_counter, unbiased=False, padding_mask_T=None
    ):
        check_tensor(
            scores_TE,
            (4096, 256),
            torch.float32,
            scores_TE.device,
            "scores",
            alignment=16,
        )
        check_tensor(
            routing_map_TE, scores_TE.shape, torch.bool, scores_TE.device, "routing map"
        )
        check_tensor(
            arrival_counter, (1,), torch.int32, scores_TE.device, "arrival counter"
        )
        if padding_mask_T is not None:
            check_tensor(
                padding_mask_T, (4096,), torch.bool, scores_TE.device, "padding mask"
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
                padding_mask_T if padding_mask_T is not None else routing_map_TE,
            ),
            unbiased,
            padding_mask_T is not None,
        )
        return raw_sum, frequencies_E

    def _kernel_backward(
        grad_raw_sum, scores_TE, frequencies_E, unbiased=False, padding_mask_T=None
    ):
        grad_raw_sum = grad_raw_sum.resolve_neg()
        check_tensor(
            scores_TE,
            (4096, 256),
            torch.float32,
            scores_TE.device,
            "scores",
            alignment=16,
        )
        check_tensor(
            frequencies_E, (256,), torch.float32, scores_TE.device, "frequencies"
        )
        check_tensor(grad_raw_sum, (), torch.float32, scores_TE.device, "loss gradient")
        if padding_mask_T is not None:
            check_tensor(
                padding_mask_T, (4096,), torch.bool, scores_TE.device, "padding mask"
            )
        grad_scores_TE = torch.empty_like(scores_TE)
        launch(
            _compiled_kernels,
            "backward",
            launch_backward,
            (
                grad_raw_sum,
                scores_TE,
                frequencies_E,
                grad_scores_TE,
                padding_mask_T if padding_mask_T is not None else scores_TE,
            ),
            unbiased,
            padding_mask_T is not None,
        )
        return grad_scores_TE


def _validate_scores(scores, padding):
    """Validate general score/mask metadata without reading storage."""
    if scores.ndim != 2 or min(scores.shape) <= 0:
        raise ValueError("sequence loss requires nonempty CUDA FP32 scores[T,E]")
    check_tensor(scores, scores.shape, torch.float32, scores.device, "scores")
    if padding is not None:
        check_tensor(
            padding, (scores.shape[0],), torch.bool, scores.device, "padding mask"
        )


def _use_tuned_loss(scores, unbiased, top_k):
    """Keep the tuned launch; unaligned and other shapes use real fused kernels."""
    return (
        _CUTEDSL_IMPORT_ERROR is None
        and scores.shape == (4096, 256)
        and (not unbiased or top_k == 8)
        and scores.data_ptr() % 16 == 0
        and torch.cuda.get_device_capability(scores.device) == (10, 3)
    )


@torch.library.custom_op(
    "torchtitan::dsv3_seqwise_loss_forward",
    mutates_args=("arrival_counter",),
    device_types="cuda",
)
def seqwise_loss_forward_op(
    scores_TE: torch.Tensor,
    routing_map_TE: torch.Tensor,
    arrival_counter: torch.Tensor,
    unbiased: bool = False,
    padding_mask_T: torch.Tensor | None = None,
    top_k: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse the raw local objective while preserving native metric/scaling ownership."""
    scores_TE = scores_TE.resolve_neg()
    _validate_scores(scores_TE, padding_mask_T)
    check_tensor(
        routing_map_TE, scores_TE.shape, torch.bool, scores_TE.device, "routing map"
    )
    check_tensor(
        arrival_counter, (1,), torch.int32, scores_TE.device, "arrival counter"
    )
    if unbiased and not 0 < top_k <= scores_TE.shape[1]:
        raise ValueError("unbiased sequence loss requires 0 < top_k <= E")
    if _use_tuned_loss(scores_TE, unbiased, top_k):
        return _kernel_forward(
            scores_TE, routing_map_TE, arrival_counter, unbiased, padding_mask_T
        )
    return _generic_forward(scores_TE, routing_map_TE, unbiased, padding_mask_T, top_k)


@seqwise_loss_forward_op.register_fake
def _seqwise_loss_forward_fake(
    scores_TE,
    routing_map_TE,
    arrival_counter,
    unbiased=False,
    padding_mask_T=None,
    top_k=8,
):
    """Infer scalar/frequency outputs from metadata without inspecting storage."""
    scores_TE = scores_TE.resolve_neg()
    _validate_scores(scores_TE, padding_mask_T)
    check_tensor(
        routing_map_TE, scores_TE.shape, torch.bool, scores_TE.device, "routing map"
    )
    check_tensor(
        arrival_counter, (1,), torch.int32, scores_TE.device, "arrival counter"
    )
    if unbiased and not 0 < top_k <= scores_TE.shape[1]:
        raise ValueError("unbiased sequence loss requires 0 < top_k <= E")
    return scores_TE.new_empty(()), scores_TE.new_empty((scores_TE.shape[1],))


@torch.library.custom_op(
    "torchtitan::dsv3_seqwise_loss_backward", mutates_args=(), device_types="cuda"
)
def seqwise_loss_backward_op(
    grad_raw_sum: torch.Tensor,
    scores_TE: torch.Tensor,
    frequencies_E: torch.Tensor,
    unbiased: bool = False,
    padding_mask_T: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the raw-loss derivative without retaining normalized score buffers."""
    _validate_scores(scores_TE, padding_mask_T)
    check_tensor(
        frequencies_E,
        (scores_TE.shape[1],),
        torch.float32,
        scores_TE.device,
        "frequencies",
    )
    check_tensor(
        grad_raw_sum.resolve_neg(), (), torch.float32, scores_TE.device, "loss gradient"
    )
    if _use_tuned_loss(scores_TE, False, 8):
        return _kernel_backward(
            grad_raw_sum, scores_TE, frequencies_E, unbiased, padding_mask_T
        )
    return _generic_backward(
        grad_raw_sum, scores_TE, frequencies_E, unbiased, padding_mask_T
    )


@seqwise_loss_backward_op.register_fake
def _seqwise_loss_backward_fake(
    grad_raw_sum, scores_TE, frequencies_E, unbiased=False, padding_mask_T=None
):
    """Infer gradient layout from validated metadata without reading storage."""
    _validate_scores(scores_TE, padding_mask_T)
    check_tensor(
        frequencies_E,
        (scores_TE.shape[1],),
        torch.float32,
        scores_TE.device,
        "frequencies",
    )
    check_tensor(
        grad_raw_sum.resolve_neg(), (), torch.float32, scores_TE.device, "loss gradient"
    )
    return torch.empty_like(scores_TE, memory_format=torch.contiguous_format)


class FusedDSv3SeqwiseLossFunction(torch.autograd.Function):
    """Raw loss and its first derivative; normalization/injection belongs to AuxLoss.

    ``arrival_counter`` is a zero-initialized, module-owned CUDA int32 scalar.
    Calls sharing a counter must be stream-ordered. Separate modules have
    separate counters, so independent layers can run on independent streams.
    """

    @staticmethod
    def spmd_typecheck(
        result,
        *,
        scores_TE,
        routing_map_TE,
        arrival_counter,
        unbiased=False,
        padding_mask_T=None,
        top_k=8,
    ):
        spmd.rules.ignore(arrival_counter)
        if padding_mask_T is not None:
            spmd.rules.ignore(padding_mask_T)
        # Both reductions require complete token/expert dimensions on the
        # non-local axes. The module makes DP local and gates CP/TP to size 1.
        spmd.rules.einsum("__,__->", scores_TE, routing_map_TE, out=result)

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx,
        scores_TE,
        routing_map_TE,
        arrival_counter,
        unbiased=False,
        padding_mask_T=None,
        top_k=8,
    ):
        scores_TE = scores_TE.resolve_neg()
        raw_sum, frequencies_E = seqwise_loss_forward_op(
            scores_TE, routing_map_TE, arrival_counter, unbiased, padding_mask_T, top_k
        )
        ctx.unbiased = unbiased
        ctx.num_inputs = len(ctx.needs_input_grad)
        ctx.save_for_backward(
            scores_TE,
            frequencies_E,
            *((padding_mask_T,) if padding_mask_T is not None else ()),
        )
        return raw_sum

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_raw_sum):  # pyrefly: ignore[bad-override]
        scores_TE, frequencies_E, *padding = ctx.saved_tensors
        gradient = seqwise_loss_backward_op(
            grad_raw_sum,
            scores_TE,
            frequencies_E,
            ctx.unbiased,
            padding[0] if padding else None,
        )
        return (gradient, *(None for _ in range(ctx.num_inputs - 1)))


def _supported(scores_TE, routing_map_TE, padding_mask_T):
    """Check the fused objective's CUDA, precision, mask and mesh requirements."""
    return (
        scores_TE.is_cuda
        and scores_TE.dtype == torch.float32
        and scores_TE.ndim == 2
        and min(scores_TE.shape) > 0
        and scores_TE.is_contiguous()
        and routing_map_TE.dtype == torch.bool
        and routing_map_TE.device == scores_TE.device
        and routing_map_TE.shape == scores_TE.shape
        and routing_map_TE.is_contiguous()
        and (
            padding_mask_T is None
            or (
                padding_mask_T.shape == (scores_TE.shape[0],)
                and padding_mask_T.dtype == torch.bool
                and padding_mask_T.device == scores_TE.device
                and padding_mask_T.is_contiguous()
            )
        )
        and spmd_mesh_size("cp") == 1
        and spmd_mesh_size("tp") == 1
        and torch.cuda.get_device_capability(scores_TE.device)[0] >= 8
    )


class FusedDSv3SeqwiseLoss(MicrobatchWiseLoadBalanceLoss):
    """Specialize the loss while retaining native scaling and metric state."""

    @dataclass(kw_only=True, slots=True)
    class Config(MicrobatchWiseLoadBalanceLoss.Config):
        top_k: int | None = None
        """None uses the provided map; a positive value selects unbiased top-k."""
        require_fusion: bool = False

        def __post_init__(self):
            self.sharding_config = _with_local_counter_sharding(self.sharding_config)
            if self.top_k is not None and self.top_k <= 0:
                raise ValueError("top_k must be positive")

    def __init__(self, config: Config):
        self.top_k = config.top_k
        self.require_fusion = config.require_fusion
        super().__init__(config)
        self.register_buffer(
            "arrival_counter", torch.zeros(1, dtype=torch.int32), persistent=False
        )

    @property
    def metric_name(self) -> str:
        return (
            "microbatch_wise_load_balance_loss"
            if self.top_k is None
            else "unbiased_top_k_load_balance_loss"
        )

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None):
        """Materialize the independent counter on the loss accumulator's device."""
        super()._init_self_buffers(buffer_device=buffer_device)
        self.arrival_counter = torch.zeros(
            1, dtype=torch.int32, device=self.instance_acc.device
        )

    def _compute_local_stats(self, scores_TE, routing_map_TE, padding_mask_T):
        """Retain the configured normalization when the geometry is not fused."""
        if self.top_k is not None:
            return UnbiasedTopKLoadBalanceLoss._compute_local_stats(
                self, scores_TE, routing_map_TE, padding_mask_T
            )
        return super()._compute_local_stats(scores_TE, routing_map_TE, padding_mask_T)

    def forward(
        self,
        scores_TE,
        routing_map_TE,
        *,
        carrier,
        padding_mask_T=None,
        denominator,
    ):
        """Fuse the raw objective while retaining native injection and metric state."""
        if (self.top_k is None or self.top_k <= scores_TE.shape[1]) and _supported(
            scores_TE, routing_map_TE, padding_mask_T
        ):
            with spmd_local_context("dp"):
                raw_sum = FusedDSv3SeqwiseLossFunction.apply(
                    scores_TE,
                    routing_map_TE,
                    self.arrival_counter,
                    self.top_k is not None,
                    padding_mask_T,
                    self.top_k if self.top_k is not None else 8,
                )
                return self.inject(raw_sum, carrier=carrier, denominator=denominator)
        if self.require_fusion:
            raise ValueError(
                "Required DSv3 sequence-loss fusion is unsupported for scores "
                f"{tuple(scores_TE.shape)} {scores_TE.dtype} or the configured mesh/device/top-k"
            )
        if self.top_k is not None:
            routing_map_TE = _unbiased_routing_map(
                scores_TE, self.top_k, padding_mask_T=padding_mask_T
            )
        return super().forward(
            scores_TE,
            routing_map_TE,
            carrier=carrier,
            padding_mask_T=padding_mask_T,
            denominator=denominator,
        )


@override(
    target=MicrobatchWiseLoadBalanceLoss.Config,
    exact=True,
    description="Fuse the DeepSeek V3 sequence-wise auxiliary loss and its backward.",
)
def fused_dsv3_seqwise_loss(
    cfg: MicrobatchWiseLoadBalanceLoss.Config,
    *,
    require_fusion: bool = False,
) -> FusedDSv3SeqwiseLoss.Config:
    """Fuse the provided-map objective while retaining coefficients and metrics."""
    return derive(cfg, FusedDSv3SeqwiseLoss.Config, require_fusion=require_fusion)


@override(
    target=UnbiasedTopKLoadBalanceLoss.Config,
    exact=True,
    description="Fuse NVIDIA's unbiased DSv3 sequence-wise objective.",
)
def fused_dsv3_unbiased_seqwise_loss(
    cfg: UnbiasedTopKLoadBalanceLoss.Config, *, require_fusion: bool = False
) -> FusedDSv3SeqwiseLoss.Config:
    """Preserve coefficients/metrics while selecting fused normalized unbiased top-k."""
    return derive(cfg, FusedDSv3SeqwiseLoss.Config, require_fusion=require_fusion)


def _fuse_aux_loss(router_cfg, aux_loss_cfg, *, require_fusion):
    """Compose the independent loss factory within the router's single claim."""
    if type(aux_loss_cfg) is UnbiasedTopKLoadBalanceLoss.Config:
        router_cfg.aux_loss = fused_dsv3_unbiased_seqwise_loss(
            aux_loss_cfg, require_fusion=require_fusion
        )
    elif type(aux_loss_cfg) is MicrobatchWiseLoadBalanceLoss.Config:
        router_cfg.aux_loss = fused_dsv3_seqwise_loss(
            aux_loss_cfg, require_fusion=require_fusion
        )
    elif require_fusion and aux_loss_cfg is not None:
        raise ValueError(
            "Required sequence-loss fusion needs a supported auxiliary loss"
        )
    return router_cfg


@override(
    target=DeepSeekV3Router.Config,
    exact=True,
    description="Compose separate DeepSeek V3 router and sequence-wise loss fusions.",
)
def fused_dsv3_router_and_seqwise_loss(
    cfg: DeepSeekV3Router.Config,
    *,
    require_fusion: bool = False,
) -> DeepSeekV3Router.Config:
    """Compose factories in one claim because nested override claims conflict."""
    from torchtitan_recipes.overrides.fused_dsv3_router import fused_dsv3_router

    router_cfg = fused_dsv3_router(cfg, require_fusion=require_fusion)
    return _fuse_aux_loss(router_cfg, cfg.aux_loss, require_fusion=require_fusion)


@override(
    target=RoundRobinTokenChoiceTopKRouter.Config,
    exact=True,
    description="Compose forced DSv3 routing and unbiased sequence-loss fusion.",
)
def fused_dsv3_round_robin_router_and_seqwise_loss(
    cfg: RoundRobinTokenChoiceTopKRouter.Config, *, require_fusion: bool = False
) -> DeepSeekV3Router.Config:
    """Keep forced benchmark assignments independent of unbiased auxiliary counts."""
    from torchtitan_recipes.overrides.fused_dsv3_router import (
        fused_dsv3_round_robin_router,
    )

    router_cfg = fused_dsv3_round_robin_router(cfg, require_fusion=require_fusion)
    return _fuse_aux_loss(router_cfg, cfg.aux_loss, require_fusion=require_fusion)
