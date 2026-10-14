# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Experimental DeepSeek V3 learned routing with fused forward/backward.

The gate stays native; the fusion consumes its FP32 [4096, 256] logits.
The specialized kernels require nvidia-cutlass-dsl >= 4.8.0."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import spmd_types as spmd
import torch
import torch_remat as remat
from packaging.version import Version
from torch.autograd.function import once_differentiable
from torchtitan.config import derive, override
from torchtitan.distributed.spmd_types import spmd_mesh_size
from torchtitan.models.common.activation import Sigmoid
from torchtitan.models.common.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.deepseek_v3.moe import DeepSeekV3Router

try:
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda
    from cutlass import Float32, Int32, Int64, Uint32, Uint8
    from cutlass.cute import arch
    from cutlass.cute.runtime import make_ptr
    from cutlass.memory import SmemAllocator

    if Version(cutlass.__version__) < Version("4.8.0"):
        raise ImportError("DSv3 routing kernels require nvidia-cutlass-dsl>=4.8.0.")
    _CUTEDSL_IMPORT_ERROR: ImportError | None = None
except ImportError as error:
    _CUTEDSL_IMPORT_ERROR = error


if TYPE_CHECKING or _CUTEDSL_IMPORT_ERROR is None:

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
            key = (name, tensors[0].device.index, alignments, *constants)
            if key not in compiled_kernels:
                compiled_kernels[key] = cute.compile(function, args, stream, *constants)
            compiled_kernels[key](args, stream)

    @cute.jit
    def bits(value):
        return arch.inline_ptx(
            "mov.b32 {$w0}, {$r0};", write_only_types=[Uint32], read_only_args=[value]
        )

    @cute.jit
    def as_float(value):
        return arch.inline_ptx(
            "mov.b32 {$w0}, {$r0};", write_only_types=[Float32], read_only_args=[value]
        )

    @cute.jit
    def max_key(a, b):
        return arch.inline_ptx(
            "max.u32 {$w0}, {$r0}, {$r1};",
            write_only_types=[Uint32],
            read_only_args=[a, b],
        )

    @cute.jit
    def min_id(a, b):
        return arch.inline_ptx(
            "min.s32 {$w0}, {$r0}, {$r1};",
            write_only_types=[Int32],
            read_only_args=[a, b],
        )

    @cute.jit
    def ordering_key(value):
        value_bits = bits(value)
        flip = choose(
            (value_bits & Uint32(0x80000000)) != 0,
            Uint32(0xFFFFFFFF),
            Uint32(0x80000000),
        )
        return choose(
            (value_bits & Uint32(0x7FFFFFFF)) > Uint32(0x7F800000),
            Uint32(0xFFFFFFFF),
            value_bits ^ flip,
        )

    @cute.jit
    def key_value(key):
        flip = choose(
            (key & Uint32(0x80000000)) != 0, Uint32(0x80000000), Uint32(0xFFFFFFFF)
        )
        return as_float(key ^ flip)

    @cute.jit
    def group_top_two(scores, half: cutlass.Constexpr, lane):
        mask = Uint32(255) << (lane & ~7)
        keys = cute.make_rmem_tensor(4, Uint32)
        chosen = cute.make_rmem_tensor(2, Uint32)
        for i in cutlass.range_constexpr(4):
            keys[i] = ordering_key(scores[half * 4 + i])
        for choice in cutlass.range_constexpr(2):
            best = max_key(max_key(keys[0], keys[1]), max_key(keys[2], keys[3]))
            best = arch.warp_redux_sync(best, "max", mask)
            expert = Int32(256)
            for i in cutlass.range_constexpr(4):
                expert = min_id(
                    expert, choose(keys[i] == best, lane * 4 + i, Int32(256))
                )
            expert = arch.warp_redux_sync(expert, "min", mask)
            for i in cutlass.range_constexpr(4):
                keys[i] = choose(lane * 4 + i == expert, Uint32(0), keys[i])
            chosen[choice] = best
        return add(add(Float32(0), key_value(chosen[0])), key_value(chosen[1]))

    @cute.jit
    def select_groups(scores, lane):
        first = group_top_two(scores, 0, lane)
        second = group_top_two(scores, 1, lane)
        first_group = arch.shuffle_sync(first, (lane & 3) * 8)
        second_group = arch.shuffle_sync(second, (lane & 3) * 8)
        key = choose(
            lane < 8,
            ordering_key(choose(lane < 4, first_group, second_group)),
            Uint32(0),
        )
        selected_groups = Uint32(0)
        for choice in cutlass.range_constexpr(4):
            best = arch.warp_redux_sync(key, "max")
            group = arch.warp_redux_sync(choose(key == best, lane, Int32(32)), "min")
            selected_groups |= Uint32(1) << group
            key = choose(lane == group, Uint32(0), key)
        return selected_groups

    @cute.jit
    def select_eight(scores, lane):
        keys = cute.make_rmem_tensor(8, Uint32)
        selected = cute.make_rmem_tensor(8, Int32)
        for i in cutlass.range_constexpr(8):
            keys[i] = ordering_key(scores[i])
            selected[i] = Int32(0)
        choice_expert = Int32(0)
        choice_key = Uint32(0)
        threshold = Uint32(0)
        for choice in cutlass.range(8, unroll=1):
            best = Uint32(0)
            for i in cutlass.range_constexpr(8):
                best = max_key(best, keys[i])
            best = arch.warp_redux_sync(best, "max")
            threshold = best
            expert = Int32(256)
            for i in cutlass.range_constexpr(8):
                index = lane * 4 + (i & 3) + (i // 4) * 128
                expert = min_id(expert, choose(keys[i] == best, index, Int32(256)))
            expert = arch.warp_redux_sync(expert, "min")
            choice_expert = choose(lane == choice, expert, choice_expert)
            choice_key = choose(lane == choice, best, choice_key)
            for i in cutlass.range_constexpr(8):
                index = lane * 4 + (i & 3) + (i // 4) * 128
                selected[i] = choose(index == expert, Int32(1), selected[i])
                keys[i] = choose(index == expert, Uint32(0), keys[i])
        return selected, choice_expert, choice_key, threshold

    @cute.jit
    def native_order(expert, choice_key, threshold, lane):
        # Native unsorted topk orders larger values by ID, then threshold ties.
        key = expert + choose(choice_key == threshold, Int32(256), Int32(0))
        for level in cutlass.range_constexpr(1, 4):
            for step in cutlass.range_constexpr(level):
                distance = 1 << (level - 1 - step)
                peer = arch.shuffle_sync_bfly(
                    key, distance, mask=255, mask_and_clamp=0x1807
                )
                take_min = ((lane & (1 << level)) == 0) == ((lane & distance) == 0)
                maximum = Int32(max_key(Uint32(key), Uint32(peer)))
                key = choose(take_min, min_id(key, peer), maximum)
        return key & 255

    @cute.jit
    def finish_counts(selected_masks, partition_counts, arrivals, dispatch, smem):
        thread = arch.thread_idx()[0]
        rank = arch.block_idx_in_cluster()
        partition = arch.block_idx()[0] // 8
        lane = thread & 31
        slot_counts = smem.allocate_tensor(Uint32, 128, byte_alignment=16)
        row_counts = smem.allocate_tensor(Int32, 128, byte_alignment=16)
        is_last = smem.allocate_tensor(Int32, 1)
        arch.sync_threads()
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
            count = Int32(0)
            for slot in cutlass.range_constexpr(4):
                source = y * 2 + (slot >> 1)
                word = remote_load(slot_counts, (slot & 1) * 64 + (expert >> 2), source)
                count += Int32((word >> ((expert & 3) * 8)) & 255)
            row_counts[thread] = count
        arch.sync_threads()
        if thread < 32:
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
                count = Int32(0)
                for index in cutlass.range_constexpr(16):
                    source = (y + index * 4) * 256 + expert
                    count += Int32(
                        arch.load(partition_counts.iterator + source, Uint8, cop="cg")
                    )
                row_counts[thread] = count
            arch.sync_threads()
            if thread < 32:
                dispatch[expert] = Int64(
                    row_counts[lane]
                    + row_counts[lane + 32]
                    + row_counts[lane + 64]
                    + row_counts[lane + 96]
                )

    @cute.kernel
    def forward_kernel(args, has_bias: cutlass.Constexpr, aligned: cutlass.Constexpr):
        logits_ptr, bias_ptr, arrivals_ptr = args[:3]
        weights_ptr, ids_ptr, map_ptr, dispatch_ptr = args[3:7]
        scores_ptr, selected_ptr, route_denom_ptr = args[7:10]
        (partition_counts_ptr,) = args[10:]
        logits = cute.make_tensor(logits_ptr, cute.make_layout(4096 * 256))
        bias = cute.make_tensor(bias_ptr, cute.make_layout(256))
        arrivals = cute.make_tensor(arrivals_ptr, cute.make_layout(1))
        weights = cute.make_tensor(weights_ptr, cute.make_layout(4096 * 8))
        ids = cute.make_tensor(ids_ptr, cute.make_layout(4096 * 8))
        routing_map = cute.make_tensor(map_ptr, cute.make_layout(4096 * 256))
        dispatch = cute.make_tensor(dispatch_ptr, cute.make_layout(256))
        selected_output = cute.make_tensor(selected_ptr, cute.make_layout(4096 * 8))
        route_denom = cute.make_tensor(route_denom_ptr, cute.make_layout(4096))
        partition_counts = cute.make_tensor(
            partition_counts_ptr, cute.make_layout(64 * 256)
        )
        thread = arch.thread_idx()[0]
        rank = arch.block_idx_in_cluster()
        partition = arch.block_idx()[0] // 8
        lane = thread & 31
        warp = thread >> 5
        slot = 2 * (rank & 1) + (warp >> 2)
        token = partition * 4 + (rank >> 1) + (slot + (warp & 3) * 4) * 256
        smem = SmemAllocator()
        shared_values = smem.allocate_tensor(Float32, 8 * 256, byte_alignment=16)
        selected_masks = smem.allocate_tensor(Uint32, 8 * 32, byte_alignment=16)
        values = cute.make_rmem_tensor(8, Float32)
        if cutlass.const_expr(aligned):
            values = row_load(logits_ptr + token * 256 + lane * 4)
        else:
            for i in cutlass.range_constexpr(8):
                values[i] = logits[token * 256 + lane * 4 + (i & 3) + (i // 4) * 128]
        scores = cute.make_rmem_tensor(8, Float32)
        choice_scores = cute.make_rmem_tensor(8, Float32)
        for i in cutlass.range_constexpr(8):
            score = div(
                Float32(1), add(Float32(1), cute.exp(-values[i], fastmath=False))
            )
            scores[i] = score
            if cutlass.const_expr(has_bias):
                score = add(score, bias[lane * 4 + (i & 3) + (i // 4) * 128])
            choice_scores[i] = score
        row_store(scores_ptr + token * 256 + lane * 4, scores)
        row_store(shared_values.iterator + warp * 256 + lane * 4, scores, True)
        groups = select_groups(choice_scores, lane)
        for i in cutlass.range_constexpr(8):
            group = lane // 8 + (i // 4) * 4
            choice_scores[i] = choose(
                (groups & (Uint32(1) << group)) != 0,
                choice_scores[i],
                Float32(float("-inf")),
            )
        dispatch_selected, choice_expert, choice_key, threshold = select_eight(
            choice_scores, lane
        )
        arch.sync_warp()
        if lane < 8:
            expert = native_order(choice_expert, choice_key, threshold, lane)
            selected = shared_values[warp * 256 + expert]
            route_denominator = selected
            for step in cutlass.range_constexpr(3):
                route_denominator = add(
                    route_denominator,
                    arch.shuffle_sync_down(
                        route_denominator, 4 >> step, mask=255, mask_and_clamp=0x1807
                    ),
                )
            route_denominator = add(
                arch.shuffle_sync(
                    route_denominator, 0, mask=255, mask_and_clamp=0x1807
                ),
                Float32(1.0e-20),
            )
            weights[token * 8 + lane] = mul(
                div(selected, route_denominator), Float32(2.5)
            )
            ids[token * 8 + lane] = Int64(expert)
            selected_output[token * 8 + lane] = selected
            if lane == 0:
                route_denom[token] = route_denominator
        selected = Uint32(0)
        for i in cutlass.range_constexpr(8):
            expert = lane * 4 + (i & 3) + (i // 4) * 128
            routing_map[token * 256 + expert] = Uint8(dispatch_selected[i])
            selected |= Uint32(dispatch_selected[i]) << i
        selected_masks[warp * 32 + lane] = selected
        finish_counts(selected_masks, partition_counts, arrivals, dispatch, smem)

    @cute.kernel
    def backward_kernel(
        args,
        has_route: cutlass.Constexpr,
        has_score_grad: cutlass.Constexpr,
        stride_t: cutlass.Constexpr,
        stride_k: cutlass.Constexpr,
        neg_route: cutlass.Constexpr,
        neg_score_grad: cutlass.Constexpr,
        deterministic: cutlass.Constexpr,
        aligned_scores: cutlass.Constexpr,
        score_stride_t: cutlass.Constexpr,
        score_stride_e: cutlass.Constexpr,
        aligned_score_grad: cutlass.Constexpr,
    ):
        (
            scores_ptr,
            ids_ptr,
            selected_ptr,
            route_denom_ptr,
            grad_weights_ptr,
            grad_scores_ptr,
            output_ptr,
        ) = args
        ids = cute.make_tensor(ids_ptr, cute.make_layout(4096 * 8))
        selected = cute.make_tensor(selected_ptr, cute.make_layout(4096 * 8))
        route_denom = cute.make_tensor(route_denom_ptr, cute.make_layout(4096))
        grad_weights = cute.make_tensor(
            grad_weights_ptr, cute.make_layout(max(1, 4096 * stride_t + 8 * stride_k))
        )
        grad_scores = cute.make_tensor(
            grad_scores_ptr,
            cute.make_layout(max(1, 4096 * score_stride_t + 256 * score_stride_e)),
        )
        thread = arch.thread_idx()[0]
        lane = thread & 31
        warp = thread >> 5
        token = arch.block_idx()[0] * 8 + warp
        smem = SmemAllocator()
        route_gradients = smem.allocate_tensor(Float32, 8 * 256, byte_alignment=16)
        if cutlass.const_expr(has_route):
            zeros = cute.make_rmem_tensor(8, Float32)
            zeros.fill(Float32(0))
            row_store(route_gradients.iterator + warp * 256 + lane * 4, zeros, True)
        if cutlass.const_expr(has_route):
            arch.sync_warp()
        if cutlass.const_expr(has_route):
            if lane < 8:
                route_upstream = grad_weights[token * stride_t + lane * stride_k]
                if cutlass.const_expr(neg_route):
                    route_upstream = -route_upstream
                scaled = mul(route_upstream, Float32(2.5))
                denominator = route_denom[token]
                contribution = mul(
                    -scaled,
                    div(div(selected[token * 8 + lane], denominator), denominator),
                )
                denominator_gradient = add(Float32(0), contribution)
                if cutlass.const_expr(stride_t > 0 and stride_k > stride_t):
                    denominator_gradient = add(
                        denominator_gradient,
                        arch.shuffle_sync_down(
                            contribution, 4, mask=255, mask_and_clamp=0x1807
                        ),
                    )
                    folded = arch.shuffle_sync(
                        denominator_gradient, 0, mask=255, mask_and_clamp=0x1807
                    )
                    for i in cutlass.range_constexpr(1, 4):
                        folded = add(
                            folded,
                            arch.shuffle_sync(
                                denominator_gradient, i, mask=255, mask_and_clamp=0x1807
                            ),
                        )
                    denominator_gradient = folded
                else:
                    for step in cutlass.range_constexpr(3):
                        denominator_gradient = add(
                            denominator_gradient,
                            arch.shuffle_sync_down(
                                denominator_gradient,
                                4 >> step,
                                mask=255,
                                mask_and_clamp=0x1807,
                            ),
                        )
                    denominator_gradient = arch.shuffle_sync(
                        denominator_gradient, 0, mask=255, mask_and_clamp=0x1807
                    )
                scattered = add(
                    Float32(0), add(div(scaled, denominator), denominator_gradient)
                )
                if cutlass.const_expr(not deterministic):
                    scattered = choose(
                        (bits(scattered) & Uint32(0x7FFFFFFF)) < Uint32(0x00800000),
                        Float32(0),
                        scattered,
                    )
                route_gradients[warp * 256 + Int32(ids[token * 8 + lane])] = scattered
            arch.sync_warp()
        scores = row_load(scores_ptr + token * 256 + lane * 4, aligned_scores)
        gradients = cute.make_rmem_tensor(8, Float32)
        gradients.fill(Float32(0))
        if cutlass.const_expr(has_score_grad):
            if cutlass.const_expr(score_stride_t == 256 and score_stride_e == 1):
                gradients = row_load(
                    grad_scores_ptr + token * 256 + lane * 4, aligned_score_grad
                )
            else:
                for i in cutlass.range_constexpr(8):
                    expert = lane * 4 + (i & 3) + (i // 4) * 128
                    gradients[i] = grad_scores[
                        token * score_stride_t + expert * score_stride_e
                    ]
            if cutlass.const_expr(neg_score_grad):
                for i in cutlass.range_constexpr(8):
                    gradients[i] = -gradients[i]
        for i in cutlass.range_constexpr(8):
            if cutlass.const_expr(has_route):
                route_gradient = route_gradients[
                    warp * 256 + lane * 4 + (i & 3) + (i // 4) * 128
                ]
                if cutlass.const_expr(has_score_grad):
                    gradients[i] = add(gradients[i], route_gradient)
                else:
                    gradients[i] = route_gradient
            gradients[i] = mul(
                mul(gradients[i], add(Float32(1), -scores[i])), scores[i]
            )
        row_store(output_ptr + token * 256 + lane * 4, gradients)

    @cute.jit
    def launch_forward(
        args, stream, has_bias: cutlass.Constexpr, aligned: cutlass.Constexpr
    ):
        forward_kernel(args, has_bias, aligned).launch(
            grid=(512, 1, 1), block=(256, 1, 1), cluster=(8, 1, 1), stream=stream
        )

    @cute.jit
    def launch_backward(args, stream, flags: cutlass.Constexpr):
        backward_kernel(args, *flags).launch(
            grid=(512, 1, 1), block=(256, 1, 1), stream=stream
        )

    _compiled_kernels = {}

    def _kernel_forward(logits_TE, expert_bias_E, arrival_counter):
        logits_TE = logits_TE.resolve_neg()
        check_tensor(logits_TE, (4096, 256), torch.float32, logits_TE.device, "logits")
        check_tensor(
            arrival_counter, (1,), torch.int32, logits_TE.device, "arrival counter"
        )
        if expert_bias_E is not None:
            expert_bias_E = expert_bias_E.resolve_neg()
            check_tensor(
                expert_bias_E, (256,), torch.float32, logits_TE.device, "expert bias"
            )

        def empty(shape, dtype=torch.float32):
            return torch.empty(shape, device=logits_TE.device, dtype=dtype)

        outputs = (
            empty((4096, 8)),
            empty((4096, 8), torch.int64),
            empty((4096, 256), torch.bool),
            empty((256,), torch.int64),
            empty((4096, 256)),
            empty((4096, 8)),
            empty((4096, 1)),
        )
        partition_counts = empty((64, 256), torch.uint8)
        launch(
            _compiled_kernels,
            "forward",
            launch_forward,
            (
                logits_TE,
                expert_bias_E if expert_bias_E is not None else logits_TE,
                arrival_counter,
                *outputs,
                partition_counts,
            ),
            expert_bias_E is not None,
            logits_TE.data_ptr() % 16 == 0,
        )
        return outputs

    def _kernel_backward(
        scores_TE,
        expert_ids_TK,
        selected_scores_TK,
        route_denominator_T1,
        grad_weights_TK,
        grad_scores_TE,
    ):
        for tensor, shape, dtype, name in (
            (scores_TE, (4096, 256), torch.float32, "scores"),
            (expert_ids_TK, (4096, 8), torch.int64, "expert IDs"),
            (selected_scores_TK, (4096, 8), torch.float32, "selected scores"),
            (route_denominator_T1, (4096, 1), torch.float32, "route denominator"),
        ):
            check_tensor(tensor, shape, dtype, scores_TE.device, name)
        for gradient, shape in (
            (grad_weights_TK, (4096, 8)),
            (grad_scores_TE, (4096, 256)),
        ):
            if gradient is not None and (
                gradient.shape != shape
                or gradient.dtype != torch.float32
                or gradient.device != scores_TE.device
            ):
                raise ValueError(f"router gradient requires CUDA FP32 {shape}")
        grad_logits_TE = torch.empty_like(scores_TE)
        has_route, has_score_grad = (
            grad_weights_TK is not None,
            grad_scores_TE is not None,
        )
        route_strides = grad_weights_TK.stride() if has_route else (0, 0)
        score_strides = grad_scores_TE.stride() if has_score_grad else (0, 0)
        flags = (
            has_route,
            has_score_grad,
            *route_strides,
            grad_weights_TK.is_neg() if has_route else False,
            grad_scores_TE.is_neg() if has_score_grad else False,
            torch.are_deterministic_algorithms_enabled(),
            scores_TE.data_ptr() % 16 == 0,
            *score_strides,
            grad_scores_TE.data_ptr() % 16 == 0 if has_score_grad else True,
        )
        launch(
            _compiled_kernels,
            "backward",
            launch_backward,
            (
                scores_TE,
                expert_ids_TK,
                selected_scores_TK,
                route_denominator_T1,
                grad_weights_TK if has_route else scores_TE,
                grad_scores_TE if has_score_grad else scores_TE,
                grad_logits_TE,
            ),
            flags,
        )
        return grad_logits_TE


@torch.library.custom_op(
    "torchtitan::dsv3_router_forward",
    mutates_args=("arrival_counter",),
    device_types="cuda",
)
def router_forward_op(
    logits_TE: torch.Tensor,
    expert_bias_E: torch.Tensor | None,
    arrival_counter: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if _CUTEDSL_IMPORT_ERROR is not None:
        raise ImportError(
            "DSv3 router fusion requires CuTeDSL; install nvidia-cutlass-dsl>=4.8.0."
        ) from _CUTEDSL_IMPORT_ERROR
    return _kernel_forward(logits_TE, expert_bias_E, arrival_counter)


@router_forward_op.register_fake
def _router_forward_fake(logits_TE, expert_bias_E, arrival_counter):
    tokens, experts = logits_TE.shape
    return (
        logits_TE.new_empty((tokens, 8)),
        logits_TE.new_empty((tokens, 8), dtype=torch.int64),
        logits_TE.new_empty((tokens, experts), dtype=torch.bool),
        logits_TE.new_empty((experts,), dtype=torch.int64),
        logits_TE.new_empty((tokens, experts)),
        logits_TE.new_empty((tokens, 8)),
        logits_TE.new_empty((tokens, 1)),
    )


@torch.library.custom_op(
    "torchtitan::dsv3_router_backward", mutates_args=(), device_types="cuda"
)
def router_backward_op(
    scores_TE: torch.Tensor,
    expert_ids_TK: torch.Tensor,
    selected_scores_TK: torch.Tensor,
    route_denominator_T1: torch.Tensor,
    grad_weights_TK: torch.Tensor | None,
    grad_scores_TE: torch.Tensor | None,
) -> torch.Tensor:
    if _CUTEDSL_IMPORT_ERROR is not None:
        raise ImportError(
            "DSv3 router fusion requires CuTeDSL; install nvidia-cutlass-dsl>=4.8.0."
        ) from _CUTEDSL_IMPORT_ERROR
    return _kernel_backward(
        scores_TE,
        expert_ids_TK,
        selected_scores_TK,
        route_denominator_T1,
        grad_weights_TK,
        grad_scores_TE,
    )


@router_backward_op.register_fake
def _router_backward_fake(
    scores_TE,
    expert_ids_TK,
    selected_scores_TK,
    route_denominator_T1,
    grad_weights_TK,
    grad_scores_TE,
):
    return torch.empty_like(scores_TE, memory_format=torch.contiguous_format)


class FusedDSv3RouterFunction(torch.autograd.Function):
    """Learned top-k routing with differentiable scores for an external loss.

    Returns weights, expert IDs, routing map, counts, and sigmoid scores.
    Weights and scores are differentiable. Bias affects discrete selection,
    so it receives no gradient, as in the native router. Calls sharing the
    zero-initialized arrival counter must be stream-ordered.
    """

    @staticmethod
    def spmd_typecheck(outputs, *, logits_TE, expert_bias_E, arrival_counter):
        spmd.rules.ignore(arrival_counter)
        weights_TK, ids_TK, routing_map_TE, counts_E, scores_TE = outputs
        operands = (logits_TE,) if expert_bias_E is None else (logits_TE, expert_bias_E)
        rows = "t_->t_" if expert_bias_E is None else "t_,_->t_"
        for output in (weights_TK, ids_TK, routing_map_TE, scores_TE):
            spmd.rules.einsum(rows, *operands, out=output)
        spmd.rules.einsum("t_->_", routing_map_TE, out=counts_E)

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx, logits_TE, expert_bias_E, arrival_counter
    ):
        (
            weights_TK,
            ids_TK,
            routing_map_TE,
            counts_E,
            scores_TE,
            selected_TK,
            route_denominator_T1,
        ) = router_forward_op(logits_TE, expert_bias_E, arrival_counter)
        ctx.save_for_backward(scores_TE, ids_TK, selected_TK, route_denominator_T1)
        ctx.mark_non_differentiable(ids_TK, routing_map_TE, counts_E)
        ctx.set_materialize_grads(False)
        return weights_TK, ids_TK, routing_map_TE, counts_E, scores_TE

    @staticmethod
    @once_differentiable
    def backward(  # pyrefly: ignore[bad-override]
        ctx, grad_weights_TK, grad_ids, grad_map, grad_counts, grad_scores_TE
    ):
        if grad_weights_TK is None and grad_scores_TE is None:
            return None, None, None
        return (
            router_backward_op(*ctx.saved_tensors, grad_weights_TK, grad_scores_TE),
            None,
            None,
        )


class FusedDSv3Router(DeepSeekV3Router):
    """Router specialization with no knowledge of the auxiliary-loss formula."""

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV3Router.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.register_buffer(
            "arrival_counter", torch.zeros(1, dtype=torch.int32), persistent=False
        )

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None):
        super()._init_self_buffers(buffer_device=buffer_device)
        self.arrival_counter = torch.zeros(
            1, dtype=torch.int32, device=self.tokens_per_expert_E.device
        )

    def _supports_fusion(self, x_TD, expert_bias_E, padding_mask_T):
        return (
            self.training
            and x_TD.is_cuda
            and x_TD.ndim == 2
            and x_TD.shape[0] == 4096
            and isinstance(self.gate, HiMidLoLinear)
            and not torch.is_autocast_enabled("cuda")
            and self.num_experts == 256
            and self.top_k == 8
            and self.num_expert_groups == 8
            and self.num_limited_groups == 4
            and type(self.score_func) is Sigmoid
            and self.route_norm
            and self.route_norm_epsilon == 1e-20
            and self.route_scale == 2.5
            and padding_mask_T is None
            and spmd_mesh_size("cp") == 1
            and spmd_mesh_size("tp") == 1
            and torch.cuda.get_device_capability(x_TD.device) == (10, 3)
            and torch.cuda.get_device_properties(x_TD.device).multi_processor_count
            >= 128
            and (
                expert_bias_E is None
                or (
                    expert_bias_E.shape == (256,)
                    and expert_bias_E.dtype == torch.float32
                    and expert_bias_E.device == x_TD.device
                    and expert_bias_E.is_contiguous()
                )
            )
        )

    def forward(
        self,
        x_TD,
        expert_bias_E=None,
        *,
        padding_mask_T=None,
        aux_loss_denominator=None,
        **router_kwargs,
    ):
        if self._supports_fusion(x_TD, expert_bias_E, padding_mask_T):
            logits_TE = self.gate(x_TD)
            remat.recompute_needs_tensor(logits_TE)
            weights_TK, ids_TK, routing_map_TE, counts_E, scores_TE = remat.region(
                FusedDSv3RouterFunction.apply, "routing_decision", recompute=False
            )(logits_TE, expert_bias_E, self.arrival_counter)
            remat.recompute_needs_tensor(ids_TK)
            if not remat.is_recomputing():
                with torch.no_grad():
                    self.tokens_per_expert_E.add_(counts_E)
            if self.aux_loss is not None:
                if aux_loss_denominator is None:
                    raise ValueError("An auxiliary-loss denominator is required.")
                remat.recompute_needs_tensor(scores_TE)
                remat.recompute_needs_tensor(routing_map_TE)
                weights_TK = self.aux_loss(
                    scores_TE,
                    routing_map_TE,
                    carrier=weights_TK,
                    padding_mask_T=padding_mask_T,
                    denominator=aux_loss_denominator,
                )
            return weights_TK, ids_TK, routing_map_TE
        return super().forward(
            x_TD,
            expert_bias_E,
            padding_mask_T=padding_mask_T,
            aux_loss_denominator=aux_loss_denominator,
            **router_kwargs,
        )


@override(
    target=DeepSeekV3Router.Config,
    exact=True,
    description="Fuse learned DeepSeek V3 routing and its backward.",
)
def fused_dsv3_router(cfg: DeepSeekV3Router.Config) -> FusedDSv3Router.Config:
    return derive(cfg, FusedDSv3Router.Config)
