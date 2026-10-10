# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""CuTeDSL kernels for the learned DSv3 router and its auxiliary loss."""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32, Uint8
from cutlass.cute import arch
from cutlass.memory import SmemAllocator

from ._dsv3_routing_math import (
    add,
    check_tensor,
    choose,
    clamp_norm,
    div,
    finish_loss,
    launch,
    mul,
    row_divide,
    row_load,
    row_reciprocal,
    row_store,
    row_sum,
)


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
        "max.u32 {$w0}, {$r0}, {$r1};", write_only_types=[Uint32], read_only_args=[a, b]
    )


@cute.jit
def min_id(a, b):
    return arch.inline_ptx(
        "min.s32 {$w0}, {$r0}, {$r1};", write_only_types=[Int32], read_only_args=[a, b]
    )


@cute.jit
def ordering_key(value):
    value_bits = bits(value)
    flip = choose(
        (value_bits & Uint32(0x80000000)) != 0, Uint32(0xFFFFFFFF), Uint32(0x80000000)
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
            expert = min_id(expert, choose(keys[i] == best, lane * 4 + i, Int32(256)))
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
        lane < 8, ordering_key(choose(lane < 4, first_group, second_group)), Uint32(0)
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


@cute.kernel
def forward_kernel(args, has_bias: cutlass.Constexpr, aligned: cutlass.Constexpr):
    logits_ptr, bias_ptr, arrivals_ptr = args[:3]
    weights_ptr, ids_ptr, map_ptr, raw_ptr, dispatch_ptr = args[3:8]
    (
        scores_ptr,
        norms_ptr,
        selected_ptr,
        route_denom_ptr,
        norm_denom_ptr,
        frequencies_ptr,
    ) = args[8:14]
    partition_sums_ptr, partition_counts_ptr = args[14:]
    logits = cute.make_tensor(logits_ptr, cute.make_layout(4096 * 256))
    bias = cute.make_tensor(bias_ptr, cute.make_layout(256))
    arrivals = cute.make_tensor(arrivals_ptr, cute.make_layout(1))
    weights = cute.make_tensor(weights_ptr, cute.make_layout(4096 * 8))
    ids = cute.make_tensor(ids_ptr, cute.make_layout(4096 * 8))
    routing_map = cute.make_tensor(map_ptr, cute.make_layout(4096 * 256))
    raw = cute.make_tensor(raw_ptr, cute.make_layout(1))
    dispatch = cute.make_tensor(dispatch_ptr, cute.make_layout(256))
    norms = cute.make_tensor(norms_ptr, cute.make_layout(4096))
    selected_output = cute.make_tensor(selected_ptr, cute.make_layout(4096 * 8))
    route_denom = cute.make_tensor(route_denom_ptr, cute.make_layout(4096))
    norm_denom = cute.make_tensor(norm_denom_ptr, cute.make_layout(4096))
    frequencies = cute.make_tensor(frequencies_ptr, cute.make_layout(256))
    partition_sums = cute.make_tensor(partition_sums_ptr, cute.make_layout(64 * 256))
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
        score = div(Float32(1), add(Float32(1), cute.exp(-values[i], fastmath=False)))
        scores[i] = score
        if cutlass.const_expr(has_bias):
            score = add(score, bias[lane * 4 + (i & 3) + (i // 4) * 128])
        choice_scores[i] = score
    row_store(scores_ptr + token * 256 + lane * 4, scores)
    row_store(shared_values.iterator + warp * 256 + lane * 4, scores, True)
    norm = row_sum(scores)
    denominator = clamp_norm(norm)
    if lane == 0:
        norms[token] = norm
        norm_denom[token] = denominator
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
            arch.shuffle_sync(route_denominator, 0, mask=255, mask_and_clamp=0x1807),
            Float32(1.0e-20),
        )
        weights[token * 8 + lane] = mul(div(selected, route_denominator), Float32(2.5))
        ids[token * 8 + lane] = Int64(expert)
        selected_output[token * 8 + lane] = selected
        if lane == 0:
            route_denom[token] = route_denominator
    selected = Uint32(0)
    for i in cutlass.range_constexpr(8):
        expert = lane * 4 + (i & 3) + (i // 4) * 128
        routing_map[token * 256 + expert] = Uint8(dispatch_selected[i])
        selected |= Uint32(dispatch_selected[i]) << i
    normalized = row_divide(scores, denominator, row_reciprocal(denominator))
    arch.sync_warp()
    row_store(shared_values.iterator + warp * 256 + lane * 4, normalized, True)
    selected_masks[warp * 32 + lane] = selected
    finish_loss(
        shared_values,
        selected_masks,
        partition_sums,
        partition_counts,
        arrivals,
        raw,
        frequencies,
        dispatch,
        smem,
        True,
    )


@cute.kernel
def backward_kernel(
    args,
    has_route: cutlass.Constexpr,
    has_aux: cutlass.Constexpr,
    stride_t: cutlass.Constexpr,
    stride_k: cutlass.Constexpr,
    neg_route: cutlass.Constexpr,
    neg_aux: cutlass.Constexpr,
    deterministic: cutlass.Constexpr,
    aligned_scores: cutlass.Constexpr,
):
    (
        scores_ptr,
        norms_ptr,
        ids_ptr,
        selected_ptr,
        route_denom_ptr,
        norm_denom_ptr,
        frequencies_ptr,
        grad_weights_ptr,
        grad_raw_ptr,
        output_ptr,
    ) = args
    norms = cute.make_tensor(norms_ptr, cute.make_layout(4096))
    ids = cute.make_tensor(ids_ptr, cute.make_layout(4096 * 8))
    selected = cute.make_tensor(selected_ptr, cute.make_layout(4096 * 8))
    route_denom = cute.make_tensor(route_denom_ptr, cute.make_layout(4096))
    norm_denom = cute.make_tensor(norm_denom_ptr, cute.make_layout(4096))
    frequencies = cute.make_tensor(frequencies_ptr, cute.make_layout(256))
    grad_weights = cute.make_tensor(
        grad_weights_ptr, cute.make_layout(max(1, 4096 * stride_t + 8 * stride_k))
    )
    grad_raw = cute.make_tensor(grad_raw_ptr, cute.make_layout(1))
    thread = arch.thread_idx()[0]
    lane = thread & 31
    warp = thread >> 5
    token = arch.block_idx()[0] * 8 + warp
    smem = SmemAllocator()
    route_gradients = smem.allocate_tensor(Float32, 8 * 256, byte_alignment=16)
    coefficients = smem.allocate_tensor(Float32, 256, byte_alignment=16)
    if cutlass.const_expr(has_route):
        zeros = cute.make_rmem_tensor(8, Float32)
        zeros.fill(Float32(0))
        row_store(route_gradients.iterator + warp * 256 + lane * 4, zeros, True)
    if cutlass.const_expr(has_aux):
        auxiliary_upstream = grad_raw[0]
        if cutlass.const_expr(neg_aux):
            auxiliary_upstream = -auxiliary_upstream
        coefficients[thread] = mul(auxiliary_upstream, frequencies[thread])
        arch.sync_threads()
    elif cutlass.const_expr(has_route):
        arch.sync_warp()
    if cutlass.const_expr(has_route):
        if lane < 8:
            route_upstream = grad_weights[token * stride_t + lane * stride_k]
            if cutlass.const_expr(neg_route):
                route_upstream = -route_upstream
            scaled = mul(route_upstream, Float32(2.5))
            denominator = route_denom[token]
            contribution = mul(
                -scaled, div(div(selected[token * 8 + lane], denominator), denominator)
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
    if cutlass.const_expr(has_aux):
        denominator = norm_denom[token]
        reciprocal = row_reciprocal(denominator)
        coefficient = cute.make_rmem_tensor(8, Float32)
        for i in cutlass.range_constexpr(8):
            coefficient[i] = coefficients[lane * 4 + (i & 3) + (i // 4) * 128]
        direct = row_divide(coefficient, denominator, reciprocal)
        normalized = row_divide(scores, denominator, reciprocal)
        quotients = row_divide(normalized, denominator, reciprocal)
        terms = cute.make_rmem_tensor(8, Float32)
        for i in cutlass.range_constexpr(8):
            terms[i] = mul(-coefficient[i], quotients[i])
        norm_gradient = choose(
            norms[token] > Float32(1.0e-12), row_sum(terms, True), Float32(0)
        )
        for i in cutlass.range_constexpr(8):
            sign = Float32(scores[i] > Float32(0)) - Float32(scores[i] < Float32(0))
            gradients[i] = add(direct[i], mul(sign, norm_gradient))
    for i in cutlass.range_constexpr(8):
        if cutlass.const_expr(has_route):
            route_gradient = route_gradients[
                warp * 256 + lane * 4 + (i & 3) + (i // 4) * 128
            ]
            if cutlass.const_expr(has_aux):
                gradients[i] = add(gradients[i], route_gradient)
            else:
                gradients[i] = route_gradient
        gradients[i] = mul(mul(gradients[i], add(Float32(1), -scores[i])), scores[i])
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


def forward(logits_TE, expert_bias_E, arrival_counter):
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
        empty(()),
        empty((256,), torch.int64),
        empty((4096, 256)),
        empty((4096, 1)),
        empty((4096, 8)),
        empty((4096, 1)),
        empty((4096, 1)),
        empty((256,)),
    )
    partition_sums = empty((64, 256))
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
            partition_sums,
            partition_counts,
        ),
        expert_bias_E is not None,
        logits_TE.data_ptr() % 16 == 0,
    )
    return outputs


def backward(
    scores_TE,
    row_norm_T1,
    expert_ids_TK,
    selected_scores_TK,
    route_denominator_T1,
    norm_denominator_T1,
    frequencies_E,
    grad_weights_TK,
    grad_raw_sum,
):
    for tensor, shape, dtype, name in (
        (scores_TE, (4096, 256), torch.float32, "scores"),
        (row_norm_T1, (4096, 1), torch.float32, "row norms"),
        (expert_ids_TK, (4096, 8), torch.int64, "expert IDs"),
        (selected_scores_TK, (4096, 8), torch.float32, "selected scores"),
        (route_denominator_T1, (4096, 1), torch.float32, "route denominator"),
        (norm_denominator_T1, (4096, 1), torch.float32, "norm denominator"),
        (frequencies_E, (256,), torch.float32, "frequencies"),
    ):
        check_tensor(tensor, shape, dtype, scores_TE.device, name)
    for gradient, shape in ((grad_weights_TK, (4096, 8)), (grad_raw_sum, ())):
        if gradient is not None and (
            gradient.shape != shape
            or gradient.dtype != torch.float32
            or gradient.device != scores_TE.device
        ):
            raise ValueError(f"router gradient requires CUDA FP32 {shape}")
    grad_logits_TE = torch.empty_like(scores_TE)
    has_route, has_aux = grad_weights_TK is not None, grad_raw_sum is not None
    strides = grad_weights_TK.stride() if has_route else (0, 0)
    flags = (
        has_route,
        has_aux,
        *strides,
        grad_weights_TK.is_neg() if has_route else False,
        grad_raw_sum.is_neg() if has_aux else False,
        torch.are_deterministic_algorithms_enabled(),
        scores_TE.data_ptr() % 16 == 0,
    )
    launch(
        _compiled_kernels,
        "backward",
        launch_backward,
        (
            scores_TE,
            row_norm_T1,
            expert_ids_TK,
            selected_scores_TK,
            route_denominator_T1,
            norm_denominator_T1,
            frequencies_E,
            grad_weights_TK if has_route else scores_TE,
            grad_raw_sum if has_aux else scores_TE,
            grad_logits_TE,
        ),
        flags,
    )
    return grad_logits_TE
