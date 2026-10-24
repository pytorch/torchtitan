# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from Yidi Wu (@ydwu4), pytorch/torchtitan#5258.
# Source revision: a5d8990c5e9b46e3a87007b33de8e03c09fb97b7.
# pyrefly: ignore-errors

"""Experimental DeepSeek V3 learned routing with fused forward/backward.

The gate stays native; fused CUDA kernels consume BF16 or FP32 [T, E] logits.
The tuned CuTe specialization requires nvidia-cutlass-dsl >= 4.8.0; other
shapes use parameterized Triton kernels. T = tokens, E = experts, K = top-k."""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import spmd_types as spmd
import torch
import torch_remat as remat
import triton
import triton.language as tl
from packaging.version import Version
from torch.autograd.function import once_differentiable
from torchtitan.config import derive, override
from torchtitan.distributed.spmd_types import spmd_mesh_size
from torchtitan.models.common.activation import Sigmoid
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.linear.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.moe import RoundRobinTokenChoiceTopKRouter
from torchtitan.models.deepseek_v3.moe import DeepSeekV3Router
from triton.language.extra.cuda import libdevice


def check_tensor(tensor, shape, dtype, device, name, *, alignment=1):
    """Validate metadata; inspect storage alignment only when explicitly required."""
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
    if alignment > 1 and tensor.data_ptr() % alignment:
        raise ValueError(f"{name} requires {alignment}-byte alignment")


@triton.jit
def _ordering_key(values):
    """Order FP32 values as unsigned integers, with NaNs above infinity."""
    bits = values.to(tl.uint32, bitcast=True)
    flip = tl.where((bits & 0x80000000) != 0, 0xFFFFFFFF, 0x80000000).to(tl.uint32)
    return tl.where((bits & 0x7FFFFFFF) > 0x7F800000, 0xFFFFFFFF, bits ^ flip).to(
        tl.uint32
    )


@triton.jit
def _topk_mask(values, indices, valid, K: tl.constexpr):
    """Select K values with lowest-index tie breaking and return the threshold."""
    keys = tl.where(valid, _ordering_key(values), 0).to(tl.uint32)
    selected = tl.full(values.shape, False, tl.int1)
    threshold = tl.full((), 0, tl.uint32)
    for _ in range(K):
        threshold = tl.max(keys, 0)
        index = tl.min(tl.where(valid & (keys == threshold), indices, 0x7FFFFFFF), 0)
        chosen = valid & (indices == index)
        selected |= chosen
        keys = tl.where(chosen, 0, keys).to(tl.uint32)
    return selected, threshold


@triton.jit
def _row_gather(values, indices, size: tl.constexpr):
    """Gather register values with zero-filled tails and bounded indices."""
    loaded = tl.gather(values, indices % values.shape[0], 0)
    return tl.where(indices < size, loaded, 0.0)


@triton.jit
def _native_row_sum(
    values,
    token,
    TOKENS: tl.constexpr,
    SIZE: tl.constexpr,
    OUTPUT_MAJOR: tl.constexpr = False,
    ALIGNMENT_OFFSET: tl.constexpr = 0,
):
    """Preserve ATen's FP32 virtual-thread, accumulator and warp reduction order.

    Block width depends on input/output sizes. At128 columns ATen switches
    to aligned vectors with four accumulators, including row head/tail values.
    Fold accumulators before the warp tree, not after it.
    """
    VECTORIZED: tl.constexpr = SIZE >= 128 and not OUTPUT_MAJOR
    if OUTPUT_MAJOR:
        # Transposed upstreams make the normalizer-gradient temporary token-major.
        # ATen vectorizes outputs and splits the reduction over block.y only
        # when enough inputs remain per output thread.
        OUTPUT_VEC: tl.constexpr = (
            4 if TOKENS % 4 == 0 else (2 if TOKENS % 2 == 0 else 1)
        )
        MAX_THREADS: tl.constexpr = 512 // OUTPUT_VEC
        DIM_POW2: tl.constexpr = min(
            MAX_THREADS, triton.next_power_of_2(TOKENS // OUTPUT_VEC + 1) // 2
        )
        INITIAL_WIDTH: tl.constexpr = min(DIM_POW2, 32)
        HEIGHT: tl.constexpr = min(
            triton.next_power_of_2(SIZE + 1) // 2, MAX_THREADS // INITIAL_WIDTH
        )
        WIDTH: tl.constexpr = HEIGHT if SIZE >= min(HEIGHT * 16, 256) else 1
    else:
        DIM: tl.constexpr = SIZE // 4 if VECTORIZED else SIZE
        DIM_POW2: tl.constexpr = triton.next_power_of_2(DIM + 1) // 2
        OUTPUT_POW2: tl.constexpr = min(512, triton.next_power_of_2(TOKENS + 1) // 2)
        INITIAL_WIDTH: tl.constexpr = min(DIM_POW2, 32)
        HEIGHT: tl.constexpr = min(OUTPUT_POW2, 512 // INITIAL_WIDTH)
        WIDTH: tl.constexpr = min(DIM_POW2, 512 // HEIGHT)
    lanes = tl.arange(0, WIDTH)
    first = tl.full((WIDTH,), 0.0, tl.float32)
    second = tl.full((WIDTH,), 0.0, tl.float32)
    third = tl.full((WIDTH,), 0.0, tl.float32)
    fourth = tl.full((WIDTH,), 0.0, tl.float32)
    if VECTORIZED:
        shift = (token * SIZE + ALIGNMENT_OFFSET) % 4
        prefix = tl.where(shift > 0, 4 - shift, 0)
        head = _row_gather(values, (lanes - shift) % values.shape[0], SIZE)
        first = tl.where((shift > 0) & (lanes >= shift) & (lanes < 4), head, 0.0)
        remaining = SIZE - prefix
        for part in range((SIZE + 4 * WIDTH - 1) // (4 * WIDTH)):
            base = prefix + (part * WIDTH + lanes) * 4
            full = base + 3 < SIZE
            first += tl.where(full, _row_gather(values, base, SIZE), 0.0)
            second += tl.where(full, _row_gather(values, base + 1, SIZE), 0.0)
            third += tl.where(full, _row_gather(values, base + 2, SIZE), 0.0)
            fourth += tl.where(full, _row_gather(values, base + 3, SIZE), 0.0)
        tail = prefix + remaining - remaining % 4 + lanes
        first += tl.where(lanes < remaining % 4, _row_gather(values, tail, SIZE), 0.0)
    else:
        for part in range((SIZE + 4 * WIDTH - 1) // (4 * WIDTH)):
            base = part * 4 * WIDTH + lanes
            first += _row_gather(values, base, SIZE)
            second += _row_gather(values, base + WIDTH, SIZE)
            third += _row_gather(values, base + 2 * WIDTH, SIZE)
            fourth += _row_gather(values, base + 3 * WIDTH, SIZE)
    folded = ((first + second) + third) + fourth
    if WIDTH > 32:
        folded = tl.sum(tl.reshape(folded, (WIDTH // 32, 32)), 0)
    return tl.sum(folded, 0)


@triton.jit
def _generic_router_forward_kernel(
    Logits,
    Bias,
    Padding,
    Weights,
    Ids,
    RoutingMap,
    Counts,
    Scores,
    Selected,
    Denominator,
    TOKENS: tl.constexpr,
    E: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    L: tl.constexpr,
    P: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_PADDING: tl.constexpr,
    FORCED: tl.constexpr,
    SCALE: tl.constexpr,
    EPSILON: tl.constexpr,
):
    """Fuse sigmoid, group/top-k selection, weights, map and integer usage counts."""
    token = tl.program_id(0).to(tl.int64)
    groups = tl.arange(0, BLOCK_G)
    columns = tl.arange(0, BLOCK_P)
    experts = groups[:, None] * P + columns[None, :]
    valid = (groups[:, None] < G) & (columns[None, :] < P)
    logits = tl.load(Logits + token * E + experts, valid, other=0).to(tl.float32)
    scores = tl.div_rn(1.0, 1.0 + libdevice.exp(-logits))
    choice_scores = scores
    if HAS_BIAS:
        choice_scores += tl.load(Bias + experts, valid, other=0)
    if G > 1 and not FORCED:
        keys = tl.where(valid, _ordering_key(choice_scores), 0).to(tl.uint32)
        first_key = tl.max(keys, 1)
        first_id = tl.min(
            tl.where(
                valid & (keys == first_key[:, None]), columns[None, :], 0x7FFFFFFF
            ),
            1,
        )
        first = tl.sum(
            tl.where(columns[None, :] == first_id[:, None], choice_scores, 0.0), 1
        )
        keys = tl.where(columns[None, :] == first_id[:, None], 0, keys).to(tl.uint32)
        second_key = tl.max(keys, 1)
        second_id = tl.min(
            tl.where(
                valid & (keys == second_key[:, None]), columns[None, :], 0x7FFFFFFF
            ),
            1,
        )
        second = tl.sum(
            tl.where(columns[None, :] == second_id[:, None], choice_scores, 0.0), 1
        )
        chosen_groups, _ = _topk_mask(first + second, groups, groups < G, L)
        choice_valid = valid & chosen_groups[:, None]
    else:
        choice_valid = valid
    flat_experts = tl.reshape(experts, (BLOCK_G * BLOCK_P,))
    flat_scores = tl.reshape(scores, (BLOCK_G * BLOCK_P,))
    flat_valid = tl.reshape(valid, (BLOCK_G * BLOCK_P,))
    choices = tl.arange(0, BLOCK_K)
    if FORCED:
        ids = ((token * K + choices) % E).to(tl.int32)
        selected = (
            tl.sum((flat_experts[:, None] == ids[None, :]) & (choices[None, :] < K), 1)
            != 0
        )
    else:
        flat_choice = tl.reshape(choice_scores, (BLOCK_G * BLOCK_P,))
        selected, threshold = _topk_mask(
            flat_choice, flat_experts, tl.reshape(choice_valid, (BLOCK_G * BLOCK_P,)), K
        )
        # Match unsorted gather-topk: larger values in index order, then threshold ties.
        order = tl.where(
            selected,
            flat_experts + tl.where(_ordering_key(flat_choice) == threshold, E, 0),
            0x7FFFFFFF,
        )
        ordered = tl.sort(order, descending=False)
        ids = tl.gather(ordered, choices, 0) % E
    picked = tl.gather(flat_scores, (ids // P) * BLOCK_P + ids % P, 0)
    denominator = (
        _native_row_sum(tl.where(choices < K, picked, 0.0), token, TOKENS, K) + EPSILON
    )
    tl.store(
        Weights + token * K + choices,
        tl.div_rn(picked, denominator) * SCALE,
        choices < K,
    )
    tl.store(Ids + token * K + choices, ids.to(tl.int64), choices < K)
    tl.store(Selected + token * K + choices, picked, choices < K)
    tl.store(Denominator + token, denominator)
    tl.store(Scores + token * E + flat_experts, flat_scores, flat_valid)
    tl.store(RoutingMap + token * E + flat_experts, selected, flat_valid)
    counted = tl.full((), True, tl.int1)
    if HAS_PADDING:
        counted = ~tl.load(Padding + token)
    tl.atomic_add(Counts + ids, 1, (choices < K) & counted, sem="relaxed")


@triton.jit
def _generic_router_backward_kernel(
    Scores,
    Ids,
    Selected,
    Denominator,
    GradWeights,
    GradScores,
    Output,
    TOKENS: tl.constexpr,
    E: tl.constexpr,
    K: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_ROUTE: tl.constexpr,
    HAS_SCORE: tl.constexpr,
    BF16_LOGITS: tl.constexpr,
    ROUTE_STRIDE_T: tl.constexpr,
    ROUTE_STRIDE_K: tl.constexpr,
    SCORE_STRIDE_T: tl.constexpr,
    SCORE_STRIDE_E: tl.constexpr,
    SCALE: tl.constexpr,
):
    """Combine score branches, preserving independent BF16 logit-gradient rounding."""
    token = tl.program_id(1).to(tl.int64)
    experts = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
    valid = experts < E
    scores = tl.load(Scores + token * E + experts, valid, other=0)
    route_gradient = tl.full((BLOCK_E,), 0.0, tl.float32)
    if HAS_ROUTE:
        choices = tl.arange(0, BLOCK_K)
        ids = tl.load(Ids + token * K + choices, choices < K, other=-1)
        picked = tl.load(Selected + token * K + choices, choices < K, other=0)
        upstream = tl.load(
            GradWeights + token * ROUTE_STRIDE_T + choices * ROUTE_STRIDE_K,
            choices < K,
            other=0,
        ).to(tl.float32)
        if BF16_LOGITS:
            upstream = upstream.to(tl.bfloat16).to(tl.float32)
        scaled = upstream * SCALE
        denominator = tl.load(Denominator + token)
        norm_gradient = _native_row_sum(
            -scaled * tl.div_rn(tl.div_rn(picked, denominator), denominator),
            token,
            TOKENS,
            K,
            ROUTE_STRIDE_T > 0 and ROUTE_STRIDE_K > ROUTE_STRIDE_T,
        )
        routed = tl.div_rn(scaled, denominator) + norm_gradient
        route_gradient = tl.sum(
            tl.where(experts[:, None] == ids[None, :], routed[None, :], 0.0), 1
        )
    score_gradient = tl.full((BLOCK_E,), 0.0, tl.float32)
    if HAS_SCORE:
        score_gradient = tl.load(
            GradScores + token * SCORE_STRIDE_T + experts * SCORE_STRIDE_E,
            valid,
            other=0,
        )
    if BF16_LOGITS:
        route_gradient = (
            (route_gradient * scores * (1.0 - scores)).to(tl.bfloat16).to(tl.float32)
        )
        score_gradient = (
            (score_gradient * scores * (1.0 - scores)).to(tl.bfloat16).to(tl.float32)
        )
        result = route_gradient + score_gradient
    else:
        result = ((score_gradient + route_gradient) * (1.0 - scores)) * scores
    tl.store(Output + token * E + experts, result, valid)


def _generic_router_forward(
    logits, bias, counter, forced, padding, top_k, groups, limited, scale, epsilon
):
    """Allocate general-shape outputs; fresh integer counts make calls stream-independent."""
    tokens, experts = logits.shape

    def empty(shape, dtype):
        """Allocate one output on the logits device."""
        return torch.empty(shape, dtype=dtype, device=logits.device)

    outputs = (
        empty((tokens, top_k), logits.dtype),
        empty((tokens, top_k), torch.int64),
        empty((tokens, experts), torch.bool),
        torch.zeros(experts, dtype=torch.int64, device=logits.device),
        empty((tokens, experts), torch.float32),
        empty((tokens, top_k), torch.float32),
        empty((tokens, 1), torch.float32),
    )
    _generic_router_forward_kernel[(tokens,)](
        logits,
        bias if bias is not None else logits,
        padding if padding is not None else logits,
        *outputs,
        tokens,
        experts,
        top_k,
        groups,
        limited,
        experts // groups,
        triton.next_power_of_2(groups),
        triton.next_power_of_2(experts // groups),
        triton.next_power_of_2(top_k),
        bias is not None,
        padding is not None,
        forced,
        scale,
        epsilon,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return outputs


def _generic_router_backward(
    scores, ids, selected, denominator, grad_weights, grad_scores, bf16_logits, scale
):
    """Launch a row/expert-tiled derivative without materializing scatter temporaries."""
    tokens, experts = scores.shape
    top_k = ids.shape[1]
    if grad_weights is not None:
        grad_weights = grad_weights.resolve_neg()
    if grad_scores is not None:
        grad_scores = grad_scores.resolve_neg()
    output = torch.empty_like(
        scores, dtype=torch.bfloat16 if bf16_logits else torch.float32
    )
    _generic_router_backward_kernel[(triton.cdiv(experts, 128), tokens)](
        scores,
        ids,
        selected,
        denominator,
        grad_weights if grad_weights is not None else scores,
        grad_scores if grad_scores is not None else scores,
        output,
        tokens,
        experts,
        top_k,
        128,
        triton.next_power_of_2(top_k),
        grad_weights is not None,
        grad_scores is not None,
        bf16_logits,
        *(grad_weights.stride() if grad_weights is not None else (0, 0)),
        *(grad_scores.stride() if grad_scores is not None else (0, 0)),
        scale,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


try:
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda
    from cutlass import BFloat16, Float32, Int32, Int64, Uint32, Uint8
    from cutlass.cute import arch
    from cutlass.cute.runtime import make_ptr
    from cutlass.memory import SmemAllocator

    if Version(cutlass.__version__) < Version("4.8.0"):
        raise ImportError("DSv3 routing kernels require nvidia-cutlass-dsl>=4.8.0.")
    _CUTEDSL_IMPORT_ERROR: ImportError | None = None
except ImportError as error:
    _CUTEDSL_IMPORT_ERROR = error


if TYPE_CHECKING or _CUTEDSL_IMPORT_ERROR is None:

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
            torch.bfloat16: BFloat16,
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
    def forward_kernel(
        args,
        has_bias: cutlass.Constexpr,
        aligned: cutlass.Constexpr,
        force_balancing: cutlass.Constexpr,
        has_padding: cutlass.Constexpr,
    ):
        logits_ptr, bias_ptr, arrivals_ptr = args[:3]
        weights_ptr, ids_ptr, map_ptr, dispatch_ptr = args[3:7]
        scores_ptr, selected_ptr, route_denom_ptr = args[7:10]
        partition_counts_ptr, padding_ptr = args[10:]
        logits = cute.make_tensor(logits_ptr, cute.make_layout(4096 * 256))
        padding = cute.make_tensor(padding_ptr, cute.make_layout(4096))
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
        if cutlass.const_expr(logits.element_type == BFloat16):
            for i in cutlass.range_constexpr(8):
                values[i] = Float32(
                    logits[token * 256 + lane * 4 + (i & 3) + (i // 4) * 128]
                )
        elif cutlass.const_expr(aligned):
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
        if cutlass.const_expr(force_balancing):
            first_expert = Int32((token * 8) % 256)
            dispatch_selected = cute.make_rmem_tensor(8, Int32)
            for i in cutlass.range_constexpr(8):
                candidate_expert = lane * 4 + (i & 3) + (i // 4) * 128
                dispatch_selected[i] = Int32(
                    (candidate_expert >= first_expert)
                    & (candidate_expert < first_expert + 8)
                )
        else:
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
            if cutlass.const_expr(force_balancing):
                selected_expert = first_expert + lane
            else:
                selected_expert = native_order(
                    choice_expert, choice_key, threshold, lane
                )
            selected = shared_values[warp * 256 + selected_expert]
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
            weights[token * 8 + lane] = weights.element_type(
                mul(div(selected, route_denominator), Float32(2.5))
            )
            ids[token * 8 + lane] = Int64(selected_expert)
            selected_output[token * 8 + lane] = selected
            if lane == 0:
                route_denom[token] = route_denominator
        selected = Uint32(0)
        for i in cutlass.range_constexpr(8):
            map_expert = lane * 4 + (i & 3) + (i // 4) * 128
            routing_map[token * 256 + map_expert] = Uint8(dispatch_selected[i])
            selected |= Uint32(dispatch_selected[i]) << i
        if cutlass.const_expr(has_padding):
            selected = choose(padding[token] != 0, Uint32(0), selected)
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
        bf16_logits: cutlass.Constexpr,
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
        output = cute.make_tensor(output_ptr, cute.make_layout(4096 * 256))
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
                route_upstream = Float32(
                    grad_weights[token * stride_t + lane * stride_k]
                )
                if cutlass.const_expr(neg_route):
                    route_upstream = -route_upstream
                if cutlass.const_expr(bf16_logits):
                    route_upstream = Float32(BFloat16(route_upstream))
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
                if cutlass.const_expr(bf16_logits):
                    route_gradient = Float32(
                        BFloat16(
                            mul(
                                mul(route_gradient, scores[i]),
                                add(Float32(1), -scores[i]),
                            )
                        )
                    )
                if cutlass.const_expr(has_score_grad and not bf16_logits):
                    gradients[i] = add(gradients[i], route_gradient)
                elif cutlass.const_expr(not bf16_logits):
                    gradients[i] = route_gradient
            if cutlass.const_expr(bf16_logits):
                if cutlass.const_expr(has_score_grad):
                    gradients[i] = Float32(
                        BFloat16(
                            mul(
                                mul(gradients[i], scores[i]),
                                add(Float32(1), -scores[i]),
                            )
                        )
                    )
                    if cutlass.const_expr(has_route):
                        gradients[i] = add(gradients[i], route_gradient)
                else:
                    gradients[i] = route_gradient
                output[token * 256 + lane * 4 + (i & 3) + (i // 4) * 128] = BFloat16(
                    gradients[i]
                )
            else:
                gradients[i] = mul(
                    mul(gradients[i], add(Float32(1), -scores[i])), scores[i]
                )
        if cutlass.const_expr(not bf16_logits):
            row_store(output_ptr + token * 256 + lane * 4, gradients)

    @cute.jit
    def launch_forward(
        args,
        stream,
        has_bias: cutlass.Constexpr,
        aligned: cutlass.Constexpr,
        force_balancing: cutlass.Constexpr,
        has_padding: cutlass.Constexpr,
    ):
        forward_kernel(args, has_bias, aligned, force_balancing, has_padding).launch(
            grid=(512, 1, 1), block=(256, 1, 1), cluster=(8, 1, 1), stream=stream
        )

    @cute.jit
    def launch_backward(args, stream, flags: cutlass.Constexpr):
        backward_kernel(args, *flags).launch(
            grid=(512, 1, 1), block=(256, 1, 1), stream=stream
        )

    _compiled_kernels = {}

    def _kernel_forward(
        logits_TE,
        expert_bias_E,
        arrival_counter,
        force_balancing=False,
        padding_mask_T=None,
    ):
        logits_TE = logits_TE.resolve_neg()
        if logits_TE.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("router logits must be BF16 or FP32")
        check_tensor(
            logits_TE, (4096, 256), logits_TE.dtype, logits_TE.device, "logits"
        )
        if padding_mask_T is not None:
            check_tensor(
                padding_mask_T, (4096,), torch.bool, logits_TE.device, "padding mask"
            )
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
            empty((4096, 8), logits_TE.dtype),
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
                padding_mask_T if padding_mask_T is not None else logits_TE,
            ),
            expert_bias_E is not None,
            logits_TE.data_ptr() % 16 == 0,
            force_balancing,
            padding_mask_T is not None,
        )
        return outputs

    def _kernel_backward(
        scores_TE,
        expert_ids_TK,
        selected_scores_TK,
        route_denominator_T1,
        grad_weights_TK,
        grad_scores_TE,
        bf16_logits=False,
    ):
        for tensor, shape, dtype, name in (
            (scores_TE, (4096, 256), torch.float32, "scores"),
            (expert_ids_TK, (4096, 8), torch.int64, "expert IDs"),
            (selected_scores_TK, (4096, 8), torch.float32, "selected scores"),
            (route_denominator_T1, (4096, 1), torch.float32, "route denominator"),
        ):
            check_tensor(tensor, shape, dtype, scores_TE.device, name)
        for gradient, shape, dtypes in (
            (grad_weights_TK, (4096, 8), (torch.float32, torch.bfloat16)),
            (grad_scores_TE, (4096, 256), (torch.float32,)),
        ):
            if gradient is not None and (
                gradient.shape != shape
                or gradient.dtype not in dtypes
                or gradient.device != scores_TE.device
            ):
                raise ValueError(f"router gradient requires CUDA FP32 {shape}")
        grad_logits_TE = torch.empty_like(
            scores_TE, dtype=torch.bfloat16 if bf16_logits else torch.float32
        )
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
            bf16_logits,
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


def _validate_router_inputs(logits, bias, counter, padding, top_k, groups, limited):
    """Check the general routing shape contract without reading tensor data."""
    if (
        logits.ndim != 2
        or min(logits.shape) <= 0
        or logits.dtype not in (torch.bfloat16, torch.float32)
    ):
        raise ValueError("router logits must be nonempty CUDA BF16 or FP32 [T,E]")
    tokens, experts = logits.shape
    if (
        not 0 < top_k <= experts
        or groups <= 0
        or experts % groups
        or not 0 < limited <= groups
    ):
        raise ValueError(
            "router requires 0 < top_k <= E, groups dividing E, and 0 < limited <= groups"
        )
    if groups > 1 and (experts // groups < 2 or top_k > limited * (experts // groups)):
        raise ValueError(
            "selected groups must contain top_k experts and at least two experts per group"
        )
    check_tensor(logits, (tokens, experts), logits.dtype, logits.device, "logits")
    check_tensor(counter, (1,), torch.int32, logits.device, "arrival counter")
    if bias is not None:
        check_tensor(bias, (experts,), torch.float32, logits.device, "expert bias")
    if padding is not None:
        check_tensor(padding, (tokens,), torch.bool, logits.device, "padding mask")


def _validate_router_backward(
    scores, ids, selected, denominator, grad_weights, grad_scores
):
    """Reject inconsistent saved-state and gradient metadata before device launches."""
    if scores.ndim != 2 or min(scores.shape) <= 0 or ids.ndim != 2:
        raise ValueError("router backward requires scores[T,E] and IDs[T,K]")
    tokens, experts = scores.shape
    top_k = ids.shape[1]
    if not 0 < top_k <= experts:
        raise ValueError("router backward requires 0 < K <= E")
    for tensor, shape, dtype, name in (
        (scores, (tokens, experts), torch.float32, "scores"),
        (ids, (tokens, top_k), torch.int64, "expert IDs"),
        (selected, (tokens, top_k), torch.float32, "selected scores"),
        (denominator, (tokens, 1), torch.float32, "route denominator"),
    ):
        check_tensor(tensor, shape, dtype, scores.device, name)
    for gradient, shape, dtypes in (
        (grad_weights, (tokens, top_k), (torch.float32, torch.bfloat16)),
        (grad_scores, (tokens, experts), (torch.float32,)),
    ):
        if gradient is not None and (
            gradient.shape != shape
            or gradient.dtype not in dtypes
            or gradient.device != scores.device
        ):
            raise ValueError(
                f"router gradient must have shape {shape}, dtype {dtypes}, and device {scores.device}"
            )


def _use_tuned_router(logits, forced, top_k, groups, limited, scale, epsilon):
    """Select the preserved CuTe launch; all other shapes use real fused kernels."""
    return (
        _CUTEDSL_IMPORT_ERROR is None
        and logits.shape == (4096, 256)
        and top_k == 8
        and (forced or (groups == 8 and limited == 4))
        and scale == 2.5
        and epsilon == 1e-20
        and torch.cuda.get_device_capability(logits.device) == (10, 3)
        and torch.cuda.get_device_properties(logits.device).multi_processor_count >= 128
    )


@torch.library.custom_op(
    "torchtitan::dsv3_router_forward",
    mutates_args=("arrival_counter",),
    device_types="cuda",
)
def router_forward_op(
    logits_TE: torch.Tensor,
    expert_bias_E: torch.Tensor | None,
    arrival_counter: torch.Tensor,
    force_balancing: bool = False,
    padding_mask_T: torch.Tensor | None = None,
    top_k: int = 8,
    num_expert_groups: int = 8,
    num_limited_groups: int = 4,
    route_scale: float = 2.5,
    route_norm_epsilon: float = 1e-20,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return routing outputs and compact backward state for legal CUDA logits.

    Outputs are weights, expert IDs, routing map, counts, FP32 scores, selected
    scores and normalization denominators. Only the synchronization counter
    mutates; discrete bias/selection does not receive gradients.
    """
    logits_TE = logits_TE.resolve_neg()
    if expert_bias_E is not None:
        expert_bias_E = expert_bias_E.resolve_neg()
    _validate_router_inputs(
        logits_TE,
        expert_bias_E,
        arrival_counter,
        padding_mask_T,
        top_k,
        num_expert_groups,
        num_limited_groups,
    )
    if _use_tuned_router(
        logits_TE,
        force_balancing,
        top_k,
        num_expert_groups,
        num_limited_groups,
        route_scale,
        route_norm_epsilon,
    ):
        return _kernel_forward(
            logits_TE, expert_bias_E, arrival_counter, force_balancing, padding_mask_T
        )
    return _generic_router_forward(
        logits_TE,
        expert_bias_E,
        arrival_counter,
        force_balancing,
        padding_mask_T,
        top_k,
        num_expert_groups,
        num_limited_groups,
        route_scale,
        route_norm_epsilon,
    )


@router_forward_op.register_fake
def _router_forward_fake(
    logits_TE,
    expert_bias_E,
    arrival_counter,
    force_balancing=False,
    padding_mask_T=None,
    top_k=8,
    num_expert_groups=8,
    num_limited_groups=4,
    route_scale=2.5,
    route_norm_epsilon=1e-20,
):
    """Validate metadata and infer output shapes without reading tensor storage."""
    logits_TE = logits_TE.resolve_neg()
    if expert_bias_E is not None:
        expert_bias_E = expert_bias_E.resolve_neg()
    _validate_router_inputs(
        logits_TE,
        expert_bias_E,
        arrival_counter,
        padding_mask_T,
        top_k,
        num_expert_groups,
        num_limited_groups,
    )
    tokens, experts = logits_TE.shape
    return (
        logits_TE.new_empty((tokens, top_k)),
        logits_TE.new_empty((tokens, top_k), dtype=torch.int64),
        logits_TE.new_empty((tokens, experts), dtype=torch.bool),
        logits_TE.new_empty((experts,), dtype=torch.int64),
        logits_TE.new_empty((tokens, experts), dtype=torch.float32),
        logits_TE.new_empty((tokens, top_k), dtype=torch.float32),
        logits_TE.new_empty((tokens, 1), dtype=torch.float32),
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
    bf16_logits: bool = False,
    route_scale: float = 2.5,
) -> torch.Tensor:
    """Combine score/weight derivatives with the native logit-gradient precision."""
    _validate_router_backward(
        scores_TE,
        expert_ids_TK,
        selected_scores_TK,
        route_denominator_T1,
        grad_weights_TK,
        grad_scores_TE,
    )
    if (
        scores_TE.shape == (4096, 256)
        and expert_ids_TK.shape[1] == 8
        and route_scale == 2.5
        and _CUTEDSL_IMPORT_ERROR is None
        and torch.cuda.get_device_capability(scores_TE.device) == (10, 3)
    ):
        return _kernel_backward(
            scores_TE,
            expert_ids_TK,
            selected_scores_TK,
            route_denominator_T1,
            grad_weights_TK,
            grad_scores_TE,
            bf16_logits,
        )
    return _generic_router_backward(
        scores_TE,
        expert_ids_TK,
        selected_scores_TK,
        route_denominator_T1,
        grad_weights_TK,
        grad_scores_TE,
        bf16_logits,
        route_scale,
    )


@router_backward_op.register_fake
def _router_backward_fake(
    scores_TE,
    expert_ids_TK,
    selected_scores_TK,
    route_denominator_T1,
    grad_weights_TK,
    grad_scores_TE,
    bf16_logits=False,
    route_scale=2.5,
):
    """Infer the contiguous gradient's shape/dtype using only saved metadata."""
    _validate_router_backward(
        scores_TE,
        expert_ids_TK,
        selected_scores_TK,
        route_denominator_T1,
        grad_weights_TK,
        grad_scores_TE,
    )
    return torch.empty_like(
        scores_TE,
        dtype=torch.bfloat16 if bf16_logits else torch.float32,
        memory_format=torch.contiguous_format,
    )


class FusedDSv3RouterFunction(torch.autograd.Function):
    """Learned top-k routing with differentiable scores for an external loss.

    Returns weights, expert IDs, routing map, counts, and sigmoid scores.
    Weights and scores are differentiable. Bias affects discrete selection,
    so it receives no gradient, as in the native router. Calls sharing the
    zero-initialized arrival counter must be stream-ordered.
    """

    @staticmethod
    def spmd_typecheck(
        outputs,
        *,
        logits_TE,
        expert_bias_E,
        arrival_counter,
        force_balancing=False,
        padding_mask_T=None,
        top_k=8,
        num_expert_groups=8,
        num_limited_groups=4,
        route_scale=2.5,
        route_norm_epsilon=1e-20,
    ):
        spmd.rules.ignore(arrival_counter)
        weights_TK, ids_TK, routing_map_TE, counts_E, scores_TE = outputs
        operands = (logits_TE,) if expert_bias_E is None else (logits_TE, expert_bias_E)
        rows = "t_->t_" if expert_bias_E is None else "t_,_->t_"
        if padding_mask_T is not None:
            operands = (*operands, padding_mask_T)
            rows = "t_,t->t_" if expert_bias_E is None else "t_,_,t->t_"
        for output in (weights_TK, ids_TK, routing_map_TE, scores_TE):
            spmd.rules.einsum(rows, *operands, out=output)
        spmd.rules.einsum("t_->_", routing_map_TE, out=counts_E)

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx,
        logits_TE,
        expert_bias_E,
        arrival_counter,
        force_balancing=False,
        padding_mask_T=None,
        top_k=8,
        num_expert_groups=8,
        num_limited_groups=4,
        route_scale=2.5,
        route_norm_epsilon=1e-20,
    ):
        (
            weights_TK,
            ids_TK,
            routing_map_TE,
            counts_E,
            scores_TE,
            selected_TK,
            route_denominator_T1,
        ) = router_forward_op(
            logits_TE,
            expert_bias_E,
            arrival_counter,
            force_balancing,
            padding_mask_T,
            top_k,
            num_expert_groups,
            num_limited_groups,
            route_scale,
            route_norm_epsilon,
        )
        ctx.bf16_logits = logits_TE.dtype == torch.bfloat16
        ctx.route_scale = route_scale
        ctx.num_inputs = len(ctx.needs_input_grad)
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
            return (None,) * ctx.num_inputs
        grad_logits = router_backward_op(
            *ctx.saved_tensors,
            grad_weights_TK,
            grad_scores_TE,
            ctx.bf16_logits,
            ctx.route_scale,
        )
        return (grad_logits, *(None for _ in range(ctx.num_inputs - 1)))


def _fused_routing(
    logits_TE,
    expert_bias_E,
    arrival_counter,
    force_balancing,
    padding_mask_T,
    top_k=8,
    num_expert_groups=8,
    num_limited_groups=4,
    route_scale=2.5,
    route_norm_epsilon=1e-20,
):
    """Expose a plain callable remat boundary around the autograd implementation."""
    return FusedDSv3RouterFunction.apply(
        logits_TE,
        expert_bias_E,
        arrival_counter,
        force_balancing,
        padding_mask_T,
        top_k,
        num_expert_groups,
        num_limited_groups,
        route_scale,
        route_norm_epsilon,
    )


def _with_local_counter_sharding(sharding_config):
    """Preserve inherited layouts and declare the counter as independent rank state."""
    if sharding_config is None:
        return None
    return replace(
        sharding_config,
        state_shardings={
            **sharding_config.state_shardings,
            "arrival_counter": spmd.SpmdType(
                {"dp": spmd.V, "cp": spmd.V, "tp": spmd.V}
            ),
        },
    )


class FusedDSv3Router(DeepSeekV3Router):
    """Router specialization with no knowledge of the auxiliary-loss formula."""

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV3Router.Config):
        force_load_balancing: bool = False
        require_fusion: bool = False

        def __post_init__(self):
            DeepSeekV3Router.Config.__post_init__(self)
            self.sharding_config = _with_local_counter_sharding(self.sharding_config)

    def __init__(self, config: Config):
        super().__init__(config)
        self.force_load_balancing = config.force_load_balancing
        self.require_fusion = config.require_fusion
        self.register_buffer(
            "arrival_counter", torch.zeros(1, dtype=torch.int32), persistent=False
        )

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None):
        """Initialize independent rank-local synchronization state after materialization."""
        super()._init_self_buffers(buffer_device=buffer_device)
        self.arrival_counter = torch.zeros(
            1, dtype=torch.int32, device=self.tokens_per_expert_E.device
        )

    def _supports_fusion(self, x_TD, expert_bias_E, padding_mask_T):
        """Check the router's numerical, layout, device and mesh fusion contract."""
        return (
            x_TD.is_cuda
            and x_TD.ndim == 2
            and x_TD.shape[0] > 0
            and type(self.score_func) is Sigmoid
            and (
                isinstance(self.gate, HiMidLoLinear)
                or (
                    type(self.gate) is Linear
                    and self.score_func.compute_dtype == torch.float32
                    and (
                        self.routing_weight_dtype == torch.bfloat16
                        or (
                            x_TD.dtype == torch.float32
                            and self.gate.weight.dtype == torch.float32
                        )
                    )
                )
            )
            and self.score_func.compute_dtype in (None, torch.float32)
            and not torch.is_autocast_enabled("cuda")
            and self.num_experts > 0
            and 0 < self.top_k <= self.num_experts
            and (
                self.force_load_balancing
                or self.num_expert_groups is None
                or (
                    self.num_expert_groups > 0
                    and self.num_experts % self.num_expert_groups == 0
                    and self.num_limited_groups is not None
                    and 0 < self.num_limited_groups <= self.num_expert_groups
                    and self.num_experts // self.num_expert_groups >= 2
                    and self.top_k
                    <= self.num_limited_groups
                    * (self.num_experts // self.num_expert_groups)
                )
            )
            and self.route_norm
            and (
                padding_mask_T is None
                or (
                    padding_mask_T.shape == (x_TD.shape[0],)
                    and padding_mask_T.dtype == torch.bool
                    and padding_mask_T.device == x_TD.device
                    and padding_mask_T.is_contiguous()
                )
            )
            and spmd_mesh_size("cp") == 1
            and spmd_mesh_size("tp") == 1
            and torch.cuda.get_device_capability(x_TD.device)[0] >= 8
            and (
                expert_bias_E is None
                or (
                    expert_bias_E.shape == (self.num_experts,)
                    and expert_bias_E.dtype == torch.float32
                    and expert_bias_E.device == x_TD.device
                    and expert_bias_E.is_contiguous()
                )
            )
        )

    def _select_experts(self, scores_TE, expert_bias_E=None, **router_kwargs):
        """Preserve round-robin selection on unsupported benchmark geometries."""
        if self.force_load_balancing:
            return RoundRobinTokenChoiceTopKRouter._select_experts(
                self, scores_TE, expert_bias_E, **router_kwargs
            )
        return super()._select_experts(scores_TE, expert_bias_E, **router_kwargs)

    def forward(
        self,
        x_TD,
        expert_bias_E=None,
        *,
        padding_mask_T=None,
        aux_loss_denominator=None,
        **router_kwargs,
    ):
        """Fuse routing while preserving native counters and auxiliary-loss ownership."""
        if self._supports_fusion(x_TD, expert_bias_E, padding_mask_T):
            logits_TE = self.gate(x_TD)
            remat.recompute_needs_tensor(logits_TE)
            weights_TK, ids_TK, routing_map_TE, counts_E, scores_TE = remat.region(
                _fused_routing, "routing_decision", recompute=False
            )(
                logits_TE,
                expert_bias_E,
                self.arrival_counter,
                self.force_load_balancing,
                padding_mask_T,
                self.top_k,
                1 if self.force_load_balancing else (self.num_expert_groups or 1),
                1 if self.force_load_balancing else (self.num_limited_groups or 1),
                self.route_scale,
                self.route_norm_epsilon,
            )
            if self.routing_weight_dtype is not None:
                weights_TK = weights_TK.to(self.routing_weight_dtype)
            remat.recompute_needs_tensor(ids_TK)
            if self.training and not remat.is_recomputing():
                with torch.no_grad():
                    self.tokens_per_expert_E.add_(counts_E)
            if self.training and self.aux_loss is not None:
                if aux_loss_denominator is None:
                    raise ValueError("An auxiliary-loss denominator is required.")
                remat.recompute_needs_tensor(scores_TE)
                remat.recompute_needs_tensor(routing_map_TE)
                masked_map_TE = (
                    routing_map_TE
                    if padding_mask_T is None
                    else routing_map_TE & ~padding_mask_T.unsqueeze(-1)
                )
                weights_TK = self.aux_loss(
                    scores_TE,
                    masked_map_TE,
                    carrier=weights_TK,
                    padding_mask_T=padding_mask_T,
                    denominator=aux_loss_denominator,
                )
            return weights_TK, ids_TK, routing_map_TE
        if self.require_fusion:
            raise ValueError(
                "Required DSv3 router fusion is unsupported for input "
                f"{tuple(x_TD.shape)} {x_TD.dtype} or the configured router/mesh/device"
            )
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
def fused_dsv3_router(
    cfg: DeepSeekV3Router.Config, *, require_fusion: bool = False
) -> FusedDSv3Router.Config:
    """Select fused learned routing; optionally reject unsupported runtime inputs."""
    result = derive(cfg, FusedDSv3Router.Config)
    result.require_fusion = require_fusion
    return result


@override(
    target=RoundRobinTokenChoiceTopKRouter.Config,
    exact=True,
    description="Fuse explicit round-robin DSv3 performance routing.",
)
def fused_dsv3_round_robin_router(
    cfg: RoundRobinTokenChoiceTopKRouter.Config, *, require_fusion: bool = False
) -> FusedDSv3Router.Config:
    """Use the same fused kernels while preserving forced balanced assignments."""
    result = derive(cfg, FusedDSv3Router.Config)
    result.force_load_balancing = True
    result.require_fusion = require_fusion
    return result
