# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Experimental DeepSeek V3 sequence-wise loss with fused forward/backward.

Current TorchTitan calls this objective ``MicrobatchWiseLoadBalanceLoss``.
The routing map, loss scaling, metric accumulation, and collectives keep their
native contracts. The kernel specializes FP32 scores shaped [4096, 256].
The specialized kernels require nvidia-cutlass-dsl >= 4.8.0."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import spmd_types as spmd
import torch
import torch.nn.functional as F
from packaging.version import Version
from torch.autograd.function import once_differentiable
from torchtitan.config import derive, override
from torchtitan.distributed.spmd_types import spmd_local_context, spmd_mesh_size
from torchtitan.models.common.moe import MicrobatchWiseLoadBalanceLoss
from torchtitan.models.deepseek_v3.moe import DeepSeekV3Router

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
        partition_sums = cute.make_tensor(
            partition_sums_ptr, cute.make_layout(64 * 256)
        )
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
        backward_kernel(*args).launch(
            grid=(512, 1, 1), block=(256, 1, 1), stream=stream
        )

    _compiled_kernels = {}

    def _kernel_forward(scores_TE, routing_map_TE, arrival_counter):
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

    def _kernel_backward(grad_raw_sum, scores_TE, frequencies_E):
        grad_raw_sum = grad_raw_sum.resolve_neg()
        check_tensor(scores_TE, (4096, 256), torch.float32, scores_TE.device, "scores")
        check_tensor(
            frequencies_E, (256,), torch.float32, scores_TE.device, "frequencies"
        )
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
            norm_grad_T1 = torch.where(
                norm_T1 > 1e-12, terms_TE.sum(-1, keepdim=True), 0.0
            )
            return direct_TE + scores_TE.sgn() * norm_grad_T1
        grad_scores_TE = torch.empty_like(scores_TE)
        launch(
            _compiled_kernels,
            "backward",
            launch_backward,
            (grad_raw_sum, scores_TE, frequencies_E, grad_scores_TE),
        )
        return grad_scores_TE


@torch.library.custom_op(
    "torchtitan::dsv3_seqwise_loss_forward",
    mutates_args=("arrival_counter",),
    device_types="cuda",
)
def seqwise_loss_forward_op(
    scores_TE: torch.Tensor,
    routing_map_TE: torch.Tensor,
    arrival_counter: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One CuTeDSL launch; the caller owns the stream-ordered arrival counter."""
    if _CUTEDSL_IMPORT_ERROR is not None:
        raise ImportError(
            "DSv3 sequence-wise fusion requires CuTeDSL; "
            "install nvidia-cutlass-dsl>=4.8.0."
        ) from _CUTEDSL_IMPORT_ERROR
    return _kernel_forward(scores_TE, routing_map_TE, arrival_counter)


@seqwise_loss_forward_op.register_fake
def _seqwise_loss_forward_fake(scores_TE, routing_map_TE, arrival_counter):
    return scores_TE.new_empty(()), scores_TE.new_empty((scores_TE.shape[1],))


@torch.library.custom_op(
    "torchtitan::dsv3_seqwise_loss_backward", mutates_args=(), device_types="cuda"
)
def seqwise_loss_backward_op(
    grad_raw_sum: torch.Tensor,
    scores_TE: torch.Tensor,
    frequencies_E: torch.Tensor,
) -> torch.Tensor:
    """One CuTeDSL launch for the raw-loss derivative with respect to scores."""
    if _CUTEDSL_IMPORT_ERROR is not None:
        raise ImportError(
            "DSv3 sequence-wise fusion requires CuTeDSL; "
            "install nvidia-cutlass-dsl>=4.8.0."
        ) from _CUTEDSL_IMPORT_ERROR
    return _kernel_backward(grad_raw_sum, scores_TE, frequencies_E)


@seqwise_loss_backward_op.register_fake
def _seqwise_loss_backward_fake(grad_raw_sum, scores_TE, frequencies_E):
    return torch.empty_like(scores_TE, memory_format=torch.contiguous_format)


class FusedDSv3SeqwiseLossFunction(torch.autograd.Function):
    """Raw loss and its first derivative; normalization/injection belongs to AuxLoss.

    ``arrival_counter`` is a zero-initialized, module-owned CUDA int32 scalar.
    Calls sharing a counter must be stream-ordered. Separate modules have
    separate counters, so independent layers can run on independent streams.
    """

    @staticmethod
    def spmd_typecheck(result, *, scores_TE, routing_map_TE, arrival_counter):
        spmd.rules.ignore(arrival_counter)
        # Both reductions require complete token/expert dimensions on the
        # non-local axes. The module makes DP local and gates CP/TP to size 1.
        spmd.rules.einsum("__,__->", scores_TE, routing_map_TE, out=result)

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx, scores_TE, routing_map_TE, arrival_counter
    ):
        scores_TE = scores_TE.resolve_neg()
        raw_sum, frequencies_E = seqwise_loss_forward_op(
            scores_TE, routing_map_TE, arrival_counter
        )
        ctx.save_for_backward(scores_TE, frequencies_E)
        return raw_sum

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_raw_sum):  # pyrefly: ignore[bad-override]
        scores_TE, frequencies_E = ctx.saved_tensors
        return (
            seqwise_loss_backward_op(grad_raw_sum, scores_TE, frequencies_E),
            None,
            None,
        )


def _supported(scores_TE, routing_map_TE, padding_mask_T):
    return (
        scores_TE.is_cuda
        and scores_TE.dtype == torch.float32
        and scores_TE.shape == (4096, 256)
        and scores_TE.is_contiguous()
        and routing_map_TE.dtype == torch.bool
        and routing_map_TE.device == scores_TE.device
        and routing_map_TE.shape == scores_TE.shape
        and routing_map_TE.is_contiguous()
        and padding_mask_T is None
        and spmd_mesh_size("cp") == 1
        and spmd_mesh_size("tp") == 1
        and torch.cuda.get_device_capability(scores_TE.device) == (10, 3)
    )


class FusedDSv3SeqwiseLoss(MicrobatchWiseLoadBalanceLoss):
    """Specialize the loss while retaining native scaling and metric state."""

    @dataclass(kw_only=True, slots=True)
    class Config(MicrobatchWiseLoadBalanceLoss.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.register_buffer(
            "arrival_counter", torch.zeros(1, dtype=torch.int32), persistent=False
        )

    @property
    def metric_name(self) -> str:
        return "microbatch_wise_load_balance_loss"

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None):
        super()._init_self_buffers(buffer_device=buffer_device)
        self.arrival_counter = torch.zeros(
            1, dtype=torch.int32, device=self.instance_acc.device
        )

    def forward(
        self,
        scores_TE,
        routing_map_TE,
        *,
        carrier,
        padding_mask_T=None,
        denominator,
    ):
        if _supported(scores_TE, routing_map_TE, padding_mask_T):
            with spmd_local_context("dp"):
                raw_sum = FusedDSv3SeqwiseLossFunction.apply(
                    scores_TE, routing_map_TE, self.arrival_counter
                )
                return self.inject(raw_sum, carrier=carrier, denominator=denominator)
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
) -> FusedDSv3SeqwiseLoss.Config:
    return derive(cfg, FusedDSv3SeqwiseLoss.Config)


@override(
    target=DeepSeekV3Router.Config,
    exact=True,
    description="Compose separate DeepSeek V3 router and sequence-wise loss fusions.",
)
def fused_dsv3_router_and_seqwise_loss(
    cfg: DeepSeekV3Router.Config,
) -> DeepSeekV3Router.Config:
    """Compose factories in one claim because nested override claims conflict."""
    from torchtitan_recipes.overrides.fused_dsv3_router import fused_dsv3_router

    router_cfg = fused_dsv3_router(cfg)
    if type(cfg.aux_loss) is MicrobatchWiseLoadBalanceLoss.Config:
        router_cfg.aux_loss = fused_dsv3_seqwise_loss(cfg.aux_loss)
    return router_cfg
