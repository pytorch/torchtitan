# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Experimental CuTeDSL backward for the DSv3 HiMidLo router gate.

T = local tokens, D = model dimension, E = experts; B/L are leading token axes.
The forward, FP32 weight gradients and native Linear/FSDP ownership are retained.
"""

import warnings
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import version
from typing import TYPE_CHECKING

import spmd_types as spmd
import torch
from packaging.version import Version
from torch._subclasses.fake_tensor import FakeTensor
from torch.autograd.function import once_differentiable

from torchtitan.config import derive, override
from torchtitan.models.common.hi_mid_lo_linear import (
    _HiMidLoLinearFunction,
    _narrow_backward,
    HiMidLoLinear,
)
from torchtitan.models.deepseek_v3.moe import DeepSeekV3Router

try:
    import cutlass
    import cutlass.utils.blackwell_helpers as sm100

    # pyrefly: ignore[missing-module-attribute]
    from cuda.bindings import driver as cuda
    from cutlass import utils
    from cutlass.cute.nvgpu import cpasync, OperandMajorMode, tcgen05
    from cutlass.cute.runtime import from_dlpack, make_fake_tensor

    if Version(version("nvidia-cutlass-dsl")) < Version("4.8.0"):
        raise ImportError("DSv3 router-gate kernels require CuTeDSL 4.8 or newer.")
    _CUTEDSL_IMPORT_ERROR: ImportError | None = None
except ImportError as error:
    _CUTEDSL_IMPORT_ERROR = error


ACCEPTED = False


if TYPE_CHECKING or _CUTEDSL_IMPORT_ERROR is None:
    import cutlass.cute as cute

    @cute.jit
    def _split_pair(a, b):
        # Shift BF16 bits back to FP32 so subnormals and each rounding survive.
        return cute.arch.inline_ptx(
            "{.reg .b32 x0,x1; .reg .f32 a0,a1,r0,r1; "
            "cvt.rn.bf16x2.f32 {$w0}, {$r1}, {$r0}; "
            "shl.b32 x0, {$w0}, 16; and.b32 x1, {$w0}, 0xffff0000; "
            "mov.b32 a0,x0; mov.b32 a1,x1; "
            "sub.rn.f32 r0,{$r0},a0; sub.rn.f32 r1,{$r1},a1; "
            "cvt.rn.bf16x2.f32 {$w1},r1,r0; "
            "shl.b32 x0, {$w1}, 16; and.b32 x1, {$w1}, 0xffff0000; "
            "mov.b32 a0,x0; mov.b32 a1,x1; "
            "sub.rn.f32 r0,r0,a0; sub.rn.f32 r1,r1,a1; "
            "cvt.rn.bf16x2.f32 {$w2},r1,r0;}",
            write_only_types=[cutlass.Uint32, cutlass.Uint32, cutlass.Uint32],
            read_only_args=[a, b],
        )

    @cute.jit
    def _load_float4(ptr):
        return cute.arch.inline_ptx(
            "ld.global.v4.f32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
            write_only_types=[cutlass.Float32] * 4,
            read_only_args=[ptr],
        )

    @cute.jit
    def _load_word4(ptr):
        return cute.arch.inline_ptx(
            "ld.global.v4.u32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
            write_only_types=[cutlass.Uint32] * 4,
            read_only_args=[ptr],
        )

    @cute.jit
    def _store_word4(ptr, a, b, c, d):
        cute.arch.inline_ptx(
            "st.global.v4.u32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
            read_only_args=[ptr, a, b, c, d],
        )

    class _RouterGatePrepare:
        def __init__(self, num_pieces, needs_grad_input):
            self.num_pieces = num_pieces
            self.needs_grad_input = needs_grad_input
            self.num_threads = 256
            self.num_gradient_chunks = 2
            self.num_weight_chunks = 2
            self.num_gradient_blocks = 256
            self.num_weight_blocks = 448 if needs_grad_input else 0

        @cute.jit
        def __call__(self, g, w, stacked, repeated, stream: cuda.CUstream):
            self.kernel(g, w, stacked, repeated).launch(
                grid=(self.num_gradient_blocks + self.num_weight_blocks, 1, 1),
                block=(self.num_threads, 1, 1),
                stream=stream,
            )

        @cute.kernel
        def kernel(self, g, w, stacked, repeated):
            tid, _, _ = cute.arch.thread_idx()
            bid, _, _ = cute.arch.block_idx()
            if bid < self.num_gradient_blocks:
                for chunk in cutlass.range_constexpr(self.num_gradient_chunks):
                    offset = (
                        bid * (self.num_threads * 8 * self.num_gradient_chunks)
                        + chunk * self.num_threads * 8
                        + tid * 8
                    )
                    a, b, c, d = _load_float4(g.iterator + offset)
                    e, f, h, i = _load_float4(g.iterator + offset + 4)
                    h0, m0, l0 = _split_pair(a, b)
                    h1, m1, l1 = _split_pair(c, d)
                    h2, m2, l2 = _split_pair(e, f)
                    h3, m3, l3 = _split_pair(h, i)
                    out = (offset // 256) * self.num_pieces * 128 + (offset % 256) // 2
                    _store_word4(stacked.iterator + out, h0, h1, h2, h3)
                    _store_word4(stacked.iterator + out + 128, m0, m1, m2, m3)
                    if cutlass.const_expr(self.num_pieces == 3):
                        _store_word4(stacked.iterator + out + 256, l0, l1, l2, l3)
            elif cutlass.const_expr(self.needs_grad_input):
                for chunk in cutlass.range_constexpr(self.num_weight_chunks):
                    offset = (
                        (bid - self.num_gradient_blocks)
                        * (self.num_threads * 4 * self.num_weight_chunks)
                        + chunk * self.num_threads * 4
                        + tid * 4
                    )
                    a, b, c, d = _load_word4(w.iterator + offset)
                    for piece in cutlass.range_constexpr(self.num_pieces):
                        _store_word4(
                            repeated.iterator + piece * 917504 + offset, a, b, c, d
                        )

    class _RouterGateWeightGradient:
        def __init__(self, num_pieces):
            self.num_pieces = num_pieces
            # One cluster covers all experts. 96 model columns give 150 CTAs
            # on the 152-SM GB300 without changing the reduction order.
            self.tile = (256, 96, 32)
            self.stages = 8
            self.tmem_columns = 1 << ((96 * num_pieces - 1).bit_length())

        @cute.jit
        def __call__(
            self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, stream: cuda.CUstream
        ):
            bm, bn, bk = self.tile
            a_major = OperandMajorMode.MN
            b_major = OperandMajorMode.MN
            mma = sm100.make_trivial_tiled_mma(
                cutlass.BFloat16,
                cutlass.BFloat16,
                a_major,
                b_major,
                cutlass.Float32,
                tcgen05.CtaGroup.TWO,
                (bm, bn),
            )
            cluster_layout = cute.tiled_divide(
                cute.make_layout((2, 1, 1)), (mma.thr_id.shape,)
            )
            a_layout = sm100.make_smem_layout_a(
                mma, self.tile, cutlass.BFloat16, self.stages * self.num_pieces
            )
            b_layout = sm100.make_smem_layout_b(
                mma, self.tile, cutlass.BFloat16, self.stages
            )
            a_load, a_tensor = cute.nvgpu.make_tiled_tma_atom_A(
                cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO),
                a,
                cute.slice_(a_layout, (None, None, None, 0)),
                self.tile,
                mma,
                cluster_layout.shape,
            )
            b_load, b_tensor = cute.nvgpu.make_tiled_tma_atom_B(
                cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO),
                b,
                cute.slice_(b_layout, (None, None, None, 0)),
                self.tile,
                mma,
                cluster_layout.shape,
            )
            output_dtype = cutlass.Float32
            c_layout = sm100.make_smem_layout(
                OperandMajorMode.K, (bm // 2, bn), output_dtype, 1
            )
            c_store, c_tensor = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                c,
                cute.slice_(c_layout, (None, None, 0)),
                (bm // 2, bn),
            )
            self.kernel(
                a_load,
                a_tensor,
                b_load,
                b_tensor,
                c_tensor,
                c_store,
                c_layout,
                mma,
                a_layout,
                b_layout,
            ).launch(
                grid=(
                    cute.size(a, mode=[0]) // (bm // 2),
                    cute.ceil_div(cute.size(b, mode=[0]), bn),
                    1,
                ),
                cluster=(2, 1, 1),
                block=(128, 1, 1),
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            a_load: cute.CopyAtom,
            a: cute.Tensor,
            b_load: cute.CopyAtom,
            b: cute.Tensor,
            c: cute.Tensor,
            c_store: cute.CopyAtom,
            c_layout: cute.ComposedLayout,
            mma: cute.TiledMma,
            a_layout: cute.ComposedLayout,
            b_layout: cute.ComposedLayout,
        ):
            bm, bn, bk = self.tile
            tid, _, _ = cute.arch.thread_idx()
            tile_m, tile_n, _ = cute.arch.block_idx()
            warp = cute.arch.warp_idx()
            smem = utils.SmemAllocator()
            barriers = smem.allocate_tensor(
                cutlass.Int64, cute.make_layout(self.stages), 8
            )
            ready = smem.allocate_tensor(
                cutlass.Int64, cute.make_layout(self.stages), 8
            )
            tmem_slot = smem.allocate_tensor(cutlass.Int32, cute.make_layout(1), 4)
            a_shared = smem.allocate_tensor(
                cutlass.BFloat16, a_layout.outer, 128, swizzle=a_layout.inner
            )
            b_shared = smem.allocate_tensor(
                cutlass.BFloat16, b_layout.outer, 128, swizzle=b_layout.inner
            )
            output_dtype = cutlass.Float32
            # The epilogue reuses operand storage after the final MMA completes.
            c_shared = cute.make_tensor(
                cute.recast_ptr(
                    a_shared.iterator, dtype=cutlass.Float32, swizzle_=c_layout.inner
                ),
                c_layout.outer,
            )
            if warp == 0:
                with cute.arch.elect_one():
                    for slot in cutlass.range_constexpr(self.stages):
                        cute.arch.mbarrier_init(barriers.iterator + slot, 1)
                        cute.arch.mbarrier_init(ready.iterator + slot, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
            if warp == 0:
                cute.arch.alloc_tmem(
                    self.tmem_columns, tmem_slot.iterator, is_two_cta=True
                )
            cute.arch.barrier()
            mma_thread = mma.get_slice(tile_m % 2)
            a_fragment = mma.make_fragment_A(a_shared)
            b_fragment = mma.make_fragment_B(b_shared)
            acc_layout = mma.make_fragment_C(mma.partition_shape_C((bm, bn))).layout
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                cutlass.Float32,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_slot.iterator,
            )
            a_global = mma_thread.partition_A(
                cute.local_tile(a, (bm, bk), (tile_m // 2, None, None))
            )
            b_global = mma_thread.partition_B(
                cute.local_tile(b, (bn, bk), (tile_n, None))
            )
            a_destination, a_source = cpasync.tma_partition(
                a_load,
                0,
                cute.make_layout(1),
                cute.group_modes(a_shared, 0, 3),
                cute.group_modes(a_global, 0, 3),
            )
            b_destination, b_source = cpasync.tma_partition(
                b_load,
                0,
                cute.make_layout(1),
                cute.group_modes(b_shared, 0, 3),
                cute.group_modes(b_global, 0, 3),
            )
            steps = cute.size(a_global.shape[3])
            if warp == 0:
                for step in cutlass.range(steps, unroll=1):
                    slot = step % self.stages
                    cute.arch.mbarrier_wait(
                        barriers.iterator + slot, (step // self.stages + 1) % 2
                    )
                    if tile_m % 2 == 0:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                ready.iterator + slot,
                                (bm * self.num_pieces + bn) * bk * 2,
                            )
                    for piece in cutlass.range_constexpr(self.num_pieces):
                        cute.copy(
                            a_load,
                            a_source[None, step, piece],
                            a_destination[None, slot * self.num_pieces + piece],
                            tma_bar_ptr=ready.iterator + slot,
                        )
                    cute.copy(
                        b_load,
                        b_source[None, step],
                        b_destination[None, slot],
                        tma_bar_ptr=ready.iterator + slot,
                    )
            if warp == 1 and tile_m % 2 == 0:
                for global_step in cutlass.range(steps, unroll=1):
                    slot = global_step % self.stages
                    cute.arch.mbarrier_wait(
                        ready.iterator + slot, (global_step // self.stages) % 2
                    )
                    for piece in cutlass.range_constexpr(self.num_pieces):
                        accumulator = cute.make_tensor(
                            tmem_ptr + piece * bn, acc_layout
                        )
                        mma.set(tcgen05.Field.ACCUMULATE, global_step != 0)
                        for block_k in cutlass.range_constexpr(
                            cute.size(a_fragment.shape[2])
                        ):
                            b_slot = slot * self.num_pieces + piece
                            cute.gemm(
                                mma,
                                accumulator,
                                a_fragment[(None, None, block_k, b_slot)],
                                b_fragment[(None, None, block_k, slot)],
                                accumulator,
                            )
                            mma.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(
                            barriers.iterator + slot,
                            mask=3,
                            cta_group=tcgen05.CtaGroup.TWO,
                        )

            cute.arch.mbarrier_wait(
                barriers.iterator + (steps - 1) % self.stages,
                ((steps - 1) // self.stages) % 2,
            )
            cute.arch.barrier()
            accumulator = cute.make_tensor(tmem_ptr, acc_layout)
            c_global = cute.local_tile(c, (bm // 2, bn), (tile_m, tile_n))
            epilogue = (bm // 2, 32)
            acc_tiles = cute.flat_divide(accumulator[((None, None), 0, 0)], epilogue)
            c_tiles = cute.flat_divide(c_global, epilogue)
            load_atom = sm100.get_tmem_load_op(
                self.tile,
                utils.LayoutEnum.ROW_MAJOR,
                output_dtype,
                cutlass.Float32,
                epilogue,
                True,
            )
            load = tcgen05.make_tmem_copy(load_atom, acc_tiles[(None, None, 0, 0)])
            thread = load.get_slice(tid)
            c_destination = thread.partition_D(c_tiles)
            result = cute.make_rmem_tensor(
                c_destination[(None, None, None, 0, 0)].shape, cutlass.Float32
            )
            total = cute.make_rmem_tensor(result.shape, cutlass.Float32)
            rounded = cute.make_rmem_tensor(result.shape, output_dtype)
            shared_store = sm100.get_smem_store_op(
                utils.LayoutEnum.ROW_MAJOR, output_dtype, cutlass.Float32, load
            )
            shared_copy = cute.make_tiled_copy_D(shared_store, load)
            shared_thread = shared_copy.get_slice(tid)
            shared_tiles = cute.flat_divide(c_shared[None, None, 0], epilogue)
            shared_destination = shared_thread.partition_D(shared_tiles)
            shared_source = shared_copy.retile(rounded)
            for column in cutlass.range_constexpr(bn // 32):
                if tid < 128:
                    for piece in cutlass.range_constexpr(self.num_pieces):
                        partial = cute.make_tensor(tmem_ptr + piece * bn, acc_layout)
                        partial_tiles = cute.flat_divide(
                            partial[((None, None), 0, 0)], epilogue
                        )
                        source = thread.partition_S(partial_tiles)
                        cute.copy(load, source[(None, None, None, 0, column)], result)
                        if cutlass.const_expr(piece == 0):
                            total.store(result.load())
                        else:
                            total.store(total.load() + result.load())
                    rounded.store(total.load().to(output_dtype))
                    cute.copy(
                        shared_copy,
                        shared_source,
                        shared_destination[(None, None, None, 0, column)],
                    )
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier()
            if tid == 0:
                source, destination = cpasync.tma_partition(
                    c_store,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(c_shared, 0, 2),
                    cute.group_modes(c_global, 0, 2),
                )
                cute.copy(c_store, source[None, 0], destination)
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.barrier()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
            if warp == 0:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=True)
                cute.arch.dealloc_tmem(tmem_ptr, self.tmem_columns, is_two_cta=True)

    @lru_cache(maxsize=8)
    def _compile_prepare(num_pieces, needs_grad_input, device):
        words = (
            1048576,
            917504 if needs_grad_input else None,
            4096 * num_pieces * 128,
            num_pieces * 917504 if needs_grad_input else None,
        )
        arguments = tuple(
            make_fake_tensor(
                cutlass.Float32 if index == 0 else cutlass.Uint32,
                (size,),
                (1,),
                assumed_align=16,
            )
            if size is not None
            else None
            for index, size in enumerate(words)
        )
        return cute.compile(
            _RouterGatePrepare(num_pieces, needs_grad_input),
            *arguments,
            cuda.CUstream(torch.cuda.current_stream(device).cuda_stream),
        )

    def _prepare_inputs(grad_output_TE, weight_ED, num_pieces, needs_grad_input):
        stacked_TPE = torch.empty(
            (4096, num_pieces * 256), device=grad_output_TE.device, dtype=torch.bfloat16
        )
        repeated_PED = (
            torch.empty(
                (num_pieces * 256, 7168), device=weight_ED.device, dtype=torch.bfloat16
            )
            if needs_grad_input
            else None
        )
        arguments = (
            grad_output_TE.flatten(),
            weight_ED.view(torch.uint32).flatten() if needs_grad_input else None,
            stacked_TPE.view(torch.uint32).flatten(),
            repeated_PED.view(torch.uint32).flatten()
            if repeated_PED is not None
            else None,
        )
        kernel = _compile_prepare(
            num_pieces, needs_grad_input, str(grad_output_TE.device)
        )
        kernel(
            *(
                from_dlpack(value.detach(), assumed_align=16)
                if value is not None
                else None
                for value in arguments
            ),
            cuda.CUstream(torch.cuda.current_stream(grad_output_TE.device).cuda_stream),
        )
        return stacked_TPE, repeated_PED

    @lru_cache(maxsize=4)
    def _compile_weight_gradient(num_pieces, device):
        arguments = (
            make_fake_tensor(
                cutlass.BFloat16,
                (256, 4096, num_pieces),
                (1, 256 * num_pieces, 256),
                assumed_align=16,
            ),
            make_fake_tensor(
                cutlass.BFloat16, (7168, 4096), (1, 7168), assumed_align=16
            ),
            make_fake_tensor(cutlass.Float32, (256, 7168), (7168, 1), assumed_align=16),
        )
        return cute.compile(
            _RouterGateWeightGradient(num_pieces),
            *arguments,
            cuda.CUstream(torch.cuda.current_stream(device).cuda_stream),
        )

    def _weight_gradient(stacked_TPE, input_TD, num_pieces):
        grad_weight_ED = input_TD.new_empty((256, 7168), dtype=torch.float32)
        arguments = (
            stacked_TPE.view(4096, num_pieces, 256).permute(2, 0, 1),
            input_TD.t(),
            grad_weight_ED,
        )
        kernel = _compile_weight_gradient(num_pieces, str(input_TD.device))
        kernel(
            *(from_dlpack(value.detach(), assumed_align=16) for value in arguments),
            cuda.CUstream(torch.cuda.current_stream(input_TD.device).cuda_stream),
        )
        return grad_weight_ED

    def _backward_kernels(
        grad_output_TE,
        input_TD,
        weight_ED,
        num_pieces,
        needs_grad_input,
        needs_grad_weight,
    ):
        stacked_TPE, repeated_PED = _prepare_inputs(
            grad_output_TE, weight_ED, num_pieces, needs_grad_input
        )
        grad_input_TD = (
            torch.mm(stacked_TPE, repeated_PED) if repeated_PED is not None else None
        )
        # Release the repeated weight before allocating the FP32 weight gradient.
        del repeated_PED
        grad_weight_ED = (
            _weight_gradient(stacked_TPE, input_TD, num_pieces)
            if needs_grad_weight
            else None
        )
        return grad_input_TD, grad_weight_ED


def _supports_gate_backward(grad_output_TE, operand_RD, *, num_pieces, wgrad):
    rows = 4096 if wgrad else 256
    tensors = (grad_output_TE, operand_RD)
    return (
        _CUTEDSL_IMPORT_ERROR is None
        and num_pieces in (2, 3)
        and all(
            type(value) in (torch.Tensor, torch.nn.Parameter, FakeTensor)
            for value in tensors
        )
        and grad_output_TE.shape == (4096, 256)
        and operand_RD.shape == (rows, 7168)
        and grad_output_TE.dtype == torch.float32
        and operand_RD.dtype == torch.bfloat16
        and grad_output_TE.is_cuda
        and grad_output_TE.device == operand_RD.device
        and all(
            value.is_contiguous() and not value.is_neg() and not value.is_conj()
            for value in tensors
        )
        and torch.__version__ == "2.16.0.dev20261007+cu130"
        and (
            isinstance(grad_output_TE, FakeTensor)
            or (
                all(value.data_ptr() % 16 == 0 for value in tensors)
                and torch.cuda.get_device_capability(grad_output_TE.device) == (10, 3)
            )
        )
    )


@torch.library.custom_op(
    "torchtitan::dsv3_router_gate_backward",
    mutates_args=(),
    device_types="cuda",
    schema="(Tensor grad_output_TE, Tensor input_TD, Tensor weight_ED, "
    "SymInt num_pieces, bool needs_grad_input, bool needs_grad_weight) -> (Tensor?, Tensor?)",
)
def backward_op(
    grad_output_TE: torch.Tensor,
    input_TD: torch.Tensor,
    weight_ED: torch.Tensor,
    num_pieces: int,
    needs_grad_input: bool,
    needs_grad_weight: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if num_pieces not in (2, 3):
        raise ValueError("Router-gate backward requires two or three BF16 pieces.")
    if not (needs_grad_input or needs_grad_weight):
        return None, None
    if not (
        (
            not needs_grad_input
            or _supports_gate_backward(
                grad_output_TE, weight_ED, num_pieces=num_pieces, wgrad=False
            )
        )
        and (
            not needs_grad_weight
            or _supports_gate_backward(
                grad_output_TE, input_TD, num_pieces=num_pieces, wgrad=True
            )
        )
    ):
        return _narrow_backward(
            grad_output_TE,
            input_TD,
            weight_ED,
            num_pieces=num_pieces,
            needs_grad_input=needs_grad_input,
            needs_grad_weight=needs_grad_weight,
        )
    return _backward_kernels(
        grad_output_TE,
        input_TD,
        weight_ED,
        num_pieces,
        needs_grad_input,
        needs_grad_weight,
    )


@backward_op.register_fake
def _backward_fake(
    grad_output_TE, input_TD, weight_ED, num_pieces, needs_grad_input, needs_grad_weight
):
    return (
        input_TD.new_empty(input_TD.shape) if needs_grad_input else None,
        weight_ED.new_empty(weight_ED.shape, dtype=torch.float32)
        if needs_grad_weight
        else None,
    )


@spmd.register_local_autograd_function
class FusedDSv3RouterGateFunction(torch.autograd.Function):
    """Native forward; shared BF16 preparation and a fused FP32 weight gradient."""

    @staticmethod
    def forward(ctx, input_TD, weight_ED, num_pieces):  # pyrefly: ignore[bad-override]
        return _HiMidLoLinearFunction.forward(ctx, input_TD, weight_ED, num_pieces)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output_TE):  # pyrefly: ignore[bad-override]
        input_TD, weight_ED = ctx.saved_tensors
        if not (
            ctx.use_bf16_gemm
            and grad_output_TE.shape == (4096, 256)
            and input_TD.shape == (4096, 7168)
            and weight_ED.shape == (256, 7168)
        ):
            return _HiMidLoLinearFunction.backward(ctx, grad_output_TE)
        grad_output_TE = grad_output_TE.float()
        # Runtime layout/alignment guards live inside the operators so AOT
        # backward tracing never inspects pointers.
        grad_input_TD, grad_weight_ED = backward_op(
            grad_output_TE,
            input_TD,
            weight_ED,
            ctx.num_pieces,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
        )
        return grad_input_TD, grad_weight_ED, None


class FusedDSv3RouterGate(HiMidLoLinear):
    @dataclass(kw_only=True, slots=True)
    class Config(HiMidLoLinear.Config):
        use_fused_backward: bool = False

    def __init__(self, config: Config):
        super().__init__(config)
        self.use_fused_backward = config.use_fused_backward

    def _linear(self, input, weight, bias):
        if not self.use_fused_backward:
            return super()._linear(input, weight, bias)
        input_BLD, weight_ED, bias_E = input, weight, bias
        output_TE = FusedDSv3RouterGateFunction.apply(
            input_BLD.reshape(-1, input_BLD.shape[-1]), weight_ED, self.num_pieces
        )
        output_BLE = output_TE.reshape(*input_BLD.shape[:-1], -1)
        return output_BLE if bias_E is None else output_BLE + bias_E.float()


@override(
    target=HiMidLoLinear.Config,
    exact=True,
    fqns=["*.moe.router.gate"],
    description="Fuse the DSv3 HiMidLo router-gate backward with CuTeDSL.",
)
def fused_dsv3_router_gate(
    cfg: HiMidLoLinear.Config, *, experimental: bool = False
) -> FusedDSv3RouterGate.Config:
    if not ACCEPTED:
        warnings.warn(
            "Router-gate roofline acceptance is open; experimental=True enables "
            "the candidate for evaluation. Native backward is the default.",
            stacklevel=2,
        )
    return derive(
        cfg,
        FusedDSv3RouterGate.Config,
        use_fused_backward=ACCEPTED or experimental,
    )


@override(
    target=DeepSeekV3Router.Config,
    exact=True,
    description="Compose DSv3 router/gate fusions with optional sequence-wise loss.",
)
def fused_dsv3_router_with_gate(
    cfg: DeepSeekV3Router.Config,
    *,
    experimental: bool = False,
    seqwise_loss: bool = False,
) -> DeepSeekV3Router.Config:
    """Use one parent claim for the independently implemented module fusions."""
    if seqwise_loss:
        from .fused_dsv3_seqwise_loss import fused_dsv3_router_and_seqwise_loss

        router_cfg = fused_dsv3_router_and_seqwise_loss(cfg)
    else:
        from .fused_dsv3_router import fused_dsv3_router

        router_cfg = fused_dsv3_router(cfg)
    if type(cfg.gate) is HiMidLoLinear.Config:
        router_cfg.gate = fused_dsv3_router_gate(cfg.gate, experimental=experimental)
    return router_cfg
