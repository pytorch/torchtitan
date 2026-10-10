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
    _split_into_bf16_pieces,
    HiMidLoLinear,
)
from torchtitan.models.deepseek_v3.moe import DeepSeekV3Router

try:
    if Version(version("nvidia-cutlass-dsl")) < Version("4.8.0"):
        raise ImportError("DSv3 router-gate kernels require CuTeDSL 4.8 or newer.")
    import cutlass
    import cutlass.utils.blackwell_helpers as sm100

    # pyrefly: ignore[missing-module-attribute]
    from cuda.bindings import driver as cuda
    from cutlass import cute, utils
    from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
    from cutlass.cute.runtime import from_dlpack

    _CUTEDSL_IMPORT_ERROR: ImportError | None = None
except ImportError as error:
    _CUTEDSL_IMPORT_ERROR = error


ACCEPTED = False


if TYPE_CHECKING or _CUTEDSL_IMPORT_ERROR is None:

    class _RouterGateGemm:
        def __init__(self, num_pieces, tile, stages, *, wgrad, threads):
            self.num_pieces = num_pieces
            self.tile = tile
            self.stages = stages
            self.wgrad = wgrad
            self.threads = threads
            self.tmem_columns = 1 << (
                (tile[1] * (num_pieces if wgrad else 1) - 1).bit_length()
            )

        @cute.jit
        def __call__(
            self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, stream: cuda.CUstream
        ):
            bm, bn, bk = self.tile
            a_major = OperandMajorMode.MN
            b_major = OperandMajorMode.MN if self.wgrad else OperandMajorMode.K
            mma = sm100.make_trivial_tiled_mma(
                cutlass.BFloat16,
                cutlass.BFloat16,
                a_major,
                b_major,
                cutlass.Float32,
                tcgen05.CtaGroup.ONE,
                (bm, bn),
            )
            a_layout = sm100.make_smem_layout(
                a_major, (bm, bk), cutlass.BFloat16, self.stages
            )
            b_layout = sm100.make_smem_layout(
                b_major,
                (bn, bk),
                cutlass.BFloat16,
                self.stages * (self.num_pieces if self.wgrad else 1),
            )
            self.kernel(a, b, c, mma, a_layout, b_layout).launch(
                grid=(cute.size(a, mode=[0]) // bm, cute.size(b, mode=[0]) // bn, 1),
                block=(self.threads + 128, 1, 1),
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            a: cute.Tensor,
            b: cute.Tensor,
            c: cute.Tensor,
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
            output_dtype = cutlass.Float32 if self.wgrad else cutlass.BFloat16
            c_layout = sm100.make_smem_layout(
                OperandMajorMode.MN, (bm, 32), output_dtype, 1
            )
            c_shared = smem.allocate_tensor(
                output_dtype, c_layout.outer, 128, swizzle=c_layout.inner
            )
            if warp == 0:
                with cute.arch.elect_one():
                    for slot in cutlass.range_constexpr(self.stages):
                        cute.arch.mbarrier_init(barriers.iterator + slot, 1)
                        cute.arch.mbarrier_init(ready.iterator + slot, 1)
                cute.arch.alloc_tmem(
                    self.tmem_columns, tmem_slot.iterator, is_two_cta=False
                )
            cute.arch.mbarrier_init_fence()
            cute.arch.barrier()
            mma_thread = mma.get_slice(0)
            a_fragment = mma.make_fragment_A(mma_thread.partition_A(a_shared))
            b_fragment = mma.make_fragment_B(mma_thread.partition_B(b_shared))
            acc_layout = mma.make_fragment_C(mma.partition_shape_C((bm, bn))).layout
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                cutlass.Float32,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_slot.iterator,
            )
            a_global = cute.local_tile(a, (bm, bk), (tile_m, None))
            b_global = cute.local_tile(b, (bn, bk), (tile_n, None))
            a_copy = cute.make_tiled_copy_tv(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    cutlass.BFloat16,
                    num_bits_per_copy=128,
                ),
                cute.make_layout(
                    (bm // 8, self.threads // (bm // 8)), stride=(1, bm // 8)
                ),
                cute.make_layout((8, 1)),
            )
            a_thread = a_copy.get_slice(tid)
            a_source = a_thread.partition_S(a_global)
            a_destination = a_thread.partition_D(a_shared)
            a_registers = cute.make_rmem_tensor(
                a_source[(None, None, None, 0)].shape, cutlass.BFloat16
            )
            if cutlass.const_expr(self.wgrad):
                b_threads = cute.make_layout(
                    (bn // 4, self.threads // (bn // 4)), stride=(1, bn // 4)
                )
                b_values = cute.make_layout((4, 1))
            else:
                b_threads = cute.make_layout(
                    (self.threads // (bk // 4), bk // 4), stride=(bk // 4, 1)
                )
                b_values = cute.make_layout((1, 4))
            b_copy = cute.make_tiled_copy_tv(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128
                ),
                b_threads,
                b_values,
            )
            b_thread = b_copy.get_slice(tid)
            b_source = b_thread.partition_S(b_global)
            b_destination = b_thread.partition_D(b_shared)
            b_registers = cute.make_rmem_tensor(
                b_source[(None, None, None, 0)].shape, cutlass.Float32
            )
            b_converted = cute.make_rmem_tensor(b_registers.shape, cutlass.BFloat16)
            b_store = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=64
            )

            steps = cute.size(a_global.shape[2])
            total_steps = steps * (1 if self.wgrad else self.num_pieces)
            if tid < self.threads:
                for global_step in cutlass.range(total_steps, unroll=1):
                    slot = global_step % self.stages
                    step = global_step % steps
                    cute.arch.mbarrier_wait(
                        barriers.iterator + slot, (global_step // self.stages + 1) % 2
                    )
                    cute.copy(a_copy, a_source[(None, None, None, step)], a_registers)
                    cute.copy(b_copy, b_source[(None, None, None, step)], b_registers)
                    cute.copy(
                        a_copy, a_registers, a_destination[(None, None, None, slot)]
                    )
                    values = b_registers.load()
                    hi = values.to(cutlass.BFloat16)
                    rest = values - hi.to(cutlass.Float32)
                    mid = rest.to(cutlass.BFloat16)
                    lo = (rest - mid.to(cutlass.Float32)).to(cutlass.BFloat16)
                    if cutlass.const_expr(self.wgrad):
                        for piece in cutlass.range_constexpr(self.num_pieces):
                            if cutlass.const_expr(piece == 0):
                                b_converted.store(hi)
                            elif cutlass.const_expr(piece == 1):
                                b_converted.store(mid)
                            else:
                                b_converted.store(lo)
                            cute.copy(
                                b_store,
                                b_converted,
                                b_destination[
                                    (None, None, None, slot * self.num_pieces + piece)
                                ],
                            )
                    else:
                        piece = global_step // steps
                        if piece == 0:
                            b_converted.store(hi)
                        elif piece == 1:
                            b_converted.store(mid)
                        else:
                            if cutlass.const_expr(self.num_pieces == 3):
                                b_converted.store(lo)
                        cute.copy(
                            b_store,
                            b_converted,
                            b_destination[(None, None, None, slot)],
                        )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.barrier(barrier_id=1, number_of_threads=self.threads)
                    if tid == 0:
                        cute.arch.mbarrier_arrive(ready.iterator + slot)
            if warp == self.threads // 32:
                for global_step in cutlass.range(total_steps, unroll=1):
                    slot = global_step % self.stages
                    cute.arch.mbarrier_wait(
                        ready.iterator + slot, (global_step // self.stages) % 2
                    )
                    for piece in cutlass.range_constexpr(
                        self.num_pieces if self.wgrad else 1
                    ):
                        accumulator = cute.make_tensor(
                            tmem_ptr + piece * bn, acc_layout
                        )
                        mma.set(tcgen05.Field.ACCUMULATE, global_step != 0)
                        for block_k in cutlass.range_constexpr(
                            cute.size(a_fragment.shape[2])
                        ):
                            b_slot = (
                                slot * self.num_pieces + piece if self.wgrad else slot
                            )
                            cute.gemm(
                                mma,
                                accumulator,
                                a_fragment[(None, None, block_k, slot)],
                                b_fragment[(None, None, block_k, b_slot)],
                                accumulator,
                            )
                            mma.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(barriers.iterator + slot)

            cute.arch.mbarrier_wait(
                barriers.iterator + (total_steps - 1) % self.stages,
                ((total_steps - 1) // self.stages) % 2,
            )
            cute.arch.barrier()
            accumulator = cute.make_tensor(tmem_ptr, acc_layout)
            c_global = cute.local_tile(c, (bm, bn), (tile_m, tile_n))
            epilogue = (bm, 32)
            acc_tiles = cute.flat_divide(accumulator[((None, None), 0, 0)], epilogue)
            c_tiles = cute.flat_divide(c_global, epilogue)
            load_atom = sm100.get_tmem_load_op(
                self.tile,
                utils.LayoutEnum.COL_MAJOR,
                output_dtype,
                cutlass.Float32,
                epilogue,
                False,
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
                utils.LayoutEnum.COL_MAJOR, output_dtype, cutlass.Float32, load
            )
            shared_copy = cute.make_tiled_copy_D(shared_store, load)
            shared_thread = shared_copy.get_slice(tid)
            shared_destination = shared_thread.partition_D(c_shared)
            shared_source = shared_copy.retile(rounded)
            vector_elements = 4 if self.wgrad else 8
            c_copy = cute.make_tiled_copy_tv(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), output_dtype, num_bits_per_copy=128
                ),
                cute.make_layout(
                    (bm // vector_elements, 128 // (bm // vector_elements)),
                    stride=(1, bm // vector_elements),
                ),
                cute.make_layout((vector_elements, 1)),
            )
            c_thread = c_copy.get_slice(tid)
            vector_source = c_thread.partition_S(c_shared)
            vector_destination = c_thread.partition_D(c_tiles)
            vector_registers = cute.make_rmem_tensor(
                vector_source[(None, None, None, 0)].shape, output_dtype
            )
            for column in cutlass.range_constexpr(bn // 32):
                if tid < 128:
                    for piece in cutlass.range_constexpr(
                        self.num_pieces if self.wgrad else 1
                    ):
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
                        shared_destination[(None, None, None, 0)],
                    )
                cute.arch.barrier()
                if tid < 128:
                    cute.copy(
                        c_copy, vector_source[(None, None, None, 0)], vector_registers
                    )
                    cute.copy(
                        c_copy,
                        vector_registers,
                        vector_destination[(None, None, None, 0, column)],
                    )
                cute.arch.barrier()
            cute.arch.barrier()
            if warp == 0:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
                cute.arch.dealloc_tmem(tmem_ptr, self.tmem_columns, is_two_cta=False)

    @lru_cache(maxsize=64)
    def _compile_gate_gemm(num_pieces, wgrad, tile, stages, device, threads):
        k, n = (4096, 256) if wgrad else (256, 4096)
        a = torch.empty((k, 7168), device=device, dtype=torch.bfloat16).t()
        gradient = torch.empty((4096, 256), device=device, dtype=torch.float32)
        b = gradient.t() if wgrad else gradient
        c = torch.empty(
            (n, 7168), device=device, dtype=torch.float32 if wgrad else torch.bfloat16
        ).t()
        return cute.compile(
            _RouterGateGemm(num_pieces, tile, stages, wgrad=wgrad, threads=threads),
            *(from_dlpack(value, assumed_align=16) for value in (a, b, c)),
            cuda.CUstream(torch.cuda.current_stream(device).cuda_stream),
        )

    def _gate_gemm(
        grad_logits,
        operand,
        *,
        num_pieces=3,
        wgrad=False,
        tile=None,
        stages=2,
        threads=512,
    ):
        tile = (128, 128 if wgrad else 256, 64) if tile is None else tile
        n = grad_logits.shape[1] if wgrad else grad_logits.shape[0]
        output = torch.empty(
            (n, 7168),
            dtype=torch.float32 if wgrad else torch.bfloat16,
            device=operand.device,
        )
        kernel = _compile_gate_gemm(
            num_pieces, wgrad, tile, stages, str(grad_logits.device), threads
        )
        a = operand.t()
        b = grad_logits.t() if wgrad else grad_logits
        c = output.t()
        kernel(
            *(from_dlpack(value.detach(), assumed_align=16) for value in (a, b, c)),
            cuda.CUstream(torch.cuda.current_stream(grad_logits.device).cuda_stream),
        )
        return output


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
    "torchtitan::dsv3_router_gate_dgrad", mutates_args=(), device_types="cuda"
)
def dgrad_op(
    grad_output_TE: torch.Tensor, weight_ED: torch.Tensor, num_pieces: int
) -> torch.Tensor:
    if num_pieces not in (2, 3):
        raise ValueError("Router-gate backward requires two or three BF16 pieces.")
    if not _supports_gate_backward(
        grad_output_TE, weight_ED, num_pieces=num_pieces, wgrad=False
    ):
        stacked_TPE = _split_into_bf16_pieces(grad_output_TE, num_pieces, dim=1)
        return torch.mm(stacked_TPE, torch.cat([weight_ED] * num_pieces))
    return _gate_gemm(grad_output_TE, weight_ED, num_pieces=num_pieces)


@dgrad_op.register_fake
def _dgrad_fake(grad_output_TE, weight_ED, num_pieces):
    return weight_ED.new_empty((grad_output_TE.shape[0], weight_ED.shape[1]))


@torch.library.custom_op(
    "torchtitan::dsv3_router_gate_wgrad", mutates_args=(), device_types="cuda"
)
def wgrad_op(
    grad_output_TE: torch.Tensor, input_TD: torch.Tensor, num_pieces: int
) -> torch.Tensor:
    if num_pieces not in (2, 3):
        raise ValueError("Router-gate backward requires two or three BF16 pieces.")
    if not _supports_gate_backward(
        grad_output_TE, input_TD, num_pieces=num_pieces, wgrad=True
    ):
        stacked_TPE = _split_into_bf16_pieces(grad_output_TE, num_pieces, dim=1)
        pieces_TE = stacked_TPE.split(grad_output_TE.shape[1], dim=1)
        grad_weight_ED = torch.mm(pieces_TE[0].T, input_TD, out_dtype=torch.float32)
        for piece_TE in pieces_TE[1:]:
            grad_weight_ED += torch.mm(piece_TE.T, input_TD, out_dtype=torch.float32)
        return grad_weight_ED
    return _gate_gemm(grad_output_TE, input_TD, num_pieces=num_pieces, wgrad=True)


@wgrad_op.register_fake
def _wgrad_fake(grad_output_TE, input_TD, num_pieces):
    return input_TD.new_empty(
        (grad_output_TE.shape[1], input_TD.shape[1]), dtype=torch.float32
    )


@spmd.register_local_autograd_function
class FusedDSv3RouterGateFunction(torch.autograd.Function):
    """Native forward; on-chip BF16 splitting, gradient GEMMs and output sums."""

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
        grad_input_TD = (
            dgrad_op(grad_output_TE, weight_ED, ctx.num_pieces)
            if ctx.needs_input_grad[0]
            else None
        )
        grad_weight_ED = (
            wgrad_op(grad_output_TE, input_TD, ctx.num_pieces)
            if ctx.needs_input_grad[1]
            else None
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
