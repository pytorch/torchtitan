# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CuTeDSL TF32 backward for the pinned DSv3 RouterGateLinear.

T = local tokens, D = model dimension, E = experts; B/L are leading token axes.
The native gate returns BF16 dX and dW, unlike public-main HiMidLoLinear.
This override requires the fbpackage's explicit per-GEMM TF32 policy.
"""

import os
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, TYPE_CHECKING

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.autograd.function import once_differentiable

from torchtitan.config import derive, override

try:
    # The pinned fbpackage retains this gate; public main uses HiMidLoLinear.
    from torchtitan.models.common.router_gate import (  # pyrefly: ignore[missing-import]
        _RouterGateLinearFunction,
        RouterGateLinear,
    )
except ImportError:
    _RouterGateLinearFunction = None
    RouterGateLinear = None

try:
    import cutlass
    import cutlass.utils.blackwell_helpers as sm100

    # pyrefly: ignore[missing-module-attribute]
    from cuda.bindings import driver as cuda
    from cutlass import utils
    from cutlass._mlir.dialects import llvm
    from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
    from cutlass.cute.runtime import from_dlpack, make_fake_tensor

    _CUTEDSL_IMPORT_ERROR: ImportError | None = None
except ImportError as error:
    _CUTEDSL_IMPORT_ERROR = error


ACCEPTED = False
_NATIVE_MM: Any = getattr(torch, "_mm_with_compute_mode", None)


def _require_native_gate():
    if _RouterGateLinearFunction is None or _NATIVE_MM is None:
        raise RuntimeError(
            "The TF32 router-gate override requires the pinned fbpackage's "
            "RouterGateLinear and torch._mm_with_compute_mode. "
            "Public HiMidLoLinear has a different backward precision policy."
        )
    return _RouterGateLinearFunction


if TYPE_CHECKING or _CUTEDSL_IMPORT_ERROR is None:
    import cutlass.cute as cute
    from cutlass.cutlass_dsl import dsl_user_op

    @dsl_user_op
    def _tf32_rn(value, *, loc=None, ip=None):
        result = llvm.inline_asm(
            cutlass.Float32.mlir_type,
            [cutlass.Float32(value).ir_value(loc=loc, ip=ip)],
            "cvt.rn.tf32.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        return cutlass.Float32(result)

    class _RouterGateTF32Gemm:
        def __init__(self, *, wgrad):
            self.tile = (128, 128 if wgrad else 256, 64)
            self.num_stages = 2
            self.wgrad = wgrad
            self.num_threads = 512

        @cute.jit
        def __call__(
            self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, stream: cuda.CUstream
        ):
            bm, bn, bk = self.tile
            a_major = OperandMajorMode.MN
            b_major = OperandMajorMode.MN if self.wgrad else OperandMajorMode.K
            mma = sm100.make_trivial_tiled_mma(
                cutlass.TFloat32,
                cutlass.TFloat32,
                a_major,
                b_major,
                cutlass.Float32,
                tcgen05.CtaGroup.ONE,
                (bm, bn),
            )
            a_layout = sm100.make_smem_layout(
                a_major, (bm, bk), cutlass.TFloat32, self.num_stages
            )
            b_layout = sm100.make_smem_layout(
                b_major, (bn, bk), cutlass.TFloat32, self.num_stages
            )
            self.kernel(a, b, c, mma, a_layout, b_layout).launch(
                grid=(cute.size(a, mode=[0]) // bm, cute.size(b, mode=[0]) // bn, 1),
                block=(self.num_threads + 128, 1, 1),
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
            operand_free = smem.allocate_tensor(
                cutlass.Int64, cute.make_layout(self.num_stages), 8
            )
            operand_ready = smem.allocate_tensor(
                cutlass.Int64, cute.make_layout(self.num_stages), 8
            )
            tmem_slot = smem.allocate_tensor(cutlass.Int32, cute.make_layout(1), 4)
            a_shared = smem.allocate_tensor(
                cutlass.TFloat32, a_layout.outer, 128, swizzle=a_layout.inner
            )
            b_shared = smem.allocate_tensor(
                cutlass.TFloat32, b_layout.outer, 128, swizzle=b_layout.inner
            )
            # Keep the shared-memory swizzle when aliasing TF32 storage as FP32.
            a_shared_float = cute.make_tensor(
                cute.recast_ptr(
                    a_shared.iterator, swizzle_=a_layout.inner, dtype=cutlass.Float32
                ),
                a_shared.layout,
            )
            b_shared_float = cute.make_tensor(
                cute.recast_ptr(
                    b_shared.iterator, swizzle_=b_layout.inner, dtype=cutlass.Float32
                ),
                b_shared.layout,
            )
            c_layout = sm100.make_smem_layout(
                OperandMajorMode.MN, (bm, 32), cutlass.BFloat16, 1
            )
            c_shared = smem.allocate_tensor(
                cutlass.BFloat16, c_layout.outer, 128, swizzle=c_layout.inner
            )
            if warp == 0:
                with cute.arch.elect_one():
                    for slot in cutlass.range_constexpr(self.num_stages):
                        cute.arch.mbarrier_init(operand_free.iterator + slot, 1)
                        cute.arch.mbarrier_init(operand_ready.iterator + slot, 1)
                cute.arch.alloc_tmem(bn, tmem_slot.iterator, is_two_cta=False)
            cute.arch.mbarrier_init_fence()
            cute.arch.barrier()
            mma_thread = mma.get_slice(0)
            a_fragment = mma.make_fragment_A(mma_thread.partition_A(a_shared))
            b_fragment = mma.make_fragment_B(mma_thread.partition_B(b_shared))
            acc_shape = mma.partition_shape_C((bm, bn))
            acc_layout = mma.make_fragment_C(acc_shape).layout
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                cutlass.Float32,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_slot.iterator,
            )
            accumulator = cute.make_tensor(tmem_ptr, acc_layout)
            a_global = cute.local_tile(a, (bm, bk), (tile_m, None))
            b_global = cute.local_tile(b, (bn, bk), (tile_n, None))
            a_threads = cute.make_layout(
                (bm // 8, self.num_threads // (bm // 8)), stride=(1, bm // 8)
            )
            a_values = cute.make_layout((8, 1))
            a_copy = cute.make_tiled_copy_tv(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    cutlass.BFloat16,
                    num_bits_per_copy=128,
                ),
                a_threads,
                a_values,
            )
            a_thread = a_copy.get_slice(tid)
            a_source = a_thread.partition_S(a_global)
            a_destination = a_thread.partition_D(a_shared_float)
            a_registers = cute.make_rmem_tensor(
                a_source[None, None, None, 0].shape, cutlass.BFloat16
            )
            a_converted = cute.make_rmem_tensor(a_registers.shape, cutlass.Float32)
            if cutlass.const_expr(self.wgrad):
                b_threads = cute.make_layout(
                    (bn // 4, self.num_threads // (bn // 4)), stride=(1, bn // 4)
                )
                b_values = cute.make_layout((4, 1))
            else:
                b_threads = cute.make_layout(
                    (self.num_threads // (bk // 4), bk // 4), stride=(bk // 4, 1)
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
            b_destination = b_thread.partition_D(b_shared_float)
            b_registers = cute.make_rmem_tensor(
                b_source[None, None, None, 0].shape, cutlass.Float32
            )
            store_float = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128
            )
            mma.set(tcgen05.Field.ACCUMULATE, False)
            steps = cute.size(a_global.shape[2])
            if tid < self.num_threads:
                for producer_step in cutlass.range(steps, unroll=1):
                    slot = producer_step % self.num_stages
                    cute.arch.mbarrier_wait(
                        operand_free.iterator + slot,
                        (producer_step // self.num_stages + 1) % 2,
                    )
                    cute.copy(
                        a_copy, a_source[None, None, None, producer_step], a_registers
                    )
                    cute.copy(
                        b_copy, b_source[None, None, None, producer_step], b_registers
                    )
                    a_converted.store(a_registers.load().to(cutlass.Float32))
                    cute.copy(
                        store_float, a_converted, a_destination[None, None, None, slot]
                    )
                    for index in cutlass.range_constexpr(cute.size(b_registers)):
                        b_registers[index] = _tf32_rn(b_registers[index])
                    cute.copy(
                        store_float, b_registers, b_destination[None, None, None, slot]
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.barrier(barrier_id=1, number_of_threads=self.num_threads)
                    if tid == 0:
                        cute.arch.mbarrier_arrive(operand_ready.iterator + slot)
            if warp == self.num_threads // 32:
                for consumer_step in cutlass.range(steps, unroll=1):
                    slot = consumer_step % self.num_stages
                    cute.arch.mbarrier_wait(
                        operand_ready.iterator + slot,
                        consumer_step // self.num_stages % 2,
                    )
                    for block_k in cutlass.range_constexpr(
                        cute.size(a_fragment.shape[2])
                    ):
                        coordinate = (None, None, block_k, slot)
                        cute.gemm(
                            mma,
                            accumulator,
                            a_fragment[coordinate],
                            b_fragment[coordinate],
                            accumulator,
                        )
                        mma.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(operand_free.iterator + slot)
            cute.arch.mbarrier_wait(
                operand_free.iterator + (steps - 1) % self.num_stages,
                (steps - 1) // self.num_stages % 2,
            )
            cute.arch.barrier()
            c_global = cute.local_tile(c, (bm, bn), (tile_m, tile_n))
            epilogue = (bm, 32)
            acc_tiles = cute.flat_divide(accumulator[(None, None), 0, 0], epilogue)
            c_tiles = cute.flat_divide(c_global, epilogue)
            load_atom = sm100.get_tmem_load_op(
                self.tile,
                utils.LayoutEnum.COL_MAJOR,
                cutlass.BFloat16,
                cutlass.Float32,
                epilogue,
                False,
            )
            load = tcgen05.make_tmem_copy(load_atom, acc_tiles[None, None, 0, 0])
            thread = load.get_slice(tid)
            acc_source = thread.partition_S(acc_tiles)
            c_destination = thread.partition_D(c_tiles)
            result = cute.make_rmem_tensor(
                c_destination[None, None, None, 0, 0].shape, cutlass.Float32
            )
            rounded = cute.make_rmem_tensor(result.shape, cutlass.BFloat16)
            shared_store = sm100.get_smem_store_op(
                utils.LayoutEnum.COL_MAJOR, cutlass.BFloat16, cutlass.Float32, load
            )
            shared_copy = cute.make_tiled_copy_D(shared_store, load)
            shared_thread = shared_copy.get_slice(tid)
            shared_destination = shared_thread.partition_D(c_shared)
            shared_source = shared_copy.retile(rounded)
            c_copy = cute.make_tiled_copy_tv(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    cutlass.BFloat16,
                    num_bits_per_copy=128,
                ),
                cute.make_layout((bm // 8, 128 // (bm // 8)), stride=(1, bm // 8)),
                a_values,
            )
            c_thread = c_copy.get_slice(tid)
            vector_source = c_thread.partition_S(c_shared)
            vector_destination = c_thread.partition_D(c_tiles)
            vector_registers = cute.make_rmem_tensor(
                vector_source[None, None, None, 0].shape, cutlass.BFloat16
            )
            for column in cutlass.range_constexpr(bn // 32):
                if tid < 128:
                    cute.copy(load, acc_source[None, None, None, 0, column], result)
                    rounded.store(result.load().to(cutlass.BFloat16))
                if tid < 128:
                    cute.copy(
                        shared_copy,
                        shared_source,
                        shared_destination[None, None, None, 0],
                    )
                cute.arch.barrier()
                if tid < 128:
                    cute.copy(
                        c_copy, vector_source[None, None, None, 0], vector_registers
                    )
                    cute.copy(
                        c_copy,
                        vector_registers,
                        vector_destination[None, None, None, 0, column],
                    )
                cute.arch.barrier()
            cute.arch.barrier()
            if warp == 0:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
                cute.arch.dealloc_tmem(tmem_ptr, bn, is_two_cta=False)

    @lru_cache(maxsize=4)
    def _compile_gate_gemm(wgrad, device):
        reduction, columns = (4096, 256) if wgrad else (256, 4096)
        arguments = (
            make_fake_tensor(
                cutlass.BFloat16, (7168, reduction), (1, 7168), assumed_align=16
            ),
            make_fake_tensor(
                cutlass.Float32,
                (columns, reduction),
                (1, 256) if wgrad else (256, 1),
                assumed_align=16,
            ),
            make_fake_tensor(
                cutlass.BFloat16, (7168, columns), (1, 7168), assumed_align=16
            ),
        )
        return cute.compile(
            _RouterGateTF32Gemm(wgrad=wgrad),
            *arguments,
            cuda.CUstream(torch.cuda.current_stream(device).cuda_stream),
        )

    def _gate_gemm(grad_output_TE, operand_RD, *, wgrad):
        rows = 256 if wgrad else 4096
        output_RD = operand_RD.new_empty((rows, 7168))
        arguments = (
            operand_RD.t(),
            grad_output_TE.t() if wgrad else grad_output_TE,
            output_RD.t(),
        )
        kernel = _compile_gate_gemm(wgrad, str(grad_output_TE.device))
        kernel(
            *(from_dlpack(value.detach(), assumed_align=16) for value in arguments),
            cuda.CUstream(torch.cuda.current_stream(grad_output_TE.device).cuda_stream),
        )
        return output_RD


def _supports_gate_backward(grad_output_TE, operand_RD, *, wgrad):
    rows = 4096 if wgrad else 256
    tensors = (grad_output_TE, operand_RD)
    return (
        _CUTEDSL_IMPORT_ERROR is None
        and _NATIVE_MM is not None
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
        and os.environ.get("NVIDIA_TF32_OVERRIDE") != "0"
        and torch.__version__ == "2.15.0a0+giteecfcac"
        and (
            isinstance(grad_output_TE, FakeTensor)
            or (
                all(value.data_ptr() % 16 == 0 for value in tensors)
                and torch.cuda.get_device_capability(grad_output_TE.device) == (10, 3)
            )
        )
    )


@torch.library.custom_op(
    "torchtitan::dsv3_router_gate_tf32_dgrad", mutates_args=(), device_types="cuda"
)
def dgrad_op(grad_output_TE: torch.Tensor, weight_ED: torch.Tensor) -> torch.Tensor:
    if not _supports_gate_backward(grad_output_TE, weight_ED, wgrad=False):
        _require_native_gate()
        return _NATIVE_MM(
            grad_output_TE.float(),
            weight_ED.float(),
            compute_mode="tf32",
            out_dtype=torch.float32,
        ).to(weight_ED.dtype)
    return _gate_gemm(grad_output_TE, weight_ED, wgrad=False)


@dgrad_op.register_fake
def _dgrad_fake(grad_output_TE, weight_ED):
    return weight_ED.new_empty((grad_output_TE.shape[0], weight_ED.shape[1]))


@torch.library.custom_op(
    "torchtitan::dsv3_router_gate_tf32_wgrad", mutates_args=(), device_types="cuda"
)
def wgrad_op(grad_output_TE: torch.Tensor, input_TD: torch.Tensor) -> torch.Tensor:
    if not _supports_gate_backward(grad_output_TE, input_TD, wgrad=True):
        _require_native_gate()
        return _NATIVE_MM(
            grad_output_TE.float().t(),
            input_TD.float(),
            compute_mode="tf32",
            out_dtype=torch.float32,
        ).to(input_TD.dtype)
    return _gate_gemm(grad_output_TE, input_TD, wgrad=True)


@wgrad_op.register_fake
def _wgrad_fake(grad_output_TE, input_TD):
    return input_TD.new_empty((grad_output_TE.shape[1], input_TD.shape[1]))


class FusedDSv3TF32RouterGateFunction(torch.autograd.Function):
    """Native forward and two fused cast/TF32-GEMM/BF16-cast backward operators."""

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx,
        input_TD,
        weight_ED,
        forward_compute,
        forward_output_dtype,
        backward_input_dtype,
        backward_compute,
        backward_output_dtype,
    ):
        return _require_native_gate().forward(
            ctx,
            input_TD,
            weight_ED,
            forward_compute,
            forward_output_dtype,
            backward_input_dtype,
            backward_compute,
            backward_output_dtype,
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output_TE):  # pyrefly: ignore[bad-override]
        input_TD, weight_ED = ctx.saved_tensors
        if not (
            ctx.backward_input_dtype == torch.float32
            and ctx.backward_compute == "tf32"
            and ctx.backward_output_dtype == torch.float32
            and (
                torch.compiler.is_compiling()
                or (
                    _supports_gate_backward(grad_output_TE, weight_ED, wgrad=False)
                    and _supports_gate_backward(grad_output_TE, input_TD, wgrad=True)
                )
            )
        ):
            return _require_native_gate().backward(ctx, grad_output_TE)
        grad_input_TD = (
            dgrad_op(grad_output_TE, weight_ED) if ctx.needs_input_grad[0] else None
        )
        grad_weight_ED = (
            wgrad_op(grad_output_TE, input_TD) if ctx.needs_input_grad[1] else None
        )
        return grad_input_TD, grad_weight_ED, None, None, None, None, None


if RouterGateLinear is not None:

    class FusedDSv3TF32RouterGate(RouterGateLinear):
        forward_config: Any
        backward_config: Any
        weight: torch.Tensor
        bias: torch.Tensor | None
        out_features: int

        @dataclass(kw_only=True, slots=True)
        class Config(RouterGateLinear.Config):
            use_fused_backward: bool = False

        def __init__(self, config):
            super().__init__(config)
            self.use_fused_backward = config.use_fused_backward

        def forward(self, input_BLD):
            if not self.use_fused_backward:
                return super().forward(input_BLD)
            input_shape = input_BLD.shape
            input_TD = input_BLD.reshape(-1, input_shape[-1])
            forward = self.forward_config
            backward = self.backward_config
            with torch.autocast(device_type=input_BLD.device.type, enabled=False):
                output_TE = FusedDSv3TF32RouterGateFunction.apply(
                    input_TD.to(forward.input_dtype),
                    self.weight.to(forward.input_dtype),
                    forward.compute_mode,
                    forward.output_dtype,
                    backward.input_dtype,
                    backward.compute_mode,
                    backward.output_dtype,
                )
                if self.bias is not None:
                    output_TE = output_TE + self.bias.to(forward.output_dtype)
            return output_TE.reshape(*input_shape[:-1], self.out_features)

else:
    FusedDSv3TF32RouterGate = None


def fused_dsv3_router_gate_tf32(cfg, *, experimental=False):
    """Fuse an existing TF32 gate without changing its numerical policy."""
    _require_native_gate()
    if FusedDSv3TF32RouterGate is None:
        raise RuntimeError("The pinned RouterGateLinear implementation is required.")
    enabled = ACCEPTED or experimental
    if not ACCEPTED:
        warnings.warn(
            "TF32 router-gate roofline acceptance is open; experimental=True "
            "enables the candidate. Native backward is the default.",
            stacklevel=2,
        )
    return derive(cfg, FusedDSv3TF32RouterGate.Config, use_fused_backward=enabled)


if RouterGateLinear is not None:
    fused_dsv3_router_gate_tf32 = override(
        target=RouterGateLinear.Config,
        exact=True,
        fqns=["*.moe.router.gate"],
        description="Fuse the DSv3 TF32 router-gate backward with CuTeDSL.",
    )(fused_dsv3_router_gate_tf32)
