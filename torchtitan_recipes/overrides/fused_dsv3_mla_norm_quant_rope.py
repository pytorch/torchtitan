# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Experimental DeepSeek V3 MLA RMSNorm, MXFP8, and RoPE fusion.

T = local tokens (4096), D = model width (7168), H = heads (128),
Q = query rank (1536), K = KV rank (512), R = rotary width (64).
The CuTeDSL kernels match the pinned runtime's CuTe RMSNorm, BF16 RoPE rounding,
and RCEIL scale bytes. The adapter retains native MXFP8 weight-cache and
gradient policy. Native RoPE autotuning must be pinned by deterministic mode.
"""

import logging
from dataclasses import dataclass
from importlib.metadata import version
from typing import TYPE_CHECKING

import spmd_types as spmd
import torch
import torch.nn.functional as F
import torch_remat as remat
import triton

from packaging.version import Version
from torch._subclasses.fake_tensor import FakeTensor
from torch.autograd.function import once_differentiable
from torch.distributed.fsdp import FSDPModule
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torchao.prototype.mx_formats.kernels import (
    mxfp8_quantize_cuda,
    triton_mx_block_rearrange,
)
from torchtitan.config import derive, override
from torchtitan.distributed.spmd_types import spmd_mesh_size
from torchtitan.models.common.linear import maybe_gather_tp_input
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import _maybe_check_max_pos
from torchtitan.models.deepseek_v3.model import Attention
from torchtitan.quantization._fsdp_tensor import _UnshardedFSDPTensor
from torchtitan.quantization.mxfp8.linear import MXFP8Linear
from torchtitan.quantization.mxfp8.tensor import (
    _LinearShardedTensorWithMXFP8Compute,
    _quantize_mxfp8_weight,
)

from torchtitan_recipes.overrides.fused_mla import (
    _fused_mla_k_rope_op,
    _fused_mla_kv_backward_op,
    _fused_mla_q_rope_op,
    _resolve_positions,
    FusedMLAAttention,
)


# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


try:
    if Version(version("nvidia-cutlass-dsl")) < Version("4.8.0"):
        raise ImportError("MLA kernels require CuTeDSL 4.8 or newer.")
    import cutlass
    from cuda.bindings import driver as cuda
    from cutlass import cute, Float32, Int32, Int64, pipeline, Uint32, Uint8, utils
    from cutlass.cute import arch
    from cutlass.cute.nvgpu import cpasync, tcgen05
    from cutlass.cute.runtime import from_dlpack, make_ptr
    from cutlass.memory import SmemAllocator
    from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
    from cutlass.utils import (
        blackwell_helpers as sm100_utils,
        blockscaled_layout as blockscaled_utils,
    )

    _CUTEDSL_IMPORT_ERROR: ImportError | None = None
except ImportError as error:
    _CUTEDSL_IMPORT_ERROR = error


if TYPE_CHECKING or _CUTEDSL_IMPORT_ERROR is None:

    @cute.jit
    def _mul(a, b):
        # Separate FP32 operations preserve native RoPE and RMSNorm rounding.
        return arch.inline_ptx(
            "mul.rn.f32 {$w0}, {$r0}, {$r1};",
            write_only_types=[Float32],
            read_only_args=[a, b],
        )

    @cute.jit
    def _add(a, b):
        return arch.inline_ptx(
            "add.rn.f32 {$w0}, {$r0}, {$r1};",
            write_only_types=[Float32],
            read_only_args=[a, b],
        )

    @cute.jit
    def _low(word):
        return arch.inline_ptx(
            "mov.b32 {$w0}, {$r0};",
            write_only_types=[Float32],
            read_only_args=[word << 16],
        )

    @cute.jit
    def _high(word):
        return arch.inline_ptx(
            "mov.b32 {$w0}, {$r0};",
            write_only_types=[Float32],
            read_only_args=[word & Uint32(4294901760)],
        )

    @cute.jit
    def _pack(lo, hi):
        return arch.inline_ptx(
            "cvt.rn.bf16x2.f32 {$w0}, {$r1}, {$r0};",
            write_only_types=[Uint32],
            read_only_args=[lo, hi],
        )

    @cute.jit
    def _absmax(a, b):
        return arch.inline_ptx(
            "max.NaN.xorsign.abs.bf16x2 {$w0}, {$r0}, {$r1};",
            write_only_types=[Uint32],
            read_only_args=[a, b],
        )

    @cute.jit
    def _maximum(a, b):
        return arch.inline_ptx(
            "max.NaN.f32 {$w0}, {$r0}, {$r1};",
            write_only_types=[Float32],
            read_only_args=[a, b],
        )

    @cute.jit
    def _scale(maximum):
        return arch.inline_ptx(
            (
                "{.reg .b16 encoded; .reg .b32 exponent, reciprocal; .reg .f32 scaled; .reg .pred "
                "finite,p254,p255; mul.rn.f32 scaled, {$r0}, 0f3B124925; cvt.rp.satfinite.ue8m0x2.f32 "
                "encoded,0.0,scaled; cvt.u32.u16 exponent,encoded; and.b32 exponent,exponent,255; "
                "setp.lt.f32 finite,{$r0},0f7F800000; selp.u32 exponent,exponent,255,finite; sub.u32 "
                "reciprocal,254,exponent; shl.b32 reciprocal,reciprocal,7; setp.eq.u32 p254,exponent,254; "
                "selp.u32 reciprocal,64,reciprocal,p254; setp.eq.u32 p255,exponent,255; selp.u32 "
                "reciprocal,32704,reciprocal,p255; mov.b32 {$w0},reciprocal; mov.b32 {$w1},exponent;}"
            ),
            write_only_types=[Uint32, Uint32],
            read_only_args=[maximum],
        )

    @cute.jit
    def _blocked(row, block, blocks: cutlass.Constexpr):
        return (
            (row // 128 * ((blocks + 3) // 4) + block // 4) * 512
            + row % 32 * 16
            + row % 128 // 32 * 4
            + block % 4
        )

    @cute.jit
    def _load4(ptr):
        return arch.inline_ptx(
            "ld.global.v4.b32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
            write_only_types=[Uint32] * 4,
            read_only_args=[ptr],
        )

    @cute.jit
    def _store4(ptr, a, b, c, d):
        arch.inline_ptx(
            "st.global.v4.b32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
            read_only_args=[ptr, a, b, c, d],
        )

    @cute.jit
    def _quant4(a, b, reciprocal):
        return arch.inline_ptx(
            (
                "{.reg .b32 a,b; .reg .b16 x,y; mul.rn.bf16x2 a, {$r0}, {$r2}; mul.rn.bf16x2 b, {$r1}, "
                "{$r2}; cvt.rn.satfinite.e4m3x2.bf16x2 x,a; cvt.rn.satfinite.e4m3x2.bf16x2 y,b; mov.b32 "
                "{$w0}, {x,y};}"
            ),
            write_only_types=[Uint32],
            read_only_args=[a, b, reciprocal],
        )

    @cute.jit
    def _quant_col4(a, b, c, d, reciprocal):
        return arch.inline_ptx(
            (
                "{.reg .b32 a,b,c,d,ab,cd; .reg .b16 x,y,z,w; mul.rn.bf16x2 a, {$r0}, {$r4}; mul.rn.bf16x2 "
                "b, {$r1}, {$r4}; mul.rn.bf16x2 c, {$r2}, {$r4}; mul.rn.bf16x2 d, {$r3}, {$r4}; "
                "cvt.rn.satfinite.e4m3x2.bf16x2 x,a; cvt.rn.satfinite.e4m3x2.bf16x2 y,b; "
                "cvt.rn.satfinite.e4m3x2.bf16x2 z,c; cvt.rn.satfinite.e4m3x2.bf16x2 w,d; mov.b32 ab,{x,y}; "
                "mov.b32 cd,{z,w}; prmt.b32 {$w0},ab,cd,0x6420; prmt.b32 {$w1},ab,cd,0x7531;}"
            ),
            write_only_types=[Uint32, Uint32],
            read_only_args=[a, b, c, d, reciprocal],
        )

    @cute.jit
    def _word_view(tensor, swizzle):
        # cute.recast_tensor drops the pointer swizzle; preserve it explicitly.
        return cute.make_tensor(
            cute.recast_ptr(tensor.iterator, swizzle_=swizzle, dtype=Uint32),
            cute.recast_layout(32, tensor.element_type.width, tensor.layout),
        )

    class _NormQuant:
        def __init__(self, m, n, stride, eps, colwise, threads, splits, tma_col):
            self.m, self.n, self.stride, self.eps = (m, n, stride, eps)
            self.colwise, self.threads, self.splits = (colwise, threads, splits)

        @cute.jit
        def __call__(self, args, stream: cuda.CUstream):
            if cutlass.const_expr(self.colwise and False):
                col = cute.make_tensor(
                    cute.recast_ptr(args[4], dtype=Uint8),
                    cute.make_layout((self.n, self.m), stride=(self.m, 1)),
                )
                layout = sm100_utils.make_smem_layout_epi(
                    Uint8,
                    utils.LayoutEnum.ROW_MAJOR,
                    (256, 32),
                    self.n // self.splits // 256,
                )
                atom, tensor = cpasync.make_tiled_tma_atom(
                    cpasync.CopyBulkTensorTileS2GOp(),
                    col,
                    cute.slice_(layout, (None, None, 0)),
                    (256, 32),
                )
            else:
                atom, tensor, layout = (None, None, None)
            self.kernel(args, atom, tensor, layout).launch(
                grid=(self.m // 32, self.splits, 1),
                block=(self.threads, 1, 1),
                stream=stream,
            )

        @cute.kernel
        def kernel(self, args, col_atom, col_tensor, col_layout):
            x, w, rstd, row, col, row_scale, col_scale = args
            rstd_tensor = cute.make_tensor(rstd, cute.make_layout(self.m))
            row_scale_tensor = cute.make_tensor(
                row_scale, cute.make_layout(self.m * self.n // 32)
            )
            col_scale_tensor = cute.make_tensor(
                col_scale, cute.make_layout(self.m * self.n // 32)
            )
            tid, _, _ = arch.thread_idx()
            tile, split, _ = arch.block_idx()
            lane = tid % 32
            warp = tid // 32
            span = self.n // self.splits
            smem = SmemAllocator()
            normalized = smem.allocate_tensor(
                Uint32, cute.make_layout((32, span // 2), stride=(span // 2, 1)), 16
            )
            if cutlass.const_expr(self.colwise and False):
                quantized = smem.allocate_tensor(
                    Uint8, col_layout.outer, 1024, swizzle=col_layout.inner
                )
                quantized_words = _word_view(quantized, col_layout.inner)
                shared_col, global_col = cpasync.tma_partition(
                    col_atom,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(quantized, 0, 2),
                    cute.group_modes(
                        cute.local_tile(col_tensor, (256, 32), (None, None)), 0, 2
                    ),
                )
            packed = cute.make_rmem_tensor(4, Uint32)
            if cutlass.const_expr(self.splits == 1):
                raw_saved = cute.make_rmem_tensor((self.n // 256, 4), Uint32)
            rows = tile * 32
            for row_iter in cutlass.range_constexpr(32 // (self.threads // 32)):
                local_row = warp + row_iter * (self.threads // 32)
                r = rows + local_row
                total = Float32(0)
                for chunk in cutlass.range_constexpr(self.n // 256):
                    words = _load4(x + r * (self.stride // 2) + chunk * 128 + lane * 4)
                    for i in cutlass.range_constexpr(4):
                        if cutlass.const_expr(self.splits == 1):
                            raw_saved[chunk, i] = words[i]
                        even_value, odd_value = (_low(words[i]), _high(words[i]))
                        total = _add(total, _mul(even_value, even_value))
                        total = _add(total, _mul(odd_value, odd_value))
                for offset in cutlass.range_constexpr(5):
                    total = _add(total, arch.shuffle_sync_bfly(total, 16 >> offset))
                mean = arch.inline_ptx(
                    "div.rn.f32 {$w0}, {$r0}, {$r1};",
                    write_only_types=[Float32],
                    read_only_args=[total, Float32(self.n)],
                )
                inverse = cute.math.rsqrt(_add(mean, Float32(self.eps)), fastmath=True)
                if lane == 0 and split == 0:
                    rstd_tensor[r] = inverse
                for chunk in cutlass.range_constexpr(span // 256):
                    source_chunk = split * (span // 256) + chunk
                    gamma = _load4(w + source_chunk * 128 + lane * 4)
                    if cutlass.const_expr(self.splits != 1):
                        raw = _load4(
                            x + r * (self.stride // 2) + source_chunk * 128 + lane * 4
                        )
                    for i in cutlass.range_constexpr(4):
                        if cutlass.const_expr(self.splits == 1):
                            word = raw_saved[chunk, i]
                        else:
                            word = raw[i]
                        packed[i] = _pack(
                            _mul(_mul(_low(word), inverse), _low(gamma[i])),
                            _mul(_mul(_high(word), inverse), _high(gamma[i])),
                        )
                        if cutlass.const_expr(self.colwise):
                            normalized[local_row, chunk * 128 + lane * 4 + i] = packed[
                                i
                            ]
                    maximum = _absmax(
                        _absmax(packed[0], packed[1]), _absmax(packed[2], packed[3])
                    )
                    maximum = _absmax(maximum, arch.shuffle_sync_bfly(maximum, 1))
                    maximum = _absmax(maximum, arch.shuffle_sync_bfly(maximum, 2))
                    reciprocal, exponent = _scale(
                        _maximum(
                            _low(maximum & Uint32(2147450879)),
                            _high(maximum & Uint32(2147450879)),
                        )
                    )
                    reciprocal = reciprocal | reciprocal << 16
                    out_word0 = _quant4(packed[0], packed[1], reciprocal)
                    out_word1 = _quant4(packed[2], packed[3], reciprocal)
                    arch.inline_ptx(
                        "st.global.v2.b32 [{$r0}], {{$r1}, {$r2}};",
                        read_only_args=[
                            row + r * (self.n // 4) + source_chunk * 64 + lane * 2,
                            out_word0,
                            out_word1,
                        ],
                    )
                    if lane % 4 == 0:
                        row_scale_tensor[
                            _blocked(r, source_chunk * 8 + lane // 4, self.n // 32)
                        ] = exponent.to(Uint8)
            if cutlass.const_expr(self.colwise):
                arch.barrier()
                column_words = cute.make_rmem_tensor(32, Uint32)
                even = cute.make_rmem_tensor(8, Uint32)
                odd = cute.make_rmem_tensor(8, Uint32)
                for column_iter in cutlass.range_constexpr(
                    (span // 2 + self.threads - 1) // self.threads
                ):
                    local_column = tid + column_iter * self.threads
                    if local_column < span // 2:
                        for r in cutlass.range_constexpr(32):
                            column_words[r] = normalized[r, local_column]
                        maximum = column_words[0]
                        for r in cutlass.range_constexpr(1, 32):
                            maximum = _absmax(maximum, column_words[r])
                        lo_inverse, lo_exponent = _scale(
                            _low(maximum & Uint32(2147450879))
                        )
                        hi_inverse, hi_exponent = _scale(
                            _high(maximum & Uint32(2147450879))
                        )
                        reciprocal = lo_inverse | hi_inverse << 16
                        for r in cutlass.range_constexpr(8):
                            even_word, odd_word = _quant_col4(
                                column_words[r * 4],
                                column_words[r * 4 + 1],
                                column_words[r * 4 + 2],
                                column_words[r * 4 + 3],
                                reciprocal,
                            )
                            even[r], odd[r] = (even_word, odd_word)
                        column = split * span + local_column * 2
                        for group in cutlass.range_constexpr(2):
                            r = group * 4
                            _store4(
                                col + column * (self.m // 4) + rows // 4 + r,
                                even[r],
                                even[r + 1],
                                even[r + 2],
                                even[r + 3],
                            )
                            _store4(
                                col + (column + 1) * (self.m // 4) + rows // 4 + r,
                                odd[r],
                                odd[r + 1],
                                odd[r + 2],
                                odd[r + 3],
                            )
                        col_scale_tensor[
                            _blocked(column, rows // 32, self.m // 32)
                        ] = lo_exponent.to(Uint8)
                        col_scale_tensor[
                            _blocked(column + 1, rows // 32, self.m // 32)
                        ] = hi_exponent.to(Uint8)

    class _MLAUpGemm:
        """Persistent two-CTA MXFP8 GEMM, derived from CUTLASS v4.3 (BSD above)."""

        def __init__(
            self,
            sf_vec_size: int,
            mma_tiler_mn: tuple[int, int],
            cluster_shape_mn: tuple[int, int],
            *,
            mode=1,
            block_k=128,
            stages=None,
            c_stages=None,
            acc_stages=None,
            order=0,
        ):
            self.mode = mode
            self.block_k = block_k
            self.stages = stages
            self.c_stages = c_stages
            self.acc_stages = acc_stages
            self.order = order
            self.acc_dtype = cutlass.Float32
            self.sf_vec_size = sf_vec_size
            self.use_2cta_instrs = mma_tiler_mn[0] == 256
            self.cluster_shape_mn = cluster_shape_mn
            self.mma_tiler = (*mma_tiler_mn, 1)
            self.cta_group = (
                tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
            )
            self.occupancy = 1
            self.epilog_warp_id = (0, 1, 2, 3)
            self.mma_warp_id = 4
            self.tma_warp_id = 5
            self.threads_per_cta = 32 * len(
                (self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id)
            )
            self.epilog_sync_barrier = pipeline.NamedBarrier(
                barrier_id=1, num_threads=32 * len(self.epilog_warp_id)
            )
            self.tmem_alloc_barrier = pipeline.NamedBarrier(
                barrier_id=2,
                num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
            )
            self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100") - (
                16384 if mode == 2 else 0
            )
            SM100_TMEM_CAPACITY_COLUMNS = 512
            self.num_tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        def _setup_attributes(self):
            self.mma_inst_shape_mn = (self.mma_tiler[0], self.mma_tiler[1])
            self.mma_inst_shape_mn_sfb = (
                self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
                cute.round_up(self.mma_inst_shape_mn[1], 128),
            )
            tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_inst_shape_mn,
            )
            tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                cute.nvgpu.tcgen05.CtaGroup.ONE,
                self.mma_inst_shape_mn_sfb,
            )
            mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
            mma_inst_tile_k = self.block_k // mma_inst_shape_k
            self.mma_tiler = (
                self.mma_inst_shape_mn[0],
                self.mma_inst_shape_mn[1],
                mma_inst_shape_k * mma_inst_tile_k,
            )
            self.mma_tiler_sfb = (
                self.mma_inst_shape_mn_sfb[0],
                self.mma_inst_shape_mn_sfb[1],
                mma_inst_shape_k * mma_inst_tile_k,
            )
            self.cta_tile_shape_mnk = (
                self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
                self.mma_tiler[1],
                self.mma_tiler[2],
            )
            self.cta_tile_shape_mnk_sfb = (
                self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
                self.mma_tiler_sfb[1],
                self.mma_tiler_sfb[2],
            )
            self.cluster_layout_vmnk = cute.tiled_divide(
                cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
            )
            self.cluster_layout_sfb_vmnk = cute.tiled_divide(
                cute.make_layout((*self.cluster_shape_mn, 1)),
                (tiled_mma_sfb.thr_id.shape,),
            )
            self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
            self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
            self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
            self.is_a_mcast = self.num_mcast_ctas_a > 1
            self.is_b_mcast = self.num_mcast_ctas_b > 1
            self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1
            self.epi_tile = (128, 64)
            self.epi_tile_n = cute.size(self.epi_tile[1])
            (
                self.num_acc_stage,
                self.num_ab_stage,
                self.num_c_stage,
            ) = self._compute_stages(
                tiled_mma,
                self.mma_tiler,
                self.a_dtype,
                self.b_dtype,
                self.epi_tile,
                self.c_dtype,
                self.c_layout,
                self.sf_dtype,
                self.sf_vec_size,
                self.smem_capacity,
                self.occupancy,
            )
            if self.stages is not None:
                self.num_ab_stage = self.stages
            if self.c_stages is not None:
                self.num_c_stage = self.c_stages
            if self.acc_stages is not None:
                self.num_acc_stage = self.acc_stages
            self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
                tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
            )
            self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
                tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
            )
            self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
                tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage
            )
            self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
                tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage
            )
            self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
                self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage
            )
            sf_atom_mn = 32
            self.num_sfa_tmem_cols = (
                self.cta_tile_shape_mnk[0] // sf_atom_mn * mma_inst_tile_k
            )
            self.num_sfb_tmem_cols = (
                self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn * mma_inst_tile_k
            )
            self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
            self.num_accumulator_tmem_cols = (
                self.cta_tile_shape_mnk[1] * self.num_acc_stage
            )
            self.iter_acc_early_release_in_epilogue = (
                self.num_sf_tmem_cols // self.epi_tile_n
            )

        @cute.jit
        def __call__(
            self,
            a_tensor: cute.Tensor,
            b_tensor: cute.Tensor,
            sfa_tensor: cute.Tensor,
            sfb_tensor: cute.Tensor,
            c_tensor: cute.Tensor,
            k_tensor: cute.Tensor,
            k_pe: cute.Tensor,
            cache: cute.Tensor,
            positions: cute.Tensor,
            max_active_clusters: cutlass.Constexpr,
            stream: cuda.CUstream,
        ):
            self.a_dtype: type[cutlass.Numeric] = a_tensor.element_type
            self.b_dtype: type[cutlass.Numeric] = b_tensor.element_type
            self.sf_dtype: type[cutlass.Numeric] = sfa_tensor.element_type
            self.c_dtype: type[cutlass.Numeric] = c_tensor.element_type
            self.a_major_mode = utils.LayoutEnum.from_tensor(a_tensor).mma_major_mode()
            self.b_major_mode = utils.LayoutEnum.from_tensor(b_tensor).mma_major_mode()
            self.c_layout = utils.LayoutEnum.from_tensor(c_tensor)
            if cutlass.const_expr(self.a_dtype != self.b_dtype):
                raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")
            self.k_tiles = a_tensor.shape[1] // self.block_k
            self._setup_attributes()
            sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
                a_tensor.shape, self.sf_vec_size
            )
            sfa_tensor = cute.make_tensor(sfa_tensor.iterator, sfa_layout)
            sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
                b_tensor.shape, self.sf_vec_size
            )
            sfb_tensor = cute.make_tensor(sfb_tensor.iterator, sfb_layout)
            tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_inst_shape_mn,
            )
            tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                cute.nvgpu.tcgen05.CtaGroup.ONE,
                self.mma_inst_shape_mn_sfb,
            )
            atom_thr_size = cute.size(tiled_mma.thr_id.shape)
            a_op = sm100_utils.cluster_shape_to_tma_atom_A(
                self.cluster_shape_mn, tiled_mma.thr_id
            )
            a_smem_layout = cute.slice_(
                self.a_smem_layout_staged, (None, None, None, 0)
            )
            tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
                a_op,
                a_tensor,
                a_smem_layout,
                self.mma_tiler,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
            b_op = sm100_utils.cluster_shape_to_tma_atom_B(
                self.cluster_shape_mn, tiled_mma.thr_id
            )
            b_smem_layout = cute.slice_(
                self.b_smem_layout_staged, (None, None, None, 0)
            )
            tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
                b_op,
                b_tensor,
                b_smem_layout,
                self.mma_tiler,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
            sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
                self.cluster_shape_mn, tiled_mma.thr_id
            )
            sfa_smem_layout = cute.slice_(
                self.sfa_smem_layout_staged, (None, None, None, 0)
            )
            tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
                sfa_op,
                sfa_tensor,
                sfa_smem_layout,
                self.mma_tiler,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )
            sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
                self.cluster_shape_mn, tiled_mma.thr_id
            )
            sfb_smem_layout = cute.slice_(
                self.sfb_smem_layout_staged, (None, None, None, 0)
            )
            tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
                sfb_op,
                sfb_tensor,
                sfb_smem_layout,
                self.mma_tiler_sfb,
                tiled_mma_sfb,
                self.cluster_layout_sfb_vmnk.shape,
                internal_type=cutlass.Int16,
            )
            if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                x = tma_tensor_sfb.stride[0][1]
                y = cute.ceil_div(tma_tensor_sfb.shape[0][1], 4)
                new_shape = (
                    (tma_tensor_sfb.shape[0][0], ((2, 2), y)),
                    tma_tensor_sfb.shape[1],
                    tma_tensor_sfb.shape[2],
                )
                x_times_3 = 3 * x
                new_stride = (
                    (tma_tensor_sfb.stride[0][0], ((x, x), x_times_3)),
                    tma_tensor_sfb.stride[1],
                    tma_tensor_sfb.stride[2],
                )
                tma_tensor_sfb_new_layout = cute.make_layout(
                    new_shape, stride=new_stride
                )
                tma_tensor_sfb = cute.make_tensor(
                    tma_tensor_sfb.iterator, tma_tensor_sfb_new_layout
                )
            a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
            b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
            sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
            sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
            self.num_tma_load_bytes = (
                a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
            ) * atom_thr_size
            epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
            tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                c_tensor,
                epi_smem_layout,
                self.epi_tile,
            )
            tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                k_tensor,
                epi_smem_layout,
                self.epi_tile,
            )
            self.tiles_m = c_tensor.shape[0] // self.mma_tiler[0]
            self.tiles_n = cute.ceil_div(c_tensor.shape[1], self.mma_tiler[1])
            self.tile_sched_params, grid = self._compute_grid(
                c_tensor,
                self.cta_tile_shape_mnk,
                self.cluster_shape_mn,
                max_active_clusters,
            )
            self.buffer_align_bytes = 1024

            @cute.struct
            class SharedStorage:
                ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
                ab_empty_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.num_ab_stage
                ]
                acc_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.num_acc_stage
                ]
                acc_empty_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.num_acc_stage
                ]
                tmem_dealloc_mbar_ptr: cutlass.Int64
                tmem_holding_buf: cutlass.Int32
                output: cute.struct.Align[
                    cute.struct.MemRange[
                        self.c_dtype, cute.cosize(self.c_smem_layout_staged.outer)
                    ],
                    self.buffer_align_bytes,
                ]
                activation: cute.struct.Align[
                    cute.struct.MemRange[
                        self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                    ],
                    self.buffer_align_bytes,
                ]
                weight: cute.struct.Align[
                    cute.struct.MemRange[
                        self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                    ],
                    self.buffer_align_bytes,
                ]
                activation_scales: cute.struct.Align[
                    cute.struct.MemRange[
                        self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                    ],
                    self.buffer_align_bytes,
                ]
                weight_scales: cute.struct.Align[
                    cute.struct.MemRange[
                        self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                    ],
                    self.buffer_align_bytes,
                ]

            self.shared_storage = SharedStorage
            self.kernel(
                tiled_mma,
                tiled_mma_sfb,
                tma_atom_a,
                tma_tensor_a,
                tma_atom_b,
                tma_tensor_b,
                tma_atom_sfa,
                tma_tensor_sfa,
                tma_atom_sfb,
                tma_tensor_sfb,
                tma_atom_c,
                tma_tensor_c,
                self.cluster_layout_vmnk,
                self.cluster_layout_sfb_vmnk,
                self.a_smem_layout_staged,
                self.b_smem_layout_staged,
                self.sfa_smem_layout_staged,
                self.sfb_smem_layout_staged,
                self.c_smem_layout_staged,
                self.epi_tile,
                self.tile_sched_params,
                tma_tensor_k,
                tma_atom_k,
                k_pe,
                cache,
                positions,
            ).launch(
                grid=grid,
                block=[self.threads_per_cta, 1, 1],
                cluster=(*self.cluster_shape_mn, 1),
                stream=stream,
                min_blocks_per_mp=1,
            )
            return

        @cute.kernel
        def kernel(
            self,
            tiled_mma: cute.TiledMma,
            tiled_mma_sfb: cute.TiledMma,
            tma_atom_a: cute.CopyAtom,
            mA_mkl: cute.Tensor,
            tma_atom_b: cute.CopyAtom,
            mB_nkl: cute.Tensor,
            tma_atom_sfa: cute.CopyAtom,
            mSFA_mkl: cute.Tensor,
            tma_atom_sfb: cute.CopyAtom,
            mSFB_nkl: cute.Tensor,
            tma_atom_c: cute.CopyAtom,
            mC_mnl: cute.Tensor,
            cluster_layout_vmnk: cute.Layout,
            cluster_layout_sfb_vmnk: cute.Layout,
            a_smem_layout_staged: cute.ComposedLayout,
            b_smem_layout_staged: cute.ComposedLayout,
            sfa_smem_layout_staged: cute.Layout,
            sfb_smem_layout_staged: cute.Layout,
            c_smem_layout_staged: cute.Layout | cute.ComposedLayout,
            epi_tile: cute.Tile,
            tile_sched_params: utils.PersistentTileSchedulerParams,
            mK_mnl: cute.Tensor,
            tma_atom_k: cute.CopyAtom,
            k_pe: cute.Tensor,
            cache: cute.Tensor,
            positions: cute.Tensor,
        ):
            warp_idx = cute.arch.warp_idx()
            warp_idx = cute.arch.make_warp_uniform(warp_idx)
            if warp_idx == self.tma_warp_id:
                cpasync.prefetch_descriptor(tma_atom_a)
                cpasync.prefetch_descriptor(tma_atom_b)
                cpasync.prefetch_descriptor(tma_atom_sfa)
                cpasync.prefetch_descriptor(tma_atom_sfb)
                cpasync.prefetch_descriptor(tma_atom_c)
                cpasync.prefetch_descriptor(tma_atom_k)
            use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
            bidx, bidy, bidz = cute.arch.block_idx()
            mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
            is_leader_cta = mma_tile_coord_v == 0
            cta_rank_in_cluster = cute.arch.make_warp_uniform(
                cute.arch.block_idx_in_cluster()
            )
            block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
                cta_rank_in_cluster
            )
            block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
                cta_rank_in_cluster
            )
            tidx, _, _ = cute.arch.thread_idx()
            smem = utils.SmemAllocator()
            storage = smem.allocate(self.shared_storage)
            if cutlass.const_expr(self.mode == 2):
                kpe_smem_layout = sm100_utils.make_smem_layout_epi(
                    self.c_dtype, self.c_layout, self.epi_tile, 1
                )
                sPe = smem.allocate_tensor(
                    self.c_dtype,
                    kpe_smem_layout.outer,
                    128,
                    swizzle=kpe_smem_layout.inner,
                )
            ab_pipeline_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread
            )
            num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
            ab_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_tma_producer
            )
            ab_pipeline = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
                num_stages=self.num_ab_stage,
                producer_group=ab_pipeline_producer_group,
                consumer_group=ab_pipeline_consumer_group,
                tx_count=self.num_tma_load_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            acc_pipeline_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread
            )
            num_acc_consumer_threads = len(self.epilog_warp_id) * (
                2 if use_2cta_instrs else 1
            )
            acc_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_acc_consumer_threads
            )
            acc_pipeline = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
                num_stages=self.num_acc_stage,
                producer_group=acc_pipeline_producer_group,
                consumer_group=acc_pipeline_consumer_group,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            tmem = utils.TmemAllocator(
                storage.tmem_holding_buf,
                barrier_for_retrieve=self.tmem_alloc_barrier,
                allocator_warp_id=self.epilog_warp_id[0],
                is_two_cta=use_2cta_instrs,
                two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
            )
            pipeline_init_arrive(
                cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True
            )
            sC = storage.output.get_tensor(
                c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
            )
            sA = storage.activation.get_tensor(
                a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
            )
            sB = storage.weight.get_tensor(
                b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
            )
            sSFA = storage.activation_scales.get_tensor(sfa_smem_layout_staged)
            sSFB = storage.weight_scales.get_tensor(sfb_smem_layout_staged)
            a_full_mcast_mask = None
            b_full_mcast_mask = None
            sfa_full_mcast_mask = None
            sfb_full_mcast_mask = None
            if cutlass.const_expr(
                self.is_a_mcast or self.is_b_mcast or use_2cta_instrs
            ):
                a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
                )
                b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
                )
                sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
                )
                sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_sfb_vmnk,
                    block_in_cluster_coord_sfb_vmnk,
                    mcast_mode=1,
                )
            gA_mkl = cute.local_tile(
                mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
            )
            gB_nkl = cute.local_tile(
                mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
            )
            gSFA_mkl = cute.local_tile(
                mSFA_mkl,
                cute.slice_(self.mma_tiler, (None, 0, None)),
                (None, None, None),
            )
            gSFB_nkl = cute.local_tile(
                mSFB_nkl,
                cute.slice_(self.mma_tiler_sfb, (0, None, None)),
                (None, None, None),
            )
            gC_mnl = cute.local_tile(
                mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
            )
            k_tile_cnt = cute.size(gA_mkl, mode=[3])
            thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
            thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
            tCgA = thr_mma.partition_A(gA_mkl)
            tCgB = thr_mma.partition_B(gB_nkl)
            tCgSFA = thr_mma.partition_A(gSFA_mkl)
            tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
            tCgC = thr_mma.partition_C(gC_mnl)
            a_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
            )
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a,
                block_in_cluster_coord_vmnk[2],
                a_cta_layout,
                cute.group_modes(sA, 0, 3),
                cute.group_modes(tCgA, 0, 3),
            )
            b_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
            )
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b,
                block_in_cluster_coord_vmnk[1],
                b_cta_layout,
                cute.group_modes(sB, 0, 3),
                cute.group_modes(tCgB, 0, 3),
            )
            sfa_cta_layout = a_cta_layout
            tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
                tma_atom_sfa,
                block_in_cluster_coord_vmnk[2],
                sfa_cta_layout,
                cute.group_modes(sSFA, 0, 3),
                cute.group_modes(tCgSFA, 0, 3),
            )
            tAsSFA = cute.filter_zeros(tAsSFA)
            tAgSFA = cute.filter_zeros(tAgSFA)
            sfb_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
            )
            tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
                tma_atom_sfb,
                block_in_cluster_coord_sfb_vmnk[1],
                sfb_cta_layout,
                cute.group_modes(sSFB, 0, 3),
                cute.group_modes(tCgSFB, 0, 3),
            )
            tBsSFB = cute.filter_zeros(tBsSFB)
            tBgSFB = cute.filter_zeros(tBgSFB)
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, self.num_acc_stage)
            )
            pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
            if warp_idx == self.tma_warp_id:
                tile_begin, tile_end, tile_step = self.tile_range()
                ab_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_ab_stage
                )
                for tile in cutlass.range(tile_begin, tile_end, tile_step):
                    cur_tile_coord = self.tile_coordinate(tile)
                    mma_tile_coord_mnl = (
                        cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                        cur_tile_coord[1],
                        cur_tile_coord[2],
                    )
                    tAgA_slice = tAgA[
                        None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2]
                    ]
                    tBgB_slice = tBgB[
                        None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2]
                    ]
                    tAgSFA_slice = tAgSFA[
                        None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2]
                    ]
                    slice_n = mma_tile_coord_mnl[1]
                    if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                        slice_n = mma_tile_coord_mnl[1] // 2
                    tBgSFB_slice = tBgSFB[None, slice_n, None, mma_tile_coord_mnl[2]]
                    ab_producer_state.reset_count()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        ab_pipeline.producer_acquire(
                            ab_producer_state, peek_ab_empty_status
                        )
                        cute.copy(
                            tma_atom_a,
                            tAgA_slice[None, ab_producer_state.count],
                            tAsA[None, ab_producer_state.index],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(
                                ab_producer_state
                            ),
                            mcast_mask=a_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB_slice[None, ab_producer_state.count],
                            tBsB[None, ab_producer_state.index],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(
                                ab_producer_state
                            ),
                            mcast_mask=b_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_sfa,
                            tAgSFA_slice[None, ab_producer_state.count],
                            tAsSFA[None, ab_producer_state.index],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(
                                ab_producer_state
                            ),
                            mcast_mask=sfa_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_sfb,
                            tBgSFB_slice[None, ab_producer_state.count],
                            tBsSFB[None, ab_producer_state.index],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(
                                ab_producer_state
                            ),
                            mcast_mask=sfb_full_mcast_mask,
                        )
                        ab_producer_state.advance()
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if ab_producer_state.count < k_tile_cnt:
                            peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                                ab_producer_state
                            )
                ab_pipeline.producer_tail(ab_producer_state)
            if warp_idx == self.mma_warp_id:
                tmem.wait_for_alloc()
                acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
                sfa_tmem_ptr = cute.recast_ptr(
                    acc_tmem_ptr + self.num_accumulator_tmem_cols, dtype=self.sf_dtype
                )
                tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                    tiled_mma,
                    self.mma_tiler,
                    self.sf_vec_size,
                    cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
                )
                tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
                sfb_tmem_ptr = cute.recast_ptr(
                    acc_tmem_ptr
                    + self.num_accumulator_tmem_cols
                    + self.num_sfa_tmem_cols,
                    dtype=self.sf_dtype,
                )
                tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                    tiled_mma,
                    self.mma_tiler,
                    self.sf_vec_size,
                    cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
                )
                tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
                (
                    tiled_copy_s2t_sfa,
                    tCsSFA_compact_s2t,
                    tCtSFA_compact_s2t,
                ) = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
                (
                    tiled_copy_s2t_sfb,
                    tCsSFB_compact_s2t,
                    tCtSFB_compact_s2t,
                ) = self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)
                tile_begin, tile_end, tile_step = self.tile_range()
                ab_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_ab_stage
                )
                acc_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_acc_stage
                )
                for tile in cutlass.range(tile_begin, tile_end, tile_step):
                    cur_tile_coord = self.tile_coordinate(tile)
                    mma_tile_coord_mnl = (
                        cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                        cur_tile_coord[1],
                        cur_tile_coord[2],
                    )
                    acc_stage_index = acc_producer_state.index
                    tCtAcc = tCtAcc_base[None, None, None, acc_stage_index]
                    ab_consumer_state.reset_count()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                        peek_ab_full_status = ab_pipeline.consumer_try_wait(
                            ab_consumer_state
                        )
                    if is_leader_cta:
                        acc_pipeline.producer_acquire(acc_producer_state)
                    tCtSFB_mma = tCtSFB
                    if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                        offset = (
                            cutlass.Int32(2)
                            if mma_tile_coord_mnl[1] % 2 == 1
                            else cutlass.Int32(0)
                        )
                        shifted_ptr = cute.recast_ptr(
                            acc_tmem_ptr
                            + self.num_accumulator_tmem_cols
                            + self.num_sfa_tmem_cols
                            + offset,
                            dtype=self.sf_dtype,
                        )
                        tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                    elif cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                        offset = cutlass.Int32(mma_tile_coord_mnl[1] % 2 * 2)
                        shifted_ptr = cute.recast_ptr(
                            acc_tmem_ptr
                            + self.num_accumulator_tmem_cols
                            + self.num_sfa_tmem_cols
                            + offset,
                            dtype=self.sf_dtype,
                        )
                        tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for k_tile in range(k_tile_cnt):
                        if is_leader_cta:
                            ab_pipeline.consumer_wait(
                                ab_consumer_state, peek_ab_full_status
                            )
                            s2t_stage_coord = (
                                None,
                                None,
                                None,
                                None,
                                ab_consumer_state.index,
                            )
                            tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[
                                s2t_stage_coord
                            ]
                            tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[
                                s2t_stage_coord
                            ]
                            cute.copy(
                                tiled_copy_s2t_sfa,
                                tCsSFA_compact_s2t_staged,
                                tCtSFA_compact_s2t,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfb,
                                tCsSFB_compact_s2t_staged,
                                tCtSFB_compact_s2t,
                            )
                            num_kblocks = cute.size(tCrA, mode=[2])
                            for kblock_idx in cutlass.range(
                                num_kblocks, unroll_full=True
                            ):
                                kblock_coord = (
                                    None,
                                    None,
                                    kblock_idx,
                                    ab_consumer_state.index,
                                )
                                sf_kblock_coord = (None, None, kblock_idx)
                                tiled_mma.set(
                                    tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator
                                )
                                tiled_mma.set(
                                    tcgen05.Field.SFB,
                                    tCtSFB_mma[sf_kblock_coord].iterator,
                                )
                                cute.gemm(
                                    tiled_mma,
                                    tCtAcc,
                                    tCrA[kblock_coord],
                                    tCrB[kblock_coord],
                                    tCtAcc,
                                )
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            ab_pipeline.consumer_release(ab_consumer_state)
                        ab_consumer_state.advance()
                        peek_ab_full_status = cutlass.Boolean(1)
                        if ab_consumer_state.count < k_tile_cnt:
                            if is_leader_cta:
                                peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                    ab_consumer_state
                                )
                    if is_leader_cta:
                        acc_pipeline.producer_commit(acc_producer_state)
                    acc_producer_state.advance()
                acc_pipeline.producer_tail(acc_producer_state)
            if warp_idx < self.mma_warp_id:
                tmem.allocate(self.num_tmem_alloc_cols)
                tmem.wait_for_alloc()
                acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
                epi_tidx = tidx
                (
                    tiled_copy_t2r,
                    tTR_tAcc_base,
                    tTR_rAcc,
                ) = self.epilog_tmem_copy_and_partition(
                    epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
                )
                if cutlass.const_expr(self.mode == 2):
                    gK_epi = cute.local_tile(mK_mnl, self.epi_tile, (None, None, 0))
                    _, bSG_gK = cpasync.tma_partition(
                        tma_atom_k,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sC, 0, 2),
                        cute.group_modes(gK_epi, 0, 2),
                    )
                    bSG_sPe, _ = cpasync.tma_partition(
                        tma_atom_k,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sPe, 0, 2),
                        cute.group_modes(gK_epi, 0, 2),
                    )
                tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
                tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                    tiled_copy_t2r, tTR_rC, epi_tidx, sC
                )
                (
                    tma_atom_c,
                    bSG_sC,
                    bSG_gC_partitioned,
                ) = self.epilog_gmem_copy_and_partition(
                    epi_tidx, tma_atom_c, tCgC, epi_tile, sC
                )
                tile_begin, tile_end, tile_step = self.tile_range()
                acc_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_acc_stage
                )
                c_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, 32 * len(self.epilog_warp_id)
                )
                c_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.num_c_stage, producer_group=c_producer_group
                )
                if cutlass.const_expr(True and self.mode == 1):
                    cosine_table = cute.make_rmem_tensor(32, cutlass.Float32)
                    sine_table = cute.make_rmem_tensor(32, cutlass.Float32)
                else:
                    cosine_table, sine_table = (None, None)
                cached_row = cutlass.Int32(-1)
                for tile in cutlass.range(tile_begin, tile_end, tile_step):
                    cur_tile_coord = self.tile_coordinate(tile)
                    mma_tile_coord_mnl = (
                        cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                        cur_tile_coord[1],
                        cur_tile_coord[2],
                    )
                    bSG_gC = bSG_gC_partitioned[None, None, None, *mma_tile_coord_mnl]
                    acc_stage_index = acc_consumer_state.index
                    tTR_tAcc = tTR_tAcc_base[
                        None, None, None, None, None, acc_stage_index
                    ]
                    if cutlass.const_expr(True and self.mode == 1):
                        for index in cutlass.range_constexpr(32):
                            linear = tidx + index * 128
                            r, pair = (linear // 32, linear % 32)
                            position = positions[cur_tile_coord[0] * 128 + r]
                            cosine_table[index] = cache[position, pair * 2]
                            sine_table[index] = cache[position, pair * 2 + 1]
                    acc_pipeline.consumer_wait(acc_consumer_state)
                    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
                    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                    num_prev_subtiles = (tile - tile_begin) // tile_step * subtile_cnt
                    for subtile_idx in cutlass.range(subtile_cnt, unroll_full=True):
                        real_subtile_idx = subtile_idx
                        tTR_tAcc_mn = tTR_tAcc[None, None, None, real_subtile_idx]
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
                        tRS_rC.store(
                            tiled_copy_r2s.retile(tTR_rAcc).load().to(self.c_dtype)
                        )
                        c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                        cute.copy(
                            tiled_copy_r2s, tRS_rC, tRS_sC[None, None, None, c_buffer]
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()
                        column = (
                            cur_tile_coord[1] * self.cta_tile_shape_mnk[1]
                            + real_subtile_idx * 64
                        )
                        row_start = cur_tile_coord[0] * 128
                        if cutlass.const_expr(self.mode == 1):
                            if 64 == 192 or column % 192 == 128:
                                self.rotate_shared(
                                    sC,
                                    c_buffer,
                                    k_pe,
                                    cache,
                                    positions,
                                    row_start,
                                    False,
                                    cosine_table,
                                    sine_table,
                                )
                                cute.arch.fence_proxy("async.shared", space="cta")
                                self.epilog_sync_barrier.arrive_and_wait()
                        if warp_idx == self.epilog_warp_id[0]:
                            if cutlass.const_expr(self.mode == 2):
                                if column % 256 < 128:
                                    cute.copy(
                                        tma_atom_k,
                                        bSG_sC[None, c_buffer],
                                        bSG_gK[
                                            None,
                                            cur_tile_coord[0],
                                            column // 256 * 3 + column % 256 // 64,
                                        ],
                                    )
                                else:
                                    cute.copy(
                                        tma_atom_c,
                                        bSG_sC[None, c_buffer],
                                        bSG_gC[None, real_subtile_idx],
                                    )
                            else:
                                cute.copy(
                                    tma_atom_c,
                                    bSG_sC[None, c_buffer],
                                    bSG_gC[None, real_subtile_idx],
                                )
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        self.epilog_sync_barrier.arrive_and_wait()
                        if cutlass.const_expr(self.mode == 2):
                            if column % 256 == 0:
                                if row_start != cached_row:
                                    if warp_idx == self.epilog_warp_id[0]:
                                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    self.rotate_shared(
                                        sPe,
                                        0,
                                        k_pe,
                                        cache,
                                        positions,
                                        row_start,
                                        True,
                                        cosine_table,
                                        sine_table,
                                    )
                                    cute.arch.fence_proxy("async.shared", space="cta")
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    cached_row = row_start
                                if warp_idx == self.epilog_warp_id[0]:
                                    cute.copy(
                                        tma_atom_k,
                                        bSG_sPe[None, 0],
                                        bSG_gK[
                                            None,
                                            cur_tile_coord[0],
                                            column // 256 * 3 + 2,
                                        ],
                                    )
                                    c_pipeline.producer_commit()
                                    c_pipeline.producer_acquire()
                                self.epilog_sync_barrier.arrive_and_wait()
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()
                tmem.relinquish_alloc_permit()
                self.epilog_sync_barrier.arrive_and_wait()
                tmem.free(acc_tmem_ptr)
                c_pipeline.producer_tail()

        def mainloop_s2t_copy_and_partition(
            self, sSF: cute.Tensor, tSF: cute.Tensor
        ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
            tCsSF_compact = cute.filter_zeros(sSF)
            tCtSF_compact = cute.filter_zeros(tSF)
            copy_atom_s2t = cute.make_copy_atom(
                tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype
            )
            tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
            thr_copy_s2t = tiled_copy_s2t.get_slice(0)
            tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
            tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
                tiled_copy_s2t, tCsSF_compact_s2t_
            )
            tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
            return (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t)

        def epilog_tmem_copy_and_partition(
            self,
            tidx: cutlass.Int32,
            tAcc: cute.Tensor,
            gC_mnl: cute.Tensor,
            epi_tile: cute.Tile,
            use_2cta_instrs: cutlass.Boolean | bool,
        ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                self.cta_tile_shape_mnk,
                self.c_layout,
                self.c_dtype,
                self.acc_dtype,
                epi_tile,
                use_2cta_instrs,
            )
            tAcc_epi = cute.flat_divide(tAcc[(None, None), 0, 0, None], epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r, tAcc_epi[None, None, 0, 0, 0]
            )
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
            gC_mnl_epi = cute.flat_divide(
                gC_mnl[(None, None), 0, 0, None, None, None], epi_tile
            )
            tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
            tTR_rAcc = cute.make_rmem_tensor(
                tTR_gC[None, None, None, 0, 0, 0, 0, 0].shape, self.acc_dtype
            )
            return (tiled_copy_t2r, tTR_tAcc, tTR_rAcc)

        def epilog_smem_copy_and_partition(
            self,
            tiled_copy_t2r: cute.TiledCopy,
            tTR_rC: cute.Tensor,
            tidx: cutlass.Int32,
            sC: cute.Tensor,
        ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
            copy_atom_r2s = sm100_utils.get_smem_store_op(
                self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
            )
            tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
            thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
            tRS_sC = thr_copy_r2s.partition_D(sC)
            tRS_rC = tiled_copy_r2s.retile(tTR_rC)
            return (tiled_copy_r2s, tRS_rC, tRS_sC)

        def epilog_gmem_copy_and_partition(
            self,
            tidx: cutlass.Int32,
            atom: cute.CopyAtom | cute.TiledCopy,
            gC_mnl: cute.Tensor,
            epi_tile: cute.Tile,
            sC: cute.Tensor,
        ) -> tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
            gC_epi = cute.flat_divide(
                gC_mnl[(None, None), 0, 0, None, None, None], epi_tile
            )
            tma_atom_c = atom
            sC_for_tma_partition = cute.group_modes(sC, 0, 2)
            gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
            bSG_sC, bSG_gC = cpasync.tma_partition(
                tma_atom_c,
                0,
                cute.make_layout(1),
                sC_for_tma_partition,
                gC_for_tma_partition,
            )
            return (tma_atom_c, bSG_sC, bSG_gC)

        @staticmethod
        def _compute_stages(
            tiled_mma: cute.TiledMma,
            mma_tiler_mnk: tuple[int, int, int],
            a_dtype: type[cutlass.Numeric],
            b_dtype: type[cutlass.Numeric],
            epi_tile: cute.Tile,
            c_dtype: type[cutlass.Numeric],
            c_layout: utils.LayoutEnum,
            sf_dtype: type[cutlass.Numeric],
            sf_vec_size: int,
            smem_capacity: int,
            occupancy: int,
        ) -> tuple[int, int, int]:
            num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2
            num_c_stage = 2
            a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
                tiled_mma, mma_tiler_mnk, a_dtype, 1
            )
            b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
                tiled_mma, mma_tiler_mnk, b_dtype, 1
            )
            sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
                tiled_mma, mma_tiler_mnk, sf_vec_size, 1
            )
            sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
                tiled_mma, mma_tiler_mnk, sf_vec_size, 1
            )
            c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
                c_dtype, c_layout, epi_tile, 1
            )
            ab_bytes_per_stage = (
                cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
                + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
                + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
                + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
            )
            mbar_helpers_bytes = 1024
            c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
            c_bytes = c_bytes_per_stage * num_c_stage
            num_ab_stage = (
                smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
            ) // ab_bytes_per_stage
            num_c_stage += (
                smem_capacity
                - occupancy * ab_bytes_per_stage * num_ab_stage
                - occupancy * (mbar_helpers_bytes + c_bytes)
            ) // (occupancy * c_bytes_per_stage)
            return (num_acc_stage, num_ab_stage, num_c_stage)

        @staticmethod
        def _compute_grid(
            c: cute.Tensor,
            cta_tile_shape_mnk: tuple[int, int, int],
            cluster_shape_mn: tuple[int, int],
            max_active_clusters: cutlass.Constexpr,
        ) -> tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]:
            c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
            gc = cute.zipped_divide(c, tiler=c_shape)
            num_ctas_mnl = gc[0, (None, None, None)].shape
            cluster_shape_mnl = (*cluster_shape_mn, 1)
            tile_sched_params = utils.PersistentTileSchedulerParams(
                num_ctas_mnl, cluster_shape_mnl
            )
            grid = utils.StaticPersistentTileScheduler.get_grid_shape(
                tile_sched_params, max_active_clusters
            )
            return (tile_sched_params, grid)

        @cute.jit
        def rotate_shared(
            self,
            shared,
            stage,
            k_pe,
            cache,
            positions,
            row_start,
            positional: cutlass.Constexpr,
            cosine_table,
            sine_table,
        ):
            tid, _, _ = cute.arch.thread_idx()
            offset = 64 - 64
            if cutlass.const_expr(positional):
                offset = 0
            for step in cutlass.range_constexpr(32):
                index = tid + step * 128
                row = index // 32
                pair = index % 32
                if cutlass.const_expr(True and (not positional)):
                    cosine, sine = (cosine_table[step], sine_table[step])
                else:
                    position = positions[row_start + row]
                    cosine = cache[position, pair * 2]
                    sine = cache[position, pair * 2 + 1]
                if cutlass.const_expr(positional):
                    even = k_pe[row_start + row, pair * 2].to(cutlass.Float32)
                    odd = k_pe[row_start + row, pair * 2 + 1].to(cutlass.Float32)
                else:
                    even = shared[row, pair * 2 + offset, stage].to(cutlass.Float32)
                    odd = shared[row, pair * 2 + 1 + offset, stage].to(cutlass.Float32)
                real = _add(_mul(even, cosine), -_mul(odd, sine))
                imag = _add(_mul(even, sine), _mul(odd, cosine))
                shared[row, pair * 2 + offset, stage] = real.to(cutlass.BFloat16)
                shared[row, pair * 2 + 1 + offset, stage] = imag.to(cutlass.BFloat16)

        @cute.jit
        def tile_range(self):
            _, _, cluster = cute.arch.block_idx()
            _, _, clusters = cute.arch.grid_dim()
            total = self.tiles_m * self.tiles_n
            if cutlass.const_expr(self.order == 1):
                begin = total * cluster // clusters
                end = total * (cluster + 1) // clusters
                step = cutlass.Int32(1)
            else:
                begin = cluster
                end = cutlass.Int32(total)
                step = clusters
            return (begin, end, step)

        @cute.jit
        def tile_coordinate(self, tile):
            cta, _, _ = cute.arch.block_idx()
            if cutlass.const_expr(self.order == 0):
                row = tile % self.tiles_m
                column = tile // self.tiles_m
            else:
                row = tile // self.tiles_n
                column = tile % self.tiles_n
            return (
                row * (2 if self.use_2cta_instrs else 1) + cta,
                column,
                cutlass.Int32(0),
            )

    class _GradientQuant:
        """TMA tiles fuse inverse RoPE or KV packing with both MXFP8 orientations."""

        def __init__(
            self, m, n, is_kv, block_m=128, stages=2, ctas_per_sm=3, limit_tiles=0
        ):
            self.m, self.n, self.is_kv = (m, n, is_kv)
            self.block_m, self.stages, self.ctas_per_sm = (block_m, stages, ctas_per_sm)
            self.threads = block_m

        @cute.jit
        def __call__(
            self,
            ga,
            gb,
            row,
            col,
            row_scale,
            col_scale,
            cache,
            positions,
            stream: cuda.CUstream,
        ):
            input_layout = sm100_utils.make_smem_layout_epi(
                cutlass.BFloat16,
                utils.LayoutEnum.ROW_MAJOR,
                (self.block_m, 64),
                self.stages,
            )
            row_layout = sm100_utils.make_smem_layout_epi(
                cutlass.Float8E4M3FN, utils.LayoutEnum.ROW_MAJOR, (self.block_m, 64), 2
            )
            col_layout = sm100_utils.make_smem_layout_epi(
                cutlass.Float8E4M3FN, utils.LayoutEnum.ROW_MAJOR, (64, self.block_m), 2
            )
            input_tile = cute.slice_(input_layout, (None, None, 0))
            row_tile = cute.slice_(row_layout, (None, None, 0))
            col_tile = cute.slice_(col_layout, (None, None, 0))
            load_a, ta = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(), ga, input_tile, (self.block_m, 64)
            )
            load_b, tb = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(), gb, input_tile, (self.block_m, 64)
            )
            store_r, tr = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(), row, row_tile, (self.block_m, 64)
            )
            store_c, tc = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(), col, col_tile, (64, self.block_m)
            )
            grid = min(self.m // self.block_m * (self.n // 64), self.ctas_per_sm * 152)
            if not self.is_kv and grid % 3 == 0:
                grid -= 1
            self.kernel(
                load_a,
                ta,
                load_b,
                tb,
                store_r,
                tr,
                store_c,
                tc,
                row_scale,
                col_scale,
                cache,
                positions,
                input_layout,
                row_layout,
                col_layout,
            ).launch(grid=(grid, 1, 1), block=(self.threads, 1, 1), stream=stream)

        @cute.jit
        def issue(self, load_a, load_b, sa, ta, tb, ready, slot, tile):
            row = tile // (self.n // 64)
            column = tile % (self.n // 64)
            with arch.elect_one():
                arch.mbarrier_arrive_and_expect_tx(ready + slot, self.block_m * 64 * 2)
            if cutlass.const_expr(self.is_kv):
                segment = column % 4
                if segment < 2:
                    cute.copy(
                        load_a,
                        ta[None, row, column // 4 * 3 + segment],
                        sa[None, slot],
                        tma_bar_ptr=ready + slot,
                    )
                else:
                    cute.copy(
                        load_b,
                        tb[None, row, column // 4 * 2 + segment - 2],
                        sa[None, slot],
                        tma_bar_ptr=ready + slot,
                    )
            else:
                cute.copy(
                    load_a,
                    ta[None, row, column],
                    sa[None, slot],
                    tma_bar_ptr=ready + slot,
                )

        @cute.kernel
        def kernel(
            self,
            load_a,
            ga,
            load_b,
            gb,
            store_r,
            gr,
            store_c,
            gc,
            row_scale,
            col_scale,
            cache,
            positions,
            input_layout: cute.ComposedLayout,
            row_layout: cute.ComposedLayout,
            col_layout: cute.ComposedLayout,
        ):
            tid, _, _ = arch.thread_idx()
            first, _, _ = arch.block_idx()
            step, _, _ = arch.grid_dim()
            total = self.m // self.block_m * (self.n // 64)
            allocator = utils.SmemAllocator()
            ready = allocator.allocate_array(Int64, self.stages)
            source = allocator.allocate_tensor(
                cutlass.BFloat16, input_layout.outer, 1024, swizzle=input_layout.inner
            )
            row = allocator.allocate_tensor(
                cutlass.Float8E4M3FN, row_layout.outer, 1024, swizzle=row_layout.inner
            )
            col = allocator.allocate_tensor(
                cutlass.Float8E4M3FN, col_layout.outer, 1024, swizzle=col_layout.inner
            )
            source_words = _word_view(source, input_layout.inner)
            row_words = _word_view(row, row_layout.inner)
            col_words = _word_view(col, col_layout.inner)
            sa, ta = cpasync.tma_partition(
                load_a,
                0,
                cute.make_layout(1),
                cute.group_modes(source, 0, 2),
                cute.group_modes(
                    cute.local_tile(ga, (self.block_m, 64), (None, None)), 0, 2
                ),
            )
            _, tb = cpasync.tma_partition(
                load_b,
                0,
                cute.make_layout(1),
                cute.group_modes(source, 0, 2),
                cute.group_modes(
                    cute.local_tile(gb, (self.block_m, 64), (None, None)), 0, 2
                ),
            )
            sr, tr = cpasync.tma_partition(
                store_r,
                0,
                cute.make_layout(1),
                cute.group_modes(row, 0, 2),
                cute.group_modes(
                    cute.local_tile(gr, (self.block_m, 64), (None, None)), 0, 2
                ),
            )
            sc, tc = cpasync.tma_partition(
                store_c,
                0,
                cute.make_layout(1),
                cute.group_modes(col, 0, 2),
                cute.group_modes(
                    cute.local_tile(gc, (64, self.block_m), (None, None)), 0, 2
                ),
            )
            if tid == 0:
                for slot in cutlass.range_constexpr(self.stages):
                    arch.mbarrier_init(ready + slot, 1)
            arch.mbarrier_init_fence()
            arch.barrier()
            if tid < 32:
                for slot in cutlass.range_constexpr(self.stages):
                    tile = first + slot * step
                    if tile < total:
                        self.issue(load_a, load_b, sa, ta, tb, ready, slot, tile)
            count = Int32(0)
            packed = cute.make_rmem_tensor(32, Uint32)
            row_values = cute.make_rmem_tensor(16, Uint32)
            column_even = cute.make_rmem_tensor(8, Uint32)
            column_odd = cute.make_rmem_tensor(8, Uint32)
            word_copy = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), Uint32, num_bits_per_copy=128
            )
            if cutlass.const_expr(not self.is_kv):
                cosine_table = cute.make_rmem_tensor(32, Float32)
                sine_table = cute.make_rmem_tensor(32, Float32)
            for tile in cutlass.range(first, total, step):
                slot = count % self.stages
                out_slot = count % 2
                row_start = tile // (self.n // 64) * self.block_m
                column_start = tile % (self.n // 64) * 64
                if cutlass.const_expr(not self.is_kv):
                    if column_start // 64 % 3 == 2:
                        for index in cutlass.range_constexpr(
                            self.block_m * 32 // self.threads
                        ):
                            linear = tid + index * self.threads
                            r, pair = (linear // 32, linear % 32)
                            position = positions[row_start + r]
                            cosine_table[index] = cache[position, pair * 2]
                            sine_table[index] = cache[position, pair * 2 + 1]
                arch.mbarrier_wait(ready + slot, count // self.stages % 2)
                if cutlass.const_expr(not self.is_kv):
                    if column_start // 64 % 3 == 2:
                        for index in cutlass.range_constexpr(
                            self.block_m * 32 // self.threads
                        ):
                            linear = tid + index * self.threads
                            r, pair = (linear // 32, linear % 32)
                            cosine, sine = (cosine_table[index], sine_table[index])
                            word = source_words[r, pair, slot]
                            even, odd = (_low(word), _high(word))
                            source_words[r, pair, slot] = _pack(
                                _add(_mul(even, cosine), _mul(odd, sine)),
                                _add(_mul(odd, cosine), -_mul(even, sine)),
                            )
                        arch.barrier()
                if tid == 0:
                    arch.cp_async_bulk_wait_group(1, read=True)
                arch.barrier()
                row_source = cute.make_tensor(
                    source_words.iterator
                    + cute.assume(source_words.layout((tid, 0, slot)), divby=4),
                    cute.make_layout(32),
                )
                cute.copy(word_copy, row_source, packed)
                for group in cutlass.range_constexpr(2):
                    maximum = packed[group * 16]
                    for i in cutlass.range_constexpr(1, 16):
                        maximum = _absmax(maximum, packed[group * 16 + i])
                    magnitude = maximum & Uint32(2147450879)
                    reciprocal, exponent = _scale(
                        _maximum(_low(magnitude), _high(magnitude))
                    )
                    reciprocal = reciprocal | reciprocal << 16
                    for i in cutlass.range_constexpr(8):
                        row_values[group * 8 + i] = _quant4(
                            packed[group * 16 + i * 2],
                            packed[group * 16 + i * 2 + 1],
                            reciprocal,
                        )
                    row_scale[
                        _blocked(
                            row_start + tid, column_start // 32 + group, self.n // 32
                        )
                    ] = exponent.to(Uint8)
                row_destination = cute.make_tensor(
                    row_words.iterator
                    + cute.assume(row_words.layout((tid, 0, out_slot)), divby=4),
                    cute.make_layout(16),
                )
                cute.copy(word_copy, row_values, row_destination)
                group, pair = (tid // 32, tid % 32)
                for i in cutlass.range_constexpr(32):
                    packed[i] = source_words[group * 32 + i, pair, slot]
                maximum = packed[0]
                for i in cutlass.range_constexpr(1, 32):
                    maximum = _absmax(maximum, packed[i])
                magnitude = maximum & Uint32(2147450879)
                inverse_lo, exponent_lo = _scale(_low(magnitude))
                inverse_hi, exponent_hi = _scale(_high(magnitude))
                reciprocal = inverse_lo | inverse_hi << 16
                for i in cutlass.range_constexpr(8):
                    even_word, odd_word = _quant_col4(
                        packed[i * 4],
                        packed[i * 4 + 1],
                        packed[i * 4 + 2],
                        packed[i * 4 + 3],
                        reciprocal,
                    )
                    column_even[i], column_odd[i] = (even_word, odd_word)
                column_even_destination = cute.make_tensor(
                    col_words.iterator
                    + cute.assume(
                        col_words.layout((pair * 2, group * 8, out_slot)), divby=4
                    ),
                    cute.make_layout(8),
                )
                column_odd_destination = cute.make_tensor(
                    col_words.iterator
                    + cute.assume(
                        col_words.layout((pair * 2 + 1, group * 8, out_slot)), divby=4
                    ),
                    cute.make_layout(8),
                )
                cute.copy(word_copy, column_even, column_even_destination)
                cute.copy(word_copy, column_odd, column_odd_destination)
                col_scale[
                    _blocked(
                        column_start + pair * 2, row_start // 32 + group, self.m // 32
                    )
                ] = exponent_lo.to(Uint8)
                col_scale[
                    _blocked(
                        column_start + pair * 2 + 1,
                        row_start // 32 + group,
                        self.m // 32,
                    )
                ] = exponent_hi.to(Uint8)
                arch.fence_proxy("async.shared", space="cta")
                arch.barrier()
                if tid == 0:
                    cute.copy(
                        store_r,
                        sr[None, out_slot],
                        tr[None, row_start // self.block_m, column_start // 64],
                    )
                    cute.copy(
                        store_c,
                        sc[None, out_slot],
                        tc[None, column_start // 64, row_start // self.block_m],
                    )
                    arch.cp_async_bulk_commit_group()
                if tid < 32:
                    next_tile = tile + self.stages * step
                    if next_tile < total:
                        self.issue(load_a, load_b, sa, ta, tb, ready, slot, next_tile)
                count += 1
            if tid == 0:
                arch.cp_async_bulk_wait_group(0, read=True)

    class _PositionGradient:
        """Match the native head-reduction tree before BF16 inverse RoPE."""

        @cute.jit
        def __call__(self, gradient, cache, positions, out, stream: cuda.CUstream):
            self.kernel(gradient, cache, positions, out).launch(
                grid=(1024, 1, 1), block=(128, 1, 1), stream=stream
            )

        @cute.kernel
        def kernel(self, gradient, cache, positions, out):
            tid, _, _ = arch.thread_idx()
            block, _, _ = arch.block_idx()
            row = block * 4 + tid // 32
            pair = tid % 32
            even = cute.make_rmem_tensor(64, Float32)
            odd = cute.make_rmem_tensor(64, Float32)
            total_even, total_odd = (Float32(0), Float32(0))
            for group in cutlass.range_constexpr(2):
                for head in cutlass.range_constexpr(64):
                    word = gradient[row, (group * 64 + head) * 96 + 64 + pair]
                    even[head], odd[head] = (_low(word), _high(word))
                for level in cutlass.range_constexpr(6):
                    distance = (16, 32, 2, 1, 8, 4)[level]
                    reduced_mask = (16, 48, 50, 51, 59, 63)[level]
                    for head in cutlass.range_constexpr(64):
                        if cutlass.const_expr(head & reduced_mask == 0):
                            even[head] = _add(even[head], even[head + distance])
                            odd[head] = _add(odd[head], odd[head + distance])
                total_even = _add(total_even, even[0])
                total_odd = _add(total_odd, odd[0])
            rounded = _pack(total_even, total_odd)
            grad_even, grad_odd = (_low(rounded), _high(rounded))
            position = positions[row]
            cosine, sine = (cache[position, pair * 2], cache[position, pair * 2 + 1])
            out[row, pair] = _pack(
                _add(_mul(grad_even, cosine), _mul(grad_odd, sine)),
                _add(_mul(grad_odd, cosine), -_mul(grad_even, sine)),
            )


def _norm_scale_numel(rows: int, columns: int) -> int:
    return (rows + 127) // 128 * ((columns + 127) // 128) * 512


def _gradient_token_matrix(value):
    """[1, T, H, W] gradient -> TMA-compatible [T, H*W] view (copy if needed)."""
    _, tokens, heads, width = value.shape
    if not (
        value.stride(3) == 1
        and value.stride(2) == width
        and (value.stride(1) % 8 == 0)
        and (value.data_ptr() % 16 == 0)
    ):
        value = value.contiguous()
    return (
        torch.as_strided(
            value, (tokens, heads * width), (value.stride(1), 1), value.storage_offset()
        ),
        value,
    )


def _gradient_side_stream(device):
    index = torch.device(device).index
    if index not in _GRADIENT_SIDE_STREAMS:
        _GRADIENT_SIDE_STREAMS[index] = torch.cuda.Stream(device=device)
    return _GRADIENT_SIDE_STREAMS[index]


def _gradient_check(cache, positions, tokens, block_m):
    """Validate operands; returns positions as a contiguous [1, T] int32/int64 tensor."""
    if positions.ndim == 1:
        positions = positions.unsqueeze(0)
    positions = positions.contiguous()
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("positions must be int32 or int64")
    if tokens % block_m:
        raise ValueError(f"token count must be a multiple of {block_m}")
    if (
        cache.dtype != torch.float32
        or cache.shape[-2:] != (32, 2)
        or (not cache.is_contiguous())
    ):
        raise ValueError("rope_cache_real must be a contiguous FP32 [S, 32, 2] tensor")
    if positions.shape != (1, tokens):
        raise ValueError("positions must have shape [1, T] or [T]")
    return positions


def _norm_rms_norm_quant(x, weight, eps, *, colwise=True):
    _require_cutedsl()
    n = x.shape[-1]
    m = x.numel() // n
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("rms_norm_quant requires BF16 input and weight")
    if (m, n) not in ((4096, 1536), (4096, 512)) or weight.shape != (n,):
        raise ValueError("MLA RMSNorm requires 4096 tokens and width 1536 or 512")
    if (
        x.ndim < 2
        or x.stride(-1) != 1
        or any(
            x.stride(i) != x.stride(i + 1) * x.shape[i + 1] for i in range(x.ndim - 2)
        )
        or not weight.is_contiguous()
        or x.data_ptr() % 16
        or x.stride(-2) % 8
    ):
        raise ValueError("MLA RMSNorm requires aligned rows and contiguous weight")
    rstd = torch.empty((*x.shape[:-1], 1), device=x.device, dtype=torch.float32)
    row = torch.empty((m, n), device=x.device, dtype=torch.float8_e4m3fn)
    col = (
        torch.empty((n, m), device=x.device, dtype=torch.float8_e4m3fn).t()
        if colwise
        else row.new_empty(0)
    )
    row_scale = torch.empty(m * n // 32, device=x.device, dtype=torch.float8_e8m0fnu)
    col_scale = torch.empty_like(row_scale) if colwise else row_scale.new_empty(0)
    pointers = tuple(
        (
            make_ptr(dtype, value.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
            for value, dtype in zip(
                (x, weight, rstd, row, col, row_scale, col_scale),
                (Uint32, Uint32, Float32, Uint32, Uint32, Uint8, Uint8),
            )
        )
    )
    key = (x.device.index, m, n, x.stride(-2), eps, colwise)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if key not in _NORM_COMPILED:
        _NORM_COMPILED[key] = cute.compile(
            _NormQuant(*key[1:], 1024, 1, False), pointers, stream
        )
    _NORM_COMPILED[key](pointers, stream)
    return (rstd, row, col, row_scale, col_scale)


def _up_launch(
    row, row_scale, weight, weight_scale, output, k, k_pe, cache, positions, mode
):
    _require_cutedsl()
    tile = (256, 192)
    cluster = (2, 1)
    values = (
        row.unsqueeze(-1),
        weight.unsqueeze(-1),
        row_scale,
        weight_scale,
        output.view(row.shape[0], -1, 1),
        k.view(row.shape[0], -1, 1),
        k_pe.view(row.shape[0], 64),
        cache.view(-1, 64),
        positions.view(-1),
    )
    dtypes = (
        cutlass.Float8E4M3FN,
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        cutlass.Float8E8M0FNU,
        cutlass.BFloat16,
        cutlass.BFloat16,
        cutlass.BFloat16,
        cutlass.Float32,
        cutlass.Int32 if positions.dtype == torch.int32 else cutlass.Int64,
    )
    tensors = []
    for value, dtype in zip(values, dtypes):
        # The autograd Function owns differentiation; DLPack needs a detached view.
        value = value.detach()
        tensor = from_dlpack(
            value.view(torch.uint8) if value.element_size() == 1 else value,
            assumed_align=16,
        )
        tensor.element_type = dtype
        tensors.append(tensor)
    key = (
        row.device.index,
        mode,
        tuple(((tuple(v.shape), v.stride(), v.dtype) for v in values)),
        tile,
        cluster,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if key not in _UP_COMPILED:
        kernel = _MLAUpGemm(
            32, tile, cluster, mode=mode, block_k=256, order=0 if mode == 1 else 1
        )
        _UP_COMPILED[key] = cute.compile(
            kernel, *tensors, 152 // (cluster[0] * cluster[1]), stream
        )
    _UP_COMPILED[key](*tensors, stream)


def _up_q_up_rope(row, row_scale, weight, weight_scale, cache, positions):
    output = torch.empty(
        (1, row.shape[0], weight.shape[0]), device=row.device, dtype=torch.bfloat16
    )
    _up_launch(
        row,
        row_scale,
        weight,
        weight_scale,
        output,
        output,
        output[..., :64],
        cache,
        positions,
        1,
    )
    return output


def _up_kv_up_rope(row, row_scale, weight, weight_scale, k_pe, cache, positions):
    kv = torch.empty(
        (1, row.shape[0], weight.shape[0] // 256, 256),
        device=row.device,
        dtype=torch.bfloat16,
    )
    k = torch.empty(
        (1, row.shape[0], weight.shape[0] // 256, 192),
        device=row.device,
        dtype=torch.bfloat16,
    )
    _up_launch(row, row_scale, weight, weight_scale, kv, k, k_pe, cache, positions, 2)
    return (kv, k)


def _gradient_launch(ga, gb, cache, positions, *, is_kv=False):
    _require_cutedsl()
    m = ga.shape[0]
    n = (ga.shape[1] // 192) * (256 if is_kv else 192)
    row = torch.empty((m, n), device=ga.device, dtype=torch.float8_e4m3fn)
    col = torch.empty((n, m), device=ga.device, dtype=torch.float8_e4m3fn)
    row_scale = torch.empty(m * n // 32, device=ga.device, dtype=torch.float8_e8m0fnu)
    col_scale = torch.empty_like(row_scale)
    values = (
        ga,
        gb,
        row,
        col,
        row_scale,
        col_scale,
        cache.view(-1, 64),
        positions.view(-1),
    )
    tensors = []
    for value in values:
        if value.element_size() == 1:
            tensor = from_dlpack(value.view(torch.uint8), assumed_align=16)
            tensor.element_type = (
                cutlass.Float8E4M3FN if value.dtype == torch.float8_e4m3fn else Uint8
            )
        else:
            tensor = from_dlpack(value, assumed_align=16)
        tensors.append(tensor)
    key = (
        ga.device.index,
        m,
        n,
        is_kv,
        tuple(((x.shape, x.stride(), x.dtype) for x in values)),
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if key not in _GRADIENT_COMPILED:
        _GRADIENT_COMPILED[key] = cute.compile(
            _GradientQuant(m, n, is_kv), *tensors, stream
        )
    _GRADIENT_COMPILED[key](*tensors, stream)
    return (row, col.t(), row_scale, col_scale)


def _position_gradient(grad_k, cache, positions, output):
    values = (
        grad_k.reshape(4096, -1).view(torch.int32),
        cache.view(-1, 64),
        positions.view(-1),
        output.view(torch.int32).view(4096, 32),
    )
    tensors = []
    for value in values:
        tensor = from_dlpack(value, assumed_align=16)
        if value.dtype == torch.int32 and value is not values[2]:
            tensor.element_type = Uint32
        tensors.append(tensor)
    key = (
        "position",
        grad_k.device.index,
        tuple(((v.shape, v.stride(), v.dtype) for v in values)),
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if key not in _GRADIENT_COMPILED:
        _GRADIENT_COMPILED[key] = cute.compile(_PositionGradient(), *tensors, stream)
    _GRADIENT_COMPILED[key](*tensors, stream)
    return output


_NORM_COMPILED = {}
_UP_COMPILED = {}
_GRADIENT_COMPILED = {}
_GRADIENT_SIDE_STREAMS = {}


def _require_cutedsl():
    if _CUTEDSL_IMPORT_ERROR is not None:
        raise RuntimeError(
            "MLA fusion requires CuTeDSL 4.8 or newer"
        ) from _CUTEDSL_IMPORT_ERROR


def _gradient_q_rope_backward_quant(grad_q, cache, positions):
    if grad_q.shape != (1, 4096, 128, 192) or grad_q.dtype != torch.bfloat16:
        raise ValueError("Q gradient must be BF16 [1, 4096, 128, 192]")
    positions = _gradient_check(cache, positions, 4096, 128)
    matrix, _ = _gradient_token_matrix(grad_q)
    return _gradient_launch(matrix, matrix, cache, positions)


def _gradient_kv_backward_quant(grad_k, grad_v, cache, positions):
    if grad_k.shape != (1, 4096, 128, 192) or grad_v.shape != (1, 4096, 128, 128):
        raise ValueError(
            "KV gradients require 4096 tokens, 128 heads, and widths 192/128"
        )
    if grad_k.dtype != torch.bfloat16 or grad_v.dtype != torch.bfloat16:
        raise ValueError("KV gradients must be BF16")
    positions = _gradient_check(cache, positions, 4096, 128)
    ga, grad_k = _gradient_token_matrix(grad_k)
    gb, _ = _gradient_token_matrix(grad_v)
    grad_position = grad_k.new_empty((1, 4096, 64))
    current = torch.cuda.current_stream(grad_k.device)
    side = _gradient_side_stream(grad_k.device)
    side.wait_stream(current)
    outputs = _gradient_launch(ga, gb, cache, positions, is_kv=True)
    with torch.cuda.stream(side):
        _position_gradient(grad_k, cache, positions, grad_position)
    current.wait_stream(side)
    return (*outputs, grad_position)


ACCEPTED = False
_STAGES = ("norm_quant", "q_up_rope", "kv_up_rope", "q_bwd", "kv_bwd")
logger = logging.getLogger(__name__)


def _quantize(value_TD, *, colwise=True):
    row_TD, column_TD, row_scale, column_scale = mxfp8_quantize_cuda(
        value_TD.contiguous(), rowwise=True, colwise=colwise, scaling_mode="rceil"
    )
    row_scale = triton_mx_block_rearrange(row_scale).flatten()
    if colwise:
        column_scale = triton_mx_block_rearrange(column_scale).flatten()
    else:
        column_TD = value_TD.new_empty((0,), dtype=torch.float8_e4m3fn)
        column_scale = value_TD.new_empty((0,), dtype=torch.float8_e8m0fnu)
    return row_TD, column_TD, row_scale, column_scale


def _scaled_mm(left, right, left_scale, right_scale, *, dtype=torch.bfloat16):
    return F.scaled_mm(
        left,
        right,
        scale_a=left_scale.flatten(),
        scale_recipe_a=F.ScalingType.BlockWise1x32,
        scale_b=right_scale.flatten(),
        scale_recipe_b=F.ScalingType.BlockWise1x32,
        swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
        output_dtype=dtype,
    )


@torch.library.custom_op(
    "torchtitan::dsv3_mla_norm_quant", mutates_args=(), device_types="cuda"
)
def norm_quant_op(
    latent_TD: torch.Tensor, weight_D: torch.Tensor, eps: float, colwise: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _norm_rms_norm_quant(latent_TD, weight_D, eps, colwise=colwise)


def _quantized_metadata(reference, tokens, width, *, colwise=True):
    return (
        reference.new_empty((tokens, width), dtype=torch.float8_e4m3fn),
        torch.empty_strided(
            (tokens, width) if colwise else (0,),
            (1, tokens) if colwise else (1,),
            dtype=torch.float8_e4m3fn,
            device=reference.device,
        ),
        reference.new_empty(
            (_norm_scale_numel(tokens, width),), dtype=torch.float8_e8m0fnu
        ),
        reference.new_empty(
            (_norm_scale_numel(width, tokens) if colwise else 0,),
            dtype=torch.float8_e8m0fnu,
        ),
    )


@norm_quant_op.register_fake
def _norm_quant_fake(latent_TD, weight_D, eps, colwise):
    tokens, width = latent_TD.numel() // latent_TD.shape[-1], latent_TD.shape[-1]
    return (
        latent_TD.new_empty((*latent_TD.shape[:-1], 1), dtype=torch.float32),
        *_quantized_metadata(latent_TD, tokens, width, colwise=colwise),
    )


@torch.library.custom_op(
    "torchtitan::dsv3_mla_q_up_rope", mutates_args=(), device_types="cuda"
)
def q_up_rope_op(
    row_TQ: torch.Tensor,
    row_scale: torch.Tensor,
    weight_NQ: torch.Tensor,
    weight_scale: torch.Tensor,
    cache_SR2: torch.Tensor,
    positions_BT: torch.Tensor,
) -> torch.Tensor:
    return _up_q_up_rope(
        row_TQ, row_scale, weight_NQ, weight_scale, cache_SR2, positions_BT
    )


@q_up_rope_op.register_fake
def _q_up_rope_fake(
    row_TQ, row_scale, weight_NQ, weight_scale, cache_SR2, positions_BT
):
    return row_TQ.new_empty(
        (1, row_TQ.shape[0], weight_NQ.shape[0]), dtype=torch.bfloat16
    )


@torch.library.custom_op(
    "torchtitan::dsv3_mla_kv_up_rope", mutates_args=(), device_types="cuda"
)
def kv_up_rope_op(
    row_TK: torch.Tensor,
    row_scale: torch.Tensor,
    weight_NK: torch.Tensor,
    weight_scale: torch.Tensor,
    k_pe_BTR: torch.Tensor,
    cache_SR2: torch.Tensor,
    positions_BT: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _up_kv_up_rope(
        row_TK, row_scale, weight_NK, weight_scale, k_pe_BTR, cache_SR2, positions_BT
    )


@kv_up_rope_op.register_fake
def _kv_up_rope_fake(
    row_TK, row_scale, weight_NK, weight_scale, k_pe_BTR, cache_SR2, positions_BT
):
    tokens, heads = row_TK.shape[0], weight_NK.shape[0] // 256
    return (
        row_TK.new_empty((1, tokens, heads, 256), dtype=torch.bfloat16),
        row_TK.new_empty((1, tokens, heads, 192), dtype=torch.bfloat16),
    )


@torch.library.custom_op(
    "torchtitan::dsv3_mla_q_backward_quant", mutates_args=(), device_types="cuda"
)
def q_backward_quant_op(
    grad_q_BTHD: torch.Tensor, cache_SR2: torch.Tensor, positions_BT: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _gradient_q_rope_backward_quant(grad_q_BTHD, cache_SR2, positions_BT)


@q_backward_quant_op.register_fake
def _q_backward_quant_fake(grad_q_BTHD, cache_SR2, positions_BT):
    return _quantized_metadata(
        grad_q_BTHD, grad_q_BTHD.shape[1], grad_q_BTHD.shape[2] * 192
    )


@torch.library.custom_op(
    "torchtitan::dsv3_mla_kv_backward_quant", mutates_args=(), device_types="cuda"
)
def kv_backward_quant_op(
    grad_k_BTHD: torch.Tensor,
    grad_v_BTHV: torch.Tensor,
    cache_SR2: torch.Tensor,
    positions_BT: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _gradient_kv_backward_quant(
        grad_k_BTHD, grad_v_BTHV, cache_SR2, positions_BT
    )


@kv_backward_quant_op.register_fake
def _kv_backward_quant_fake(grad_k_BTHD, grad_v_BTHV, cache_SR2, positions_BT):
    return (
        *_quantized_metadata(
            grad_k_BTHD, grad_k_BTHD.shape[1], grad_k_BTHD.shape[2] * 256
        ),
        grad_k_BTHD.new_empty((1, grad_k_BTHD.shape[1], 64)),
    )


@torch.library.custom_op(
    "torchtitan::dsv3_mla_norm_backward",
    mutates_args=(),
    device_types="cuda",
    schema="(Tensor grad_normalized_BTD, Tensor latent_BTD, Tensor rstd_BT1, "
    "Tensor weight_D, bool input_grad, bool weight_grad) -> (Tensor?, Tensor?)",
)
def norm_backward_op(
    grad_normalized_BTD: torch.Tensor,
    latent_BTD: torch.Tensor,
    rstd_BT1: torch.Tensor,
    weight_D: torch.Tensor,
    input_grad: bool,
    weight_grad: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    # Inductor decomposes this ATen operation and changes reduction order.
    # Keep native RMSNorm backward opaque so compiled fusion retains the
    # same rounding as eager execution.
    return torch.ops.aten._fused_rms_norm_backward(
        grad_normalized_BTD,
        latent_BTD,
        [latent_BTD.shape[-1]],
        rstd_BT1,
        weight_D,
        [input_grad, weight_grad],
    )


@norm_backward_op.register_fake
def _norm_backward_fake(
    grad_normalized_BTD, latent_BTD, rstd_BT1, weight_D, input_grad, weight_grad
):
    return (
        latent_BTD.new_empty(latent_BTD.shape) if input_grad else None,
        weight_D.new_empty(weight_D.shape) if weight_grad else None,
    )


def _weight_operands(weight):
    if isinstance(weight, _UnshardedFSDPTensor):
        return weight.operands
    storage = (
        weight._tensor
        if isinstance(weight, _LinearShardedTensorWithMXFP8Compute)
        else weight
    )
    with torch.no_grad():
        return _quantize_mxfp8_weight(storage)


def _save_chain(
    ctx,
    latent_BTD,
    rstd_BT1,
    norm_weight_D,
    column_TD,
    column_scale,
    weight,
    operands,
    cache_SR2,
    positions_BT,
    stages,
    accumulate,
):
    # FSDP may free and refill operand storage between forward and backward.
    # Retain its wrapper, and re-read that lifetime's operands in backward.
    if isinstance(weight, _UnshardedFSDPTensor):
        weight_data = operands.weight_qdata_dgrad_NK.new_empty((0,))
        weight_scale = operands.weight_scale_dgrad_swizzled.new_empty((0,))
    else:
        weight_data = operands.weight_qdata_dgrad_NK
        weight_scale = operands.weight_scale_dgrad_swizzled
    ctx.save_for_backward(
        latent_BTD,
        rstd_BT1,
        norm_weight_D,
        column_TD,
        column_scale,
        weight,
        weight_data,
        weight_scale,
        cache_SR2,
        positions_BT,
    )
    ctx.weight_param = weight if accumulate else None
    ctx.wgrad_dtype = weight.grad_dtype or weight.dtype
    ctx.stages = stages


def _norm_forward(latent_BTD, weight_D, eps, colwise, stages):
    if "norm_quant" in stages:
        return norm_quant_op(latent_BTD, weight_D, eps, colwise)
    normalized_BTD, rstd_BT1 = torch.ops.aten._fused_rms_norm(
        latent_BTD.contiguous(), [latent_BTD.shape[-1]], weight_D, eps
    )
    return rstd_BT1, *_quantize(normalized_BTD.flatten(0, 1), colwise=colwise)


def _projection_backward(ctx, gradient, saved):
    (
        latent_BTD,
        rstd_BT1,
        norm_weight_D,
        column_TD,
        column_scale,
        weight,
        weight_data,
        weight_scale,
        _,
        _,
    ) = saved
    if isinstance(weight, _UnshardedFSDPTensor):
        weight_data = weight.operands.weight_qdata_dgrad_NK
        weight_scale = weight.operands.weight_scale_dgrad_swizzled
    grad_row_TN, grad_column_TN, grad_row_scale, grad_column_scale = gradient
    grad_latent_BTD = grad_norm_D = grad_weight = None
    if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
        grad_normalized_BTD = _scaled_mm(
            grad_row_TN, weight_data, grad_row_scale, weight_scale
        ).view(latent_BTD.shape)
        grad_latent_BTD, grad_norm_D = norm_backward_op(
            grad_normalized_BTD,
            latent_BTD,
            rstd_BT1,
            norm_weight_D,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
        )
    if ctx.needs_input_grad[2]:
        running_grad = None if ctx.weight_param is None else ctx.weight_param.grad
        if running_grad is None:
            grad_weight = _scaled_mm(
                grad_column_TN.t(),
                column_TD,
                grad_column_scale,
                column_scale,
                dtype=ctx.wgrad_dtype,
            )
        else:
            F.scaled_addmm_(
                running_grad,
                grad_column_TN.t(),
                column_TD,
                scale_a=grad_column_scale.flatten(),
                scale_recipe_a=F.ScalingType.BlockWise1x32,
                scale_b=column_scale.flatten(),
                scale_recipe_b=F.ScalingType.BlockWise1x32,
                swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
                swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
            )
            ctx.weight_param.grad = None
            grad_weight = running_grad
    return grad_latent_BTD, grad_norm_D, grad_weight


@spmd.register_local_autograd_function
@torch._dynamo.allow_in_graph
class FusedMLAQChainFunction(torch.autograd.Function):
    """RMSNorm, MXFP8 Q up-projection, and RoPE with a fused gradient quantizer."""

    @staticmethod
    def forward(
        ctx,
        latent_BTD,
        norm_weight_D,
        weight,
        cache_SR2,
        positions_BT,
        eps,
        stages,
        accumulate,
    ):
        operands = _weight_operands(weight)
        rstd_BT1, row_TD, column_TD, row_scale, column_scale = _norm_forward(
            latent_BTD, norm_weight_D, eps, ctx.needs_input_grad[2], stages
        )
        if "q_up_rope" in stages:
            q_BTN = q_up_rope_op(
                row_TD,
                row_scale,
                operands.weight_qdata_dgrad_NK,
                operands.weight_scale_fprop_swizzled,
                cache_SR2,
                positions_BT,
            )
        else:
            q_BTN = _scaled_mm(
                row_TD,
                operands.weight_qdata_fprop_KN,
                row_scale,
                operands.weight_scale_fprop_swizzled,
            ).unsqueeze(0)
            _fused_mla_q_rope_op(
                q_BTN.view(1, row_TD.shape[0], -1, 192),
                cache_SR2,
                positions_BT,
                128,
                False,
            )
        _save_chain(
            ctx,
            latent_BTD,
            rstd_BT1,
            norm_weight_D,
            column_TD,
            column_scale,
            weight,
            operands,
            cache_SR2,
            positions_BT,
            stages,
            accumulate,
        )
        return q_BTN.view(1, row_TD.shape[0], -1, 192)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_q_BTHD):
        saved = ctx.saved_tensors
        cache_SR2, positions_BT = saved[-2:]
        if "q_bwd" in ctx.stages:
            gradient = q_backward_quant_op(grad_q_BTHD, cache_SR2, positions_BT)
        else:
            # This gradient may have another consumer. The native fused MLA
            # backward mutates its contiguous buffer, so give it owned storage.
            grad_q_BTHD = grad_q_BTHD.clone(memory_format=torch.contiguous_format)
            _fused_mla_q_rope_op(grad_q_BTHD, cache_SR2, positions_BT, 128, True)
            gradient = _quantize(grad_q_BTHD.flatten(2).squeeze(0))
        return (
            *_projection_backward(ctx, gradient, saved),
            None,
            None,
            None,
            None,
            None,
        )


@spmd.register_local_autograd_function
@torch._dynamo.allow_in_graph
class FusedMLAKVChainFunction(torch.autograd.Function):
    """KV norm/up-projection, K RoPE/broadcast, and packed gradient quantization."""

    @staticmethod
    def forward(
        ctx,
        down_BTD,
        norm_weight_D,
        weight,
        cache_SR2,
        positions_BT,
        eps,
        stages,
        accumulate,
    ):
        latent_BTK, k_pe_BTR = down_BTD.split((512, 64), dim=-1)
        operands = _weight_operands(weight)
        rstd_BT1, row_TK, column_TK, row_scale, column_scale = _norm_forward(
            latent_BTK, norm_weight_D, eps, ctx.needs_input_grad[2], stages
        )
        if "kv_up_rope" in stages:
            kv_BTHD, k_BTHD = kv_up_rope_op(
                row_TK,
                row_scale,
                operands.weight_qdata_dgrad_NK,
                operands.weight_scale_fprop_swizzled,
                k_pe_BTR,
                cache_SR2,
                positions_BT,
            )
        else:
            kv_BTHD = _scaled_mm(
                row_TK,
                operands.weight_qdata_fprop_KN,
                row_scale,
                operands.weight_scale_fprop_swizzled,
            ).view(1, row_TK.shape[0], -1, 256)
            k_BTHD = _fused_mla_k_rope_op(
                kv_BTHD, k_pe_BTR, cache_SR2, positions_BT, 128
            )
        # RMSNorm's backward saves the original strided KV slice.
        _save_chain(
            ctx,
            latent_BTK,
            rstd_BT1,
            norm_weight_D,
            column_TK,
            column_scale,
            weight,
            operands,
            cache_SR2,
            positions_BT,
            stages,
            accumulate,
        )
        return k_BTHD, kv_BTHD[..., 128:]

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_k_BTHD, grad_v_BTHV):
        saved = ctx.saved_tensors
        cache_SR2, positions_BT = saved[-2:]
        if "kv_bwd" in ctx.stages:
            *gradient, grad_k_pe_BTR = kv_backward_quant_op(
                grad_k_BTHD, grad_v_BTHV, cache_SR2, positions_BT
            )
        else:
            grad_kv_BTHD, grad_k_pe_BTR = _fused_mla_kv_backward_op(
                grad_k_BTHD, grad_v_BTHV, cache_SR2, positions_BT, 128, 64
            )
            gradient = _quantize(grad_kv_BTHD.flatten(2).squeeze(0))
        grad_latent_BTK, grad_norm_K, grad_weight = _projection_backward(
            ctx, gradient, saved
        )
        grad_down_BTD = (
            torch.cat((grad_latent_BTK, grad_k_pe_BTR), dim=-1)
            if ctx.needs_input_grad[0]
            else None
        )
        return grad_down_BTD, grad_norm_K, grad_weight, None, None, None, None, None


def _chain_inputs(norm, projection, latent_BTD, cache, positions_BT, stages):
    weight = projection.weight
    accumulate = (
        weight.is_leaf
        and not torch.compiler.is_compiling()
        and get_proxy_mode() is None
    )
    return (
        latent_BTD,
        norm.weight,
        weight,
        torch.view_as_real(cache).contiguous(),
        positions_BT,
        norm.eps,
        stages,
        accumulate,
    )


def _supports_children(module):
    hooks = torch.nn.modules.module
    if any(
        (
            hooks._global_forward_pre_hooks,
            hooks._global_forward_hooks,
            hooks._global_backward_pre_hooks,
            hooks._global_backward_hooks,
        )
    ):
        return False
    for child in (module.q_norm, module.kv_norm, module.wq_b, module.wkv_b):
        # The fusion spans these child boundaries. An independently sharded
        # child or a selective remat policy must retain its native calls.
        if (
            isinstance(child, FSDPModule)
            or child._forward_pre_hooks
            or child._forward_hooks
            or child._backward_pre_hooks
            or child._backward_hooks
            or child._remat_save_patterns
            or child._remat_recompute_patterns
        ):
            return False
    return True


_NATIVE_CUTEDSL_AVAILABLE = False
_NATIVE_FILTER_STATE = None
if torch.__version__ == "2.16.0.dev20261007+cu130" and triton.__version__ == "3.9.0":
    from torch.backends import python_native

    # The public controller methods are on Dynamo's skip list. Cache
    # availability outside tracing, but let Dynamo guard the mutable filter
    # sets so disabling native RMSNorm invalidates a compiled fast path.
    _NATIVE_CUTEDSL_AVAILABLE = python_native.cutedsl.available
    _NATIVE_FILTER_STATE = python_native._get_filter_state()


def _native_runtime_matches():
    return (
        _CUTEDSL_IMPORT_ERROR is None
        and _NATIVE_CUTEDSL_AVAILABLE
        and torch.are_deterministic_algorithms_enabled()
        and "cutedsl" not in _NATIVE_FILTER_STATE._dsl_names
        and "_fused_rms_norm" not in _NATIVE_FILTER_STATE._op_symbols
        and "CUDA" not in _NATIVE_FILTER_STATE._dispatch_keys
    )


class FusedDSv3MLANormQuantRoPE(FusedMLAAttention):
    """Preserve native parameters, attention, FSDP owner, and remat boundaries."""

    @dataclass(kw_only=True, slots=True)
    class Config(FusedMLAAttention.Config):
        mla_nqr_stages: tuple[str, ...] = ()

    def __init__(self, config: Config):
        super().__init__(config)
        self.mla_nqr_stages = config.mla_nqr_stages

    def _supports_chain_fusion(self, x_TD):
        return (
            bool(self.mla_nqr_stages)
            and type(x_TD) in (torch.Tensor, FakeTensor)
            and x_TD.is_cuda
            and x_TD.dtype == torch.bfloat16
            and x_TD.shape == (4096, 7168)
            and x_TD.is_contiguous()
            and self.q_lora_rank == 1536
            and self.kv_lora_rank == 512
            and self.qk_nope_head_dim == 128
            and self.qk_rope_head_dim == 64
            and self.v_head_dim == 128
            and self.n_heads == 128
            and type(self.q_norm) is RMSNorm
            and type(self.kv_norm) is RMSNorm
            and self.q_norm.eps == 1e-5
            and self.kv_norm.eps == 1e-5
            and self.q_norm.weight.shape == (1536,)
            and self.kv_norm.weight.shape == (512,)
            and self.q_norm.weight.is_contiguous()
            and self.kv_norm.weight.is_contiguous()
            and type(self.wq_b) is MXFP8Linear
            and type(self.wkv_b) is MXFP8Linear
            and self.wq_b.bias is None
            and self.wkv_b.bias is None
            # The handoff saves the normalized columnwise MXFP8 operand.
            # Keep the native BF16 save-policy branch intact.
            and self.wq_b.input_activation_format_for_backward == "mxfp8"
            and self.wkv_b.input_activation_format_for_backward == "mxfp8"
            and all(
                weight.dtype == torch.bfloat16
                for weight in (
                    self.q_norm.weight,
                    self.kv_norm.weight,
                    self.wq_b.weight,
                    self.wkv_b.weight,
                )
            )
            and self.wq_b.weight.shape == (24576, 1536)
            and self.wkv_b.weight.shape == (32768, 512)
            and spmd_mesh_size("tp") == 1
            and spmd_mesh_size("cp") == 1
            and _supports_children(self)
            and not torch.is_autocast_enabled("cuda")
            and _native_runtime_matches()
            and (
                isinstance(x_TD, FakeTensor)
                or torch.cuda.get_device_capability(x_TD.device) == (10, 3)
            )
        )

    def forward(self, x_TD, attention_metadata, positions=None):
        if not self._supports_chain_fusion(x_TD):
            return super().forward(x_TD, attention_metadata, positions)
        x_TD = maybe_gather_tp_input(self, x_TD)
        if positions is not None:
            _maybe_check_max_pos(positions, max_valid_pos=self.rope.cache.shape[0] - 1)
        positions_BT = _resolve_positions(positions, x_TD.unsqueeze(0))
        latent_TQ = self.wq_a(x_TD)
        remat.recompute_needs_tensor(latent_TQ)
        q_BTHD = FusedMLAQChainFunction.apply(
            *_chain_inputs(
                self.q_norm,
                self.wq_b,
                latent_TQ.unsqueeze(0),
                self.rope.cache,
                positions_BT,
                self.mla_nqr_stages,
            )
        )
        down_TD = self.wkv_a(x_TD)
        remat.recompute_needs_tensor(down_TD)
        k_BTHD, v_BTHV = FusedMLAKVChainFunction.apply(
            *_chain_inputs(
                self.kv_norm,
                self.wkv_b,
                down_TD.unsqueeze(0),
                self.rope.cache,
                positions_BT,
                self.mla_nqr_stages,
            )
        )
        output_THV = remat.region(
            self.inner_attention,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(
            q_BTHD.squeeze(0),
            k_BTHD.squeeze(0),
            v_BTHV.squeeze(0),
            attention_metadata=attention_metadata,
            scale=self.softmax_scale,
        )
        remat.recompute_needs_tensor(output_THV)
        return self.wo(output_THV.contiguous().view(x_TD.shape[0], -1))


@override(
    target=Attention.Config,
    exact=True,
    description="Fuse DeepSeek V3 MLA RMSNorm, MXFP8 projections, RoPE, and their gradient preparation.",
)
def fused_mla_norm_quant_rope(
    cfg: Attention.Config,
    *,
    stages: tuple[str, ...] = _STAGES,
    experimental: bool = False,
) -> FusedDSv3MLANormQuantRoPE.Config:
    if not set(stages).issubset(_STAGES):
        raise ValueError(f"Unknown MLA fusion stages: {set(stages) - set(_STAGES)}")
    if not (ACCEPTED or experimental):
        logger.warning(
            "MLA norm/quant/RoPE fusion has not met all roofline gates; "
            "keeping the existing fused_mla path. "
            "Set experimental=True to measure it."
        )
    return derive(
        cfg,
        FusedDSv3MLANormQuantRoPE.Config,
        mla_nqr_stages=tuple(stages) if ACCEPTED or experimental else (),
    )
