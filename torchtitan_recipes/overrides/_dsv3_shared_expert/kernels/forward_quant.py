# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""W13 epilogue: BF16 SwiGLU and native row/column MXFP8.

The W13 load/MMA pipeline is shared with shared_swiglu. This epilogue retains
the original BF16 rounding after GEMM and after SwiGLU. The real BF16 hidden
tensor is written for the existing W2 module boundary and observable hooks.
"""

import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.nvidia.blackwell import mbarrier, tma

from .quantize import _quantize


@gluon.jit
def _epilogue_partition(
    p,
    row_output,
    column_output,
    row_scale,
    column_scale,
    EPILOGUE_LAYOUT: gl.constexpr,
    TILE_LAYOUT: gl.constexpr,
    ROW_LAYOUT: gl.constexpr,
    COL_LAYOUT: gl.constexpr,
    ROW_SCALE_LAYOUT: gl.constexpr,
    COL_SCALE_LAYOUT: gl.constexpr,
):
    TOTAL: gl.constexpr = (p.M // p.BM) * (p.HIDDEN // p.BN)
    hidden_tile = gl.allocate_shared_memory(gl.bfloat16, (p.BM, 64), TILE_LAYOUT)
    row_stages = gl.allocate_shared_memory(
        gl.float8e4nv,
        (2, p.BM, 64),
        row_output.layout,
    )
    column_stages = gl.allocate_shared_memory(
        gl.float8e4nv,
        (2, 64, p.BM),
        column_output.layout,
    )
    tile_count = 0
    store_count = 0
    for tile in range(gl.program_id(0), TOTAL, gl.num_programs(0)):
        row_tile = tile % (p.M // p.BM)
        column_tile = tile // (p.M // p.BM)
        acc_slot = tile_count % p.ACC_BUFFERS
        mbarrier.wait(p.acc_ready.index(acc_slot), (tile_count // p.ACC_BUFFERS) % 2)
        accumulator = p.accumulators.index(acc_slot)
        gate_chunks = ()
        up_chunks = ()
        for chunk_index in gl.static_range(p.BN // 64):
            gate_part = accumulator.slice(chunk_index * 64, 64)
            up_part = accumulator.slice(p.BN + chunk_index * 64, 64)
            gate_chunks += (
                gate_part.load(
                    gate_part.get_reg_layout(instr_variant=p.LOAD_VARIANT)
                ).to(gl.bfloat16),
            )
            up_chunks += (
                up_part.load(up_part.get_reg_layout(instr_variant=p.LOAD_VARIANT)).to(
                    gl.bfloat16
                ),
            )
        mbarrier.arrive(p.acc_empty.index(acc_slot), count=1)
        for chunk_index in gl.static_range(p.BN // 64):
            gate = gl.convert_layout(gate_chunks[chunk_index], EPILOGUE_LAYOUT)
            up = gl.convert_layout(up_chunks[chunk_index], EPILOGUE_LAYOUT)
            row_start = row_tile * p.BM
            column_start = column_tile * p.BN + chunk_index * 64
            rows = row_start + gl.arange(
                0, p.BM, layout=gl.SliceLayout(1, EPILOGUE_LAYOUT)
            )
            columns = column_start + gl.arange(
                0, 64, layout=gl.SliceLayout(0, EPILOGUE_LAYOUT)
            )
            packed_offsets = rows[:, None] * (2 * p.HIDDEN) + columns[None, :]
            if p.SAVE_INTERMEDIATES:
                gl.store(p.gate_up_pointer + packed_offsets, gate)
                gl.store(p.gate_up_pointer + packed_offsets + p.HIDDEN, up)
            gate_f32, up_f32 = gate.to(gl.float32), up.to(gl.float32)
            sigmoid = 1.0 / (1.0 + gl.exp(-gate_f32))
            hidden = (gate_f32 * sigmoid * up_f32).to(gl.bfloat16)
            gl.store(
                p.output_pointer + rows[:, None] * p.HIDDEN + columns[None, :], hidden
            )
            hidden_tile.store(hidden)
            tma.store_wait(2)
            stage = store_count % 2
            _quantize(
                hidden_tile,
                row_stages.index(stage),
                column_stages.index(stage),
                row_output,
                column_output,
                row_scale,
                column_scale,
                row_start,
                column_start,
                p.M,
                p.HIDDEN,
                p.BM,
                True,
                True,
                gl.num_warps(),
                ROW_LAYOUT,
                COL_LAYOUT,
                ROW_SCALE_LAYOUT,
                COL_SCALE_LAYOUT,
            )
            store_count += 1
        tile_count += 1
    tma.store_wait(0)
