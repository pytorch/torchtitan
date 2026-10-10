# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Shared W2 dgrad, native SwiGLU derivative, and both MXFP8 orientations.

The sequential-K accumulator is rounded to BF16 before the derivative. Each
derivative is rounded to BF16 before native RCEIL quantization. Gate and up
occupy the same two halves as the original concatenation.
"""

from dataclasses import replace

import torch
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    allocate_tensor_memory,
    mbarrier,
    tcgen05_commit,
    tcgen05_copy,
    tcgen05_mma_scaled,
    tensor_memory_descriptor,
    TensorMemoryLayout,
    TensorMemoryScalesLayout,
    tma,
)

from .descriptors import _matrix, _scales

from .quantize import _blocked, _quantize, _scale


@gluon.aggregate
class _Pipeline:
    gradient: tma.tensor_descriptor
    weight: tma.tensor_descriptor
    gradient_scale: tma.tensor_descriptor
    weight_scale: tma.tensor_descriptor
    row_output: tma.tensor_descriptor
    column_output: tma.tensor_descriptor
    gate_up: gl.tensor
    row_pointer: gl.tensor
    column_pointer: gl.tensor
    row_scale: gl.tensor
    column_scale: gl.tensor
    saved_hidden_gradient: gl.tensor
    saved_packed_gradient: gl.tensor
    a_buffers: gl.shared_memory_descriptor
    b_buffers: gl.shared_memory_descriptor
    as_buffers: gl.shared_memory_descriptor
    bs_buffers: gl.shared_memory_descriptor
    load_empty: gl.shared_memory_descriptor
    load_ready: gl.shared_memory_descriptor
    accumulators: tensor_memory_descriptor
    acc_empty: gl.shared_memory_descriptor
    acc_ready: gl.shared_memory_descriptor
    M: gl.constexpr
    N: gl.constexpr
    K: gl.constexpr
    BM: gl.constexpr
    BN: gl.constexpr
    BK: gl.constexpr
    BUFFERS: gl.constexpr
    ACC_BUFFERS: gl.constexpr
    SAVE_INTERMEDIATES: gl.constexpr
    TRANSPOSE_WEIGHT: gl.constexpr
    NUM_WARPS: gl.constexpr
    REGISTER_QUANT: gl.constexpr
    EPILOGUE_LAYOUT: gl.constexpr
    COLUMN_LAYOUT: gl.constexpr
    TILE_LAYOUT: gl.constexpr
    QUANT_ROW_LAYOUT: gl.constexpr
    QUANT_COL_LAYOUT: gl.constexpr
    QUANT_ROW_SCALE_LAYOUT: gl.constexpr
    QUANT_COL_SCALE_LAYOUT: gl.constexpr


@gluon.jit
def _coordinates(p, tile):
    return tile % (p.M // p.BM), tile // (p.M // p.BM)


@gluon.jit
def _load_partition(p):
    TOTAL: gl.constexpr = (p.M // p.BM) * (p.N // p.BN)
    count = 0
    for tile in range(gl.program_id(0), TOTAL, gl.num_programs(0)):
        row_tile, column_tile = _coordinates(p, tile)
        for start in range(0, p.K, p.BK):
            slot = count % p.BUFFERS
            mbarrier.wait(p.load_empty.index(slot), (count // p.BUFFERS) % 2 ^ 1)
            bar = p.load_ready.index(slot)
            mbarrier.expect(
                bar,
                p.a_buffers.index(slot).nbytes_per_cta
                + p.b_buffers.index(slot).nbytes_per_cta
                + p.as_buffers.index(slot).nbytes_per_cta
                + p.bs_buffers.index(slot).nbytes_per_cta,
            )
            tma.async_load(
                p.gradient, [row_tile * p.BM, start], bar, p.a_buffers.index(slot)
            )
            if p.TRANSPOSE_WEIGHT:
                tma.async_load(
                    p.weight, [column_tile * p.BN, start], bar, p.b_buffers.index(slot)
                )
            else:
                tma.async_load(
                    p.weight, [start, column_tile * p.BN], bar, p.b_buffers.index(slot)
                )
            tma.async_load(
                p.gradient_scale,
                [0, row_tile * (p.BM // 128), start // 128, 0, 0],
                bar,
                p.as_buffers.index(slot),
            )
            tma.async_load(
                p.weight_scale,
                [0, column_tile * (p.BN // 128), start // 128, 0, 0],
                bar,
                p.bs_buffers.index(slot),
            )
            count += 1


@gluon.jit
def _mma_partition(p):
    TOTAL: gl.constexpr = (p.M // p.BM) * (p.N // p.BN)
    load_count = 0
    tile_count = 0
    for tile in range(gl.program_id(0), TOTAL, gl.num_programs(0)):
        acc_slot = tile_count % p.ACC_BUFFERS
        mbarrier.wait(
            p.acc_empty.index(acc_slot), (tile_count // p.ACC_BUFFERS) % 2 ^ 1
        )
        accumulator = p.accumulators.index(acc_slot)
        use_acc = False
        for _ in range(0, p.K, p.BK):
            slot = load_count % p.BUFFERS
            mbarrier.wait(p.load_ready.index(slot), (load_count // p.BUFFERS) % 2)
            left_scale = p.as_buffers.index(slot).reshape(
                (p.BM // 128, p.BK // 128, 32, 4, 4)
            )
            left_scale = left_scale.permute((0, 3, 2, 1, 4)).reshape((p.BM, p.BK // 32))
            right_scale = p.bs_buffers.index(slot).reshape(
                (p.BN // 128, p.BK // 128, 32, 4, 4)
            )
            right_scale = right_scale.permute((0, 3, 2, 1, 4)).reshape(
                (p.BN, p.BK // 32)
            )
            left_tmem = allocate_tensor_memory(
                gl.uint8,
                (p.BM, p.BK // 32),
                TensorMemoryScalesLayout(
                    cga_layout=[[1, 0]] if gl.num_ctas() == 2 else []
                ),
            )
            right_tmem = allocate_tensor_memory(
                gl.uint8,
                (p.BN, p.BK // 32),
                TensorMemoryScalesLayout(
                    cga_layout=[[0, 0]] if gl.num_ctas() == 2 else []
                ),
            )
            tcgen05_copy(left_scale, left_tmem)
            tcgen05_copy(right_scale, right_tmem)
            if p.TRANSPOSE_WEIGHT:
                right = p.b_buffers.index(slot).permute((1, 0))
            else:
                right = p.b_buffers.index(slot)
            tcgen05_mma_scaled(
                p.a_buffers.index(slot),
                right,
                accumulator,
                left_tmem,
                right_tmem,
                "e4m3",
                "e4m3",
                use_acc=use_acc,
            )
            tcgen05_commit(
                p.load_empty.index(slot),
                descs=[p.a_buffers.index(slot), p.b_buffers.index(slot)],
            )
            load_count += 1
            use_acc = True
        tcgen05_commit(
            p.acc_ready.index(acc_slot),
            descs=[p.a_buffers.index(0), p.b_buffers.index(0)],
        )
        tile_count += 1


@gluon.jit
def _quantize_registers(p, values, row_start, column_start):
    row_values = values.to(gl.float32).reshape((p.BM, 2, 32))
    magnitudes = gl.abs(row_values)
    row_reciprocal, row_biased = _scale(gl.max(magnitudes, 2))
    row_quantized = (
        (row_values * row_reciprocal[:, :, None]).reshape((p.BM, 64)).to(gl.float8e4nv)
    )
    layout: gl.constexpr = values.type.layout
    rows = row_start + gl.arange(0, p.BM, layout=gl.SliceLayout(1, layout))
    columns = column_start + gl.arange(0, 64, layout=gl.SliceLayout(0, layout))
    gl.store(
        p.row_pointer + rows[:, None] * (2 * p.N) + columns[None, :],
        gl.convert_layout(row_quantized, layout),
    )
    row_scale_layout: gl.constexpr = row_biased.type.layout
    scale_rows = row_start + gl.arange(
        0, p.BM, layout=gl.SliceLayout(1, row_scale_layout)
    )
    scale_groups = column_start // 32 + gl.arange(
        0, 2, layout=gl.SliceLayout(0, row_scale_layout)
    )
    gl.store(
        p.row_scale
        + _blocked(scale_rows[:, None], scale_groups[None, :], 2 * p.N // 32),
        row_biased,
    )
    column_layout: gl.constexpr = p.COLUMN_LAYOUT
    column_values = gl.convert_layout(values.to(gl.float32), column_layout).reshape(
        (p.BM // 32, 32, 64)
    )
    magnitudes = gl.abs(column_values)
    column_reciprocal, column_biased = _scale(gl.max(magnitudes, 1))
    column_quantized = (
        (column_values * column_reciprocal[:, None, :])
        .reshape((p.BM, 64))
        .to(gl.float8e4nv)
    )
    rows = row_start + gl.arange(0, p.BM, layout=gl.SliceLayout(1, column_layout))
    columns = column_start + gl.arange(0, 64, layout=gl.SliceLayout(0, column_layout))
    gl.store(
        p.column_pointer + columns[None, :] * p.M + rows[:, None],
        gl.convert_layout(column_quantized, column_layout),
    )
    column_scale_layout: gl.constexpr = column_biased.type.layout
    scale_groups = row_start // 32 + gl.arange(
        0, p.BM // 32, layout=gl.SliceLayout(1, column_scale_layout)
    )
    scale_columns = column_start + gl.arange(
        0, 64, layout=gl.SliceLayout(0, column_scale_layout)
    )
    gl.store(
        p.column_scale
        + _blocked(scale_columns[None, :], scale_groups[:, None], p.M // 32),
        column_biased,
    )


@gluon.jit
def _epilogue_partition(p):
    TOTAL: gl.constexpr = (p.M // p.BM) * (p.N // p.BN)
    layout: gl.constexpr = p.EPILOGUE_LAYOUT
    if not p.REGISTER_QUANT:
        tile_layout: gl.constexpr = p.TILE_LAYOUT
        gradient_tile = gl.allocate_shared_memory(gl.bfloat16, (p.BM, 64), tile_layout)
        row_stages = gl.allocate_shared_memory(
            gl.float8e4nv, (2, p.BM, 64), p.row_output.layout
        )
        column_stages = gl.allocate_shared_memory(
            gl.float8e4nv, (2, 64, p.BM), p.column_output.layout
        )
    tile_count = 0
    store_count = 0
    for tile in range(gl.program_id(0), TOTAL, gl.num_programs(0)):
        row_tile, column_tile = _coordinates(p, tile)
        acc_slot = tile_count % p.ACC_BUFFERS
        mbarrier.wait(p.acc_ready.index(acc_slot), (tile_count // p.ACC_BUFFERS) % 2)
        accumulator = p.accumulators.index(acc_slot)
        chunks = ()
        for chunk_index in gl.static_range(p.BN // 64):
            part = accumulator.slice(chunk_index * 64, 64)
            chunks += (
                part.load(part.get_reg_layout(instr_variant="32x32b")).to(gl.bfloat16),
            )
        mbarrier.arrive(p.acc_empty.index(acc_slot), count=1)
        for chunk_index in gl.static_range(p.BN // 64):
            hidden_gradient = gl.convert_layout(chunks[chunk_index], layout).to(
                gl.float32
            )
            row_start = row_tile * p.BM
            column_start = column_tile * p.BN + chunk_index * 64
            rows = row_start + gl.arange(0, p.BM, layout=gl.SliceLayout(1, layout))
            columns = column_start + gl.arange(0, 64, layout=gl.SliceLayout(0, layout))
            offsets = rows[:, None] * (2 * p.N) + columns[None, :]
            gate = gl.load(p.gate_up + offsets).to(gl.float32)
            up = gl.load(p.gate_up + offsets + p.N).to(gl.float32)
            sigmoid = 1.0 / (1.0 + gl.exp(-gate))
            silu = gate * sigmoid
            silu_gradient = sigmoid * (1.0 + gate * (1.0 - sigmoid))
            gate_gradient = (hidden_gradient * up * silu_gradient).to(gl.bfloat16)
            up_gradient = (hidden_gradient * silu).to(gl.bfloat16)
            if p.SAVE_INTERMEDIATES:
                gl.store(
                    p.saved_hidden_gradient + rows[:, None] * p.N + columns[None, :],
                    hidden_gradient.to(gl.bfloat16),
                )
                gl.store(p.saved_packed_gradient + offsets, gate_gradient)
                gl.store(p.saved_packed_gradient + offsets + p.N, up_gradient)
            for branch in gl.static_range(2):
                branch_gradient = gate_gradient if branch == 0 else up_gradient
                if p.REGISTER_QUANT:
                    _quantize_registers(
                        p, branch_gradient, row_start, column_start + branch * p.N
                    )
                else:
                    gradient_tile.store(branch_gradient)
                    tma.store_wait(2)
                    stage = store_count % 2
                    _quantize(
                        gradient_tile,
                        row_stages.index(stage),
                        column_stages.index(stage),
                        p.row_output,
                        p.column_output,
                        p.row_scale,
                        p.column_scale,
                        row_start,
                        column_start + branch * p.N,
                        p.M,
                        2 * p.N,
                        p.BM,
                        True,
                        True,
                        p.NUM_WARPS,
                        p.QUANT_ROW_LAYOUT,
                        p.QUANT_COL_LAYOUT,
                        p.QUANT_ROW_SCALE_LAYOUT,
                        p.QUANT_COL_SCALE_LAYOUT,
                    )
                    store_count += 1
        tile_count += 1
    if not p.REGISTER_QUANT:
        tma.store_wait(0)


@gluon.jit
def _kernel(
    gradient,
    weight,
    gradient_scale,
    weight_scale,
    row_output,
    column_output,
    gate_up,
    row_pointer,
    column_pointer,
    row_scale,
    column_scale,
    saved_hidden_gradient,
    saved_packed_gradient,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    BUFFERS: gl.constexpr,
    ACC_BUFFERS: gl.constexpr,
    SAVE_INTERMEDIATES: gl.constexpr,
    TRANSPOSE_WEIGHT: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    REGISTER_QUANT: gl.constexpr,
    EPILOGUE_LAYOUT: gl.constexpr,
    COLUMN_LAYOUT: gl.constexpr,
    TILE_LAYOUT: gl.constexpr,
    QUANT_ROW_LAYOUT: gl.constexpr,
    QUANT_COL_LAYOUT: gl.constexpr,
    QUANT_ROW_SCALE_LAYOUT: gl.constexpr,
    QUANT_COL_SCALE_LAYOUT: gl.constexpr,
):
    BM: gl.constexpr = 128 * gl.num_ctas()
    a_buffers = gl.allocate_shared_memory(
        gradient.dtype, [BUFFERS] + gradient.block_type.shape, gradient.layout
    )
    b_buffers = gl.allocate_shared_memory(
        weight.dtype, [BUFFERS] + weight.block_type.shape, weight.layout
    )
    as_buffers = gl.allocate_shared_memory(
        gradient_scale.dtype,
        [BUFFERS] + gradient_scale.block_type.shape,
        gradient_scale.layout,
    )
    bs_buffers = gl.allocate_shared_memory(
        weight_scale.dtype,
        [BUFFERS] + weight_scale.block_type.shape,
        weight_scale.layout,
    )
    load_empty = mbarrier.allocate_mbarrier(batch=BUFFERS)
    load_ready = mbarrier.allocate_mbarrier(batch=BUFFERS, two_ctas=gl.num_ctas() == 2)
    for index in gl.static_range(BUFFERS):
        mbarrier.init(load_empty.index(index), count=1)
        mbarrier.init(load_ready.index(index), count=1)
    accumulators = allocate_tensor_memory(
        gl.float32,
        (ACC_BUFFERS, BM, BN),
        TensorMemoryLayout(
            [128, BN],
            col_stride=1,
            cga_layout=[[1, 0]] if gl.num_ctas() == 2 else [],
            two_ctas=gl.num_ctas() == 2,
        ),
    )
    acc_empty = mbarrier.allocate_mbarrier(
        batch=ACC_BUFFERS, two_ctas=gl.num_ctas() == 2
    )
    acc_ready = mbarrier.allocate_mbarrier(batch=ACC_BUFFERS)
    for index in gl.static_range(ACC_BUFFERS):
        mbarrier.init(acc_empty.index(index), count=1)
        mbarrier.init(acc_ready.index(index), count=1)
    p = _Pipeline(
        gradient,
        weight,
        gradient_scale,
        weight_scale,
        row_output,
        column_output,
        gate_up,
        row_pointer,
        column_pointer,
        row_scale,
        column_scale,
        saved_hidden_gradient,
        saved_packed_gradient,
        a_buffers,
        b_buffers,
        as_buffers,
        bs_buffers,
        load_empty,
        load_ready,
        accumulators,
        acc_empty,
        acc_ready,
        M,
        N,
        K,
        BM,
        BN,
        BK,
        BUFFERS,
        ACC_BUFFERS,
        SAVE_INTERMEDIATES,
        TRANSPOSE_WEIGHT,
        NUM_WARPS,
        REGISTER_QUANT,
        EPILOGUE_LAYOUT,
        COLUMN_LAYOUT,
        TILE_LAYOUT,
        QUANT_ROW_LAYOUT,
        QUANT_COL_LAYOUT,
        QUANT_ROW_SCALE_LAYOUT,
        QUANT_COL_SCALE_LAYOUT,
    )
    gl.warp_specialize(
        [(_epilogue_partition, (p,)), (_mma_partition, (p,)), (_load_partition, (p,))],
        [1, 1],
        [24, 24],
    )


def launch_swiglu_backward(
    gradient,
    weight,
    gradient_scale,
    weight_scale,
    gate_up,
    *,
    block_n=128,
    block_k=128,
    buffers=None,
    acc_buffers=2,
    num_warps=None,
    maxnreg=None,
    grid_size=None,
    transpose_weight=False,
    save_intermediates=False,
    num_ctas=None,
    register_quant=False,
    kernel_metadata=None,
):
    """Return native logical (row, column, row_scale, column_scale) operands.

    ``weight`` keeps native [K,N] contiguous storage. The explicit transpose
    experiment includes its preparation launch in every call.
    """
    rows, reduction = gradient.shape
    hidden = gate_up.shape[1] // 2
    production_default = num_ctas is None and (rows, hidden, reduction) == (
        4096,
        2048,
        7168,
    )
    num_ctas = (2 if production_default else 1) if num_ctas is None else num_ctas
    buffers = (6 if production_default else 4) if buffers is None else buffers
    num_warps = (8 if production_default else 4) if num_warps is None else num_warps
    maxnreg = (144 if production_default else 192) if maxnreg is None else maxnreg
    if production_default and grid_size is None:
        grid_size = 64
    block_m = 128 * num_ctas
    if num_ctas not in (1, 2) or num_warps not in (4, 8):
        raise ValueError(
            "SwiGLU backward supports one/two CTAs and four/eight epilogue warps"
        )
    if rows % block_m or hidden % block_n or reduction % block_k:
        raise ValueError("SwiGLU backward requires whole GEMM tiles")
    row = gradient.new_empty((rows, 2 * hidden))
    column_storage = gradient.new_empty((2 * hidden, rows))
    row_scale = gradient_scale.new_empty((rows * 2 * hidden // 32,))
    column_scale = gradient_scale.new_empty((rows * 2 * hidden // 32,))
    saved_hidden = gate_up.new_empty((rows, hidden) if save_intermediates else (0,))
    saved_packed = gate_up.new_empty(gate_up.shape if save_intermediates else (0,))
    if transpose_weight:
        weight = weight.t().contiguous()
        weight_descriptor = _matrix(weight, hidden, reduction, block_n, block_k)
    else:
        weight_descriptor = _matrix(weight, reduction, hidden, block_k, block_n)
    gradient_descriptor = _matrix(gradient, rows, reduction, block_m, block_k)
    gradient_scale_descriptor = _scales(
        gradient_scale, rows, reduction, block_m, block_k
    )
    weight_scale_descriptor = _scales(weight_scale, hidden, reduction, block_n, block_k)
    if num_ctas == 2:
        gradient_descriptor = replace(
            gradient_descriptor,
            layout=replace(gradient_descriptor.layout, cga_layout=[[1, 0]]),
        )
        gradient_scale_descriptor = replace(
            gradient_scale_descriptor,
            layout=replace(
                gradient_scale_descriptor.layout, cga_layout=[[0, 1, 0, 0, 0]]
            ),
        )
        weight_descriptor = replace(
            weight_descriptor,
            layout=gl.NVMMASharedLayout.get_default_for(
                [block_n, block_k] if transpose_weight else [block_k, block_n],
                gl.float8e4nv,
                cga_layout=[[1, 0]] if transpose_weight else [[0, 1]],
            ),
        )
        weight_scale_descriptor = replace(
            weight_scale_descriptor,
            layout=replace(
                weight_scale_descriptor.layout, cga_layout=[[0, 0, 0, 0, 0]]
            ),
        )
    total_tiles = rows // block_m * (hidden // block_n)
    sms = (
        torch.cuda.get_device_properties(gradient.device).multi_processor_count
        // num_ctas
    )
    grid = (min(grid_size or sms, total_tiles),)
    cga_layout = [[1, 0]] if num_ctas == 2 else []
    epilogue_layout = gl.BlockedLayout(
        [1, 2], [4, 8], [4, num_warps // 4], [1, 0], cga_layout=cga_layout
    )
    column_layout = gl.BlockedLayout(
        [2, 1], [8, 4], [4, num_warps // 4], [0, 1], cga_layout=cga_layout
    )
    tile_layout = gl.NVMMASharedLayout(
        swizzle_byte_width=128, element_bitwidth=16, rank=2, cga_layout=cga_layout
    )
    row_descriptor = _matrix(row, rows, 2 * hidden, block_m, 64)
    column_descriptor = _matrix(column_storage, 2 * hidden, rows, 64, block_m)
    if num_ctas == 2:
        row_descriptor = replace(
            row_descriptor, layout=replace(row_descriptor.layout, cga_layout=[[1, 0]])
        )
        column_descriptor = replace(
            column_descriptor,
            layout=gl.NVMMASharedLayout.get_default_for(
                [64, block_m],
                gl.float8e4nv,
                cga_layout=[[0, 1]],
            ),
        )
    row_warps = block_m // (32 * num_ctas)
    column_warps = num_warps // row_warps
    quant_row_layout = gl.BlockedLayout(
        [1, 32 // column_warps],
        [32, 1],
        [row_warps, column_warps],
        [1, 0],
        cga_layout=cga_layout,
    )
    quant_column_layout = gl.BlockedLayout(
        [32 // column_warps, 1],
        [column_warps, 32 // column_warps],
        [row_warps, column_warps],
        [0, 1],
        cga_layout=cga_layout,
    )
    quant_row_scale_layout = gl.BlockedLayout(
        [1, 2], [32, 1], [row_warps, column_warps], [1, 0], cga_layout=cga_layout
    )
    quant_column_scale_layout = gl.BlockedLayout(
        [1, 2], [1, 32], [row_warps, column_warps], [1, 0], cga_layout=cga_layout
    )
    compiled = _kernel[grid](
        gradient_descriptor,
        weight_descriptor,
        gradient_scale_descriptor,
        weight_scale_descriptor,
        row_descriptor,
        column_descriptor,
        gate_up,
        row,
        column_storage,
        row_scale.view(torch.uint8),
        column_scale.view(torch.uint8),
        saved_hidden,
        saved_packed,
        rows,
        hidden,
        reduction,
        block_n,
        block_k,
        buffers,
        acc_buffers,
        save_intermediates,
        transpose_weight,
        num_warps,
        register_quant,
        epilogue_layout,
        column_layout,
        tile_layout,
        quant_row_layout,
        quant_column_layout,
        quant_row_scale_layout,
        quant_column_scale_layout,
        # Inherit the same floating-point contraction default as FusedSwiGLU.
        # Forcing FMA changes the BF16 gate gradient on Triton 3.9.
        num_warps=num_warps,
        num_ctas=num_ctas,
        maxnreg=maxnreg,
    )
    if kernel_metadata is not None:
        kernel_metadata.update(
            registers=compiled.n_regs,
            spills=compiled.n_spills,
            shared_bytes=compiled.metadata.shared,
            threads=compiled.metadata.num_warps * 32,
        )
    outputs = row, column_storage.t(), row_scale, column_scale
    return (*outputs, saved_hidden, saved_packed) if save_intermediates else outputs
