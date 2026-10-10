# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Blackwell packed W1/W3 MXFP8 GEMM with the native SwiGLU epilogue."""

from dataclasses import replace

import torch
import triton
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    allocate_tensor_memory,
    fence_async_shared,
    mbarrier,
    tcgen05_commit,
    tcgen05_copy,
    tcgen05_mma_scaled,
    tensor_memory_descriptor,
    TensorMemoryLayout,
    TensorMemoryScalesLayout,
    tma,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from .descriptors import _matrix, _scales
from .forward_quant import _epilogue_partition as _quantized_epilogue


_EPILOGUE_LAYOUT_4 = gl.BlockedLayout(
    [1, 2], [4, 8], [4, 1], [1, 0], cga_layout=[[1, 0]]
)
_EPILOGUE_LAYOUT_8 = gl.BlockedLayout(
    [1, 2], [4, 8], [4, 2], [1, 0], cga_layout=[[1, 0]]
)


@gluon.aggregate
class _Pipeline:
    activation: tma.tensor_descriptor
    weight: tma.tensor_descriptor
    activation_scale: tma.tensor_descriptor
    weight_scale: tma.tensor_descriptor
    gate_up: tma.tensor_descriptor
    output: tma.tensor_descriptor
    gate_up_pointer: gl.tensor
    output_pointer: gl.tensor
    activation_buffers: gl.shared_memory_descriptor
    weight_buffers: gl.shared_memory_descriptor
    activation_scale_buffers: gl.shared_memory_descriptor
    weight_scale_buffers: gl.shared_memory_descriptor
    load_empty: gl.shared_memory_descriptor
    load_ready: gl.shared_memory_descriptor
    accumulators: tensor_memory_descriptor
    acc_empty: gl.shared_memory_descriptor
    acc_ready: gl.shared_memory_descriptor
    M: gl.constexpr
    HIDDEN: gl.constexpr
    K: gl.constexpr
    BM: gl.constexpr
    BN: gl.constexpr
    BK: gl.constexpr
    BUFFERS: gl.constexpr
    ACC_BUFFERS: gl.constexpr
    SAVE_INTERMEDIATES: gl.constexpr
    TMA_STORES: gl.constexpr
    LOAD_VARIANT: gl.constexpr
    COALESCED: gl.constexpr


@gluon.jit
def _coordinates(p, tile):
    row_tile = tile % gl.cdiv(p.M, p.BM)
    column_tile = tile // gl.cdiv(p.M, p.BM)
    return row_tile, column_tile


@gluon.jit
def _load_partition(p):
    TOTAL: gl.constexpr = gl.cdiv(p.M, p.BM) * gl.cdiv(p.HIDDEN, p.BN)
    count = 0
    for tile in range(gl.program_id(0), TOTAL, gl.num_programs(0)):
        row_tile, column_tile = _coordinates(p, tile)
        for start in range(0, p.K, p.BK):
            slot = count % p.BUFFERS
            mbarrier.wait(p.load_empty.index(slot), (count // p.BUFFERS) % 2 ^ 1)
            bar = p.load_ready.index(slot)
            mbarrier.expect(
                bar,
                p.activation_buffers.index(slot).nbytes_per_cta
                + p.weight_buffers.index(slot).nbytes_per_cta
                + p.activation_scale_buffers.index(slot).nbytes_per_cta
                + p.weight_scale_buffers.index(slot).nbytes_per_cta,
            )
            tma.async_load(
                p.activation,
                [row_tile * p.BM, start],
                bar,
                p.activation_buffers.index(slot),
            )
            tma.async_load(
                p.activation_scale,
                [0, row_tile * (p.BM // 128), start // 128, 0, 0],
                bar,
                p.activation_scale_buffers.index(slot),
            )
            tma.async_load(
                p.weight,
                [0, column_tile * p.BN, start],
                bar,
                p.weight_buffers.index(slot),
            )
            tma.async_load(
                p.weight_scale,
                [0, column_tile * (p.BN // 128), start // 128, 0, 0],
                bar,
                p.weight_scale_buffers.index(slot),
            )
            count += 1


@gluon.jit
def _mma_partition(p):
    TOTAL: gl.constexpr = gl.cdiv(p.M, p.BM) * gl.cdiv(p.HIDDEN, p.BN)
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
            left_scale = p.activation_scale_buffers.index(slot).reshape(
                (p.BM // 128, p.BK // 128, 32, 4, 4)
            )
            left_scale = left_scale.permute((0, 3, 2, 1, 4)).reshape((p.BM, p.BK // 32))
            right_scale = p.weight_scale_buffers.index(slot).reshape(
                (2 * p.BN // 128, p.BK // 128, 32, 4, 4)
            )
            right_scale = right_scale.permute((0, 3, 2, 1, 4)).reshape(
                (2 * p.BN, p.BK // 32)
            )
            layout: gl.constexpr = TensorMemoryScalesLayout(
                cga_layout=[[1, 0]] if gl.num_ctas() == 2 else []
            )
            right_layout: gl.constexpr = TensorMemoryScalesLayout(
                cga_layout=[[0, 0]] if gl.num_ctas() == 2 else []
            )
            left_tmem = allocate_tensor_memory(gl.uint8, (p.BM, p.BK // 32), layout)
            right_tmem = allocate_tensor_memory(
                gl.uint8, (2 * p.BN, p.BK // 32), right_layout
            )
            tcgen05_copy(left_scale, left_tmem)
            tcgen05_copy(right_scale, right_tmem)
            tcgen05_mma_scaled(
                p.activation_buffers.index(slot),
                p.weight_buffers.index(slot).reshape((2 * p.BN, p.BK)).permute((1, 0)),
                accumulator,
                left_tmem,
                right_tmem,
                "e4m3",
                "e4m3",
                use_acc=use_acc,
            )
            tcgen05_commit(
                p.load_empty.index(slot),
                descs=[p.activation_buffers.index(slot), p.weight_buffers.index(slot)],
            )
            load_count += 1
            use_acc = True
        tcgen05_commit(
            p.acc_ready.index(acc_slot),
            descs=[p.activation_buffers.index(0), p.weight_buffers.index(0)],
        )
        tile_count += 1


@gluon.jit
def _store(destination, staging, value, row, column, store_count):
    tma.store_wait(1)
    buffer = staging.index(store_count % 2)
    buffer.store(value)
    fence_async_shared()
    tma.async_store(destination, [row, column], buffer)
    return store_count + 1


@gluon.jit
def _epilogue_partition(p):
    TOTAL: gl.constexpr = gl.cdiv(p.M, p.BM) * gl.cdiv(p.HIDDEN, p.BN)
    if p.TMA_STORES:
        staging = gl.allocate_shared_memory(
            gl.bfloat16, (2, p.BM, p.BN), p.output.layout
        )
    tile_count = 0
    store_count = 0
    for tile in range(gl.program_id(0), TOTAL, gl.num_programs(0)):
        row_tile, column_tile = _coordinates(p, tile)
        acc_slot = tile_count % p.ACC_BUFFERS
        mbarrier.wait(p.acc_ready.index(acc_slot), (tile_count // p.ACC_BUFFERS) % 2)
        if gl.num_warps() >= 8:
            accumulator = p.accumulators.index(acc_slot)
            gate_accumulator = accumulator.slice(0, p.BN)
            up_accumulator = accumulator.slice(p.BN, p.BN)
            gate = gate_accumulator.load(
                gate_accumulator.get_reg_layout(instr_variant=p.LOAD_VARIANT)
            ).to(gl.bfloat16)
            up = up_accumulator.load(
                up_accumulator.get_reg_layout(instr_variant=p.LOAD_VARIANT)
            ).to(gl.bfloat16)
        else:
            accumulator = p.accumulators.index(acc_slot)
            values = accumulator.load(
                accumulator.get_reg_layout(instr_variant=p.LOAD_VARIANT)
            ).to(gl.bfloat16)
            gate, up = values.reshape((p.BM, 2, p.BN)).permute((0, 2, 1)).split()
        mbarrier.arrive(p.acc_empty.index(acc_slot), count=1)
        if p.COALESCED:
            store_layout: gl.constexpr = (
                _EPILOGUE_LAYOUT_8 if gl.num_warps() == 8 else _EPILOGUE_LAYOUT_4
            )
            gate = gl.convert_layout(gate, store_layout)
            up = gl.convert_layout(up, store_layout)
        row, column = row_tile * p.BM, column_tile * p.BN
        if p.SAVE_INTERMEDIATES and p.TMA_STORES:
            store_count = _store(p.gate_up, staging, gate, row, column, store_count)
            store_count = _store(
                p.gate_up, staging, up, row, column + p.HIDDEN, store_count
            )
        # The native torchtitan::silu_and_mul consumes the BF16 GEMM result,
        # computes gate * sigmoid(gate) * up in FP32, and rounds only the output.
        gate_f32, up_f32 = gate.to(gl.float32), up.to(gl.float32)
        sigmoid = 1.0 / (1.0 + gl.exp(-gate_f32))
        output = (gate_f32 * sigmoid * up_f32).to(gl.bfloat16)
        if p.TMA_STORES:
            store_count = _store(p.output, staging, output, row, column, store_count)
        else:
            layout: gl.constexpr = gate.type.layout
            rows = row + gl.arange(0, p.BM, layout=gl.SliceLayout(1, layout))
            columns = column + gl.arange(0, p.BN, layout=gl.SliceLayout(0, layout))
            if p.SAVE_INTERMEDIATES:
                packed_offset = rows[:, None] * (2 * p.HIDDEN) + columns[None, :]
                gl.store(p.gate_up_pointer + packed_offset, gate)
                gl.store(p.gate_up_pointer + packed_offset + p.HIDDEN, up)
            gl.store(
                p.output_pointer + rows[:, None] * p.HIDDEN + columns[None, :], output
            )
        tile_count += 1
    if p.TMA_STORES:
        tma.store_wait(0)


@gluon.jit
def _kernel(
    activation,
    weight,
    activation_scale,
    weight_scale,
    gate_up,
    output,
    gate_up_pointer,
    output_pointer,
    M: gl.constexpr,
    HIDDEN: gl.constexpr,
    K: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    BUFFERS: gl.constexpr,
    ACC_BUFFERS: gl.constexpr,
    SAVE_INTERMEDIATES: gl.constexpr,
    TMA_STORES: gl.constexpr,
    LOAD_VARIANT: gl.constexpr,
    COALESCED: gl.constexpr,
    row_output,
    column_output,
    row_scale,
    column_scale,
    QUANTIZE_OUTPUT: gl.constexpr,
    EPILOGUE_LAYOUT: gl.constexpr,
    TILE_LAYOUT: gl.constexpr,
    ROW_LAYOUT: gl.constexpr,
    COL_LAYOUT: gl.constexpr,
    ROW_SCALE_LAYOUT: gl.constexpr,
    COL_SCALE_LAYOUT: gl.constexpr,
):
    activation_buffers = gl.allocate_shared_memory(
        activation.dtype, [BUFFERS] + activation.block_type.shape, activation.layout
    )
    weight_buffers = gl.allocate_shared_memory(
        weight.dtype, [BUFFERS] + weight.block_type.shape, weight.layout
    )
    activation_scale_buffers = gl.allocate_shared_memory(
        activation_scale.dtype,
        [BUFFERS] + activation_scale.block_type.shape,
        activation_scale.layout,
    )
    weight_scale_buffers = gl.allocate_shared_memory(
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
        (ACC_BUFFERS, BM, 2 * BN),
        TensorMemoryLayout(
            [BM // gl.num_ctas(), 2 * BN],
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
        activation,
        weight,
        activation_scale,
        weight_scale,
        gate_up,
        output,
        gate_up_pointer,
        output_pointer,
        activation_buffers,
        weight_buffers,
        activation_scale_buffers,
        weight_scale_buffers,
        load_empty,
        load_ready,
        accumulators,
        acc_empty,
        acc_ready,
        M,
        HIDDEN,
        K,
        BM,
        BN,
        BK,
        BUFFERS,
        ACC_BUFFERS,
        SAVE_INTERMEDIATES,
        TMA_STORES,
        LOAD_VARIANT,
        COALESCED,
    )
    if QUANTIZE_OUTPUT:
        gl.warp_specialize(
            [
                (
                    _quantized_epilogue,
                    (
                        p,
                        row_output,
                        column_output,
                        row_scale,
                        column_scale,
                        EPILOGUE_LAYOUT,
                        TILE_LAYOUT,
                        ROW_LAYOUT,
                        COL_LAYOUT,
                        ROW_SCALE_LAYOUT,
                        COL_SCALE_LAYOUT,
                    ),
                ),
                (_mma_partition, (p,)),
                (_load_partition, (p,)),
            ],
            [1, 1],
            [24, 24],
        )
    else:
        gl.warp_specialize(
            [
                (_epilogue_partition, (p,)),
                (_mma_partition, (p,)),
                (_load_partition, (p,)),
            ],
            [1, 1],
            [24, 24],
        )


def shared_w13_swiglu(
    activation,
    weight,
    activation_scale,
    weight_scale,
    *,
    save_intermediates=True,
    block_m=None,
    block_n=128,
    block_k=128,
    buffers=None,
    acc_buffers=1,
    maxnreg=128,
    ctas_per_sm=1,
    tma_stores=False,
    num_ctas=None,
    grid_size=None,
    num_warps=None,
    load_variant="32x32b",
    coalesced=None,
    quantize_output=False,
    kernel_metadata=None,
):
    """Return the activation and the exact BF16 packed values needed by backward."""
    rows, reduction = activation.shape
    hidden = weight.shape[0] // 2
    num_ctas = (2 if rows % 256 == 0 else 1) if num_ctas is None else num_ctas
    num_warps = (8 if num_ctas == 2 else 4) if num_warps is None else num_warps
    coalesced = (
        num_ctas == 2 and num_warps in (4, 8) if coalesced is None else coalesced
    )
    if coalesced and (num_ctas != 2 or num_warps not in (4, 8)):
        raise ValueError(
            "coalesced shared SwiGLU requires two CTAs and four/eight epilogue warps"
        )
    if buffers is None:
        buffers = (
            (4 if tma_stores or quantize_output else 6)
            if num_ctas == 2
            else (3 if tma_stores or quantize_output else 4)
        )
    block_m = 128 * num_ctas if block_m is None else block_m
    if (
        num_ctas not in (1, 2)
        or block_m != 128 * num_ctas
        or block_n != 128
        or block_k not in (128, 256)
    ):
        raise ValueError(
            "shared SwiGLU requires 128x128 half-tiles and K tiles of 128/256"
        )
    if rows % block_m or hidden % block_n or reduction % block_k:
        raise ValueError("shared SwiGLU tile dimensions must divide the matrices")
    output = torch.empty((rows, hidden), device=activation.device, dtype=torch.bfloat16)
    gate_up = torch.empty(
        (rows, 2 * hidden) if save_intermediates else (0,),
        device=activation.device,
        dtype=torch.bfloat16,
    )
    total_tiles = (rows // block_m) * (hidden // block_n)
    clusters = (
        torch.cuda.get_device_properties(activation.device).multi_processor_count
        * ctas_per_sm
        // num_ctas
    )
    # Balance the persistent grid's final wave: 256 tiles use 64 paired CTAs
    # on GB300, instead of leaving a partially occupied fourth wave.
    balanced_clusters = triton.cdiv(total_tiles, triton.cdiv(total_tiles, clusters))
    grid = (min(grid_size or balanced_clusters, total_tiles),)
    weight_matrix = weight.view(2, hidden, reduction)
    weight_shape = [2, block_n, block_k]
    weight_descriptor = TensorDescriptor.from_tensor(
        weight_matrix,
        weight_shape,
        gl.NVMMASharedLayout.get_default_for(
            weight_shape,
            gl.float8e4nv,
            cga_layout=[[1, 0, 0]] if num_ctas == 2 else [],
        ),
    )
    weight_scales = weight_scale.view(torch.uint8).view(
        2, hidden // 128, reduction // 128, 2, 256
    )
    weight_scale_descriptor = TensorDescriptor.from_tensor(
        weight_scales,
        [2, block_n // 128, block_k // 128, 2, 256],
        gl.NVMMASharedLayout(
            swizzle_byte_width=0,
            element_bitwidth=8,
            rank=5,
            cga_layout=[[0, 0, 0, 0, 0]] if num_ctas == 2 else [],
        ),
    )
    activation_descriptor = _matrix(activation, rows, reduction, block_m, block_k)
    activation_scale_descriptor = _scales(
        activation_scale, rows, reduction, block_m, block_k
    )
    gate_up_descriptor = _matrix(
        gate_up if save_intermediates else output,
        rows,
        2 * hidden if save_intermediates else hidden,
        block_m,
        block_n,
        dtype=gl.bfloat16,
    )
    output_descriptor = _matrix(
        output, rows, hidden, block_m, block_n, dtype=gl.bfloat16
    )
    if num_ctas == 2:
        activation_descriptor = replace(
            activation_descriptor,
            layout=replace(activation_descriptor.layout, cga_layout=[[1, 0]]),
        )
        activation_scale_descriptor = replace(
            activation_scale_descriptor,
            layout=replace(
                activation_scale_descriptor.layout, cga_layout=[[0, 1, 0, 0, 0]]
            ),
        )
        gate_up_descriptor = replace(
            gate_up_descriptor,
            layout=replace(gate_up_descriptor.layout, cga_layout=[[1, 0]]),
        )
        output_descriptor = replace(
            output_descriptor,
            layout=replace(output_descriptor.layout, cga_layout=[[1, 0]]),
        )
    row_descriptor = column_descriptor = output_descriptor
    row_scale = column_scale = activation_scale
    epilogue_layout = tile_layout = row_layout = col_layout = None
    row_scale_layout = col_scale_layout = None
    if quantize_output:
        if tma_stores:
            raise ValueError("quantized SwiGLU uses its own staged MXFP8 stores")
        row = activation.new_empty((rows, hidden))
        column = activation.new_empty((hidden, rows))
        row_scale = activation_scale.new_empty((rows * hidden // 32,))
        column_scale = torch.empty_like(row_scale)
        row_descriptor = _matrix(row, rows, hidden, block_m, 64)
        column_descriptor = _matrix(column, hidden, rows, 64, block_m)
        cga_layout = [[1, 0]] if num_ctas == 2 else []
        if num_ctas == 2:
            row_descriptor = replace(
                row_descriptor,
                layout=replace(row_descriptor.layout, cga_layout=cga_layout),
            )
            column_descriptor = replace(
                column_descriptor,
                layout=gl.NVMMASharedLayout.get_default_for(
                    [64, block_m],
                    gl.float8e4nv,
                    cga_layout=[[0, 1]],
                ),
            )
        epilogue_layout = gl.BlockedLayout(
            [1, 2], [4, 8], [4, num_warps // 4], [1, 0], cga_layout=cga_layout
        )
        tile_layout = gl.NVMMASharedLayout(
            swizzle_byte_width=128, element_bitwidth=16, rank=2, cga_layout=cga_layout
        )
        row_warps = block_m // (32 * num_ctas)
        column_warps = num_warps // row_warps
        row_layout = gl.BlockedLayout(
            [1, 32 // column_warps],
            [32, 1],
            [row_warps, column_warps],
            [1, 0],
            cga_layout=cga_layout,
        )
        col_layout = gl.BlockedLayout(
            [32 // column_warps, 1],
            [column_warps, 32 // column_warps],
            [row_warps, column_warps],
            [0, 1],
            cga_layout=cga_layout,
        )
        row_scale_layout = gl.BlockedLayout(
            [1, 2], [32, 1], [row_warps, column_warps], [1, 0], cga_layout=cga_layout
        )
        col_scale_layout = gl.BlockedLayout(
            [1, 2], [1, 32], [row_warps, column_warps], [1, 0], cga_layout=cga_layout
        )
    compiled = _kernel[grid](
        activation_descriptor,
        weight_descriptor,
        activation_scale_descriptor,
        weight_scale_descriptor,
        gate_up_descriptor,
        output_descriptor,
        gate_up,
        output,
        rows,
        hidden,
        reduction,
        block_m,
        block_n,
        block_k,
        buffers,
        acc_buffers,
        save_intermediates,
        tma_stores,
        load_variant,
        coalesced,
        row_descriptor,
        column_descriptor,
        row_scale.view(torch.uint8),
        column_scale.view(torch.uint8),
        quantize_output,
        epilogue_layout,
        tile_layout,
        row_layout,
        col_layout,
        row_scale_layout,
        col_scale_layout,
        num_warps=num_warps,
        num_ctas=num_ctas,
        maxnreg=maxnreg,
        enable_fp_fusion=True,
    )
    if kernel_metadata is not None:
        kernel_metadata.update(
            registers=compiled.n_regs,
            spills=compiled.n_spills,
            shared_bytes=compiled.metadata.shared,
            threads=compiled.metadata.num_warps * 32,
        )
    if quantize_output:
        return output, gate_up, row, column.t(), row_scale, column_scale
    return output, gate_up
