# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused Triton kernel for the NVFP4 W4A16 expert-weight QDQ.

One pass over the weight: load a tile, compute each 1x16 block's E4M3 scale,
round to E2M1, dequantize, and store in the weight's dtype. It reproduces
``nvfp4_w4a16._nvfp4_qdq_reference`` bit for bit, which the GPU unit test
checks: every division is IEEE round-to-nearest (``tl.div_rn``, not Triton's
approximate ``/``), the FP8 cast rounds to nearest even, and the E2M1 sign is
copied from the scaled value's sign bit so ``-0.0`` survives.

Imported lazily by ``nvfp4_w4a16`` so CPU-only installs never need Triton.

Shape suffixes: ``E`` experts, ``N`` rows, ``K`` columns (the blocked input
dimension), ``R``/``C`` a tile's rows/columns, ``B`` blocks in a tile row,
``V`` values in one block.
"""

import torch
import triton
import triton.language as tl

_BLOCK_ROWS = 16
_BLOCK_COLS = 256


@triton.jit
def _nvfp4_qdq_kernel(
    x_ptr,
    global_scale_ptr,
    out_ptr,
    num_rows,
    num_cols,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    expert = tl.program_id(2).to(tl.int64)
    rows_R = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols_C = tl.program_id(1) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask_RC = (rows_R[:, None] < num_rows) & (cols_C[None, :] < num_cols)
    offsets_RC = (
        expert * num_rows * num_cols
        + rows_R[:, None].to(tl.int64) * num_cols
        + cols_C[None, :]
    )
    x_RC = tl.load(x_ptr + offsets_RC, mask=mask_RC, other=0.0).to(tl.float32)
    global_scale = tl.load(global_scale_ptr + expert)

    # num_cols is a multiple of 16 and tiles start on block boundaries, so a
    # masked tail never splits a block.
    x_RBV = tl.reshape(x_RC, (BLOCK_ROWS, BLOCK_COLS // 16, 16))
    block_amax_RB = tl.max(tl.abs(x_RBV), axis=2)
    # (amax * global_scale) * fl32(1/6), clamped to [2^-9, 448], rounded to E4M3.
    block_scale_RB = (block_amax_RB * global_scale) * 0.16666666666666666
    block_scale_RB = tl.clamp(
        block_scale_RB, 0.001953125, 448.0, propagate_nan=tl.PropagateNan.ALL
    )
    block_scale_RB = block_scale_RB.to(tl.float8e4nv, fp_downcast_rounding="rtne").to(
        tl.float32
    )

    scaled_RBV = x_RBV * tl.div_rn(global_scale, block_scale_RB)[:, :, None]
    # E2M1 round-to-nearest-even with saturation; see _round_to_e2m1.
    magnitude_RBV = tl.abs(scaled_RBV)
    code_RBV = tl.where(magnitude_RBV > 5.0, 6.0, 4.0)
    code_RBV = tl.where(magnitude_RBV < 3.5, 3.0, code_RBV)
    code_RBV = tl.where(magnitude_RBV <= 2.5, 2.0, code_RBV)
    code_RBV = tl.where(magnitude_RBV < 1.75, 1.5, code_RBV)
    code_RBV = tl.where(magnitude_RBV <= 1.25, 1.0, code_RBV)
    code_RBV = tl.where(magnitude_RBV < 0.75, 0.5, code_RBV)
    code_RBV = tl.where(magnitude_RBV <= 0.25, 0.0, code_RBV)
    sign_RBV = scaled_RBV.to(tl.uint32, bitcast=True) & 0x80000000
    code_RBV = (code_RBV.to(tl.uint32, bitcast=True) | sign_RBV).to(
        tl.float32, bitcast=True
    )

    dequant_RBV = tl.div_rn(code_RBV * block_scale_RB[:, :, None], global_scale)
    dequant_RC = tl.reshape(dequant_RBV, (BLOCK_ROWS, BLOCK_COLS))
    tl.store(
        out_ptr + offsets_RC,
        dequant_RC.to(out_ptr.dtype.element_ty),
        mask=mask_RC,
    )


def nvfp4_qdq_into(
    weight_ENK: torch.Tensor, global_scale_E: torch.Tensor, out_ENK: torch.Tensor
) -> None:
    """Write the NVFP4 QDQ of contiguous ``weight_ENK`` into ``out_ENK``."""
    assert weight_ENK.is_contiguous() and out_ENK.is_contiguous()
    assert out_ENK.shape == weight_ENK.shape
    assert global_scale_E.dtype == torch.float32
    num_experts, num_rows, num_cols = weight_ENK.shape
    grid = (
        triton.cdiv(num_rows, _BLOCK_ROWS),
        triton.cdiv(num_cols, _BLOCK_COLS),
        num_experts,
    )
    _nvfp4_qdq_kernel[grid](
        weight_ENK,
        global_scale_E,
        out_ENK,
        num_rows,
        num_cols,
        BLOCK_ROWS=_BLOCK_ROWS,
        BLOCK_COLS=_BLOCK_COLS,
    )
