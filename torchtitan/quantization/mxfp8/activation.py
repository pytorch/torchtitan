# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Activation casts with directly consumable blocked MXFP8 scales."""

from inspect import signature

import torch


__all__: list[str] = []


def _require_cutedsl_activation_quantizers() -> None:
    """Check the opted-in AO kernel contract before model execution."""
    try:
        from torchao.prototype.moe_training.kernels.mxfp8 import quant
    except ImportError as error:
        raise ImportError(
            "CuTeDSL activation casts require torchao with fused dual-axis MXFP8 quantizers"
        ) from error
    required = (
        "mxfp8_quantize_2d_1x32_32x1_cutedsl",
        "mxfp8_quantize_2d_1x32_cutedsl",
        "mxfp8_quantize_2d_32x1_cutedsl",
    )
    if any(not hasattr(quant, name) for name in required):
        raise ImportError(
            "CuTeDSL activation casts require torchao with fused dual-axis MXFP8 quantizers"
        )
    offs_parameter = signature(
        quant.mxfp8_quantize_2d_1x32_32x1_cutedsl
    ).parameters.get("offs")
    if offs_parameter is None or offs_parameter.default is not None:
        raise ImportError(
            "CuTeDSL activation casts require torchao's dense None-offset MXFP8 API"
        )
    if not quant._mxfp8_cutedsl_kernels_available:
        raise ImportError(
            "CuTeDSL activation casts require SM10.x, CUDA 12.8+, "
            "nvidia-cutlass-dsl and apache-tvm-ffi"
        )


def _quantize_activation(
    x_MK: torch.Tensor, *, rowwise: bool, colwise: bool, backend: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return row/column operands and blocked E8M0 scales without extra casts.

    CuTeDSL is explicit opt-in for 128-aligned BF16 matrices. The fused dense
    API returns both operands and compact blocked scales without offsets.
    Both backends explicitly request RCEIL to match the 32x32 weight cast.

    Args:
        x_MK: Contiguous two-dimensional BF16 activation or output gradient.
        rowwise: Produce the rowwise FPROP/DGRAD operand.
        colwise: Produce the columnwise WGRAD operand.
        backend: Validated ``cuda`` or ``cutedsl`` configuration choice.

    Returns:
        Rowwise qdata, columnwise qdata, and their blocked E8M0 scales.
        Unrequested operands are empty tensors.
    """
    if backend == "cuda":
        from torchao.prototype.mx_formats.kernels import (
            mxfp8_quantize_cuda,
            triton_mx_block_rearrange,
        )

        row, col, row_scale, col_scale = mxfp8_quantize_cuda(
            x_MK, rowwise=rowwise, colwise=colwise, scaling_mode="rceil"
        )
        return (
            row,
            col,
            triton_mx_block_rearrange(row_scale) if rowwise else row_scale,
            triton_mx_block_rearrange(col_scale) if colwise else col_scale,
        )
    if x_MK.dtype != torch.bfloat16 or any(size % 128 for size in x_MK.shape):
        raise ValueError(
            "CuTeDSL activation casts require BF16 with both matrix axes "
            f"divisible by 128; got {x_MK.dtype}, {tuple(x_MK.shape)}. "
            "Use activation_quantization_backend='cuda' for other shapes."
        )
    from torchao.prototype.moe_training.kernels.mxfp8.quant import (
        mxfp8_quantize_2d_1x32_32x1_cutedsl,
        mxfp8_quantize_2d_1x32_cutedsl,
        mxfp8_quantize_2d_32x1_cutedsl,
    )

    if rowwise and colwise:
        row, row_scale, col, col_scale = mxfp8_quantize_2d_1x32_32x1_cutedsl(
            x_MK, scaling_mode="rceil"
        )
        return (
            row,
            col.t(),
            row_scale.flatten().view(torch.float8_e8m0fnu),
            col_scale.flatten().view(torch.float8_e8m0fnu),
        )
    empty_data = x_MK.new_empty(0, dtype=torch.float8_e4m3fn)
    empty_scale = x_MK.new_empty(0, dtype=torch.float8_e8m0fnu)
    if rowwise:
        row, row_scale = mxfp8_quantize_2d_1x32_cutedsl(x_MK, scaling_mode="rceil")
        return (
            row,
            empty_data,
            row_scale.flatten().view(torch.float8_e8m0fnu),
            empty_scale,
        )
    col, col_scale = mxfp8_quantize_2d_32x1_cutedsl(x_MK, scaling_mode="rceil")
    return empty_data, col, empty_scale, col_scale.flatten().view(torch.float8_e8m0fnu)
