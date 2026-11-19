# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 activation casts and FSDP-managed weight quantization.

Tensor shape suffixes:
    M: flattened token rows
    N: output features
    K: input features
"""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature

import torch

from torchao.prototype.mx_formats.kernels import (
    triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale,
)

from .._fsdp_tensor import _ShardedFSDPTensor


# Everything here is internal to the MXFP8 component; nothing is re-exported.
__all__: list[str] = []

# One E8M0 scale per 32 elements along the scaled axis, per the OCP
# microscaling spec. Weight quantization uses a *square* 32x32 tile --
# side equal to the block size is the only shape that is a valid MX
# group along both axes at once, which is what makes it transpose-
# invariant and lets FPROP and DGRAD share one qdata allocation.
_MXFP8_BLOCK_SIZE = 32
# The square weight cast has a fixed RCEIL contract; activations must match it.
_MXFP8_SCALING_MODE = "rceil"


def _require_cutedsl_activation_quantizers() -> None:
    """Validate the dense AO CuTeDSL API and runtime before model execution."""
    try:
        from torchao.prototype.moe_training.kernels.mxfp8 import quant
    except ImportError as error:
        raise ImportError(
            "MXFP8 activation casts require torchao with fused dual-axis MXFP8 "
            "quantizers; install a current torchao nightly."
        ) from error
    required = (
        "mxfp8_quantize_2d_1x32_32x1_cutedsl",
        "mxfp8_quantize_2d_1x32_cutedsl",
        "mxfp8_quantize_2d_32x1_cutedsl",
    )
    if any(not hasattr(quant, name) for name in required):
        raise ImportError(
            "MXFP8 activation casts require torchao with fused dual-axis MXFP8 "
            "quantizers; install a current torchao nightly."
        )
    offs_parameter = signature(
        quant.mxfp8_quantize_2d_1x32_32x1_cutedsl
    ).parameters.get("offs")
    if offs_parameter is None or offs_parameter.default is not None:
        raise ImportError(
            "MXFP8 activation casts require torchao's dense None-offset MXFP8 API; "
            "install a current torchao nightly."
        )
    if not quant._mxfp8_cutedsl_kernels_available:
        raise ImportError(
            "MXFP8 activation casts require SM10.x, CUDA 12.8+, and "
            "the CuTeDSL runtime; install nvidia-cutlass-dsl and apache-tvm-ffi."
        )


def _quantize_activation(
    x_MK: torch.Tensor, *, rowwise: bool, colwise: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return MXFP8 operands with blocked E8M0 scales for the requested axes.

    CuTeDSL handles positive, 128-aligned BF16 matrices. Other matrix shapes
    use the existing CUDA cast and scale rearrangement, without padding a
    larger GEMM. Selection uses the actual operand, so a linear's forward and
    backward can use different kernels. Missing CuTe dependencies are errors,
    not a reason to select CUDA. All casts use the weight's RCEIL policy.

    Args:
        x_MK: Contiguous two-dimensional BF16 activation or output gradient.
        rowwise: Produce the rowwise FPROP/DGRAD operand.
        colwise: Produce the columnwise WGRAD operand.

    Returns:
        Rowwise qdata, columnwise qdata, and their blocked E8M0 scales.
        Unrequested operands are empty tensors.
    """
    if x_MK.ndim != 2 or x_MK.dtype != torch.bfloat16:
        raise ValueError(
            "MXFP8 activation casts require a 2D BF16 matrix; "
            f"got {x_MK.dtype}, {tuple(x_MK.shape)}."
        )
    if not rowwise and not colwise:
        raise ValueError("MXFP8 activation casts require at least one axis.")
    if any(size <= 0 or size % 128 for size in x_MK.shape):
        from torchao.prototype.mx_formats.kernels import (
            mxfp8_quantize_cuda,
            triton_mx_block_rearrange,
        )

        row, col, row_scale, col_scale = mxfp8_quantize_cuda(
            x_MK, rowwise=rowwise, colwise=colwise, scaling_mode=_MXFP8_SCALING_MODE
        )
        return (
            row,
            col,
            triton_mx_block_rearrange(row_scale) if rowwise else row_scale,
            triton_mx_block_rearrange(col_scale) if colwise else col_scale,
        )
    from torchao.prototype.moe_training.kernels.mxfp8.quant import (
        mxfp8_quantize_2d_1x32_32x1_cutedsl,
        mxfp8_quantize_2d_1x32_cutedsl,
        mxfp8_quantize_2d_32x1_cutedsl,
    )

    if rowwise and colwise:
        row, row_scale, col, col_scale = mxfp8_quantize_2d_1x32_32x1_cutedsl(
            x_MK, scaling_mode=_MXFP8_SCALING_MODE
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
        row, row_scale = mxfp8_quantize_2d_1x32_cutedsl(
            x_MK, scaling_mode=_MXFP8_SCALING_MODE
        )
        return (
            row,
            empty_data,
            row_scale.flatten().view(torch.float8_e8m0fnu),
            empty_scale,
        )
    col, col_scale = mxfp8_quantize_2d_32x1_cutedsl(
        x_MK, scaling_mode=_MXFP8_SCALING_MODE
    )
    return empty_data, col, empty_scale, col_scale.flatten().view(torch.float8_e8m0fnu)


@dataclass(frozen=True, slots=True)
class _MXFP8LinearOperands:
    """The independent MXFP8 tensors owned by one FSDP unshard lifetime.

    Everything here is quantized the same way: square 32x32 tiles, E4M3 qdata
    and one E8M0 scale per tile. ``swizzled`` on the scales names their memory
    layout, not their format -- they are pre-arranged into the blocked grid
    ``scaled_mm`` wants for ``SwizzleType.SWIZZLE_32_4_4``, so no rearrange is
    needed at the GEMM. Square tiles are transpose-invariant, so FPROP and
    DGRAD share one qdata allocation and differ only in scale layout.
    """

    weight_qdata_dgrad_NK: torch.Tensor  # noqa: N815
    weight_scale_fprop_swizzled: torch.Tensor
    weight_scale_dgrad_swizzled: torch.Tensor

    @property
    def weight_qdata_fprop_KN(self) -> torch.Tensor:  # noqa: N802
        return self.weight_qdata_dgrad_NK.t()


def _quantize_mxfp8_weight(weight_NK: torch.Tensor) -> _MXFP8LinearOperands:
    """Quantize a BF16 weight using fixed square 32x32 scale tiles."""
    assert _MXFP8_SCALING_MODE == "rceil", "The AO 32x32 weight cast requires RCEIL"
    if weight_NK.ndim != 2:
        raise ValueError(
            "MXFP8 32x32 weight quantization requires a 2D weight, "
            f"got {weight_NK.ndim} dimensions."
        )
    if weight_NK.dtype != torch.bfloat16:
        raise ValueError(
            "MXFP8 32x32 weight quantization requires BF16 weights, "
            f"got {weight_NK.dtype}."
        )
    if any(size % _MXFP8_BLOCK_SIZE for size in weight_NK.shape):
        raise ValueError(
            "MXFP8 32x32 weight quantization requires both matrix dimensions "
            f"divisible by {_MXFP8_BLOCK_SIZE}, got {tuple(weight_NK.shape)}."
        )
    (
        weight_qdata_dgrad_NK,
        weight_scale_fprop_swizzled,
        weight_scale_dgrad_swizzled,
    ) = triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale(weight_NK)
    return _MXFP8LinearOperands(
        weight_qdata_dgrad_NK=weight_qdata_dgrad_NK,
        weight_scale_fprop_swizzled=weight_scale_fprop_swizzled,
        weight_scale_dgrad_swizzled=weight_scale_dgrad_swizzled,
    )


class _LinearShardedTensorWithMXFP8Compute(_ShardedFSDPTensor):
    """The persistent BF16 linear parameter; quantizes to MXFP8 on unshard.

    This is the sharded state only. FSDP shards, all-gathers, reduces
    gradients into, and checkpoints this BF16 parameter. The MXFP8 operands it
    produces live on a ``_UnshardedFSDPTensor`` for one unshard lifetime; that
    holder is generic, so quantization is the only thing a format supplies.
    """

    def _build_operands(
        self,
        logical_tensor: torch.Tensor,
        out: _MXFP8LinearOperands | None = None,
    ) -> _MXFP8LinearOperands:
        if logical_tensor.ndim > 2 and logical_tensor.shape[-2] % _MXFP8_BLOCK_SIZE:
            raise ValueError(
                "MXFP8 requires local matrix out_features divisible by "
                f"{_MXFP8_BLOCK_SIZE}; got {logical_tensor.shape[-2]}. Adjust "
                "the Linear out_features or TP degree so quantization "
                "blocks do not span projection boundaries."
            )
        operands = _quantize_mxfp8_weight(logical_tensor.flatten(0, -2))
        if out is None:
            return operands
        out.weight_qdata_dgrad_NK.copy_(operands.weight_qdata_dgrad_NK)
        out.weight_scale_fprop_swizzled.copy_(operands.weight_scale_fprop_swizzled)
        out.weight_scale_dgrad_swizzled.copy_(operands.weight_scale_dgrad_swizzled)
        return out
