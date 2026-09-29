# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 specialization of the generic FSDP unsharded-tensor lifecycle."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torchao.prototype.mx_formats.utils import from_blocked, to_blocked

from .._fsdp_tensor import _ShardedFSDPTensor


# Everything here is internal to the MXFP8 component; nothing is re-exported.
__all__: list[str] = []

# One E8M0 scale per 32 elements along the scaled axis, per the OCP
# microscaling spec. Weight quantization uses a *square* 32x32 tile --
# side equal to the block size is the only shape that is a valid MX
# group along both axes at once, which is what makes it transpose-
# invariant and lets FPROP and DGRAD share one qdata allocation.
_MXFP8_BLOCK_SIZE = 32


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


@torch.compile
def _transpose_mxfp8_weight_scales(
    scale_fprop_swizzled_bytes: torch.Tensor,
    num_weight_rows: int,
    num_weight_cols: int,
) -> torch.Tensor:
    """Rearrange 32x32 tile scales for a transposed weight operand."""
    scale_fprop = from_blocked(
        scale_fprop_swizzled_bytes,
        num_weight_rows,
        num_weight_cols // _MXFP8_BLOCK_SIZE,
    )
    # Square-tile scales are repeated across each group of 32 rows. Keep one
    # per tile, transpose the tile grid, then expand over the new row axis.
    tile_scales = scale_fprop[::_MXFP8_BLOCK_SIZE]
    scale_dgrad = tile_scales.t().repeat_interleave(_MXFP8_BLOCK_SIZE, dim=0)
    return to_blocked(scale_dgrad).view(
        scale_fprop_swizzled_bytes.shape[1],
        scale_fprop_swizzled_bytes.shape[0],
        32,
        16,
    )


def _quantize_mxfp8_weight(weight_NK: torch.Tensor) -> _MXFP8LinearOperands:
    """Quantize a BF16 weight using fixed square 32x32 scale tiles."""
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
    weight_qdata_dgrad_NK, weight_scale_fprop_swizzled = F.quantize_tensor(
        weight_NK,
        qdata_dtype=torch.float8_e4m3fn,
        scaling_algorithm=F.ScalingAlgorithm.MXFP_E8M0_RU,
        scaling_type=F.ScalingType.BlockWise1x32,
        swizzle_type=F.SwizzleType.SWIZZLE_32_4_4,
        scaling_type_use_square_block_size=True,
    )
    # Inductor cannot lower an E8M0-to-uint8 view combined with the padded
    # scale layout transform, so keep the dtype views outside compilation.
    weight_scale_dgrad_swizzled = _transpose_mxfp8_weight_scales(
        weight_scale_fprop_swizzled.view(torch.uint8), *weight_NK.shape
    ).view(torch.float8_e8m0fnu)
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
