# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 specialization of the generic FSDP unsharded-tensor lifecycle.

Tensor shape suffixes:
    E: experts
    N: output features
    K: input features
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchao.prototype.mx_formats.kernels import (
    triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale,
)

from .._fsdp_tensor import _ShardedFSDPTensor
from ._common import _MXFP8_BLOCK_SIZE


# Everything here is internal to the MXFP8 component; nothing is re-exported.
__all__: list[str] = []


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


@dataclass(frozen=True, slots=True)
class _MXFP8GroupedWeightOperands:
    """The independent MXFP8 tensors owned by one grouped-expert unshard.

    Each expert is quantized with square 32x32 tiles: E4M3 qdata and one E8M0
    scale per tile. ``swizzled`` on the scales names their memory layout, not
    their format -- they are pre-arranged into the blocked grid the grouped
    GEMM wants, so no rearrange is needed at the call site.

    Unlike the dense linear case, FPROP and DGRAD cannot share one qdata
    allocation. Both grouped GEMMs require a right operand that is
    column-major within each expert, so the ``(E, K, N)`` operand consumed by
    FPROP and the ``(E, N, K)`` operand consumed by DGRAD are different
    physical layouts. Square 32x32 tiles make their *values* identical, so
    what we duplicate is purely layout.

    ``torch._scaled_grouped_mm_v2`` takes a ``contraction_dim``, but it does
    not lift this today: passing one allocation for both rejects with
    "Expected mat2 to be transposed", and a non-default contraction dim with
    "Currently contraction dims must be (-1, -2) only". If PyTorch accepts a
    strided right operand, DGRAD can read a transposed view of the FPROP
    qdata as the dense path already does, dropping one E*N*K-byte allocation
    per expert weight.
    """

    weight_qdata_fprop_EKN: torch.Tensor  # noqa: N815
    weight_scale_fprop_swizzled: torch.Tensor
    weight_qdata_dgrad_ENK: torch.Tensor  # noqa: N815
    weight_scale_dgrad_swizzled: torch.Tensor


def _quantize_mxfp8_grouped_weight(
    weight_ENK: torch.Tensor,
) -> _MXFP8GroupedWeightOperands:
    """Quantize each expert weight using fixed square 32x32 scale tiles."""
    if weight_ENK.ndim != 3:
        raise ValueError(
            "MXFP8 grouped 32x32 weight quantization requires a 3D weight, "
            f"got {weight_ENK.ndim} dimensions."
        )
    if weight_ENK.dtype != torch.bfloat16:
        raise ValueError(
            "MXFP8 grouped 32x32 weight quantization requires BF16 weights, "
            f"got {weight_ENK.dtype}."
        )
    if any(size % _MXFP8_BLOCK_SIZE for size in weight_ENK.shape[1:]):
        raise ValueError(
            "MXFP8 grouped 32x32 weight quantization requires both expert "
            f"matrix dimensions divisible by {_MXFP8_BLOCK_SIZE}, got "
            f"{tuple(weight_ENK.shape)}."
        )

    # TODO: replace this per-expert loop with a single fused grouped cast once
    # TorchAO's 3D 32x32 kernel emits both swizzled scale layouts from one
    # pass. The 2D kernel already does, so quantizing expert by expert keeps
    # the layout contract identical to MXFP8Linear at the cost of E launches.
    # The scale swizzle blocks rows in groups of 128, so the experts cannot be
    # flattened into one 2D cast unless the output dimension is a multiple of
    # 128, which we do not require.
    weight_qdata_per_expert_NK = []  # noqa: N806
    weight_scale_fprop_per_expert = []
    weight_scale_dgrad_per_expert = []
    for weight_NK in weight_ENK:
        (
            weight_qdata_NK,
            weight_scale_fprop_swizzled,
            weight_scale_dgrad_swizzled,
        ) = triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale(weight_NK.contiguous())
        weight_qdata_per_expert_NK.append(weight_qdata_NK)
        weight_scale_fprop_per_expert.append(weight_scale_fprop_swizzled.reshape(-1))
        weight_scale_dgrad_per_expert.append(weight_scale_dgrad_swizzled.reshape(-1))

    # ``torch._scaled_grouped_mm`` requires a right operand that is column-major
    # within each expert. Stacking gives a row-major (E, N, K) buffer, whose
    # transpose is the column-major (E, K, N) operand FPROP needs; DGRAD needs
    # a column-major (E, N, K) operand, which is a second physical layout.
    weight_qdata_ENK = torch.stack(weight_qdata_per_expert_NK)
    weight_qdata_fprop_EKN = weight_qdata_ENK.transpose(-2, -1)
    weight_qdata_dgrad_ENK = weight_qdata_fprop_EKN.contiguous().transpose(-2, -1)
    return _MXFP8GroupedWeightOperands(
        weight_qdata_fprop_EKN=weight_qdata_fprop_EKN,
        weight_scale_fprop_swizzled=torch.stack(weight_scale_fprop_per_expert),
        weight_qdata_dgrad_ENK=weight_qdata_dgrad_ENK,
        weight_scale_dgrad_swizzled=torch.stack(weight_scale_dgrad_per_expert),
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
    (
        weight_qdata_dgrad_NK,
        weight_scale_fprop_swizzled,
        weight_scale_dgrad_swizzled,
    ) = triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale(weight_NK.contiguous())
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
        logical_weight: torch.Tensor,
        out: _MXFP8LinearOperands | None = None,
    ) -> _MXFP8LinearOperands:
        operands = _quantize_mxfp8_weight(logical_weight)
        if out is None:
            return operands
        out.weight_qdata_dgrad_NK.copy_(operands.weight_qdata_dgrad_NK)
        out.weight_scale_fprop_swizzled.copy_(operands.weight_scale_fprop_swizzled)
        out.weight_scale_dgrad_swizzled.copy_(operands.weight_scale_dgrad_swizzled)
        return out


class _GroupedExpertsShardedTensorWithMXFP8Compute(_ShardedFSDPTensor):
    """The persistent BF16 grouped-expert parameter; quantizes on unshard.

    FSDP shards, all-gathers, reduces gradients into, and checkpoints this
    BF16 parameter; the MXFP8 operands it produces live on a
    ``_UnshardedFSDPTensor`` for one unshard lifetime. Four of them here rather
    than the three a 2D weight needs -- see ``_MXFP8GroupedWeightOperands``
    for why FPROP and DGRAD cannot share a qdata allocation.
    """

    def _build_operands(
        self,
        logical_weight: torch.Tensor,
        out: _MXFP8GroupedWeightOperands | None = None,
    ) -> _MXFP8GroupedWeightOperands:
        operands = _quantize_mxfp8_grouped_weight(logical_weight)
        if out is None:
            return operands
        out.weight_qdata_fprop_EKN.copy_(operands.weight_qdata_fprop_EKN)
        out.weight_scale_fprop_swizzled.copy_(operands.weight_scale_fprop_swizzled)
        out.weight_qdata_dgrad_ENK.copy_(operands.weight_qdata_dgrad_ENK)
        out.weight_scale_dgrad_swizzled.copy_(operands.weight_scale_dgrad_swizzled)
        return out
