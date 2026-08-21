# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 32x32 weight quantization.

Tensor shape suffixes:
    N: output features
    K: input features

Weights use square 32x32 scale tiles, while activations use standard
one-dimensional MXFP8 scale tiles. Square weight tiles are invariant under
transpose, so FPROP and DGRAD share one qdata allocation.
"""

from dataclasses import dataclass

import torch

from torchao.prototype.mx_formats.kernels import (
    triton_to_mxfp8_32x32_swizzle_dim0_and_dim1,
)


_MXFP8_WEIGHT_TILE_SIZE = 32


@dataclass(frozen=True)
class MXFP8WeightOperands:
    """The weight qdata and blocked scales consumed by FPROP and DGRAD."""

    q_weight_fprop_KN: torch.Tensor  # noqa: N815
    s_weight_fprop_blocked: torch.Tensor
    q_weight_dgrad_NK: torch.Tensor  # noqa: N815
    s_weight_dgrad_blocked: torch.Tensor


def _validate_weight(weight_NK: torch.Tensor) -> None:
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
    if any(size % _MXFP8_WEIGHT_TILE_SIZE for size in weight_NK.shape):
        raise ValueError(
            "MXFP8 32x32 weight quantization requires both matrix dimensions "
            f"divisible by {_MXFP8_WEIGHT_TILE_SIZE}, got {tuple(weight_NK.shape)}."
        )


def quantize_mxfp8_weight(weight_NK: torch.Tensor) -> MXFP8WeightOperands:
    """Quantize a BF16 weight using square 32x32 scale tiles."""
    _validate_weight(weight_NK)
    (
        q_weight_dgrad_NK,
        s_weight_fprop_blocked,
        _,
        s_weight_dgrad_blocked,
    ) = triton_to_mxfp8_32x32_swizzle_dim0_and_dim1(weight_NK.contiguous())
    # The fused kernel also materializes the transposed qdata for an optimized
    # DGRAD layout. Square quantization makes it an exact transpose, so retain
    # one allocation and use views in both GEMMs.
    q_weight_fprop_KN = q_weight_dgrad_NK.t()
    return MXFP8WeightOperands(
        q_weight_fprop_KN=q_weight_fprop_KN,
        s_weight_fprop_blocked=s_weight_fprop_blocked,
        q_weight_dgrad_NK=q_weight_dgrad_NK,
        s_weight_dgrad_blocked=s_weight_dgrad_blocked,
    )


__all__ = [
    "MXFP8WeightOperands",
    "quantize_mxfp8_weight",
]
