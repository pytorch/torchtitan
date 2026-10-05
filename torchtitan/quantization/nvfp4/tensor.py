# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NVFP4 specialization of the generic FSDP unsharded-tensor lifecycle."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchao.prototype.moe_training.nvfp4_training.nvfp4_linear import (
    _resolve_use_cutedsl,
    _weight_quantize_2d,
)
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference

from .._fsdp_tensor import _ShardedFSDPTensor


__all__: list[str] = []

_NVFP4_ALIGNMENT = 128


@dataclass(frozen=True, slots=True)
class _NVFP4LinearOperands:
    """The independent NVFP4 tensors owned by one FSDP unshard lifetime."""

    weight_qdata_fprop: torch.Tensor
    weight_scale_fprop: torch.Tensor
    weight_qdata_dgrad: torch.Tensor
    weight_scale_dgrad: torch.Tensor
    weight_amax: torch.Tensor


def _quantize_nvfp4_weight(weight_NK: torch.Tensor) -> _NVFP4LinearOperands:
    """Quantize a BF16 weight in both orientations for FPROP and DGRAD.

    TorchAO's ``nvfp4_linear._weight_quantize_2d`` owns the quantization; this
    wrapper names its outputs for TorchTitan's FSDP operand cache.
    """
    if weight_NK.ndim != 2:
        raise ValueError(
            f"NVFP4 weight quantization requires a 2D weight, got {weight_NK.ndim} dimensions."
        )
    if weight_NK.dtype != torch.bfloat16:
        raise ValueError(
            f"NVFP4 weight quantization requires BF16 weights, got {weight_NK.dtype}."
        )
    if any(size % _NVFP4_ALIGNMENT for size in weight_NK.shape):
        raise ValueError(
            "NVFP4 weight quantization requires both matrix dimensions divisible "
            f"by {_NVFP4_ALIGNMENT}, got {tuple(weight_NK.shape)}."
        )

    use_cutedsl = _resolve_use_cutedsl(KernelPreference.AUTO)
    (
        weight_qdata_fprop,
        weight_scale_fprop,
        _weight_global_scale,
        weight_qdata_dgrad,
        weight_scale_dgrad,
        weight_amax,
    ) = _weight_quantize_2d(weight_NK, use_cutedsl)
    return _NVFP4LinearOperands(
        weight_qdata_fprop=weight_qdata_fprop,
        weight_scale_fprop=weight_scale_fprop,
        weight_qdata_dgrad=weight_qdata_dgrad,
        weight_scale_dgrad=weight_scale_dgrad,
        weight_amax=weight_amax,
    )


class _LinearShardedTensorWithNVFP4Compute(_ShardedFSDPTensor):
    """Persistent high-precision parameter that quantizes on FSDP unshard."""

    def _build_operands(
        self,
        logical_tensor: torch.Tensor,
        out: _NVFP4LinearOperands | None = None,
    ) -> _NVFP4LinearOperands:
        if logical_tensor.ndim > 2 and logical_tensor.shape[-2] % _NVFP4_ALIGNMENT:
            raise ValueError(
                "NVFP4 requires local matrix out_features divisible by "
                f"{_NVFP4_ALIGNMENT}; got {logical_tensor.shape[-2]}. Adjust "
                "the Linear out_features or TP degree."
            )
        operands = _quantize_nvfp4_weight(logical_tensor.flatten(0, -2))
        if out is None:
            return operands
        out.weight_qdata_fprop.copy_(operands.weight_qdata_fprop)
        out.weight_scale_fprop.copy_(operands.weight_scale_fprop)
        out.weight_qdata_dgrad.copy_(operands.weight_qdata_dgrad)
        out.weight_scale_dgrad.copy_(operands.weight_scale_dgrad)
        out.weight_amax.copy_(operands.weight_amax)
        return out
