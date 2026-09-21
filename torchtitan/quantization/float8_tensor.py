# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Float8 specialization of the generic FSDP unsharded-tensor lifecycle."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchao.float8.config import e4m3_dtype
from torchao.float8.float8_utils import amax_to_scale, to_fp8_saturated

from ._fsdp_tensor import _ShardedFSDPTensor


__all__: list[str] = []

_FLOAT8_ALIGNMENT = 16


@torch.no_grad()
def _quantize_float8(
    tensor: torch.Tensor,
    *,
    reduction_axis: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamically quantize a tensor with TorchAO's Float8 cast primitives."""
    if reduction_axis is None:
        amax = torch.max(torch.abs(tensor))
    else:
        amax = torch.amax(torch.abs(tensor), dim=reduction_axis, keepdim=True)
    scale = amax_to_scale(
        amax,
        e4m3_dtype,
        round_scales_to_power_of_2=True,
    )
    qdata = to_fp8_saturated(tensor.to(torch.float32) * scale, e4m3_dtype)
    return qdata, scale


@dataclass(frozen=True, slots=True)
class _Float8LinearOperands:
    """The independent Float8 tensors owned by one FSDP unshard lifetime."""

    weight_qdata_fprop_NK: torch.Tensor  # noqa: N815
    weight_scale_fprop_N1: torch.Tensor  # noqa: N815
    weight_qdata_dgrad_NK: torch.Tensor  # noqa: N815
    weight_scale_dgrad: torch.Tensor

    @property
    def weight_qdata_fprop_KN(self) -> torch.Tensor:  # noqa: N802
        return self.weight_qdata_fprop_NK.t()

    @property
    def weight_scale_fprop_1N(self) -> torch.Tensor:  # noqa: N802
        return self.weight_scale_fprop_N1.t()


def _quantize_float8_weight(
    weight_NK: torch.Tensor,
    *,
    dgrad_weight_tensorwise: bool,
) -> _Float8LinearOperands:
    """Build the two weight orientations used by Float8 FPROP and DGRAD."""
    if weight_NK.ndim != 2:
        raise ValueError(
            f"Float8 weight quantization requires a 2D weight, got {weight_NK.ndim} dimensions."
        )
    if any(size % _FLOAT8_ALIGNMENT for size in weight_NK.shape):
        raise ValueError(
            "Float8 weight quantization requires both matrix dimensions divisible "
            f"by {_FLOAT8_ALIGNMENT}, got {tuple(weight_NK.shape)}."
        )

    weight_qdata_fprop_NK, weight_scale_fprop_N1 = _quantize_float8(
        weight_NK,
        reduction_axis=-1,
    )
    weight_qdata_dgrad_NK, weight_scale_dgrad = _quantize_float8(
        weight_NK,
        reduction_axis=None if dgrad_weight_tensorwise else 0,
    )
    return _Float8LinearOperands(
        weight_qdata_fprop_NK=weight_qdata_fprop_NK,
        weight_scale_fprop_N1=weight_scale_fprop_N1,
        weight_qdata_dgrad_NK=weight_qdata_dgrad_NK,
        weight_scale_dgrad=weight_scale_dgrad,
    )


class _LinearShardedTensorWithFloat8Compute(_ShardedFSDPTensor):
    """Persistent high-precision parameter with rowwise Float8 compute weights."""

    dgrad_weight_tensorwise = False

    def _build_operands(
        self,
        logical_tensor: torch.Tensor,
        out: _Float8LinearOperands | None = None,
    ) -> _Float8LinearOperands:
        if logical_tensor.ndim > 2 and logical_tensor.shape[-2] % _FLOAT8_ALIGNMENT:
            raise ValueError(
                "Float8 requires local matrix out_features divisible by "
                f"{_FLOAT8_ALIGNMENT}; got {logical_tensor.shape[-2]}. Adjust "
                "the Linear out_features or TP degree."
            )
        operands = _quantize_float8_weight(
            logical_tensor.flatten(0, -2),
            dgrad_weight_tensorwise=self.dgrad_weight_tensorwise,
        )
        if out is None:
            return operands
        out.weight_qdata_fprop_NK.copy_(operands.weight_qdata_fprop_NK)
        out.weight_scale_fprop_N1.copy_(operands.weight_scale_fprop_N1)
        out.weight_qdata_dgrad_NK.copy_(operands.weight_qdata_dgrad_NK)
        out.weight_scale_dgrad.copy_(operands.weight_scale_dgrad)
        return out


class _LinearShardedTensorWithFloat8GWHPCompute(_LinearShardedTensorWithFloat8Compute):
    """Float8 weight whose DGRAD operand uses tensorwise scaling."""

    dgrad_weight_tensorwise = True
