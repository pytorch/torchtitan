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

from .._fsdp_tensor import _ShardedFSDPTensor


__all__: list[str] = []

_FLOAT8_GEMM_ALIGNMENT = 16


# Quantized operands are derived compute state, not differentiable model state.
# The custom autograd functions attach gradients to the high-precision weights,
# so recording scale computation and casting would only retain temporary storage.
# This follows torchao.float8.float8_scaling_utils.hp_tensor_to_float8_dynamic,
# but returns its plain qdata and scale for TorchTitan's FSDP-owned cache.
@torch.no_grad()
def _quantize_float8(
    tensor: torch.Tensor,
    *,
    reduction_axis: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamically quantize with one scale per unreduced index.

    For a 2D tensor, ``reduction_axis=-1`` is rowwise and returns ``(M, 1)``
    scales, ``reduction_axis=0`` is columnwise and returns ``(1, K)`` scales,
    and ``None`` is tensorwise and returns one scalar scale.
    """
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
    """The independent Float8 tensors owned by one FSDP unshard lifetime.

    FPROP quantizes stored ``weight_NK`` rowwise over K, producing ``(N, 1)``
    scales. Transposing both for the GEMM RHS gives ``weight_KN`` with one
    scale per N column, shaped ``(1, N)``.

    DGRAD uses ``weight_NK`` directly as its GEMM RHS. The rowwise recipe
    reduces over N and therefore has one scale per K column, shaped ``(1, K)``;
    the high-precision weight-gradient recipe instead uses one tensorwise
    scalar.
    """

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
    grad_input_weight_tensorwise: bool,
) -> _Float8LinearOperands:
    """Build the two weight orientations used by Float8 FPROP and DGRAD."""
    if weight_NK.ndim != 2:
        raise ValueError(
            f"Float8 weight quantization requires a 2D weight, got {weight_NK.ndim} dimensions."
        )
    if any(size % _FLOAT8_GEMM_ALIGNMENT for size in weight_NK.shape):
        raise ValueError(
            "Float8 weight quantization requires both matrix dimensions divisible "
            f"by {_FLOAT8_GEMM_ALIGNMENT}, got {tuple(weight_NK.shape)}."
        )

    # Stored W[N, K] is quantized per N row for X[M, K] @ W.T[K, N].
    weight_qdata_fprop_NK, weight_scale_fprop_N1 = _quantize_float8(
        weight_NK,
        reduction_axis=-1,
    )
    # DGRAD is dY[M, N] @ W[N, K], so W needs either one scale per K
    # output column or one scale for the entire tensor.
    weight_qdata_dgrad_NK, weight_scale_dgrad = _quantize_float8(
        weight_NK,
        reduction_axis=None if grad_input_weight_tensorwise else 0,
    )
    return _Float8LinearOperands(
        weight_qdata_fprop_NK=weight_qdata_fprop_NK,
        weight_scale_fprop_N1=weight_scale_fprop_N1,
        weight_qdata_dgrad_NK=weight_qdata_dgrad_NK,
        weight_scale_dgrad=weight_scale_dgrad,
    )


class _LinearShardedTensorWithFloat8Compute(_ShardedFSDPTensor):
    """Persistent high-precision parameter with rowwise Float8 compute weights."""

    grad_input_weight_tensorwise = False

    def _build_operands(
        self,
        logical_tensor: torch.Tensor,
        out: _Float8LinearOperands | None = None,
    ) -> _Float8LinearOperands:
        if (
            logical_tensor.ndim > 2
            and logical_tensor.shape[-2] % _FLOAT8_GEMM_ALIGNMENT
        ):
            raise ValueError(
                "Float8 requires local matrix out_features divisible by "
                f"{_FLOAT8_GEMM_ALIGNMENT}; got {logical_tensor.shape[-2]}. Adjust "
                "the Linear out_features or TP degree."
            )
        operands = _quantize_float8_weight(
            logical_tensor.flatten(0, -2),
            grad_input_weight_tensorwise=self.grad_input_weight_tensorwise,
        )
        if out is None:
            return operands
        out.weight_qdata_fprop_NK.copy_(operands.weight_qdata_fprop_NK)
        out.weight_scale_fprop_N1.copy_(operands.weight_scale_fprop_N1)
        out.weight_qdata_dgrad_NK.copy_(operands.weight_qdata_dgrad_NK)
        out.weight_scale_dgrad.copy_(operands.weight_scale_dgrad)
        return out


class _LinearShardedTensorWithFloat8HighPrecisionWeightGradient(
    _LinearShardedTensorWithFloat8Compute
):
    """Float8 weight cache for the high-precision weight-gradient recipe."""

    grad_input_weight_tensorwise = True


@dataclass(frozen=True, slots=True)
class _Float8GroupedExpertsOperands:
    """Float8 expert-weight operands owned by one FSDP unshard lifetime.

    FPROP uses transposed expert weights ``(E, I, O)`` with one scale per
    expert and O column, shaped ``(E, 1, O)``. DGRAD uses ``(E, O, I)`` with
    one scale per expert and I column, shaped ``(E, I)``. Thus scaling is
    axiswise within each expert; experts never share a scale.
    """

    weight_qdata_fprop_EIO: torch.Tensor  # noqa: N815
    weight_scale_fprop_E1O: torch.Tensor  # noqa: N815
    weight_qdata_dgrad_EOI: torch.Tensor  # noqa: N815
    weight_scale_dgrad_EI: torch.Tensor  # noqa: N815


@torch.no_grad()
def _quantize_float8_grouped_weight(
    weight_EOI: torch.Tensor,
) -> _Float8GroupedExpertsOperands:
    """Build the two expert-weight orientations used by FPROP and DGRAD.

    The kernel sequence is adapted from
    ``torchao.prototype.moe_training.fp8_grouped_mm._Float8GroupedMM``. It is
    hoisted out of that autograd function so FSDP can cache the results.
    """
    from torchao.prototype.moe_training.kernels import (
        triton_fp8_colwise_3d_scale_and_cast,
        triton_fp8_rowwise_3d_transpose_rhs,
    )

    if weight_EOI.ndim != 3:
        raise ValueError(
            "Float8 grouped weight quantization requires a 3D weight, "
            f"got {weight_EOI.ndim} dimensions."
        )
    if any(size % _FLOAT8_GEMM_ALIGNMENT for size in weight_EOI.shape[-2:]):
        raise ValueError(
            "Float8 grouped weight quantization requires both matrix dimensions "
            f"divisible by {_FLOAT8_GEMM_ALIGNMENT}, got {tuple(weight_EOI.shape)}."
        )

    weight_EIO = weight_EOI.bfloat16().transpose(-2, -1)
    # Reduce I independently for every (expert, output-feature) pair.
    (
        weight_qdata_fprop_EIO,
        weight_scale_fprop_E1O,
    ) = triton_fp8_colwise_3d_scale_and_cast(
        weight_EIO,
        output_dtype=e4m3_dtype,
        round_scales_to_power_of_2=True,
    )
    # Reduce O independently for every (expert, input-feature) pair.
    weight_qdata_dgrad_EOI, weight_scale_dgrad_EI = triton_fp8_rowwise_3d_transpose_rhs(
        weight_EIO,
        output_dtype=e4m3_dtype,
        round_scales_to_power_of_2=True,
    )
    return _Float8GroupedExpertsOperands(
        weight_qdata_fprop_EIO=weight_qdata_fprop_EIO,
        weight_scale_fprop_E1O=weight_scale_fprop_E1O,
        weight_qdata_dgrad_EOI=weight_qdata_dgrad_EOI,
        weight_scale_dgrad_EI=weight_scale_dgrad_EI,
    )


class _GroupedExpertsShardedTensorWithFloat8Compute(_ShardedFSDPTensor):
    """Persistent expert parameter with cached Float8 compute operands."""

    def _build_operands(
        self,
        logical_tensor: torch.Tensor,
        out: _Float8GroupedExpertsOperands | None = None,
    ) -> _Float8GroupedExpertsOperands:
        operands = _quantize_float8_grouped_weight(logical_tensor)
        if out is None:
            return operands
        out.weight_qdata_fprop_EIO.copy_(operands.weight_qdata_fprop_EIO)
        out.weight_scale_fprop_E1O.copy_(operands.weight_scale_fprop_E1O)
        out.weight_qdata_dgrad_EOI.copy_(operands.weight_qdata_dgrad_EOI)
        out.weight_scale_dgrad_EI.copy_(operands.weight_scale_dgrad_EI)
        return out
