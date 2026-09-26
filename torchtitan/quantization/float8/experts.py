# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Float8 grouped-expert training with FSDP-managed weight operands.

Tensor shape suffixes:
    R: routed tokens
    E: experts
    O: expert output features
    I: expert input features
"""

from dataclasses import dataclass
from typing import cast

import spmd_types as spmd
import torch
from torch import nn
from torch.autograd.function import once_differentiable

from torchao.prototype.moe_training.kernels import (
    triton_fp8_per_group_colwise_scales_dual,
    triton_fp8_rowwise_2d_scale_and_cast,
)

from .._fsdp_tensor import _UnshardedFSDPTensor
from .tensor import (
    _FLOAT8_GEMM_ALIGNMENT,
    _GroupedLinearShardedTensorWithFloat8Compute,
    _quantize_float8_grouped_weight,
)


__all__: list[str] = []


# Adapted from
# torchao.prototype.moe_training.fp8_grouped_mm._Float8GroupedMM. TorchTitan
# owns autograd and accepts pre-quantized weights from its FSDP cache; TorchAO
# continues to provide the quantization kernels.
@torch._dynamo.allow_in_graph
class _Float8GroupedMMFunction(torch.autograd.Function):
    """Grouped Float8 GEMM with TorchTitan-owned autograd state."""

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        A_RI: torch.Tensor,
        weight_EOI: torch.Tensor,
        weight_qdata_fprop_EIO: torch.Tensor,
        weight_scale_fprop_E1O: torch.Tensor,
        weight_qdata_dgrad_EOI: torch.Tensor,
        weight_scale_dgrad_EI: torch.Tensor,
        offsets_E: torch.Tensor,
    ) -> torch.Tensor:
        if A_RI.ndim != 2:
            raise ValueError(
                "Float8 grouped GEMM requires a 2D input, "
                f"got {A_RI.ndim} dimensions."
            )
        if weight_EOI.ndim != 3:
            raise ValueError(
                "Float8 grouped GEMM requires a 3D expert weight, "
                f"got {weight_EOI.ndim} dimensions."
            )
        if A_RI.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(
                "Float8 grouped GEMM requires FP32 or BF16 activations; "
                f"got {A_RI.dtype}."
            )
        if offsets_E.dtype != torch.int32:
            raise ValueError(
                "Float8 grouped GEMM requires int32 offsets; " f"got {offsets_E.dtype}."
            )
        if A_RI.shape[-1] != weight_EOI.shape[-1]:
            raise ValueError(
                "Float8 grouped GEMM activation and weight contraction "
                f"dimensions must match; got {A_RI.shape[-1]} and "
                f"{weight_EOI.shape[-1]}."
            )
        if any(size % _FLOAT8_GEMM_ALIGNMENT for size in weight_EOI.shape[-2:]):
            raise ValueError(
                "Float8 grouped GEMM requires local input and output features "
                f"divisible by {_FLOAT8_GEMM_ALIGNMENT}; "
                f"got {tuple(weight_EOI.shape[-2:])}."
            )

        # FPROP activations use one scale per routed-token row. Expert weights
        # use one scale per (expert, output-feature) column of the GEMM RHS.
        A_qdata_RI, A_scale_R1 = triton_fp8_rowwise_2d_scale_and_cast(
            A_RI,
            output_dtype=weight_qdata_fprop_EIO.dtype,
            round_scales_to_power_of_2=True,
        )
        output_RO = torch._scaled_grouped_mm(
            A_qdata_RI,
            weight_qdata_fprop_EIO,
            A_scale_R1.squeeze(-1).reciprocal(),
            weight_scale_fprop_E1O.squeeze(1).reciprocal(),
            offsets_E,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )

        has_unsharded_tensor = isinstance(weight_EOI, _UnshardedFSDPTensor)
        saved_weight_tensors = (
            (weight_EOI,)
            if has_unsharded_tensor
            else (weight_qdata_dgrad_EOI, weight_scale_dgrad_EI)
        )
        # TODO: Honor ctx.needs_input_grad like _Float8LinearFunction so frozen
        # expert weights do not save A_RI for or compute WGRAD.
        ctx.save_for_backward(A_RI, offsets_E, *saved_weight_tensors)
        ctx.has_unsharded_tensor = has_unsharded_tensor
        ctx.weight_dtype = weight_EOI.dtype
        return output_RO

    @staticmethod
    @once_differentiable
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output_RO: torch.Tensor):
        A_RI, offsets_E, *saved_weight_tensors = ctx.saved_tensors
        if ctx.has_unsharded_tensor:
            (weight_EOI,) = saved_weight_tensors
            if not isinstance(weight_EOI, _UnshardedFSDPTensor):
                raise RuntimeError("FSDP restored an incompatible Float8 expert weight")
            operands = weight_EOI.operands
            weight_qdata_dgrad_EOI = operands.weight_qdata_dgrad_EOI
            weight_scale_dgrad_EI = operands.weight_scale_dgrad_EI
        else:
            weight_qdata_dgrad_EOI, weight_scale_dgrad_EI = saved_weight_tensors

        grad_output_RO = grad_output_RO.contiguous()
        # DGRAD uses one scale per routed-token row for dY and one scale per
        # (expert, input-feature) column for the cached weight operand.
        # WGRAD computes separate columnwise scales inside every expert's
        # token group; offsets_E defines those group boundaries.
        (
            grad_output_qdata_RO,
            grad_output_scale_R1,
        ) = triton_fp8_rowwise_2d_scale_and_cast(
            grad_output_RO,
            output_dtype=weight_qdata_dgrad_EOI.dtype,
            round_scales_to_power_of_2=True,
        )
        grad_A_RI = torch._scaled_grouped_mm(
            grad_output_qdata_RO,
            weight_qdata_dgrad_EOI,
            grad_output_scale_R1.squeeze(-1).reciprocal(),
            weight_scale_dgrad_EI.reciprocal(),
            offsets_E,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )

        (
            grad_output_qdata_col,
            grad_output_scale_col,
            A_qdata_col,
            A_scale_col,
        ) = triton_fp8_per_group_colwise_scales_dual(
            grad_output_RO,
            A_RI,
            offsets_E,
            weight_qdata_dgrad_EOI.dtype,
            round_scales_to_power_of_2=True,
        )
        grad_weight_EOI = torch._scaled_grouped_mm(
            grad_output_qdata_col.t(),
            A_qdata_col,
            grad_output_scale_col.t().reciprocal(),
            A_scale_col.reciprocal(),
            offsets_E,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        ).to(ctx.weight_dtype)
        return grad_A_RI, grad_weight_EOI, None, None, None, None, None


spmd.register_local_autograd_function(_Float8GroupedMMFunction)


_float8_grouped_linear_cache: dict[type, type] = {}


def _get_float8_grouped_linear_cls(parent_cls: type) -> type:
    """Get or create a Float8-quantized grouped-linear subclass."""
    if parent_cls in _float8_grouped_linear_cache:
        return _float8_grouped_linear_cache[parent_cls]

    parent_config_cls = parent_cls.Config  # type: ignore[attr-defined]

    class Float8GroupedLinear(parent_cls):  # type: ignore[valid-type, misc]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            pass

        def __init__(self, config: Config):
            super().__init__(config)
            module = cast(nn.Module, self)
            parameter = module.get_parameter("weight")
            module.weight = nn.Parameter(
                _GroupedLinearShardedTensorWithFloat8Compute(parameter.data),
                requires_grad=parameter.requires_grad,
            )

        def _save_to_state_dict(self, destination, prefix, keep_vars):
            super()._save_to_state_dict(destination, prefix, keep_vars)
            module = cast(nn.Module, self)
            parameter = module.get_parameter("weight")
            if isinstance(parameter, _GroupedLinearShardedTensorWithFloat8Compute):
                tensor = parameter._tensor
                destination[prefix + "weight"] = (
                    tensor if keep_vars else tensor.detach()
                )

        def _grouped_mm(
            self,
            *,
            input_RI: torch.Tensor,
            weight_EOI: torch.Tensor,
            offsets_E: torch.Tensor,
        ) -> torch.Tensor:
            physical_weight = weight_EOI
            if isinstance(physical_weight, _UnshardedFSDPTensor):
                operands = physical_weight.operands
            else:
                with torch.no_grad():
                    high_precision_weight = (
                        physical_weight._tensor
                        if isinstance(
                            physical_weight,
                            _GroupedLinearShardedTensorWithFloat8Compute,
                        )
                        else physical_weight
                    )
                    operands = _quantize_float8_grouped_weight(high_precision_weight)

            return _Float8GroupedMMFunction.apply(
                input_RI,
                weight_EOI,
                operands.weight_qdata_fprop_EIO,
                operands.weight_scale_fprop_E1O,
                operands.weight_qdata_dgrad_EOI,
                operands.weight_scale_dgrad_EI,
                offsets_E,
            )

    Float8GroupedLinear.__name__ = f"Float8{parent_cls.__name__}"
    Float8GroupedLinear.__qualname__ = f"Float8{parent_cls.__name__}"
    _float8_grouped_linear_cache[parent_cls] = Float8GroupedLinear
    return Float8GroupedLinear
