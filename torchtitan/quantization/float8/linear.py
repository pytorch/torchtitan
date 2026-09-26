# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Float8 linear training with FSDP-managed weight operands."""

from dataclasses import dataclass
from typing import Any, cast, Literal

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.function import once_differentiable

from torchao.float8.float8_ops import addmm_float8_unwrapped

from torchtitan.models.common.linear import Linear

from .._fsdp_tensor import _UnshardedFSDPTensor
from .tensor import (
    _LinearShardedTensorWithFloat8Compute,
    _LinearShardedTensorWithFloat8HighPrecisionWeightGradient,
    _quantize_float8,
    _quantize_float8_weight,
)


__all__ = ["Float8Linear"]

Float8RecipeName = Literal["rowwise", "rowwise_with_gw_hp"]
_FLOAT8_RECIPE_NAMES = ("rowwise", "rowwise_with_gw_hp")


def _prepare_float8_mm(
    lhs_qdata: torch.Tensor,
    lhs_scale: torch.Tensor,
    rhs_qdata: torch.Tensor,
    rhs_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize layouts and mixed tensorwise/rowwise scales for scaled_mm."""
    if not (lhs_qdata.stride(0) > lhs_qdata.stride(1) == 1):
        lhs_qdata = lhs_qdata.contiguous()
    if rhs_qdata.stride(0) > rhs_qdata.stride(1) == 1:
        rhs_qdata = rhs_qdata.t().contiguous().t()
    if lhs_scale.ndim == 0 and rhs_scale.ndim != 0:
        lhs_scale = lhs_scale.repeat(lhs_qdata.shape[0]).reshape(-1, 1)
    elif lhs_scale.ndim != 0 and rhs_scale.ndim == 0:
        rhs_scale = rhs_scale.repeat(rhs_qdata.shape[1]).reshape(1, -1)
    return lhs_qdata, lhs_scale, rhs_qdata, rhs_scale


def _float8_mm(
    lhs_qdata: torch.Tensor,
    lhs_scale: torch.Tensor,
    rhs_qdata: torch.Tensor,
    rhs_scale: torch.Tensor,
    *,
    output_dtype: torch.dtype,
    use_fast_accum: bool,
    emulate: bool,
) -> torch.Tensor:
    lhs_qdata, lhs_scale, rhs_qdata, rhs_scale = _prepare_float8_mm(
        lhs_qdata,
        lhs_scale,
        rhs_qdata,
        rhs_scale,
    )
    if emulate:
        return torch.mm(
            lhs_qdata.float() / lhs_scale,
            rhs_qdata.float() / rhs_scale,
        ).to(output_dtype)
    return addmm_float8_unwrapped(
        lhs_qdata,
        lhs_scale,
        rhs_qdata,
        rhs_scale,
        output_dtype,
        use_fast_accum=use_fast_accum,
    )


def _float8_mm_out(
    lhs_qdata: torch.Tensor,
    lhs_scale: torch.Tensor,
    rhs_qdata: torch.Tensor,
    rhs_scale: torch.Tensor,
    *,
    out: torch.Tensor,
    use_fast_accum: bool,
) -> torch.Tensor:
    """Write a Float8 matrix product into caller-owned BF16 storage.

    Adapted from ``torchao.float8.float8_ops.addmm_float8_unwrapped`` to use
    the ``out`` overload required by TorchTitan's storage-owning path.
    """
    lhs_qdata, lhs_scale, rhs_qdata, rhs_scale = _prepare_float8_mm(
        lhs_qdata,
        lhs_scale,
        rhs_qdata,
        rhs_scale,
    )
    lhs_inverse_scale = lhs_scale.reciprocal()
    rhs_inverse_scale = rhs_scale.reciprocal()
    post_inverse_scale = None
    is_rowwise = lhs_scale.shape == (lhs_qdata.shape[0], 1) and rhs_scale.shape == (
        1,
        rhs_qdata.shape[1],
    )
    if is_rowwise and not use_fast_accum:
        post_inverse_scale = lhs_inverse_scale * rhs_inverse_scale
        lhs_inverse_scale = lhs_inverse_scale.new_ones(())
        rhs_inverse_scale = rhs_inverse_scale.new_ones(())

    lhs_recipe = (
        F.ScalingType.TensorWise
        if lhs_inverse_scale.ndim == 0
        else F.ScalingType.RowWise
    )
    rhs_recipe = (
        F.ScalingType.TensorWise
        if rhs_inverse_scale.ndim == 0
        else F.ScalingType.RowWise
    )
    scaled_mm_v2 = cast(Any, torch.ops.aten)._scaled_mm_v2
    scaled_mm_v2.out(
        lhs_qdata,
        rhs_qdata,
        [lhs_inverse_scale],
        [lhs_recipe.value],
        [],
        [rhs_inverse_scale],
        [rhs_recipe.value],
        [],
        None,
        out.dtype,
        [],
        use_fast_accum,
        out=out,
    )
    if post_inverse_scale is not None:
        out.mul_(post_inverse_scale)
    return out


# Adapted from
# torchao.float8.float8_linear.matmul_with_hp_or_float8_args. TorchTitan passes
# cached weight operands explicitly so its autograd and FSDP lifetimes compose.
@torch._dynamo.allow_in_graph
class _Float8LinearFunction(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        x: torch.Tensor,
        weight_NK: torch.Tensor,
        weight_qdata_fprop_KN: torch.Tensor,
        weight_scale_fprop_1N: torch.Tensor,
        weight_qdata_dgrad_NK: torch.Tensor,
        weight_scale_dgrad: torch.Tensor,
        weight_gradient_in_high_precision: bool,
        emulate: bool,
    ) -> torch.Tensor:
        if x.dtype != weight_NK.dtype:
            raise ValueError(
                "Float8Linear requires activations and weights with the same "
                f"dtype; got {x.dtype} and {weight_NK.dtype}."
            )
        if x.shape[-1] != weight_NK.shape[1]:
            raise ValueError(
                "Float8Linear activation and weight contraction dimensions must "
                f"match; got {x.shape[-1]} and {weight_NK.shape[1]}."
            )

        input_shape = x.shape
        x_MK = x.reshape(-1, input_shape[-1])
        # FPROP activations use one scale per token row: (M, K) -> (M, 1).
        x_qdata_row_MK, x_scale_row_M1 = _quantize_float8(
            x_MK,
            reduction_axis=-1,
        )
        output_shape = (*input_shape[:-1], weight_NK.shape[0])
        if emulate or x.dtype is not torch.bfloat16:
            output_MN = _float8_mm(
                x_qdata_row_MK,
                x_scale_row_M1,
                weight_qdata_fprop_KN,
                weight_scale_fprop_1N,
                output_dtype=x.dtype,
                use_fast_accum=True,
                emulate=emulate,
            )
            output = output_MN.reshape(output_shape).clone()
        else:
            output = x.new_empty(output_shape)
            _float8_mm_out(
                x_qdata_row_MK,
                x_scale_row_M1,
                weight_qdata_fprop_KN,
                weight_scale_fprop_1N,
                out=output.view(-1, weight_NK.shape[0]),
                use_fast_accum=True,
            )

        has_unsharded_tensor = isinstance(weight_NK, _UnshardedFSDPTensor)
        saved_weight_tensors = (
            (weight_NK,)
            if has_unsharded_tensor
            else (weight_qdata_dgrad_NK, weight_scale_dgrad)
        )
        ctx.save_for_backward(x, *saved_weight_tensors)
        ctx.has_unsharded_tensor = has_unsharded_tensor
        ctx.input_shape = input_shape
        ctx.weight_gradient_in_high_precision = weight_gradient_in_high_precision
        ctx.emulate = emulate
        return output

    @staticmethod
    @once_differentiable
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output: torch.Tensor):
        x, *saved_weight_tensors = ctx.saved_tensors
        if ctx.has_unsharded_tensor:
            (weight_NK,) = saved_weight_tensors
            if not isinstance(weight_NK, _UnshardedFSDPTensor):
                raise RuntimeError("FSDP restored an incompatible Float8 weight")
            operands = weight_NK.operands
            weight_qdata_dgrad_NK = operands.weight_qdata_dgrad_NK
            weight_scale_dgrad = operands.weight_scale_dgrad
        else:
            weight_qdata_dgrad_NK, weight_scale_dgrad = saved_weight_tensors

        grad_output_MN = grad_output.reshape(-1, grad_output.shape[-1])
        grad_input = None
        if ctx.needs_input_grad[0]:
            # DGRAD output gradients use one scale per token row.
            grad_output_qdata_row_MN, grad_output_scale_row_M1 = _quantize_float8(
                grad_output_MN, reduction_axis=-1
            )
            grad_input_MK = _float8_mm(
                grad_output_qdata_row_MN,
                grad_output_scale_row_M1,
                weight_qdata_dgrad_NK,
                weight_scale_dgrad,
                output_dtype=x.dtype,
                use_fast_accum=False,
                emulate=ctx.emulate,
            )
            grad_input = grad_input_MK.reshape(ctx.input_shape)

        grad_weight_NK = None
        if ctx.needs_input_grad[1]:
            x_MK = x.reshape(-1, ctx.input_shape[-1])
            if ctx.weight_gradient_in_high_precision:
                grad_weight_NK = torch.mm(grad_output_MN.t(), x_MK)
            else:
                # WGRAD quantizes both inputs per feature column. Transposing
                # dY turns its (1, N) scales into per-row (N, 1) scales.
                (
                    grad_output_qdata_col_MN,
                    grad_output_scale_col_1N,
                ) = _quantize_float8(grad_output_MN, reduction_axis=0)
                x_qdata_col_MK, x_scale_col_1K = _quantize_float8(
                    x_MK,
                    reduction_axis=0,
                )
                grad_weight_NK = _float8_mm(
                    grad_output_qdata_col_MN.t(),
                    grad_output_scale_col_1N.t(),
                    x_qdata_col_MK,
                    x_scale_col_1K,
                    output_dtype=x.dtype,
                    use_fast_accum=False,
                    emulate=ctx.emulate,
                )

        return grad_input, grad_weight_NK, None, None, None, None, None, None


spmd.register_local_autograd_function(_Float8LinearFunction)


class Float8Linear(Linear):
    """Linear with TorchTitan-owned Float8 autograd and FSDP operands."""

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        """Drop-in replacement for ``Linear.Config``."""

        recipe_name: Float8RecipeName = "rowwise"
        emulate: bool = False

        def __post_init__(self) -> None:
            if self.recipe_name not in _FLOAT8_RECIPE_NAMES:
                raise ValueError(
                    f"Float8 recipe_name must be one of {_FLOAT8_RECIPE_NAMES}; "
                    f"got {self.recipe_name!r}."
                )
            for name in ("in_features", "out_features"):
                value = getattr(self, name)
                if value % 16:
                    raise ValueError(
                        f"Float8 requires {name} divisible by 16; got {name}={value}."
                    )

    def __init__(self, config: Config):
        super().__init__(config)
        self.recipe_name = config.recipe_name
        self.emulate = config.emulate
        wrapper_cls = (
            _LinearShardedTensorWithFloat8HighPrecisionWeightGradient
            if config.recipe_name == "rowwise_with_gw_hp"
            else _LinearShardedTensorWithFloat8Compute
        )
        self.weight = nn.Parameter(
            wrapper_cls(self.weight.data),
            requires_grad=self.weight.requires_grad,
        )

    def _parallelize(self, parallel_dims) -> None:
        # spmd_types returns a plain tensor when TP shards the weight. Restore
        # the FSDP extension wrapper before fully_shard() consumes it.
        super()._parallelize(parallel_dims)
        wrapper_cls = (
            _LinearShardedTensorWithFloat8HighPrecisionWeightGradient
            if self.recipe_name == "rowwise_with_gw_hp"
            else _LinearShardedTensorWithFloat8Compute
        )
        if isinstance(self.weight, wrapper_cls):
            return
        distributed_weight = self.weight
        wrapped_weight = nn.Parameter(
            wrapper_cls(distributed_weight.data),
            requires_grad=distributed_weight.requires_grad,
        )
        spmd.assert_type_like(wrapped_weight, distributed_weight)
        self.weight = wrapped_weight

    def _linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if torch.is_autocast_enabled():
            input = input.to(torch.get_autocast_gpu_dtype())

        physical_weight = self.weight
        local_out_features = physical_weight.shape[-2]
        if local_out_features % 16:
            raise ValueError(
                "Float8 requires local out_features divisible by 16; got "
                f"{local_out_features}. Adjust the Linear out_features or TP degree."
            )
        if isinstance(physical_weight, _UnshardedFSDPTensor):
            operands = physical_weight.operands
        else:
            with torch.no_grad():
                high_precision_weight = (
                    physical_weight._tensor
                    if isinstance(
                        physical_weight, _LinearShardedTensorWithFloat8Compute
                    )
                    else physical_weight
                )
                operands = _quantize_float8_weight(
                    high_precision_weight.flatten(0, -2),
                    grad_input_weight_tensorwise=(
                        self.recipe_name == "rowwise_with_gw_hp"
                    ),
                )

        output = _Float8LinearFunction.apply(
            input,
            weight,
            operands.weight_qdata_fprop_KN,
            operands.weight_scale_fprop_1N,
            operands.weight_qdata_dgrad_NK,
            operands.weight_scale_dgrad,
            self.recipe_name == "rowwise_with_gw_hp",
            self.emulate,
        )
        if bias is not None:
            output = output + bias.to(output.dtype)
        return output
