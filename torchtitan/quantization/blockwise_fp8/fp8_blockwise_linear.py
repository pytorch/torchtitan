# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Blockwise FP8 linear consumer for prequantized weights."""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn
from torchao.prototype.blockwise_fp8_training.kernels import (
    BLOCKWISE_128X128_SCALING_TYPE as _TORCHAO_BLOCKWISE_128X128_SCALING_TYPE,
    BLOCKWISE_1X128_SCALING_TYPE as _TORCHAO_BLOCKWISE_1X128_SCALING_TYPE,
    triton_fp8_blockwise_act_quant_lhs as _torchao_act_quant_lhs,
    triton_fp8_blockwise_act_quant_rhs as _torchao_act_quant_rhs,
    triton_fp8_blockwise_act_quant_transposed_lhs as _torchao_act_quant_transposed_lhs,
    triton_fp8_gemm_1x128_128x1 as _torchao_gemm_1x128_128x1,
    triton_fp8_gemm_1x128_128x128 as _torchao_gemm_1x128_128x128,
)
from torchao.prototype.blockwise_fp8_training.linear import (
    _run_blockwise_mm as _torchao_run_blockwise_mm,
    Float8BlockwiseLinear as TorchAOFloat8BlockwiseLinear,
)

from pytorch.flex_shard.custom_placements.fp8_bucketed_block_shard import (
    _pad_2d_to_block_shape,
)

from .blockwise_fp8_weight import BlockwiseFp8Weight


class FlexShardFloat8BlockwiseLinear(TorchAOFloat8BlockwiseLinear):
    """torchao Float8BlockwiseLinear that consumes FlexShard FP8 all-gather output."""

    @classmethod
    def from_float(
        cls,
        mod,
        use_triton: bool = False,
    ):
        with torch.device("meta"):
            new_mod = cls(
                mod.in_features,
                mod.out_features,
                bias=mod.bias is not None,
                use_triton=use_triton,
            )
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        new_mod.train(mod.training)
        return new_mod

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if not isinstance(weight, BlockwiseFp8Weight):
            raise TypeError(
                "FlexShardFloat8BlockwiseLinear requires a BlockwiseFp8Weight "
                f"produced by FP8 all-gather, got {type(weight).__name__}."
            )
        return _PrequantizedBlockwiseFp8Linear.apply(
            input,
            weight,
            self.bias,
            self.block_size,
            self.dtype,
            self.use_triton,
        )


def convert_to_flex_shard_float8_blockwise_linear(
    module: nn.Module,
    filter_fn: Callable[[str, nn.Linear], bool] | None = None,
    use_triton: bool = False,
    linear_cls: type[FlexShardFloat8BlockwiseLinear] = FlexShardFloat8BlockwiseLinear,
) -> nn.Module:
    """Replace matching nn.Linear children with the FlexShard torchao wrapper."""

    def should_convert(fqn: str, child: nn.Linear) -> bool:
        return filter_fn(fqn, child) if filter_fn is not None else True

    def convert_children(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            fqn = f"{prefix}.{name}" if prefix else name
            if isinstance(child, FlexShardFloat8BlockwiseLinear):
                continue
            if isinstance(child, nn.Linear) and should_convert(fqn, child):
                setattr(
                    parent,
                    name,
                    linear_cls.from_float(
                        child,
                        use_triton=use_triton,
                    ),
                )
            else:
                convert_children(child, fqn)

    convert_children(module)
    return module


def _pad_last_dim_to(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
    if tensor.shape[-1] == target_dim:
        return tensor
    if tensor.shape[-1] > target_dim:
        raise ValueError(
            f"Cannot pad last dim from {tensor.shape[-1]} down to {target_dim}."
        )
    padded = torch.zeros(
        (*tensor.shape[:-1], target_dim),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded[..., : tensor.shape[-1]].copy_(tensor)
    return padded


def _as_column_major_2d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 2:
        raise ValueError(f"expected a 2D tensor, got {tuple(tensor.shape)}")
    if tensor.stride(-2) == 1:
        return tensor
    return tensor.t().contiguous().t()


def _blockwise_fp8_linear_forward(
    ctx: Any,
    x: torch.Tensor,
    weight: BlockwiseFp8Weight,
    bias: torch.Tensor | None,
    block_size: int,
    out_dtype: torch.dtype,
    use_triton: bool,
) -> torch.Tensor:
    if block_size != weight.block_size:
        raise ValueError(
            f"linear block_size {block_size} does not match gathered weight "
            f"block_size {weight.block_size}"
        )
    if block_size != 128:
        raise AssertionError("torchao Float8BlockwiseLinear only supports 128")

    logical_out_dim, logical_in_dim = weight.shape
    x_orig_shape = x.shape
    x_2d = x.reshape(-1, x_orig_shape[-1])
    if x_2d.shape[-1] != logical_in_dim:
        raise ValueError(
            f"input last dim {x_2d.shape[-1]} does not match weight input dim "
            f"{logical_in_dim}"
        )
    weight_fp8 = _pad_2d_to_block_shape(weight.fp8_data, block_size)
    _, padded_in_dim = weight_fp8.shape
    x_2d = _pad_last_dim_to(x_2d, padded_in_dim)
    x_fp8, x_scale = _torchao_act_quant_lhs(
        x_2d,
        block_size,
        dtype=weight.fp8_data.dtype,
    )
    weight_t_fp8 = weight_fp8.t()
    weight_t_scale = weight.recip_scale.t()
    out = _torchao_run_blockwise_mm(
        use_triton=use_triton,
        triton_kernel=_torchao_gemm_1x128_128x128,
        mat_a=x_fp8,
        mat_b=weight_t_fp8,
        scale_a=x_scale,
        scale_recipe_a=_TORCHAO_BLOCKWISE_1X128_SCALING_TYPE,
        scale_b=weight_t_scale,
        scale_recipe_b=_TORCHAO_BLOCKWISE_128X128_SCALING_TYPE,
        out_dtype=out_dtype,
    )
    out = out[:, :logical_out_dim].reshape(*x_orig_shape[:-1], logical_out_dim)
    if bias is not None:
        out = out + bias
    ctx.save_for_backward(x, weight)
    ctx.block_size = block_size
    ctx.has_bias = bias is not None
    ctx.out_dtype = out_dtype
    ctx.use_triton = use_triton
    return out


def _blockwise_fp8_linear_backward(
    ctx: Any,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    x, weight = ctx.saved_tensors
    block_size = ctx.block_size
    out_dtype = ctx.out_dtype
    use_triton = ctx.use_triton
    fp8_dtype = weight.fp8_data.dtype
    grad_output_orig_shape = grad_output.shape
    grad_output_2d = grad_output.reshape(-1, grad_output_orig_shape[-1]).contiguous()
    logical_out_dim, logical_in_dim = weight.shape
    if grad_output_2d.shape[-1] != logical_out_dim:
        raise ValueError(
            f"grad_output last dim {grad_output_2d.shape[-1]} does not match "
            f"weight output dim {logical_out_dim}"
        )
    weight_fp8 = _pad_2d_to_block_shape(weight.fp8_data, block_size)
    padded_out_dim = weight_fp8.shape[0]
    grad_output_2d_padded = _pad_last_dim_to(grad_output_2d, padded_out_dim)

    grad_x = grad_weight = grad_bias = None
    needs_x_grad, needs_weight_grad, needs_bias_grad = ctx.needs_input_grad[:3]
    if needs_x_grad:
        grad_output_fp8, grad_output_scale = _torchao_act_quant_lhs(
            grad_output_2d_padded,
            block_size,
            dtype=fp8_dtype,
        )
        weight_fp8 = _as_column_major_2d(weight_fp8)
        weight_scale = _as_column_major_2d(weight.recip_scale)
        grad_x = _torchao_run_blockwise_mm(
            use_triton=use_triton,
            triton_kernel=_torchao_gemm_1x128_128x128,
            mat_a=grad_output_fp8,
            mat_b=weight_fp8,
            scale_a=grad_output_scale,
            scale_recipe_a=_TORCHAO_BLOCKWISE_1X128_SCALING_TYPE,
            scale_b=weight_scale,
            scale_recipe_b=_TORCHAO_BLOCKWISE_128X128_SCALING_TYPE,
            out_dtype=out_dtype,
        )
        grad_x = grad_x[:, :logical_in_dim].reshape(
            *grad_output_orig_shape[:-1],
            logical_in_dim,
        )
    if needs_weight_grad:
        # Both wgrad operands must share a block-aligned token dimension.
        grad_output_2d_wgrad = _pad_2d_to_block_shape(
            grad_output_2d_padded,
            block_size,
        )
        x_2d = _pad_2d_to_block_shape(
            x.reshape(-1, x.shape[-1]),
            block_size,
        )
        grad_output_t_fp8, grad_output_t_scale = _torchao_act_quant_transposed_lhs(
            grad_output_2d_wgrad,
            block_size,
            dtype=fp8_dtype,
        )
        x_fp8, x_scale = _torchao_act_quant_rhs(
            x_2d,
            block_size,
            dtype=fp8_dtype,
        )
        grad_weight = _torchao_run_blockwise_mm(
            use_triton=use_triton,
            triton_kernel=_torchao_gemm_1x128_128x1,
            mat_a=grad_output_t_fp8,
            mat_b=x_fp8,
            scale_a=grad_output_t_scale,
            scale_recipe_a=_TORCHAO_BLOCKWISE_1X128_SCALING_TYPE,
            scale_b=x_scale.transpose(-1, -2),
            scale_recipe_b=_TORCHAO_BLOCKWISE_1X128_SCALING_TYPE,
            triton_scale_b=x_scale,
            out_dtype=out_dtype,
        )
        # Slicing padded columns (e.g. w2 in_dim=10944 padded to 11008) yields a
        # non-contiguous view; the FlexShard grad reduce path requires contiguous
        # grads, so materialize before returning.
        grad_weight = (
            grad_weight[:logical_out_dim, :logical_in_dim].to(weight.dtype).contiguous()
        )
    if ctx.has_bias and needs_bias_grad:
        grad_bias = grad_output_2d.sum(dim=0)
    return grad_x, grad_weight, grad_bias


class _PrequantizedBlockwiseFp8Linear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: BlockwiseFp8Weight,
        bias: torch.Tensor | None,
        block_size: int,
        out_dtype: torch.dtype,
        use_triton: bool,
    ) -> torch.Tensor:
        return _blockwise_fp8_linear_forward(
            ctx,
            x,
            weight,
            bias,
            block_size,
            out_dtype,
            use_triton,
        )

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
    ]:
        grad_x, grad_weight, grad_bias = _blockwise_fp8_linear_backward(
            ctx,
            grad_output,
        )
        return grad_x, grad_weight, grad_bias, None, None, None


__all__ = [
    "convert_to_flex_shard_float8_blockwise_linear",
    "FlexShardFloat8BlockwiseLinear",
    "TorchAOFloat8BlockwiseLinear",
]
