# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 linear training with pluggable FSDP-managed weight caches.

Tensor shape suffixes:
    M: flattened token rows
    N: output features
    K: input features
"""

from dataclasses import dataclass, replace
from typing import Literal

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.function import once_differentiable

from torchao.prototype.mx_formats.kernels import (
    mxfp8_quantize_cuda,
    triton_mx_block_rearrange,
)

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.decoder_sharding import dense_activation_placement
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.sharding import LocalMapConfig

from .quantize import get_mxfp8_weight_quantization_strategy, quantize_mxfp8_weight
from .tensor import MXFP8FSDPComputeWeight, MXFP8FSDPWeight


TP = MeshAxisName.TP

_MXFP8_BLOCK_SIZE = 32
_MXFP8_SCALING_MODE = "rceil"

InputActivationSaveFormat = Literal["bf16", "mxfp8"]
_INPUT_ACTIVATION_SAVE_FORMATS = ("bf16", "mxfp8")


def _pad_rows(x_MK: torch.Tensor) -> tuple[torch.Tensor, int]:
    num_rows = x_MK.shape[0]
    num_padded_rows = (
        (num_rows + _MXFP8_BLOCK_SIZE - 1) // _MXFP8_BLOCK_SIZE
    ) * _MXFP8_BLOCK_SIZE
    if num_padded_rows == num_rows:
        return x_MK, num_rows
    return F.pad(x_MK, (0, 0, 0, num_padded_rows - num_rows)), num_rows


# Adapted from torchao.prototype.moe_training.mxfp8_linear.mx_mm. This variant
# lives in TorchTitan so its autograd state and weight cache can integrate with
# FSDP and other parallelisms.
@torch._dynamo.allow_in_graph
class _MXFP8LinearFunction(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        x: torch.Tensor,
        weight_NK: torch.Tensor,
        bias_N: torch.Tensor | None,
        weight_quantization: str,
        input_activation_save_format: InputActivationSaveFormat,
    ) -> torch.Tensor:
        # FPROP always consumes rowwise MXFP8. WGRAD can either retain the
        # original BF16 input and quantize it columnwise in backward, or retain
        # a columnwise MXFP8 representation produced in forward. The former is
        # memory-safe when another operation already keeps BF16 x alive; the
        # latter reduces storage for a unique input at the cost of an extra
        # cached representation when BF16 x is retained elsewhere. Under full
        # activation checkpointing, the selected state is created by recompute.
        if isinstance(weight_NK, MXFP8FSDPComputeWeight):
            if weight_NK.weight_quantization != weight_quantization:
                raise AssertionError(
                    "MXFP8Linear weight strategy does not match its FSDP cache: "
                    f"{weight_quantization!r} != "
                    f"{weight_NK.weight_quantization!r}."
                )
            weight_shape = weight_NK.shape
            q_weight_fprop_KN = weight_NK.q_weight_fprop_KN
            s_weight_fprop_blocked = weight_NK.s_weight_fprop_blocked
            q_weight_dgrad_NK = weight_NK.q_weight_dgrad_NK
            s_weight_dgrad_blocked = weight_NK.s_weight_dgrad_blocked
        else:
            # FSDP2 supplies MXFP8FSDPComputeWeight with prepared operands.
            # Non-FSDP execution uses the inner BF16 tensor, while GraphTrainer
            # SimpleFSDP supplies a plain/Fake BF16 all-gathered tensor.
            weight_hp_NK = (
                weight_NK._data if isinstance(weight_NK, MXFP8FSDPWeight) else weight_NK
            )
            weight_shape = weight_hp_NK.shape
            operands = quantize_mxfp8_weight(weight_hp_NK, weight_quantization)
            q_weight_fprop_KN = operands.q_weight_fprop_KN
            s_weight_fprop_blocked = operands.s_weight_fprop_blocked
            q_weight_dgrad_NK = operands.q_weight_dgrad_NK
            s_weight_dgrad_blocked = operands.s_weight_dgrad_blocked

        if x.dtype != torch.bfloat16 or weight_NK.dtype != torch.bfloat16:
            raise ValueError(
                "MXFP8Linear requires BF16 activations and weights; "
                f"got activation dtype {x.dtype} and weight dtype {weight_NK.dtype}."
            )
        if bias_N is not None and bias_N.dtype != torch.bfloat16:
            raise ValueError(
                f"MXFP8Linear requires a BF16 bias; got bias dtype {bias_N.dtype}."
            )
        if x.shape[-1] != weight_shape[1]:
            raise ValueError(
                "MXFP8Linear activation and weight contraction dimensions must "
                f"match; got {x.shape[-1]} and {weight_shape[1]}."
            )
        for name, value in (
            ("local in_features", weight_shape[1]),
            ("local out_features", weight_shape[0]),
        ):
            if value % _MXFP8_BLOCK_SIZE:
                raise ValueError(
                    f"MXFP8Linear requires {name} divisible by "
                    f"{_MXFP8_BLOCK_SIZE}; got {value}."
                )

        input_shape = x.shape
        x_MK, num_rows = _pad_rows(x.reshape(-1, input_shape[-1]).contiguous())
        requires_wgrad = ctx.needs_input_grad[1]
        quantize_wgrad_input_in_forward = (
            requires_wgrad and input_activation_save_format == "mxfp8"
        )

        # The save format controls both computation and saved state. BF16 mode
        # requests only the rowwise FPROP operand here; backward produces the
        # columnwise WGRAD operand from the saved BF16 input.
        x_row_MK, x_col_MK, x_row_scales, x_col_scales = mxfp8_quantize_cuda(
            x_MK,
            rowwise=True,
            colwise=quantize_wgrad_input_in_forward,
            scaling_mode=_MXFP8_SCALING_MODE,
        )
        x_row_scales = triton_mx_block_rearrange(x_row_scales)
        if quantize_wgrad_input_in_forward:
            x_col_scales = triton_mx_block_rearrange(x_col_scales)

        # Weight strategies must return both qdata/scale pairs ready for this
        # exact BlockWise1x32 and SWIZZLE_32_4_4 B-operand contract. Strategies
        # using another logical tile must expand and swizzle before returning.
        output_MN = F.scaled_mm(
            x_row_MK,
            q_weight_fprop_KN,
            scale_a=x_row_scales,
            scale_recipe_a=F.ScalingType.BlockWise1x32,
            scale_b=s_weight_fprop_blocked,
            scale_recipe_b=F.ScalingType.BlockWise1x32,
            swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
            swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
            bias=bias_N,
            output_dtype=torch.bfloat16,
        )

        # Save exactly one input-activation representation for WGRAD. BF16 mode
        # keeps the original tensor and builds the columnwise operand in
        # backward. MXFP8 mode keeps the columnwise qdata and scales produced
        # above. A weight strategy may share qdata between FPROP and DGRAD or
        # provide orientation-specific storage.
        if requires_wgrad and input_activation_save_format == "bf16":
            ctx.save_for_backward(
                x,
                q_weight_dgrad_NK,
                s_weight_dgrad_blocked,
            )
        else:
            ctx.save_for_backward(
                x_col_MK,
                x_col_scales,
                q_weight_dgrad_NK,
                s_weight_dgrad_blocked,
            )
        ctx.input_shape = input_shape
        ctx.num_rows = num_rows
        ctx.requires_dgrad = ctx.needs_input_grad[0]
        ctx.requires_wgrad = requires_wgrad
        ctx.input_activation_save_format = input_activation_save_format
        ctx.has_bias = bias_N is not None

        return output_MN[:num_rows].reshape(*input_shape[:-1], weight_shape[0])

    @staticmethod
    @once_differentiable
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output: torch.Tensor):
        # WGRAD consumes either the saved columnwise activation pair or a pair
        # rebuilt from the saved BF16 input. DGRAD consumes the weight pair.
        x_hp = None
        x_col_MK = None
        x_col_scales = None
        if ctx.requires_wgrad and ctx.input_activation_save_format == "bf16":
            (
                x_hp,
                q_weight_dgrad_NK,
                s_weight_dgrad_blocked,
            ) = ctx.saved_tensors
        else:
            (
                x_col_MK,
                x_col_scales,
                q_weight_dgrad_NK,
                s_weight_dgrad_blocked,
            ) = ctx.saved_tensors

        grad_output_MN = grad_output.contiguous().reshape(-1, grad_output.shape[-1])
        grad_bias_N = grad_output_MN.sum(dim=0) if ctx.has_bias else None

        grad_input = None
        grad_weight_NK = None
        if ctx.requires_dgrad or ctx.requires_wgrad:
            padded_grad_output_MN, _ = _pad_rows(grad_output_MN)
            (
                grad_output_row_MN,
                grad_output_col_MN,
                grad_output_row_scales,
                grad_output_col_scales,
            ) = mxfp8_quantize_cuda(
                padded_grad_output_MN,
                rowwise=ctx.requires_dgrad,
                colwise=ctx.requires_wgrad,
                scaling_mode=_MXFP8_SCALING_MODE,
            )

            if ctx.requires_dgrad:
                grad_output_row_scales = triton_mx_block_rearrange(
                    grad_output_row_scales
                )
                grad_input_MK = F.scaled_mm(
                    grad_output_row_MN,
                    q_weight_dgrad_NK,
                    scale_a=grad_output_row_scales,
                    scale_recipe_a=F.ScalingType.BlockWise1x32,
                    scale_b=s_weight_dgrad_blocked,
                    scale_recipe_b=F.ScalingType.BlockWise1x32,
                    swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
                    swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
                    output_dtype=torch.bfloat16,
                )
                grad_input = grad_input_MK[: ctx.num_rows].reshape(ctx.input_shape)

            if ctx.requires_wgrad:
                if ctx.input_activation_save_format == "bf16":
                    assert x_hp is not None
                    x_MK, _ = _pad_rows(
                        x_hp.reshape(-1, ctx.input_shape[-1]).contiguous()
                    )
                    _, x_col_MK, _, x_col_scales = mxfp8_quantize_cuda(
                        x_MK,
                        rowwise=False,
                        colwise=True,
                        scaling_mode=_MXFP8_SCALING_MODE,
                    )
                    x_col_scales = triton_mx_block_rearrange(x_col_scales)

                assert x_col_MK is not None
                assert x_col_scales is not None
                grad_output_col_scales = triton_mx_block_rearrange(
                    grad_output_col_scales
                )
                grad_weight_NK = F.scaled_mm(
                    grad_output_col_MN.t(),
                    x_col_MK,
                    scale_a=grad_output_col_scales,
                    scale_recipe_a=F.ScalingType.BlockWise1x32,
                    scale_b=x_col_scales,
                    scale_recipe_b=F.ScalingType.BlockWise1x32,
                    swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
                    swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
                    output_dtype=torch.bfloat16,
                )

        return grad_input, grad_weight_NK, grad_bias_N, None, None


spmd.register_local_autograd_function(_MXFP8LinearFunction)


class MXFP8Linear(Linear):
    """Linear using 1D activations and pluggable cached weight quantization."""

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        """Drop-in replacement for ``Linear.Config``."""

        weight_quantization: str = "32x32"
        """Dense-weight quantization strategy.

        ``"32x32"`` uses square scale tiles. The quantization groups are
        invariant under transpose, so FPROP and DGRAD can share qdata. Custom
        ``MXFP8Linear`` subclasses may select another registered strategy.
        """
        input_activation_save_format: InputActivationSaveFormat = "bf16"
        """Format used to save the input activation needed by WGRAD.

        ``"bf16"`` saves the original input and quantizes it columnwise during
        backward. ``"mxfp8"`` produces the columnwise representation during
        forward and saves its qdata and scales for backward.
        """

        def __post_init__(self) -> None:
            get_mxfp8_weight_quantization_strategy(self.weight_quantization)
            if self.input_activation_save_format not in _INPUT_ACTIVATION_SAVE_FORMATS:
                raise ValueError(
                    "MXFP8 input_activation_save_format must be one of "
                    f"{_INPUT_ACTIVATION_SAVE_FORMATS}; got "
                    f"{self.input_activation_save_format!r}."
                )
            for name in ("in_features", "out_features"):
                value = getattr(self, name)
                if value % _MXFP8_BLOCK_SIZE:
                    raise ValueError(
                        f"MXFP8 requires {name} divisible by {_MXFP8_BLOCK_SIZE}; "
                        f"got {name}={value}."
                    )

        def build(self, **kwargs):
            # Model converters run before update_from_config() attaches the
            # stock Linear sharding config. Adapt that late-bound config here so
            # the opaque MXFP8 autograd function runs on local tensors with the
            # correct TP input and input-gradient placements.
            instance = Linear.Config.build(self, **kwargs)
            if instance._sharding_config is not None:
                sharding_config = instance._sharding_config
                weight_tp = (
                    sharding_config.state_shardings["weight"]
                    .per_axis_spmd_types()
                    .get(TP)
                )
                rowwise = isinstance(weight_tp, spmd.Shard) and weight_tp.dim == 1
                if rowwise:
                    input_layout = dense_activation_placement(
                        tp=spmd.S(-1), cp=spmd.S(1)
                    )
                    input_grad_layout = dense_activation_placement(
                        tp=spmd.S(-1), cp=spmd.S(1)
                    )
                else:
                    input_layout = dense_activation_placement(tp=spmd.R, cp=spmd.S(1))
                    input_grad_layout = dense_activation_placement(
                        tp=spmd.P, cp=spmd.S(1)
                    )
                instance._sharding_config = replace(
                    sharding_config,
                    in_src_shardings={
                        **(sharding_config.in_src_shardings or {}),
                        "input": input_layout,
                    },
                    in_dst_shardings={
                        **(sharding_config.in_dst_shardings or {}),
                        "input": input_layout,
                    },
                    local_map=LocalMapConfig(in_grad_placements=(input_grad_layout,)),
                )
            return instance

    def __init__(self, config: Config):
        super().__init__(config)
        self.weight_quantization = config.weight_quantization
        self.input_activation_save_format = config.input_activation_save_format
        self.weight = nn.Parameter(
            MXFP8FSDPWeight(self.weight, self.weight_quantization),
            requires_grad=self.weight.requires_grad,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return _MXFP8LinearFunction.apply(
            input,
            self.weight,
            self.bias,
            self.weight_quantization,
            self.input_activation_save_format,
        )


__all__ = ["InputActivationSaveFormat", "MXFP8Linear"]
