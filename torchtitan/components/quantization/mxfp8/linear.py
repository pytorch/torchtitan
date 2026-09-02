# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 linear training with FSDP-managed 32x32 weight caches.

Tensor shape suffixes:
    M: flattened token rows
    N: output features
    K: input features
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.function import once_differentiable

from torchao.prototype.mx_formats.kernels import (
    mxfp8_quantize_cuda,
    triton_mx_block_rearrange,
)

from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.linear import Linear

from .._fsdp_tensor import _UnshardedFSDPTensor
from ._common import (
    _INPUT_ACTIVATION_FORMATS_FOR_BACKWARD,
    _MXFP8_BLOCK_SIZE,
    _MXFP8_SCALING_MODE,
    InputActivationFormatForBackward,
)
from .tensor import _LinearShardedTensorWithMXFP8Compute, _quantize_mxfp8_weight


__all__ = ["InputActivationFormatForBackward", "MXFP8Linear"]


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
        weight_qdata_fprop_KN: torch.Tensor,
        weight_scale_fprop_swizzled: torch.Tensor,
        weight_qdata_dgrad_NK: torch.Tensor,
        weight_scale_dgrad_swizzled: torch.Tensor,
        bias_N: torch.Tensor | None,
        input_activation_format_for_backward: InputActivationFormatForBackward,
    ) -> torch.Tensor:
        # FPROP always consumes rowwise MXFP8. WGRAD can either retain the
        # original BF16 input and quantize it columnwise in backward, or retain
        # a columnwise MXFP8 operands produced in forward. The former is
        # memory-safe when another operation already keeps BF16 x alive; the
        # latter reduces storage for a single-consumer input at the cost of an
        # extra cached operands when BF16 x is retained elsewhere. Under
        # full activation checkpointing, the selected state is created by
        # recompute.
        if x.dtype != torch.bfloat16 or weight_NK.dtype != torch.bfloat16:
            raise ValueError(
                "MXFP8Linear requires BF16 activations and weights; "
                f"got activation dtype {x.dtype} and weight dtype {weight_NK.dtype}."
            )
        if bias_N is not None and bias_N.dtype != torch.bfloat16:
            raise ValueError(
                f"MXFP8Linear requires a BF16 bias; got bias dtype {bias_N.dtype}."
            )
        if x.shape[-1] != weight_NK.shape[1]:
            raise ValueError(
                "MXFP8Linear activation and weight contraction dimensions must "
                f"match; got {x.shape[-1]} and {weight_NK.shape[1]}."
            )
        for name, value in (
            ("local in_features", weight_NK.shape[1]),
            ("local out_features", weight_NK.shape[0]),
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
            requires_wgrad and input_activation_format_for_backward == "mxfp8"
        )

        # The save format controls both computation and saved state. BF16 mode
        # requests only the rowwise FPROP operand here; backward produces the
        # columnwise WGRAD operand from the saved BF16 input.
        x_qdata_row_MK, x_qdata_col_MK, x_scale_row, x_scale_col = mxfp8_quantize_cuda(
            x_MK,
            rowwise=True,
            colwise=quantize_wgrad_input_in_forward,
            scaling_mode=_MXFP8_SCALING_MODE,
        )
        x_scale_row = triton_mx_block_rearrange(x_scale_row)
        if quantize_wgrad_input_in_forward:
            x_scale_col = triton_mx_block_rearrange(x_scale_col)

        # The 32x32 weight quantizer returns both qdata/scale pairs ready for
        # this exact BlockWise1x32 and SWIZZLE_32_4_4 B-operand contract.
        output_MN = F.scaled_mm(
            x_qdata_row_MK,
            weight_qdata_fprop_KN,
            scale_a=x_scale_row,
            scale_recipe_a=F.ScalingType.BlockWise1x32,
            scale_b=weight_scale_fprop_swizzled,
            scale_recipe_b=F.ScalingType.BlockWise1x32,
            swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
            swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
            bias=bias_N,
            output_dtype=torch.bfloat16,
        )

        # Save exactly one input-activation operands for WGRAD. BF16 mode
        # keeps the original tensor and builds the columnwise operand in
        # backward. MXFP8 mode keeps the columnwise qdata and scales produced
        # above. FPROP and DGRAD share the same weight qdata allocation.
        # An unsharded tensor's storage is FSDP's to free at reshard and refill
        # before backward, so save the wrapper and read the operands off it
        # then. Anything else carries no operands to refill, so save them.
        has_unsharded_tensor = isinstance(weight_NK, _UnshardedFSDPTensor)
        saved_weight_tensors = (
            (weight_NK,)
            if has_unsharded_tensor
            else (weight_qdata_dgrad_NK, weight_scale_dgrad_swizzled)
        )
        if requires_wgrad and input_activation_format_for_backward == "bf16":
            ctx.save_for_backward(x, *saved_weight_tensors)
        else:
            ctx.save_for_backward(x_qdata_col_MK, x_scale_col, *saved_weight_tensors)
        ctx.has_unsharded_tensor = has_unsharded_tensor
        ctx.input_shape = input_shape
        ctx.num_rows = num_rows
        ctx.requires_dgrad = ctx.needs_input_grad[0]
        ctx.requires_wgrad = requires_wgrad
        ctx.input_activation_format_for_backward = input_activation_format_for_backward
        ctx.has_bias = bias_N is not None

        return output_MN[:num_rows].reshape(*input_shape[:-1], weight_NK.shape[0])

    @staticmethod
    @once_differentiable
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output: torch.Tensor):
        # WGRAD consumes either the saved columnwise activation pair or a pair
        # rebuilt from the saved BF16 input. DGRAD consumes the weight pair.
        x_hp = None
        x_qdata_col_MK = None
        x_scale_col = None
        saved_tensors = ctx.saved_tensors
        if ctx.requires_wgrad and ctx.input_activation_format_for_backward == "bf16":
            x_hp = saved_tensors[0]
            saved_weight_tensors = saved_tensors[1:]
        else:
            x_qdata_col_MK, x_scale_col = saved_tensors[:2]
            saved_weight_tensors = saved_tensors[2:]

        if ctx.has_unsharded_tensor:
            (weight_NK,) = saved_weight_tensors
            if not isinstance(weight_NK, _UnshardedFSDPTensor):
                raise RuntimeError("FSDP restored an incompatible MXFP8 weight")
            operands = weight_NK.operands
            weight_qdata_dgrad_NK = operands.weight_qdata_dgrad_NK
            weight_scale_dgrad_swizzled = operands.weight_scale_dgrad_swizzled
        else:
            weight_qdata_dgrad_NK, weight_scale_dgrad_swizzled = saved_weight_tensors

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
                    weight_qdata_dgrad_NK,
                    scale_a=grad_output_row_scales,
                    scale_recipe_a=F.ScalingType.BlockWise1x32,
                    scale_b=weight_scale_dgrad_swizzled,
                    scale_recipe_b=F.ScalingType.BlockWise1x32,
                    swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
                    swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
                    output_dtype=torch.bfloat16,
                )
                grad_input = grad_input_MK[: ctx.num_rows].reshape(ctx.input_shape)

            if ctx.requires_wgrad:
                if ctx.input_activation_format_for_backward == "bf16":
                    assert x_hp is not None
                    x_MK, _ = _pad_rows(
                        x_hp.reshape(-1, ctx.input_shape[-1]).contiguous()
                    )
                    _, x_qdata_col_MK, _, x_scale_col = mxfp8_quantize_cuda(
                        x_MK,
                        rowwise=False,
                        colwise=True,
                        scaling_mode=_MXFP8_SCALING_MODE,
                    )
                    x_scale_col = triton_mx_block_rearrange(x_scale_col)

                assert x_qdata_col_MK is not None
                assert x_scale_col is not None
                grad_output_col_scales = triton_mx_block_rearrange(
                    grad_output_col_scales
                )
                grad_weight_NK = F.scaled_mm(
                    grad_output_col_MN.t(),
                    x_qdata_col_MK,
                    scale_a=grad_output_col_scales,
                    scale_recipe_a=F.ScalingType.BlockWise1x32,
                    scale_b=x_scale_col,
                    scale_recipe_b=F.ScalingType.BlockWise1x32,
                    swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
                    swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
                    output_dtype=torch.bfloat16,
                )

        return grad_input, grad_weight_NK, None, None, None, None, grad_bias_N, None


# Marks the function local-only so SPMD type checking can propagate through
# an autograd function it cannot see into.
# TODO(anijain2305, pianpwk): drop this once register_local_autograd_function
# is removed tree-wide. nvfp4 and qwen3_5's gdn still rely on the same
# registration, so it has to go everywhere at once.
spmd.register_local_autograd_function(_MXFP8LinearFunction)


class MXFP8Linear(Linear):
    """Linear using 1D activations and cached 32x32 weight quantization."""

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        """Drop-in replacement for ``Linear.Config``."""

        input_activation_format_for_backward: InputActivationFormatForBackward = "bf16"
        """Format used to save the input activation needed by WGRAD.

        ``"bf16"`` saves the original input and quantizes it columnwise during
        backward. ``"mxfp8"`` produces the columnwise operands during
        forward and saves its qdata and scales for backward.
        """

        def __post_init__(self) -> None:
            if (
                self.input_activation_format_for_backward
                not in _INPUT_ACTIVATION_FORMATS_FOR_BACKWARD
            ):
                raise ValueError(
                    "MXFP8 input_activation_format_for_backward must be one of "
                    f"{_INPUT_ACTIVATION_FORMATS_FOR_BACKWARD}; got "
                    f"{self.input_activation_format_for_backward!r}."
                )
            for name in ("in_features", "out_features"):
                value = getattr(self, name)
                if value % _MXFP8_BLOCK_SIZE:
                    raise ValueError(
                        f"MXFP8 requires {name} divisible by {_MXFP8_BLOCK_SIZE}; "
                        f"got {name}={value}."
                    )

        def build(self, **kwargs):
            # The MXFP8 matmul is an opaque autograd function, so DTensor has
            # no sharding strategy for it. Under partial_dtensor with TP,
            # propagation reaches into the storage-free unsharded tensor and
            # fails; making it work needs local_map plus hand-declared input
            # and input-gradient placements. spmd_types instead annotates the
            # function itself (see register_local_autograd_function above), so
            # the stock Linear sharding config suffices there. Reject the
            # backend rather than carry a second sharding path for it.
            if get_spmd_backend() == "partial_dtensor":
                raise ValueError(
                    "MXFP8Linear requires parallelism.spmd_backend="
                    "'spmd_types'; got 'partial_dtensor'. The MXFP8 matmul is "
                    "an opaque autograd function with no DTensor sharding "
                    "rule, so tensor parallelism cannot propagate through it."
                )
            return Linear.Config.build(self, **kwargs)

    def __init__(self, config: Config):
        super().__init__(config)
        self.input_activation_format_for_backward = (
            config.input_activation_format_for_backward
        )
        # Install the unsharded-tensor wrapper up front so no caller has to
        # remember to do it. The wrapper is inert until a data parallel
        # implementation drives its unshard lifecycle: until then it just holds
        # the BF16 weight, and forward rejects it.
        self.weight = nn.Parameter(
            _LinearShardedTensorWithMXFP8Compute(self.weight.data),
            requires_grad=self.weight.requires_grad,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Always a plain tensor: the weight is only re-wrapped as a DTensor on
        # a non-data-parallel mesh under partial_dtensor, which Config.build
        # rejects, and spmd_types carries TP and EP as annotations instead.
        weight_NK = self.weight
        # __init__ installs a _LinearShardedTensorWithMXFP8Compute, but that is
        # not what forward usually sees. Under FSDP the post-all-gather hook has
        # already replaced it for this unshard lifetime with the storage-free
        # _UnshardedFSDPTensor holding the quantized operands, so the weight
        # arrives here already quantized and the type identifies which state we
        # are in.
        if isinstance(weight_NK, _UnshardedFSDPTensor):
            operands = weight_NK.operands
        else:
            # No data parallel implementation owns this weight's lifecycle, so
            # it still holds high-precision storage and the operands are built
            # per invocation. Eager FSDP2 always installs an unsharded tensor, but
            # GraphTrainer under the spmd_types backend does not: its runtime
            # hands forward a plain annotated local tensor, so the wrapper
            # SimpleFSDP's parametrization built never reaches here. Quantize
            # the storage rather than the wrapper, which the kernels cannot
            # consume; ``weight_NK`` itself stays wrapped so autograd returns
            # the gradient to the parameter.
            with torch.no_grad():
                operands = _quantize_mxfp8_weight(
                    weight_NK._tensor
                    if isinstance(weight_NK, _LinearShardedTensorWithMXFP8Compute)
                    else weight_NK
                )
            # Nothing caches this across calls, so a frozen weight is
            # requantized on every forward. Training pays that anyway, since
            # the weight changes each optimizer step; inference does not.
            # TODO(anijain2305): key the operands on the parameter's
            # version counter so a frozen weight is quantized once.
        return _MXFP8LinearFunction.apply(
            input,
            weight_NK,
            operands.weight_qdata_fprop_KN,
            operands.weight_scale_fprop_swizzled,
            operands.weight_qdata_dgrad_NK,
            operands.weight_scale_dgrad_swizzled,
            self.bias,
            self.input_activation_format_for_backward,
        )
