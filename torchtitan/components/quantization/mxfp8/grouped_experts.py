# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 grouped-expert training with FSDP-managed 32x32 weight caches.

Tensor shape suffixes:
    M: flattened routed-token rows
    E: experts
    N: expert output features
    K: expert input features
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch
from torch import nn
from torch.autograd.function import once_differentiable

from torchao.prototype.moe_training.kernels.mxfp8.quant import (
    triton_mx_block_rearrange_2d_K_groups,
    triton_mx_block_rearrange_2d_M_groups,
)
from torchao.prototype.mx_formats.kernels import mxfp8_quantize_cuda

from .._fsdp_tensor import _UnshardedFSDPTensor

from ._common import (
    _INPUT_ACTIVATION_FORMATS_FOR_BACKWARD,
    _MXFP8_BLOCK_SIZE,
    _MXFP8_SCALING_MODE,
    InputActivationFormatForBackward,
)
from .tensor import (
    _GroupedExpertsShardedTensorWithMXFP8Compute,
    _quantize_mxfp8_grouped_weight,
)


# The MXFP8 experts class is created per experts variant, so the factory is
# the only thing callers name.
__all__ = ["get_mxfp8_grouped_experts_cls"]


def _rowwise_operands(
    x_MK: torch.Tensor, offs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize along K and lay the scales out for a grouped GEMM A operand."""
    x_row_MK, _, x_row_scales, _ = mxfp8_quantize_cuda(
        x_MK,
        rowwise=True,
        colwise=False,
        scaling_mode=_MXFP8_SCALING_MODE,
    )
    return x_row_MK, triton_mx_block_rearrange_2d_M_groups(x_row_scales, offs)


def _colwise_operands(
    x_MK: torch.Tensor, offs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize along M and lay the scales out for a grouped WGRAD operand.

    Scaling along M groups 32 rows together, so a scale block must not span two
    experts. The MXFP8 converter enforces that by padding each expert's token
    group to a multiple of the block size, which also makes the per-group scale
    offsets an exact division of the token offsets.
    """
    _, x_col_MK, _, x_col_scales = mxfp8_quantize_cuda(
        x_MK,
        rowwise=False,
        colwise=True,
        scaling_mode=_MXFP8_SCALING_MODE,
    )
    scale_offs = offs // _MXFP8_BLOCK_SIZE
    return x_col_MK, triton_mx_block_rearrange_2d_K_groups(x_col_scales, scale_offs)


# Lives in TorchTitan rather than TorchAO so its autograd state and weight
# cache can integrate with FSDP and the routed-expert parallelisms; TorchAO
# stays a source of quantization and layout kernels. Mirrors the dense
# _MXFP8LinearFunction, which is structured the same way.
@torch._dynamo.allow_in_graph
class _MXFP8GroupedMMFunction(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        x_MK: torch.Tensor,
        weight_ENK: torch.Tensor,
        weight_qdata_fprop_EKN: torch.Tensor,
        weight_scale_fprop_swizzled: torch.Tensor,
        weight_qdata_dgrad_ENK: torch.Tensor,
        weight_scale_dgrad_swizzled: torch.Tensor,
        offs: torch.Tensor,
        input_activation_format_for_backward: InputActivationFormatForBackward,
    ) -> torch.Tensor:
        # FPROP always consumes rowwise MXFP8. WGRAD can either retain the
        # original BF16 input and quantize it columnwise in backward, or retain
        # a columnwise MXFP8 operands produced in forward. See
        # MXFP8GroupedExperts.Config for the trade-off.
        if x_MK.dtype != torch.bfloat16 or weight_ENK.dtype != torch.bfloat16:
            raise ValueError(
                "MXFP8 grouped experts require BF16 activations and weights; "
                f"got activation dtype {x_MK.dtype} and weight dtype "
                f"{weight_ENK.dtype}."
            )
        if x_MK.ndim != 2:
            raise ValueError(
                "MXFP8 grouped experts require 2D routed activations; got "
                f"{x_MK.ndim} dimensions."
            )
        if x_MK.shape[-1] != weight_ENK.shape[-1]:
            raise ValueError(
                "MXFP8 grouped-expert activation and weight contraction "
                f"dimensions must match; got {x_MK.shape[-1]} and "
                f"{weight_ENK.shape[-1]}."
            )
        for name, value in (
            ("local expert in_features", weight_ENK.shape[2]),
            ("local expert out_features", weight_ENK.shape[1]),
        ):
            if value % _MXFP8_BLOCK_SIZE:
                raise ValueError(
                    f"MXFP8 grouped experts require {name} divisible by "
                    f"{_MXFP8_BLOCK_SIZE}; got {value}."
                )

        x_MK = x_MK.contiguous()
        requires_wgrad = ctx.needs_input_grad[1]
        quantize_wgrad_input_in_forward = (
            requires_wgrad and input_activation_format_for_backward == "mxfp8"
        )

        x_row_MK, x_row_scales_blocked = _rowwise_operands(x_MK, offs)
        output_MN = torch._scaled_grouped_mm(
            x_row_MK,
            weight_qdata_fprop_EKN,
            x_row_scales_blocked,
            weight_scale_fprop_swizzled,
            offs=offs,
            out_dtype=torch.bfloat16,
        )

        # Save exactly one input-activation operands for WGRAD, and let
        # FSDP own the weight operands whenever it manages them: saving the
        # wrapper rather than its current operands means a reshard between
        # forward and backward refills the same tensors in place.
        # An unsharded weight carries operands FSDP will refill before
        # backward, so save the wrapper. Anything else has none, so the DGRAD
        # operands have to be saved directly.
        has_unsharded_tensor = isinstance(weight_ENK, _UnshardedFSDPTensor)
        saved_weight_tensors = (
            (weight_ENK,)
            if has_unsharded_tensor
            else (weight_qdata_dgrad_ENK, weight_scale_dgrad_swizzled)
        )
        if quantize_wgrad_input_in_forward:
            x_col_MK, x_col_scales_blocked = _colwise_operands(x_MK, offs)
            ctx.save_for_backward(
                x_col_MK, x_col_scales_blocked, offs, *saved_weight_tensors
            )
        else:
            ctx.save_for_backward(x_MK, offs, *saved_weight_tensors)
        ctx.requires_dgrad = ctx.needs_input_grad[0]
        ctx.requires_wgrad = requires_wgrad
        ctx.input_activation_format_for_backward = input_activation_format_for_backward
        ctx.saved_quantized_input = quantize_wgrad_input_in_forward
        ctx.has_unsharded_tensor = has_unsharded_tensor
        return output_MN

    @staticmethod
    @once_differentiable
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output_MN: torch.Tensor):
        saved_tensors = ctx.saved_tensors
        if ctx.saved_quantized_input:
            x_col_MK, x_col_scales_blocked, offs = saved_tensors[:3]
            x_hp_MK = None
            saved_weight_tensors = saved_tensors[3:]
        else:
            x_hp_MK, offs = saved_tensors[:2]
            x_col_MK = None
            x_col_scales_blocked = None
            saved_weight_tensors = saved_tensors[2:]

        if ctx.has_unsharded_tensor:
            (weight_ENK,) = saved_weight_tensors
            if not isinstance(weight_ENK, _UnshardedFSDPTensor):
                raise RuntimeError("FSDP restored an incompatible MXFP8 weight")
            operands = weight_ENK.operands
            if operands is None:
                raise RuntimeError("FSDP did not build MXFP8 weight state for backward")
            weight_qdata_dgrad_ENK = operands.weight_qdata_dgrad_ENK
            weight_scale_dgrad_swizzled = operands.weight_scale_dgrad_swizzled
        else:
            weight_qdata_dgrad_ENK, weight_scale_dgrad_swizzled = saved_weight_tensors

        grad_output_MN = grad_output_MN.contiguous()
        grad_input_MK = None
        grad_weight_ENK = None
        if ctx.requires_dgrad or ctx.requires_wgrad:
            (
                grad_output_row_MN,
                grad_output_col_MN,
                grad_output_row_scales,
                grad_output_col_scales,
            ) = mxfp8_quantize_cuda(
                grad_output_MN,
                rowwise=ctx.requires_dgrad,
                colwise=ctx.requires_wgrad,
                scaling_mode=_MXFP8_SCALING_MODE,
            )

            if ctx.requires_dgrad:
                grad_output_row_scales_blocked = triton_mx_block_rearrange_2d_M_groups(
                    grad_output_row_scales, offs
                )
                grad_input_MK = torch._scaled_grouped_mm(
                    grad_output_row_MN,
                    weight_qdata_dgrad_ENK,
                    grad_output_row_scales_blocked,
                    weight_scale_dgrad_swizzled,
                    offs=offs,
                    out_dtype=torch.bfloat16,
                )

            if ctx.requires_wgrad:
                if x_col_MK is None:
                    assert x_hp_MK is not None
                    x_col_MK, x_col_scales_blocked = _colwise_operands(x_hp_MK, offs)
                grad_output_col_scales_blocked = triton_mx_block_rearrange_2d_K_groups(
                    grad_output_col_scales, offs // _MXFP8_BLOCK_SIZE
                )
                grad_weight_ENK = torch._scaled_grouped_mm(
                    grad_output_col_MN.transpose(-2, -1),
                    x_col_MK,
                    grad_output_col_scales_blocked,
                    x_col_scales_blocked,
                    offs=offs,
                    out_dtype=torch.bfloat16,
                )

        return grad_input_MK, grad_weight_ENK, None, None, None, None, None, None


# Marks the function local-only so SPMD type checking can propagate through
# an autograd function it cannot see into.
# TODO(anijain2305, pianpwk): drop this once register_local_autograd_function
# is removed tree-wide. MXFP8Linear, nvfp4 and qwen3_5's gdn rely on the same
# registration, so it has to go everywhere at once.
spmd.register_local_autograd_function(_MXFP8GroupedMMFunction)


_mxfp8_experts_cache: dict[type, type] = {}


def get_mxfp8_grouped_experts_cls(parent_cls: type) -> type:
    """Get or create an MXFP8 subclass of *parent_cls*.

    Works for any experts module exposing the ``_grouped_mm`` seam (the common
    ``GroupedExperts`` and its per-model variants). The returned class has a
    proper ``_owner`` set by ``__init_subclass__``.
    """
    if parent_cls in _mxfp8_experts_cache:
        return _mxfp8_experts_cache[parent_cls]

    parent_config_cls = parent_cls.Config  # type: ignore[attr-defined]

    class MXFP8GroupedExperts(parent_cls):  # type: ignore[valid-type, misc]
        """Grouped experts using cached 32x32 expert-weight quantization."""

        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            input_activation_format_for_backward: InputActivationFormatForBackward = (
                "bf16"
            )
            """Format used to save the input activation needed by WGRAD.

            ``"bf16"`` saves the original input and quantizes it columnwise
            during backward. ``"mxfp8"`` produces the columnwise operands
            during forward and saves its qdata and scales for backward.

            The default is conservative because saving a quantized
            operands only reduces memory when no other operation retains
            the same BF16 input; if one does, the quantized copy is additional
            rather than a replacement. In the common SwiGLU experts the routed
            input feeds both the gate and up projections, so its BF16 form
            stays alive regardless.

            TODO: select this per projection rather than per module, so the
            down projection -- whose input is produced and consumed once -- can
            use MXFP8 while the gate and up projections keep BF16.
            """

            def __post_init__(self) -> None:
                if (
                    self.input_activation_format_for_backward
                    not in _INPUT_ACTIVATION_FORMATS_FOR_BACKWARD
                ):
                    raise ValueError(
                        "MXFP8 input_activation_format_for_backward must be one "
                        f"of {_INPUT_ACTIVATION_FORMATS_FOR_BACKWARD}; got "
                        f"{self.input_activation_format_for_backward!r}."
                    )

        def __init__(self, config: Config):
            super().__init__(config)
            self.input_activation_format_for_backward = (
                config.input_activation_format_for_backward
            )
            self._install_unsharded_tensors()

        def _install_unsharded_tensors(self) -> None:
            """Wrap each grouped expert weight at construction.

            Only the grouped expert weights are wrapped. Experts variants may
            also own per-expert biases, which are not grouped GEMM operands and
            stay ordinary parameters. The wrapper is inert until a data
            parallel implementation drives its unshard lifecycle.
            """
            for name, parameter in list(self.named_parameters(recurse=False)):
                if parameter.ndim != 3:
                    continue
                setattr(
                    self,
                    name,
                    nn.Parameter(
                        _GroupedExpertsShardedTensorWithMXFP8Compute(parameter.data),
                        requires_grad=parameter.requires_grad,
                    ),
                )

        def _grouped_mm(self, *, A, weight_EOI, offs):
            # The seam speaks moe.py's legend, where O and I are the
            # expert output and input features. This module calls those
            # N and K (see the legend at the top of the file), so bind
            # once and use the local vocabulary below.
            weight_ENK = weight_EOI
            # __init__ installs the sharded wrapper, but under FSDP the
            # post-all-gather hook has already replaced it for this unshard
            # lifetime with the storage-free unsharded tensor, so the type says
            # which state we are in.
            if isinstance(weight_ENK, _UnshardedFSDPTensor):
                operands = weight_ENK.operands
            else:
                # Still the sharded parameter, so no data parallel
                # implementation owns this weight's lifecycle -- or the grouped
                # weight is a view of a differently shaped parameter, as with
                # the fused SwiGLU override. Either way the operands are built
                # per invocation, from the BF16 storage rather than the wrapper
                # the kernels cannot consume. ``weight_ENK`` itself stays
                # wrapped so autograd returns the gradient to the parameter.
                with torch.no_grad():
                    operands = _quantize_mxfp8_grouped_weight(
                        weight_ENK._tensor
                        if isinstance(
                            weight_ENK, _GroupedExpertsShardedTensorWithMXFP8Compute
                        )
                        else weight_ENK
                    )
            return _MXFP8GroupedMMFunction.apply(
                A,
                weight_ENK,
                operands.weight_qdata_fprop_EKN,
                operands.weight_scale_fprop_swizzled,
                operands.weight_qdata_dgrad_ENK,
                operands.weight_scale_dgrad_swizzled,
                offs,
                self.input_activation_format_for_backward,
            )

    MXFP8GroupedExperts.__name__ = f"MXFP8{parent_cls.__name__}"
    MXFP8GroupedExperts.__qualname__ = f"MXFP8{parent_cls.__name__}"
    _mxfp8_experts_cache[parent_cls] = MXFP8GroupedExperts
    return MXFP8GroupedExperts
