# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared FFN autograd with native MXFP8 weights, WGRAD, and save policies.

Shape suffixes: T = tokens, D = model width, F = hidden width, P = packed 2F.
W13 preparation runs under the owning module's FSDP unshard. The Function is
called inside W2.forward, after W2's own pre-forward hooks have run.
"""

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torchao.prototype.mx_formats.kernels import (
    mxfp8_quantize_cuda,
    triton_mx_block_rearrange,
)

from torchtitan.quantization._fsdp_tensor import _UnshardedFSDPTensor
from torchtitan.quantization.mxfp8.tensor import (
    _LinearShardedTensorWithMXFP8Compute,
    _quantize_mxfp8_weight,
)

from torchtitan_recipes.overrides.fused_swiglu import (
    silu_and_mul_backward_op,
    silu_and_mul_op,
)

from .ops import shared_expert_backward_op, shared_expert_forward_op


def quantize(value, *, rowwise=True, colwise=True):
    """Native activation quantization and blocked scales, without changing policy."""
    row, column, row_scale, column_scale = mxfp8_quantize_cuda(
        value.resolve_neg().contiguous().reshape(-1, value.shape[-1]),
        rowwise=rowwise,
        colwise=colwise,
        scaling_mode="rceil",
    )
    if rowwise:
        row_scale = triton_mx_block_rearrange(row_scale).flatten()
    else:
        row = value.new_empty((0,), dtype=torch.float8_e4m3fn)
        row_scale = value.new_empty((0,), dtype=torch.float8_e8m0fnu)
    if colwise:
        column_scale = triton_mx_block_rearrange(column_scale).flatten()
    else:
        column = value.new_empty((0,), dtype=torch.float8_e4m3fn)
        column_scale = value.new_empty((0,), dtype=torch.float8_e8m0fnu)
    return row, column, row_scale, column_scale


def scaled_mm(left, right, left_scale, right_scale, *, dtype=torch.bfloat16):
    return F.scaled_mm(
        left,
        right,
        scale_a=left_scale.flatten(),
        scale_recipe_a=F.ScalingType.BlockWise1x32,
        scale_b=right_scale.flatten(),
        scale_recipe_b=F.ScalingType.BlockWise1x32,
        swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
        output_dtype=dtype,
    )


def weight_operands(weight):
    if isinstance(weight, _UnshardedFSDPTensor):
        return weight.operands
    storage = (
        weight._tensor
        if isinstance(weight, _LinearShardedTensorWithMXFP8Compute)
        else weight
    )
    with torch.no_grad():
        return _quantize_mxfp8_weight(storage.flatten(0, -2))


def _saved_weight(weight, operands):
    if isinstance(weight, _UnshardedFSDPTensor):
        return (
            operands.weight_qdata_dgrad_NK.new_empty((0,)),
            operands.weight_scale_dgrad_swizzled.new_empty((0,)),
        )
    return operands.weight_qdata_dgrad_NK, operands.weight_scale_dgrad_swizzled


def _backward_weight(weight, qdata, scale):
    if isinstance(weight, _UnshardedFSDPTensor):
        operands = weight.operands
        return operands.weight_qdata_dgrad_NK, operands.weight_scale_dgrad_swizzled
    return qdata, scale


def _can_accumulate(weight):
    return (
        not torch.compiler.is_compiling()
        and get_proxy_mode() is None
        and weight.is_leaf
    )


class PreparedSharedExpert:
    """Opaque to FSDP's recursive input cast; FP8 operands must stay FP8.

    FSDP traverses dataclasses as well as tuples and lists. The BF16 hidden
    activation is the ordinary W2 argument; these prepared operands are not.
    """

    __slots__ = (
        "input",
        "weight",
        "packed",
        "hidden",
        "input_column",
        "input_column_scale",
        "hidden_operands",
        "weight_qdata",
        "weight_dgrad_scale",
        "input_save_format",
        "fused_backward",
        "accumulate_wgrad",
    )

    def __init__(
        self,
        input,
        weight,
        packed,
        hidden,
        input_column,
        input_column_scale,
        hidden_operands,
        weight_qdata,
        weight_dgrad_scale,
        input_save_format,
        fused_backward,
        accumulate_wgrad,
    ):
        self.input = input
        self.weight = weight
        self.packed = packed
        self.hidden = hidden
        self.input_column = input_column
        self.input_column_scale = input_column_scale
        self.hidden_operands = hidden_operands
        self.weight_qdata = weight_qdata
        self.weight_dgrad_scale = weight_dgrad_scale
        self.input_save_format = input_save_format
        self.fused_backward = fused_backward
        self.accumulate_wgrad = accumulate_wgrad


def _check_prepared_input(x, prepared_input):
    tokens = x.numel() // x.shape[-1]
    dim = x.shape[-1]
    if len(prepared_input) != 4:
        raise ValueError(
            "Prepared input requires row/column payloads and blocked scales"
        )
    specs = (
        ((tokens, dim), (dim, 1), torch.float8_e4m3fn),
        ((tokens, dim), (1, tokens), torch.float8_e4m3fn),
        ((tokens * dim // 32,), (1,), torch.float8_e8m0fnu),
        ((tokens * dim // 32,), (1,), torch.float8_e8m0fnu),
    )
    for value, (shape, stride, dtype) in zip(prepared_input, specs):
        if (
            value.shape != shape
            or value.stride() != stride
            or value.dtype != dtype
            or value.device != x.device
            or value.requires_grad
        ):
            raise ValueError(
                "Prepared input does not match the BF16 input's MXFP8 contract"
            )


@spmd.register_local_autograd_function
@torch._dynamo.allow_in_graph
class _PreparedHiddenFunction(torch.autograd.Function):
    """Keep W2's FSDP input hook ordered before the W13/input gradient hooks.

    The full FFN Function below computes the real gradients. This view exposes
    the hidden activation's W13 dependencies to the ordinary W2 module call,
    so separate FSDP groups reduce W2 before W13 as in the native graph.
    """

    @staticmethod
    def forward(ctx, hidden, x, weight):  # pyrefly: ignore[bad-override]
        ctx.set_materialize_grads(False)
        return hidden

    @staticmethod
    def backward(ctx, gradient):  # pyrefly: ignore[bad-override]
        return None, None, None


def prepare_shared_expert(
    x,
    weight,
    *,
    fused_forward,
    fused_backward,
    input_save_format,
    prepared_input=None,
):
    """Prepare W13 and hidden operands; keep the differentiable BF16 input explicit."""
    if prepared_input is not None:
        _check_prepared_input(x, prepared_input)
    operands = weight_operands(weight)
    with torch.no_grad():
        row, column, row_scale, column_scale = (
            quantize(x, colwise=weight.requires_grad and input_save_format == "mxfp8")
            if prepared_input is None
            else prepared_input
        )
        if fused_forward:
            hidden, packed, *hidden_operands = shared_expert_forward_op(
                row,
                operands.weight_qdata_dgrad_NK,
                row_scale,
                operands.weight_scale_fprop_swizzled,
            )
        else:
            packed = scaled_mm(
                row,
                operands.weight_qdata_fprop_KN,
                row_scale,
                operands.weight_scale_fprop_swizzled,
            )
            hidden = silu_and_mul_op(packed.unflatten(-1, (2, packed.shape[-1] // 2)))
            hidden_operands = (None, None, None, None)
    qdata, dgrad_scale = _saved_weight(weight, operands)
    hidden = _PreparedHiddenFunction.apply(
        hidden.reshape(*x.shape[:-1], hidden.shape[-1]), x, weight
    )
    return PreparedSharedExpert(
        x,
        weight,
        packed,
        hidden,
        column,
        column_scale,
        tuple(hidden_operands),
        qdata,
        dgrad_scale,
        input_save_format,
        fused_backward,
        _can_accumulate(weight),
    )


def _weight_gradient(column, scale, input_column, input_scale, shape, dtype, parameter):
    running_grad = None if parameter is None else parameter.grad
    if running_grad is None:
        return scaled_mm(
            column.t(), input_column, scale, input_scale, dtype=dtype
        ).view(shape)
    F.scaled_addmm_(
        running_grad.view(-1, shape[-1]),
        column.t(),
        input_column,
        scale_a=scale.flatten(),
        scale_recipe_a=F.ScalingType.BlockWise1x32,
        scale_b=input_scale.flatten(),
        scale_recipe_b=F.ScalingType.BlockWise1x32,
        swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
    )
    parameter.grad = None
    return running_grad


@spmd.register_local_autograd_function
@torch._dynamo.allow_in_graph
class FusedDSv3SharedExpertFunction(torch.autograd.Function):
    """Full shared FFN first derivative with native dx and both WGRAD GEMMs."""

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx,
        x,
        w13,
        w2,
        packed,
        hidden,
        input_column,
        input_scale,
        hidden_row,
        hidden_column,
        hidden_scale,
        hidden_column_scale,
        w13_qdata,
        w13_dgrad_scale,
        w2_qdata,
        w2_fprop_scale,
        w2_dgrad_scale,
        input_save_format,
        hidden_save_format,
        fused_backward,
        accumulate_w13,
        accumulate_w2,
    ):
        if hidden_row is None:
            hidden_row, hidden_column, hidden_scale, hidden_column_scale = quantize(
                hidden, colwise=w2.requires_grad and hidden_save_format == "mxfp8"
            )
        output = scaled_mm(hidden_row, w2_qdata.t(), hidden_scale, w2_fprop_scale)
        if isinstance(w2, _UnshardedFSDPTensor):
            w2_qdata = w2_qdata.new_empty((0,))
            w2_dgrad_scale = w2_dgrad_scale.new_empty((0,))
        saved_input = (
            x
            if w13.requires_grad and input_save_format == "bf16"
            else x.new_empty((0,))
        )
        saved_hidden = (
            hidden
            if w2.requires_grad and hidden_save_format == "bf16"
            else hidden.new_empty((0,))
        )
        if input_save_format != "mxfp8" or not w13.requires_grad:
            input_column = x.new_empty((0,), dtype=torch.float8_e4m3fn)
            input_scale = x.new_empty((0,), dtype=torch.float8_e8m0fnu)
        if hidden_save_format != "mxfp8" or not w2.requires_grad:
            hidden_column = x.new_empty((0,), dtype=torch.float8_e4m3fn)
            hidden_column_scale = x.new_empty((0,), dtype=torch.float8_e8m0fnu)
        ctx.save_for_backward(
            w13,
            w2,
            packed,
            input_column,
            input_scale,
            hidden_column,
            hidden_column_scale,
            w13_qdata,
            w13_dgrad_scale,
            w2_qdata,
            w2_dgrad_scale,
            saved_input,
            saved_hidden,
        )
        ctx.input_shape = x.shape
        ctx.w13_shape, ctx.w2_shape = w13.shape, w2.shape
        ctx.w13_grad_dtype = w13.grad_dtype or w13.dtype
        ctx.w2_grad_dtype = w2.grad_dtype or w2.dtype
        ctx.input_save_format, ctx.hidden_save_format = (
            input_save_format,
            hidden_save_format,
        )
        ctx.w13_parameter = w13 if accumulate_w13 else None
        ctx.w2_parameter = w2 if accumulate_w2 else None
        ctx.fused_backward = fused_backward
        return output.reshape(*x.shape[:-1], output.shape[-1])

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):  # pyrefly: ignore[bad-override]
        (
            w13,
            w2,
            packed,
            input_column,
            input_scale,
            hidden_column,
            hidden_column_scale,
            w13_qdata,
            w13_dgrad_scale,
            w2_qdata,
            w2_dgrad_scale,
            saved_input,
            saved_hidden,
        ) = ctx.saved_tensors
        ctx.maybe_clear_saved_tensors()
        needs_input, needs_w13, needs_w2 = ctx.needs_input_grad[:3]
        dx = dw13 = dw2 = None
        packed_operands = None
        row, column, row_scale, column_scale = quantize(
            gradient, rowwise=needs_input or needs_w13, colwise=needs_w2
        )
        if needs_input or needs_w13:
            w2_qdata, w2_dgrad_scale = _backward_weight(w2, w2_qdata, w2_dgrad_scale)
            if ctx.fused_backward:
                packed_operands = shared_expert_backward_op(
                    row, w2_qdata, row_scale, w2_dgrad_scale, packed
                )
            else:
                d_hidden = scaled_mm(row, w2_qdata, row_scale, w2_dgrad_scale)
                d_packed = silu_and_mul_backward_op(
                    d_hidden, packed.unflatten(-1, (2, packed.shape[-1] // 2))
                ).flatten(-2)
                packed_operands = quantize(
                    d_packed, rowwise=needs_input, colwise=needs_w13
                )
                del d_hidden, d_packed
        del packed
        if needs_w2:
            if ctx.hidden_save_format == "bf16":
                _, hidden_column, _, hidden_column_scale = quantize(
                    saved_hidden, rowwise=False
                )
            dw2 = _weight_gradient(
                column,
                column_scale,
                hidden_column,
                hidden_column_scale,
                ctx.w2_shape,
                ctx.w2_grad_dtype,
                ctx.w2_parameter,
            )
        del hidden_column, hidden_column_scale, saved_hidden
        del row, column, row_scale, column_scale
        if needs_input or needs_w13:
            assert packed_operands is not None
            (
                packed_row,
                packed_column,
                packed_row_scale,
                packed_column_scale,
            ) = packed_operands
            if needs_input:
                w13_qdata, w13_dgrad_scale = _backward_weight(
                    w13, w13_qdata, w13_dgrad_scale
                )
                dx = scaled_mm(
                    packed_row, w13_qdata, packed_row_scale, w13_dgrad_scale
                ).reshape(ctx.input_shape)
            if needs_w13:
                if ctx.input_save_format == "bf16":
                    _, input_column, _, input_scale = quantize(
                        saved_input, rowwise=False
                    )
                dw13 = _weight_gradient(
                    packed_column,
                    packed_column_scale,
                    input_column,
                    input_scale,
                    ctx.w13_shape,
                    ctx.w13_grad_dtype,
                    ctx.w13_parameter,
                )
        return dx, dw13, dw2, *([None] * 18)


def shared_expert_linear(hidden, parameter, prepared, *, hidden_save_format):
    """Run inside the ordinary W2 module call, after its FSDP unshard hook."""
    operands = weight_operands(parameter)
    return FusedDSv3SharedExpertFunction.apply(
        prepared.input,
        prepared.weight,
        parameter,
        prepared.packed,
        hidden,
        prepared.input_column,
        prepared.input_column_scale,
        *prepared.hidden_operands,
        prepared.weight_qdata,
        prepared.weight_dgrad_scale,
        operands.weight_qdata_dgrad_NK,
        operands.weight_scale_fprop_swizzled,
        operands.weight_scale_dgrad_swizzled,
        prepared.input_save_format,
        hidden_save_format,
        prepared.fused_backward,
        prepared.accumulate_wgrad,
        _can_accumulate(parameter),
    )
