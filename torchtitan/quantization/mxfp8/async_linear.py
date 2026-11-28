# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Async tensor-parallel MXFP8 linears.

Tensor shape suffixes follow ``async_linear.py``: over ``R`` TP ranks, ``M``
token rows, ``N`` output features and ``K`` input features.
"""

import math
from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_remat as remat
from torch.autograd.function import once_differentiable

from torchao.prototype.mx_formats.kernels import (
    mxfp8_quantize_cuda,
    triton_mx_block_rearrange,
)

from torchtitan.distributed.spmd_types import spmd_dense_sp_enabled
from torchtitan.models.common.linear import ColumnParallelLinear, RowParallelLinear
from torchtitan.models.common.linear.async_linear import (
    _tp_group_from_context,
    ensure_symm_mem_ops,
)
from torchtitan.models.common.linear.inplace_wgrad_accum import uses_inplace_wgrad_accum

from .linear import (
    _check_mxfp8_linear_inputs,
    _MXFP8_SCALING_MODE,
    _mxfp8_weight_operands,
    _mxfp8_wgrad,
    _restore_dgrad_weight_operands,
    _save_dgrad_weight_operands,
    InputActivationFormatForBackward,
    MXFP8Linear,
)

__all__ = ["MXFP8AsyncColumnParallelLinear", "MXFP8AsyncRowParallelLinear"]

_MX = F.ScalingType.BlockWise1x32.value
_SWIZZLE = F.SwizzleType.SWIZZLE_32_4_4.value
# The fused symm_mem MXFP8 ops gather or slice the swizzled scale per rank, so
# each rank's rows must fill whole 128-row scale tiles.
_ROW_ALIGNMENT = 128


def _quantize_rowwise(x_MK: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    qdata, _, scale, _ = mxfp8_quantize_cuda(
        x_MK, rowwise=True, colwise=False, scaling_mode=_MXFP8_SCALING_MODE
    )
    return qdata, triton_mx_block_rearrange(scale)


def _quantize_colwise(x_MK: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Columnwise qdata and unswizzled scales, the operands WGRAD gathers."""
    _, qdata, _, scale = mxfp8_quantize_cuda(
        x_MK, rowwise=False, colwise=True, scaling_mode=_MXFP8_SCALING_MODE
    )
    return qdata, scale


class _ColwiseAllGather:
    """All-gather a columnwise-quantized ``[M / R, K]`` operand along M.

    Quantizing each shard and gathering equals quantizing the gathered tensor:
    columnwise blocks span 32 rows, and each shard's rows are a multiple of 128,
    so no block straddles two ranks. That halves the bytes of gathering BF16 and
    quantizes only the local shard. The scales must be gathered unswizzled,
    since a swizzled scale pads each shard to 128-row tiles along K here, and
    are swizzled once afterwards. NCCL has no fp8 dtypes, so both move as bytes.
    """

    def __init__(
        self, qdata: torch.Tensor, scale: torch.Tensor, group: dist.ProcessGroup
    ) -> None:
        # Columnwise qdata is stored K-major, [M / R, K] with strides (1, M / R),
        # and the scales likewise as [K, M / 32 / R] with strides (1, K).
        # Transposed, both are contiguous with the gathered M leading or
        # trailing: the scales concatenate along M directly, the qdata along
        # its trailing dim and needs one permute afterwards.
        self.world_size = group.size()
        self.qdata_dtype, self.scale_dtype = qdata.dtype, scale.dtype
        qdata_KM = qdata.t().contiguous().view(torch.uint8)
        scale_MK = scale.t().contiguous().view(torch.uint8)
        self.qdata_out = qdata_KM.new_empty(
            self.world_size * qdata_KM.shape[0], qdata_KM.shape[1]
        )
        self.scale_out = scale_MK.new_empty(
            self.world_size * scale_MK.shape[0], scale_MK.shape[1]
        )
        self.works: list[dist.Work] = []
        for out, inp in ((self.qdata_out, qdata_KM), (self.scale_out, scale_MK)):
            work = dist.all_gather_into_tensor(out, inp, group=group, async_op=True)
            assert work is not None
            self.works.append(work)

    def wait(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Gathered ``[M, K]`` qdata and swizzled scales, as WGRAD consumes them."""
        for work in self.works:
            work.wait()
        K, rows = self.qdata_out.shape[0] // self.world_size, self.qdata_out.shape[1]
        qdata_KM = (
            self.qdata_out.view(self.world_size, K, rows)
            .permute(1, 0, 2)
            .reshape(K, self.world_size * rows)
        )
        scale = self.scale_out.view(self.scale_dtype).t()
        return qdata_KM.view(self.qdata_dtype).t(), triton_mx_block_rearrange(scale)


class _MXFP8AllGatherLinearFunction(torch.autograd.Function):
    """MXFP8 version of ``AsyncAllGatherLinear``.

        forward  y_shard_n  = all_gather(x_shard_m) @ w_shard_n.T
                 quantized rowwise per rank, then a fused all-gather + matmul
        dgrad    dx_shard_m = reduce_scatter(dy_shard_n @ w_shard_n)
                 a fused matmul + reduce-scatter
        wgrad    dw_shard_n = dy_shard_n.T @ all_gather(x_shard_m)
                 x quantized columnwise per rank and gathered as MXFP8; see
                 _ColwiseAllGather. The gather starts before dgrad so the two
                 overlap.

    Forward saves only this rank's sequence shard: BF16 ``x_shard_m``, or its
    columnwise MXFP8 operands when ``input_activation_format_for_backward`` is
    ``"mxfp8"``.
    """

    @staticmethod
    def spmd_typecheck(
        result: torch.Tensor,
        *,
        x_shard_m: torch.Tensor,
        weight: torch.Tensor,
        weight_qdata_fprop_KN: torch.Tensor,
        weight_scale_fprop_swizzled: torch.Tensor,
        weight_qdata_dgrad_NK: torch.Tensor,
        weight_scale_dgrad_swizzled: torch.Tensor,
        bias_shard_n: torch.Tensor | None,
        group: dist.ProcessGroup,
    ) -> None:
        """SPMD type: x S(0)@TP, w S(0)@TP -> y S(1)@TP."""
        spmd.assert_type(x_shard_m, {group: spmd.S(0)})
        # Output features are the second-to-last dim of a (stacked) weight.
        spmd.assert_type(weight, {group: spmd.S(weight.dim() - 2)})
        # Quantized from ``weight``, so typed by it.
        spmd.rules.ignore(
            weight_qdata_fprop_KN,
            weight_scale_fprop_swizzled,
            weight_qdata_dgrad_NK,
            weight_scale_dgrad_swizzled,
        )
        if bias_shard_n is not None:
            spmd.assert_type(bias_shard_n, {group: spmd.S(0)})
        spmd.assert_local_type_like(
            result,
            x_shard_m,
            {group: spmd.S(1)},  # pyrefly: ignore [bad-argument-type]
        )

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        x_shard_m: torch.Tensor,
        weight: torch.Tensor,
        weight_qdata_fprop_KN: torch.Tensor,
        weight_scale_fprop_swizzled: torch.Tensor,
        weight_qdata_dgrad_NK: torch.Tensor,
        weight_scale_dgrad_swizzled: torch.Tensor,
        bias_shard_n: torch.Tensor | None,
        group: dist.ProcessGroup,
        group_name: str,
        input_activation_format_for_backward: InputActivationFormatForBackward,
        inplace_wgrad_accum: bool,
    ) -> torch.Tensor:
        ensure_symm_mem_ops()
        _check_mxfp8_linear_inputs(x_shard_m, weight, bias_shard_n)
        x_shard_m = x_shard_m.contiguous()
        x_MK = x_shard_m.view(-1, x_shard_m.shape[-1])
        save_mxfp8 = input_activation_format_for_backward == "mxfp8"
        x_qdata, x_col, x_scale, x_col_scale = mxfp8_quantize_cuda(
            x_MK, rowwise=True, colwise=save_mxfp8, scaling_mode=_MXFP8_SCALING_MODE
        )
        x_scale = triton_mx_block_rearrange(x_scale)

        _, (y_shard_n,) = torch.ops.symm_mem.fused_all_gather_scaled_matmul(
            x_qdata.view(x_shard_m.shape),
            [weight_qdata_fprop_KN],
            x_scale,
            _MX,
            _SWIZZLE,
            [weight_scale_fprop_swizzled],
            _MX,
            _SWIZZLE,
            0,
            group_name,
            [bias_shard_n],
            [torch.bfloat16],
            [False],
        )

        saved_input = (x_col, x_col_scale) if save_mxfp8 else (x_shard_m,)
        ctx.num_saved_input = len(saved_input)
        ctx.input_shape = x_shard_m.shape
        ctx.save_for_backward(
            *saved_input,
            *_save_dgrad_weight_operands(
                weight, weight_qdata_dgrad_NK, weight_scale_dgrad_swizzled
            ),
        )
        ctx.group = group
        ctx.group_name = group_name
        ctx.has_bias = bias_shard_n is not None
        ctx.weight_shape = weight.shape
        # See _MXFP8LinearFunction.forward for why these are read here.
        ctx.wgrad_dtype = weight.grad_dtype or weight.dtype
        ctx.weight_param = weight if inplace_wgrad_accum else None
        return y_shard_n

    @staticmethod
    @once_differentiable
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_y_shard_n: torch.Tensor):
        saved = ctx.saved_tensors
        saved_input = saved[: ctx.num_saved_input]
        (
            weight_qdata_dgrad_NK,
            weight_scale_dgrad_swizzled,
        ) = _restore_dgrad_weight_operands(saved[ctx.num_saved_input :])
        input_shape = ctx.input_shape

        # Start gathering x for WGRAD; DGRAD does not need it, so it overlaps.
        if len(saved_input) == 1:
            (x_shard_m,) = saved_input
            x_col, x_col_scale = _quantize_colwise(x_shard_m.view(-1, input_shape[-1]))
        else:
            x_col, x_col_scale = saved_input
        x_gather = _ColwiseAllGather(x_col, x_col_scale, ctx.group)

        grad_y_MN = grad_y_shard_n.contiguous().view(-1, grad_y_shard_n.shape[-1])
        dy_row, dy_col, dy_row_scale, dy_col_scale = mxfp8_quantize_cuda(
            grad_y_MN, rowwise=True, colwise=True, scaling_mode=_MXFP8_SCALING_MODE
        )

        grad_x_MK = torch.ops.symm_mem.fused_scaled_matmul_reduce_scatter(
            dy_row,
            weight_qdata_dgrad_NK,
            triton_mx_block_rearrange(dy_row_scale),
            _MX,
            _SWIZZLE,
            weight_scale_dgrad_swizzled,
            _MX,
            _SWIZZLE,
            "sum",
            0,
            0,
            ctx.group_name,
            [grad_y_MN.shape[0], input_shape[-1]],
        )
        grad_x_shard_m = grad_x_MK.view(input_shape)

        x_col, x_col_scale = x_gather.wait()
        grad_weight = _mxfp8_wgrad(
            ctx, dy_col, triton_mx_block_rearrange(dy_col_scale), x_col, x_col_scale
        )
        grad_bias = grad_y_MN.sum(dim=0) if ctx.has_bias else None
        return (
            grad_x_shard_m,
            grad_weight,
            None,
            None,
            None,
            None,
            grad_bias,
            None,
            None,
            None,
            None,
        )


class _MXFP8LinearReduceScatterFunction(torch.autograd.Function):
    """MXFP8 version of ``AsyncLinearReduceScatter``.

    forward  y_shard_m  = reduce_scatter(x_shard_k @ w_shard_k.T)
             a fused matmul + reduce-scatter
    dgrad    dx_shard_k = all_gather(dy_shard_m) @ w_shard_k
             dy quantized rowwise per rank, then a fused all-gather + matmul
    wgrad    dw_shard_k = all_gather(dy_shard_m).T @ x_shard_k
             dy quantized columnwise per rank and gathered as MXFP8; see
             _ColwiseAllGather. The fused dgrad gathers rowwise MXFP8,
             which WGRAD cannot use, so this is a second gather. It starts
             before dgrad so the two overlap.
    dbias    [N]        = all_reduce(dy_shard_m.sum(0))
    """

    @staticmethod
    def spmd_typecheck(
        result: torch.Tensor,
        *,
        x_shard_k: torch.Tensor,
        weight: torch.Tensor,
        weight_qdata_fprop_KN: torch.Tensor,
        weight_scale_fprop_swizzled: torch.Tensor,
        weight_qdata_dgrad_NK: torch.Tensor,
        weight_scale_dgrad_swizzled: torch.Tensor,
        bias: torch.Tensor | None,
        group: dist.ProcessGroup,
    ) -> None:
        """SPMD type: x S(-1)@TP, w S(-1)@TP, bias I@TP -> y S(0)@TP."""
        spmd.assert_type(x_shard_k, {group: spmd.S(x_shard_k.dim() - 1)})
        spmd.assert_type(weight, {group: spmd.S(weight.dim() - 1)})
        # Quantized from ``weight``, so typed by it.
        spmd.rules.ignore(
            weight_qdata_fprop_KN,
            weight_scale_fprop_swizzled,
            weight_qdata_dgrad_NK,
            weight_scale_dgrad_swizzled,
        )
        if bias is not None:
            spmd.assert_type(bias, {group: spmd.I})
        spmd.assert_local_type_like(
            result,
            x_shard_k,
            {group: spmd.S(0)},  # pyrefly: ignore [bad-argument-type]
        )

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        x_shard_k: torch.Tensor,
        weight: torch.Tensor,
        weight_qdata_fprop_KN: torch.Tensor,
        weight_scale_fprop_swizzled: torch.Tensor,
        weight_qdata_dgrad_NK: torch.Tensor,
        weight_scale_dgrad_swizzled: torch.Tensor,
        bias: torch.Tensor | None,
        group: dist.ProcessGroup,
        group_name: str,
        input_activation_format_for_backward: InputActivationFormatForBackward,
        inplace_wgrad_accum: bool,
    ) -> torch.Tensor:
        ensure_symm_mem_ops()
        local_out_features = _check_mxfp8_linear_inputs(x_shard_k, weight, bias)
        x_shard_k = x_shard_k.contiguous()
        x_MK = x_shard_k.view(-1, x_shard_k.shape[-1])
        save_mxfp8 = input_activation_format_for_backward == "mxfp8"
        x_qdata, x_col, x_scale, x_col_scale = mxfp8_quantize_cuda(
            x_MK, rowwise=True, colwise=save_mxfp8, scaling_mode=_MXFP8_SCALING_MODE
        )

        y_MN = torch.ops.symm_mem.fused_scaled_matmul_reduce_scatter(
            x_qdata,
            weight_qdata_fprop_KN,
            triton_mx_block_rearrange(x_scale),
            _MX,
            _SWIZZLE,
            weight_scale_fprop_swizzled,
            _MX,
            _SWIZZLE,
            "sum",
            0,
            0,
            group_name,
            [x_MK.shape[0], local_out_features],
        )
        # Rows are scattered rank-major, so dim 0 of x carries the shard.
        y_shard_m = y_MN.view(-1, *x_shard_k.shape[1:-1], local_out_features)
        if bias is not None:
            # After the reduction, so it is added once.
            y_shard_m = y_shard_m + bias

        # WGRAD contracts over the full, local M, so x needs no gather and its
        # columnwise operands are swizzled once here or in backward.
        if save_mxfp8:
            saved_input = (x_col, triton_mx_block_rearrange(x_col_scale))
        else:
            saved_input = (x_shard_k,)
        ctx.num_saved_input = len(saved_input)
        ctx.input_shape = x_shard_k.shape
        ctx.save_for_backward(
            *saved_input,
            *_save_dgrad_weight_operands(
                weight, weight_qdata_dgrad_NK, weight_scale_dgrad_swizzled
            ),
        )
        ctx.group = group
        ctx.group_name = group_name
        ctx.has_bias = bias is not None
        ctx.weight_shape = weight.shape
        # See _MXFP8LinearFunction.forward for why these are read here.
        ctx.wgrad_dtype = weight.grad_dtype or weight.dtype
        ctx.weight_param = weight if inplace_wgrad_accum else None
        return y_shard_m

    @staticmethod
    @once_differentiable
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_y_shard_m: torch.Tensor):
        saved = ctx.saved_tensors
        saved_input = saved[: ctx.num_saved_input]
        (
            weight_qdata_dgrad_NK,
            weight_scale_dgrad_swizzled,
        ) = _restore_dgrad_weight_operands(saved[ctx.num_saved_input :])
        input_shape = ctx.input_shape

        grad_y_MN = grad_y_shard_m.contiguous().view(-1, grad_y_shard_m.shape[-1])
        dy_row, dy_col, dy_row_scale, dy_col_scale = mxfp8_quantize_cuda(
            grad_y_MN, rowwise=True, colwise=True, scaling_mode=_MXFP8_SCALING_MODE
        )
        # Start gathering dy for WGRAD; DGRAD gathers its own rowwise copy.
        dy_gather = _ColwiseAllGather(dy_col, dy_col_scale, ctx.group)

        _, (grad_x_MK,) = torch.ops.symm_mem.fused_all_gather_scaled_matmul(
            dy_row,
            [weight_qdata_dgrad_NK],
            triton_mx_block_rearrange(dy_row_scale),
            _MX,
            _SWIZZLE,
            [weight_scale_dgrad_swizzled],
            _MX,
            _SWIZZLE,
            0,
            ctx.group_name,
            [None],
            [torch.bfloat16],
            [False],
        )
        grad_x_shard_k = grad_x_MK.view(input_shape)

        if len(saved_input) == 1:
            (x_shard_k,) = saved_input
            x_col, x_col_scale = _quantize_colwise(x_shard_k.view(-1, input_shape[-1]))
            x_col_scale = triton_mx_block_rearrange(x_col_scale)
        else:
            x_col, x_col_scale = saved_input
        dy_col_full, dy_col_full_scale = dy_gather.wait()
        grad_weight = _mxfp8_wgrad(
            ctx, dy_col_full, dy_col_full_scale, x_col, x_col_scale
        )

        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_y_MN.sum(dim=0)
            dist.all_reduce(grad_bias, group=ctx.group)
        return (
            grad_x_shard_k,
            grad_weight,
            None,
            None,
            None,
            None,
            grad_bias,
            None,
            None,
            None,
            None,
        )


class MXFP8AsyncColumnParallelLinear(MXFP8Linear, ColumnParallelLinear):
    """Overlap an MXFP8 input all-gather with a column-parallel GEMM.

    Falls back to the synchronous MXFP8 column-parallel linear when there is no
    sequence-parallel TP collective to fuse, or when a rank's token rows are not
    a multiple of 128, which the fused MXFP8 ops require.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MXFP8Linear.Config):  # pyrefly: ignore [bad-override]
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tp_group = _tp_group_from_context()
        rows = math.prod(input.shape[:-1])
        if tp_group is None or not spmd_dense_sp_enabled() or rows % _ROW_ALIGNMENT:
            return super().forward(input)

        weight, bias = self.weight, self.bias
        operands = _mxfp8_weight_operands(weight)
        output = remat.region(
            _MXFP8AllGatherLinearFunction.apply,
            self.remat_region_name("linear"),
            recompute=self.remat_should_recompute("linear"),
        )(
            input,
            weight,
            operands.weight_qdata_fprop_KN,
            operands.weight_scale_fprop_swizzled,
            operands.weight_qdata_dgrad_NK,
            operands.weight_scale_dgrad_swizzled,
            None if bias is None else bias.flatten(),
            tp_group,
            tp_group.group_name,
            self.input_activation_format_for_backward,
            uses_inplace_wgrad_accum(self.inplace_wgrad_accum, weight, self),
        )
        return self._unflatten_output(output, weight)


class MXFP8AsyncRowParallelLinear(MXFP8Linear, RowParallelLinear):
    """Overlap a row-parallel MXFP8 GEMM with its output reduce-scatter.

    Falls back to the synchronous MXFP8 row-parallel linear when there is no
    sequence-parallel TP collective to fuse, or when a rank's share of the token
    rows is not a multiple of 128, which the fused MXFP8 ops require.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MXFP8Linear.Config):  # pyrefly: ignore [bad-override]
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tp_group = _tp_group_from_context()
        if (
            tp_group is None
            or not spmd_dense_sp_enabled()
            or input.shape[0] % tp_group.size()
            or math.prod(input.shape[:-1]) % (tp_group.size() * _ROW_ALIGNMENT)
        ):
            return super().forward(input)

        weight, bias = self.weight, self.bias
        operands = _mxfp8_weight_operands(weight)
        output = remat.region(
            _MXFP8LinearReduceScatterFunction.apply,
            self.remat_region_name("linear"),
            recompute=self.remat_should_recompute("linear"),
        )(
            input,
            weight,
            operands.weight_qdata_fprop_KN,
            operands.weight_scale_fprop_swizzled,
            operands.weight_qdata_dgrad_NK,
            operands.weight_scale_dgrad_swizzled,
            None if bias is None else bias.flatten(),
            tp_group,
            tp_group.group_name,
            self.input_activation_format_for_backward,
            uses_inplace_wgrad_accum(self.inplace_wgrad_accum, weight, self),
        )
        return self._unflatten_output(output, weight)
