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
    R: routed-token rows of the expert MLP (moe.py's name for M)
    D: model dim, the MLP's input and output features
    F: hidden dim; FC1's N is FC2's K, so the fused MLP keeps moe.py's suffixes
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
from torchao.prototype.mx_formats.utils import to_blocked

from .._fsdp_tensor import _UnshardedFSDPTensor

from ._common import (
    _INPUT_ACTIVATION_FORMATS_FOR_BACKWARD,
    _MXFP8_BLOCK_SIZE,
    _MXFP8_FUSED_MLP_DIM_ALIGNMENT,
    _MXFP8_FUSED_MLP_ROW_ALIGNMENT,
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


# ---------------------------------------------------------------------------
# Fused expert MLP over TorchAO's cuDNN grouped-GEMM + SwiGLU + quantization ops
# ---------------------------------------------------------------------------
#
# The four ``torchao::mxfp8_grouped_gemm_*_cudnn`` ops each fuse a ragged
# grouped GEMM with the SwiGLU (or its derivative) and the MXFP8 quantization
# of their output, so the expert MLP runs as: FC1 + SwiGLU + quantize (one
# launch), FC2 (one launch); backward: FC2 dgrad + dSwiGLU + quantize (one
# launch), FC1 dgrad (one launch), two weight gradients. They consume the same
# 32x32-tile weight operands FSDP caches for the per-GEMM path: the FC2
# operands as they are, and FC1's from the gate and up operands interleaved in
# 32-row bands in the fp8 domain (a square tile never straddles a band), so a
# fused step quantizes no weight.
#
# Contract (TorchAO ``cudnn_grouped_mlp``): every expert's token group and
# the allocated row count are multiples of 256 (the kernels' fixed group
# padding; 128-aligned groups corrupt silently), D and F are multiples of 128,
# SM 10.0. Rows past ``offs[-1]`` of the kernel-allocated outputs are garbage
# and never read.

# The blocked scale layout tiles a logical ``[rows, cols/32]`` scale matrix in
# squares of 128 rows by 4 scale columns (128 elements), stored as (32, 4, 4).
_SCALE_TILE_SIDE = 128


def _interleave_rows(a_ENK: torch.Tensor, b_ENK: torch.Tensor) -> torch.Tensor:
    """``(E, N, K)`` pair -> ``(E, 2N, K)`` alternating 32-row bands, the
    cuDNN GLU row order ``[a0(32) | b0(32) | a1(32) | b1(32) | ...]``."""
    e, n, k = a_ENK.shape
    bands = (e, n // _MXFP8_BLOCK_SIZE, _MXFP8_BLOCK_SIZE, k)
    return torch.stack([a_ENK.view(bands), b_ENK.view(bands)], dim=2).view(e, 2 * n, k)


def _interleave_cols(a_ENK: torch.Tensor, b_ENK: torch.Tensor) -> torch.Tensor:
    """``(E, N, K)`` pair -> ``(E, N, 2K)`` alternating 32-column bands."""
    e, n, k = a_ENK.shape
    bands = (e, n, k // _MXFP8_BLOCK_SIZE, _MXFP8_BLOCK_SIZE)
    return torch.stack([a_ENK.view(bands), b_ENK.view(bands)], dim=3).view(e, n, 2 * k)


def _interleave_row_scales(
    a: torch.Tensor, b: torch.Tensor, *, rows: int, cols: int
) -> torch.Tensor:
    """Blocked scales of two logical ``[rows, cols/32]`` scale matrices -> the
    blocked scales of their 32-row-band interleave ``[2*rows, cols/32]``.

    The blocked layout stores each 128x4 tile as ``(32, 4, 4)``: row within
    the band, band within the tile, column. A band is one index of the middle
    axis, so the interleave is a permute of whole 32x4 slabs.
    """
    e = a.shape[0]
    tiles = (
        e,
        rows // _SCALE_TILE_SIDE,
        cols // _SCALE_TILE_SIDE,
        _MXFP8_BLOCK_SIZE,
        4,
        4,
    )

    def by_band(scales):  # (E, rows/32, cols/128, 32, 4)
        return (
            scales.view(tiles)
            .permute(0, 1, 4, 2, 3, 5)
            .reshape(
                e,
                rows // _MXFP8_BLOCK_SIZE,
                cols // _SCALE_TILE_SIDE,
                _MXFP8_BLOCK_SIZE,
                4,
            )
        )

    interleaved = torch.stack([by_band(a), by_band(b)], dim=2).view(
        e,
        2 * rows // _SCALE_TILE_SIDE,
        4,
        cols // _SCALE_TILE_SIDE,
        _MXFP8_BLOCK_SIZE,
        4,
    )
    return interleaved.permute(0, 1, 3, 4, 2, 5).reshape(e, -1)


def _interleave_col_scales(
    a: torch.Tensor, b: torch.Tensor, *, rows: int, cols: int
) -> torch.Tensor:
    """Blocked scales of two logical ``[rows, cols/32]`` scale matrices -> the
    blocked scales of their 32-column-band interleave ``[rows, 2*cols/32]``: a
    scale column is one index of the tile's last axis."""
    e = a.shape[0]
    tiles = (
        e,
        rows // _SCALE_TILE_SIDE,
        cols // _SCALE_TILE_SIDE,
        _MXFP8_BLOCK_SIZE,
        4,
        4,
    )

    def by_band(scales):  # (E, rows/128, cols/32, 32, 4)
        return (
            scales.view(tiles)
            .permute(0, 1, 2, 5, 3, 4)
            .reshape(
                e,
                rows // _SCALE_TILE_SIDE,
                cols // _MXFP8_BLOCK_SIZE,
                _MXFP8_BLOCK_SIZE,
                4,
            )
        )

    interleaved = torch.stack([by_band(a), by_band(b)], dim=3).view(
        e,
        rows // _SCALE_TILE_SIDE,
        2 * cols // _SCALE_TILE_SIDE,
        4,
        _MXFP8_BLOCK_SIZE,
        4,
    )
    return interleaved.permute(0, 1, 2, 4, 5, 3).reshape(e, -1)


def _w13_fprop_operands(
    w1_qdata_fprop_EDF, w1_scale_fprop, w3_qdata_fprop_EDF, w3_scale_fprop
):
    """The FC1 operand of the fused forward: ``(E, 2F, D)`` gate and up rows
    interleaved in 32-row bands, quantized along D, plus its blocked scales."""
    _, d, f = w1_qdata_fprop_EDF.shape
    qdata_E2FD = _interleave_rows(
        w1_qdata_fprop_EDF.transpose(-2, -1), w3_qdata_fprop_EDF.transpose(-2, -1)
    )
    scales = _interleave_row_scales(w1_scale_fprop, w3_scale_fprop, rows=f, cols=d)
    return qdata_E2FD, scales


def _w13_dgrad_operands(
    w1_qdata_dgrad_EFD, w1_scale_dgrad, w3_qdata_dgrad_EFD, w3_scale_dgrad
):
    """The FC1 operand of the fused activation gradient: ``(E, D, 2F)`` with
    the gate and up features interleaved in 32-column bands along the
    contraction axis, plus the blocked scales of the logical ``[D, 2F/32]``."""
    _, f, d = w1_qdata_dgrad_EFD.shape
    qdata_ED2F = _interleave_cols(
        w1_qdata_dgrad_EFD.transpose(-2, -1), w3_qdata_dgrad_EFD.transpose(-2, -1)
    )
    scales = _interleave_col_scales(w1_scale_dgrad, w3_scale_dgrad, rows=d, cols=f)
    return qdata_ED2F, scales


def _split_w13_grad(grad_E2FD: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The fused FC1 weight gradient, in the interleaved row order, back to
    the stock gate and up gradients ``(E, F, D)``."""
    e, two_f, d = grad_E2FD.shape
    f = two_f // 2
    bands = grad_E2FD.view(e, f // _MXFP8_BLOCK_SIZE, 2, _MXFP8_BLOCK_SIZE, d)
    return bands[:, :, 0].reshape(e, f, d), bands[:, :, 1].reshape(e, f, d)


def _pad_offsets_pow2(offs: torch.Tensor) -> torch.Tensor:
    # The K-groups scale swizzle sizes a tl.arange by the group count, which
    # Triton requires to be a power of two; repeated end offsets are zero-size
    # groups the kernel skips.
    e = offs.shape[0]
    e_pow2 = 1 << (e - 1).bit_length()
    if e_pow2 == e:
        return offs
    return torch.cat([offs, offs[-1:].expand(e_pow2 - e)])


def _blocked_rowwise_scales(scales: torch.Tensor) -> torch.Tensor:
    """Rowwise 1x32 scales of ``[M, K]`` -> the flat whole-matrix blocked
    buffer the ops read (equal to the per-group concatenation because every
    group is a 256-multiple, so 128-row tiles never straddle groups)."""
    return to_blocked(scales).reshape(-1)


def _blocked_colwise_scales(
    scales: torch.Tensor, offs: torch.Tensor, *, rows: int
) -> torch.Tensor:
    """Columnwise 32x1 scales (logical ``[K, M/32]``) of ``[M, K]`` -> the flat
    per-group blocked buffer of ``K * M/32`` bytes the WGRAD op reads. The
    swizzle lays the groups out back to back and pads only past them, so the
    static slice drops that tail without a device sync."""
    blocked = triton_mx_block_rearrange_2d_K_groups(
        scales, _pad_offsets_pow2(offs // _MXFP8_BLOCK_SIZE)
    )
    return blocked.reshape(-1)[: scales.shape[0] * (rows // _MXFP8_BLOCK_SIZE)]


def _validate_fused_mlp_inputs(
    x_RD: torch.Tensor, w1_EFD: torch.Tensor, offsets_E: torch.Tensor
) -> None:
    """Checks at the module's forward: the local expert dims (shards under
    tensor parallelism) and the routing-dependent token count on the host, the
    group boundaries on the device."""
    _, f, d = w1_EFD.shape
    if f % _MXFP8_FUSED_MLP_DIM_ALIGNMENT or d % _MXFP8_FUSED_MLP_DIM_ALIGNMENT:
        raise ValueError(
            "MXFP8 fuse_grouped_mlp requires the local expert dimensions to be "
            f"multiples of {_MXFP8_FUSED_MLP_DIM_ALIGNMENT}; got hidden_dim={f}, "
            f"dim={d}. Choose a tensor_parallel_degree that keeps the shards "
            "aligned, or disable the fusion for this model."
        )
    row_alignment = _MXFP8_FUSED_MLP_ROW_ALIGNMENT
    # R is an unbacked SymInt under compile, so identity tests: literal bools
    # raise, SymBools become deferred runtime asserts. The >= and % 32 forms
    # are implied by the row multiple but recorded separately: the kernel
    # wrappers and GEMM metas check exactly those forms, and the symbolic
    # engine matches expressions rather than deriving them.
    r = x_RD.shape[0]
    for cond, requirement in (
        (r >= row_alignment, f"at least {row_alignment}"),
        (
            r % row_alignment == 0,
            f"a multiple of {row_alignment} (configure the token dispatcher with "
            f"pad_multiple={row_alignment})",
        ),
        (r % _MXFP8_BLOCK_SIZE == 0, f"a multiple of {_MXFP8_BLOCK_SIZE}"),
    ):
        if cond is False:
            raise ValueError(
                f"MXFP8 fuse_grouped_mlp: token count {r} must be {requirement}; "
                "there is no silent fallback."
            )
        if cond is not True:
            torch._check(
                cond,
                lambda: f"MXFP8 fuse_grouped_mlp: token count must be {requirement}",
            )
    # Group boundaries must be aligned too (the dispatcher's pad_multiple
    # guarantees it). Their values live on the device, so a host-side check
    # would sync; a device-side assertion keeps the stream ordered and fails
    # at the next sync instead of corrupting silently. It runs after the host
    # checks so a rejected buffer never enqueues a kernel.
    torch._assert_async(
        (offsets_E % row_alignment == 0).all(),
        "MXFP8 fuse_grouped_mlp: every expert's token group must be a multiple "
        f"of {row_alignment} rows (configure the token dispatcher with "
        f"pad_multiple={row_alignment})",
    )


# Lives in TorchTitan for the same reason as _MXFP8GroupedMMFunction: it
# reads the weight operands off FSDP's unsharded tensors in both passes.
@torch._dynamo.allow_in_graph
class _MXFP8FusedGroupedMLPFunction(torch.autograd.Function):
    """``x_RD [R, D] -> y_RD [R, D]``, the SwiGLU expert MLP over the four
    fused cuDNN ops, on the cached 32x32 weight operands.

    ``w1_EFD``/``w3_EFD``/``w2_EDF`` are the stock parameters (FSDP wrappers
    or plain tensors) that receive the gradients; the twelve operand tensors
    are their FPROP and DGRAD qdata and blocked scales for this call.
    ``offs`` holds int32 exclusive-end row offsets of the 256-row-padded
    groups, ``offs[-1] <= R``; rows past it in ``y_RD``/``grad_x_RD`` are
    left unwritten.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        x_RD: torch.Tensor,
        w1_EFD: torch.Tensor,
        w3_EFD: torch.Tensor,
        w2_EDF: torch.Tensor,
        w1_qdata_fprop_EDF: torch.Tensor,
        w1_scale_fprop: torch.Tensor,
        w1_qdata_dgrad_EFD: torch.Tensor,
        w1_scale_dgrad: torch.Tensor,
        w3_qdata_fprop_EDF: torch.Tensor,
        w3_scale_fprop: torch.Tensor,
        w3_qdata_dgrad_EFD: torch.Tensor,
        w3_scale_dgrad: torch.Tensor,
        w2_qdata_fprop_EFD: torch.Tensor,
        w2_scale_fprop: torch.Tensor,
        w2_qdata_dgrad_EDF: torch.Tensor,
        w2_scale_dgrad: torch.Tensor,
        offs: torch.Tensor,
        input_activation_format_for_backward: InputActivationFormatForBackward,
    ) -> torch.Tensor:
        # Imported here, not at module scope: the ops ship separately from the
        # per-GEMM path's kernels, and converter.py imports this module inside
        # the try that decides whether MXFP8 is available at all.
        from torchao.prototype.moe_training.kernels.mxfp8.cudnn_grouped_mlp import (
            mxfp8_grouped_gemm_cudnn,
            mxfp8_grouped_gemm_swiglu_fwd_cudnn,
        )

        if x_RD.dtype != torch.bfloat16 or any(
            weight.dtype != torch.bfloat16 for weight in (w1_EFD, w3_EFD, w2_EDF)
        ):
            raise ValueError(
                "MXFP8 fused grouped experts require BF16 activations and weights; "
                f"got activation dtype {x_RD.dtype} and weight dtypes "
                f"{w1_EFD.dtype}, {w3_EFD.dtype}, {w2_EDF.dtype}."
            )
        x_RD = x_RD.contiguous()
        requires_wgrad = any(ctx.needs_input_grad[1:4])
        quantize_wgrad_input_in_forward = (
            requires_wgrad and input_activation_format_for_backward == "mxfp8"
        )

        x_row_RD, _, x_row_scales, _ = mxfp8_quantize_cuda(
            x_RD, rowwise=True, colwise=False, scaling_mode=_MXFP8_SCALING_MODE
        )
        w13_qdata_E2FD, w13_scale = _w13_fprop_operands(
            w1_qdata_fprop_EDF, w1_scale_fprop, w3_qdata_fprop_EDF, w3_scale_fprop
        )
        # FC1 + SwiGLU + quantization: the BF16 pre-activation feeds the
        # backward op; ``h`` comes out quantized both ways, rowwise for FC2 and
        # columnwise for the FC2 weight gradient.
        (
            z_R2F,
            h_row_RF,
            h_row_scales,
            h_col_RF,
            h_col_scales,
        ) = mxfp8_grouped_gemm_swiglu_fwd_cudnn(
            x_row_RD,
            _blocked_rowwise_scales(x_row_scales),
            w13_qdata_E2FD,
            w13_scale,
            offs,
        )
        # FC2: the right operand is w2 quantized along F, i.e. the FPROP
        # operand viewed in its stored (E, D, F) orientation.
        y_RD = mxfp8_grouped_gemm_cudnn(
            h_row_RF,
            h_row_scales,
            w2_qdata_fprop_EFD.transpose(-2, -1),
            w2_scale_fprop,
            offs,
        )

        # Save exactly one input-activation operands for WGRAD, and let FSDP
        # own the weight operands whenever it manages them: an unsharded
        # weight carries operands FSDP will refill before backward, so save
        # the wrapper and read them off it then. Anything else has none, so
        # its DGRAD operands are saved directly.
        saved_weight_tensors = []
        unsharded_weights = []
        for weight, qdata_dgrad, scale_dgrad in (
            (w1_EFD, w1_qdata_dgrad_EFD, w1_scale_dgrad),
            (w3_EFD, w3_qdata_dgrad_EFD, w3_scale_dgrad),
            (w2_EDF, w2_qdata_dgrad_EDF, w2_scale_dgrad),
        ):
            is_unsharded = isinstance(weight, _UnshardedFSDPTensor)
            unsharded_weights.append(is_unsharded)
            saved_weight_tensors.extend(
                (weight,) if is_unsharded else (qdata_dgrad, scale_dgrad)
            )
        if quantize_wgrad_input_in_forward:
            _, x_col_RD, _, x_col_scales = mxfp8_quantize_cuda(
                x_RD, rowwise=False, colwise=True, scaling_mode=_MXFP8_SCALING_MODE
            )
            ctx.save_for_backward(
                z_R2F,
                h_col_RF,
                h_col_scales,
                offs,
                x_col_RD,
                _blocked_colwise_scales(x_col_scales, offs, rows=x_RD.shape[0]),
                *saved_weight_tensors,
            )
        else:
            ctx.save_for_backward(
                z_R2F, h_col_RF, h_col_scales, offs, x_RD, *saved_weight_tensors
            )
        ctx.requires_dgrad = ctx.needs_input_grad[0]
        ctx.requires_wgrad = requires_wgrad
        ctx.saved_quantized_input = quantize_wgrad_input_in_forward
        ctx.unsharded_weights = tuple(unsharded_weights)
        return y_RD

    @staticmethod
    @once_differentiable
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_y_RD: torch.Tensor):
        # Same import placement as forward.
        from torchao.prototype.moe_training.kernels.mxfp8.cudnn_grouped_mlp import (
            mxfp8_grouped_gemm_cudnn,
            mxfp8_grouped_gemm_dswiglu_bwd_cudnn,
            mxfp8_grouped_gemm_wgrad_cudnn,
        )

        saved_tensors = list(ctx.saved_tensors)
        z_R2F, h_col_RF, h_col_scales, offs = saved_tensors[:4]
        if ctx.saved_quantized_input:
            x_col_RD, x_col_scales = saved_tensors[4:6]
            x_hp_RD = None
            saved_weight_tensors = saved_tensors[6:]
        else:
            x_hp_RD = saved_tensors[4]
            x_col_RD = x_col_scales = None
            saved_weight_tensors = saved_tensors[5:]

        dgrad_operands = []
        for is_unsharded in ctx.unsharded_weights:
            if is_unsharded:
                weight = saved_weight_tensors.pop(0)
                if not isinstance(weight, _UnshardedFSDPTensor):
                    raise RuntimeError("FSDP restored an incompatible MXFP8 weight")
                operands = weight.operands
                if operands is None:
                    raise RuntimeError(
                        "FSDP did not build MXFP8 weight state for backward"
                    )
                dgrad_operands.append(
                    (
                        operands.weight_qdata_dgrad_ENK,
                        operands.weight_scale_dgrad_swizzled,
                    )
                )
            else:
                dgrad_operands.append(
                    (saved_weight_tensors.pop(0), saved_weight_tensors.pop(0))
                )
        (
            (w1_qdata_dgrad_EFD, w1_scale_dgrad),
            (w3_qdata_dgrad_EFD, w3_scale_dgrad),
            (w2_qdata_dgrad_EDF, w2_scale_dgrad),
        ) = dgrad_operands

        grad_y_RD = grad_y_RD.contiguous()
        rows = grad_y_RD.shape[0]
        dy_row_RD, dy_col_RD, dy_row_scales, dy_col_scales = mxfp8_quantize_cuda(
            grad_y_RD,
            rowwise=True,
            colwise=ctx.requires_wgrad,
            scaling_mode=_MXFP8_SCALING_MODE,
        )
        # FC2 dgrad + dSwiGLU + quantization: the right operand is w2
        # quantized along D, i.e. the DGRAD operand as stored. ``dz`` comes
        # out quantized both ways, rowwise for the FC1 dgrad GEMM and
        # columnwise for the FC1 weight gradient.
        (
            dz_row_R2F,
            dz_row_scales,
            dz_col_R2F,
            dz_col_scales,
        ) = mxfp8_grouped_gemm_dswiglu_bwd_cudnn(
            dy_row_RD,
            _blocked_rowwise_scales(dy_row_scales),
            w2_qdata_dgrad_EDF,
            w2_scale_dgrad,
            z_R2F,
            offs,
        )

        grad_x_RD = None
        if ctx.requires_dgrad:
            w13_qdata_ED2F, w13_scale = _w13_dgrad_operands(
                w1_qdata_dgrad_EFD, w1_scale_dgrad, w3_qdata_dgrad_EFD, w3_scale_dgrad
            )
            grad_x_RD = mxfp8_grouped_gemm_cudnn(
                dz_row_R2F, dz_row_scales, w13_qdata_ED2F, w13_scale, offs
            )

        grad_w1_EFD = grad_w3_EFD = grad_w2_EDF = None
        if ctx.requires_wgrad:
            grad_w2_EDF = mxfp8_grouped_gemm_wgrad_cudnn(
                dy_col_RD,
                _blocked_colwise_scales(dy_col_scales, offs, rows=rows),
                h_col_RF,
                h_col_scales,
                offs,
            )
            if x_col_RD is None:
                assert x_hp_RD is not None
                _, x_col_RD, _, x_col_scales_unblocked = mxfp8_quantize_cuda(
                    x_hp_RD,
                    rowwise=False,
                    colwise=True,
                    scaling_mode=_MXFP8_SCALING_MODE,
                )
                x_col_scales = _blocked_colwise_scales(
                    x_col_scales_unblocked, offs, rows=rows
                )
            grad_w1_EFD, grad_w3_EFD = _split_w13_grad(
                mxfp8_grouped_gemm_wgrad_cudnn(
                    dz_col_R2F, dz_col_scales, x_col_RD, x_col_scales, offs
                )
            )

        return (grad_x_RD, grad_w1_EFD, grad_w3_EFD, grad_w2_EDF, *([None] * 14))


# Local-only for SPMD type checking; see the note on _MXFP8GroupedMMFunction.
spmd.register_local_autograd_function(_MXFP8FusedGroupedMLPFunction)


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

            fuse_grouped_mlp: bool = False
            """Run the expert SwiGLU MLP as fused grouped-GEMM + SwiGLU +
            MXFP8-quantization kernels (TorchAO's ``cudnn_grouped_mlp`` ops)
            instead of three grouped GEMMs with the SwiGLU in BF16 between them.

            The fused ops consume the same FSDP-managed 32x32 weight operands
            as the per-GEMM path (the gate and up operands interleaved in the
            fp8 domain), so they add no weight quantization. They require the
            routed token groups padded to multiples of 256, ``dim`` and
            ``hidden_dim`` multiples of 128 (per rank under tensor
            parallelism) and SM 10.0; see ``MXFP8GroupedExpertsConverter``.
            The fusion applies to the stock ``GroupedExperts`` forward; a
            variant with its own forward (``GptOssGroupedExperts``,
            ``KimiGroupedExperts``, ``FusedGroupedExperts``) never reaches
            ``_grouped_mlp`` and keeps the per-GEMM path, so the converter
            rejects the flag for it.
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
            self.fuse_grouped_mlp = config.fuse_grouped_mlp
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

        def _grouped_mlp(self, *, x_RD, w1_EFD, w2_EDF, w3_EFD, offsets_E):
            if not self.fuse_grouped_mlp:
                return super()._grouped_mlp(
                    x_RD=x_RD,
                    w1_EFD=w1_EFD,
                    w2_EDF=w2_EDF,
                    w3_EFD=w3_EFD,
                    offsets_E=offsets_E,
                )
            _validate_fused_mlp_inputs(x_RD, w1_EFD, offsets_E)
            # Same operand selection as ``_grouped_mm``: an FSDP-unsharded weight
            # carries this unshard lifetime's cached operands; anything else is
            # quantized here, from the BF16 storage.
            with torch.no_grad():
                w1, w3, w2 = (
                    w.operands
                    if isinstance(w, _UnshardedFSDPTensor)
                    else _quantize_mxfp8_grouped_weight(
                        w._tensor
                        if isinstance(w, _GroupedExpertsShardedTensorWithMXFP8Compute)
                        else w
                    )
                    for w in (w1_EFD, w3_EFD, w2_EDF)
                )
            # The weights are passed as they are (FSDP wrappers included) so
            # autograd returns the gradients to the parameters; the output
            # takes ``x_RD``'s dtype like the stock MLP.
            return _MXFP8FusedGroupedMLPFunction.apply(
                x_RD.bfloat16(),
                w1_EFD,
                w3_EFD,
                w2_EDF,
                w1.weight_qdata_fprop_EKN,
                w1.weight_scale_fprop_swizzled,
                w1.weight_qdata_dgrad_ENK,
                w1.weight_scale_dgrad_swizzled,
                w3.weight_qdata_fprop_EKN,
                w3.weight_scale_fprop_swizzled,
                w3.weight_qdata_dgrad_ENK,
                w3.weight_scale_dgrad_swizzled,
                w2.weight_qdata_fprop_EKN,
                w2.weight_scale_fprop_swizzled,
                w2.weight_qdata_dgrad_ENK,
                w2.weight_scale_dgrad_swizzled,
                offsets_E,
                self.input_activation_format_for_backward,
            ).type_as(x_RD)

    MXFP8GroupedExperts.__name__ = f"MXFP8{parent_cls.__name__}"
    MXFP8GroupedExperts.__qualname__ = f"MXFP8{parent_cls.__name__}"
    _mxfp8_experts_cache[parent_cls] = MXFP8GroupedExperts
    return MXFP8GroupedExperts
