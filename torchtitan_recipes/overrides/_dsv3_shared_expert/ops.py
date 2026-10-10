# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tensor contracts for shared-expert MXFP8 GEMM epilogues.

Shape suffixes: T = tokens, D = model width, F = hidden width, P = packed 2F.
Prepared FP8 operands are nondifferentiable; autograd lives in ``autograd``.
"""

from typing import cast

import torch
from torch._subclasses.fake_tensor import FakeTensor


ACCEPTED = False
PRODUCTION_SHAPE = (4096, 2048, 7168)
_VALIDATED_SHAPES = {(128, 128, 128), PRODUCTION_SHAPE}


def _check_operands(x_TD, weight, x_scale, weight_scale, *, packed_TP=None):
    if x_TD.ndim != 2 or weight.ndim != 2:
        raise ValueError("Shared-expert epilogues require matrix operands")
    tokens, dim = x_TD.shape
    hidden = weight.shape[0] // 2 if packed_TP is None else weight.shape[1]
    expected_weight = (2 * hidden, dim) if packed_TP is None else (dim, hidden)
    if (tokens, hidden, dim) not in _VALIDATED_SHAPES:
        raise ValueError("Shared-expert epilogues require a validated T/F/D shape")
    if weight.shape != expected_weight:
        raise ValueError("Shared-expert weight shape does not match its projection")
    if x_TD.dtype != torch.float8_e4m3fn or weight.dtype != x_TD.dtype:
        raise ValueError("Shared-expert epilogues require E4M3 matrix operands")
    for scale, count in (
        (x_scale, x_TD.numel() // 32),
        (weight_scale, weight.numel() // 32),
    ):
        if scale.dtype != torch.float8_e8m0fnu or scale.numel() != count:
            raise ValueError("Shared-expert epilogues require blocked E8M0 scales")
    values = (x_TD, weight, x_scale, weight_scale)
    if packed_TP is not None:
        if packed_TP.shape != (tokens, 2 * hidden) or packed_TP.dtype != torch.bfloat16:
            raise ValueError("Backward requires the saved packed BF16 gate/up tensor")
        values += (packed_TP,)
    for value in values:
        if (
            not value.is_cuda
            or value.device != x_TD.device
            or not value.is_contiguous()
        ):
            raise ValueError("Prepared operands must be contiguous on one CUDA device")
        if value.requires_grad:
            raise ValueError("Use the shared-expert autograd adapter for gradients")
        if not isinstance(value, FakeTensor) and value.data_ptr() % 16:
            raise ValueError("MXFP8 TMA operands require 16-byte alignment")
    return tokens, hidden


@torch.library.custom_op(
    "torchtitan::dsv3_shared_expert_forward", mutates_args=(), device_types="cuda"
)
def shared_expert_forward_op(
    x_row_TD: torch.Tensor,
    w13_PD: torch.Tensor,
    x_scale: torch.Tensor,
    w13_scale: torch.Tensor,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Fuse W13, BF16 SwiGLU, and both hidden MXFP8 orientations in one launch."""
    from .kernels.forward import shared_w13_swiglu

    _check_operands(x_row_TD, w13_PD, x_scale, w13_scale)
    return cast(
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        shared_w13_swiglu(
            x_row_TD, w13_PD, x_scale, w13_scale, quantize_output=True, maxnreg=160
        ),
    )


@shared_expert_forward_op.register_fake
def _forward_fake(x_row_TD, w13_PD, x_scale, w13_scale):
    tokens, hidden = _check_operands(x_row_TD, w13_PD, x_scale, w13_scale)
    return (
        x_row_TD.new_empty((tokens, hidden), dtype=torch.bfloat16),
        x_row_TD.new_empty((tokens, 2 * hidden), dtype=torch.bfloat16),
        x_row_TD.new_empty((tokens, hidden)),
        torch.empty_strided(
            (tokens, hidden), (1, tokens), dtype=x_row_TD.dtype, device=x_row_TD.device
        ),
        x_scale.new_empty((tokens * hidden // 32,)),
        x_scale.new_empty((tokens * hidden // 32,)),
    )


@torch.library.custom_op(
    "torchtitan::dsv3_shared_expert_backward", mutates_args=(), device_types="cuda"
)
def shared_expert_backward_op(
    dy_row_TD: torch.Tensor,
    w2_DF: torch.Tensor,
    dy_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    packed_TP: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse W2 DGRAD, SwiGLU backward, packing, and both gradient quantizations."""
    from .kernels.backward import launch_swiglu_backward

    _check_operands(dy_row_TD, w2_DF, dy_scale, w2_scale, packed_TP=packed_TP)
    return cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        launch_swiglu_backward(dy_row_TD, w2_DF, dy_scale, w2_scale, packed_TP),
    )


@shared_expert_backward_op.register_fake
def _backward_fake(dy_row_TD, w2_DF, dy_scale, w2_scale, packed_TP):
    tokens, hidden = _check_operands(
        dy_row_TD, w2_DF, dy_scale, w2_scale, packed_TP=packed_TP
    )
    return (
        dy_row_TD.new_empty((tokens, 2 * hidden)),
        torch.empty_strided(
            (tokens, 2 * hidden),
            (1, tokens),
            dtype=dy_row_TD.dtype,
            device=dy_row_TD.device,
        ),
        dy_scale.new_empty((tokens * 2 * hidden // 32,)),
        dy_scale.new_empty((tokens * 2 * hidden // 32,)),
    )
