# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bilinear ``grid_sample`` with a deterministic input-gradient backward.

The CUDA ``grid_sample`` backward accumulates with atomic adds, so its input
gradient changes from run to run. This op runs the same forward and computes
the input gradient as a matmul with a dense bilinear weight matrix.

Shape suffixes: N = batch, C = channels, H/W = input height/width,
P = sampled points (output height x output width), R = H x W.

TODO: Switch to the upstream deterministic grid_sample backward once
pytorch/pytorch#196856 lands.
"""

import torch
import torch.nn.functional as F


@torch.library.custom_op("torchtitan::deterministic_grid_sample", mutates_args=())
def deterministic_grid_sample(input: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """``F.grid_sample`` with bilinear mode, zeros padding, align_corners=False.

    ``grid`` must not require grad: only the input gradient is computed.
    """
    return F.grid_sample(
        input, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )


@deterministic_grid_sample.register_fake
def _(input: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    return input.new_empty(input.shape[0], input.shape[1], *grid.shape[1:3])


def _input_grad(
    grad_output: torch.Tensor, grid: torch.Tensor, input_shape: torch.Size
) -> torch.Tensor:
    N, C, H, W = input_shape
    num_points = grid.shape[1] * grid.shape[2]
    # Accumulate in at least fp32, as the CUDA kernel does for low precision.
    compute_dtype = torch.promote_types(grad_output.dtype, torch.float32)
    grid_NP2 = grid.reshape(N, num_points, 2).to(compute_dtype)
    # Unnormalize with align_corners=False: -1 and 1 are the outer pixel edges.
    x_NP = ((grid_NP2[..., 0] + 1) * W - 1) / 2
    y_NP = ((grid_NP2[..., 1] + 1) * H - 1) / 2
    x0_NP = torch.floor(x_NP)
    y0_NP = torch.floor(y_NP)
    wx1_NP = x_NP - x0_NP
    wy1_NP = y_NP - y0_NP

    # Dense bilinear weights, 4 nonzeros per point. The input grad is then one
    # matmul, which is deterministic. The matrix is P x H*W, so its cost grows
    # with the table size.
    pixel_R = torch.arange(H * W, device=grid.device)
    weight_NPR = grid_NP2.new_zeros(N, num_points, H * W)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        xs_NP = x0_NP + dx
        ys_NP = y0_NP + dy
        weight_NP = (wx1_NP if dx else 1 - wx1_NP) * (wy1_NP if dy else 1 - wy1_NP)
        # Zeros padding: out-of-bounds corners get weight 0, so it does not
        # matter which pixel their index lands on.
        in_bounds_NP = (xs_NP >= 0) & (xs_NP <= W - 1) & (ys_NP >= 0) & (ys_NP <= H - 1)
        weight_NP = torch.where(in_bounds_NP, weight_NP, 0)
        pixel_NP = ys_NP.long() * W + xs_NP.long()
        weight_NPR += torch.where(
            pixel_NP.unsqueeze(-1) == pixel_R, weight_NP.unsqueeze(-1), 0
        )
    grad_NPC = grad_output.reshape(N, C, num_points).transpose(1, 2).to(compute_dtype)
    grad_NRC = weight_NPR.transpose(1, 2) @ grad_NPC
    return grad_NRC.reshape(N, H, W, C).permute(0, 3, 1, 2).to(grad_output.dtype)


def _backward(
    ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
) -> tuple[torch.Tensor, None]:
    (grid,) = ctx.saved_tensors  # pyrefly: ignore[missing-attribute]
    input_shape = ctx.input_shape  # pyrefly: ignore[missing-attribute]
    return _input_grad(grad_output, grid, input_shape), None


def _setup_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple[torch.Tensor, torch.Tensor],
    output: torch.Tensor,
) -> None:
    input, grid = inputs
    if grid.requires_grad:
        raise ValueError("deterministic_grid_sample does not compute a grid gradient")
    ctx.save_for_backward(grid)
    ctx.input_shape = input.shape  # pyrefly: ignore[missing-attribute]


deterministic_grid_sample.register_autograd(_backward, setup_context=_setup_context)
