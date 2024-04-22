# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn

import triton
import triton.language as tl


def create_norm(norm_type: str, dim: int, eps: float = 1e-6):
    """
    Creates the specified normalization layer based on the norm_type.

    Args:
        norm_type (str): The type of normalization layer to create.
            Supported types: 1. rmsnorm 2. fused_rmsnorm 3. layernorm 4. np_layernorm
        dim (int): The dimension of the normalization layer.
        eps (float, optional): The epsilon value for numerical stability. Defaults to 1e-6.

    Returns:
        The created normalization layer.

    Raises:
        NotImplementedError: If an unknown norm_type is provided.
    """
    norm_type = norm_type.lower()  # Normalize to lowercase

    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps, bias=False)
    elif norm_type == "np_layernorm":
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=False, bias=False)
    elif norm_type == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    elif norm_type == "fused_rmsnorm":
        return FusedRMSNorm(dim, eps=eps)
    else:
        raise NotImplementedError(f"Unknown norm_type: '{norm_type}'")


class FusedRMSNorm(nn.Module):
    """Fused RMS Norm, wraps a fused Triton Kernel"""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.fused_rms_norm_fn = fused_rms_norm_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """leverages Triton Fused RMS Norm kernel"""
        return self.fused_rms_norm_fn(
            x,
            self.weight,
            eps=self.eps,
        )

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore


class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore


# FusedRMSNorm in Triton

# Credit
# Tri Dao's Triton LayerNorm: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/layer_norm.py
# Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["N"],
)
@triton.jit
def _rms_norm_fwd_kernel(
    X,
    stride_x,
    Y,
    stride_y,
    W,
    Rstd,
    eps,
    M,  # num rows
    N,  # num cols
    block_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block_N)

    # Load input data and weights
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    # Compute mean and variance
    xbar = tl.where(cols < N, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    # Store the reciprocal standard deviation
    tl.store(Rstd + row, rstd)

    # Normalize and apply linear transformation
    x_hat = x * rstd
    y = x_hat * w

    # Write output
    tl.store(Y + row * stride_y + cols, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["N"],
)
@triton.jit
def _rms_norm_bwd_kernel_sm(
    X,
    stride_x,
    W,
    DY,
    stride_dy,
    DX,
    stride_dx,
    Rstd,
    DW,
    eps,
    M,  # num rows
    N,  # num cols
    rows_per_program,
    block_N: tl.constexpr,
):
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, block_N)
    mask = cols < N

    # Load weights
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    # Accumulate gradients for weights
    dw = tl.zeros((block_N,), dtype=tl.float32)

    row_end = min(row_start + rows_per_program, M)
    for row in range(row_start, row_end):
        # Load input, output gradient, and reciprocal standard deviation
        x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + row * stride_dy + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(Rstd + row)

        # Compute normalized input and gradients
        x_hat = x * rstd
        wdy = w * dy
        dw += dy * x_hat
        c1 = tl.sum(x_hat * wdy, axis=0) / N
        dx = (wdy - x_hat * c1) * rstd

        # Store input gradient
        tl.store(DX + row * stride_dx + cols, dx, mask=mask)

    # Store weight gradients
    tl.store(DW + row_block_id * N + cols, dw, mask=mask)


class TritonFusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        x_shape_start = x.shape

        # Flatten input
        x = x.view(-1, x.shape[-1])
        if x.stride(-1) != 1:
            x = x.contiguous()
        if weight.stride(-1) != 1:
            weight = weight.contiguous()

        M, N = x.shape
        y = torch.empty_like(x)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

        max_size = 65536 // x.element_size()
        block_N = min(max_size, triton.next_power_of_2(N))

        if N > block_N:
            raise ValueError(f"N {N} must be <= {block_N=}")

        grid = lambda meta: (M,)
        _rms_norm_fwd_kernel[grid](
            x,
            x.stride(0),
            y,
            y.stride(0),
            weight,
            rstd,
            eps,
            M,
            N,
            block_N,
        )

        ctx.eps = eps
        ctx.save_for_backward(x, weight, rstd)
        ctx.x_shape_start = x_shape_start

        y = y.reshape(x_shape_start)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, rstd = ctx.saved_tensors
        eps = ctx.eps
        x_shape_start = ctx.x_shape_start

        # Flatten input and output gradients
        dy = dy.view(-1, dy.shape[-1])
        if dy.stride(-1) != 1:
            dy = dy.contiguous()

        M, N = dy.shape
        dx = torch.empty_like(x)
        dw = torch.empty_like(weight)

        sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
        _dw = torch.empty((sm_count, N), dtype=torch.float32, device=weight.device)

        max_size = 65536 // x.element_size()
        block_N = min(max_size, triton.next_power_of_2(N))
        rows_per_sm = math.ceil(M / sm_count)

        if N > block_N:
            raise ValueError(f"N {N} must be <= {block_N=}")

        grid = lambda meta: (sm_count,)
        _rms_norm_bwd_kernel_sm[grid](
            x,
            x.stride(0),
            weight,
            dy,
            dy.stride(0),
            dx,
            dx.stride(0),
            rstd,
            _dw,
            eps,
            M,
            N,
            rows_per_sm,
            block_N,
        )
        dw = _dw.sum(0).to(weight.dtype)
        dx = dx.view(x_shape_start)
        return dx, dw, None


# expose fusedRMSNorm as a function
def fused_rms_norm_fn(
    x,
    weight,
    eps=1e-6,
):
    return TritonFusedRMSNorm.apply(
        x,
        weight,
        eps,
    )
