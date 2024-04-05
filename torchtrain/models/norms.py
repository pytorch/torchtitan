# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import math

from abc import abstractmethod

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


def create_norm(norm_type: str, dim: int, eps: float = 1e-6):
    """
    Creates the specified normalization layer based on the norm_type.

    Args:
        norm_type (str): The type of normalization layer to create.
            Supported types: 'layernorm', 'nplayernorm', 'rmsnorm', 'fusedrmsnorm'.
        dim (int): The dimension of the normalization layer.
        eps (float, optional): The epsilon value for numerical stability. Defaults to 1e-6.

    Returns:
        The created normalization layer.

    Raises:
        NotImplementedError: If an unknown norm_type is provided.
    """
    norm_type = norm_type.lower()  # Normalize to lowercase

    if norm_type in ("layernorm", "layer_norm"):
        return LayerNorm(dim, eps=eps)
    elif norm_type in ("np_layernorm", "np_layer_norm", "nplayernorm"):
        return NPLayerNorm(dim, eps=eps)
    elif norm_type in ("rms", "rmsnorm", "rms_norm", "rms_layernorm"):
        return RMSNorm(dim, eps=eps)
    elif norm_type in (
        "fused_rms",
        "fused_rmsnorm",
        "fused_rms_norm",
        "fused_rms_layernorm",
        "fusedrms",
        "fusedrmsnorm",
    ):
        return FusedRMSNorm(dim, eps=eps)
    else:
        raise NotImplementedError(f"Unknown norm_type: '{norm_type}'")


class NormBase(nn.Module):
    """
    Base class for normalization layers.
    """

    def __init__(
        self,
        size: int,
        eps: float = 1e-06,
        *,
        elementwise_affine: Optional[bool] = True,
    ):
        super().__init__()

        self.eps = eps
        self.normalized_shape = (size,)
        if elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(
                    self.normalized_shape,
                )
            )
        else:
            self.register_parameter("weight", None)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def init_weights(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)  # type: ignore

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)  # type: ignore


class LayerNorm(NormBase):
    """Classical LayerNorm, without bias."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-06,
        elementwise_affine: Optional[bool] = True,
    ):
        super().__init__(size=dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, weight=self.weight, eps=self.eps)


class NPLayerNorm(NormBase):
    """Non Parametric LayerNorm - no affine transform."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: Optional[bool] = False,
    ):
        super().__init__(size=dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, weight=self.weight, eps=self.eps)


class FusedRMSNorm(NormBase):
    """Fused RMS Norm"""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ):
        super().__init__(size=dim, elementwise_affine=True, eps=eps)
        self.fused_rms_norm_fn = fused_rms_norm_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """leverages Triton Fused RMS Norm kernel"""
        return self.fused_rms_norm_fn(
            x,
            self.weight,
            eps=self.eps,
        )


class RMSNorm(NormBase):
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
        super().__init__(size=dim, eps=eps)

    def _norm(self, x: torch.Tensor):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


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


class TTRMSNorm(torch.autograd.Function):
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
    return TTRMSNorm.apply(
        x,
        weight,
        eps,
    )
