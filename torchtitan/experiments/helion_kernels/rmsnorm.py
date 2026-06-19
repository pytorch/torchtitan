# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

import helion
import helion.language as hl


@helion.kernel()
def rms_norm_helion_fwd_2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    m, n = x.size()
    assert weight.size(0) == n, f"weight size mismatch {weight.size(0)} != {n}"

    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = x[tile_m, :].to(torch.float32)
        variance = torch.mean(acc * acc, dim=-1)
        inv_rms = torch.rsqrt(variance + eps)
        out[tile_m, :] = (acc * inv_rms[:, None] * weight[:].to(torch.float32)).to(
            x.dtype
        )
    return out


@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def rms_norm_helion_bwd_2d(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.size()
    assert grad_out.size(0) == m, f"grad_out rows mismatch {grad_out.size(0)} != {m}"
    assert grad_out.size(1) == n, f"grad_out cols mismatch {grad_out.size(1)} != {n}"
    assert weight.size(0) == n, f"weight size mismatch {weight.size(0)} != {n}"

    m_block = hl.register_block_size(m)
    grad_x = torch.empty_like(x)
    grad_weight_partials = torch.empty(
        [(m + m_block - 1) // m_block, n],
        dtype=torch.float32,
        device=x.device,
    )
    n_static = hl.specialize(n)

    for tile_m_block in hl.tile(m, block_size=m_block):
        grad_weight_acc = weight.new_zeros(n_static, dtype=torch.float32)
        for tile_m in hl.tile(tile_m_block.begin, tile_m_block.end):
            x_tile = x[tile_m, :].to(torch.float32)
            grad_out_tile = grad_out[tile_m, :].to(torch.float32)
            weight_tile = weight[None, :].to(torch.float32)

            variance = torch.sum(x_tile * x_tile, dim=-1) / n_static
            inv_rms = torch.rsqrt(variance + eps)
            grad_weighted = grad_out_tile * weight_tile
            dot_mean = torch.sum(grad_weighted * x_tile, dim=-1) / n_static

            grad_x[tile_m, :] = (
                grad_weighted * inv_rms[:, None]
                - x_tile * (inv_rms[:, None] ** 3) * dot_mean[:, None]
            ).to(x.dtype)
            grad_weight_acc += (grad_out_tile * x_tile * inv_rms[:, None]).sum(0)

        grad_weight_partials[tile_m_block.id, :] = grad_weight_acc

    return grad_x, grad_weight_partials.sum(0).to(weight.dtype)


def _nearest_power_of_2_bucket(value: int) -> int:
    assert value > 0
    lower = 1 << (value.bit_length() - 1)
    upper = lower << 1
    return lower if value - lower < upper - value else upper


def _config(
    block_sizes: list[int],
    *,
    num_warps: int,
    num_stages: int,
    pid_type: str = "flat",
    indexing: list[str] | None = None,
    load_eviction_policies: list[str] | None = None,
    maxnreg: int | None = None,
    num_sm_multiplier: int | None = None,
    reduction_loops: list[int | None] | None = None,
) -> helion.Config:
    kwargs: dict[str, Any] = {
        "block_sizes": block_sizes,
        "num_stages": num_stages,
        "num_warps": num_warps,
        "pid_type": pid_type,
    }
    if indexing is not None:
        kwargs["indexing"] = indexing
    if load_eviction_policies is not None:
        kwargs["load_eviction_policies"] = load_eviction_policies
    if maxnreg is not None:
        kwargs["maxnreg"] = maxnreg
    if num_sm_multiplier is not None:
        kwargs["num_sm_multiplier"] = num_sm_multiplier
    if reduction_loops is not None:
        kwargs["reduction_loops"] = reduction_loops
    return helion.Config(**kwargs)


_RMS_NORM_HELION_CONFIGS_BY_BUCKET: dict[tuple[int, int], helion.Config] = {
    # (normalized dim, nearest power-of-two row bucket)
    (5120, 1): _config(
        [1],
        num_warps=16,
        num_stages=8,
        pid_type="persistent_blocked",
        maxnreg=128,
        num_sm_multiplier=8,
        reduction_loops=[None],
    ),
    (5120, 256): _config(
        [1],
        num_warps=16,
        num_stages=8,
        reduction_loops=[None],
    ),
    (5120, 1024): _config(
        [1],
        num_warps=4,
        num_stages=4,
        pid_type="persistent_interleaved",
        maxnreg=128,
        num_sm_multiplier=16,
        reduction_loops=[1024],
    ),
    (128, 1): _config(
        [1],
        num_warps=1,
        num_stages=2,
        reduction_loops=[None],
    ),
    (128, 8): _config(
        [4],
        num_warps=4,
        num_stages=6,
        reduction_loops=[None],
    ),
    (128, 256): _config(
        [4],
        num_warps=4,
        num_stages=2,
        reduction_loops=[None],
    ),
    (128, 1024): _config(
        [16],
        num_warps=8,
        num_stages=1,
        reduction_loops=[None],
    ),
    (128, 2048): _config(
        [16],
        num_warps=8,
        num_stages=1,
        reduction_loops=[None],
    ),
    (128, 8192): _config(
        [32],
        num_warps=16,
        num_stages=7,
        reduction_loops=[None],
    ),
}


_RMS_NORM_HELION_BWD_CONFIGS_BY_BUCKET: dict[tuple[int, int], helion.Config] = {
    # (normalized dim, nearest power-of-two row bucket)
    (5120, 1): _config([1, 1], num_warps=8, num_stages=6),
    (5120, 256): _config(
        [2, 8],
        num_warps=8,
        num_stages=3,
        pid_type="persistent_interleaved",
        maxnreg=256,
        num_sm_multiplier=1,
    ),
    (5120, 1024): _config([8, 1], num_warps=8, num_stages=3),
    (128, 1): _config([1, 1], num_warps=1, num_stages=1),
    (128, 8): _config(
        [2, 2],
        num_warps=8,
        num_stages=6,
        pid_type="persistent_blocked",
        maxnreg=256,
        num_sm_multiplier=4,
    ),
    (128, 256): _config([2, 2], num_warps=1, num_stages=4),
    (128, 1024): _config(
        [16, 32],
        num_warps=8,
        num_stages=1,
        pid_type="persistent_interleaved",
        num_sm_multiplier=1,
    ),
    (128, 2048): _config([16, 16], num_warps=8, num_stages=6),
    (128, 8192): _config([128, 128], num_warps=8, num_stages=8),
}


def _rms_norm_helion_config(x_2d: torch.Tensor) -> helion.Config | None:
    rows, dim = x_2d.shape
    row_bucket = _nearest_power_of_2_bucket(rows)
    return _RMS_NORM_HELION_CONFIGS_BY_BUCKET.get((dim, row_bucket))


def _rms_norm_helion_bwd_config(x_2d: torch.Tensor) -> helion.Config | None:
    rows, dim = x_2d.shape
    row_bucket = _nearest_power_of_2_bucket(rows)
    return _RMS_NORM_HELION_BWD_CONFIGS_BY_BUCKET.get((dim, row_bucket))


def _rms_norm_helion_fwd_2d_with_config(
    x_2d: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    config = _rms_norm_helion_config(x_2d)
    if config is None:
        return rms_norm_helion_fwd_2d(x_2d, weight, eps)

    bound_kernel = rms_norm_helion_fwd_2d.bind((x_2d, weight, eps))
    if getattr(bound_kernel, "_config", None) != config:
        bound_kernel.set_config(config)
    return bound_kernel(x_2d, weight, eps)


def _rms_norm_helion_bwd_2d_with_config(
    grad_out_2d: torch.Tensor,
    x_2d: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    config = _rms_norm_helion_bwd_config(x_2d)
    if config is None:
        return rms_norm_helion_bwd_2d(grad_out_2d, x_2d, weight, eps)

    bound_kernel = rms_norm_helion_bwd_2d.bind((grad_out_2d, x_2d, weight, eps))
    if getattr(bound_kernel, "_config", None) != config:
        bound_kernel.set_config(config)
    return bound_kernel(grad_out_2d, x_2d, weight, eps)


def _can_use_helion_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
) -> bool:
    return (
        weight is not None
        and x.ndim >= 1
        and weight.ndim == 1
        and x.shape[-1] == weight.shape[0]
        and x.is_cuda
        and weight.is_cuda
        and x.is_contiguous()
        and weight.is_contiguous()
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and weight.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


def _is_qwen3_b200_fast_shape(x: torch.Tensor, weight: torch.Tensor) -> bool:
    normalized_shape = weight.shape[0]
    num_rows = x.numel() // normalized_shape

    return (normalized_shape == 5120 and num_rows <= 256) or (
        normalized_shape == 128 and num_rows >= 4096
    )


def rms_norm_helion_raw(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float = 1e-5,
) -> torch.Tensor:
    normalized_shape = (x.shape[-1],)
    if not _can_use_helion_rmsnorm(x, weight):
        return F.rms_norm(x, normalized_shape, weight, eps)

    assert weight is not None
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        return _RMSNormHelionFunction.apply(x, weight, eps)

    x_2d = x.reshape(-1, x.shape[-1])
    out = _rms_norm_helion_fwd_2d_with_config(x_2d, weight, eps)
    return out.reshape_as(x)


class _RMSNormHelionFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1])
        out = _rms_norm_helion_fwd_2d_with_config(x_2d, weight, eps).reshape_as(x)
        ctx.x_shape = x.shape
        ctx.eps = eps
        ctx.save_for_backward(x, weight)
        return out

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx: Any,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        x, weight = ctx.saved_tensors
        grad_x, grad_weight = _rms_norm_helion_bwd_2d_with_config(
            grad_out.contiguous().reshape(-1, x.shape[-1]),
            x.reshape(-1, x.shape[-1]),
            weight,
            ctx.eps,
        )
        if not ctx.needs_input_grad[0]:
            grad_x = None
        else:
            grad_x = grad_x.reshape(ctx.x_shape)
        if not ctx.needs_input_grad[1]:
            grad_weight = None
        return grad_x, grad_weight, None


def rms_norm_helion(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Apply RMSNorm with experimental Helion kernels for selected shapes.

    The public wrapper is deliberately selective: on B200, standalone Helion
    is faster for the short Qwen3 hidden-size norms and the largest Q/K norm
    rows from the current sweep, but ATen is still better for other forward
    RMSNorm shapes. Unsupported layouts fall back to PyTorch.
    """
    normalized_shape = (x.shape[-1],)
    if not _can_use_helion_rmsnorm(x, weight):
        return F.rms_norm(x, normalized_shape, weight, eps)

    assert weight is not None
    if not _is_qwen3_b200_fast_shape(x, weight):
        return F.rms_norm(x, normalized_shape, weight, eps)

    return rms_norm_helion_raw(x, weight, eps)
