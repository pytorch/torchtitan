# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
import torch.nn.functional as F

import helion.experimental
import helion.language as hl


@helion.experimental.aot_kernel()
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
        and not (torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad))
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
    x_2d = x.reshape(-1, x.shape[-1])
    out = rms_norm_helion_fwd_2d(x_2d, weight, eps)
    return out.reshape_as(x)


def rms_norm_helion(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Apply RMSNorm with an experimental Helion forward kernel.

    The public wrapper is deliberately selective: on B200, standalone Helion
    is faster for the short Qwen3 hidden-size norms and the largest Q/K norm
    rows from the current sweep, but ATen is still better for other RMSNorm
    shapes. Unsupported layouts and training cases also fall back to PyTorch.
    """
    normalized_shape = (x.shape[-1],)
    if not _can_use_helion_rmsnorm(x, weight):
        return F.rms_norm(x, normalized_shape, weight, eps)

    assert weight is not None
    if not _is_qwen3_b200_fast_shape(x, weight):
        return F.rms_norm(x, normalized_shape, weight, eps)

    return rms_norm_helion_raw(x, weight, eps)
