# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Differentiable dequantized representations of V4.1's activation formats.

These operations model the published inference formats during training. They
retain floating point tensors; they do not allocate a packed serving KV cache.
"""

import torch


def _power_of_two_ceiling(x: torch.Tensor) -> torch.Tensor:
    mantissa, exponent = torch.frexp(x)
    return torch.ldexp(torch.ones_like(x), exponent - (mantissa == 0.5).int())


def fp8_activation(x: torch.Tensor, *, block_size: int = 32) -> torch.Tensor:
    """E4M3 values with power-of-two block scales and a straight-through gradient."""
    blocks = x.float().unflatten(-1, (-1, block_size))
    scale = _power_of_two_ceiling(
        blocks.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448
    )
    values = (blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    dequantized = (values * scale).flatten(-2).to(x.dtype)
    return x + (dequantized - x).detach()


def fp4_values(x: torch.Tensor) -> torch.Tensor:
    """Round to E2M1, including round-to-nearest-even at each midpoint."""
    levels = x.new_tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    midpoints = x.new_tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    magnitude = x.abs().clamp_max(6)
    codes = torch.bucketize(magnitude.contiguous(), midpoints)
    midpoint = midpoints[codes.clamp_max(6)]
    codes = codes + ((magnitude == midpoint) & (codes.remainder(2) == 1)).long()
    return torch.copysign(levels[codes], x)


def fp4_activation(x: torch.Tensor, *, compressed_kv: bool = False) -> torch.Tensor:
    """NVFP4 (16 values, E4M3 scales) or MXFP4 (32 values, E8M0 scales)."""
    block_size = 16 if compressed_kv else 32
    blocks = x.float().unflatten(-1, (-1, block_size))
    minimum = 6 * (2**-9 if compressed_kv else 2**-126)
    scale = blocks.abs().amax(-1, keepdim=True).clamp_min(minimum) / 6
    scale = (
        scale.to(torch.float8_e4m3fn).float()
        if compressed_kv
        else _power_of_two_ceiling(scale)
    )
    dequantized = (fp4_values(blocks / scale) * scale).flatten(-2).to(x.dtype)
    return x + (dequantized - x).detach()
