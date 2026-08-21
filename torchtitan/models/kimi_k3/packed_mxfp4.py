# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Load K3's packed-MXFP4 routed-expert weights.

The released checkpoint stores every routed expert as two tensors instead of one:

    ...block_sparse_moe.experts.{e}.w{1,2,3}.weight_packed   uint8
    ...block_sparse_moe.experts.{e}.w{1,2,3}.weight_scale     uint8

from ``quantization_config``: ``mxfp4-pack-quantized``, 4 bits, group_size 32,
group strategy, symmetric, ``scale_dtype torch.uint8``. That is OCP MX with an
E8M0 block scale stored as a raw byte, two E2M1 values per packed byte.

Nothing else in the checkpoint is quantized (see ``quant_scope.py``), so this is
the only place a load has to do anything unusual.

Two shapes matter and they are easy to conflate:

* per-expert HF layout is ``[out, in]`` with ``weight_packed`` at
  ``[out, in // 2]`` and ``weight_scale`` at ``[out, in // 32]``;
* our ``GroupedExperts`` stacks experts, so ``w1_EFD`` is ``[E, F, D]`` and
  ``w2_EDF`` is ``[E, D, F]``. The stacking axis is the expert index, and the
  group axis is always the LAST dim, which is what makes the per-expert
  dequantized block droppable straight into ``[e]``.

E8M0 decode: the byte is a biased power of two, ``scale = 2 ** (byte - 127)``,
with ``byte == 0`` meaning zero rather than ``2 ** -127``. E2M1 decode is a
16-entry table, so it is done by lookup rather than bit arithmetic.
"""

from __future__ import annotations

import torch

MXFP4_GROUP_SIZE = 32
_E8M0_BIAS = 127

# E2M1: 1 sign, 2 exponent, 1 mantissa. The 16 representable magnitudes, in
# nibble order 0..15 (sign bit is the high bit of the nibble).
_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _e2m1_table(device, dtype) -> torch.Tensor:
    return torch.tensor(_E2M1_VALUES, device=device, dtype=dtype)


def dequantize_mxfp4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    group_size: int = MXFP4_GROUP_SIZE,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """``[..., in // 2]`` uint8 + ``[..., in // 32]`` uint8 -> ``[..., in]``.

    Args:
        packed: two E2M1 nibbles per byte, low nibble first.
        scale: one E8M0 byte per group of ``group_size`` values.

    The low-nibble-first convention is the one compressed-tensors writes, and it
    is asserted by round-tripping our own packer in the tests -- getting it
    backwards swaps adjacent weights, which no shape check would catch and which
    a loss curve would absorb.
    """
    if packed.dtype != torch.uint8 or scale.dtype != torch.uint8:
        raise ValueError(
            f"expected uint8 packed data and scales, got {packed.dtype} and "
            f"{scale.dtype}"
        )
    in_half = packed.shape[-1]
    in_features = in_half * 2
    if in_features % group_size:
        raise ValueError(
            f"in_features {in_features} is not a multiple of group_size "
            f"{group_size}"
        )
    expected_groups = in_features // group_size
    if scale.shape[-1] != expected_groups:
        raise ValueError(
            f"scale has {scale.shape[-1]} groups but the packed data implies "
            f"{expected_groups}"
        )

    # torchao's MX dequantizer, not a local nibble table (finding 56). Checked
    # bit-for-bit before delegating, on the cases that actually distinguish the two:
    # three shapes at bf16 and float32 are identical, and so are all three E8M0
    # special values -- 0x00 as 2**-127, 0x7F as 2**0, and 0xFF as NaN. That last one
    # matters most: it is a fix this function already carries (mapping 0x00 to zero or
    # letting 0xFF reach exp2(128) = inf is wrong in both directions, and
    # quantize_mxfp4 emits neither, so the round-trip test cannot see it). Delegating
    # to something that got it wrong would have reintroduced it.
    #
    # float16 targets are NOT equivalent: E8M0 scales reach 2**23, which overflows
    # fp16, and torchao computes in float32 before casting. Nothing here asks for
    # fp16; the overflow is real rather than an artifact of either implementation.
    from torchao.prototype.mx_formats.mx_tensor import to_dtype

    return to_dtype(packed, scale, torch.float4_e2m1fn_x2, group_size, dtype)


def quantize_mxfp4(
    weight: torch.Tensor, *, group_size: int = MXFP4_GROUP_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`dequantize_mxfp4`, for building test fixtures.

    Not the training path -- ``lora.quantize_grouped_experts_mxfp4`` uses
    torchao's MX primitives for that. This exists so a synthetic checkpoint can
    be written in the RELEASED byte layout and read back, which is how the load
    path gets exercised without the 1.56 TB download.
    """
    *lead, in_features = weight.shape
    if in_features % group_size:
        raise ValueError(f"{in_features} is not a multiple of {group_size}")
    groups = weight.float().reshape(*lead, in_features // group_size, group_size)

    amax = groups.abs().amax(dim=-1, keepdim=True)
    # OCP MX: shared exponent = floor(log2(amax)) - emax_elem, where E2M1's
    # largest magnitude 6 = 1.5 * 2**2 gives emax_elem = 2. Using
    # floor(log2(amax / 6)) instead loses up to a full binade of range.
    exp = torch.where(
        amax == 0,
        torch.zeros_like(amax),
        torch.floor(torch.log2(amax)) - 2 + _E8M0_BIAS,
    ).clamp(0, 255)
    factor = torch.where(exp == 0, torch.ones_like(exp), torch.exp2(exp - _E8M0_BIAS))
    normalized = groups / factor

    table = _e2m1_table(weight.device, torch.float32)
    # Nearest representable E2M1 value, by exhaustive comparison over 16 entries.
    idx = (normalized.unsqueeze(-1) - table).abs().argmin(dim=-1)
    nibbles = idx.reshape(*lead, in_features).to(torch.uint8)
    lo, hi = nibbles[..., 0::2], nibbles[..., 1::2]
    packed = (lo | (hi << 4)).contiguous()
    scale = exp.squeeze(-1).to(torch.uint8).contiguous()
    return packed, scale


def load_packed_experts(
    experts: torch.nn.Module,
    tensors: dict[str, torch.Tensor],
    *,
    num_experts: int,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Fill a ``GroupedExperts`` from per-expert packed tensors.

    ``tensors`` is keyed by our own naming with an expert index, i.e. what
    ``hf_key_map.official_to_titan`` returns: ``"...w1_EFD[3]"`` mapped to the
    packed byte tensor, plus the same key for the scale. Both kinds are passed
    together under separate dicts to keep the caller honest about which is which.

    Returns the number of expert slices written. Raises if any slice is missing --
    a partially loaded expert tensor is worse than a failed load, because the
    remaining slices keep their init values and the model still trains.
    """
    written = 0
    for name in ("w1_EFD", "w2_EDF", "w3_EFD"):
        param = experts._parameters.get(name)
        if param is None:
            continue
        for e in range(num_experts):
            key = f"{name}[{e}]"
            if key not in tensors or f"{key}:scale" not in tensors:
                raise KeyError(
                    f"missing packed data for expert slice {key}; refusing a "
                    "partial load, which would leave that expert at init values"
                )
            block = dequantize_mxfp4(tensors[key], tensors[f"{key}:scale"], dtype=dtype)
            if block.shape != param.shape[1:]:
                raise ValueError(
                    f"{key} dequantized to {tuple(block.shape)} but the slice "
                    f"expects {tuple(param.shape[1:])}"
                )
            with torch.no_grad():
                param[e].copy_(block)
            written += 1
    return written
