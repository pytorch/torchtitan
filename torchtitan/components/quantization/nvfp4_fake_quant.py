# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NVFP4 fake-quantization primitives.

Note: these primitives are self-contained, can be moved to torchao if needed.

A minimal, self-contained implementation of NVFP4 fake quantization applied to a
model's MoE experts and dense linear layers during training (e.g. for
quantization-aware distillation or QAT), using a straight-through estimator.

NVFP4 numerics (two-level block-scale quantization):

* **Global scale (per tensor):** ``(448 * 6) / amax(x)`` -- 448 is the max
  FP8-E4M3 value, 6 is the max FP4-E2M1 value.
* **Block scale (per 16 elements):** each block gets its own FP8-E4M3 scale
  ``block_amax / 6``, so precision adapts to local magnitude.

Every value is rounded to the nearest E2M1 grid point
(0, 0.5, 1, 1.5, 2, 3, 4, 6) then dequantized. Training uses a straight-through
estimator (STE): forward returns the dequantized value, backward is identity.

The single public entry point is :func:`apply_nvfp4_fake_quant`, which
fake-quantizes BOTH the MoE experts and the dense linear layers (weight + input
activation) -- the full set of layers an NVFP4 eval quantizes (everything
except ``lm_head`` / ``router`` / ``gate``).
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.models.common.moe import GroupedExperts

logger = logging.getLogger(__name__)


# Representable positive E2M1 values and the round-to-nearest midpoint
# boundaries between consecutive values.
_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_E2M1_BOUNDARIES = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
_SF_BLOCK_SIZE = 16

# Module-name substrings the NVFP4 eval does NOT quantize, so QAD skips them too
# to match the eval scope exactly: the LM head and the MoE router gate.
_LINEAR_FQ_EXCLUDE = ("lm_head", "router", "gate")


def _nvfp4_quant_dequant(x: torch.Tensor) -> torch.Tensor:
    """NVFP4 E2M1 block-scale quantize-dequantize roundtrip.

    For 3D tensors (expert weights) each expert slice is quantized independently
    to avoid OOM from materializing the full tensor in float32. Returns the
    dequantized tensor in the same dtype as *x*.
    """
    if x.ndim == 3:
        out = torch.empty_like(x)
        for i in range(x.shape[0]):
            out[i] = _nvfp4_quant_dequant(x[i])
        return out

    device = x.device
    orig_dtype = x.dtype

    # Level 1: global scale factor maps tensor range into FP8*FP4 range
    gsf = (448 * 6) / x.float().abs().nan_to_num().max()
    x = x.float() * gsf

    # Level 2: per-block FP8 scale factor adapts to local magnitudes
    orig_shape = x.shape
    x_flat = x.reshape(-1, _SF_BLOCK_SIZE)
    block_amax = x_flat.abs().amax(dim=-1, keepdim=True)
    block_sf = (block_amax / 6.0).to(torch.float8_e4m3fn).float()

    # Round to nearest E2M1 value
    x_norm = x_flat / block_sf.clamp(min=torch.finfo(torch.float8_e4m3fn).tiny)
    sign = x_norm.sign()
    x_abs = x_norm.abs()
    boundaries = torch.tensor(_E2M1_BOUNDARIES, device=device)
    values = torch.tensor(_E2M1_VALUES, device=device)
    indices = torch.bucketize(x_abs, boundaries)
    x_quant = sign * values[indices]

    # Dequantize: reverse block scale, reshape, reverse global scale
    x_dq = (x_quant * block_sf).reshape(orig_shape) / gsf
    return x_dq.to(orig_dtype)


def _nvfp4_fake_quantize_ste(x: torch.Tensor) -> torch.Tensor:
    """NVFP4 block-scale fake quantize with STE (straight-through estimator).

    Forward returns the dequantized value; backward treats quantization as
    identity. Used for the MoE ``fake_quant_fn`` hook and for linear weights.
    """
    dq = _nvfp4_quant_dequant(x.detach())
    return x + (dq - x).detach()


def _nvfp4_fake_quantize_act_ste(x: torch.Tensor) -> torch.Tensor:
    """STE NVFP4 fake-quant for a linear layer's INPUT activation of any rank.

    ``_nvfp4_quant_dequant`` treats a 3D tensor as independent per-expert slices
    (looping dim 0), which is wrong for an activation shaped
    ``(batch, seq, features)``. Flatten to 2D ``(tokens, features)`` first so the
    per-tensor global scale spans all tokens and the FP8 block scales tile along
    the feature dim -- matching how the eval quantizes a linear's input -- then
    restore the original shape. STE preserved through the reshape.
    """
    if x.ndim <= 2:
        return _nvfp4_fake_quantize_ste(x)
    orig_shape = x.shape
    flat = x.reshape(-1, orig_shape[-1])
    return _nvfp4_fake_quantize_ste(flat).reshape(orig_shape)


def _wrap_linear_nvfp4_fake_quant(module: nn.Linear) -> None:
    """Override a linear's forward so its WEIGHT and INPUT ACTIVATION are both
    NVFP4 fake-quantized (STE) before the matmul, matching the NVFP4 eval.
    Instance-level forward override (eager only)."""

    def fq_forward(input: torch.Tensor) -> torch.Tensor:
        w = _nvfp4_fake_quantize_ste(module.weight)
        x = _nvfp4_fake_quantize_act_ste(input)
        return F.linear(x, w, module.bias)

    module.forward = fq_forward


def apply_nvfp4_fake_quant(model: nn.Module) -> nn.Module:
    """Enable NVFP4 fake quantization on BOTH the MoE experts AND the dense
    linear layers -- the full set of layers quantized by the NVFP4 eval.

    * **MoE experts:** sets ``fake_quant_fn`` on every ``GroupedExperts`` module
      (identified by having both ``num_experts`` and ``fake_quant_fn``), which
      fake-quantizes the expert weights and activations inside
      ``_experts_forward`` (FSDP2/DTensor-compatible).
    * **Dense linears:** overrides the forward of every ``nn.Linear`` whose name
      does not match :data:`_LINEAR_FQ_EXCLUDE` (``lm_head``/``router``/``gate``)
      so its weight and input activation are fake-quantized.

    Applied in-place; the same model is returned. Eager-only (linear forwards are
    overridden at the instance level).
    """
    # Apply fake quant to the MoE experts (via the GroupedExperts hook).
    n_moe = 0
    for module in model.modules():
        if isinstance(module, GroupedExperts):
            module.fake_quant_fn = _nvfp4_fake_quantize_ste
            n_moe += 1

    # Apply fake quant to the dense linear layers (excl lm_head/router/gate).
    n_lin = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not any(
            excl in name for excl in _LINEAR_FQ_EXCLUDE
        ):
            _wrap_linear_nvfp4_fake_quant(module)
            n_lin += 1

    logger.info(
        "Applied NVFP4 fake quant to %d GroupedExperts modules and %d linear "
        "layers (excluded lm_head/router/gate)",
        n_moe,
        n_lin,
    )
    return model
