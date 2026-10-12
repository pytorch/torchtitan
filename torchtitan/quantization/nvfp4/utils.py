# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NVFP4 recipe utilities."""

import math
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from torchtitan.models.common import TransformerBlock


# TorchAO's NVFP4 Triton kernels require each local GEMM dimension to be a
# multiple of 128. This is a kernel tiling requirement, not the NVFP4 format's
# 16-element microscaling block (F.ScalingType.BlockWise1x16).
_NVFP4_GEMM_ALIGNMENT = 128


def nvfp4_bf16_tail_fqns(num_layers: int, bf16_tail_fraction: float) -> list[str]:
    """Select leading decoder layers while leaving a BF16 tail.

    Each fqn has a trailing '.' so 'layers.1.' matches layer 1 only, not
    'layers.10'. An empty list would convert every Linear, so reject fractions
    that leave no layer to convert.
    """
    num_bf16 = math.ceil(num_layers * bf16_tail_fraction)
    convert_upto = num_layers - num_bf16
    if convert_upto <= 0:
        raise ValueError(
            f"bf16_tail_fraction={bf16_tail_fraction} keeps all {num_layers} "
            "layers in bf16; nothing to convert to NVFP4."
        )
    return [f"layers.{i}." for i in range(convert_upto)]


def nvfp4_ffn_submodules(
    layers: list["TransformerBlock.Config"],
) -> tuple[str, ...]:
    """Select the FFN submodules NVFP4 can convert for this model.

    NVFP4 requires in_features and out_features to be multiples of
    _NVFP4_GEMM_ALIGNMENT, rejected in NVFP4Linear.Config.__post_init__.
    DeepSeek-V3 16B's dense FFN is 10944 wide, so its feed_forward cannot be
    converted while its shared_experts can; every other flavor converts both.
    Deriving this from the shape keeps the exclusion at the model that needs
    it, rather than as a literal each config has to pick correctly.
    """
    dense = next(
        (layer.feed_forward for layer in layers if layer.feed_forward is not None),
        None,
    )
    if dense is not None and dense.w13.out_features % _NVFP4_GEMM_ALIGNMENT:
        return ("moe.shared_experts.",)
    return ("feed_forward.", "moe.shared_experts.")
