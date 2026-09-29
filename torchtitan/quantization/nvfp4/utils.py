# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NVFP4 recipe utilities."""

import math


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
