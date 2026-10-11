# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

from torchtitan.models.common.attention import InnerAttention


class SDPAReferenceAttention(InnerAttention):
    """Batched causal SDPA reference for standalone checkpoint comparisons."""

    def forward(
        self,
        q_BLHK,
        k_BLHK,
        v_BLHV,
        *,
        attention_metadata=None,
        scale=None,
        enable_gqa=False,
        **kwargs,
    ):
        if attention_metadata is not None:
            raise ValueError(
                "The checkpoint SDPA reference requires a single document."
            )
        with sdpa_kernel(
            [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH],
            set_priority=True,
        ):
            out_BHLV = torch.nn.functional.scaled_dot_product_attention(
                q_BLHK.transpose(1, 2),
                k_BLHK.transpose(1, 2),
                v_BLHV.transpose(1, 2),
                scale=scale,
                is_causal=True,
                enable_gqa=enable_gqa,
            )
        return out_BHLV.transpose(1, 2)
