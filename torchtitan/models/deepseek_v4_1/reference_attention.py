# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""An explicit PyTorch math backend for Flex masks, including CPU backward.

T: query tokens, S: key tokens, H: heads, K: query/key width, V: value width.
"""

from dataclasses import dataclass

import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.attention.flex_attention import create_mask

from torchtitan.models.common.attention import FlexInnerAttention, InnerAttention


class ReferenceFlexInnerAttention(InnerAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(FlexInnerAttention.Config):
        def __post_init__(self):
            if self.kernel_options:
                raise ValueError(
                    "The reference Flex backend does not accept fused kernel options."
                )

    def __init__(self, config: Config):
        super().__init__()

    def forward(
        self,
        q_THK,
        k_THK,
        v_THV,
        *,
        attention_metadata,
        score_mod=None,
        scale=None,
        out_transform=None,
        **kwargs
    ):
        if score_mod is not None or out_transform is not None:
            raise ValueError(
                "Reference vision attention supports masks and scaling only."
            )
        mask_1HTS = create_mask(
            attention_metadata.mask_mod,
            1,
            q_THK.size(1),
            q_THK.size(0),
            k_THK.size(0),
            device=q_THK.device,
        )
        with sdpa_kernel(SDPBackend.MATH):
            out_1HTV = F.scaled_dot_product_attention(
                q_THK.transpose(0, 1)[None],
                k_THK.transpose(0, 1)[None],
                v_THV.transpose(0, 1)[None],
                attn_mask=mask_1HTS,
                scale=scale,
            )
        return out_1HTV.squeeze(0).transpose(0, 1)
