# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.vision_encoder import VisionAttention


class _IdentityAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_shape: torch.Size | None = None

    def forward(self, q_THK, k_THK, v_THV, *, attention_masks):
        self.input_shape = q_THK.shape
        return q_THK


class TestVisionAttention(unittest.TestCase):
    def test_td_layout(self) -> None:
        dim, num_heads, num_tokens = 16, 4, 7
        attention = VisionAttention(
            VisionAttention.Config(
                dim=dim,
                num_heads=num_heads,
                wq=Linear.Config(in_features=dim, out_features=dim),
                wk=Linear.Config(in_features=dim, out_features=dim),
                wv=Linear.Config(in_features=dim, out_features=dim),
                proj=Linear.Config(in_features=dim, out_features=dim),
            )
        )
        identity_attention = _IdentityAttention()
        attention.flex_attention = identity_attention

        x_TD = torch.randn(num_tokens, dim)
        attention_mask = create_block_mask(
            lambda b, h, q_idx, kv_idx: torch.tensor(True),
            B=1,
            H=None,
            Q_LEN=num_tokens,
            KV_LEN=num_tokens,
            device="cpu",
        )

        out_TD = attention(
            x_TD,
            rope_cache=torch.empty(0),
            rope_apply=lambda q, k, cache: (q, k),
            attention_mask=attention_mask,
        )

        self.assertEqual(identity_attention.input_shape, (num_tokens, num_heads, 4))
        self.assertEqual(out_TD.shape, (num_tokens, dim))


if __name__ == "__main__":
    unittest.main()
