# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask

from torchtitan.config import apply_overrides, derive, override, OverrideConfig
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.models.common.vision_encoder import VisionAttention


class _IdentityAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_shape: torch.Size | None = None

    def forward(self, q_THK, k_THK, v_THV, *, attention_metadata):
        self.input_shape = q_THK.shape
        return q_THK


class TestVisionAttention(unittest.TestCase):
    def test_td_layout(self) -> None:
        dim, num_heads, num_tokens = 16, 4, 7
        config = VisionAttention.Config(
            dim=dim,
            num_heads=num_heads,
            wq=Linear.Config(in_features=dim, out_features=dim),
            wk=Linear.Config(in_features=dim, out_features=dim),
            wv=Linear.Config(in_features=dim, out_features=dim),
            proj=Linear.Config(in_features=dim, out_features=dim),
            rope=ComplexRoPE.Config(),
        )

        class OffsetRoPE(ComplexRoPE):
            @dataclass(kw_only=True, slots=True)
            class Config(ComplexRoPE.Config):
                pass

            def forward(
                self,
                query_THDh,
                key_THDh=None,
                positions=None,
                *,
                rope_cache=None,
                inverse=False,
            ):
                query_THDh = query_THDh + 1
                return query_THDh if key_THDh is None else (query_THDh, key_THDh)

        with patch.dict("torchtitan.config.override._REGISTRY"):

            @override(target=ComplexRoPE.Config, fqns=["rope"], exact=True)
            def replace_vision_rope(cfg):
                return derive(cfg, OffsetRoPE.Config)

            replacements = apply_overrides(
                OverrideConfig(imports=[f"{__name__}.replace_vision_rope"]), config
            )
            self.assertEqual(len(replacements), 1)

        attention = config.build()
        self.assertIsNone(attention.rope.cache)
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
            rope_cache=torch.ones(num_tokens, 1, 2, dtype=torch.complex64),
            attention_mask=attention_mask,
        )

        self.assertEqual(identity_attention.input_shape, (num_tokens, num_heads, 4))
        self.assertEqual(out_TD.shape, (num_tokens, dim))
        torch.testing.assert_close(
            out_TD, attention.proj(attention.wq(x_TD) + 1), rtol=0, atol=0
        )
        torch.testing.assert_close(
            attention(
                x_TD,
                rope_cache=torch.empty(0),
                attention_mask=attention_mask,
                use_rope=False,
            ),
            attention.proj(attention.wq(x_TD)),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
