# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import unittest
from dataclasses import dataclass
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask

from torchtitan.config import apply_overrides, derive, override, OverrideConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rope import ComplexRoPE, IdentityRoPE, RoPE
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
            rope_apply=IdentityRoPE.Config().build(),
            attention_mask=attention_mask,
        )

        self.assertEqual(identity_attention.input_shape, (num_tokens, num_heads, 4))
        self.assertEqual(out_TD.shape, (num_tokens, dim))


def test_encoder_rope_and_attention_overrides_are_independent() -> None:
    class MarkerRoPE(Module):
        @dataclass(kw_only=True, slots=True)
        class Config(RoPE.Config):
            pass

        def __init__(self, config):
            super().__init__()
            self.calls = 0
            self.cache_is_complex = None

        def forward(
            self, query, key=None, positions=None, *, rope_cache=None, inverse=False
        ):
            self.calls += 1
            assert rope_cache is not None
            self.cache_is_complex = rope_cache.is_complex()
            return torch.full_like(query, 42), key

    class MarkerAttention(FlexInnerAttention):
        @dataclass(kw_only=True, slots=True)
        class Config(FlexInnerAttention.Config):
            pass

        def __init__(self, config):
            super().__init__(config)
            self.calls = 0

        def forward(self, query, key, value, *, attention_metadata):
            self.calls += 1
            torch.testing.assert_close(
                query, torch.full_like(query, 42), rtol=0, atol=0
            )
            return query

    with patch.dict("torchtitan.config.override._REGISTRY"):

        @override(target=RoPE.Config)
        def marker_rope(config):
            return derive(config, MarkerRoPE.Config)

        @override(target=FlexInnerAttention.Config, exact=True)
        def marker_attention(config):
            return derive(config, MarkerAttention.Config)

        for family in ("kimi_k2_7", "kimi_k3", "qwen3_5", "muse_spark", "muse_glimmer"):
            config = (
                importlib.import_module(f"torchtitan.models.{family}")
                .build_model_config(
                    "debugmodel_mm" if family.startswith("muse_") else "debugmodel",
                    seq_len=128,
                    attn_backend="flex",
                    parallelism=ParallelismConfig(),
                )
                .vision_encoder
            )
            complex_cache = isinstance(config.rope_apply, ComplexRoPE.Config)
            replacements = apply_overrides(
                OverrideConfig(
                    imports=[
                        f"{__name__}.marker_rope",
                        f"{__name__}.marker_attention",
                    ]
                ),
                config,
            )
            assert len(replacements) == 2
            assert isinstance(config.rope_apply, MarkerRoPE.Config)
            assert isinstance(config.block.attn.inner_attention, MarkerAttention.Config)
            encoder = config.build()
            encoder.init_states()
            with patch(
                f"{encoder.forward.__module__}.create_block_diagonal_mask",
                return_value=object(),
            ):
                projection = (
                    encoder.conv1_linear
                    if family.startswith("muse_")
                    else encoder.patch_embed
                )
                encoder(
                    torch.randn(16, projection.in_features),
                    grid_thw=torch.tensor([[1, 4, 4]]),
                )
            assert encoder.rope_apply.calls == config.num_layers
            assert encoder.rope_apply.cache_is_complex == complex_cache
            assert all(
                layer.attn.flex_attention.calls == 1
                for layer in encoder.layers.values()
            )


if __name__ == "__main__":
    unittest.main()
