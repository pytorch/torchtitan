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
from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rope import ComplexRoPE, CosSinRoPE, IdentityRoPE, RoPE
from torchtitan.models.common.vision_encoder import VisionAttention
from torchtitan.models.kimi_k2_7.flavors import _vision_encoder_config as kimi27_vision
from torchtitan.models.kimi_k3.flavors import _vision_encoder_config as kimi3_vision
from torchtitan.models.muse_spark.flavors import muse_spark_navit_encoder_config
from torchtitan.models.qwen3_5.flavors import (
    _qwen35_vision_encoder_config as qwen35_vision,
)


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

    cases = [
        (
            kimi27_vision(
                dim=16,
                ffn_dim=32,
                num_layers=2,
                num_heads=2,
                patch_size=2,
                init_pos_emb_height=4,
                init_pos_emb_width=4,
                text_hidden_size=16,
            ),
            12,
            16,
        ),
        (
            kimi3_vision(
                text_dim=16,
                dim=16,
                qkv_dim=16,
                hidden_dim=32,
                num_layers=2,
                num_heads=2,
                patch_size=2,
                init_pos_emb_height=4,
                init_pos_emb_width=4,
            ),
            12,
            16,
        ),
        (
            qwen35_vision(
                dim=16,
                ffn_dim=32,
                num_layers=2,
                num_heads=2,
                patch_size=2,
                temporal_patch_size=2,
                spatial_merge_size=2,
                out_hidden_size=16,
                num_position_embeddings=16,
            ),
            24,
            16,
        ),
    ]
    for grid_order in ("legacy_spark", "raster"):
        config = muse_spark_navit_encoder_config(
            latent_dim=16,
            num_heads=2,
            num_layers=2,
            mlp_hidden_dim=32,
            patch_size=2,
            patch_temporal=2,
            downsample_factor=2,
            sparse_attention_factor=2,
            pos_emb_grid_h=2,
            pos_emb_grid_w=2,
        )
        config.pos_emb_grid_order = grid_order
        cases.append((config, 24, 64))

    with patch.dict("torchtitan.config.override._REGISTRY"):

        @override(target=ComplexRoPE.Config, exact=True)
        def marker_complex_rope(config):
            return derive(config, MarkerRoPE.Config)

        @override(target=CosSinRoPE.Config, exact=True)
        def marker_cos_sin_rope(config):
            return derive(config, MarkerRoPE.Config)

        @override(target=FlexInnerAttention.Config, exact=True)
        def marker_attention(config):
            return derive(config, MarkerAttention.Config)

        for config, patch_dim, output_dim in cases:
            complex_cache = isinstance(config.rope_apply, ComplexRoPE.Config)
            rope_override = (
                "marker_complex_rope" if complex_cache else "marker_cos_sin_rope"
            )
            replacements = apply_overrides(
                OverrideConfig(
                    imports=[
                        f"{__name__}.{rope_override}",
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
                output = encoder(
                    torch.randn(16, patch_dim), grid_thw=torch.tensor([[1, 4, 4]])
                )
            assert output.shape == (4, output_dim)
            assert encoder.rope_apply.calls == config.num_layers
            assert encoder.rope_apply.cache_is_complex == complex_cache
            assert all(
                layer.attn.flex_attention.calls == 1
                for layer in encoder.layers.values()
            )


if __name__ == "__main__":
    unittest.main()
