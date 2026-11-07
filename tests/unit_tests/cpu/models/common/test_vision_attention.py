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
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rope import IdentityRoPE, RoPE
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
                rope=IdentityRoPE.Config(),
            )
        )
        identity_attention = _IdentityAttention()
        attention.flex_attention = identity_attention
        self.assertIsInstance(attention.rope, RoPE)
        self.assertIsInstance(attention.rope.config, RoPE.Config)
        self.assertEqual(list(attention.rope.named_buffers()), [])

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
            positions=torch.zeros(num_tokens, 2, dtype=torch.long),
            attention_mask=attention_mask,
        )

        self.assertEqual(identity_attention.input_shape, (num_tokens, num_heads, 4))
        self.assertEqual(out_TD.shape, (num_tokens, dim))


def test_encoder_rope_and_attention_overrides_are_independent() -> None:
    class MarkerRoPE(RoPE):
        @dataclass(kw_only=True, slots=True)
        class Config(RoPE.Config):
            pass

        def __init__(self, config):
            super().__init__(config)
            self.calls = 0
            self.prepare_calls = 0
            self.prepared_positions = None
            self.received_positions = None
            self.positions_shape = None

        def prepare_positions(self, positions):
            self.prepare_calls += 1
            self.prepared_positions = positions + 1
            return self.prepared_positions

        def forward(self, query, key=None, positions=None, *, inverse=False):
            self.calls += 1
            assert positions is not None
            assert positions.dtype == torch.long
            self.positions_shape = positions.shape
            self.received_positions = positions
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

        for family, enabled in (
            ("kimi_k2_7", True),
            ("kimi_k3", True),
            ("qwen3_5", True),
            ("muse_spark", True),
            ("muse_glimmer", True),
            ("muse_spark", False),
        ):
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
            if not enabled:
                config.block.attn.rope = IdentityRoPE.Config()
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
            assert isinstance(config.block.attn.rope, MarkerRoPE.Config)
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
                for calls, grid in enumerate(([1, 4, 4], [1, 2, 8]), start=1):
                    encoder(
                        torch.randn(16, projection.in_features),
                        grid_thw=torch.tensor([grid]),
                    )
                    ropes = [layer.attn.rope for layer in encoder.layers.values()]
                    assert sum(rope.prepare_calls for rope in ropes) == calls
                    prepared_positions = ropes[0].prepared_positions
                    for layer in encoder.layers.values():
                        assert isinstance(layer.attn.rope, RoPE)
                        assert not list(layer.attn.rope.named_buffers())
                        assert layer.attn.rope.calls == calls
                        assert layer.attn.rope.positions_shape == (16, 2)
                        assert layer.attn.rope.received_positions is prepared_positions
                        assert layer.attn.flex_attention.calls == calls


if __name__ == "__main__":
    unittest.main()
