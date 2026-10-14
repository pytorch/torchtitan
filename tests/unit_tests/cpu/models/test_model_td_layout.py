# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.deepseek_v3 import (
    build_model_config as build_deepseek_v3_model_config,
    MODEL_FLAVORS as DEEPSEEK_V3_MODEL_FLAVORS,
)
from torchtitan.models.gpt_oss import (
    build_model_config as build_gpt_oss_model_config,
    MODEL_FLAVORS as GPT_OSS_MODEL_FLAVORS,
)
from torchtitan.models.muse_glimmer import MODEL_FLAVORS as MUSE_GLIMMER_MODEL_FLAVORS
from torchtitan.models.qwen3_5 import MODEL_FLAVORS as QWEN3_5_MODEL_FLAVORS


class _AttentionOutput(nn.Module):
    def forward(self, q_THK, k_THK, v_THV, *, out_transform=None, **kwargs):
        num_q_heads = q_THK.shape[1]
        num_v_heads = v_THV.shape[1]
        out_THV = v_THV.repeat_interleave(num_q_heads // num_v_heads, dim=1)
        if out_transform is not None:
            lse_TH = torch.zeros(
                q_THK.shape[:2], device=q_THK.device, dtype=q_THK.dtype
            )
            out_THV = out_transform(out_THV, lse_TH)
        return out_THV


class TestModelTDLayout(unittest.TestCase):
    def test_gpt_oss_attention_preserves_td_shape(self):
        _, max_context_length = GPT_OSS_MODEL_FLAVORS["debugmodel"]
        config = build_gpt_oss_model_config(
            "debugmodel",
            attn_backend="varlen",
            seq_len=max_context_length,
            parallelism=ParallelismConfig(),
        )
        attention = config.layers[0].attention.build()
        attention.inner_attention = _AttentionOutput()
        x_TD = torch.randn(8, config.dim)
        positions_T = torch.arange(8)

        out_TD = attention(x_TD, None, positions_T)

        self.assertEqual(out_TD.shape, x_TD.shape)

    def test_deepseek_attention_preserves_td_shape(self):
        _, max_context_length = DEEPSEEK_V3_MODEL_FLAVORS["debugmodel"]
        config = build_deepseek_v3_model_config(
            "debugmodel",
            attn_backend="flex",
            seq_len=max_context_length,
            parallelism=ParallelismConfig(),
        )
        attention = config.layers[0].attention.build()
        attention.inner_attention = _AttentionOutput()
        x_TD = torch.randn(8, config.dim)
        positions_T = torch.arange(8)

        out_TD = attention(x_TD, None, positions_T)

        self.assertEqual(out_TD.shape, x_TD.shape)

    def test_muse_attention_preserves_td_shape(self):
        build_config, max_context_length = MUSE_GLIMMER_MODEL_FLAVORS["debugmodel"]
        config = build_config(
            "varlen", seq_len=max_context_length, parallelism=ParallelismConfig()
        )
        attention = config.layers[0].attention.build()
        attention.inner_attention = _AttentionOutput()
        x_TD = torch.randn(8, config.dim)
        positions_T = torch.arange(8)
        attention_metadata = {"swa_128": None}

        out_TD = attention(x_TD, attention_metadata, positions_T)

        self.assertEqual(out_TD.shape, x_TD.shape)

    def test_qwen35_attention_preserves_td_shape(self):
        build_config, max_context_length = QWEN3_5_MODEL_FLAVORS["debugmodel"]
        config = build_config(
            "varlen", seq_len=max_context_length, parallelism=ParallelismConfig()
        )
        attention_config = config.first_full_attention
        assert attention_config is not None
        attention = attention_config.build()
        attention.inner_attention = _AttentionOutput()
        x_TD = torch.randn(8, config.dim)
        positions_T = torch.arange(8)

        out_TD = attention(x_TD, None, positions_T)

        self.assertEqual(out_TD.shape, x_TD.shape)


if __name__ == "__main__":
    unittest.main()
