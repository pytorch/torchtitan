# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from torchtitan.experiments.graph_trainer.common_utils import (
    GraphTrainerScaledDotProductAttention,
)
from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.models.deepseek_v3 import deepseekv3_configs
from torchtitan.models.gpt_oss import gptoss_configs
from torchtitan.models.muse_glimmer import muse_glimmer_configs
from torchtitan.models.qwen3_5 import qwen3_5_configs


class _AttentionOutput(nn.Module):
    def forward(self, q_TNH, k_TNH, v_TNH, *, out_transform=None, **kwargs):
        num_q_heads = q_TNH.shape[1]
        num_v_heads = v_TNH.shape[1]
        out_TNH = v_TNH.repeat_interleave(num_q_heads // num_v_heads, dim=1)
        if out_transform is not None:
            lse_TN = torch.zeros(
                q_TNH.shape[:2], device=q_TNH.device, dtype=q_TNH.dtype
            )
            out_TNH = out_transform(out_TNH, lse_TN)
        return out_TNH


class TestModelTDLayout(unittest.TestCase):
    def test_sdpa_preserves_blnh_shape(self):
        attention = ScaledDotProductAttention.Config().build()
        q_BLNH = torch.randn(2, 8, 4, 16)
        k_BLNH = torch.randn(2, 8, 2, 16)
        v_BLNH = torch.randn(2, 8, 2, 16)

        out_BLNH = attention(q_BLNH, k_BLNH, v_BLNH, enable_gqa=True)

        self.assertEqual(out_BLNH.shape, q_BLNH.shape)

    def test_graph_trainer_sdpa_preserves_tnh_shape(self):
        attention = GraphTrainerScaledDotProductAttention.Config().build()
        q_TNH = torch.randn(8, 4, 16)
        k_TNH = torch.randn(8, 2, 16)
        v_TNH = torch.randn(8, 2, 16)

        out_TNH = attention(q_TNH, k_TNH, v_TNH, enable_gqa=True)

        self.assertEqual(out_TNH.shape, q_TNH.shape)

    def test_gpt_oss_attention_preserves_td_shape(self):
        config = gptoss_configs["debugmodel"]("standard", "varlen")
        attention = config.layers[0].attention.build()
        attention.inner_attention = _AttentionOutput()
        x_TD = torch.randn(8, config.dim)
        positions_T = torch.arange(8)

        out_TD = attention(x_TD, None, positions_T)

        self.assertEqual(out_TD.shape, x_TD.shape)

    def test_deepseek_attention_preserves_td_shape(self):
        config = deepseekv3_configs["debugmodel"]("flex", "standard")
        attention = config.layers[0].attention.build()
        attention.inner_attention = _AttentionOutput()
        x_TD = torch.randn(8, config.dim)
        positions_T = torch.arange(8)

        out_TD = attention(x_TD, None, positions_T)

        self.assertEqual(out_TD.shape, x_TD.shape)

    def test_muse_attention_preserves_td_shape(self):
        config = muse_glimmer_configs["debugmodel"]("varlen")
        attention = config.layers[0].attention.build()
        attention.inner_attention = _AttentionOutput()
        x_TD = torch.randn(8, config.dim)
        positions_T = torch.arange(8)
        attention_masks = {"swa_128": None}

        out_TD = attention(x_TD, attention_masks, positions_T)

        self.assertEqual(out_TD.shape, x_TD.shape)

    def test_qwen35_attention_preserves_td_shape(self):
        config = qwen3_5_configs["debugmodel"]("varlen")
        attention_config = next(
            layer.attention for layer in config.layers if layer.attention is not None
        )
        attention = attention_config.build()
        attention.inner_attention = _AttentionOutput()
        x_TD = torch.randn(8, config.dim)
        positions_T = torch.arange(8)

        out_TD = attention(x_TD, None, positions_T)

        self.assertEqual(out_TD.shape, x_TD.shape)


if __name__ == "__main__":
    unittest.main()
