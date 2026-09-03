# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn.attention.flex_attention import create_block_mask

from torchtitan.models.common.attention import FlexAttention


class TestFlexAttentionLayouts(unittest.TestCase):
    def setUp(self) -> None:
        self.attention = FlexAttention(FlexAttention.Config())

    @staticmethod
    def _mask(seq_len: int, batch_size: int):
        return create_block_mask(
            lambda b, h, q_idx, kv_idx: q_idx >= kv_idx,
            B=batch_size,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device="cpu",
        )

    def test_thk_thv_layout(self) -> None:
        num_tokens, num_heads, head_dim = 8, 4, 16
        q_THK = torch.randn(num_tokens, num_heads, head_dim)
        k_THK = torch.randn_like(q_THK)
        v_THV = torch.randn_like(q_THK)

        def kernel(q_1HTK, k_1HTK, v_1HTV, **kwargs):
            self.assertEqual(q_1HTK.shape, (1, num_heads, num_tokens, head_dim))
            self.assertEqual(k_1HTK.shape, q_1HTK.shape)
            self.assertEqual(v_1HTV.shape, q_1HTK.shape)
            lse_1HT = torch.randn(1, num_heads, num_tokens)
            return q_1HTK, SimpleNamespace(lse=lse_1HT)

        with patch.object(FlexAttention, "compiled_flex_attn", side_effect=kernel):
            out_THV = self.attention(
                q_THK,
                k_THK,
                v_THV,
                attention_masks=self._mask(num_tokens, 1),
            )

        torch.testing.assert_close(out_THV, q_THK)

    def test_thk_thv_out_transform_layout(self) -> None:
        num_tokens, num_heads, head_dim = 8, 4, 16
        q_THK = torch.randn(num_tokens, num_heads, head_dim)
        expected_lse_TH = torch.randn(num_tokens, num_heads)

        def kernel(q_1HTK, k_1HTK, v_1HTV, **kwargs):
            return q_1HTK, SimpleNamespace(
                lse=expected_lse_TH.transpose(0, 1).unsqueeze(0)
            )

        def out_transform(out_THV, lse_TH):
            torch.testing.assert_close(lse_TH, expected_lse_TH)
            return out_THV

        with patch.object(FlexAttention, "compiled_flex_attn", side_effect=kernel):
            out_THV = self.attention(
                q_THK,
                q_THK,
                q_THK,
                attention_masks=self._mask(num_tokens, 1),
                out_transform=out_transform,
            )

        torch.testing.assert_close(out_THV, q_THK)


if __name__ == "__main__":
    unittest.main()
