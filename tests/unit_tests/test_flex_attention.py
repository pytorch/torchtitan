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

    def test_tnh_layout(self) -> None:
        num_tokens, num_heads, head_dim = 8, 4, 16
        q_TNH = torch.randn(num_tokens, num_heads, head_dim)
        k_TNH = torch.randn_like(q_TNH)
        v_TNH = torch.randn_like(q_TNH)

        def kernel(q_BNTH, k_BNTH, v_BNTH, **kwargs):
            self.assertEqual(q_BNTH.shape, (1, num_heads, num_tokens, head_dim))
            self.assertEqual(k_BNTH.shape, q_BNTH.shape)
            self.assertEqual(v_BNTH.shape, q_BNTH.shape)
            lse_BNT = torch.randn(1, num_heads, num_tokens)
            return q_BNTH, SimpleNamespace(lse=lse_BNT)

        with patch.object(FlexAttention, "compiled_flex_attn", side_effect=kernel):
            out_TNH = self.attention(
                q_TNH,
                k_TNH,
                v_TNH,
                attention_masks=self._mask(num_tokens, 1),
            )

        torch.testing.assert_close(out_TNH, q_TNH)

    def test_tnh_out_transform_layout(self) -> None:
        num_tokens, num_heads, head_dim = 8, 4, 16
        q_TNH = torch.randn(num_tokens, num_heads, head_dim)
        expected_lse_TN = torch.randn(num_tokens, num_heads)

        def kernel(q_BNTH, k_BNTH, v_BNTH, **kwargs):
            return q_BNTH, SimpleNamespace(
                lse=expected_lse_TN.transpose(0, 1).unsqueeze(0)
            )

        def out_transform(out_TNH, lse_TN):
            torch.testing.assert_close(lse_TN, expected_lse_TN)
            return out_TNH

        with patch.object(FlexAttention, "compiled_flex_attn", side_effect=kernel):
            out_TNH = self.attention(
                q_TNH,
                q_TNH,
                q_TNH,
                attention_masks=self._mask(num_tokens, 1),
                out_transform=out_transform,
            )

        torch.testing.assert_close(out_TNH, q_TNH)


if __name__ == "__main__":
    unittest.main()
