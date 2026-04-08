# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.models.gpt_oss.state_dict_adapter import GptOssStateDictAdapter
from torchtitan.protocols.state_dict_adapter import StateDictAdapter


class TestFusedQKVWeightRoundTrip(unittest.TestCase):
    """Test that fusing and unfusing QKV weights is lossless."""

    def _round_trip_weight(self, n_heads, n_kv_heads, head_dim, dim):
        """Verify separate → fused → separate round-trip for weights."""
        wq = torch.randn(n_heads * head_dim, dim)
        wk = torch.randn(n_kv_heads * head_dim, dim)
        wv = torch.randn(n_kv_heads * head_dim, dim)

        fused = StateDictAdapter.separate_to_fused_qkv(
            wq, wk, wv, n_heads, n_kv_heads, head_dim
        )
        wq2, wk2, wv2 = StateDictAdapter.fused_to_separate_qkv(
            fused, n_heads, n_kv_heads, head_dim
        )

        torch.testing.assert_close(wq, wq2)
        torch.testing.assert_close(wk, wk2)
        torch.testing.assert_close(wv, wv2)

    def test_mha(self):
        """MHA: n_heads == n_kv_heads."""
        self._round_trip_weight(n_heads=8, n_kv_heads=8, head_dim=64, dim=512)

    def test_gqa(self):
        """GQA: n_heads > n_kv_heads."""
        self._round_trip_weight(n_heads=32, n_kv_heads=8, head_dim=64, dim=2048)

    def test_gqa_small(self):
        """GQA with small dims (debug model sizes)."""
        self._round_trip_weight(n_heads=4, n_kv_heads=2, head_dim=32, dim=128)


class TestFusedQKVBiasRoundTrip(unittest.TestCase):
    """Test that fusing and unfusing QKV biases is lossless (GPT-OSS specific)."""

    def _round_trip_bias(self, n_heads, n_kv_heads, head_dim):
        """Verify separate → fused → separate round-trip for biases."""
        bq = torch.randn(n_heads * head_dim)
        bk = torch.randn(n_kv_heads * head_dim)
        bv = torch.randn(n_kv_heads * head_dim)

        fused = GptOssStateDictAdapter._separate_to_fused_qkv_bias(
            bq, bk, bv, n_heads, n_kv_heads, head_dim
        )

        # Verify fused shape
        heads_per_kv = n_heads // n_kv_heads
        r_dim = heads_per_kv + 2
        self.assertEqual(fused.shape, (n_kv_heads * r_dim * head_dim,))

        bq2, bk2, bv2 = GptOssStateDictAdapter._fused_to_separate_qkv_bias(
            fused, n_heads, n_kv_heads, head_dim
        )

        torch.testing.assert_close(bq, bq2)
        torch.testing.assert_close(bk, bk2)
        torch.testing.assert_close(bv, bv2)

    def test_mha(self):
        self._round_trip_bias(n_heads=8, n_kv_heads=8, head_dim=64)

    def test_gqa(self):
        self._round_trip_bias(n_heads=32, n_kv_heads=8, head_dim=64)

    def test_gqa_gptoss_dims(self):
        """GPT-OSS default: 64 heads, 8 kv heads, head_dim=64."""
        self._round_trip_bias(n_heads=64, n_kv_heads=8, head_dim=64)

    def test_reverse_round_trip(self):
        """Verify fused → separate → fused round-trip."""
        n_heads, n_kv_heads, head_dim = 32, 8, 64
        heads_per_kv = n_heads // n_kv_heads
        r_dim = heads_per_kv + 2
        fused = torch.randn(n_kv_heads * r_dim * head_dim)

        bq, bk, bv = GptOssStateDictAdapter._fused_to_separate_qkv_bias(
            fused, n_heads, n_kv_heads, head_dim
        )
        fused2 = GptOssStateDictAdapter._separate_to_fused_qkv_bias(
            bq, bk, bv, n_heads, n_kv_heads, head_dim
        )

        torch.testing.assert_close(fused, fused2)


if __name__ == "__main__":
    unittest.main()
