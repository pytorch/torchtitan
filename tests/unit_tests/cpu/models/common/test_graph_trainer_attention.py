# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.experiments.graph_trainer.attention import (
    ScaledDotProductInnerAttention,
)
from torchtitan.experiments.graph_trainer.common_utils import (
    GraphTrainerScaledDotProductInnerAttention,
)


class TestScaledDotProductInnerAttention(unittest.TestCase):
    def test_sdpa_preserves_blhv_shape(self):
        attention = ScaledDotProductInnerAttention.Config().build()
        q_BLHK = torch.randn(2, 8, 4, 16)
        k_BLHK = torch.randn(2, 8, 2, 16)
        v_BLHV = torch.randn(2, 8, 2, 16)

        out_BLHV = attention(q_BLHK, k_BLHK, v_BLHV, enable_gqa=True)

        self.assertEqual(out_BLHV.shape, q_BLHK.shape)

    def test_graph_trainer_sdpa_preserves_thv_shape(self):
        attention = GraphTrainerScaledDotProductInnerAttention.Config().build()
        q_THK = torch.randn(8, 4, 16)
        k_THK = torch.randn(8, 2, 16)
        v_THV = torch.randn(8, 2, 16)

        out_THV = attention(q_THK, k_THK, v_THV, enable_gqa=True)

        self.assertEqual(out_THV.shape, q_THK.shape)
