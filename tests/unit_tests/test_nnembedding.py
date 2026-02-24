# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from torchtitan.models.common.embedding import NNEmbedding


class TestNNEmbedding(unittest.TestCase):
    """Tests for the NNEmbedding class used in the codebase."""

    def test_config_build(self):
        """NNEmbedding.Config.build() creates a working embedding."""
        config = NNEmbedding.Config()
        config.num_embeddings = 100
        config.embedding_dim = 32
        emb = config.build()
        self.assertIsInstance(emb, NNEmbedding)
        self.assertIsInstance(emb, nn.Embedding)
        self.assertEqual(emb.weight.shape, torch.Size([100, 32]))

    def test_config_build_without_fields_raises(self):
        """NNEmbedding.Config.build() raises when fields are not set."""
        config = NNEmbedding.Config()
        with self.assertRaises(ValueError):
            config.build()

    def test_init_weights(self):
        """NNEmbedding.init_weights re-initializes the weight tensor."""
        config = NNEmbedding.Config()
        config.num_embeddings = 50
        config.embedding_dim = 16
        emb = config.build()

        # Set all weights to zero, then call init_weights
        nn.init.zeros_(emb.weight)
        self.assertTrue(torch.all(emb.weight == 0))
        emb.init_weights()
        # After init_weights, weights should no longer be all zero
        self.assertFalse(torch.all(emb.weight == 0))

    def test_custom_init_std(self):
        """NNEmbedding respects custom init_mean and init_std."""
        config = NNEmbedding.Config(init_mean=0.0, init_std=0.02)
        config.num_embeddings = 1000
        config.embedding_dim = 64
        emb = config.build()

        torch.manual_seed(42)
        emb.init_weights()
        # Deterministic check: std must be within 2x of the requested init_std
        self.assertLess(emb.weight.std().item(), config.init_std * 2)


if __name__ == "__main__":
    unittest.main()
