# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass

import torch
import torch.nn as nn

from torchtitan.models.common.embedding import Embedding


class TestEmbedding(unittest.TestCase):
    """Tests for the Embedding class used in the codebase."""

    def test_config_build(self):
        """Embedding.Config.build() creates a working embedding."""
        config = Embedding.Config()
        emb = config.build(num_embeddings=100, embedding_dim=32)
        self.assertIsInstance(emb, Embedding)
        self.assertIsInstance(emb, nn.Embedding)
        self.assertEqual(emb.weight.shape, torch.Size([100, 32]))

    def test_config_build_without_fields_raises(self):
        """Embedding.Config.build() raises when fields are not passed."""
        config = Embedding.Config()
        with self.assertRaises(TypeError):
            config.build()

    def test_init_weights(self):
        """Embedding.init_weights re-initializes the weight tensor."""
        config = Embedding.Config()
        emb = config.build(num_embeddings=50, embedding_dim=16)

        nn.init.zeros_(emb.weight)
        self.assertTrue(torch.all(emb.weight == 0))
        emb.init_weights()
        # After init_weights, weights should no longer be all zero
        self.assertFalse(torch.all(emb.weight == 0))

    def test_custom_init_std(self):
        """Embedding respects custom init_mean and init_std."""
        config = Embedding.Config(init_mean=0.1, init_std=0.02)
        emb = config.build(num_embeddings=1000, embedding_dim=160)

        torch.manual_seed(42)
        emb.init_weights()
        # With large amount of samples (160 * 1000) the sample statistics should
        # be close to the requested values. places=3 checks within 0.0005, which
        # is well within statistical tolerance for this sample size.
        self.assertAlmostEqual(emb.weight.mean().item(), 0.1, places=3)
        self.assertAlmostEqual(emb.weight.std().item(), 0.02, places=3)

    def test_config_pre_specified_build(self):
        """Embedding.Config with both fields pre-specified builds with no kwargs."""
        config = Embedding.Config()
        config.num_embeddings = 100
        config.embedding_dim = 32
        emb = config.build()
        self.assertIsInstance(emb, Embedding)
        self.assertEqual(emb.weight.shape, torch.Size([100, 32]))

    def test_config_partial_pre_specified(self):
        """Embedding.Config with one field pre-specified, other via build()."""
        config = Embedding.Config()
        config.num_embeddings = 100
        emb = config.build(embedding_dim=32)
        self.assertIsInstance(emb, Embedding)
        self.assertEqual(emb.weight.shape, torch.Size([100, 32]))

    def test_config_inheritance_preset(self):
        """Inheriting Embedding.Config can put fields back in __init__."""

        @dataclass(kw_only=True, slots=True)
        class PresetConfig(Embedding.Config):
            num_embeddings: int = 100
            embedding_dim: int = 32

        config = PresetConfig()
        emb = config.build()
        self.assertIsInstance(emb, Embedding)
        self.assertEqual(emb.weight.shape, torch.Size([100, 32]))


if __name__ == "__main__":
    unittest.main()
