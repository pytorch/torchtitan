# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn

from torchtitan.models.common.embedding import Embedding


class TestEmbedding(unittest.TestCase):
    """Tests for the Embedding class used in the codebase."""

    def test_config_build(self):
        """Embedding.Config.build() creates a working embedding."""
        config = Embedding.Config(num_embeddings=100, embedding_dim=32)
        emb = config.build()
        self.assertIsInstance(emb, Embedding)
        self.assertIsInstance(emb, nn.Embedding)
        self.assertEqual(emb.weight.shape, torch.Size([100, 32]))

    def test_config_build_without_fields_raises(self):
        """Embedding.Config() raises TypeError when required fields are not provided."""
        with self.assertRaises(TypeError):
            Embedding.Config()

    def test_init_states(self):
        """init_states re-initializes the weight tensor."""
        config = Embedding.Config(
            num_embeddings=50,
            embedding_dim=16,
            param_init={"weight": partial(nn.init.trunc_normal_, std=0.02)},
        )
        emb = config.build()

        nn.init.zeros_(emb.weight)
        self.assertTrue(torch.all(emb.weight == 0))
        emb.init_states()
        self.assertFalse(torch.all(emb.weight == 0))

    def test_custom_init_std(self):
        """Embedding respects custom mean and std."""
        config = Embedding.Config(
            num_embeddings=1000,
            embedding_dim=160,
            param_init={"weight": partial(nn.init.normal_, mean=0.1, std=0.02)},
        )
        emb = config.build()

        torch.manual_seed(42)
        emb.init_states()
        # With large amount of samples (160 * 1000) the sample statistics should
        # be close to the requested values. places=3 checks within 0.0005, which
        # is well within statistical tolerance for this sample size.
        self.assertAlmostEqual(emb.weight.mean().item(), 0.1, places=3)
        self.assertAlmostEqual(emb.weight.std().item(), 0.02, places=3)

    def test_config_pre_specified_build(self):
        """Embedding.Config with both fields pre-specified builds with no kwargs."""
        config = Embedding.Config(num_embeddings=100, embedding_dim=32)
        emb = config.build()
        self.assertIsInstance(emb, Embedding)
        self.assertEqual(emb.weight.shape, torch.Size([100, 32]))

    def test_config_partial_pre_specified(self):
        """Embedding.Config with fields specified at construction builds correctly."""
        config = Embedding.Config(num_embeddings=100, embedding_dim=32)
        emb = config.build()
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
