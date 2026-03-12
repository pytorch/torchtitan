# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from torchtitan.models.common.rmsnorm import RMSNorm


class TestRMSNorm(unittest.TestCase):
    """Tests for the RMSNorm class used in the codebase."""

    def test_config_build(self):
        """RMSNorm.Config.build() creates a working norm."""
        config = RMSNorm.Config()
        norm = config.build(normalized_shape=32)
        self.assertIsInstance(norm, RMSNorm)
        self.assertIsInstance(norm, nn.RMSNorm)
        self.assertEqual(norm.weight.shape, torch.Size([32]))

    def test_config_build_without_fields_raises(self):
        """RMSNorm.Config.build() raises when normalized_shape is not passed."""
        config = RMSNorm.Config()
        with self.assertRaises(TypeError):
            config.build()

    def test_init_weights(self):
        """RMSNorm.init_weights re-initializes the weight tensor."""
        config = RMSNorm.Config()
        norm = config.build(normalized_shape=16)

        # Set weights to zero, then call init_weights
        nn.init.zeros_(norm.weight)
        self.assertTrue(torch.all(norm.weight == 0))
        norm.init_weights()
        # After init_weights, weights should be all ones (RMSNorm default)
        self.assertTrue(torch.all(norm.weight == 1))

    def test_custom_eps(self):
        """RMSNorm respects custom eps."""
        config = RMSNorm.Config(eps=1e-6)
        norm = config.build(normalized_shape=32)
        self.assertEqual(norm.eps, 1e-6)

    def test_elementwise_affine_false(self):
        """RMSNorm supports elementwise_affine=False."""
        config = RMSNorm.Config(elementwise_affine=False)
        norm = config.build(normalized_shape=16)
        self.assertIsNone(norm.weight)

    def test_forward(self):
        """Forward pass works through nn.RMSNorm's implementation."""
        config = RMSNorm.Config()
        norm = config.build(normalized_shape=32)
        x = torch.randn(2, 10, 32)
        out = norm(x)
        self.assertEqual(out.shape, torch.Size([2, 10, 32]))

    def test_normalized_shape_excluded_from_config_init(self):
        """normalized_shape uses field(init=False), so it cannot be passed to Config()."""
        with self.assertRaises(TypeError):
            RMSNorm.Config(normalized_shape=32)

    def test_shared_config(self):
        """A single RMSNorm.Config can build multiple independent norms.

        This verifies that sharing a config instance across model variants
        is safe because build() clones the config internally.
        """
        config = RMSNorm.Config(eps=1e-6)
        norm1 = config.build(normalized_shape=16)
        norm2 = config.build(normalized_shape=32)

        # They are separate module instances
        self.assertIsNot(norm1, norm2)

        # Each gets its own normalized_shape
        self.assertEqual(norm1.weight.shape, torch.Size([16]))
        self.assertEqual(norm2.weight.shape, torch.Size([32]))

        # But share the same eps configuration
        self.assertEqual(norm1.eps, norm2.eps)

        # Modifying one doesn't affect the other
        nn.init.zeros_(norm1.weight)
        self.assertTrue(torch.all(norm1.weight == 0))
        self.assertTrue(torch.all(norm2.weight == 1))


if __name__ == "__main__":
    unittest.main()
