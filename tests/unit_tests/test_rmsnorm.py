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
        config = RMSNorm.Config(normalized_shape=32)
        norm = config.build()
        self.assertIsInstance(norm, RMSNorm)
        self.assertIsInstance(norm, nn.RMSNorm)
        self.assertEqual(norm.weight.shape, torch.Size([32]))

    def test_config_build_without_fields_raises(self):
        """RMSNorm.Config() raises TypeError when normalized_shape is not provided."""
        with self.assertRaises(TypeError):
            RMSNorm.Config()

    def test_init_states(self):
        """init_states re-initializes the weight tensor."""
        config = RMSNorm.Config(
            normalized_shape=16, param_init={"weight": nn.init.ones_}
        )
        norm = config.build()

        nn.init.zeros_(norm.weight)
        self.assertTrue(torch.all(norm.weight == 0))
        norm.init_states()
        self.assertTrue(torch.all(norm.weight == 1))

    def test_custom_eps(self):
        """RMSNorm respects custom eps."""
        config = RMSNorm.Config(normalized_shape=32, eps=1e-6)
        norm = config.build()
        self.assertEqual(norm.eps, 1e-6)

    def test_elementwise_affine_false(self):
        """RMSNorm supports elementwise_affine=False."""
        config = RMSNorm.Config(normalized_shape=16, elementwise_affine=False)
        norm = config.build()
        self.assertIsNone(norm.weight)

    def test_forward(self):
        """Forward pass works through nn.RMSNorm's implementation."""
        config = RMSNorm.Config(normalized_shape=32)
        norm = config.build()
        x = torch.randn(2, 10, 32)
        out = norm(x)
        self.assertEqual(out.shape, torch.Size([2, 10, 32]))

    def test_normalized_shape_required_at_config_init(self):
        """normalized_shape is a required field and must be passed to Config()."""
        # Can be passed at construction
        config = RMSNorm.Config(normalized_shape=32)
        self.assertEqual(config.normalized_shape, 32)
        # Omitting it raises TypeError
        with self.assertRaises(TypeError):
            RMSNorm.Config()

    def test_shared_config(self):
        """Multiple RMSNorm.Config instances with different shapes can be built independently."""
        config1 = RMSNorm.Config(normalized_shape=16, eps=1e-6)
        norm1 = config1.build()
        config2 = RMSNorm.Config(normalized_shape=32, eps=1e-6)
        norm2 = config2.build()

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
