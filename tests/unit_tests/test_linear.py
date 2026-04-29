# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from functools import partial

import torch
import torch.nn as nn

from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module


class TestLinear(unittest.TestCase):
    """Tests for the Linear class used in the codebase."""

    def test_config_build(self):
        """Linear.Config.build() creates a working linear."""
        config = Linear.Config(in_features=32, out_features=16)
        linear = config.build()
        self.assertIsInstance(linear, Linear)
        self.assertIsInstance(linear, nn.Linear)
        self.assertEqual(linear.weight.shape, torch.Size([16, 32]))
        self.assertIsNone(linear.bias)

    def test_config_build_with_bias(self):
        """Linear.Config(bias=True).build() creates a linear with bias."""
        config = Linear.Config(in_features=32, out_features=16, bias=True)
        linear = config.build()
        self.assertIsNotNone(linear.bias)
        self.assertEqual(linear.bias.shape, torch.Size([16]))

    def test_config_build_without_fields_raises(self):
        """Linear.Config() raises TypeError when required features are not provided."""
        with self.assertRaises(TypeError):
            Linear.Config()

    def test_init_states(self):
        """init_states re-initializes the weight tensor."""
        config = Linear.Config(
            in_features=16,
            out_features=8,
            param_init={
                "weight": partial(nn.init.trunc_normal_, std=0.02),
                "bias": nn.init.zeros_,
            },
        )
        linear = config.build()

        with torch.no_grad():
            nn.init.zeros_(linear.weight)
            self.assertTrue(torch.all(linear.weight == 0))
            linear.init_states()
            self.assertFalse(torch.all(linear.weight == 0))

    def test_custom_init_std(self):
        """Linear respects custom mean and std."""
        config = Linear.Config(
            in_features=1000,
            out_features=500,
            param_init={
                "weight": partial(nn.init.normal_, mean=0.1, std=0.02),
                "bias": nn.init.zeros_,
            },
        )
        linear = config.build()

        torch.manual_seed(42)
        with torch.no_grad():
            linear.init_states()
        # With large amount of samples (1000 * 500) the sample statistics should
        # be close to the requested values. places=3 checks within 0.0005, which
        # is well within statistical tolerance for this sample size.
        self.assertAlmostEqual(linear.weight.mean().item(), 0.1, places=3)
        self.assertAlmostEqual(linear.weight.std().item(), 0.02, places=3)

    def test_forward(self):
        """Forward pass works through nn.Linear's implementation."""
        config = Linear.Config(in_features=32, out_features=16)
        linear = config.build()
        x = torch.randn(2, 10, 32)
        out = linear(x)
        self.assertEqual(out.shape, torch.Size([2, 10, 16]))

    def test_shared_config_builds_independent_instances(self):
        """A single Linear.Config can build multiple independent linears."""
        cfg1 = Linear.Config(in_features=32, out_features=16)
        l1 = cfg1.build()
        cfg2 = Linear.Config(in_features=64, out_features=8)
        l2 = cfg2.build()
        self.assertIsNot(l1, l2)
        self.assertEqual(l1.weight.shape, torch.Size([16, 32]))
        self.assertEqual(l2.weight.shape, torch.Size([8, 64]))

    def test_isinstance_checks(self):
        """Linear is instance of nn.Linear, and Module."""
        config = Linear.Config(in_features=8, out_features=4)
        linear = config.build()
        self.assertIsInstance(linear, nn.Linear)
        self.assertIsInstance(linear, Module)

    def test_default_bias_false(self):
        """Linear.Config defaults to bias=False."""
        config = Linear.Config(in_features=4, out_features=4)
        self.assertFalse(config.bias)

    def test_direct_construction(self):
        """Linear can be constructed directly (Flux-style, non-Configurable parents)."""
        config = Linear.Config(in_features=32, out_features=16, bias=True)
        linear = Linear(config)
        self.assertIsInstance(linear, Linear)
        self.assertIsNotNone(linear.bias)

    def test_config_pre_specified_build(self):
        """Linear.Config with both fields pre-specified builds with no kwargs."""
        config = Linear.Config(in_features=32, out_features=16)
        linear = config.build()
        self.assertIsInstance(linear, Linear)
        self.assertEqual(linear.weight.shape, torch.Size([16, 32]))

    def test_config_partial_pre_specified(self):
        """Linear.Config with fields specified at construction builds correctly."""
        config = Linear.Config(in_features=32, out_features=16)
        linear = config.build()
        self.assertIsInstance(linear, Linear)
        self.assertEqual(linear.weight.shape, torch.Size([16, 32]))


if __name__ == "__main__":
    unittest.main()
