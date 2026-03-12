# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from torchtitan.components.quantization.module_utils import (
    capture_module_attrs,
    inject_module_protocol,
    verify_module_protocol,
)
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module


class TestLinear(unittest.TestCase):
    """Tests for the Linear class used in the codebase."""

    def test_config_build(self):
        """Linear.Config.build() creates a working linear."""
        config = Linear.Config()
        linear = config.build(in_features=32, out_features=16)
        self.assertIsInstance(linear, Linear)
        self.assertIsInstance(linear, nn.Linear)
        self.assertEqual(linear.weight.shape, torch.Size([16, 32]))
        self.assertIsNone(linear.bias)

    def test_config_build_with_bias(self):
        """Linear.Config(bias=True).build() creates a linear with bias."""
        config = Linear.Config(bias=True)
        linear = config.build(in_features=32, out_features=16)
        self.assertIsNotNone(linear.bias)
        self.assertEqual(linear.bias.shape, torch.Size([16]))

    def test_config_build_without_fields_raises(self):
        """Linear.Config.build() raises when features are not passed."""
        config = Linear.Config()
        with self.assertRaises(TypeError):
            config.build()

    def test_init_weights(self):
        """Linear.init_weights re-initializes the weight tensor."""
        config = Linear.Config()
        linear = config.build(in_features=16, out_features=8)

        with torch.no_grad():
            # Set weights to zero, then call init_weights
            nn.init.zeros_(linear.weight)
            self.assertTrue(torch.all(linear.weight == 0))
            linear.init_weights()
            # After init_weights, weights should no longer be all zero
            self.assertFalse(torch.all(linear.weight == 0))

    def test_custom_init_std(self):
        """Linear respects custom init_mean and init_std."""
        config = Linear.Config(init_mean=0.1, init_std=0.02)
        linear = config.build(in_features=1000, out_features=500)

        torch.manual_seed(42)
        with torch.no_grad():
            linear.init_weights()
        # With large amount of samples (1000 * 500) the sample statistics should
        # be close to the requested values. places=3 checks within 0.0005, which
        # is well within statistical tolerance for this sample size.
        self.assertAlmostEqual(linear.weight.mean().item(), 0.1, places=3)
        self.assertAlmostEqual(linear.weight.std().item(), 0.02, places=3)

    def test_forward(self):
        """Forward pass works through nn.Linear's implementation."""
        config = Linear.Config()
        linear = config.build(in_features=32, out_features=16)
        x = torch.randn(2, 10, 32)
        out = linear(x)
        self.assertEqual(out.shape, torch.Size([2, 10, 16]))

    def test_shared_config_builds_independent_instances(self):
        """A single Linear.Config can build multiple independent linears."""
        config = Linear.Config()
        l1 = config.build(in_features=32, out_features=16)
        l2 = config.build(in_features=64, out_features=8)
        self.assertIsNot(l1, l2)
        self.assertEqual(l1.weight.shape, torch.Size([16, 32]))
        self.assertEqual(l2.weight.shape, torch.Size([8, 64]))

    def test_isinstance_checks(self):
        """Linear is instance of nn.Linear, and Module."""
        config = Linear.Config()
        linear = config.build(in_features=8, out_features=4)
        self.assertIsInstance(linear, nn.Linear)
        self.assertIsInstance(linear, Module)

    def test_default_bias_false(self):
        """Linear.Config defaults to bias=False."""
        config = Linear.Config()
        self.assertFalse(config.bias)

    def test_direct_construction(self):
        """Linear can be constructed directly (Flux-style, non-Configurable parents)."""
        config = Linear.Config(bias=True)
        with self.assertRaises(TypeError):
            linear = Linear(config)
        config.in_features = 32
        config.out_features = 16
        linear = Linear(config)
        self.assertIsInstance(linear, Linear)
        self.assertIsNotNone(linear.bias)

    def test_init_attrs_stored(self):
        """_init_mean and _init_std are stored on the instance."""
        config = Linear.Config(init_mean=0.1, init_std=0.05)
        linear = config.build(in_features=8, out_features=4)
        self.assertEqual(linear._init_mean, 0.1)
        self.assertEqual(linear._init_std, 0.05)

    def test_config_pre_specified_build(self):
        """Linear.Config with both fields pre-specified builds with no kwargs."""
        config = Linear.Config()
        config.in_features = 32
        config.out_features = 16
        linear = config.build()
        self.assertIsInstance(linear, Linear)
        self.assertEqual(linear.weight.shape, torch.Size([16, 32]))

    def test_config_partial_pre_specified(self):
        """Linear.Config with one field pre-specified, other via build()."""
        config = Linear.Config()
        config.in_features = 32
        linear = config.build(out_features=16)
        self.assertIsInstance(linear, Linear)
        self.assertEqual(linear.weight.shape, torch.Size([16, 32]))


class TestModuleInjection(unittest.TestCase):
    """Tests for post-quantization Module protocol injection."""

    def test_inject_on_plain_nn_linear(self):
        """Injection adds Linear (and thus Module) to plain nn.Linear subclass."""

        class FakeQuantLinear(nn.Linear):
            pass

        model = nn.Module()
        model.fc = FakeQuantLinear(8, 4)
        self.assertNotIsInstance(model.fc, Module)

        inject_module_protocol(model, Linear)
        self.assertIsInstance(model.fc, Linear)
        self.assertIsInstance(model.fc, nn.Linear)
        self.assertIsInstance(model.fc, FakeQuantLinear)

    def test_no_inject_on_our_linear(self):
        """Our Linear (already Module) is not patched."""
        model = nn.Module()
        config = Linear.Config()
        model.fc = config.build(in_features=8, out_features=4)
        orig_cls = type(model.fc)

        inject_module_protocol(model, Linear)
        self.assertIs(type(model.fc), orig_cls)  # class unchanged

    def test_init_weights_after_injection(self):
        """init_weights works on injected module via Linear's MRO."""

        class FakeQuantLinear(nn.Linear):
            pass

        model = nn.Module()
        config = Linear.Config(init_std=0.03)
        model.fc = config.build(in_features=8, out_features=4)

        # Capture attrs, simulate conversion, inject
        saved_attrs = capture_module_attrs(model, ["_init_mean", "_init_std"])
        model.fc = FakeQuantLinear(8, 4)
        inject_module_protocol(model, Linear, saved_attrs)

        # Should not raise — init_weights comes from Linear via MRO
        with torch.no_grad():
            model.fc.init_weights()

    def test_injection_cached_across_instances(self):
        """Same original class gets the same patched class."""

        class FakeQuantLinear(nn.Linear):
            pass

        model = nn.Module()
        model.fc1 = FakeQuantLinear(8, 4)
        model.fc2 = FakeQuantLinear(16, 8)
        inject_module_protocol(model, Linear)

        self.assertIs(type(model.fc1), type(model.fc2))

    def test_capture_and_reattach_attrs(self):
        """capture_module_attrs + inject_module_protocol round-trips attrs."""

        class FakeQuantLinear(nn.Linear):
            pass

        # Build model with our Linear
        model = nn.Module()
        config = Linear.Config(init_std=0.05, bias=True)
        model.fc = config.build(in_features=8, out_features=4)

        # Capture attrs
        saved = capture_module_attrs(model, ["_init_mean", "_init_std"])
        self.assertIn("fc", saved)
        self.assertIn("_init_mean", saved["fc"])
        self.assertIn("_init_std", saved["fc"])

        # Simulate Float8 conversion: replace with a new FakeQuantLinear
        model.fc = FakeQuantLinear(8, 4, bias=True)
        self.assertNotIsInstance(model.fc, Module)
        self.assertFalse(hasattr(model.fc, "_init_std"))

        # Inject and re-attach
        inject_module_protocol(model, Linear, saved)
        self.assertIsInstance(model.fc, Module)
        self.assertIsInstance(model.fc, Linear)
        self.assertEqual(model.fc._init_std, 0.05)

    def test_inject_idempotent(self):
        """Calling inject_module_protocol twice is a no-op the second time."""

        class FakeQuantLinear(nn.Linear):
            pass

        model = nn.Module()
        model.fc = FakeQuantLinear(8, 4)
        inject_module_protocol(model, Linear)

        cls_after_first = type(model.fc)
        self.assertIsInstance(model.fc, Module)

        # Second call should not change the class again
        inject_module_protocol(model, Linear)
        self.assertIs(type(model.fc), cls_after_first)
        self.assertIsInstance(model.fc, Module)

    def test_mx_style_class_swap_preserves_attrs(self):
        """MX-style class swap preserves instance attributes."""

        class FakeQuantLinear(nn.Linear):
            pass

        # Build model with our Linear
        model = nn.Module()
        config = Linear.Config(init_std=0.03)
        model.fc = config.build(in_features=8, out_features=4)

        # Simulate MX conversion: class swap (instance attrs survive)
        model.fc.__class__ = FakeQuantLinear
        self.assertFalse(isinstance(model.fc, Module))
        self.assertTrue(hasattr(model.fc, "_init_std"))  # attrs survive

        # Inject Linear back
        inject_module_protocol(model, Linear)
        self.assertIsInstance(model.fc, Module)
        self.assertIsInstance(model.fc, Linear)
        self.assertEqual(model.fc._init_std, 0.03)

    def test_patched_class_cannot_be_constructed(self):
        """Patched class __init__ raises RuntimeError."""

        class FakeQuantLinear(nn.Linear):
            pass

        model = nn.Module()
        model.fc = FakeQuantLinear(8, 4)
        inject_module_protocol(model, Linear)

        patched_cls = type(model.fc)
        with self.assertRaises(RuntimeError):
            patched_cls(8, 4)


class TestVerifyModuleProtocol(unittest.TestCase):
    """Tests for verify_module_protocol."""

    def test_passes_when_all_satisfy(self):
        """No error when all nn.Linear modules are Linear."""
        model = nn.Module()
        config = Linear.Config()
        model.fc1 = config.build(in_features=8, out_features=4)
        model.fc2 = config.build(in_features=4, out_features=2)

        # Should not raise
        verify_module_protocol(model, nn.Linear, Linear)

    def test_raises_when_missing(self):
        """Raises RuntimeError listing offending FQNs."""
        model = nn.Module()
        model.fc1 = nn.Linear(8, 4)  # plain nn.Linear, not Module

        with self.assertRaises(RuntimeError) as ctx:
            verify_module_protocol(model, nn.Linear, Linear)
        self.assertIn("fc1", str(ctx.exception))

    def test_mixed_passes_after_injection(self):
        """After injection, verify passes even with mixed original classes."""

        class FakeQuantLinear(nn.Linear):
            pass

        model = nn.Module()
        config = Linear.Config()
        model.fc1 = config.build(in_features=8, out_features=4)
        model.fc2 = FakeQuantLinear(16, 8)

        inject_module_protocol(model, Linear)
        # Should not raise — all are now Linear
        verify_module_protocol(model, nn.Linear, Linear)


if __name__ == "__main__":
    unittest.main()
