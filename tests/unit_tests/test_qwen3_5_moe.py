# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the Qwen3.5 MoE hybrid decoder model.

Uses ``fla_backend="torch_naive"`` so the ``fla`` package is not required.
Forward pass tests require CUDA due to MoE's use of torch.histc (not
supported for Long tensors on CPU).
"""

import dataclasses
import unittest

import torch

from torchtitan.models.qwen3_5_moe import build, model_registry, qwen35_moe_configs
from torchtitan.models.qwen3_5_moe.model import Model


def _debugmodel_naive() -> Model:
    """Return debugmodel with fla_backend='torch_naive' for CPU testing."""
    cfg = qwen35_moe_configs["debugmodel"]
    layer_cfg = dataclasses.replace(
        cfg.layer,
        deltanet=dataclasses.replace(cfg.layer.deltanet, fla_backend="torch_naive"),
    )
    new_cfg = dataclasses.replace(cfg, layer=layer_cfg)
    return new_cfg.build()


class TestQwen35MoEConstruction(unittest.TestCase):
    def test_build_debugmodel(self):
        model = _debugmodel_naive()
        self.assertIsInstance(model, Model)
        self.assertEqual(len(model.layers), 8)

    def test_layer_types(self):
        """Layers 3, 7 (0-indexed) should be full_attention; others linear_attention."""
        model = _debugmodel_naive()
        for i, layer in enumerate(model.layers.values()):
            expected = "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
            self.assertEqual(
                layer.layer_type,
                expected,
                f"Layer {i}: expected {expected}, got {layer.layer_type}",
            )

    def test_all_flavors_construct(self):
        """All model flavors should construct their Config without error."""
        for flavor, cfg in qwen35_moe_configs.items():
            self.assertIsInstance(cfg, Model.Config, f"Flavor '{flavor}' Config failed")

    def test_model_registry(self):
        spec = model_registry("debugmodel")
        self.assertEqual(spec.name, "qwen3_5_moe")
        self.assertEqual(spec.flavor, "debugmodel")

    def test_build_helper(self):
        model = _debugmodel_naive()
        self.assertIsInstance(model, Model)

    def test_build_unknown_flavor(self):
        with self.assertRaises(ValueError):
            build("nonexistent_flavor")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for MoE forward pass")
class TestQwen35MoEForward(unittest.TestCase):
    def setUp(self):
        self.model = _debugmodel_naive().cuda()
        self.model.init_weights()
        self.model.eval()

    def test_forward_shape(self):
        tokens = torch.randint(0, 2048, (1, 32), device="cuda")
        with torch.no_grad():
            out = self.model(tokens, None)
        self.assertEqual(out.shape, (1, 32, 2048))

    def test_forward_batch(self):
        tokens = torch.randint(0, 2048, (2, 16), device="cuda")
        with torch.no_grad():
            out = self.model(tokens, None)
        self.assertEqual(out.shape, (2, 16, 2048))

    def test_forward_deterministic(self):
        tokens = torch.randint(0, 2048, (1, 32), device="cuda")
        with torch.no_grad():
            out1 = self.model(tokens, None)
            out2 = self.model(tokens, None)
        torch.testing.assert_close(out1, out2)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for MoE forward pass")
class TestQwen35MoEStateDictRoundtrip(unittest.TestCase):
    def test_state_dict_roundtrip(self):
        model = _debugmodel_naive().cuda()
        model.init_weights()
        model.eval()

        tokens = torch.randint(0, 2048, (1, 16), device="cuda")
        with torch.no_grad():
            out_before = model(tokens, None)

        # Save and reload state dict
        sd = model.state_dict()
        model2 = _debugmodel_naive().cuda()
        model2.load_state_dict(sd)
        model2.eval()

        with torch.no_grad():
            out_after = model2(tokens, None)

        torch.testing.assert_close(out_before, out_after)


if __name__ == "__main__":
    unittest.main()
