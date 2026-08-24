# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Wiring tests for the self-contained MXFP8 fused-MLP overrides.

Everything here is a config-tree transform plus a meta-device build, so it
runs without a GPU: the factories' SM100 gate is patched out. Numerics of the
underlying composites are validated on SM100 hardware in NVIDIA-internal CI.
"""

import unittest
from unittest import mock

import torch

from torchtitan.config.override import apply_overrides, OverrideConfig
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.token_dispatcher import TorchAOTokenDispatcher
from torchtitan.models.deepseek_v3 import model_registry as deepseek_v3_model_registry
from torchtitan.models.llama3 import model_registry as llama3_model_registry
from torchtitan.overrides.fused_swiglu import FusedSwiGLU

try:
    from torchtitan.overrides.mxfp8_fused_mlp import (
        mxfp8_fused_mlp,
        MXFP8FusedGroupedMLP,
        MXFP8FusedMLP,
    )
except ImportError as e:  # torchao (or a transitive dep) not installed
    raise unittest.SkipTest(
        f"torchao is required for the MXFP8 fused-MLP overrides: {e}"
    ) from e

_DENSE_OVERRIDE = "torchtitan.overrides.mxfp8_fused_mlp.mxfp8_fused_mlp"
_GROUPED_OVERRIDE = "torchtitan.overrides.mxfp8_fused_mlp.mxfp8_fused_grouped_mlp"


class TestMXFP8FusedMLPOverride(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch(
            "torchtitan.overrides.mxfp8_fused_mlp.has_cuda_capability",
            lambda *args: True,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_dense_override_builds_mxfp8_fused_mlp(self):
        model_config = llama3_model_registry("debugmodel").model
        apply_overrides(OverrideConfig(imports=[_DENSE_OVERRIDE]), model_config)
        with torch.device("meta"):
            model = model_config.build()
        fused = [m for m in model.modules() if isinstance(m, MXFP8FusedMLP)]
        self.assertTrue(fused)
        self.assertTrue(all(m.fuse_activation for m in fused))

    def test_dense_override_kwargs_configure_the_composite(self):
        model_config = llama3_model_registry("debugmodel").model
        apply_overrides(
            OverrideConfig(imports=[(_DENSE_OVERRIDE, {"fuse_activation": False})]),
            model_config,
        )
        with torch.device("meta"):
            model = model_config.build()
        fused = [m for m in model.modules() if isinstance(m, MXFP8FusedMLP)]
        self.assertTrue(fused)
        self.assertFalse(any(m.fuse_activation for m in fused))

    def _grouped_model_config(self):
        model_config = deepseek_v3_model_registry("debugmodel").model
        apply_overrides(OverrideConfig(imports=[_GROUPED_OVERRIDE]), model_config)
        return model_config

    def _grouped_experts_config(self, model_config):
        nodes = list(model_config.traverse(MXFP8FusedGroupedMLP.Config))
        self.assertTrue(nodes)
        return nodes[0][1]

    def test_grouped_override_builds_experts_and_padded_dispatcher(self):
        model_config = self._grouped_model_config()
        pads = [
            dispatcher_cfg.pad_multiple
            for _fqn, dispatcher_cfg, _parent, _attr in model_config.traverse(
                TorchAOTokenDispatcher.Config
            )
        ]
        self.assertTrue(pads)
        self.assertTrue(all(pad == 128 for pad in pads))
        with torch.device("meta"):
            model = model_config.build()
        fused = [m for m in model.modules() if isinstance(m, MXFP8FusedGroupedMLP)]
        self.assertTrue(fused)
        self.assertTrue(all(type(m) is MXFP8FusedGroupedMLP for m in fused))
        self.assertTrue(all(m.fuse_activation for m in fused))

    def test_grouped_forward_validates_and_applies_the_function(self):
        cfg = self._grouped_experts_config(self._grouped_model_config())
        module = cfg.build()
        num_tokens = torch.zeros(cfg.num_experts, dtype=torch.int64)
        num_tokens[0] = 2
        x = torch.randn(2, cfg.dim)
        sentinel = torch.zeros(2, cfg.dim, dtype=torch.bfloat16)
        with mock.patch(
            "torchtitan.overrides.mxfp8_fused_mlp._validate_grouped_inputs"
        ) as validate, mock.patch(
            "torchtitan.overrides.mxfp8_fused_mlp._MXFP8GroupedMLP.apply",
            return_value=sentinel,
        ) as function:
            out = module(x, num_tokens)
        function.assert_called_once()
        args = function.call_args.args
        self.assertEqual(args[0].dtype, torch.bfloat16)
        self.assertEqual(
            tuple(args[1].shape), (cfg.num_experts, cfg.hidden_dim, 2, cfg.dim)
        )
        self.assertEqual(
            tuple(args[2].shape), (cfg.num_experts, cfg.hidden_dim, cfg.dim)
        )
        self.assertEqual(args[3].dtype, torch.int32)
        self.assertEqual(args[3].tolist(), torch.cumsum(num_tokens, dim=0).tolist())
        self.assertEqual(args[4], True)
        validate.assert_called_once_with(args[0], args[1], args[2], args[3])
        self.assertEqual(out.dtype, x.dtype)

    def test_grouped_checkpoint_keys_and_param_shapes_unchanged(self):
        stock_nodes = list(
            deepseek_v3_model_registry("debugmodel").model.traverse(
                GroupedExperts.Config
            )
        )
        self.assertTrue(stock_nodes)
        fused_cfg = self._grouped_experts_config(self._grouped_model_config())
        with torch.device("meta"):
            stock = stock_nodes[0][1].build()
            fused = fused_cfg.build()
        self.assertEqual(set(fused.state_dict().keys()), set(stock.state_dict().keys()))
        self.assertEqual(
            tuple(fused.w13.shape),
            (fused_cfg.num_experts, fused_cfg.hidden_dim, 2, fused_cfg.dim),
        )

    def test_dense_factory_raises_on_non_stock_ffn(self):
        # A FeedForward.Config SUBCLASS (already fused) must raise, not no-op.
        gate = Linear.Config(in_features=128, out_features=256)
        cfg = FusedSwiGLU.Config(
            w1=gate,
            w2=Linear.Config(in_features=256, out_features=128),
            w3=gate,
        )
        with self.assertRaisesRegex(ValueError, "stock FeedForward.Config"):
            mxfp8_fused_mlp(cfg)

    def test_dense_factory_raises_on_converted_projection(self):
        from torchtitan.components.quantization.mx import MXFP8Linear

        if MXFP8Linear is None:
            self.skipTest("torchao not installed")
        gate = Linear.Config(in_features=128, out_features=256)
        cfg = FeedForward.Config(
            w1=gate,
            w2=MXFP8Linear.Config(in_features=256, out_features=128),
            w3=gate,
        )
        with self.assertRaisesRegex(ValueError, "quantization converter"):
            mxfp8_fused_mlp(cfg)


if __name__ == "__main__":
    unittest.main()
