# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Wiring tests for the self-contained MXFP8 fused-SwiGLU overrides.

The overrides must produce the MXFP8 fused modules with the right config
fields and (for the grouped path) the padded token dispatcher; the factories
must fail loud on non-stock configs. Everything here is a config-tree
transform plus a meta-device build, so it runs without a GPU: the factories'
SM100 gate is patched out (hardware is irrelevant to the transforms under
test). Numerics of the underlying composites are validated on SM100 hardware
in NVIDIA-internal CI.
"""

import unittest
from unittest import mock

import torch

from torchtitan.config.override import apply_overrides, OverrideConfig
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.token_dispatcher import TorchAOTokenDispatcher
from torchtitan.models.deepseek_v3 import model_registry as deepseek_v3_model_registry
from torchtitan.models.llama3 import model_registry as llama3_model_registry
from torchtitan.overrides.fused_swiglu import FusedSwiGLU

try:
    from torchtitan.overrides.mxfp8_fused_swiglu import (
        mxfp8_fused_swiglu,
        MXFP8FusedGroupedExperts,
        MXFP8FusedSwiGLU,
    )
except ImportError as e:  # torchao (or a transitive dep) not installed
    raise unittest.SkipTest(
        f"torchao is required for the MXFP8 SwiGLU overrides: {e}"
    ) from e

_DENSE_OVERRIDE = "torchtitan.overrides.mxfp8_fused_swiglu.mxfp8_fused_swiglu"
_GROUPED_OVERRIDE = (
    "torchtitan.overrides.mxfp8_fused_swiglu.mxfp8_fused_grouped_experts"
)


class TestMXFP8FusedSwiGLUOverride(unittest.TestCase):
    def setUp(self):
        # The factories gate on SM100 at config-application time; hardware is
        # irrelevant to the config-tree transforms under test.
        patcher = mock.patch(
            "torchtitan.overrides.mxfp8_fused_swiglu.has_cuda_capability",
            lambda *args: True,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_dense_override_builds_mxfp8_fused_swiglu(self):
        model_config = llama3_model_registry("debugmodel").model
        apply_overrides(OverrideConfig(imports=[_DENSE_OVERRIDE]), model_config)
        with torch.device("meta"):
            model = model_config.build()
        fused = [m for m in model.modules() if isinstance(m, MXFP8FusedSwiGLU)]
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
        fused = [m for m in model.modules() if isinstance(m, MXFP8FusedSwiGLU)]
        self.assertTrue(fused)
        self.assertFalse(any(m.fuse_activation for m in fused))

    def test_grouped_override_builds_experts_and_padded_dispatcher(self):
        model_config = deepseek_v3_model_registry("debugmodel").model
        apply_overrides(OverrideConfig(imports=[_GROUPED_OVERRIDE]), model_config)
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
        fused = [m for m in model.modules() if isinstance(m, MXFP8FusedGroupedExperts)]
        self.assertTrue(fused)
        self.assertTrue(all(m.fuse_activation for m in fused))

    def test_dense_factory_raises_on_non_stock_ffn(self):
        # A FeedForward.Config SUBCLASS (already fused) must raise, not no-op.
        gate = Linear.Config(in_features=128, out_features=256)
        cfg = FusedSwiGLU.Config(
            w1=gate,
            w2=Linear.Config(in_features=256, out_features=128),
            w3=gate,
        )
        with self.assertRaisesRegex(ValueError, "stock FeedForward.Config"):
            mxfp8_fused_swiglu(cfg)

    def test_dense_factory_raises_on_converted_projection(self):
        # The composite quantizes every GEMM itself; combining with a linear
        # quantization converter on the same module must raise.
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
            mxfp8_fused_swiglu(cfg)


if __name__ == "__main__":
    unittest.main()
