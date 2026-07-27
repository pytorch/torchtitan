# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the TorchTitan-name -> renderers mapping.

Guards that qwen3_5 resolves to the dedicated Qwen3.5 renderer rather than the
"auto" fallback (which raises for this VLM family on a local checkpoint path).
"""

import unittest

from torchtitan.experiments.rl.renderer import _RENDERER_BY_MODEL


class TestRendererMapping(unittest.TestCase):
    def test_qwen3_5_maps_to_dedicated_renderer(self):
        self.assertEqual(_RENDERER_BY_MODEL["qwen3_5"], "qwen3.5")

    def test_qwen3_5_renderer_config_resolves(self):
        try:
            from renderers import config_from_name
        except ImportError as e:
            self.skipTest(f"renderers unavailable: {e}")

        cfg = config_from_name("qwen3.5")
        self.assertEqual(type(cfg).__name__, "Qwen35RendererConfig")
        self.assertIn("enable_thinking", type(cfg).model_fields)


if __name__ == "__main__":
    unittest.main()
