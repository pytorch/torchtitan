# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The dynamo carve-out for fla's triton kernels.

These assert that the carve-out is *applied*, which is a separate question from
whether it is correct. ``torch.compiler.disable`` returns a wrapper rather than
marking a function in place, so a version of this code that called it and dropped
the result compiled cleanly, ran, and protected nothing.
"""

from __future__ import annotations

import unittest


class TestFlaCarveOut(unittest.TestCase):
    def setUp(self):
        from torchtitan.models.kimi_k3 import (
            attn_res as attn_res_mod,
            attn_res_model as attn_res_model_mod,
            model as model_mod,
            parallelize as pz,
        )

        self.pz = pz
        self.model_mod = model_mod
        self.op_names = ("chunk_kda", "fused_recurrent_kda", "fused_kda_gate")
        # The carve-out is global state, so anything it touches is restored.
        self._saved = {n: getattr(model_mod, n) for n in self.op_names}
        self._saved_flag = pz._fla_dynamo_carveout_done
        self._saved_kda_forward = model_mod.KimiDeltaAttention.forward
        self._saved_block = (
            attn_res_mod.block_attn_res,
            attn_res_model_mod.block_attn_res,
        )
        self._attn_res_mod = attn_res_mod
        self._attn_res_model_mod = attn_res_model_mod
        pz._fla_dynamo_carveout_done = False

    def tearDown(self):
        for name, fn in self._saved.items():
            setattr(self.model_mod, name, fn)
        self.model_mod.KimiDeltaAttention.forward = self._saved_kda_forward
        self._attn_res_mod.block_attn_res = self._saved_block[0]
        self._attn_res_model_mod.block_attn_res = self._saved_block[1]
        self.pz._fla_dynamo_carveout_done = self._saved_flag

    def test_disable_returns_a_wrapper_rather_than_marking_in_place(self):
        """The exact property the broken version assumed the other way."""
        import torch

        def f(x):
            return x

        self.assertIsNot(torch.compiler.disable(f, recursive=True), f)

    def test_the_ops_the_model_calls_are_rebound(self):
        import fla.ops.kda

        self.pz._disable_dynamo_on_fla_ops()
        for name in self.op_names:
            with self.subTest(op=name):
                patched = getattr(self.model_mod, name)
                self.assertIsNot(patched, self._saved[name], f"{name} not rebound")
        # Rebinding fla's own module would not help: model.py bound these names
        # at import time, so the call site reads its own global.
        self.assertIsNot(self.model_mod.chunk_kda, fla.ops.kda.chunk_kda)

    def test_block_attn_res_is_rebound_in_both_modules(self):
        self.pz._disable_dynamo_on_fla_ops()
        self.assertIs(
            self._attn_res_mod.block_attn_res,
            self._attn_res_model_mod.block_attn_res,
        )
        self.assertIsNot(self._attn_res_mod.block_attn_res, self._saved_block[0])

    def test_applying_it_twice_does_not_wrap_twice(self):
        self.pz._disable_dynamo_on_fla_ops()
        once = self.model_mod.chunk_kda
        once_forward = self.model_mod.KimiDeltaAttention.forward
        # _apply_compile_kimi_k3 runs per model part, so under PP this would
        # otherwise stack one wrapper per part.
        self.pz._disable_dynamo_on_fla_ops()
        self.assertIs(self.model_mod.chunk_kda, once)
        self.assertIs(self.model_mod.KimiDeltaAttention.forward, once_forward)


if __name__ == "__main__":
    unittest.main()
