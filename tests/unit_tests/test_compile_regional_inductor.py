# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import unittest

import torch

import torchtitan.distributed.compile as compile_mod
from torchtitan.distributed.compile import (
    _maybe_regional_inductor_backend,
    maybe_regional_inductor,
)
from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.common.linear import Linear


class TestRegionalInductorBackend(unittest.TestCase):
    """CPU tests for FlexAttention regional_inductor backend selection.

    These exercise only the backend-selection decision and the resulting
    annotation toggle; compilation is never run, so no GPU is required.
    """

    def setUp(self):
        self._flex = FlexAttention(FlexAttention.Config())
        self._dense = Linear.Config(in_features=4, out_features=4, bias=False).build()
        compile_mod._regional_inductor_enabled = False

    def tearDown(self):
        compile_mod._regional_inductor_enabled = False

    def test_inductor_backend_left_unchanged(self):
        backend = _maybe_regional_inductor_backend(self._flex, "inductor")
        self.assertEqual(backend, "inductor")
        self.assertFalse(compile_mod._regional_inductor_enabled)

    def test_aot_eager_with_flex_scoops(self):
        backend = _maybe_regional_inductor_backend(self._flex, "aot_eager")
        self.assertTrue(callable(backend))
        self.assertTrue(compile_mod._regional_inductor_enabled)

    def test_aot_eager_without_flex_left_unchanged(self):
        backend = _maybe_regional_inductor_backend(self._dense, "aot_eager")
        self.assertEqual(backend, "aot_eager")
        self.assertFalse(compile_mod._regional_inductor_enabled)

    def test_other_backend_with_flex_raises(self):
        with self.assertRaises(ValueError):
            _maybe_regional_inductor_backend(self._flex, "eager")
        self.assertFalse(compile_mod._regional_inductor_enabled)

    def test_region_annotation_follows_flag(self):
        # Disabled -> null context, so no annotation metadata is emitted.
        ctx = maybe_regional_inductor(FlexAttention.inductor_configs)
        self.assertIsInstance(ctx, contextlib.nullcontext)
        # Enabled -> a real annotation context manager.
        compile_mod._regional_inductor_enabled = True
        ctx = maybe_regional_inductor(FlexAttention.inductor_configs)
        self.assertNotIsInstance(ctx, contextlib.nullcontext)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestRegionalInductorCodegen(unittest.TestCase):
    """GPU test: the compiled program lowers FlexAttention to a triton kernel.

    Compiles flex under aot_eager and inspects the generated code. With the
    regional scoop the flex region produces a triton kernel; with plain
    aot_eager it decomposes to eager aten (no triton flex kernel).
    """

    def setUp(self):
        # Disable autotune so codegen is fast and deterministic; the regional
        # annotation reads inductor_configs, so this also speeds up the scoop.
        self._cfg_backup = dict(FlexAttention.inductor_configs)
        FlexAttention.inductor_configs["max_autotune"] = False
        FlexAttention.inductor_configs["coordinate_descent_tuning"] = False
        compile_mod._regional_inductor_enabled = False

    def tearDown(self):
        FlexAttention.inductor_configs.clear()
        FlexAttention.inductor_configs.update(self._cfg_backup)
        compile_mod._regional_inductor_enabled = False
        torch._dynamo.reset()

    @staticmethod
    def _triton_flex_kernel_count(backend) -> int:
        from torch._inductor.utils import run_fw_bw_and_get_code
        from torch.nn.attention.flex_attention import create_block_mask

        torch._dynamo.reset()
        attn = FlexAttention(FlexAttention.Config())
        bs, seq, heads, dim = 2, 256, 4, 64
        shape = (bs, seq, heads, dim)
        q, k, v = (
            torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            for _ in range(3)
        )
        block_mask = create_block_mask(
            lambda b, h, q_idx, kv_idx: q_idx >= kv_idx,
            B=None,
            H=None,
            Q_LEN=seq,
            KV_LEN=seq,
            device="cuda",
        )

        def fn(q, k, v):
            return (attn(q, k, v, attention_masks=block_mask) ** 2).sum()

        compiled = torch.compile(fn, backend=backend, fullgraph=True)
        _, codes = run_fw_bw_and_get_code(lambda: compiled(q, k, v))
        return sum(1 for c in codes if "triton" in c and "flex_attention" in c)

    def test_plain_aot_eager_decomposes_flex(self):
        # Flag stays False -> forward emits no annotation -> flex decomposes.
        self.assertEqual(self._triton_flex_kernel_count("aot_eager"), 0)

    def test_regional_scoop_lowers_flex_to_triton(self):
        backend = _maybe_regional_inductor_backend(
            FlexAttention(FlexAttention.Config()), "aot_eager"
        )
        # Forward + backward flex regions both lower to triton.
        self.assertGreaterEqual(self._triton_flex_kernel_count(backend), 1)


if __name__ == "__main__":
    unittest.main()
