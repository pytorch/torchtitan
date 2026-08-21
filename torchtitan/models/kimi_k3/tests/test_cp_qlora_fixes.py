# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the 2026-07-24 CP/QLoRA fixes.

Covers: the meta-first packed-MXFP4 LoRA layout (layout registration
must exactly match on-device quantization output shapes/ctx), and the
frozen-base-LoRA dtype alignment in the AttnRes read path (fp32 masters
meeting a bf16 stream must not crash nor promote the stream).
"""

import os
import unittest

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate

from torchtitan.models.kimi_k3.attn_res import (
    AttnResProjection,
    block_attn_res,
)
from torchtitan.models.kimi_k3.lora import KimiLoRALinear


class TestPackedMXFP4MetaLayout(unittest.TestCase):
    def test_meta_layout_matches_on_device_quantize(self):
        out_f, in_f = 8, 64
        with torch.device("meta"):
            meta_lin = nn.Linear(in_f, out_f, bias=False)
        meta_mod = KimiLoRALinear(meta_lin, rank=4, alpha=8.0, quantize_base="mxfp4")

        real_lin = nn.Linear(in_f, out_f, bias=False)
        real_mod = KimiLoRALinear(real_lin, rank=4, alpha=8.0, quantize_base="mxfp4")

        # base.weight dropped in both flows
        self.assertNotIn("weight", meta_mod.base._parameters)
        self.assertNotIn("weight", real_mod.base._parameters)
        # packed layout identical to the on-device quantization output
        self.assertEqual(meta_mod.base_qdata.shape, real_mod.base_qdata.shape)
        self.assertEqual(meta_mod.base_qdata.dtype, real_mod.base_qdata.dtype)
        self.assertEqual(meta_mod.base_scale.shape, real_mod.base_scale.shape)
        self.assertEqual(meta_mod.base_scale.dtype, real_mod.base_scale.dtype)
        # flatten ctx carries no shape/data -> must be reproducible on meta
        self.assertEqual(meta_mod._mx_ctx, real_mod._mx_ctx)
        self.assertEqual(meta_mod._mx_scale_dtype, real_mod._mx_scale_dtype)

    def test_non_alignable_dim_stays_bf16(self):
        lin = nn.Linear(30, 8, bias=False)  # 30 % 32 != 0
        mod = KimiLoRALinear(lin, rank=2, alpha=4.0, quantize_base="mxfp4")
        self.assertIsNone(mod._quantize_base)
        self.assertIn("weight", mod.base._parameters)


class TestAttnResDtypeAlignment(unittest.TestCase):
    def test_fp32_masters_bf16_stream(self):
        d = 32
        proj = AttnResProjection(AttnResProjection.Config(dim=d))
        norm = nn.RMSNorm(d)
        # fp32 masters (frozen-base LoRA keeps trainable AttnRes params
        # fp32), bf16 stream:
        blocks = [torch.randn(2, 4, d, dtype=torch.bfloat16) for _ in range(2)]
        partial = torch.randn(2, 4, d, dtype=torch.bfloat16)
        h = block_attn_res(blocks, partial, proj, norm)
        # no crash, and the stream dtype is preserved (no fp32 leak)
        self.assertEqual(h.dtype, torch.bfloat16)

    def test_uniform_dtype_unchanged(self):
        d = 32
        proj = AttnResProjection(AttnResProjection.Config(dim=d))
        norm = nn.RMSNorm(d)
        blocks = [torch.randn(2, 4, d) for _ in range(2)]
        partial = torch.randn(2, 4, d)
        h = block_attn_res(blocks, partial, proj, norm)
        self.assertEqual(h.dtype, torch.float32)


class TestPackedMXFP4NoParallelInput(unittest.TestCase):
    """Packed base under a NoParallel descent (MoE shared experts).

    NoParallel's prepare_input wraps the plain input into a DTensor while
    the packed base dequantizes to a plain tensor, so the base matmul saw
    mixed Tensor/DTensor operands and raised. Only the Colwise/Rowwise
    styles set ``_tp_style``; shared experts never do.
    """

    def setUp(self):
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")
        self._owns_pg = not dist.is_initialized()
        if self._owns_pg:
            dist.init_process_group("gloo", rank=0, world_size=1)
        self.mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))

    def tearDown(self):
        if self._owns_pg and dist.is_initialized():
            dist.destroy_process_group()

    def test_dtensor_input_against_packed_base(self):
        torch.manual_seed(0)
        mod = KimiLoRALinear(
            nn.Linear(64, 16, bias=False), rank=4, alpha=8.0, quantize_base="mxfp4"
        )
        self.assertEqual(mod._quantize_base, "mxfp4")
        self.assertIsNone(getattr(mod, "_tp_style", None))

        x = torch.randn(2, 3, 64)
        y_plain = mod(x)
        y_dt = mod(DTensor.from_local(x, self.mesh, [Replicate()], run_check=False))
        self.assertIsInstance(y_dt, DTensor)
        torch.testing.assert_close(y_dt.full_tensor(), y_plain)


class TestLoRAAdapterDtypeAlignment(unittest.TestCase):
    def test_fp32_adapters_bf16_input(self):
        lin = nn.Linear(32, 16, bias=False)
        mod = KimiLoRALinear(lin, rank=4, alpha=8.0)
        # emulate the frozen-base cast: base bf16, adapters left fp32
        mod.base.weight.data = mod.base.weight.data.to(torch.bfloat16)
        x = torch.randn(2, 3, 32, dtype=torch.bfloat16)
        y = mod(x)
        self.assertEqual(y.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
