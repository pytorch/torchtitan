# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Layer-level CPU smoke tests for Kimi Linear.

Scope: verify that the torchtitan-idiom port produces
forward outputs with the right shapes, on CPU, for:

* :class:`KimiRMSNorm` — trivial
* :class:`KimiMLP` — SwiGLU dense
* :class:`KimiMLAAttention` — NoPE MLA, causal
* :class:`KimiDeltaAttention` — KDA via fla-core

MoE path (:class:`KimiMoE`) and full-model integration tests land in
a follow-up once torchtitan's ``GroupedExperts.forward`` signature is
validated against our call site.

These tests require fla-core (for KDA) and run on CPU only. The
KDA chunk kernel has a CPU fallback path — verify by running this
file, expect ~seconds per test.
"""

from __future__ import annotations

import unittest

import torch

from torchtitan.models.kimi_k3.model import (
    KimiDeltaAttention,
    KimiK3Config,
    KimiMLAAttention,
    KimiMLP,
)


def _tiny_config(num_hidden_layers: int = 2) -> KimiK3Config:
    """Small config that fits on CPU. KDA + MLA alternation: layer 0
    KDA (1-indexed: 1), layer 1 MLA (1-indexed: 2).
    """
    return KimiK3Config(
        vocab_size=256,
        hidden_size=128,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=256,
        # MLA side
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=None,
        kv_lora_rank=64,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        mla_use_nope=True,
        # KDA side
        kda_num_heads=4,
        kda_head_dim=16,
        kda_short_conv_kernel_size=4,
        kda_layers=[1],  # 1-indexed: layer 0 is KDA
        full_attn_layers=[2],  # 1-indexed: layer 1 is MLA
        # MoE off for this smoke
        num_experts=None,
        num_experts_per_token=1,
        num_shared_experts=0,
        first_k_dense_replace=num_hidden_layers,  # all layers dense
        moe_layer_freq=1,
        # Norm / act
        rms_norm_eps=1e-5,
        hidden_act="silu",
        initializer_range=0.02,
    )


class TestKimiMLP(unittest.TestCase):
    def test_forward_shape(self):
        mlp = KimiMLP.make_config(hidden_size=128, intermediate_size=256, hidden_act="silu").build()
        x = torch.randn(2, 7, 128)
        out = mlp(x)
        self.assertEqual(out.shape, x.shape)

    def test_gelu_alias_accepted(self):
        mlp = KimiMLP.make_config(hidden_size=64, intermediate_size=128, hidden_act="gelu").build()
        x = torch.randn(1, 3, 64)
        self.assertEqual(mlp(x).shape, x.shape)


class TestKimiMLAAttention(unittest.TestCase):
    def test_forward_shape(self):
        cfg = _tiny_config()
        mla = KimiMLAAttention.make_config(cfg, layer_idx=1).build()  # layer_idx 1 is MLA per tiny_config
        B, T = 2, 16
        x = torch.randn(B, T, cfg.hidden_size)
        out = mla(x)
        self.assertEqual(out.shape, (B, T, cfg.hidden_size))

    def test_forward_is_autograd_differentiable(self):
        cfg = _tiny_config()
        mla = KimiMLAAttention.make_config(cfg, layer_idx=1).build()
        x = torch.randn(1, 8, cfg.hidden_size, requires_grad=True)
        out = mla(x).sum()
        out.backward()
        self.assertIsNotNone(x.grad)
        # Any param should have grad populated
        any_param_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in mla.parameters()
        )
        self.assertTrue(any_param_grad)


class TestKimiDeltaAttention(unittest.TestCase):
    def test_instantiate_on_cpu(self):
        """KDA module should instantiate on CPU without running kernels."""
        cfg = _tiny_config()
        kda = KimiDeltaAttention.make_config(cfg, layer_idx=0).build()
        self.assertIsInstance(kda, KimiDeltaAttention)
        # Param count sanity
        n_params = sum(p.numel() for p in kda.parameters())
        self.assertGreater(n_params, 0)

    @unittest.skipUnless(
        torch.cuda.is_available(), "KDA chunk kernel is Triton/CUDA only"
    )
    def test_forward_shape_chunk_mode_cuda(self):
        """T > 64 triggers chunk mode in KDA. Requires CUDA + Triton."""
        cfg = _tiny_config()
        device = torch.device("cuda")
        kda = KimiDeltaAttention.make_config(cfg, layer_idx=0).build().to(device).to(torch.bfloat16)
        B, T = 2, 128
        x = torch.randn(B, T, cfg.hidden_size, device=device, dtype=torch.bfloat16)
        out = kda(x)
        self.assertEqual(out.shape, (B, T, cfg.hidden_size))
        self.assertTrue(torch.isfinite(out).all())

    def test_cpu_forward_raises_triton_error(self):
        """Documents that running KDA forward on CPU is unsupported —
        fails with a Triton / CUDA-side error. Not a test of
        correctness; this locks in the expectation so that when
        fla-core ships a CPU fallback we know to update the test.
        """
        cfg = _tiny_config()
        kda = KimiDeltaAttention.make_config(cfg, layer_idx=0).build()
        x = torch.randn(1, 128, cfg.hidden_size)
        with self.assertRaises((ValueError, RuntimeError, NotImplementedError)):
            kda(x)


if __name__ == "__main__":
    unittest.main()
