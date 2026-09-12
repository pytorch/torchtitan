# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the fused-activation SwiGLU override.

The override replaces ``FeedForward.activation_fn`` while preserving the module
and its physical ``w13`` projection. These tests run on CPU unless marked CUDA.

``TestFusedSwiGLUDistGemmComposition`` covers stacking the override on
``tp_gemm_backend="dist_gemm"``, which must keep the TP overlap rather than
silently replacing the overlapping FFN with the plain fused one.
"""

import unittest
from dataclasses import dataclass

import torch

from torchtitan.models.common.activation import SiTUGLU
from torchtitan.models.common.dist_gemm import DistGEMMFeedForward
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.llama3 import llama3_configs
from torchtitan.models.llama3.model import Llama3Model
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.overrides.fused_swiglu import (
    dist_gemm_fused_swiglu,
    fused_swiglu,
    FusedSwiGLU,
)

_DIM = 16
_HIDDEN = 32


class _ConvertedLinear(Linear):
    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass


def _build_fused() -> FeedForward:
    fused = fused_swiglu(_feed_forward_config()).build()
    with torch.no_grad():
        fused.w13.weight.copy_(torch.randn(2 * _HIDDEN, _DIM))
        fused.w2.weight.copy_(torch.randn(_DIM, _HIDDEN))
    return fused


def _logical_w13(feed_forward: FeedForward) -> torch.Tensor:
    return feed_forward.w13.weight.unflatten(0, (_HIDDEN, 2))


def _feed_forward_config() -> FeedForward.Config:
    return FeedForward.Config(
        w13=Linear.Config(in_features=_DIM, out_features=2 * _HIDDEN),
        w2=Linear.Config(in_features=_HIDDEN, out_features=_DIM),
    )


def _build_native() -> FeedForward:
    native = _feed_forward_config().build()
    with torch.no_grad():
        for p in native.parameters():
            p.copy_(torch.randn_like(p))
    return native


class TestFusedSwiGLU(unittest.TestCase):
    def test_replaces_only_activation(self):
        fused = fused_swiglu(_feed_forward_config())

        self.assertIs(type(fused), FeedForward.Config)
        self.assertIsInstance(fused.activation_fn, FusedSwiGLU.Config)

    def test_rejects_non_swiglu_activation(self):
        config = _feed_forward_config()
        config.activation_fn = SiTUGLU.Config()

        with self.assertRaisesRegex(ValueError, "requires the default SwiGLU"):
            fused_swiglu(config)

    def test_gate_up_projection_is_linear(self):
        fused = _build_fused()
        self.assertIsInstance(fused.w13, Linear)
        self.assertEqual(tuple(fused.w13.weight.shape), (2 * _HIDDEN, _DIM))
        self.assertEqual(
            {name for name, _ in fused.named_parameters()},
            {"w13.weight", "w2.weight"},
        )

    def test_preserves_converted_linear_config(self):
        cfg = FeedForward.Config(
            w13=_ConvertedLinear.Config(in_features=_DIM, out_features=2 * _HIDDEN),
            w2=Linear.Config(in_features=_HIDDEN, out_features=_DIM),
        )

        fused = fused_swiglu(cfg)

        self.assertIsInstance(fused.w13, _ConvertedLinear.Config)
        self.assertIsInstance(fused.build().w13, _ConvertedLinear)

    def test_saves_physical_layout(self):
        fused = _build_fused()
        sd = fused.state_dict()
        self.assertEqual(set(sd), {"w13.weight", "w2.weight"})
        self.assertTrue(torch.equal(sd["w13.weight"], fused.w13.weight))

    @unittest.skipUnless(torch.cuda.is_available(), "silu_and_mul op is CUDA-only")
    def test_triton_checkpoint_loads_into_native(self):
        fused = _build_fused().cuda()
        native = _build_native().cuda()
        native.load_state_dict(fused.state_dict())
        self.assertTrue(
            torch.equal(_logical_w13(native)[:, 0], _logical_w13(fused)[:, 0])
        )
        self.assertTrue(
            torch.equal(_logical_w13(native)[:, 1], _logical_w13(fused)[:, 1])
        )
        self.assertTrue(torch.equal(native.w2.weight, fused.w2.weight))
        x = torch.randn(4, _DIM, device="cuda")
        self.assertTrue(torch.allclose(fused(x), native(x), atol=1e-4, rtol=1e-5))

    @unittest.skipUnless(torch.cuda.is_available(), "silu_and_mul op is CUDA-only")
    def test_native_checkpoint_loads_into_triton(self):
        native = _build_native().cuda()
        fused = _build_fused().cuda()
        fused.load_state_dict(native.state_dict())
        self.assertTrue(
            torch.equal(_logical_w13(fused)[:, 0], _logical_w13(native)[:, 0])
        )
        self.assertTrue(
            torch.equal(_logical_w13(fused)[:, 1], _logical_w13(native)[:, 1])
        )
        self.assertTrue(torch.equal(fused.w2.weight, native.w2.weight))
        x = torch.randn(4, _DIM, device="cuda")
        self.assertTrue(torch.allclose(fused(x), native(x), atol=1e-4, rtol=1e-5))

    def test_fused_roundtrip(self):
        """fused -> save -> load into a fresh fused preserves w13 exactly."""
        src = _build_fused()
        dst = _build_fused()
        dst.load_state_dict(src.state_dict())
        self.assertTrue(torch.equal(dst.w13.weight, src.w13.weight))
        self.assertTrue(torch.equal(dst.w2.weight, src.w2.weight))

    def test_strict_load_reports_missing(self):
        """strict load still flags a genuinely incomplete checkpoint."""
        fused = _build_fused()
        with self.assertRaises(RuntimeError):
            fused.load_state_dict({"w2.weight": fused.w2.weight.detach().clone()})


def _dist_gemm_ffn_config(**kwargs):
    from torchtitan.models.common.config_utils import make_ffn_config

    init = {"weight": torch.nn.init.zeros_}
    return make_ffn_config(
        dim=_DIM, hidden_dim=_HIDDEN, w1_param_init=init, w2w3_param_init=init, **kwargs
    )


class TestFusedSwiGLUDistGemmComposition(unittest.TestCase):
    """The dist-GEMM override must keep TP overlap.

    Asserts on the *built module* rather than the config type: the failure mode is
    getting a working-but-unoverlapped FFN, which a config-type check on the
    override's declared return type would not catch.
    """

    def test_dist_gemm_config_keeps_overlap(self):
        self.assertIsInstance(
            fused_swiglu(_dist_gemm_ffn_config()).build(), FeedForward
        )
        fused = dist_gemm_fused_swiglu(
            _dist_gemm_ffn_config(tp_gemm_backend="dist_gemm")
        ).build()
        self.assertIsInstance(fused, DistGEMMFeedForward)
        self.assertIsInstance(fused.activation_fn, FusedSwiGLU)

    def test_overlapping_variant_keeps_w13_checkpoint_layout(self):
        fused = dist_gemm_fused_swiglu(
            _dist_gemm_ffn_config(tp_gemm_backend="dist_gemm")
        ).build()
        with torch.no_grad():
            fused.w13.weight.copy_(torch.randn(2 * _HIDDEN, _DIM))
        state_dict = fused.state_dict()
        self.assertEqual(set(state_dict), {"w13.weight", "w2.weight"})
        torch.testing.assert_close(state_dict["w13.weight"], fused.w13.weight)

        reloaded = dist_gemm_fused_swiglu(
            _dist_gemm_ffn_config(tp_gemm_backend="dist_gemm")
        ).build()
        reloaded.load_state_dict(state_dict)
        torch.testing.assert_close(reloaded.w13.weight, fused.w13.weight)


class TestFusedSwiGLUHFAdapter(unittest.TestCase):
    def test_hf_adapter_roundtrip(self):
        """A fused-SwiGLU model interoperates with the HF state-dict adapter.

        The adapter maps HF mlp.gate_proj/up_proj to the physical
        feed_forward.w13 parameter.
        """
        build_config, max_context_length = llama3_configs["debugmodel"]
        config = build_config(attn_backend="flex", seq_len=max_context_length)
        # Apply the fused override factory directly, independent of the global
        # override registry (which other tests may clear).
        for layer in config.layers:
            layer.feed_forward = fused_swiglu(layer.feed_forward)
        model = Llama3Model(config)
        model.init_states()
        ffn = model.get_submodule("layers.0.feed_forward")
        self.assertIsInstance(ffn, FeedForward)
        self.assertIsInstance(ffn.activation_fn, FusedSwiGLU)

        sd = model.state_dict()
        self.assertTrue(any(k.endswith("feed_forward.w13.weight") for k in sd))

        adapter = Llama3StateDictAdapter(config, hf_assets_path=None)
        hf_sd = adapter.to_hf(sd)
        self.assertIn("model.layers.0.mlp.gate_proj.weight", hf_sd)
        self.assertIn("model.layers.0.mlp.up_proj.weight", hf_sd)

        orig_w13 = ffn.w13.weight.detach().clone()
        restored = adapter.from_hf(hf_sd)
        self.assertIn("layers.0.feed_forward.w13.weight", restored)
        model.load_state_dict(restored, strict=False)
        self.assertTrue(torch.equal(ffn.w13.weight, orig_w13))


if __name__ == "__main__":
    unittest.main()
