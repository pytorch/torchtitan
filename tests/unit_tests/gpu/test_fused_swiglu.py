# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checkpoint interop tests for the fused-activation SwiGLU override.

The override replaces ``FeedForward.activation_fn`` while preserving the module
and its logical ``w1.weight`` / ``w3.weight`` checkpoint layout. These tests run
on CPU unless marked CUDA.

``TestFusedSwiGLUDistGemmComposition`` covers stacking the override on the
async tensor-parallel transform, which must preserve the TP overlap.
"""

import unittest
from dataclasses import dataclass

import torch

from torchtitan.config.transform import AsyncTensorParallelTransform
from torchtitan.models.common.activation import ActivationFn, SiTUGLU
from torchtitan.models.common.dist_gemm import (
    AsyncAllGatherLinear,
    AsyncLinearReduceScatter,
)
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.llama3 import llama3_configs
from torchtitan.models.llama3.model import Llama3Model
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.overrides.fused_swiglu import _fused_silu_and_mul, fused_swiglu

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
        w1=Linear.Config(in_features=_DIM, out_features=_HIDDEN),
        w2=Linear.Config(in_features=_HIDDEN, out_features=_DIM),
        w3=Linear.Config(in_features=_DIM, out_features=_HIDDEN),
    )


def _build_native() -> FeedForward:
    native = _feed_forward_config().build()
    with torch.no_grad():
        for p in native.parameters():
            p.copy_(torch.randn_like(p))
    return native


class TestFusedSwiGLUCheckpointInterop(unittest.TestCase):
    def test_replaces_only_activation(self):
        fused = fused_swiglu(_feed_forward_config())

        self.assertIs(type(fused), FeedForward.Config)
        self.assertIs(fused.activation_fn.fn, _fused_silu_and_mul)

    def test_rejects_non_swiglu_activation(self):
        config = _feed_forward_config()
        config.activation_fn = ActivationFn.Config(
            fn=SiTUGLU()  # pyrefly: ignore[bad-argument-type]
        )

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
            w1=_ConvertedLinear.Config(in_features=_DIM, out_features=_HIDDEN),
            w2=Linear.Config(in_features=_HIDDEN, out_features=_DIM),
            w3=_ConvertedLinear.Config(in_features=_DIM, out_features=_HIDDEN),
        )

        fused = fused_swiglu(cfg)

        self.assertIsInstance(fused.w1, _ConvertedLinear.Config)
        self.assertIsInstance(fused.w3, _ConvertedLinear.Config)
        self.assertIsInstance(fused.build().w13, _ConvertedLinear)

    def test_saves_in_logical_layout(self):
        """state_dict() emits logical w1/w3 keys, not the physical w13."""
        fused = _build_fused()
        sd = fused.state_dict()
        self.assertEqual(set(sd), {"w1.weight", "w3.weight", "w2.weight"})
        self.assertTrue(torch.equal(sd["w1.weight"], _logical_w13(fused)[:, 0]))
        self.assertTrue(torch.equal(sd["w3.weight"], _logical_w13(fused)[:, 1]))

    @unittest.skipUnless(torch.cuda.is_available(), "silu_and_mul op is CUDA-only")
    def test_triton_checkpoint_loads_into_native(self):
        """A Triton checkpoint loads into native FeedForward, weights + output."""
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
        """A native checkpoint loads into fused SwiGLU, weights + output."""
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

    def test_loads_native_w13(self):
        """A legacy checkpoint keyed by the native w13 still loads (back-compat)."""
        src = _build_fused()
        native = {
            "w13": _logical_w13(src).detach().clone(),
            "w2.weight": src.w2.weight.detach().clone(),
        }
        dst = _build_fused()
        dst.load_state_dict(native)
        self.assertTrue(torch.equal(dst.w13.weight, src.w13.weight))

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
        config = _dist_gemm_ffn_config()
        AsyncTensorParallelTransform().transform(config)
        fused = fused_swiglu(config).build()
        self.assertIsInstance(fused.w13, AsyncAllGatherLinear)
        self.assertIsInstance(fused.w2, AsyncLinearReduceScatter)
        self.assertIs(fused.activation_fn.fn, _fused_silu_and_mul)

    def test_overlapping_variant_keeps_w13_checkpoint_layout(self):
        config = _dist_gemm_ffn_config()
        AsyncTensorParallelTransform().transform(config)
        fused = fused_swiglu(config).build()
        with torch.no_grad():
            fused.w13.weight.copy_(torch.randn(2 * _HIDDEN, _DIM))
        state_dict = fused.state_dict()
        self.assertEqual(set(state_dict), {"w1.weight", "w2.weight", "w3.weight"})
        logical_w13 = fused.w13.weight.unflatten(0, (_HIDDEN, 2))
        torch.testing.assert_close(state_dict["w1.weight"], logical_w13[:, 0])

        reload_config = _dist_gemm_ffn_config()
        AsyncTensorParallelTransform().transform(reload_config)
        reloaded = fused_swiglu(reload_config).build()
        reloaded.load_state_dict(state_dict)
        torch.testing.assert_close(reloaded.w13.weight, fused.w13.weight)


class TestFusedSwiGLUHFAdapter(unittest.TestCase):
    def test_hf_adapter_roundtrip(self):
        """A fused-SwiGLU model interoperates with the HF state-dict adapter.

        The adapter maps HF mlp.gate_proj/up_proj <-> the logical
        feed_forward.w1/w3 FQNs, which the fused module emits and consumes via
        its state_dict hooks.
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
        self.assertIs(ffn.activation_fn.fn, _fused_silu_and_mul)

        sd = model.state_dict()
        # The FFN presents logical checkpoint FQNs, not its physical w13.
        self.assertTrue(any(k.endswith("feed_forward.w1.weight") for k in sd))
        self.assertFalse(any("feed_forward.w13" in k for k in sd))

        adapter = Llama3StateDictAdapter(config, hf_assets_path=None)
        hf_sd = adapter.to_hf(sd)
        self.assertIn("model.layers.0.mlp.gate_proj.weight", hf_sd)
        self.assertIn("model.layers.0.mlp.up_proj.weight", hf_sd)

        # Load the HF checkpoint back through the adapter (which reads logical
        # FQNs) into the fused model; the load hook merges w1/w3 into w13.
        orig_w13 = ffn.w13.weight.detach().clone()
        restored = adapter.from_hf(hf_sd)
        self.assertIn("layers.0.feed_forward.w1.weight", restored)
        model.load_state_dict(restored, strict=False)
        self.assertTrue(torch.equal(ffn.w13.weight, orig_w13))


if __name__ == "__main__":
    unittest.main()
