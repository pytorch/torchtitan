# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass, field
from unittest.mock import patch

import torch
import torchtitan.overrides.helion_rope as helion_rope_module
from torchtitan.config import apply_overrides, Configurable, OverrideConfig
from torchtitan.config.override import _REGISTRY
from torchtitan.models.common.rope import ComplexRoPE, CosSinRoPE

# Importing the override module registers the "helion_rope" override. The import
# is safe without helion, but explicitly applying the override requires helion.
from torchtitan.overrides.helion_rope import (
    helion_complex_rope,
    helion_rope,
    HelionComplexRoPE,
    HelionCosSinRoPE,
)

# The override registers at import time. Capture it now so the registry-dependent
# tests below stay robust if a sibling test (e.g. test_override.py) calls
# clear_overrides() first; setUp restores it.
_HELION_OVERRIDE = _REGISTRY.get("helion_rope")
_HELION_COMPLEX_OVERRIDE = _REGISTRY.get("helion_complex_rope")


class _RoPEHolder(Configurable):
    """Minimal config tree with one cos/sin and one complex RoPE node."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        cos_rope: CosSinRoPE.Config = field(
            default_factory=lambda: CosSinRoPE.Config(dim=64, max_seq_len=128)
        )
        complex_rope: ComplexRoPE.Config = field(
            default_factory=lambda: ComplexRoPE.Config(dim=64, max_seq_len=128)
        )

    def __init__(self, config: Config):
        self.config = config


class _ContractSubclassRoPE(CosSinRoPE):
    """CosSinRoPE subclass with a different cache contract, like MRoPE."""

    @dataclass(kw_only=True, slots=True)
    class Config(CosSinRoPE.Config):
        extra: int = 7


class _SubclassRoPEHolder(Configurable):
    """Config tree containing a CosSinRoPE subclass node."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        rope: CosSinRoPE.Config = field(
            default_factory=lambda: _ContractSubclassRoPE.Config(
                dim=64, max_seq_len=128
            )
        )

    def __init__(self, config: Config):
        self.config = config


class TestHelionRoPEOverride(unittest.TestCase):
    """Registration, factory, and PyTorch-fallback parity."""

    def setUp(self):
        # Restore the override if a previously run test cleared the registry.
        if _HELION_OVERRIDE is not None:
            _REGISTRY.setdefault("helion_rope", _HELION_OVERRIDE)
        if _HELION_COMPLEX_OVERRIDE is not None:
            _REGISTRY.setdefault("helion_complex_rope", _HELION_COMPLEX_OVERRIDE)

    def test_registered_against_cossin(self):
        self.assertIn("helion_rope", _REGISTRY)
        self.assertIs(_REGISTRY["helion_rope"].target_cls, CosSinRoPE.Config)
        self.assertTrue(_REGISTRY["helion_rope"].exact)

    def test_registered_against_complex(self):
        self.assertIn("helion_complex_rope", _REGISTRY)
        self.assertIs(_REGISTRY["helion_complex_rope"].target_cls, ComplexRoPE.Config)
        self.assertTrue(_REGISTRY["helion_complex_rope"].exact)

    def test_cossin_factory_preserves_fields(self):
        cfg = CosSinRoPE.Config(dim=64, max_seq_len=128, theta=5000.0, scaling="yarn")
        with patch.object(helion_rope_module, "_HELION_IMPORT_ERROR", None):
            replacement = helion_rope(cfg)
        self.assertIsInstance(replacement, HelionCosSinRoPE.Config)
        self.assertEqual(replacement.dim, 64)
        self.assertEqual(replacement.max_seq_len, 128)
        self.assertEqual(replacement.theta, 5000.0)
        self.assertEqual(replacement.scaling, "yarn")

    def test_complex_factory_preserves_fields(self):
        cfg = ComplexRoPE.Config(
            dim=64,
            max_seq_len=128,
            theta=5000.0,
            scaling="yarn",
            rope_factor=40.0,
        )
        with patch.object(helion_rope_module, "_HELION_IMPORT_ERROR", None):
            replacement = helion_complex_rope(cfg)
        self.assertIsInstance(replacement, HelionComplexRoPE.Config)
        self.assertEqual(replacement.dim, 64)
        self.assertEqual(replacement.max_seq_len, 128)
        self.assertEqual(replacement.theta, 5000.0)
        self.assertEqual(replacement.scaling, "yarn")
        self.assertEqual(replacement.rope_factor, 40.0)

    def test_override_requires_helion_install(self):
        root = _RoPEHolder.Config()
        import_error = ImportError("No module named 'helion'")
        with patch.object(helion_rope_module, "_HELION_IMPORT_ERROR", import_error):
            with self.assertRaisesRegex(ImportError, "helion.*not installed"):
                apply_overrides(
                    OverrideConfig(imports=["torchtitan.overrides.helion_rope"]), root
                )

    def test_override_claims_cossin_and_complex(self):
        root = _RoPEHolder.Config()
        with patch.object(helion_rope_module, "_HELION_IMPORT_ERROR", None):
            replacements = apply_overrides(
                OverrideConfig(imports=["torchtitan.overrides.helion_rope"]), root
            )
        self.assertEqual(len(replacements), 2)
        self.assertIsInstance(root.cos_rope, HelionCosSinRoPE.Config)
        self.assertIsInstance(root.complex_rope, HelionComplexRoPE.Config)

    def test_override_does_not_claim_cossin_subclass(self):
        root = _SubclassRoPEHolder.Config()
        replacements = apply_overrides(
            OverrideConfig(imports=["torchtitan.overrides.helion_rope"]), root
        )
        self.assertEqual(replacements, [])
        self.assertIs(type(root.rope), _ContractSubclassRoPE.Config)
        self.assertEqual(root.rope.extra, 7)

    def test_cpu_falls_back_to_cossin(self):
        """On CPU the module falls back to CosSinRoPE, bit-for-bit identical."""
        torch.manual_seed(0)
        dim, seqlen = 64, 16
        cossin = CosSinRoPE.Config(dim=dim, max_seq_len=seqlen).build()
        helion_module = HelionCosSinRoPE.Config(dim=dim, max_seq_len=seqlen).build()
        self.assertIsInstance(helion_module, HelionCosSinRoPE)

        xq = torch.randn(2, seqlen, 4, dim, dtype=torch.bfloat16)
        xk = torch.randn(2, seqlen, 1, dim, dtype=torch.bfloat16)
        positions = torch.arange(seqlen).unsqueeze(0).expand(2, -1)

        ref_q, ref_k = cossin(xq, xk, positions)
        out_q, out_k = helion_module(xq, xk, positions)
        torch.testing.assert_close(out_q, ref_q, rtol=0, atol=0)
        torch.testing.assert_close(out_k, ref_k, rtol=0, atol=0)

    def test_cpu_falls_back_to_complex(self):
        """On CPU the module falls back to ComplexRoPE, bit-for-bit identical."""
        torch.manual_seed(0)
        dim, seqlen = 64, 16
        complex_rope = ComplexRoPE.Config(dim=dim, max_seq_len=seqlen).build()
        helion_module = HelionComplexRoPE.Config(dim=dim, max_seq_len=seqlen).build()
        self.assertIsInstance(helion_module, HelionComplexRoPE)

        xq = torch.randn(2, seqlen, 4, dim, dtype=torch.bfloat16)
        xk = torch.randn(2, seqlen, 1, dim, dtype=torch.bfloat16)
        positions = torch.arange(seqlen).unsqueeze(0).expand(2, -1)

        ref_q, ref_k = complex_rope(xq, xk, positions)
        out_q, out_k = helion_module(xq, xk, positions)
        torch.testing.assert_close(out_q, ref_q, rtol=0, atol=0)
        torch.testing.assert_close(out_k, ref_k, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestHelionRoPEKernel(unittest.TestCase):
    """Fused-kernel numerics vs the PyTorch RoPE modules (helion + CUDA only)."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda")
        self.dim = 128
        self.seqlen = 64
        self.cossin = CosSinRoPE.Config(dim=self.dim, max_seq_len=self.seqlen).build()
        self.complex = ComplexRoPE.Config(
            dim=self.dim,
            max_seq_len=self.seqlen * 2,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=self.seqlen,
        ).build()
        self.helion = HelionCosSinRoPE.Config(
            dim=self.dim, max_seq_len=self.seqlen
        ).build()
        self.helion_complex = HelionComplexRoPE.Config(
            dim=self.dim,
            max_seq_len=self.seqlen * 2,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=self.seqlen,
        ).build()
        self.cossin.to(self.device)
        self.complex.to(self.device)
        self.helion.to(self.device)
        self.helion_complex.to(self.device)

    def _inputs(self, *, batch=2, n_heads=8, n_kv_heads=1, requires_grad=False):
        xq = torch.randn(
            batch,
            self.seqlen,
            n_heads,
            self.dim,
            device=self.device,
            dtype=torch.bfloat16,
            requires_grad=requires_grad,
        )
        xk = torch.randn(
            batch,
            self.seqlen,
            n_kv_heads,
            self.dim,
            device=self.device,
            dtype=torch.bfloat16,
            requires_grad=requires_grad,
        )
        positions = (
            torch.arange(self.seqlen, device=self.device)
            .unsqueeze(0)
            .expand(batch, -1)
            .contiguous()
        )
        return xq, xk, positions

    def test_forward_matches_cossin(self):
        xq, xk, positions = self._inputs()
        ref_q, ref_k = self.cossin(xq, xk, positions)
        out_q, out_k = self.helion(xq, xk, positions)
        self.assertEqual(out_q.shape, ref_q.shape)
        self.assertEqual(out_q.dtype, ref_q.dtype)
        torch.testing.assert_close(out_q, ref_q, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(out_k, ref_k, rtol=2e-2, atol=2e-2)

    def test_forward_matches_with_implicit_positions(self):
        # positions=None must match cache[0:seq_len] (the kernel synthesizes arange).
        xq, xk, _ = self._inputs()
        ref_q, ref_k = self.cossin(xq, xk, None)
        out_q, out_k = self.helion(xq, xk, None)
        torch.testing.assert_close(out_q, ref_q, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(out_k, ref_k, rtol=2e-2, atol=2e-2)

    def test_complex_forward_matches(self):
        xq, xk, positions = self._inputs()
        ref_q, ref_k = self.complex(xq, xk, positions)
        out_q, out_k = self.helion_complex(xq, xk, positions)
        self.assertEqual(out_q.shape, ref_q.shape)
        self.assertEqual(out_q.dtype, ref_q.dtype)
        torch.testing.assert_close(out_q, ref_q, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(out_k, ref_k, rtol=2e-2, atol=2e-2)

    def test_complex_forward_matches_with_implicit_positions(self):
        xq, xk, _ = self._inputs()
        ref_q, ref_k = self.complex(xq, xk, None)
        out_q, out_k = self.helion_complex(xq, xk, None)
        torch.testing.assert_close(out_q, ref_q, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(out_k, ref_k, rtol=2e-2, atol=2e-2)

    def test_backward_matches_cossin(self):
        xq, xk, positions = self._inputs(requires_grad=True)
        xq_ref = xq.detach().clone().requires_grad_(True)
        xk_ref = xk.detach().clone().requires_grad_(True)

        self.helion(xq, xk, positions)[0].sum().backward()
        self.cossin(xq_ref, xk_ref, positions)[0].sum().backward()

        torch.testing.assert_close(xq.grad, xq_ref.grad, rtol=2e-2, atol=2e-2)
        # xk does not feed the summed output, so its grad is zero on both paths.
        self.assertIsNotNone(xk.grad)

        xq2, xk2, positions2 = self._inputs(requires_grad=True)
        xk2_ref = xk2.detach().clone().requires_grad_(True)
        xq2_ref = xq2.detach().clone().requires_grad_(True)
        self.helion(xq2, xk2, positions2)[1].sum().backward()
        self.cossin(xq2_ref, xk2_ref, positions2)[1].sum().backward()
        torch.testing.assert_close(xk2.grad, xk2_ref.grad, rtol=2e-2, atol=2e-2)

    def test_complex_backward_matches(self):
        xq, xk, positions = self._inputs(requires_grad=True)
        xq_ref = xq.detach().clone().requires_grad_(True)
        xk_ref = xk.detach().clone().requires_grad_(True)

        helion_out = self.helion_complex(xq, xk, positions)
        ref_out = self.complex(xq_ref, xk_ref, positions)
        (helion_out[0].float().square().mean() + helion_out[1].float().sum()).backward()
        (ref_out[0].float().square().mean() + ref_out[1].float().sum()).backward()

        torch.testing.assert_close(xq.grad, xq_ref.grad, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(xk.grad, xk_ref.grad, rtol=2e-2, atol=2e-2)

    def test_complex_matches_noncontiguous_split_views(self):
        # DeepSeek V3 passes q/k RoPE slices as split views. Stock ComplexRoPE
        # accepts those strided inputs and returns contiguous outputs, so the
        # Helion override must preserve that boundary contract without requiring
        # model-side layout changes.
        batch, n_heads = 2, 8
        q_full = torch.randn(
            batch,
            self.seqlen,
            n_heads,
            self.dim + 32,
            device=self.device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        k_full = torch.randn(
            batch,
            self.seqlen,
            self.dim + 32,
            device=self.device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        q = q_full[..., -self.dim :]
        k = k_full[..., -self.dim :].unsqueeze(2)
        self.assertFalse(q.is_contiguous())
        self.assertFalse(k.is_contiguous())

        q_full_ref = q_full.detach().clone().requires_grad_(True)
        k_full_ref = k_full.detach().clone().requires_grad_(True)
        q_ref = q_full_ref[..., -self.dim :]
        k_ref = k_full_ref[..., -self.dim :].unsqueeze(2)
        positions = (
            torch.arange(self.seqlen, device=self.device).unsqueeze(0).expand(batch, -1)
        )

        self.assertIsNotNone(
            helion_rope_module._apply_helion_complex_rope(
                q, k, self.helion_complex.cache, positions
            )
        )
        helion_out = self.helion_complex(q, k, positions)
        ref_out = self.complex(q_ref, k_ref, positions)

        self.assertTrue(helion_out[0].is_contiguous())
        self.assertTrue(helion_out[1].is_contiguous())
        self.assertEqual(helion_out[0].stride(), ref_out[0].stride())
        self.assertEqual(helion_out[1].stride(), ref_out[1].stride())
        torch.testing.assert_close(helion_out[0], ref_out[0], rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(helion_out[1], ref_out[1], rtol=2e-2, atol=2e-2)

        q_grad_full = torch.randn(
            batch,
            self.seqlen,
            n_heads,
            self.dim + 32,
            device=self.device,
            dtype=torch.bfloat16,
        )
        k_grad_full = torch.randn(
            batch,
            self.seqlen,
            self.dim + 32,
            device=self.device,
            dtype=torch.bfloat16,
        )
        q_grad = q_grad_full[..., -self.dim :]
        k_grad = k_grad_full[..., -self.dim :].unsqueeze(2)
        self.assertFalse(q_grad.is_contiguous())
        self.assertFalse(k_grad.is_contiguous())

        torch.autograd.backward(helion_out, grad_tensors=(q_grad, k_grad))
        torch.autograd.backward(ref_out, grad_tensors=(q_grad, k_grad))
        torch.testing.assert_close(q_full.grad, q_full_ref.grad, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(k_full.grad, k_full_ref.grad, rtol=2e-2, atol=2e-2)

    def test_complex_matches_transposed_head_stride_layout(self):
        # The complex kernel only requires adjacent real/imag pairs to be
        # contiguous in the last dimension. Other dimensions may have arbitrary
        # strides; Helion indexes them directly and still returns contiguous
        # outputs like stock ComplexRoPE.
        batch, n_heads = 2, 8
        q_storage = torch.randn(
            batch,
            n_heads,
            self.seqlen,
            self.dim,
            device=self.device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        q = q_storage.transpose(1, 2)
        k = torch.randn(
            batch,
            self.seqlen,
            1,
            self.dim,
            device=self.device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        self.assertFalse(q.is_contiguous())
        self.assertEqual(q.stride(-1), 1)

        q_storage_ref = q_storage.detach().clone().requires_grad_(True)
        q_ref = q_storage_ref.transpose(1, 2)
        k_ref = k.detach().clone().requires_grad_(True)
        positions = (
            torch.arange(self.seqlen, device=self.device)
            .unsqueeze(0)
            .expand(batch, -1)
            .contiguous()
        )

        self.assertIsNotNone(
            helion_rope_module._apply_helion_complex_rope(
                q, k, self.helion_complex.cache, positions
            )
        )
        helion_out = self.helion_complex(q, k, positions)
        ref_out = self.complex(q_ref, k_ref, positions)

        self.assertTrue(helion_out[0].is_contiguous())
        self.assertTrue(helion_out[1].is_contiguous())
        torch.testing.assert_close(helion_out[0], ref_out[0], rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(helion_out[1], ref_out[1], rtol=2e-2, atol=2e-2)

        q_grad = torch.randn_like(ref_out[0])
        k_grad = torch.randn_like(ref_out[1])
        torch.autograd.backward(helion_out, grad_tensors=(q_grad, k_grad))
        torch.autograd.backward(ref_out, grad_tensors=(q_grad, k_grad))
        torch.testing.assert_close(
            q_storage.grad, q_storage_ref.grad, rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(k.grad, k_ref.grad, rtol=2e-2, atol=2e-2)

    def test_complex_matches_split_query_contiguous_key_grad_layout(self):
        # Autograd may give RoPE a split-view q grad and a contiguous k grad.
        # The optimized path must preserve stock ComplexRoPE semantics for that
        # layout as well as for split/split grads.
        batch, n_heads = 2, 8
        q_full = torch.randn(
            batch,
            self.seqlen,
            n_heads,
            self.dim + 32,
            device=self.device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        k_full = torch.randn(
            batch,
            self.seqlen,
            self.dim + 32,
            device=self.device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        q = q_full[..., -self.dim :]
        k = k_full[..., -self.dim :].unsqueeze(2)

        q_full_ref = q_full.detach().clone().requires_grad_(True)
        k_full_ref = k_full.detach().clone().requires_grad_(True)
        q_ref = q_full_ref[..., -self.dim :]
        k_ref = k_full_ref[..., -self.dim :].unsqueeze(2)
        positions = (
            torch.arange(self.seqlen, device=self.device).unsqueeze(0).expand(batch, -1)
        )

        helion_out = self.helion_complex(q, k, positions)
        ref_out = self.complex(q_ref, k_ref, positions)

        q_grad_full = torch.randn(
            batch,
            self.seqlen,
            n_heads,
            self.dim + 32,
            device=self.device,
            dtype=torch.bfloat16,
        )
        q_grad = q_grad_full[..., -self.dim :]
        k_grad = torch.randn_like(ref_out[1])
        self.assertFalse(q_grad.is_contiguous())
        self.assertTrue(k_grad.is_contiguous())

        torch.autograd.backward(helion_out, grad_tensors=(q_grad, k_grad))
        torch.autograd.backward(ref_out, grad_tensors=(q_grad, k_grad))
        torch.testing.assert_close(q_full.grad, q_full_ref.grad, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(k_full.grad, k_full_ref.grad, rtol=2e-2, atol=2e-2)

    def test_eligible_rejects_unsupported_inputs(self):
        # The eligibility gate must reject inputs the kernel can't safely gather,
        # so they fall back to PyTorch instead of failing inside the kernel.
        from torchtitan.overrides.helion_rope import _complex_eligible, _cossin_eligible

        d = self.device
        xq = torch.randn(2, self.seqlen, 8, self.dim, device=d, dtype=torch.bfloat16)
        xk = torch.randn(2, self.seqlen, 1, self.dim, device=d, dtype=torch.bfloat16)
        cache = self.helion.cache  # (max_seq, 2 * dim) on CUDA
        pos = (
            torch.arange(self.seqlen, device=d, dtype=torch.int32)
            .unsqueeze(0)
            .expand(2, -1)
            .contiguous()
        )
        wrong_width = torch.randn(self.seqlen, 2 * self.dim - 2, device=d)

        self.assertTrue(_cossin_eligible(xq, xk, cache, pos))
        self.assertFalse(_cossin_eligible(xq, xk, cache.cpu(), pos))
        self.assertFalse(_cossin_eligible(xq, xk, cache, pos.cpu()))
        self.assertFalse(_cossin_eligible(xq, xk, cache, pos.float()))
        self.assertFalse(_cossin_eligible(xq, xk, wrong_width, pos))
        self.assertFalse(_cossin_eligible(xq, xk, cache, pos[:1]))

        # q/k batch/seq must match: the kernel indexes xk with xq's (b, s) tiles.
        xk_short = torch.randn(
            2, self.seqlen // 2, 1, self.dim, device=d, dtype=torch.bfloat16
        )
        self.assertFalse(_cossin_eligible(xq, xk_short, cache, pos))
        # non-4D q/k (kernel unpacks both as 4D).
        self.assertFalse(_cossin_eligible(xq[:, :, 0, :], xk[:, :, 0, :], cache, pos))

        cache_real = torch.view_as_real(self.helion_complex.cache).contiguous()
        q_transposed_layout = torch.empty(
            2, 8, self.seqlen, self.dim, device=d, dtype=torch.bfloat16
        ).transpose(1, 2)
        self.assertTrue(_complex_eligible(q_transposed_layout, xk, cache_real, pos))
        q_bad_last_dim = torch.empty(
            2, self.seqlen, 8, self.dim * 2, device=d, dtype=torch.bfloat16
        )[..., ::2]
        self.assertFalse(_complex_eligible(q_bad_last_dim, xk, cache_real, pos))

        if torch.cuda.device_count() >= 2:
            # All-CUDA but split across devices: caught by the same-device check,
            # not is_cuda (the kernel can't gather across devices).
            self.assertFalse(_cossin_eligible(xq, xk, cache, pos.to("cuda:1")))

    def test_custom_op_opcheck(self):
        from torchtitan.overrides.helion_rope import _helion_cossin_rope_fwd

        xq, xk, positions = self._inputs()
        torch.library.opcheck(
            _helion_cossin_rope_fwd, (xq, xk, self.helion.cache, positions)
        )

    def test_complex_custom_op_opcheck(self):
        from torchtitan.overrides.helion_rope import _helion_complex_rope_fwd

        xq, xk, positions = self._inputs()
        torch.library.opcheck(
            _helion_complex_rope_fwd,
            (xq, xk, torch.view_as_real(self.helion_complex.cache), positions),
        )

    def test_complex_make_fx_traces_custom_op(self):
        from torch.fx.experimental.proxy_tensor import make_fx

        xq, xk, positions = self._inputs()

        def fn(q, k, pos):
            return self.helion_complex(q, k, pos)

        gm = make_fx(fn)(xq, xk, positions)
        targets = {node.target for node in gm.graph.nodes}
        self.assertIn(
            torch.ops.torchtitan.helion_complex_rope_fwd.default,
            targets,
        )

    def test_backward_custom_op_opcheck_with_noncontiguous_grads(self):
        from torchtitan.overrides.helion_rope import _helion_cossin_rope_bwd

        grad_xq_out = torch.randn(
            self.seqlen,
            2,
            8,
            self.dim,
            device=self.device,
            dtype=torch.bfloat16,
        ).transpose(0, 1)
        grad_xk_out = torch.randn(
            self.seqlen,
            2,
            1,
            self.dim,
            device=self.device,
            dtype=torch.bfloat16,
        ).transpose(0, 1)
        positions = (
            torch.arange(self.seqlen, device=self.device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(2, -1)
            .contiguous()
        )

        self.assertFalse(grad_xq_out.is_contiguous())
        self.assertFalse(grad_xk_out.is_contiguous())
        torch.library.opcheck(
            _helion_cossin_rope_bwd,
            (grad_xq_out, grad_xk_out, self.helion.cache, positions),
        )


if __name__ == "__main__":
    unittest.main()
