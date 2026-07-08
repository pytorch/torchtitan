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
from torchtitan.overrides.helion_rope import helion_rope, HelionRoPE

_HELION_INSTALLED = helion_rope_module._HELION_IMPORT_ERROR is None
_HELION_GPU = _HELION_INSTALLED and torch.cuda.is_available()

# The override registers at import time. Capture it now so the registry-dependent
# tests below stay robust if a sibling test (e.g. test_override.py) calls
# clear_overrides() first; setUp restores it.
_HELION_OVERRIDE = _REGISTRY.get("helion_rope")


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

    def test_registered_against_cossin(self):
        self.assertIn("helion_rope", _REGISTRY)
        self.assertIs(_REGISTRY["helion_rope"].target_cls, CosSinRoPE.Config)
        self.assertTrue(_REGISTRY["helion_rope"].exact)

    def test_factory_preserves_fields(self):
        cfg = CosSinRoPE.Config(dim=64, max_seq_len=128, theta=5000.0, scaling="yarn")
        with patch.object(helion_rope_module, "_HELION_IMPORT_ERROR", None):
            replacement = helion_rope(cfg)
        self.assertIsInstance(replacement, HelionRoPE.Config)
        self.assertEqual(replacement.dim, 64)
        self.assertEqual(replacement.max_seq_len, 128)
        self.assertEqual(replacement.theta, 5000.0)
        self.assertEqual(replacement.scaling, "yarn")

    def test_override_requires_helion_install(self):
        root = _RoPEHolder.Config()
        import_error = ImportError("No module named 'helion'")
        with patch.object(helion_rope_module, "_HELION_IMPORT_ERROR", import_error):
            with self.assertRaisesRegex(ImportError, "helion.*not installed"):
                apply_overrides(
                    OverrideConfig(imports=["torchtitan.overrides.helion_rope"]), root
                )

    def test_override_claims_only_cossin(self):
        root = _RoPEHolder.Config()
        with patch.object(helion_rope_module, "_HELION_IMPORT_ERROR", None):
            replacements = apply_overrides(
                OverrideConfig(imports=["torchtitan.overrides.helion_rope"]), root
            )
        self.assertEqual(len(replacements), 1)
        # cos/sin node is swapped; complex node (Llama 3 / DeepSeek) is untouched.
        self.assertIsInstance(root.cos_rope, HelionRoPE.Config)
        self.assertIs(type(root.complex_rope), ComplexRoPE.Config)

    def test_override_does_not_claim_cossin_subclass(self):
        root = _SubclassRoPEHolder.Config()
        replacements = apply_overrides(
            OverrideConfig(imports=["torchtitan.overrides.helion_rope"]), root
        )
        self.assertEqual(replacements, [])
        self.assertIs(type(root.rope), _ContractSubclassRoPE.Config)
        self.assertEqual(root.rope.extra, 7)

    @unittest.skipUnless(_HELION_INSTALLED, "requires helion")
    def test_cpu_falls_back_to_cossin(self):
        """On CPU the module falls back to CosSinRoPE, bit-for-bit identical."""
        torch.manual_seed(0)
        dim, seqlen = 64, 16
        cossin = CosSinRoPE.Config(dim=dim, max_seq_len=seqlen).build()
        helion_module = HelionRoPE.Config(dim=dim, max_seq_len=seqlen).build()
        self.assertIsInstance(helion_module, HelionRoPE)

        xq = torch.randn(2, seqlen, 4, dim, dtype=torch.bfloat16)
        xk = torch.randn(2, seqlen, 1, dim, dtype=torch.bfloat16)
        positions = torch.arange(seqlen).unsqueeze(0).expand(2, -1)

        ref_q, ref_k = cossin(xq, xk, positions)
        out_q, out_k = helion_module(xq, xk, positions)
        torch.testing.assert_close(out_q, ref_q, rtol=0, atol=0)
        torch.testing.assert_close(out_k, ref_k, rtol=0, atol=0)

    @unittest.skipIf(_HELION_INSTALLED, "requires helion to be missing")
    def test_forward_errors_without_helion(self):
        torch.manual_seed(0)
        dim, seqlen = 64, 16
        helion_module = HelionRoPE.Config(dim=dim, max_seq_len=seqlen).build()

        xq = torch.randn(2, seqlen, 4, dim, dtype=torch.bfloat16)
        xk = torch.randn(2, seqlen, 1, dim, dtype=torch.bfloat16)
        positions = torch.arange(seqlen).unsqueeze(0).expand(2, -1)

        with self.assertRaisesRegex(ImportError, "helion.*not installed"):
            helion_module(xq, xk, positions)


@unittest.skipUnless(_HELION_GPU, "requires helion + CUDA")
class TestHelionRoPEKernel(unittest.TestCase):
    """Fused-kernel numerics vs the PyTorch cos/sin RoPE (helion + CUDA only)."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda")
        self.dim = 128
        self.seqlen = 64
        self.cossin = CosSinRoPE.Config(dim=self.dim, max_seq_len=self.seqlen).build()
        self.helion = HelionRoPE.Config(dim=self.dim, max_seq_len=self.seqlen).build()
        self.cossin.to(self.device)
        self.helion.to(self.device)

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

    def test_eligible_rejects_unsupported_inputs(self):
        # The eligibility gate must reject inputs the kernel can't safely gather,
        # so they fall back to PyTorch instead of failing inside the kernel.
        from torchtitan.overrides.helion_rope import _eligible

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

        self.assertTrue(_eligible(xq, xk, cache, pos))  # baseline is eligible
        self.assertFalse(_eligible(xq, xk, cache.cpu(), pos))  # cache off-device
        self.assertFalse(_eligible(xq, xk, cache, pos.cpu()))  # positions off-device
        self.assertFalse(_eligible(xq, xk, cache, pos.float()))  # non-integer ids
        self.assertFalse(_eligible(xq, xk, wrong_width, pos))  # wrong cache width
        self.assertFalse(_eligible(xq, xk, cache, pos[:1]))  # positions shape mismatch

        # q/k batch/seq must match: the kernel indexes xk with xq's (b, s) tiles.
        xk_short = torch.randn(
            2, self.seqlen // 2, 1, self.dim, device=d, dtype=torch.bfloat16
        )
        self.assertFalse(_eligible(xq, xk_short, cache, pos))
        # non-4D q/k (kernel unpacks both as 4D).
        self.assertFalse(_eligible(xq[:, :, 0, :], xk[:, :, 0, :], cache, pos))

        if torch.cuda.device_count() >= 2:
            # All-CUDA but split across devices: caught by the same-device check,
            # not is_cuda (the kernel can't gather across devices).
            self.assertFalse(_eligible(xq, xk, cache, pos.to("cuda:1")))

    def test_custom_op_opcheck(self):
        from torchtitan.overrides.helion_rope import _helion_rope_fwd

        xq, xk, positions = self._inputs()
        torch.library.opcheck(_helion_rope_fwd, (xq, xk, self.helion.cache, positions))

    def test_backward_custom_op_opcheck_with_noncontiguous_grads(self):
        from torchtitan.overrides.helion_rope import _helion_rope_bwd

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
            _helion_rope_bwd,
            (grad_xq_out, grad_xk_out, self.helion.cache, positions),
        )


if __name__ == "__main__":
    unittest.main()
