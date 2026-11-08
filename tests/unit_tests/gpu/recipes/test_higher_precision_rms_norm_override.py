# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
import unittest
from dataclasses import dataclass
from unittest import mock

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

from torchtitan.config import apply_overrides, Configurable, OverrideConfig
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan_recipes.overrides import (
    higher_precision_rms_norm as higher_precision_rms_norm_module,
)
from torchtitan_recipes.overrides.higher_precision_rms_norm import (
    higher_precision_rms_norm,
    higher_precision_rms_norm_backward_op,
    higher_precision_rms_norm_op,
    HigherPrecisionRMSNorm,
    rms_norm_backward_kernel,
    rms_norm_forward_kernel,
)

_EPS = 1e-5


def _reference(x, weight, eps=_EPS):
    """FP64 reference RMSNorm, returning ``(x, w, y)``.

    ``x`` and ``w`` are FP64 leaf copies that require grad, and ``y`` is the FP64
    output, so callers can backpropagate through ``y`` and read ``x.grad`` and
    ``w.grad``.
    """
    x = x.detach().double().requires_grad_()
    w = None if weight is None else weight.detach().double().requires_grad_()
    y = torch.nn.functional.rms_norm(x, (x.shape[-1],), w, eps)
    return x, w, y


class TestHigherPrecisionRMSNormOverrideConfig(unittest.TestCase):
    def test_config_is_replaced_and_fields_carried_over(self):
        cfg = RMSNorm.Config(normalized_shape=32, eps=1e-6, elementwise_affine=False)

        replacement = higher_precision_rms_norm(cfg)

        self.assertIsInstance(replacement, HigherPrecisionRMSNorm.Config)
        self.assertEqual(replacement.normalized_shape, 32)
        self.assertEqual(replacement.eps, 1e-6)
        self.assertFalse(replacement.elementwise_affine)
        self.assertTrue(replacement.use_kahan_summation)

    def test_kwarg_sets_accumulation_mode(self):
        for use_kahan in (True, False):
            replacement = higher_precision_rms_norm(
                RMSNorm.Config(normalized_shape=8), use_kahan_summation=use_kahan
            )
            self.assertEqual(replacement.use_kahan_summation, use_kahan)

    def test_builds_with_native_parameter_layout(self):
        native = RMSNorm.Config(normalized_shape=16).build()
        fused = higher_precision_rms_norm(RMSNorm.Config(normalized_shape=16)).build()
        self.assertEqual(native.state_dict().keys(), fused.state_dict().keys())

    def test_apply_overrides_replaces_every_norm(self):
        class Block(Configurable):
            @dataclass(kw_only=True, slots=True)
            class Config(Configurable.Config):
                norm_a: RMSNorm.Config
                norm_b: RMSNorm.Config

        cfg = Block.Config(
            norm_a=RMSNorm.Config(normalized_shape=4),
            norm_b=RMSNorm.Config(normalized_shape=4),
        )
        apply_overrides(
            OverrideConfig(
                imports=[
                    "torchtitan_recipes.overrides.higher_precision_rms_norm.higher_precision_rms_norm"
                ]
            ),
            cfg,
        )
        self.assertIsInstance(cfg.norm_a, HigherPrecisionRMSNorm.Config)
        self.assertIsInstance(cfg.norm_b, HigherPrecisionRMSNorm.Config)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestHigherPrecisionRMSNormNumerics(unittest.TestCase):
    @parametrize("dtype", [torch.bfloat16, torch.float32])
    @parametrize("use_kahan", [False, True])
    @parametrize("dim", [1, 96, 128, 4096, 5120, 12288])
    @parametrize("has_weight", [True, False])
    def test_forward_backward_matches_fp64_reference(
        self, dtype, use_kahan, dim, has_weight
    ):
        torch.manual_seed(0)
        x = (torch.randn(3, 37, dim, device="cuda") * 3).to(dtype).requires_grad_()
        weight = (
            (torch.rand(dim, device="cuda") + 0.5).to(dtype).requires_grad_()
            if has_weight
            else None
        )
        dy = torch.randn(3, 37, dim, device="cuda").to(dtype)

        y, rstd = higher_precision_rms_norm_op(x, weight, _EPS, use_kahan)
        y.backward(dy)

        x_ref, w_ref, y_ref = _reference(x, weight)
        y_ref.backward(dy.double())

        # Error of the kernel vs FP64 must be at most one rounding of the output
        # dtype (the kernel rounds once), plus a small multiple for dx/dw stores.
        atol = {torch.bfloat16: 2e-2, torch.float32: 2e-5}[dtype]
        rtol = {torch.bfloat16: 2e-2, torch.float32: 2e-5}[dtype]
        torch.testing.assert_close(y.double(), y_ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(x.grad.double(), x_ref.grad, atol=atol, rtol=rtol)
        if has_weight:
            # dw sums over 111 rows; compare against its own scale.
            dw_scale = w_ref.grad.abs().max().item()
            torch.testing.assert_close(
                weight.grad.double(), w_ref.grad, atol=atol * dw_scale, rtol=rtol
            )
        self.assertEqual(rstd.dtype, torch.float32)
        self.assertEqual(rstd.shape, x.shape[:-1])

    @parametrize("use_kahan", [False, True])
    @parametrize("dim", [4096, 12288])
    def test_high_precision_accumulation_over_fp32_torch_norm(self, dim, use_kahan):
        """FP32 in and out: closer to the FP64 reference than F.rms_norm."""
        torch.manual_seed(0)
        x = torch.randn(512, dim, device="cuda") * 3
        weight = torch.rand(dim, device="cuda") + 0.5

        y_ref = _reference(x, weight)[2].detach()
        err_torch = (
            torch.nn.functional.rms_norm(x, (dim,), weight, _EPS).double() - y_ref
        ).abs()
        y, _ = rms_norm_forward_kernel(x, weight, _EPS, use_kahan)
        err = (y.double() - y_ref).abs()
        self.assertLess(err.mean().item(), err_torch.mean().item())
        self.assertLessEqual(err.max().item(), err_torch.max().item())

    def test_forward_kernel_has_no_fma_contraction(self):
        """TwoSum is only exact if no mul + add is contracted into an FMA, so the
        Kahan forward kernel must leave no plain (contractible) mul/add/sub."""
        module = higher_precision_rms_norm_module
        real_kernel = module._rms_norm_forward_kernel
        compiled = {}

        class _SpyKernel:
            def __getitem__(self, grid):
                launch = real_kernel[grid]

                def run(*args, **kwargs):
                    compiled["kernel"] = launch(*args, **kwargs)

                return run

        x = torch.randn(64, 4096, device="cuda")
        with mock.patch.object(module, "_rms_norm_forward_kernel", _SpyKernel()):
            rms_norm_forward_kernel(x, None, _EPS, True)
        ptx = compiled["kernel"].asm["ptx"]
        self.assertEqual(len(re.findall(r"\b(?:mul|add|sub)\.f32\b", ptx)), 0)

    @parametrize("use_kahan", [False, True])
    @parametrize("dim", [4096, 12288])
    def test_rstd_is_correctly_rounded_beyond_fp32_sum(self, dim, use_kahan):
        """Both modes sum squares beyond FP32 precision at any dim, including
        dim <= 8192 where the row is a single column block. The stored FP32 rstd
        is then almost always the correctly rounded FP64 value; an FP32 sum is not.
        """
        torch.manual_seed(0)
        scale = torch.exp(2 * torch.randn(4096, dim, device="cuda"))
        x = torch.randn(4096, dim, device="cuda") * scale
        _, rstd = rms_norm_forward_kernel(x, None, _EPS, use_kahan)
        rstd_ref = torch.rsqrt(x.double().pow(2).mean(-1) + _EPS).float()
        rstd_fp32_sum = torch.rsqrt(x.pow(2).mean(-1) + _EPS)
        kernel_exact = (rstd == rstd_ref).float().mean().item()
        fp32_sum_exact = (rstd_fp32_sum == rstd_ref).float().mean().item()
        self.assertGreater(kernel_exact, 0.999)
        self.assertGreater(kernel_exact, fp32_sum_exact + 0.1)

    def test_default_is_kahan_not_fp64(self):
        torch.manual_seed(0)
        x = torch.randn(64, 12288, device="cuda")
        weight = torch.rand(12288, device="cuda") + 0.5
        y_default, rstd_default = rms_norm_forward_kernel(x, weight, _EPS)
        y_kahan, rstd_kahan = rms_norm_forward_kernel(x, weight, _EPS, True)
        y_fp64, rstd_fp64 = rms_norm_forward_kernel(x, weight, _EPS, False)
        self.assertTrue(torch.equal(y_default, y_kahan))
        self.assertTrue(torch.equal(rstd_default, rstd_kahan))
        # The two modes are distinguishable, so the check above is meaningful.
        self.assertFalse(torch.equal(y_kahan, y_fp64))

    def test_kahan_and_fp64_agree_closely(self):
        torch.manual_seed(0)
        x = torch.randn(64, 12288, device="cuda")
        weight = torch.rand(12288, device="cuda") + 0.5
        y64, rstd64 = rms_norm_forward_kernel(x, weight, _EPS, False)
        yk, rstdk = rms_norm_forward_kernel(x, weight, _EPS, True)
        torch.testing.assert_close(rstdk, rstd64, atol=0, rtol=1e-6)
        torch.testing.assert_close(yk, y64, atol=1e-6, rtol=1e-6)

    def test_non_contiguous_rows_and_grad(self):
        torch.manual_seed(0)
        base = torch.randn(40, 2 * 128, device="cuda", dtype=torch.bfloat16)
        x = base[:, :128].requires_grad_()  # row stride 256, unit column stride
        weight = torch.rand(128, device="cuda", dtype=torch.bfloat16)
        y, _ = higher_precision_rms_norm_op(x, weight, _EPS, True)
        # Expanded (stride-0) grad, as produced by e.g. a sum() loss.
        y.sum().backward()
        x_ref, _, y_ref = _reference(x, weight)
        y_ref.sum().backward()
        torch.testing.assert_close(y.double(), y_ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(x.grad.double(), x_ref.grad, atol=2e-2, rtol=2e-2)

    def test_deterministic(self):
        torch.manual_seed(0)
        x = torch.randn(1000, 512, device="cuda", dtype=torch.bfloat16)
        weight = torch.rand(512, device="cuda", dtype=torch.bfloat16)
        dy = torch.randn_like(x)
        _, rstd = rms_norm_forward_kernel(x, weight, _EPS)
        first = rms_norm_backward_kernel(dy, x, weight, rstd)
        second = rms_norm_backward_kernel(dy, x, weight, rstd)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))

    def test_module_matches_native_module(self):
        torch.manual_seed(0)
        native = RMSNorm.Config(normalized_shape=256).build().cuda()
        fused = (
            higher_precision_rms_norm(RMSNorm.Config(normalized_shape=256))
            .build()
            .cuda()
        )
        with torch.no_grad():
            native.weight.copy_(torch.rand(256) + 0.5)
        fused.load_state_dict(native.state_dict())
        x = torch.randn(4, 9, 256, device="cuda").requires_grad_()
        x2 = x.detach().clone().requires_grad_()
        native(x).sum().backward()
        fused(x2).sum().backward()
        torch.testing.assert_close(fused(x), native(x), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(x2.grad, x.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            fused.weight.grad, native.weight.grad, atol=1e-4, rtol=1e-4
        )

    def test_compiles_without_graph_breaks(self):
        fused = (
            higher_precision_rms_norm(RMSNorm.Config(normalized_shape=128))
            .build()
            .cuda()
        )
        x = torch.randn(8, 128, device="cuda", requires_grad=True)
        compiled = torch.compile(fused, fullgraph=True)
        torch.testing.assert_close(compiled(x), fused(x))
        compiled(x).sum().backward()

    def test_opcheck_forward_and_backward_ops(self):
        x = torch.randn(6, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = torch.rand(64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        torch.library.opcheck(higher_precision_rms_norm_op, (x, weight, _EPS, True))
        y, rstd = higher_precision_rms_norm_op(x.detach(), weight.detach(), _EPS, True)
        torch.library.opcheck(
            higher_precision_rms_norm_backward_op,
            (torch.randn_like(y), x.detach(), weight.detach(), rstd),
        )


class TestHigherPrecisionRMSNormDtypeOptions(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_unsupported_input_dtype_raises_value_error(self):
        x = torch.ones(4, 16, device="cuda", dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "Unsupported norm dtype"):
            rms_norm_forward_kernel(x, None, _EPS)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fp64_input_requires_fp64_accumulation(self):
        x = torch.randn(4, 16, device="cuda", dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "use_kahan_summation=False"):
            rms_norm_forward_kernel(x, None, _EPS, True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fp64_input_forward_and_backward_are_fp64(self):
        torch.manual_seed(0)
        x = torch.randn(64, 4096, device="cuda", dtype=torch.float64) * 3
        weight = torch.rand(4096, device="cuda", dtype=torch.float64) + 0.5
        x.requires_grad_()
        weight.requires_grad_()
        dy = torch.randn(64, 4096, device="cuda", dtype=torch.float64)

        y, rstd = higher_precision_rms_norm_op(x, weight, _EPS, False)
        y.backward(dy)

        x_ref, w_ref, y_ref = _reference(x, weight)
        y_ref.backward(dy)
        self.assertEqual(y.dtype, torch.float64)
        self.assertEqual(rstd.dtype, torch.float64)
        self.assertEqual(x.grad.dtype, torch.float64)
        self.assertEqual(weight.grad.dtype, torch.float64)
        torch.testing.assert_close(y, y_ref, atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(x.grad, x_ref.grad, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(weight.grad, w_ref.grad, atol=1e-11, rtol=1e-11)

    """``input_dtype`` / ``out_dtype`` with an FP32 (residual-stream) input."""

    def test_factory_takes_dtype_names(self):
        cfg = higher_precision_rms_norm(
            RMSNorm.Config(normalized_shape=8), input_dtype="bfloat16"
        )
        self.assertEqual(cfg.input_dtype, torch.bfloat16)
        self.assertIsNone(cfg.out_dtype)
        cfg = higher_precision_rms_norm(
            RMSNorm.Config(normalized_shape=8), out_dtype="float32"
        )
        self.assertEqual(cfg.out_dtype, torch.float32)
        cfg = higher_precision_rms_norm(
            RMSNorm.Config(normalized_shape=8),
            input_dtype="bfloat16",
            out_dtype="float32",
        )
        self.assertEqual(cfg.input_dtype, torch.bfloat16)
        self.assertEqual(cfg.out_dtype, torch.float32)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestHigherPrecisionRMSNormDtypeNumerics(unittest.TestCase):
    @parametrize("dim", [96, 4096, 12288])
    @parametrize("use_kahan", [False, True])
    def test_fp32_input_bf16_input_dtype_matches_explicit_cast(self, dim, use_kahan):
        """input_dtype=bf16 on FP32 x == casting x to BF16 first, minus the copy."""
        torch.manual_seed(0)
        x = (torch.randn(5, 33, dim, device="cuda") * 3).requires_grad_()
        weight = (torch.rand(dim, device="cuda") + 0.5).bfloat16().requires_grad_()
        dy = torch.randn(5, 33, dim, device="cuda").bfloat16()

        y, rstd = higher_precision_rms_norm_op(
            x, weight, _EPS, use_kahan, torch.bfloat16
        )
        y.backward(dy)

        x_bf16 = x.detach().bfloat16().requires_grad_()
        weight_explicit = weight.detach().clone().requires_grad_()
        y_explicit, rstd_explicit = higher_precision_rms_norm_op(
            x_bf16, weight_explicit, _EPS, use_kahan
        )
        y_explicit.backward(dy)

        self.assertEqual(y.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(y, y_explicit))
        self.assertTrue(torch.equal(rstd, rstd_explicit))
        # dx is FP32 (the original input's dtype); rounding it to BF16 reproduces
        # the explicit path's BF16 dx, and dw is bitwise equal.
        self.assertEqual(x.grad.dtype, torch.float32)
        self.assertTrue(torch.equal(x.grad.bfloat16(), x_bf16.grad))
        self.assertTrue(torch.equal(weight.grad, weight_explicit.grad))

    def test_input_dtype_matches_fp64_reference_on_rounded_input(self):
        torch.manual_seed(0)
        dim = 4096
        x = (torch.randn(64, dim, device="cuda") * 3).requires_grad_()
        weight = (torch.rand(dim, device="cuda") + 0.5).requires_grad_()
        dy = torch.randn(64, dim, device="cuda")

        y, _ = higher_precision_rms_norm_op(x, weight, _EPS, True, torch.bfloat16)
        y.backward(dy.bfloat16())

        # The norm treats the BF16-rounded input as its input; the rounding is a
        # straight-through identity for the gradient.
        x_ref, w_ref, y_ref = _reference(x.detach().bfloat16(), weight)
        y_ref.backward(dy.bfloat16().double())
        torch.testing.assert_close(y.double(), y_ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(x.grad.double(), x_ref.grad, atol=2e-2, rtol=2e-2)

    @parametrize("x_dtype", [torch.float32, torch.bfloat16])
    @parametrize("out_dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_out_dtype_leaves_input_unrounded(self, x_dtype, out_dtype):
        torch.manual_seed(0)
        dim = 4096
        x = (torch.randn(40, dim, device="cuda") * 3).to(x_dtype).requires_grad_()
        weight = (torch.rand(dim, device="cuda") + 0.5).requires_grad_()
        dy = torch.randn(40, dim, device="cuda").to(out_dtype)

        y, _ = higher_precision_rms_norm_op(x, weight, _EPS, True, None, out_dtype)
        y.backward(dy)

        x_ref, w_ref, y_ref = _reference(x, weight)
        y_ref.backward(dy.double())
        self.assertEqual(y.dtype, out_dtype)
        self.assertEqual(x.grad.dtype, x_dtype)
        # One rounding to out_dtype, no rounding of the input.
        tol = {torch.float32: 1e-5, torch.bfloat16: 2e-2, torch.float16: 3e-3}[
            out_dtype
        ]
        torch.testing.assert_close(y.double(), y_ref, atol=tol, rtol=tol)
        dx_tol = {torch.float32: 1e-5, torch.bfloat16: 2e-2}[x_dtype]
        torch.testing.assert_close(
            x.grad.double(), x_ref.grad, atol=dx_tol * 10, rtol=dx_tol * 10
        )

    def test_fp32_out_dtype_is_exact_to_fp32_rounding(self):
        """BF16 in, FP32 out: no BF16 rounding of the output at all."""
        torch.manual_seed(0)
        x = torch.randn(32, 4096, device="cuda").bfloat16()
        weight = (torch.rand(4096, device="cuda") + 0.5).bfloat16()
        y, _ = rms_norm_forward_kernel(x, weight, _EPS, True, None, torch.float32)
        y_ref = _reference(x, weight)[2].detach()
        torch.testing.assert_close(y.double(), y_ref, atol=1e-6, rtol=1e-6)

    def test_fp64_mode_forward_multiply_is_fp64(self):
        """FP64 mode keeps rstd in FP64 for the multiply, so FP32 output is within
        ~1 FP32 ulp of the FP64 reference."""
        torch.manual_seed(0)
        x = torch.randn(256, 4096, device="cuda") * 3
        weight = torch.rand(4096, device="cuda") + 0.5
        y, _ = rms_norm_forward_kernel(x, weight, _EPS, False, None, torch.float32)
        y_ref = _reference(x, weight)[2].detach()
        rel_err = ((y.double() - y_ref).abs() / y_ref.abs()).max().item()
        self.assertLess(rel_err, 1.5e-7)

    def test_input_dtype_and_out_dtype_are_independent(self):
        """FP32 x, input rounded to BF16, output emitted in FP32."""
        torch.manual_seed(0)
        x = (torch.randn(32, 4096, device="cuda") * 3).requires_grad_()
        weight = (torch.rand(4096, device="cuda") + 0.5).requires_grad_()
        dy = torch.randn(32, 4096, device="cuda")

        y, _ = higher_precision_rms_norm_op(
            x, weight, _EPS, True, torch.bfloat16, torch.float32
        )
        y.backward(dy)

        # The norm sees the BF16-rounded input, but the output is not rounded to
        # BF16.
        _, _, y_ref = _reference(x.detach().bfloat16(), weight)
        self.assertEqual(y.dtype, torch.float32)
        self.assertEqual(x.grad.dtype, torch.float32)
        torch.testing.assert_close(y.double(), y_ref, atol=1e-5, rtol=1e-5)

    def test_module_with_dtype_options_compiles_and_matches_op(self):
        for kwargs in (
            {"input_dtype": "bfloat16"},
            {"out_dtype": "float32"},
            {"input_dtype": "bfloat16", "out_dtype": "float32"},
        ):
            fused = (
                higher_precision_rms_norm(
                    RMSNorm.Config(normalized_shape=128), **kwargs
                )
                .build()
                .cuda()
            )
            x = torch.randn(8, 128, device="cuda", requires_grad=True)
            compiled = torch.compile(fused, fullgraph=True)
            torch.testing.assert_close(compiled(x), fused(x))
            compiled(x).sum().backward()
            self.assertEqual(x.grad.dtype, torch.float32)

    def test_opcheck_with_dtype_options(self):
        x = torch.randn(6, 64, device="cuda", requires_grad=True)
        weight = torch.rand(64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        torch.library.opcheck(
            higher_precision_rms_norm_op, (x, weight, _EPS, True, torch.bfloat16, None)
        )
        torch.library.opcheck(
            higher_precision_rms_norm_op, (x, weight, _EPS, True, None, torch.bfloat16)
        )
        torch.library.opcheck(
            higher_precision_rms_norm_op,
            (x, weight, _EPS, True, torch.bfloat16, torch.float32),
        )
        y, rstd = higher_precision_rms_norm_op(
            x.detach(), weight.detach(), _EPS, True, torch.bfloat16
        )
        torch.library.opcheck(
            higher_precision_rms_norm_backward_op,
            (torch.randn_like(y), x.detach(), weight.detach(), rstd, torch.bfloat16),
        )


instantiate_parametrized_tests(TestHigherPrecisionRMSNormNumerics)
instantiate_parametrized_tests(TestHigherPrecisionRMSNormDtypeNumerics)

if __name__ == "__main__":
    unittest.main()
