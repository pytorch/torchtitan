# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from collections.abc import Callable

import torch
import torch.nn.functional as F

import torchtitan.distributed.compile as compile_mod
from torchtitan.overrides.compiled_gated_rmsnorm import CompiledGatedRMSNorm


_EPS = 1e-5
_FUDGE_FACTOR = 4.0
_PROJECT_ATOL = 0.0
_MAX_REFERENCE_RELATIVE_ERROR = 0.1


def _kimi_gated_rms_norm_reference(
    input: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    activation_fn: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
) -> torch.Tensor:
    input_dtype = input.dtype
    normalized = F.rms_norm(
        input.float(),
        (input.shape[-1],),
        weight.float(),
        _EPS,
    )
    return (normalized * activation_fn(gate.float())).to(input_dtype)


def _kimi_gated_rms_norm_golden(
    input: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    input_fp64 = input.double()
    inverse_rms = torch.rsqrt(input_fp64.square().mean(-1, keepdim=True) + _EPS)
    return input_fp64 * inverse_rms * weight.double() * gate.double().sigmoid()


def _max_abs(tensor: torch.Tensor) -> float:
    return tensor.abs().max().item() if tensor.numel() else 0.0


def _assert_matches_golden(
    testcase: unittest.TestCase,
    *,
    name: str,
    golden: torch.Tensor,
    reference: torch.Tensor,
    target: torch.Tensor,
) -> None:
    testcase.assertEqual(target.shape, reference.shape, name)
    testcase.assertEqual(target.dtype, reference.dtype, name)
    testcase.assertTrue(torch.equal(torch.isnan(target), torch.isnan(reference)), name)
    testcase.assertTrue(
        torch.equal(torch.isposinf(target), torch.isposinf(reference)), name
    )
    testcase.assertTrue(
        torch.equal(torch.isneginf(target), torch.isneginf(reference)), name
    )

    golden_fp64 = golden.double()
    reference_fp64 = reference.double()
    target_fp64 = target.double()
    reference_error = _max_abs(reference_fp64 - golden_fp64)
    target_error = _max_abs(target_fp64 - golden_fp64)
    rounding_floor = _max_abs(golden_fp64.to(target.dtype).double() - golden_fp64)
    absolute_floor = max(_PROJECT_ATOL, rounding_floor)
    threshold = _FUDGE_FACTOR * reference_error + absolute_floor
    golden_scale = _max_abs(golden_fp64)
    reference_relative_error = reference_error / max(
        golden_scale,
        absolute_floor,
        torch.finfo(torch.float64).tiny,
    )
    testcase.assertLessEqual(
        reference_relative_error,
        _MAX_REFERENCE_RELATIVE_ERROR,
        f"{name}: reference is too inaccurate to gate the target",
    )
    testcase.assertLessEqual(
        target_error,
        threshold,
        (
            f"{name}: target_error={target_error:.6e}, "
            f"reference_error={reference_error:.6e}, "
            f"rounding_floor={rounding_floor:.6e}, threshold={threshold:.6e}"
        ),
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCompiledGatedRMSNormNumerics(unittest.TestCase):
    def _run_case(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        seed: int,
        *,
        input_scale: float = 1.0,
        gate_scale: float = 1.0,
    ) -> None:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        input_data = (
            torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
            * input_scale
        )
        gate_data = (
            torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
            * gate_scale
        )
        weight_data = torch.randn(
            shape[-1],
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        grad_output_data = torch.randn(
            shape,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )

        golden_input = input_data.double().requires_grad_()
        golden_gate = gate_data.double().requires_grad_()
        golden_weight = weight_data.double().requires_grad_()
        golden_output = _kimi_gated_rms_norm_golden(
            golden_input,
            golden_gate,
            golden_weight,
        )
        golden_grads = torch.autograd.grad(
            golden_output,
            (golden_input, golden_gate, golden_weight),
            grad_output_data.double(),
        )

        reference_input = input_data.detach().clone().requires_grad_()
        reference_gate = gate_data.detach().clone().requires_grad_()
        reference_weight = weight_data.detach().clone().requires_grad_()
        reference_output = _kimi_gated_rms_norm_reference(
            reference_input,
            reference_gate,
            reference_weight,
        )
        reference_grads = torch.autograd.grad(
            reference_output,
            (reference_input, reference_gate, reference_weight),
            grad_output_data,
        )

        target = (
            CompiledGatedRMSNorm(CompiledGatedRMSNorm.Config(dim=shape[-1], eps=_EPS))
            .cuda()
            .to(dtype)
        )
        with torch.no_grad():
            target.weight.copy_(weight_data)
        target_input = input_data.detach().clone().requires_grad_()
        target_gate = gate_data.detach().clone().requires_grad_()
        target_output = target(target_input, target_gate)
        target_grads = torch.autograd.grad(
            target_output,
            (target_input, target_gate, target.weight),
            grad_output_data,
        )

        _assert_matches_golden(
            self,
            name="output",
            golden=golden_output,
            reference=reference_output,
            target=target_output,
        )
        for name, golden_grad, reference_grad, target_grad in zip(
            ("grad_input", "grad_gate", "grad_weight"),
            golden_grads,
            reference_grads,
            target_grads,
        ):
            _assert_matches_golden(
                self,
                name=name,
                golden=golden_grad,
                reference=reference_grad,
                target=target_grad,
            )

    def test_forward_and_backward_against_golden(self):
        cases = (
            ((3, 128), torch.float32, 1),
            ((8, 6, 128), torch.bfloat16, 2),
            ((3, 128), torch.float16, 3),
        )
        for shape, dtype, seed in cases:
            with self.subTest(shape=shape, dtype=dtype, seed=seed):
                self._run_case(shape, dtype, seed)

    def test_near_zero_variance(self):
        self._run_case((8, 6, 128), torch.bfloat16, 4, input_scale=1e-5)

    def test_saturated_gate(self):
        self._run_case((8, 6, 128), torch.bfloat16, 5, gate_scale=20.0)

    def test_configurable_unary_activation(self):
        shape = (8, 6, 128)
        generator = torch.Generator(device="cuda").manual_seed(6)
        input_data = torch.randn(
            shape, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        gate_data = torch.randn(
            shape, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        weight_data = torch.randn(
            shape[-1], device="cuda", dtype=torch.bfloat16, generator=generator
        )
        grad_output = torch.randn(
            shape, device="cuda", dtype=torch.bfloat16, generator=generator
        )

        for activation_fn in (F.silu, F.relu):
            with self.subTest(activation_fn=activation_fn.__name__):
                reference_input = input_data.detach().clone().requires_grad_()
                reference_gate = gate_data.detach().clone().requires_grad_()
                reference_weight = weight_data.detach().clone().requires_grad_()
                reference_output = _kimi_gated_rms_norm_reference(
                    reference_input,
                    reference_gate,
                    reference_weight,
                    activation_fn,
                )
                reference_grads = torch.autograd.grad(
                    reference_output,
                    (reference_input, reference_gate, reference_weight),
                    grad_output,
                )

                target = (
                    CompiledGatedRMSNorm(
                        CompiledGatedRMSNorm.Config(
                            dim=shape[-1],
                            eps=_EPS,
                            activation_fn=activation_fn,
                        )
                    )
                    .cuda()
                    .to(torch.bfloat16)
                )
                with torch.no_grad():
                    target.weight.copy_(weight_data)
                target_input = input_data.detach().clone().requires_grad_()
                target_gate = gate_data.detach().clone().requires_grad_()
                target_output = target(target_input, target_gate)
                target_grads = torch.autograd.grad(
                    target_output,
                    (target_input, target_gate, target.weight),
                    grad_output,
                )

                torch.testing.assert_close(target_output, reference_output)
                for target_grad, reference_grad in zip(
                    target_grads,
                    reference_grads,
                ):
                    torch.testing.assert_close(target_grad, reference_grad)

    def test_bitwise_deterministic(self):
        shape = (8, 6, 128)
        generator = torch.Generator(device="cuda").manual_seed(6)
        input_data = torch.randn(
            shape, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        gate_data = torch.randn(
            shape, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        grad_output = torch.randn(
            shape, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        target = (
            CompiledGatedRMSNorm(CompiledGatedRMSNorm.Config(dim=shape[-1], eps=_EPS))
            .cuda()
            .to(torch.bfloat16)
        )

        results = []
        for _ in range(3):
            input = input_data.detach().clone().requires_grad_()
            gate = gate_data.detach().clone().requires_grad_()
            output = target(input, gate)
            grads = torch.autograd.grad(
                output,
                (input, gate, target.weight),
                grad_output,
            )
            results.append((output, *grads))

        for result in results[1:]:
            for actual, expected in zip(result, results[0]):
                self.assertTrue(torch.equal(actual, expected))

    def test_aot_eager_regional_inductor_codegen(self):
        from torch._dynamo.backends.common import aot_autograd
        from torch._inductor.utils import run_fw_bw_and_get_code
        from torch.fx.passes.regional_inductor import regional_inductor

        torch._dynamo.reset()
        compile_mod._regional_inductor_enabled = True
        try:
            target = (
                CompiledGatedRMSNorm(CompiledGatedRMSNorm.Config(dim=128, eps=_EPS))
                .cuda()
                .to(torch.bfloat16)
            )
            input = torch.randn(
                8,
                6,
                128,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            gate = torch.randn_like(input, requires_grad=True)
            backend = aot_autograd(
                fw_compiler=regional_inductor,
                bw_compiler=regional_inductor,
            )
            compiled = torch.compile(target, backend=backend, fullgraph=True)
            _result, codes = run_fw_bw_and_get_code(lambda: compiled(input, gate))
        finally:
            compile_mod._regional_inductor_enabled = False
            torch._dynamo.reset()

        self.assertGreaterEqual(sum("triton" in code for code in codes), 2)
        self.assertTrue(any("sigmoid" in code and "rsqrt" in code for code in codes))


if __name__ == "__main__":
    unittest.main()
