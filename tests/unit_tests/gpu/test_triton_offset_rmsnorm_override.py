# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.overrides.offset_rmsnorm import (
    _triton_offset_rms_norm_backward_op,
    _triton_offset_rms_norm_op,
    triton_offset_rms_norm,
)


_EPS = 1e-6
_FUDGE_FACTOR = 2.0
_PROJECT_ATOL = 0.0
_MAX_REFERENCE_RELATIVE_ERROR = 0.1


def _offset_rms_norm_reference(
    input: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    input_dtype = input.dtype
    input_fp32 = input.float()
    inverse_rms = torch.rsqrt(input_fp32.square().mean(-1, keepdim=True) + _EPS)
    return ((1.0 + weight.float()) * input_fp32 * inverse_rms).to(input_dtype)


def _offset_rms_norm_golden(
    input: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    input_fp64 = input.double()
    inverse_rms = torch.rsqrt(input_fp64.square().mean(-1, keepdim=True) + _EPS)
    return (1.0 + weight.double()) * input_fp64 * inverse_rms


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
class TestTritonOffsetRMSNormNumerics(unittest.TestCase):
    def _run_case(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        seed: int,
        *,
        scale: float = 1.0,
    ) -> None:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        input_data = (
            torch.randn(shape, device="cuda", dtype=dtype, generator=generator) * scale
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
        golden_weight = weight_data.double().requires_grad_()
        golden_output = _offset_rms_norm_golden(golden_input, golden_weight)
        golden_grads = torch.autograd.grad(
            golden_output,
            (golden_input, golden_weight),
            grad_output_data.double(),
        )

        reference_input = input_data.detach().clone().requires_grad_()
        reference_weight = weight_data.detach().clone().requires_grad_()
        reference_output = _offset_rms_norm_reference(
            reference_input,
            reference_weight,
        )
        reference_grads = torch.autograd.grad(
            reference_output,
            (reference_input, reference_weight),
            grad_output_data,
        )

        target_input = input_data.detach().clone().requires_grad_()
        target_weight = weight_data.detach().clone().requires_grad_()
        target_output = triton_offset_rms_norm(
            target_input,
            target_weight,
            _EPS,
        )
        target_grads = torch.autograd.grad(
            target_output,
            (target_input, target_weight),
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
            ("grad_input", "grad_weight"),
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
            ((3, 255), torch.float32, 1),
            ((3, 256), torch.bfloat16, 2),
            ((3, 257), torch.bfloat16, 3),
            ((3, 257), torch.float16, 10),
            ((8, 6, 256), torch.bfloat16, 4),
            ((8, 4095), torch.bfloat16, 5),
            ((8, 4096), torch.bfloat16, 6),
            ((8, 4097), torch.bfloat16, 7),
            ((8, 5120), torch.bfloat16, 8),
        )
        for shape, dtype, seed in cases:
            with self.subTest(shape=shape, dtype=dtype, seed=seed):
                self._run_case(shape, dtype, seed)

    def test_near_zero_variance(self):
        self._run_case((8, 5120), torch.bfloat16, 9, scale=1e-5)

    def test_zero_variance(self):
        self._run_case((8, 5120), torch.bfloat16, 11, scale=0.0)

    def test_custom_op_contract(self):
        input = torch.randn(
            8,
            256,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        weight = torch.randn(
            256,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        torch.library.opcheck(
            _triton_offset_rms_norm_op,
            (input, weight, _EPS),
            test_utils=(
                "test_schema",
                "test_faketensor",
                "test_autograd_registration",
            ),
        )
        output, inverse_rms = _triton_offset_rms_norm_op(input, weight, _EPS)
        torch.library.opcheck(
            _triton_offset_rms_norm_backward_op,
            (torch.randn_like(output), input, weight, inverse_rms),
            test_utils=("test_schema", "test_faketensor"),
        )

    def test_torch_compile(self):
        input = torch.randn(
            8,
            5120,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        weight = torch.randn(
            5120,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        grad_output = torch.randn_like(input)
        compiled = torch.compile(triton_offset_rms_norm, fullgraph=True)

        expected = triton_offset_rms_norm(input, weight, _EPS)
        expected_grads = torch.autograd.grad(
            expected,
            (input, weight),
            grad_output,
        )

        compiled_input = input.detach().clone().requires_grad_()
        compiled_weight = weight.detach().clone().requires_grad_()
        actual = compiled(compiled_input, compiled_weight, _EPS)
        actual_grads = torch.autograd.grad(
            actual,
            (compiled_input, compiled_weight),
            grad_output,
        )

        torch.testing.assert_close(actual, expected)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)


if __name__ == "__main__":
    unittest.main()
