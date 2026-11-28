# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import unittest

import torch

from torchtitan_recipes.overrides.kimi_attention_residual import (
    _attention_residual_forward_op,
    attention_residual_op,
)


_FUDGE_FACTOR = 2.0
_MAX_REFERENCE_RELATIVE_ERROR = 0.1


def _attention_residual_reference(
    partial_block_TD: torch.Tensor | None,
    block_residual_TND: torch.Tensor,
    projection_weight_1D: torch.Tensor,
    norm_weight_D: torch.Tensor,
    eps: float,
    *,
    golden: bool,
) -> torch.Tensor:
    values_TND = (
        block_residual_TND
        if partial_block_TD is None
        else torch.cat(
            (block_residual_TND, partial_block_TD.unsqueeze(1)),
            dim=1,
        )
    )
    values_compute_TND = values_TND if golden else values_TND.float()
    projection_compute_1D = (
        projection_weight_1D if golden else projection_weight_1D.float()
    )
    norm_compute_D = norm_weight_D if golden else norm_weight_D.float()
    variance_TN1 = values_compute_TND.square().mean(dim=-1, keepdim=True)
    keys_TND = values_compute_TND * torch.rsqrt(variance_TN1 + eps)
    score_weight_D = norm_compute_D * projection_compute_1D.squeeze(0)
    scores_TN = (keys_TND * score_weight_D).sum(dim=-1)
    probs_T1N = torch.softmax(scores_TN, dim=-1).unsqueeze(1)
    output_TD = torch.matmul(probs_T1N, values_compute_TND).squeeze(1)
    return output_TD if golden else output_TD.to(values_TND.dtype)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestKimiAttentionResidualOverride(unittest.TestCase):
    def _assert_numerics(
        self,
        golden: torch.Tensor,
        reference: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        golden = golden.detach()
        reference_dtype = reference.dtype
        reference = reference.detach().double()
        target_double = target.detach().double()
        self.assertEqual(target.shape, golden.shape)
        self.assertEqual(target.dtype, reference_dtype)
        self.assertTrue(torch.equal(torch.isnan(target_double), torch.isnan(golden)))
        self.assertTrue(
            torch.equal(torch.isposinf(target_double), torch.isposinf(golden))
        )
        self.assertTrue(
            torch.equal(torch.isneginf(target_double), torch.isneginf(golden))
        )

        reference_error = (reference - golden).abs().max().item()
        target_error = (target_double - golden).abs().max().item()
        rounding_floor = (golden.to(target.dtype).double() - golden).abs().max().item()
        threshold = _FUDGE_FACTOR * reference_error + rounding_floor
        golden_scale = golden.abs().max().item()
        reference_relative_error = reference_error / max(
            golden_scale,
            rounding_floor,
            torch.finfo(torch.float64).tiny,
        )
        self.assertLessEqual(
            reference_relative_error,
            _MAX_REFERENCE_RELATIVE_ERROR,
        )
        self.assertLessEqual(target_error, threshold)

    def _run_numerics_case(
        self,
        shape: tuple[int, int, int],
        *,
        has_partial: bool,
        dtype: torch.dtype,
        input_scale: float = 1.0,
    ) -> None:
        num_tokens, _, dim = shape
        torch.manual_seed(num_tokens * 100_000 + shape[1] * 1_000 + dim)
        block_residual_TND = (torch.randn(*shape, device="cuda") * input_scale).to(
            dtype
        )
        partial_block_TD = (
            (torch.randn(num_tokens, dim, device="cuda") * input_scale).to(dtype)
            if has_partial
            else None
        )
        projection_weight_1D = (torch.randn(1, dim, device="cuda") / math.sqrt(dim)).to(
            dtype
        )
        norm_weight_D = (1 + 0.1 * torch.randn(dim, device="cuda")).to(dtype)
        grad_output_TD = torch.randn(
            num_tokens,
            dim,
            device="cuda",
            dtype=dtype,
        )
        low_precision_values = (
            [partial_block_TD] if partial_block_TD is not None else []
        ) + [
            block_residual_TND,
            projection_weight_1D,
            norm_weight_D,
        ]

        def leaves(values: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
            return tuple(value.detach().clone().requires_grad_() for value in values)

        target_inputs = leaves(low_precision_values)
        target_partial_TD = target_inputs[0] if has_partial else None
        target_offset = int(has_partial)
        target_output_TD = attention_residual_op(
            target_partial_TD,
            target_inputs[target_offset],
            target_inputs[target_offset + 1],
            target_inputs[target_offset + 2],
            1e-5,
        )
        target_grads = torch.autograd.grad(
            target_output_TD,
            target_inputs,
            grad_output_TD,
        )

        reference_inputs = leaves(low_precision_values)
        reference_partial_TD = reference_inputs[0] if has_partial else None
        reference_offset = int(has_partial)
        reference_output_TD = _attention_residual_reference(
            reference_partial_TD,
            reference_inputs[reference_offset],
            reference_inputs[reference_offset + 1],
            reference_inputs[reference_offset + 2],
            1e-5,
            golden=False,
        )
        reference_grads = torch.autograd.grad(
            reference_output_TD,
            reference_inputs,
            grad_output_TD,
        )

        golden_inputs = leaves([value.double() for value in low_precision_values])
        golden_partial_TD = golden_inputs[0] if has_partial else None
        golden_offset = int(has_partial)
        golden_output_TD = _attention_residual_reference(
            golden_partial_TD,
            golden_inputs[golden_offset],
            golden_inputs[golden_offset + 1],
            golden_inputs[golden_offset + 2],
            1e-5,
            golden=True,
        )
        golden_grads = torch.autograd.grad(
            golden_output_TD,
            golden_inputs,
            grad_output_TD.double(),
        )

        self._assert_numerics(
            golden_output_TD,
            reference_output_TD,
            target_output_TD,
        )
        for golden_grad, reference_grad, target_grad in zip(
            golden_grads,
            reference_grads,
            target_grads,
            strict=True,
        ):
            self._assert_numerics(
                golden_grad,
                reference_grad,
                target_grad,
            )

    def test_opcheck_with_and_without_partial_residual(self) -> None:
        for has_partial in (False, True):
            with self.subTest(has_partial=has_partial):
                dim = 256
                partial_block_TD = (
                    torch.randn(
                        4,
                        dim,
                        device="cuda",
                        dtype=torch.bfloat16,
                        requires_grad=True,
                    )
                    if has_partial
                    else torch.empty(
                        0,
                        dim,
                        device="cuda",
                        dtype=torch.bfloat16,
                        requires_grad=True,
                    )
                )
                torch.library.opcheck(
                    _attention_residual_forward_op,
                    (
                        partial_block_TD,
                        torch.randn(
                            4,
                            5,
                            dim,
                            device="cuda",
                            dtype=torch.bfloat16,
                            requires_grad=True,
                        ),
                        torch.randn(
                            1,
                            dim,
                            device="cuda",
                            dtype=torch.bfloat16,
                            requires_grad=True,
                        ),
                        torch.randn(
                            dim,
                            device="cuda",
                            dtype=torch.bfloat16,
                            requires_grad=True,
                        ),
                        1e-5,
                    ),
                )

    def test_forward_and_backward_numerics(self) -> None:
        cases = (
            ((1, 1, 1), False, torch.bfloat16, 1.0),
            ((3, 7, 255), True, torch.bfloat16, 1.0),
            ((4, 9, 257), True, torch.bfloat16, 1.0),
            ((2, 8, 7168), True, torch.bfloat16, 1.0),
            ((4, 5, 256), True, torch.float16, 1.0),
            ((4, 5, 256), True, torch.bfloat16, 1e-3),
        )
        for shape, has_partial, dtype, input_scale in cases:
            with self.subTest(
                shape=shape,
                has_partial=has_partial,
                dtype=dtype,
                input_scale=input_scale,
            ):
                self._run_numerics_case(
                    shape,
                    has_partial=has_partial,
                    dtype=dtype,
                    input_scale=input_scale,
                )


if __name__ == "__main__":
    unittest.main()
