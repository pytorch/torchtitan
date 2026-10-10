# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.common import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.kimi_k3.model import AttentionResidual


def _build_modules(dim: int) -> tuple[Linear, RMSNorm]:
    projection = Linear.Config(in_features=dim, out_features=1).build()
    norm = RMSNorm.Config(normalized_shape=dim).build()
    return projection.cuda().bfloat16(), norm.cuda().bfloat16()


def _reference(
    partial_block_TD: torch.Tensor | None,
    block_residual_TND: torch.Tensor,
    projection_weight_1D: torch.Tensor,
    norm_weight_D: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    values_TND = (
        block_residual_TND
        if partial_block_TD is None
        else torch.cat((block_residual_TND, partial_block_TD.unsqueeze(1)), dim=1)
    )
    values_float_TND = values_TND.float()
    variance_TN1 = values_float_TND.pow(2).mean(dim=-1, keepdim=True)
    keys_TND = values_float_TND * torch.rsqrt(variance_TN1 + eps)
    score_weight_D = norm_weight_D.float() * projection_weight_1D.squeeze(0).float()
    scores_TN = (keys_TND * score_weight_D).sum(dim=-1)
    probs_T1N = torch.softmax(scores_TN, dim=-1).unsqueeze(1)
    output_TD = torch.matmul(probs_T1N, values_float_TND).squeeze(1)
    return output_TD.to(values_TND.dtype)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestKimiK3AttentionResidualLocalCompile(unittest.TestCase):
    def setUp(self):
        apply_local_compile(["attention_residual"])
        self.attention_residual = AttentionResidual.Config().build()

    def tearDown(self):
        apply_local_compile([])
        torch._dynamo.reset()

    def test_forward_and_backward_emit_triton(self):
        from torch._inductor.utils import run_fw_bw_and_get_code

        projection, norm = _build_modules(128)
        partial_block_TD = torch.randn(
            64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        block_residual_TND = torch.randn(
            64, 3, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

        _, codes = run_fw_bw_and_get_code(
            lambda: self.attention_residual(
                partial_block_TD,
                block_residual_TND,
                projection,
                norm,
            )
        )

        self.assertGreaterEqual(sum("triton" in code for code in codes), 2)

    def test_forward_and_backward_match_reference(self):
        torch.manual_seed(42)
        for has_partial_block in (False, True):
            with self.subTest(has_partial_block=has_partial_block):
                projection, norm = _build_modules(128)
                partial_block_TD = (
                    torch.randn(
                        64,
                        128,
                        device="cuda",
                        dtype=torch.bfloat16,
                        requires_grad=True,
                    )
                    if has_partial_block
                    else None
                )
                block_residual_TND = torch.randn(
                    64,
                    3,
                    128,
                    device="cuda",
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                grad_output_TD = torch.randn_like(block_residual_TND[:, 0])

                output_TD = self.attention_residual(
                    partial_block_TD,
                    block_residual_TND,
                    projection,
                    norm,
                )
                compiled_inputs = (
                    (partial_block_TD,) if partial_block_TD is not None else ()
                ) + (block_residual_TND, projection.weight, norm.weight)
                compiled_grads = torch.autograd.grad(
                    output_TD,
                    compiled_inputs,
                    grad_output_TD,
                )

                reference_partial_block_TD = (
                    partial_block_TD.detach().clone().requires_grad_()
                    if partial_block_TD is not None
                    else None
                )
                reference_block_residual_TND = (
                    block_residual_TND.detach().clone().requires_grad_()
                )
                reference_projection_weight_1D = (
                    projection.weight.detach().clone().requires_grad_()
                )
                reference_norm_weight_D = norm.weight.detach().clone().requires_grad_()
                assert norm.eps is not None
                reference_output_TD = _reference(
                    reference_partial_block_TD,
                    reference_block_residual_TND,
                    reference_projection_weight_1D,
                    reference_norm_weight_D,
                    norm.eps,
                )
                reference_inputs = (
                    (reference_partial_block_TD,)
                    if reference_partial_block_TD is not None
                    else ()
                ) + (
                    reference_block_residual_TND,
                    reference_projection_weight_1D,
                    reference_norm_weight_D,
                )
                reference_grads = torch.autograd.grad(
                    reference_output_TD,
                    reference_inputs,
                    grad_output_TD,
                )

                torch.testing.assert_close(output_TD, reference_output_TD)
                for compiled_grad, reference_grad in zip(
                    compiled_grads, reference_grads, strict=True
                ):
                    torch.testing.assert_close(compiled_grad, reference_grad)

    def test_forward_and_backward_are_batch_invariant(self):
        torch.manual_seed(42)
        projection, norm = _build_modules(128)
        partial_block_TD = torch.randn(
            64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        block_residual_TND = torch.randn(
            64, 3, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        grad_output_TD = torch.randn_like(partial_block_TD)

        def run(
            partial_block_TD: torch.Tensor,
            block_residual_TND: torch.Tensor,
            grad_output_TD: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            output_TD = self.attention_residual(
                partial_block_TD,
                block_residual_TND,
                projection,
                norm,
            )
            grad_partial_TD, grad_residual_TND = torch.autograd.grad(
                output_TD,
                (partial_block_TD, block_residual_TND),
                grad_output_TD,
            )
            return output_TD, grad_partial_TD, grad_residual_TND

        full_outputs = run(
            partial_block_TD,
            block_residual_TND,
            grad_output_TD,
        )
        split_outputs = [[], [], []]
        for inputs in zip(
            partial_block_TD.detach().chunk(2),
            block_residual_TND.detach().chunk(2),
            grad_output_TD.chunk(2),
            strict=True,
        ):
            partial_part_TD, residual_part_TND, grad_part_TD = inputs
            partial_part_TD.requires_grad_()
            residual_part_TND.requires_grad_()
            for outputs, output in zip(
                split_outputs,
                run(partial_part_TD, residual_part_TND, grad_part_TD),
                strict=True,
            ):
                outputs.append(output)

        for full_output, split_output in zip(full_outputs, split_outputs, strict=True):
            self.assertTrue(torch.equal(full_output, torch.cat(split_output)))


if __name__ == "__main__":
    unittest.main()
