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
    sources_TD: list[torch.Tensor],
    projection_weight_1D: torch.Tensor,
    norm_weight_D: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    values_TND = torch.stack(sources_TD, dim=1)
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
        sources_TD = [
            torch.randn(
                64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            for _ in range(4)
        ]

        _, codes = run_fw_bw_and_get_code(
            lambda: self.attention_residual(sources_TD, projection, norm)
        )

        self.assertGreaterEqual(sum("triton" in code for code in codes), 2)

    def test_forward_and_backward_match_reference(self):
        # Ten source counts, more graphs than dynamo's default recompile limit.
        torch.manual_seed(42)
        projection, norm = _build_modules(128)
        assert norm.eps is not None
        for num_sources in range(1, 11):
            with self.subTest(num_sources=num_sources):
                sources_TD = [
                    torch.randn(
                        64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
                    )
                    for _ in range(num_sources)
                ]
                grad_output_TD = torch.randn_like(sources_TD[0])

                output_TD = self.attention_residual(sources_TD, projection, norm)
                compiled_inputs = (*sources_TD, projection.weight, norm.weight)
                compiled_grads = torch.autograd.grad(
                    output_TD, compiled_inputs, grad_output_TD
                )

                reference_inputs = tuple(
                    value.detach().clone().requires_grad_() for value in compiled_inputs
                )
                reference_output_TD = _reference(
                    list(reference_inputs[:num_sources]),
                    reference_inputs[num_sources],
                    reference_inputs[num_sources + 1],
                    norm.eps,
                )
                reference_grads = torch.autograd.grad(
                    reference_output_TD, reference_inputs, grad_output_TD
                )

                torch.testing.assert_close(output_TD, reference_output_TD)
                for compiled_grad, reference_grad in zip(
                    compiled_grads, reference_grads, strict=True
                ):
                    torch.testing.assert_close(compiled_grad, reference_grad)

    def test_forward_and_backward_are_batch_invariant(self):
        torch.manual_seed(42)
        projection, norm = _build_modules(128)
        partial_block_TD = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        block_residual_TND = torch.randn(
            64, 3, 128, device="cuda", dtype=torch.bfloat16
        )
        grad_output_TD = torch.randn_like(partial_block_TD)
        sources_TD = [
            source.contiguous().requires_grad_()
            for source in (*block_residual_TND.unbind(1), partial_block_TD)
        ]

        def run(
            sources_TD: list[torch.Tensor], grad_output_TD: torch.Tensor
        ) -> tuple[torch.Tensor, ...]:
            output_TD = self.attention_residual(sources_TD, projection, norm)
            return output_TD, *torch.autograd.grad(
                output_TD, sources_TD, grad_output_TD
            )

        full_outputs = run(sources_TD, grad_output_TD)
        split_outputs = [[] for _ in full_outputs]
        for part in range(2):
            sources_part_TD = [
                source.detach().chunk(2)[part].requires_grad_() for source in sources_TD
            ]
            for outputs, output in zip(
                split_outputs,
                run(sources_part_TD, grad_output_TD.chunk(2)[part]),
                strict=True,
            ):
                outputs.append(output)

        for full_output, split_output in zip(full_outputs, split_outputs, strict=True):
            self.assertTrue(torch.equal(full_output, torch.cat(split_output)))


if __name__ == "__main__":
    unittest.main()
