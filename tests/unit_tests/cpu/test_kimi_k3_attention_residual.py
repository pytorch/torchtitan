# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import types
import unittest

import torch

from torchtitan.models.kimi_k3.model import (
    _apply_attention_residual,
    _AttentionResidualAggregation,
)

EPS = 1e-5
TOKENS, BLOCKS, DIM = 16, 3, 8
WIDE_DIM = 1024


def _reference(
    prefix_sum_TD: torch.Tensor,
    block_residual_TND: torch.Tensor,
    projection_1D: torch.Tensor,
    norm_weight_D: torch.Tensor,
) -> torch.Tensor:
    """The same aggregation written as plain autograd ops."""
    values_TND = torch.cat((block_residual_TND, prefix_sum_TD.unsqueeze(1)), dim=1)
    values_float = values_TND.float()
    variance = values_float.pow(2).mean(dim=-1, keepdim=True)
    keys_TND = values_float * torch.rsqrt(variance + EPS)
    score_weight_D = norm_weight_D.float() * projection_1D.squeeze(0).float()
    scores_TN = (keys_TND * score_weight_D).sum(dim=-1)
    probs_T1N = torch.softmax(scores_TN, dim=-1).unsqueeze(1)
    return torch.matmul(probs_T1N, values_float).squeeze(1).to(values_TND.dtype)


def _inputs(blocks: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)

    def make(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator).requires_grad_()

    return (
        make(TOKENS, DIM),
        make(TOKENS, blocks, DIM),
        make(1, DIM),
        make(DIM),
        torch.randn(TOKENS, DIM, generator=generator),
    )


class TestKimiK3AttentionResidual(unittest.TestCase):
    def _compare(self, blocks: int) -> None:
        prefix, stack, projection, norm_weight, grad_output = _inputs(blocks)
        leaves = [prefix, stack, projection, norm_weight]
        outputs, grads = [], []
        for aggregate in (
            lambda: _reference(prefix, stack, projection, norm_weight),
            lambda: _apply_attention_residual(
                prefix,
                stack,
                types.SimpleNamespace(weight=projection),
                types.SimpleNamespace(weight=norm_weight, eps=EPS),
            ),
        ):
            for leaf in leaves:
                leaf.grad = None
            aggregate().backward(grad_output)
            outputs.append(aggregate().detach())
            grads.append([leaf.grad.detach().clone() for leaf in leaves])
        torch.testing.assert_close(outputs[0], outputs[1])
        for expected, actual in zip(*grads):
            torch.testing.assert_close(expected, actual)

    def test_matches_plain_autograd(self):
        for blocks in (1, BLOCKS):
            with self.subTest(blocks=blocks):
                self._compare(blocks)

    def test_bf16_weights_keep_fp32_score_precision(self):
        # The score weight is a product of two parameters. Rounding that
        # product to bfloat16 before the upcast, rather than upcasting both
        # factors first, moves the output by half a percent.
        generator = torch.Generator().manual_seed(0)

        def make(*shape: int) -> torch.Tensor:
            return torch.randn(*shape, generator=generator, dtype=torch.bfloat16)

        prefix, stack = make(TOKENS, WIDE_DIM), make(TOKENS, BLOCKS, WIDE_DIM)
        # Scaled so the depth softmax is not saturated: a saturated one is
        # insensitive to the score weight and hides the precision loss.
        projection = (make(1, WIDE_DIM).float() * WIDE_DIM**-0.5).bfloat16()
        norm_weight = make(WIDE_DIM)
        expected = _reference(prefix, stack, projection, norm_weight)
        actual = _apply_attention_residual(
            prefix,
            stack,
            types.SimpleNamespace(weight=projection),
            types.SimpleNamespace(weight=norm_weight, eps=EPS),
        )
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=2e-3, atol=2e-3
        )

    def test_zero_projection_gives_uniform_depth_weights(self):
        # Zero initialisation makes the initial depth weights uniform, so the
        # aggregation returns the mean of its sources exactly.
        generator = torch.Generator().manual_seed(0)
        prefix = torch.randn(TOKENS, DIM, generator=generator)
        stack = torch.randn(TOKENS, BLOCKS, DIM, generator=generator)
        actual = _apply_attention_residual(
            prefix,
            stack,
            types.SimpleNamespace(weight=torch.zeros(1, DIM)),
            types.SimpleNamespace(weight=torch.ones(DIM), eps=EPS),
        )
        expected = torch.cat((stack, prefix.unsqueeze(1)), dim=1).mean(dim=1)
        torch.testing.assert_close(actual, expected)

    def test_zero_projection_moves_the_projection_but_not_the_norm(self):
        # The norm weight reaches the loss only through its product with the
        # projection, so at zero initialisation it has no gradient. It gains one
        # as soon as the projection leaves zero.
        generator = torch.Generator().manual_seed(0)
        prefix = torch.randn(TOKENS, DIM, generator=generator)
        stack = torch.randn(TOKENS, BLOCKS, DIM, generator=generator)
        projection = torch.zeros(1, DIM, requires_grad=True)
        norm_weight = torch.ones(DIM, requires_grad=True)
        output = _apply_attention_residual(
            prefix,
            stack,
            types.SimpleNamespace(weight=projection),
            types.SimpleNamespace(weight=norm_weight, eps=EPS),
        )
        output.backward(torch.randn(TOKENS, DIM, generator=generator))
        self.assertGreater(projection.grad.abs().max().item(), 0.0)
        self.assertEqual(norm_weight.grad.abs().max().item(), 0.0)

    def test_residual_projections_are_zero_initialised(self):
        from torchtitan.models.kimi_k3 import model_registry

        model = model_registry("debugmodel").model
        self.assertIsNone(model.layers[0].attention_res_proj)
        for projection in (
            model.layers[0].ffn_res_proj,
            model.layers[1].attention_res_proj,
            model.output_res_proj,
        ):
            self.assertIs(projection.param_init["weight"], torch.nn.init.zeros_)

    def test_registered_for_spmd_type_checking(self):
        # An autograd Function the checker does not know about raises under the
        # strict type checking the multimodal cell runs with.
        from spmd_types._local_registration import _LOCAL_AUTOGRAD_FUNCTIONS

        self.assertIn(_AttentionResidualAggregation, _LOCAL_AUTOGRAD_FUNCTIONS)


if __name__ == "__main__":
    unittest.main()
