# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch._dynamo.eval_frame import _debug_get_cache_entry_list

from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.kimi_k3.model import (
    _attention_residual_backward,
    _attention_residual_forward,
    _AttentionResidual,
)

EPS = 1e-5


def _reference(partial_TD, residual_TND, projection_1D, norm_D):
    """The attention residual as concat + RMS norm + softmax-weighted sum, in FP64."""
    values_TND = torch.cat((residual_TND, partial_TD.unsqueeze(1)), dim=1).double()
    keys_TND = values_TND * torch.rsqrt(
        values_TND.square().mean(-1, keepdim=True) + EPS
    )
    score_weight_D = norm_D.double() * projection_1D.squeeze(0).double()
    probs_TN = torch.softmax((keys_TND * score_weight_D).sum(-1), dim=-1)
    return (probs_TN.unsqueeze(-1) * values_TND).sum(1)


def _inputs(num_entries: int, dtype: torch.dtype, num_tokens: int = 64):
    generator = torch.Generator(device="cuda").manual_seed(num_entries)

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda", generator=generator) * scale).to(
            dtype
        )

    return [
        randn(num_tokens, 256).requires_grad_(),
        randn(num_tokens, num_entries, 256).requires_grad_(),
        randn(1, 256, scale=0.05).requires_grad_(),
        (1 + randn(256, scale=0.1)).detach().requires_grad_(),
    ]


def _apply(partial, residual, projection, norm):
    return _AttentionResidual.apply(residual, partial, norm, projection, EPS)


def _output_and_grads(fn, inputs, grad_output):
    output = fn(*inputs)
    return [output, *torch.autograd.grad(output, inputs, grad_output)]


def _relative_error(actual, expected):
    return ((actual.double() - expected).norm() / expected.norm()).item()


def _num_graphs(region_fn) -> int:
    return len(_debug_get_cache_entry_list(region_fn.__wrapped__.__code__))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestKimiK3AttentionResidual(unittest.TestCase):
    def tearDown(self):
        apply_local_compile([])
        torch._dynamo.reset()

    def _check_against_fp64(self, compiled: bool):
        apply_local_compile(["attention_residual"] if compiled else [])
        # Widths 1, 7, 3, 8 in one binding: the width goes symbolic at its second
        # value, so 3 and 8 run the dynamic-width graphs.
        for dtype, tolerance in ((torch.float32, 1e-5), (torch.bfloat16, 5e-3)):
            for num_entries in (1, 7, 3, 8):
                inputs = _inputs(num_entries, dtype)
                grad_output = torch.randn_like(inputs[0])
                expected = _output_and_grads(
                    _reference,
                    [t.detach().double().requires_grad_() for t in inputs],
                    grad_output.double(),
                )
                actual = _output_and_grads(_apply, inputs, grad_output)
                for name, a, e in zip(
                    ("output", "partial", "residual", "projection", "norm"),
                    actual,
                    expected,
                    strict=True,
                ):
                    with self.subTest(dtype=dtype, entries=num_entries, tensor=name):
                        self.assertEqual(a.dtype, dtype)
                        self.assertLess(_relative_error(a, e), tolerance)

    def test_eager_matches_fp64_reference(self):
        self._check_against_fp64(compiled=False)

    def test_compiled_matches_fp64_reference(self):
        self._check_against_fp64(compiled=True)

    def test_graph_count_with_no_grad_and_token_counts(self):
        apply_local_compile(["attention_residual"])
        # Kimi K3's widths 1-8, training at two token counts, then a no_grad pass.
        for num_tokens in (64, 32):
            for num_entries in range(1, 9):
                inputs = _inputs(num_entries, torch.bfloat16, num_tokens)
                _apply(*inputs).sum().backward()
        with torch.no_grad():
            for num_entries in range(1, 9):
                _apply(*_inputs(num_entries, torch.bfloat16, 48))
        # no_grad reuses the training forward graphs; neither region nears the
        # default recompile limit of 8.
        self.assertLessEqual(_num_graphs(_attention_residual_forward), 4)
        self.assertLessEqual(_num_graphs(_attention_residual_backward), 4)


if __name__ == "__main__":
    unittest.main()
