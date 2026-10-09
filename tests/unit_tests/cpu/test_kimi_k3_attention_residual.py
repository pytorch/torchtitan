# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.kimi_k3.model import _apply_attention_residual


def _reference(partial, stack, projection_weight, norm_weight, eps):
    """Evaluate the normalized-key definition in FP64, including its gradients."""
    values = stack.double()
    if partial is not None:
        values = torch.cat((values, partial.double().unsqueeze(1)), dim=1)
    keys = values / (values.square().mean(-1, keepdim=True) + eps).sqrt()
    scores = torch.einsum(
        "tnd,d,d->tn", keys, norm_weight.double(), projection_weight.double()[0]
    )
    return torch.einsum("tn,tnd->td", scores.softmax(-1), values)


class TestKimiK3AttentionResidual(unittest.TestCase):
    def _check_case(self, *, dtype, entries, partial, strided, scale, weight_scale):
        torch.manual_seed(42)
        tokens, dim = 7, 64
        stride = 2 if strided else 1
        stack = (
            torch.randn(tokens, entries, dim * stride, dtype=dtype)[..., ::stride]
            * scale
        )
        # Preserve a noncontiguous leaf instead of testing a contiguous clone.
        if strided:
            backing = torch.empty(tokens, entries, dim * 2, dtype=dtype)
            backing[..., ::2] = stack
            stack = backing[..., ::2]
        stack.requires_grad_()
        partial_tensor = (
            (torch.randn(tokens, dim, dtype=dtype) * scale).requires_grad_()
            if partial
            else None
        )
        projection = Linear.Config(in_features=dim, out_features=1).build().to(dtype)
        norm = RMSNorm.Config(normalized_shape=dim, eps=1e-6).build().to(dtype)
        with torch.no_grad():
            projection.weight.normal_(std=weight_scale / dim**0.5)
            norm.weight.uniform_(0.5, 1.5)
        inputs = [stack, projection.weight, norm.weight]
        if partial_tensor is not None:
            inputs.append(partial_tensor)
        expected_inputs = [x.detach().double().requires_grad_() for x in inputs]
        expected = _reference(
            expected_inputs[3] if partial else None,
            *expected_inputs[:3],
            norm.eps,
        )
        # Bypass mutable global compile configuration for this eager CPU test.
        actual = _apply_attention_residual.__wrapped__(
            partial_tensor, stack, projection, norm
        )
        upstream = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, inputs, upstream)
        expected_grads = torch.autograd.grad(
            expected, expected_inputs, upstream.double()
        )
        self.assertEqual(actual.dtype, dtype)
        self.assertEqual(actual.shape, (tokens, dim))
        # FP32 reassociation is allowed; BF16 results include final rounding.
        rtol, atol = (2e-4, 2e-5) if dtype == torch.float32 else (2e-2, 2e-3)
        for result, reference in zip(
            [actual, *actual_grads], [expected, *expected_grads], strict=True
        ):
            self.assertTrue(torch.isfinite(result).all())
            torch.testing.assert_close(result.double(), reference, rtol=rtol, atol=atol)

    def test_forward_and_all_gradients(self):
        for dtype in (torch.float32, torch.bfloat16):
            # Includes the first entry, a committed-only stack, a partial block,
            # low-variance normalization, and strongly concentrated softmax.
            for entries, partial, strided, scale, weight_scale in (
                (0, True, False, 1.0, 1.0),
                (1, False, False, 1.0, 1.0),
                (8, False, True, 1.0, 1.0),
                (8, True, False, 1.0, 1.0),
                (8, True, True, 1e-4, 1.0),
                (8, True, False, 1.0, 20.0),
            ):
                with self.subTest(
                    dtype=dtype,
                    entries=entries,
                    partial=partial,
                    strided=strided,
                    scale=scale,
                    weight_scale=weight_scale,
                ):
                    self._check_case(
                        dtype=dtype,
                        entries=entries,
                        partial=partial,
                        strided=strided,
                        scale=scale,
                        weight_scale=weight_scale,
                    )


if __name__ == "__main__":
    unittest.main()
