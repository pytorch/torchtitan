# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compare cached and freshly built DeepSeek V4 SWA/HCA attention masks."""

import unittest
from unittest import mock

import torch

from torchtitan.models.deepseek_v4 import model_registry
from torchtitan.models.deepseek_v4.attention import dsv4_mask_key, DSV4FlexAttention


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class TestDeepSeekV4AttentionCache(unittest.TestCase):
    def _make_model(self):
        # Only mask construction and parameter-free inner attention are used.
        with torch.device("meta"):
            model = model_registry("debugmodel", seq_len=512).model.build()
        return model

    def _assert_outputs_and_grads_match(self, inner, cached_mask, seqlen):
        device = torch.device("cuda")
        num_heads, head_dim = 2, 256
        inputs = [
            torch.randn(seqlen, num_heads, head_dim, device=device, requires_grad=True),
            torch.randn(seqlen, head_dim, device=device, requires_grad=True),
        ]
        if inner.compress_ratio > 1:
            inputs.append(
                torch.randn(
                    seqlen // inner.compress_ratio,
                    head_dim,
                    device=device,
                    requires_grad=True,
                )
            )
        inputs.append(torch.randn(num_heads, device=device, requires_grad=True))
        reference_inputs = [x.detach().clone().requires_grad_(True) for x in inputs]

        fresh_mask = DSV4FlexAttention.build_block_mask(
            inner, seqlen=seqlen, device=device
        )
        self.assertIsNot(cached_mask, fresh_mask)
        cached_output = inner(*inputs, attention_masks=cached_mask)
        reference_output = inner(*reference_inputs, attention_masks=fresh_mask)
        grad_output = torch.randn_like(cached_output)
        cached_grads = torch.autograd.grad(cached_output, inputs, grad_output)
        reference_grads = torch.autograd.grad(
            reference_output, reference_inputs, grad_output
        )

        torch.testing.assert_close(
            cached_output, reference_output, atol=1e-5, rtol=1e-5
        )
        names = ["q", "swa_k"]
        if inner.compress_ratio > 1:
            names.append("cmp_k")
        names.append("attn_sink")
        for name, actual, expected in zip(
            names, cached_grads, reference_grads, strict=True
        ):
            torch.testing.assert_close(
                actual, expected, atol=1e-5, rtol=1e-5, msg=f"grad[{name}] mismatch"
            )

    def test_cached_masks_match_fresh_masks_across_sequence_lengths(self):
        torch.manual_seed(0)
        model = self._make_model()
        masks_by_length = {}
        build_mask = DSV4FlexAttention.build_block_mask

        for seqlen, expected_builds in [(256, 2), (256, 0), (512, 2), (256, 0)]:
            with self.subTest(seqlen=seqlen, expected_builds=expected_builds):
                positions = torch.arange(seqlen, device="cuda")
                with mock.patch.object(
                    DSV4FlexAttention,
                    "build_block_mask",
                    autospec=True,
                    side_effect=build_mask,
                ) as builder:
                    masks = model.get_attention_masks(positions)
                self.assertEqual(builder.call_count, expected_builds)
                self.assertEqual(set(masks), {"swa", "hca_128"})

                if seqlen in masks_by_length:
                    for key, mask in masks.items():
                        self.assertIs(mask, masks_by_length[seqlen][key])
                else:
                    for other_masks in masks_by_length.values():
                        for key, mask in masks.items():
                            self.assertIsNot(mask, other_masks[key])
                    masks_by_length[seqlen] = masks
                self.assertEqual(len(model.mask_cache), 2 * len(masks_by_length))

                for layer in model.layers.values():
                    inner = layer.attention.inner_attention
                    key = dsv4_mask_key(inner.compress_ratio)
                    if key is not None:
                        with self.subTest(mask=key):
                            self._assert_outputs_and_grads_match(
                                inner, masks[key], seqlen
                            )

    def test_rejects_incompatible_layers_before_deduplication(self):
        for field, value in [("window_size", 32), ("block_size", 64)]:
            for populate_cache in (False, True):
                with self.subTest(field=field, populate_cache=populate_cache):
                    model = self._make_model()
                    positions = torch.arange(256, device="cuda")
                    if populate_cache:
                        model.get_attention_masks(positions)
                    inner = model.layers["1"].attention.inner_attention
                    self.assertNotEqual(getattr(inner, field), value)
                    setattr(inner, field, value)
                    with self.assertRaises(AssertionError):
                        model.get_attention_masks(positions)


if __name__ == "__main__":
    unittest.main()
