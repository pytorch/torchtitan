# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch

from torchtitan.models.common.attention import VarlenInnerAttention
from torchtitan.models.flux.attention import FluxInnerAttention


def _reference_varlen(self, q, k, v, *, attention_metadata):
    outputs = []
    for q_start, q_end, kv_start, kv_end in zip(
        attention_metadata.cu_seq_q[:-1],
        attention_metadata.cu_seq_q[1:],
        attention_metadata.cu_seq_k[:-1],
        attention_metadata.cu_seq_k[1:],
    ):
        scores = torch.einsum("lhk,shk->hls", q[q_start:q_end], k[kv_start:kv_end])
        probs = (scores / q.shape[-1] ** 0.5).softmax(dim=-1)
        outputs.append(torch.einsum("hls,shv->lhv", probs, v[kv_start:kv_end]))
    return torch.cat(outputs)


class TestFluxInnerAttention(unittest.TestCase):
    def test_defaults_to_bidirectional_varlen(self):
        config = FluxInnerAttention.Config()
        self.assertIsInstance(config.varlen_attention, VarlenInnerAttention.Config)
        self.assertEqual(config.varlen_attention.window_size, (-1, -1))

    @patch.object(VarlenInnerAttention, "forward", _reference_varlen)
    def test_folded_batches_match_independent_attention(self):
        torch.manual_seed(42)
        attention = FluxInnerAttention.Config().build()
        # Unequal lengths cover local queries with CP-gathered keys and values.
        for query_len in (3, 8):
            with self.subTest(query_len=query_len):
                q = torch.randn(2, query_len, 4, 8, dtype=torch.float64)
                k = torch.randn(2, 8, 4, 8, dtype=torch.float64)
                v = torch.randn(2, 8, 4, 6, dtype=torch.float64)
                inputs = [t.requires_grad_() for t in (q, k, v)]
                reference_inputs = [t.detach().clone().requires_grad_() for t in inputs]

                actual = attention(*inputs)
                ref_q, ref_k, ref_v = reference_inputs
                scores = torch.einsum("blhk,bshk->bhls", ref_q, ref_k) / 8**0.5
                expected = torch.einsum(
                    "bhls,bshv->blhv", scores.softmax(dim=-1), ref_v
                )

                torch.testing.assert_close(actual, expected)
                grad_output = torch.randn_like(actual)
                actual_grads = torch.autograd.grad(actual, inputs, grad_output)
                expected_grads = torch.autograd.grad(
                    expected, reference_inputs, grad_output
                )
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual_grad, expected_grad)

    def test_mismatched_batches_or_kv_lengths_raise(self):
        attention = FluxInnerAttention.Config().build()
        q = torch.randn(2, 3, 4, 8)
        k = torch.randn(2, 8, 4, 8)
        v = torch.randn_like(k)
        for bad_k, bad_v in ((k[:1], v), (k, v[:1]), (k, v[:, :7])):
            with self.subTest(k_shape=bad_k.shape, v_shape=bad_v.shape):
                with self.assertRaisesRegex(ValueError, "matching Q/K/V batches"):
                    attention(q, bad_k, bad_v)
