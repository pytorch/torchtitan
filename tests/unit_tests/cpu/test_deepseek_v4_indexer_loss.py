# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import bisect
import unittest
from unittest.mock import patch

import torch
from attn_gym.sparse.gather_attn import AuxRequest, gather_attn

from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.deepseek_v4 import build_model_config
from torchtitan.models.deepseek_v4.attention import CompressedSparseAttention
from torchtitan.models.deepseek_v4.compressor import (
    compressed_cu_seqlens,
    Indexer,
    SparseIndexerLoss,
)


def dense_attention(q, local, compressed, indices, sink, scale, window, cu):
    """Enumerate actual keys per query/head, independently of the loss."""
    offsets = [0, len(q)] if cu is None else cu.tolist()
    compressed_offsets = [0]
    for start, end in zip(offsets[:-1], offsets[1:]):
        compressed_offsets.append(compressed_offsets[-1] + (end - start) // 4)
    output, teacher = torch.zeros_like(q), torch.zeros(indices.shape)
    lse = torch.empty(q.shape[:2])
    for token in range(len(q)):
        doc = bisect.bisect_right(offsets, token) - 1
        local_indices = list(range(max(offsets[doc], token - window + 1), token + 1))
        slots = [slot for slot, index in enumerate(indices[token]) if index >= 0]
        selected = [
            compressed_offsets[doc] + int(indices[token, slot]) for slot in slots
        ]
        keys = torch.cat((local[local_indices], compressed[selected]))
        for head in range(q.size(1)):
            logits = torch.cat((keys @ q[token, head] * scale, sink[head : head + 1]))
            probabilities = logits.softmax(0)
            output[token, head] = probabilities[:-1] @ keys
            lse[token, head] = logits.logsumexp(0)
            teacher[token, slots] += probabilities[len(local_indices) : -1] / q.size(1)
    return output, lse, teacher


class TestSparseIndexerLoss(unittest.TestCase):
    def setUp(self):
        AuxLoss._group_counts.clear()
        AuxLoss.group_acc.clear()
        torch.manual_seed(42)
        self.q = torch.randn(12, 4, 8)
        self.local, self.compressed = torch.randn(12, 8), torch.randn(3, 8)
        self.sink = torch.randn(4)
        self.scale, self.window = 8**-0.5, 3

    def tearDown(self):
        AuxLoss._group_counts.clear()
        AuxLoss.group_acc.clear()

    def case(self, packed):
        cu = torch.tensor([0, 4, 4, 12], dtype=torch.int32) if packed else None
        indices = torch.full((12, 2), -1, dtype=torch.int32)
        for token in range(12):
            start = 0 if not packed or token < 4 else 4
            count = min((token - start + 1) // 4, 2)
            indices[token, :count] = torch.arange(count)
        return indices, cu

    def kernel(self, indices, cu):
        out, aux = gather_attn(
            self.q.transpose(0, 1)[None],
            self.local[None, None],
            self.compressed[None, None],
            indices[None],
            attention_sink=self.sink,
            sliding_window_size=self.window,
            cu_seqlens=cu,
            cu_seqlens_k=None if cu is None else compressed_cu_seqlens(cu, 4),
            scale=self.scale,
            impl="reference",
            return_aux=AuxRequest(lse=True),
        )
        return out[0].transpose(0, 1), aux.lse[0].transpose(0, 1)

    def loss(self, **kwargs):
        return SparseIndexerLoss.Config(
            coeff=0.01,
            reduce_mesh="loss",
            softmax_scale=self.scale,
            num_heads=4,
            chunk_size=3,
            **kwargs,
        ).build()

    def test_teacher_matches_independent_dense_attention_and_kernel(self):
        for packed in (False, True):
            with self.subTest(packed=packed):
                indices, cu = self.case(packed)
                expected_out, expected_lse, expected_teacher = dense_attention(
                    self.q,
                    self.local,
                    self.compressed,
                    indices,
                    self.sink,
                    self.scale,
                    self.window,
                    cu,
                )
                out, lse = self.kernel(indices, cu)
                torch.testing.assert_close(out, expected_out)
                torch.testing.assert_close(lse, expected_lse)
                teacher = self.loss()._teacher(
                    self.q, self.compressed, indices, lse, cu
                )
                torch.testing.assert_close(teacher, expected_teacher)

    def test_padding_gradient_and_metric_match_independent_kl(self):
        indices, cu = self.case(True)
        _, lse = self.kernel(indices, cu)
        _, _, teacher = dense_attention(
            self.q,
            self.local,
            self.compressed,
            indices,
            self.sink,
            self.scale,
            self.window,
            cu,
        )
        valid_tokens = torch.arange(12) < 10
        for weighted in (False, True):
            with self.subTest(weighted=weighted):
                scores = torch.randn(12, 2, requires_grad=True)
                loss = self.loss(mass_weighted=weighted)
                carrier = torch.randn_like(self.q, requires_grad=True)
                out = loss(
                    self.q,
                    self.compressed,
                    indices,
                    scores.masked_fill(indices < 0, -torch.inf),
                    lse,
                    cu,
                    carrier=carrier,
                    denominator=torch.tensor(10.0),
                    padding_mask_T=~valid_tokens,
                )
                torch.testing.assert_close(out, carrier, rtol=0, atol=0)
                out.sum().backward()
                ref_scores = scores.detach().clone().requires_grad_()
                reference = ref_scores.sum() * 0
                for token in range(10):
                    valid = indices[token] >= 0
                    if not valid.any():
                        continue
                    p = teacher[token, valid]
                    target = p / p.sum()
                    kl = (
                        target
                        * (target.log() - ref_scores[token, valid].log_softmax(0))
                    ).sum()
                    reference = reference + kl * (p.sum() if weighted else 1)
                (reference * 0.01 / 10).backward()
                torch.testing.assert_close(scores.grad, ref_scores.grad)
                self.assertEqual(scores.grad[~valid_tokens].count_nonzero(), 0)
                torch.testing.assert_close(loss.instance_acc, reference.detach() / 10)

    def test_chunk_size_does_not_change_teacher(self):
        indices, cu = self.case(True)
        _, lse = self.kernel(indices, cu)
        loss = self.loss()
        small = loss._teacher(self.q, self.compressed, indices, lse, cu)
        loss.chunk_size = 128
        large = loss._teacher(self.q, self.compressed, indices, lse, cu)
        torch.testing.assert_close(small, large)

    def test_bfloat16_teacher_dot_products_accumulate_in_float32(self):
        indices, cu = self.case(True)
        q = self.q.bfloat16()
        compressed = self.compressed.bfloat16()
        _, lse, expected = dense_attention(
            q.float(),
            self.local,
            compressed.float(),
            indices,
            self.sink,
            self.scale,
            self.window,
            cu,
        )
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual = self.loss()._teacher(q, compressed, indices, lse, cu)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)

    def test_configuration_defaults_to_unweighted_kl(self):
        self.assertFalse(self.loss().mass_weighted)
        cfg = build_model_config("debugmodel", seq_len=512)
        losses = [
            layer.attention.inner_attention.aux_loss
            for layer in cfg.layers
            if layer.attention.inner_attention.aux_loss is not None
        ]
        self.assertTrue(losses)
        for loss in losses:
            self.assertFalse(loss.mass_weighted)

    def test_configuration_can_be_tuned_or_disabled(self):
        for coeff in (0.0, 0.02):
            cfg = build_model_config(
                "debugmodel",
                seq_len=512,
                indexer_loss_coeff=coeff,
                indexer_loss_mass_weighted=True,
                indexer_loss_chunk_size=32,
            )
            for layer in cfg.layers:
                aux = layer.attention.inner_attention.aux_loss
                if coeff == 0 or layer.attention.compress_ratio != 4:
                    self.assertIsNone(aux)
                else:
                    self.assertEqual(aux.coeff, coeff)
                    self.assertEqual(aux.num_heads, layer.attention.n_heads)
                    self.assertTrue(aux.mass_weighted)
                    self.assertEqual(aux.chunk_size, 32)


class TestIndexerSelect(unittest.TestCase):
    def test_selection_does_not_depend_on_rescoring(self):
        torch.manual_seed(0)
        q = torch.randn(12, 3, 8, requires_grad=True)
        k = torch.randn(3, 8, requires_grad=True)
        w = torch.randn(12, 3, requires_grad=True)
        for cu in (None, torch.tensor([0, 4, 12], dtype=torch.int32)):
            indices, scores = Indexer.select(
                q, k, w, max_seqlen=12, ratio=4, topk=2, cu_seqlens=cu
            )
            index_only, no_scores = Indexer.select(
                q,
                k,
                w,
                max_seqlen=12,
                ratio=4,
                topk=2,
                cu_seqlens=cu,
                return_scores=False,
            )
            torch.testing.assert_close(indices, index_only)
            self.assertIsNone(no_scores)
            scores[indices >= 0].sum().backward()
            for tensor in (q, k, w):
                self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_eval_and_disabled_loss_do_not_rescore(self):
        q, kv = torch.randn(8, 4, 8), torch.randn(8, 8)
        iq, ik, iw = torch.randn(8, 2, 8), torch.randn(2, 8), torch.randn(8, 2)
        for training, enabled in ((False, True), (True, False)):
            cfg = CompressedSparseAttention.Config(
                window_size=3,
                compress_ratio=4,
                softmax_scale=8**-0.5,
                index_topk=2,
                aux_loss=SparseIndexerLoss.Config(
                    coeff=0.01, softmax_scale=8**-0.5, num_heads=4
                )
                if enabled
                else None,
            )
            module = cfg.build().train(training)
            with patch.object(Indexer, "select", wraps=Indexer.select) as select:
                module(q, kv, kv[:2], iq, ik, iw, torch.zeros(4))
                self.assertFalse(select.call_args.kwargs["return_scores"])


if __name__ == "__main__":
    unittest.main()
