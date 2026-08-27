# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MTP (report sec 3.3): does the forward produce usable extra predictions?

The architecture existed before this and was never in the forward, so a run with
``num_nextn_predict_layers`` set trained nothing extra and said nothing about it.
"""

from __future__ import annotations

import dataclasses as dc
import unittest

import torch

from torchtitan.models.kimi_k3.attn_res_model import KimiK3AttnResModel
from torchtitan.models.kimi_k3.model_configs import build_kimi_linear_config


# KDA's training kernel is chunk-mode only and asserts T > 64.
SEQ = 128


def _config(num_mtp: int):
    kc = build_kimi_linear_config("k3mini", vocab_size=256)
    n = 4
    # The layer lists must be re-derived, not inherited: k3mini's cover 21 layers
    # and carrying them onto a 4-layer model leaves the two descriptions of the
    # same stack contradicting each other.
    full_attn = [4]
    return dc.replace(
        kc,
        num_hidden_layers=n,
        full_attn_layers=full_attn,
        kda_layers=[i for i in range(1, n + 1) if i not in full_attn],
        num_nextn_predict_layers=num_mtp,
    )


@unittest.skipUnless(torch.cuda.is_available(), "KDA and MoE need CUDA")
class TestMTPForward(unittest.TestCase):
    def _model(self, num_mtp: int):
        torch.manual_seed(0)
        m = KimiK3AttnResModel(_config(num_mtp), num_blocks=2).cuda().bfloat16()
        m.init_weights(buffer_device="cuda")
        return m

    def test_off_by_default_produces_no_mtp_logits(self):
        m = self._model(0)
        self.assertIsNone(m.mtp_layers)
        m(torch.randint(0, 256, (1, SEQ), device="cuda"))
        self.assertIsNone(m._mtp_logits)

    def test_logits_per_depth_are_shifted_and_finite(self):
        m = self._model(2)
        T = SEQ
        out = m(torch.randint(0, 256, (1, T), device="cuda"))
        self.assertEqual(out.shape[:2], (1, T))
        self.assertIsNotNone(m._mtp_logits)
        self.assertEqual(len(m._mtp_logits), 2)
        for k, logits in enumerate(m._mtp_logits):
            # depth k predicts k+1 ahead, so it is shorter by exactly that much
            self.assertEqual(logits.shape[1], T - (k + 1))
            self.assertEqual(logits.shape[2], 256)
            self.assertTrue(torch.isfinite(logits).all())

    def test_gradients_reach_the_mtp_layers(self):
        """A prediction nothing trains is not multi-token prediction."""
        m = self._model(1)
        out = m(torch.randint(0, 256, (1, SEQ), device="cuda"))
        # Loss on the MTP head only, so any gradient must have come through it.
        m._mtp_logits[0].float().sum().backward()
        g = m.mtp_layers["0"].eh_proj.weight.grad
        self.assertIsNotNone(g, "no gradient reached the MTP projection")
        self.assertTrue(torch.isfinite(g).all())
        self.assertGreater(g.abs().sum().item(), 0.0)
        del out


    def test_chunked_loss_is_rejected_rather_than_silently_materialising_logits(self):
        """finding 44. The MTP branch used to sit ahead of the _skip_lm_head return.

        A chunked-loss run therefore built a full [B, L, V] logits tensor per MTP depth --
        the exact allocation chunking exists to avoid, retaining ~1.3 GiB per depth and
        handing the loss chunk-misaligned labels.

        The check is that it RAISES. Skipping instead would leave take_mtp_logits()
        returning None and the MTP loss contributing nothing, so the run would look like
        it was training MTP while it was not.
        """
        m = self._model(1)
        m._skip_lm_head = True
        with self.assertRaises(ValueError) as caught:
            m(torch.randint(0, 256, (1, SEQ), device="cuda"))
        self.assertIn("chunked loss", str(caught.exception))

    def test_the_unchunked_path_is_unaffected(self):
        m = self._model(1)
        self.assertFalse(m._skip_lm_head)
        out = m(torch.randint(0, 256, (1, SEQ), device="cuda"))
        self.assertEqual(out.shape[-1], 256)
        self.assertIsNotNone(m._mtp_logits)


@unittest.skipUnless(torch.cuda.is_available(), "KDA and MoE need CUDA")
class TestMTPLoss(unittest.TestCase):
    """The loss half. Recorded as blocked on a core interface change; it is not,
    because MTP's targets are the same labels shifted."""

    def _model_and_loss(self, num_mtp: int, weight: float = 0.3):
        from torchtitan.components.loss import CrossEntropyLoss
        from torchtitan.models.kimi_k3.mtp_loss import KimiMTPLoss

        torch.manual_seed(0)
        m = KimiK3AttnResModel(_config(num_mtp), num_blocks=2).cuda().bfloat16()
        m.init_weights(buffer_device="cuda")
        loss = KimiMTPLoss.Config(
            mtp_weight=weight,
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=256),
        ).build()
        return m, loss

    def test_reduces_to_plain_ce_when_mtp_is_off(self):
        from torchtitan.components.loss import CrossEntropyLoss

        m, mtp_loss = self._model_and_loss(0)
        plain = CrossEntropyLoss.Config(global_vocab_size=256).build()
        tokens = torch.randint(0, 256, (1, SEQ), device="cuda")
        labels = torch.randint(0, 256, (1, SEQ), device="cuda")
        pred = m(tokens)
        a, _ = mtp_loss(pred, labels)
        b, _ = plain(pred, labels)
        self.assertEqual(a.item(), b.item())

    def test_mtp_raises_the_loss_and_reports_it(self):
        m, mtp_loss = self._model_and_loss(2)
        tokens = torch.randint(0, 256, (1, SEQ), device="cuda")
        labels = torch.randint(0, 256, (1, SEQ), device="cuda")
        pred = m(tokens)
        total, metrics = mtp_loss(pred, labels)
        self.assertIn("loss/mtp", metrics)
        self.assertTrue(torch.isfinite(total))
        # With a positive weight the total must exceed the main term alone.
        main_only, _ = mtp_loss.inner(pred, labels)
        self.assertGreater(total.item(), main_only.item())

    def test_weight_zero_leaves_the_main_loss_untouched(self):
        m, mtp_loss = self._model_and_loss(2, weight=0.0)
        tokens = torch.randint(0, 256, (1, SEQ), device="cuda")
        labels = torch.randint(0, 256, (1, SEQ), device="cuda")
        pred = m(tokens)
        total, _ = mtp_loss(pred, labels)
        main_only, _ = mtp_loss.inner(pred, labels)
        self.assertAlmostEqual(total.item(), main_only.item(), places=4)

    def test_positions_keyword_masks_cross_document_targets(self):
        """The trainer passes positions=; rejecting it was a TypeError before
        any step ran. And it carries the packing boundaries: a depth-k target
        past a document restart must not be trained on.
        """
        import torch

        from torchtitan.components.loss import CrossEntropyLoss
        from torchtitan.models.kimi_k3.mtp_loss import KimiMTPLoss, put_mtp_logits

        loss = KimiMTPLoss(
            KimiMTPLoss.Config(mtp_weight=1.0, loss_fn=CrossEntropyLoss.Config())
        )
        torch.manual_seed(0)
        vocab, seq = 32, 8
        pred = torch.randn(1, seq, vocab)
        labels = torch.randint(0, vocab, (1, seq))
        # Two packed documents of four tokens each: ids restart mid-sequence.
        positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])

        depth = torch.randn(1, seq - 1, vocab)
        put_mtp_logits([depth])
        with_mask, _ = loss(pred, labels, positions=positions)
        put_mtp_logits([depth])
        without, _ = loss(pred, labels)
        # Position 3's target crosses into document two; masking it changes
        # the depth loss, so the two totals must differ.
        self.assertNotAlmostEqual(with_mask.item(), without.item(), places=6)
