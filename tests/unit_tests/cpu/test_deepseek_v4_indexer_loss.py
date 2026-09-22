# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.models.common.aux_loss import AuxLoss, _zero_aux_losses
from torchtitan.models.deepseek_v4 import build_model_config
from torchtitan.models.deepseek_v4.compressor import Indexer, SparseIndexerLoss


def _clear_aux_loss_registry():
    AuxLoss._group_counts.clear()
    AuxLoss.group_acc.clear()


def _reference_teacher(q_THD, swa_k_TD, cmp_k_SD, topk_TK, sink_H, scale, window):
    valid_TK = topk_TK >= 0
    selected_TKD = cmp_k_SD[topk_TK.clamp_min(0)]
    cmp_logits_THK = torch.einsum("thd,tkd->thk", q_THD, selected_TKD).float()
    cmp_logits_THK = cmp_logits_THK * scale
    cmp_logits_THK = cmp_logits_THK.masked_fill(~valid_TK.unsqueeze(1), -torch.inf)

    T = q_THD.size(0)
    offsets = torch.arange(window)
    key_TW = torch.arange(T).unsqueeze(1) - (window - 1 - offsets)
    valid_TW = key_TW >= 0
    selected_TWD = swa_k_TD[key_TW.clamp_min(0)]
    window_logits_THW = torch.einsum("thd,twd->thw", q_THD, selected_TWD).float()
    window_logits_THW = (window_logits_THW * scale).masked_fill(
        ~valid_TW.unsqueeze(1), -torch.inf
    )

    cmp_lse_TH = torch.logsumexp(cmp_logits_THK, dim=-1)
    full_lse_TH = torch.logsumexp(
        torch.stack(
            [
                torch.logsumexp(window_logits_THW, dim=-1),
                cmp_lse_TH,
                sink_H.float().unsqueeze(0).expand_as(cmp_lse_TH),
            ],
            dim=-1,
        ),
        dim=-1,
    )
    mass_TH = torch.exp(cmp_lse_TH - full_lse_TH)
    row_valid_T = valid_TK.any(dim=-1)
    conditional_THK = torch.softmax(
        cmp_logits_THK.masked_fill(~row_valid_T[:, None, None], 0.0),
        dim=-1,
    )
    return (mass_TH.unsqueeze(-1) * conditional_THK).sum(dim=1) / q_THD.size(1)


def _reference_loss(
    q_THD, swa_k_TD, cmp_k_SD, topk_TK, scores_TK, sink_H, scale, window
):
    p_TK = _reference_teacher(
        q_THD, swa_k_TD, cmp_k_SD, topk_TK, sink_H, scale, window
    )
    t_TK = p_TK / p_TK.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    row_valid_T = torch.isfinite(scores_TK).any(dim=-1)
    slot_valid_TK = torch.isfinite(scores_TK) & row_valid_T.unsqueeze(-1)
    logits_TK = scores_TK.float().masked_fill(~row_valid_T.unsqueeze(-1), 0.0)
    log_student_TK = torch.log_softmax(
        logits_TK.masked_fill(~slot_valid_TK, -torch.inf).masked_fill(
            ~slot_valid_TK.any(dim=-1, keepdim=True), 0.0
        ),
        dim=-1,
    )
    weighted_TK = torch.special.xlogy(p_TK, t_TK) - p_TK * log_student_TK
    return weighted_TK.masked_fill(~slot_valid_TK, 0.0).sum()


class TestSparseIndexerLoss(unittest.TestCase):
    def setUp(self):
        _clear_aux_loss_registry()

    def tearDown(self):
        _clear_aux_loss_registry()

    def test_value_gradient_and_metric_match_reference(self):
        torch.manual_seed(0)
        T, H, D, N, K = 6, 3, 4, 5, 3
        coeff, denominator, softmax_scale, window = 0.25, 2.0, 0.5, 4
        q_THD = torch.randn(T, H, D, dtype=torch.float64)
        swa_k_TD = torch.randn(T, D, dtype=torch.float64)
        cmp_k_SD = torch.randn(N, D, dtype=torch.float64)
        sink_H = torch.randn(H, dtype=torch.float64)
        topk_TK = torch.tensor(
            [
                [-1, -1, -1],
                [0, -1, -1],
                [1, 0, -1],
                [2, 1, 0],
                [3, 2, 1],
                [4, 3, 2],
            ]
        )
        score_base_TK = torch.randn(T, K, dtype=torch.float64, requires_grad=True)
        scores_TK = score_base_TK.masked_fill(topk_TK < 0, -torch.inf)
        carrier = torch.randn(T, H, D, dtype=torch.float64)

        loss = SparseIndexerLoss(
            SparseIndexerLoss.Config(
                coeff=coeff,
                reduce_mesh="loss",
                softmax_scale=softmax_scale,
                window_size=window,
            )
        )
        loss.train()
        out = loss(
            q_THD,
            swa_k_TD,
            cmp_k_SD,
            topk_TK,
            scores_TK,
            sink_H,
            None,
            carrier=carrier,
            denominator=torch.tensor(denominator, dtype=torch.float64),
        )
        self.assertTrue(torch.equal(out, carrier))
        out.sum().backward()

        ref_scores_TK = score_base_TK.detach().clone().requires_grad_(True)
        ref_topk_scores_TK = ref_scores_TK.masked_fill(topk_TK < 0, -torch.inf)
        ref_raw = _reference_loss(
            q_THD,
            swa_k_TD,
            cmp_k_SD,
            topk_TK,
            ref_topk_scores_TK,
            sink_H,
            softmax_scale,
            window,
        )
        (ref_raw * (coeff / denominator)).backward()
        self.assertLess((score_base_TK.grad - ref_scores_TK.grad).abs().max(), 1e-10)

        _zero_aux_losses([loss])
        metric = AuxLoss.group_acc[("loss", "sparse_indexer_loss")]
        self.assertAlmostEqual(metric.item(), ref_raw.item() / denominator, places=5)

    def test_deepseek_v4_config_attaches_loss_to_csa_layers(self):
        config = build_model_config("debugmodel", seq_len=512)
        for layer in config.layers:
            inner_attention = layer.attention.inner_attention
            aux_loss = inner_attention.aux_loss
            if inner_attention.compress_ratio == 4:
                self.assertIsInstance(aux_loss, SparseIndexerLoss.Config)
                self.assertEqual(aux_loss.coeff, 0.01)
                self.assertEqual(aux_loss.reduce_mesh, "loss")
                self.assertEqual(aux_loss.softmax_scale, inner_attention.softmax_scale)
                self.assertEqual(aux_loss.window_size, inner_attention.window_size)
            else:
                self.assertIsNone(aux_loss)


class TestIndexerSelect(unittest.TestCase):
    def test_select_returns_live_logits_at_selected_entries(self):
        torch.manual_seed(0)
        T, Hi, Di, K = 8, 3, 4, 3
        N = T // 2
        rng = torch.Generator().manual_seed(11)
        idx_q = torch.randn(T, Hi, Di, generator=rng, requires_grad=True)
        idx_k = torch.randn(N, Di, generator=rng)
        idx_w = torch.randn(T, Hi, generator=rng)

        topk_indices, scores_TK = Indexer.select(
            idx_q, idx_k, idx_w, max_seqlen=T, ratio=2, topk=K
        )
        self.assertEqual(topk_indices.shape, (T, K))
        self.assertEqual(scores_TK.shape, (T, K))
        self.assertTrue((scores_TK[topk_indices < 0] == -torch.inf).all())
        self.assertTrue(torch.isfinite(scores_TK[topk_indices >= 0]).all())

        scores_TK[topk_indices >= 0].sum().backward()
        self.assertIsNotNone(idx_q.grad)
        self.assertTrue(torch.isfinite(idx_q.grad).all())


if __name__ == "__main__":
    unittest.main()
