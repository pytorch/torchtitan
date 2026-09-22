# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.models.common.aux_loss import AuxLoss, _zero_aux_losses
from torchtitan.models.deepseek_v4 import Indexer, SparseIndexerLoss, model_registry


def _clear_aux_loss_registry():
    AuxLoss._group_counts.clear()
    AuxLoss.group_acc.clear()
    AuxLoss._step_denominator = None


def _teacher(q_THK, cmp_k_NK, topk_indices_TJ, lse_TH, softmax_scale):
    valid_TJ = topk_indices_TJ >= 0
    row_valid_T = valid_TJ.any(dim=-1)
    selected_TJK = cmp_k_NK[topk_indices_TJ.clamp_min(0)]
    logits_THJ = (
        torch.einsum("thk,tjk->thj", q_THK, selected_TJK).float() * softmax_scale
    )
    logits_THJ = logits_THJ.masked_fill(~valid_TJ.unsqueeze(1), -torch.inf)
    comp_lse_TH = torch.logsumexp(logits_THJ, dim=-1)
    mass_TH = torch.exp(comp_lse_TH - lse_TH.float())
    conditional_THJ = torch.softmax(
        logits_THJ.masked_fill(~row_valid_T[:, None, None], 0.0),
        dim=-1,
    )
    return (mass_TH.unsqueeze(-1) * conditional_THJ).sum(dim=1) / q_THK.size(1)


def _reference_loss(q_THK, cmp_k_NK, topk_indices_TJ, lse_TH, topk_scores_TJ, scale):
    p_TJ = _teacher(q_THK, cmp_k_NK, topk_indices_TJ, lse_TH, scale)
    t_TJ = p_TJ / p_TJ.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    row_valid_T = torch.isfinite(topk_scores_TJ).any(dim=-1)
    slot_valid_TJ = torch.isfinite(topk_scores_TJ) & row_valid_T.unsqueeze(-1)
    logits_TJ = topk_scores_TJ.float().masked_fill(~row_valid_T.unsqueeze(-1), 0.0)
    log_student_TJ = torch.log_softmax(
        logits_TJ.masked_fill(~slot_valid_TJ, -torch.inf).masked_fill(
            ~slot_valid_TJ.any(dim=-1, keepdim=True), 0.0
        ),
        dim=-1,
    )
    weighted_TJ = torch.special.xlogy(p_TJ, t_TJ) - p_TJ * log_student_TJ
    return weighted_TJ.masked_fill(~slot_valid_TJ, 0.0).sum()


class TestSparseIndexerLoss(unittest.TestCase):
    def setUp(self):
        _clear_aux_loss_registry()

    def tearDown(self):
        _clear_aux_loss_registry()

    def test_value_gradient_and_metric_match_reference(self):
        torch.manual_seed(0)
        T, H, D, N, K = 6, 3, 4, 5, 3
        coeff, denominator, softmax_scale = 0.25, 2.0, 0.5
        q_THK = torch.randn(T, H, D, dtype=torch.float64)
        cmp_k_NK = torch.randn(N, D, dtype=torch.float64)
        topk_indices_TJ = torch.tensor(
            [
                [-1, -1, -1],
                [0, -1, -1],
                [1, 0, -1],
                [2, 1, 0],
                [3, 2, 1],
                [4, 3, 2],
            ]
        )
        lse_TH = torch.logsumexp(
            torch.randn(T, H, 7, dtype=torch.float64), dim=-1
        ) + 2.0
        score_base_TJ = torch.randn(T, K, dtype=torch.float64, requires_grad=True)
        topk_scores_TJ = score_base_TJ.masked_fill(topk_indices_TJ < 0, -torch.inf)
        carrier = torch.randn(T, H, D, dtype=torch.float64)

        AuxLoss.set_step_denominator(torch.tensor(denominator, dtype=torch.float64))
        loss = SparseIndexerLoss(
            SparseIndexerLoss.Config(
                coeff=coeff, reduce_mesh="loss", softmax_scale=softmax_scale
            )
        )
        loss.train()
        out = loss(
            q_THK,
            cmp_k_NK,
            topk_indices_TJ,
            lse_TH,
            topk_scores_TJ,
            carrier=carrier,
        )
        self.assertTrue(torch.equal(out, carrier))
        out.sum().backward()

        ref_scores_TJ = score_base_TJ.detach().clone().requires_grad_(True)
        ref_topk_scores_TJ = ref_scores_TJ.masked_fill(
            topk_indices_TJ < 0, -torch.inf
        )
        ref_raw = _reference_loss(
            q_THK,
            cmp_k_NK,
            topk_indices_TJ,
            lse_TH,
            ref_topk_scores_TJ,
            softmax_scale,
        )
        (ref_raw * (coeff / denominator)).backward()
        self.assertLess((score_base_TJ.grad - ref_scores_TJ.grad).abs().max(), 1e-10)

        _zero_aux_losses([loss])
        metric = AuxLoss.group_acc[("loss", "sparse_indexer_loss")]
        self.assertAlmostEqual(metric.item(), ref_raw.item() / denominator, places=5)

    def test_deepseek_v4_config_attaches_loss_to_csa_layers(self):
        spec = model_registry("debugmodel", seq_len=512)
        for layer in spec.model.layers:
            inner_attention = layer.attention.inner_attention
            aux_loss = inner_attention.aux_loss
            if inner_attention.compress_ratio == 4:
                self.assertIsInstance(aux_loss, SparseIndexerLoss.Config)
                self.assertEqual(aux_loss.coeff, 0.01)
                self.assertEqual(aux_loss.reduce_mesh, "loss")
                self.assertEqual(aux_loss.softmax_scale, inner_attention.softmax_scale)
            else:
                self.assertIsNone(aux_loss)
        self.assertEqual(
            spec.post_optimizer_build_fn.__name__,
            "_post_optimizer_build_fn",
        )


class TestIndexerSelect(unittest.TestCase):
    def test_select_returns_live_logits_and_masked_indices(self):
        torch.manual_seed(0)
        T, Hi, Di, K = 6, 3, 4, 3
        N = T // 2  # compressed entries for ratio=2
        rng = torch.Generator().manual_seed(11)
        idx_q = torch.randn(T, Hi, Di, generator=rng, requires_grad=True)
        idx_k = torch.randn(N, Di, generator=rng)
        idx_w = torch.randn(T, Hi, generator=rng)

        topk_indices, scores_TK = Indexer.select(
            idx_q, idx_k, idx_w, seqlen=T, ratio=2, topk=K
        )
        self.assertEqual(topk_indices.shape, (T, K))
        self.assertEqual(scores_TK.shape, (T, K))

        # Invalid causal slots are already represented as -1 / -inf by select.
        self.assertTrue((scores_TK[topk_indices < 0] == -torch.inf).all())
        self.assertTrue(torch.isfinite(scores_TK[topk_indices >= 0]).all())
        causal_limit = torch.arange(1, T + 1).unsqueeze(1) // 2
        valid = topk_indices >= 0
        expanded_limit = causal_limit.expand_as(topk_indices)
        self.assertTrue((topk_indices[valid] < expanded_limit[valid]).all())

        # The returned student logits carry gradient into the indexer query path.
        scores_TK[topk_indices >= 0].sum().backward()
        self.assertIsNotNone(idx_q.grad)
        self.assertTrue(torch.isfinite(idx_q.grad).all())


if __name__ == "__main__":
    unittest.main()
