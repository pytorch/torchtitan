# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from unittest.mock import patch

import torch
import torch.nn.functional as F

from torchtitan.models.common.attention import materialize_mla_kv
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.deepseek_v4.compressor import SparseIndexerLoss
from torchtitan.models.kimi_k3_1.dsa import AbsorbedKVProjection, KimiSparseAttention


def _reference(
    q_THK,
    k_THK,
    v_THV,
    idx_q_TIE,
    idx_k_TE,
    idx_w_TI,
    *,
    topk,
    scale,
    lengths,
    padding_mask
):
    """Per-token sparse attention outputs and the summed indexer KL."""
    outputs_THV, losses = [], []
    start = 0
    for length in lengths:
        for local_t in range(length):
            token = start + local_t
            candidates = torch.arange(start, token + 1)
            dots_IC = torch.einsum("ie,ce->ic", idx_q_TIE[token], idx_k_TE[candidates])
            scores_C = (dots_IC.relu() * idx_w_TI[token, :, None]).sum(dim=0)
            scores_S, slots_S = scores_C.topk(min(topk, local_t + 1))
            selected = candidates[slots_S]
            logits_HS = (
                torch.einsum("hk,shk->hs", q_THK[token], k_THK[selected]) * scale
            )
            probs_HS = logits_HS.softmax(dim=-1)
            outputs_THV.append(torch.einsum("hs,shv->hv", probs_HS, v_THV[selected]))
            if not padding_mask[token]:
                teacher_S = probs_HS.detach().sum(dim=0)
                teacher_S = teacher_S / teacher_S.sum()
                student_log_S = scores_S.log_softmax(dim=-1)
                losses.append((teacher_S * (teacher_S.log() - student_log_S)).sum())
        start += length
    return torch.stack(outputs_THV), torch.stack(losses).sum()


def test_sparse_attention_and_indexer_loss_match_oracle() -> None:
    torch.manual_seed(19)
    lengths = [3, 5, 2]
    padding_mask = torch.tensor([False] * 8 + [True, True])
    q_THK, kv_c_normed_TL, k_shared_TR, idx_q_TIE, idx_k_TE, idx_w_TI = (
        torch.randn(*shape, dtype=torch.float64)
        for shape in ((10, 2, 6), (10, 14), (10, 2), (10, 3, 8), (10, 8), (10, 3))
    )
    idx_k_TE.abs_()
    idx_q_TIE[:, 0].abs_()
    main = (q_THK, kv_c_normed_TL, k_shared_TR)
    indexer = (idx_q_TIE, idx_k_TE, idx_w_TI)
    for tensor in main + indexer:
        tensor.requires_grad_()
    with patch.dict(AuxLoss._group_counts, clear=True), patch.dict(
        AuxLoss.group_acc, clear=True
    ):
        config = KimiSparseAttention.Config(
            wkv_b=AbsorbedKVProjection.Config(
                kv_lora_rank=14, num_heads=2, qk_nope_head_dim=4, v_head_dim=5
            ),
            index_topk=3,
            aux_loss=SparseIndexerLoss.Config(
                coeff=0.01,
                reduce_mesh="loss",
                softmax_scale=0.173,
                num_heads=2,
                compress_ratio=1,
            ),
        )
        attention = config.build().train()
        attention.wkv_b.double()
        with torch.no_grad():
            attention.wkv_b.weight.normal_()
        positions = torch.cat([torch.arange(length) for length in lengths]).int()
        actual = attention(
            *main,
            idx_q_TIE=idx_q_TIE,
            idx_k_TE=idx_k_TE,
            idx_w_TI=idx_w_TI,
            attention_metadata=config.build_attention_metadata(
                positions, padding_mask=padding_mask, max_context_length=10
            ),
            scale=0.173,
            aux_loss_denominator=torch.tensor(8.0),
            padding_mask=padding_mask,
        )
        kv_THP = F.linear(kv_c_normed_TL, attention.wkv_b.weight).unflatten(-1, (2, -1))
        k_THK, v_THV = materialize_mla_kv(q_THK, kv_THP, k_shared_TR)
        expected, expected_kl = _reference(
            q_THK,
            k_THK,
            v_THV,
            *indexer,
            topk=3,
            scale=0.173,
            lengths=lengths,
            padding_mask=padding_mask,
        )

        grad = torch.randn_like(actual)
        main += (attention.wkv_b.weight,)
        actual_grads = torch.autograd.grad(actual, main + indexer, grad)
        expected_grads = torch.autograd.grad(
            expected, main, grad, retain_graph=True
        ) + torch.autograd.grad(expected_kl * (0.01 / 8.0), indexer)

        torch.testing.assert_close(actual, expected)
        for actual_grad, expected_grad in zip(
            actual_grads, expected_grads, strict=True
        ):
            torch.testing.assert_close(actual_grad, expected_grad)
        assert attention.aux_loss is not None
        torch.testing.assert_close(
            attention.aux_loss.instance_acc, (expected_kl.detach() / 8.0).float()
        )
