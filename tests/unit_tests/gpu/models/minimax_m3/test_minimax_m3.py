# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_mask

from torchtitan.models.minimax_m3.attention import (
    msa_block_mask,
    MSAIndexerLoss,
    MSAInnerAttention,
    select_msa_blocks,
)

_NUM_GROUPS = 2
_HEADS_PER_GROUP = 2
_HEAD_DIM = 32
_INDEX_HEAD_DIM = 32
_DOCUMENT_LENGTHS = (300, 150, 62)


def _inputs(device: torch.device) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    num_tokens = sum(_DOCUMENT_LENGTHS)
    num_heads = _NUM_GROUPS * _HEADS_PER_GROUP
    return {
        "positions": torch.cat(
            [torch.arange(length, device=device) for length in _DOCUMENT_LENGTHS]
        ),
        "q": torch.randn(num_tokens, num_heads, _HEAD_DIM, device=device),
        "k": torch.randn(num_tokens, _NUM_GROUPS, _HEAD_DIM, device=device),
        "v": torch.randn(num_tokens, _NUM_GROUPS, _HEAD_DIM, device=device),
        "index_q": torch.randn(num_tokens, _NUM_GROUPS, _INDEX_HEAD_DIM, device=device),
        "index_k": torch.randn(num_tokens, _INDEX_HEAD_DIM, device=device),
    }


_CONFIG = MSAInnerAttention.Config(
    index_block_size=16,
    topk_blocks=4,
    aux_loss=MSAIndexerLoss.Config(
        coeff=0.5,
        softmax_scale=_HEAD_DIM**-0.5,
        index_softmax_scale=_INDEX_HEAD_DIM**-0.5,
    ),
)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestMiniMaxSparseAttention(unittest.TestCase):
    def _run(self, training: bool):
        inputs = _inputs(torch.device("cuda"))
        attention = _CONFIG.build().cuda().train(training)
        metadata = _CONFIG.build_attention_metadata(inputs["positions"])
        leaves = {
            name: inputs[name].clone().requires_grad_()
            for name in ("q", "k", "v", "index_q", "index_k")
        }
        out = attention(
            leaves["q"],
            leaves["k"],
            leaves["v"],
            attention_metadata=metadata,
            index_q_TGI=leaves["index_q"],
            index_k_TI=leaves["index_k"],
            scale=_HEAD_DIM**-0.5,
            enable_gqa=True,
            aux_loss_denominator=torch.tensor(1.0, device="cuda"),
        )
        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        return inputs, metadata, out, grad_out, leaves

    def test_sparse_flex_attention_matches_masked_sdpa(self) -> None:
        inputs, metadata, out, grad_out, leaves = self._run(training=False)
        num_tokens = inputs["q"].shape[0]
        num_heads = inputs["q"].shape[1]
        block_indices = select_msa_blocks(
            inputs["index_q"], inputs["index_k"], metadata, topk_blocks=4
        )
        block_mask, _ = msa_block_mask(block_indices, metadata, num_heads=num_heads)
        mask = create_mask(
            block_mask.mask_mod, 1, num_heads, num_tokens, num_tokens, device="cuda"
        )

        q, k, v = (inputs[name].clone().requires_grad_() for name in ("q", "k", "v"))
        expected = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None],
            k.transpose(0, 1).repeat_interleave(_HEADS_PER_GROUP, dim=0)[None],
            v.transpose(0, 1).repeat_interleave(_HEADS_PER_GROUP, dim=0)[None],
            attn_mask=mask,
            scale=_HEAD_DIM**-0.5,
        )[0].transpose(0, 1)
        expected.backward(grad_out)

        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
        for name, reference in (("q", q), ("k", k), ("v", v)):
            torch.testing.assert_close(
                leaves[name].grad, reference.grad, atol=1e-4, rtol=1e-4
            )
        self.assertIsNone(leaves["index_q"].grad)
        self.assertIsNone(leaves["index_k"].grad)

    def test_training_injects_the_kl_gradient_into_the_indexer_only(self) -> None:
        _, _, eval_out, _, eval_leaves = self._run(training=False)
        _, _, train_out, _, train_leaves = self._run(training=True)

        torch.testing.assert_close(train_out, eval_out, atol=0, rtol=0)
        for name in ("q", "k", "v"):
            torch.testing.assert_close(
                train_leaves[name].grad, eval_leaves[name].grad, atol=0, rtol=0
            )
        for name in ("index_q", "index_k"):
            grad = train_leaves[name].grad
            self.assertIsNotNone(grad)
            self.assertGreater(grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
