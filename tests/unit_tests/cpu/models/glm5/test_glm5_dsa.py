# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import spmd_types as spmd
import torch

from torchtitan.models.common.attn_gym import lightning_indexer
from torchtitan.models.glm5.dsa import (
    create_dsa_causal_mask,
    DSAIndexerTopK,
    selected_mask,
)
from torchtitan.models.glm5.sharding import glm5_indexer_topk_sharding_config


class TestGlm5Dsa(unittest.TestCase):
    @staticmethod
    def _dense_indices(q, k, weights, mask, topk):
        scores = (q.float().transpose(0, 1) @ k.float().T) * q.shape[-1] ** -0.5
        scores = scores.relu()
        weights = weights.float() * q.shape[1] ** -0.5
        scores = (weights.unsqueeze(1) @ scores.transpose(0, 1)).squeeze(1)
        return (scores + mask[0]).topk(min(topk, k.shape[0]), dim=-1).indices

    def test_indexer_matches_dense_ranking(self) -> None:
        torch.manual_seed(42)
        for dtype in (torch.float32, torch.bfloat16):
            q = torch.randn(32, 4, 8, dtype=dtype)
            k = torch.randn(32, 8, dtype=dtype)
            weights = torch.randn(32, 4)
            for positions, valid in (
                (torch.arange(32), None),
                (torch.arange(8).repeat(4), None),
                (torch.arange(8).repeat(4), torch.rand(32) > 0.3),
                (torch.arange(32), torch.zeros(32, dtype=torch.bool)),
            ):
                mask = create_dsa_causal_mask(
                    positions, dtype=torch.float32, padding_mask_T=valid
                )
                for topk in (1, 4, 64):
                    with self.subTest(dtype=dtype, positions=positions, topk=topk):
                        indexer = DSAIndexerTopK.Config(index_topk=topk).build()
                        actual = indexer(q, k, weights, mask[0])
                        expected = self._dense_indices(q, k, weights, mask, topk)
                        self.assertEqual(actual.dtype, torch.int32)
                        self.assertEqual(actual.shape, (32, min(topk, 32)))
                        torch.testing.assert_close(
                            selected_mask(actual, mask), selected_mask(expected, mask)
                        )
                        self.assertTrue(
                            (
                                (actual == -1)
                                | (mask[0].gather(1, actual.clamp_min(0).long()) == 0)
                            ).all()
                        )

    def test_indexer_excludes_padding_before_topk(self) -> None:
        q = torch.ones(4, 1, 1)
        k = torch.tensor([[1.0], [1000.0], [2.0], [3.0]])
        weights = torch.ones(4, 1)
        mask = create_dsa_causal_mask(
            torch.arange(4),
            dtype=torch.float32,
            padding_mask_T=torch.tensor([True, False, True, True]),
        )
        actual = DSAIndexerTopK.Config(index_topk=2).build()(q, k, weights, mask[0])
        torch.testing.assert_close(actual[1], torch.tensor([-1, -1], dtype=torch.int32))
        torch.testing.assert_close(
            actual[-1].sort().values, torch.tensor([2, 3], dtype=torch.int32)
        )

    def test_indexer_uses_attention_gym_without_gradients(self) -> None:
        q = torch.ones(4, 2, 8, requires_grad=True)
        k = torch.ones(4, 8, requires_grad=True)
        weights = torch.ones(4, 2, requires_grad=True)
        mask = create_dsa_causal_mask(torch.arange(4), dtype=torch.float32)
        indexer = DSAIndexerTopK.Config(index_topk=2).build()
        with patch(
            "torchtitan.models.glm5.dsa.lightning_indexer", wraps=lightning_indexer
        ) as kernel:
            indices = indexer(q, k, weights, mask[0])
        kernel.assert_called_once()
        self.assertFalse(kernel.call_args.args[0].requires_grad)
        self.assertFalse(indices.requires_grad)

    def test_indexer_compiles_with_fixed_metadata_capacity(self) -> None:
        torch.manual_seed(42)
        q = torch.randn(16, 4, 8)
        k = torch.randn(16, 8)
        weights = torch.randn(16, 4)
        indexer = DSAIndexerTopK.Config(index_topk=4).build()
        compiled = torch.compile(indexer, backend="eager", fullgraph=True)
        for positions, valid in (
            (torch.arange(16), None),
            (torch.arange(8).repeat(2), torch.arange(16) % 3 != 0),
        ):
            mask = create_dsa_causal_mask(
                positions, dtype=torch.float32, padding_mask_T=valid
            )[0]
            torch.testing.assert_close(
                compiled(q, k, weights, mask), indexer(q, k, weights, mask)
            )

    def test_cp_indexer_returns_local_queries_with_global_key_indices(self) -> None:
        torch.manual_seed(42)
        q = torch.randn(16, 4, 8)
        k = torch.randn(16, 8)
        weights = torch.randn(16, 4)
        mask = create_dsa_causal_mask(
            torch.cat((torch.arange(6), torch.arange(10))), dtype=torch.float32
        )
        indexer = DSAIndexerTopK.Config(index_topk=4).build()
        expected = indexer(q, k, weights, mask[0])
        mask_values, document_starts = mask[0].max(dim=-1)
        for rank in range(2):
            local = slice(rank * 8, (rank + 1) * 8)

            def gather(tensors, *, reduce_dtype):
                for actual, full in zip(
                    tensors,
                    (q, weights, document_starts, mask_values == 0),
                    strict=True,
                ):
                    torch.testing.assert_close(actual, full[local])
                return q, weights, document_starts, mask_values == 0

            def shard(indices, group, *, src, dst):
                self.assertEqual(src, spmd.R)
                self.assertEqual(dst, spmd.S(0))
                torch.testing.assert_close(indices, expected)
                return indices[local]

            with (
                patch(
                    "torchtitan.models.glm5.dsa.spmd_mesh_group",
                    return_value=SimpleNamespace(rank=lambda: rank),
                ),
                patch(
                    "torchtitan.models.glm5.dsa.cp_all_gather_tensors",
                    side_effect=gather,
                ),
                patch(
                    "torchtitan.models.glm5.dsa.spmd.redistribute", side_effect=shard
                ),
            ):
                actual = indexer(q[local], k, weights[local], mask[0, local])
            torch.testing.assert_close(actual, expected[local])

    def test_indexer_layouts_match_tensor_ranks(self) -> None:
        config = glm5_indexer_topk_sharding_config()
        self.assertEqual(len(config.in_src_shardings["q_QNH"].partition_spec), 3)
        for name in ("k_KH", "weights_QN", "attention_mask_QK"):
            self.assertEqual(len(config.in_src_shardings[name].partition_spec), 2)

    def test_indexer_requires_complete_sequences(self) -> None:
        indexer = DSAIndexerTopK.Config(index_topk=2).build()
        with self.assertRaisesRegex(ValueError, "complete query and key sequences"):
            indexer(
                torch.ones(2, 1, 8),
                torch.ones(4, 8),
                torch.ones(2, 1),
                torch.zeros(2, 4),
            )

    def test_packed_document_causal_mask(self) -> None:
        positions = torch.tensor([0, 1, 2, 0, 1])
        mask = create_dsa_causal_mask(positions, dtype=torch.float32)
        allowed = mask[0] == 0
        expected = torch.tensor(
            [
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 1, 1],
            ],
            dtype=torch.bool,
        )
        torch.testing.assert_close(allowed, expected)

    def test_selected_mask_intersects_topk_and_causality(self) -> None:
        positions = torch.arange(4)
        additive = create_dsa_causal_mask(positions, dtype=torch.float32)
        indices = torch.tensor([[0, -1], [1, 0], [3, 1], [3, 3]])
        actual = selected_mask(indices, additive)
        expected = torch.tensor(
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.bool,
        )
        torch.testing.assert_close(actual, expected)

    def test_padding_mask_excludes_queries_and_keys(self) -> None:
        mask = create_dsa_causal_mask(
            torch.arange(4),
            dtype=torch.float32,
            padding_mask_T=torch.tensor([True, True, False, False]),
        )
        allowed = mask[0] == 0
        expected = torch.tensor(
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=torch.bool,
        )
        torch.testing.assert_close(allowed, expected)


if __name__ == "__main__":
    unittest.main()
