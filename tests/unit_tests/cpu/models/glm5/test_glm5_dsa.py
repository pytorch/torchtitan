# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from torchtitan.models.common.attn_gym import lightning_indexer
from torchtitan.models.glm5.dsa import (
    DSAIndexerTopK,
    DsaMetadata,
    filter_dsa_indices,
    Glm5CPMLAAttention,
    Glm5SparseAttention,
)
from torchtitan.models.glm5.sharding import glm5_indexer_topk_sharding_config


def create_dsa_causal_mask(
    positions_T: torch.Tensor,
    *,
    dtype: torch.dtype,
    padding_mask_T: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a packed-document causal additive mask of shape ``[1, T, T]``.

    ``positions_T`` restarts at zero for every packed document. The cumulative
    restart count is therefore a document id. A query can see a key only when
    both tokens belong to the same document and the key is not in the future.
    Disallowed entries use the minimum finite value for indexer ranking.
    The attention mask also excludes them after top-k selection.
    """
    T = positions_T.shape[0]
    token_indices_T = torch.arange(T, device=positions_T.device)
    document_ids_T = (positions_T == 0).cumsum(0)
    allowed_TT = (token_indices_T[:, None] >= token_indices_T[None, :]) & (
        document_ids_T[:, None] == document_ids_T[None, :]
    )
    if padding_mask_T is not None:
        allowed_TT = allowed_TT & ~padding_mask_T[:, None] & ~padding_mask_T[None, :]
    return torch.zeros(1, T, T, dtype=dtype, device=positions_T.device).masked_fill(
        ~allowed_TT.unsqueeze(0), torch.finfo(dtype).min
    )


def selected_mask(
    indices_QS: torch.Tensor, additive_mask_1QK: torch.Tensor
) -> torch.Tensor:
    """Intersect selected positions with the existing causal/document mask.

    Negative entries denote padding. Duplicate indices represent a set, not
    repeated probability mass. Invalid positive indices are excluded as well.
    The supplied mask uses zero for allowed entries, not arbitrary score bias.
    """
    Q, K = additive_mask_1QK.shape[-2:]
    valid_QS = (indices_QS >= 0) & (indices_QS < K)
    safe_QS = indices_QS.clamp(0, K - 1).long()
    counts_QK = torch.zeros((Q, K), dtype=torch.int32, device=indices_QS.device)
    counts_QK.scatter_add_(1, safe_QS, valid_QS.to(torch.int32))
    return (counts_QK > 0) & (additive_mask_1QK[0] == 0)


class TestGlm5Dsa(unittest.TestCase):
    @staticmethod
    def _dense_indices(q, k, weights, mask, topk):
        scores = (q.float().transpose(0, 1) @ k.float().T) * q.shape[-1] ** -0.5
        scores = scores.relu()
        weights = weights.float() * q.shape[1] ** -0.5
        scores = (weights.unsqueeze(1) @ scores.transpose(0, 1)).squeeze(1)
        return (scores + mask[0]).topk(min(topk, k.shape[0]), dim=-1).indices

    def test_collator_padding_reaches_model_mask(self) -> None:
        from torchtitan.components.data.collators import TextCollator
        from torchtitan.components.data.dataset import TextSequence
        from torchtitan.config.parallelism import ParallelismConfig
        from torchtitan.models.glm5 import build_model_config

        collator = TextCollator(
            TextCollator.Config(),
            context=SimpleNamespace(
                num_tokens_per_microbatch=8, max_context_length=8, num_mtp_layers=0
            ),
        )
        batch = collator([TextSequence(input_ids=[1, 2, 3], labels=[2, 3, 4])])
        model = build_model_config(
            "debugmodel", parallelism=ParallelismConfig(), seq_len=8
        ).build()
        metadata = model.get_attention_masks(batch.positions, batch.padding_mask)
        mask = (
            (metadata.documents_T[:, None] == metadata.documents_T[None, :])
            & (metadata.positions_T[:, None] >= metadata.positions_T[None, :])
            & metadata.valid_T[:, None]
            & metadata.valid_T[None, :]
        )
        expected = torch.zeros(8, 8, dtype=torch.bool)
        expected[:3, :3] = torch.ones(3, 3, dtype=torch.bool).tril()
        torch.testing.assert_close(mask, expected)

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
                    positions,
                    dtype=torch.float32,
                    padding_mask_T=None if valid is None else ~valid,
                )
                for topk in (1, 4, 64):
                    with self.subTest(dtype=dtype, positions=positions, topk=topk):
                        indexer = DSAIndexerTopK.Config(index_topk=topk).build()
                        actual = indexer(
                            q,
                            k,
                            weights,
                            metadata=DsaMetadata.from_positions(
                                positions, None if valid is None else ~valid
                            ),
                        )
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
            padding_mask_T=torch.tensor([False, True, False, False]),
        )
        actual = DSAIndexerTopK.Config(index_topk=2).build()(
            q,
            k,
            weights,
            metadata=DsaMetadata.from_positions(
                torch.arange(4), torch.tensor([False, True, False, False])
            ),
        )
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
        # Exercise the same local-region construction as distributed parallelize().
        indexer._sharding_config = glm5_indexer_topk_sharding_config()
        indexer.forward = indexer._maybe_wrap_with_local_region(indexer.forward)
        with patch(
            "torchtitan.models.glm5.dsa.lightning_indexer", wraps=lightning_indexer
        ) as kernel:
            indices = indexer(
                q, k, weights, metadata=DsaMetadata.from_positions(torch.arange(4))
            )
        kernel.assert_called_once()
        self.assertEqual(kernel.call_args.kwargs["impl"], "reference")
        self.assertEqual(kernel.call_args.args[0].dtype, torch.float32)
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
                positions,
                dtype=torch.float32,
                padding_mask_T=None if valid is None else ~valid,
            )[0]
            torch.testing.assert_close(
                compiled(
                    q,
                    k,
                    weights,
                    metadata=DsaMetadata.from_positions(
                        positions, None if valid is None else ~valid
                    ),
                ),
                indexer(
                    q,
                    k,
                    weights,
                    metadata=DsaMetadata.from_positions(
                        positions, None if valid is None else ~valid
                    ),
                ),
            )

    def test_cp_indexer_returns_local_queries_with_global_key_indices(self) -> None:
        torch.manual_seed(42)
        q, k, weights = torch.randn(16, 4, 8), torch.randn(16, 8), torch.randn(16, 4)
        metadata = DsaMetadata.from_positions(torch.arange(8).repeat(2))
        indexer = DSAIndexerTopK.Config(index_topk=4).build()
        expected = indexer(q, k, weights, metadata=metadata)
        # Head-tail style permutation, mixing tokens from both documents.
        order = torch.tensor([0, 1, 14, 15, 2, 3, 12, 13, 4, 5, 10, 11, 6, 7, 8, 9])
        for rank in range(2):
            local = slice(rank * 8, (rank + 1) * 8)
            group = SimpleNamespace(rank=lambda: rank, size=lambda: 2)
            with patch(
                "torchtitan.models.glm5.dsa.spmd_mesh_group", return_value=group
            ):
                permuted = Glm5CPMLAAttention.prepare_cp_metadata(
                    metadata, permutation=order
                )

            def gather(tensors, *, reduce_dtype):
                return q[order], weights[order]

            def shard(indices, group, *, src, dst):
                return indices[local]

            with (
                patch("torchtitan.models.glm5.dsa.spmd_mesh_group", return_value=group),
                patch(
                    "torchtitan.models.glm5.dsa.cp_all_gather_tensors",
                    side_effect=gather,
                ),
                patch(
                    "torchtitan.models.glm5.dsa.spmd.redistribute", side_effect=shard
                ),
            ):
                actual = indexer(
                    q[order][local], k[order], weights[order][local], metadata=permuted
                )
            # Compare token-level membership; top-k ties need not have equal order.
            actual_original = torch.where(
                actual >= 0, order[actual.clamp_min(0).long()], -1
            )
            torch.testing.assert_close(
                actual_original.to(torch.int32).sort(-1).values,
                expected[order][local].sort(-1).values,
            )

    def test_indexer_layouts_match_tensor_ranks(self) -> None:
        config = glm5_indexer_topk_sharding_config()
        self.assertEqual(len(config.in_src_shardings["q_QNH"].partition_spec), 3)
        for name in ("k_KH", "weights_QN"):
            self.assertEqual(len(config.in_src_shardings[name].partition_spec), 2)

    def test_indexer_requires_complete_sequences(self) -> None:
        indexer = DSAIndexerTopK.Config(index_topk=2).build()
        with self.assertRaisesRegex(ValueError, "complete query and key sequences"):
            indexer(
                torch.ones(2, 1, 8),
                torch.ones(4, 8),
                torch.ones(2, 1),
                metadata=DsaMetadata.from_positions(torch.arange(4)),
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

    def test_gather_matches_dense_output_and_gradients_with_permuted_padding(
        self,
    ) -> None:
        positions = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        padding = torch.tensor([False, False, False, True, False, False, True, True])
        order = torch.tensor([0, 7, 1, 6, 2, 5, 3, 4])
        original = DsaMetadata.from_positions(positions, padding)
        metadata = DsaMetadata(
            positions[order],
            original.documents_T[order],
            ~padding[order],
            torch.tensor([2, 3, 4, 5]),
            order,
        )
        indices = torch.tensor(
            [[0, 0, 2, -1], [3, 6, 99, -1], [4, 2, 0, 7], [5, 7, 1, -2]]
        )
        selected = filter_dsa_indices(indices, metadata)
        k = torch.arange(8)[None, :]
        actual = (selected[:, :, None] == k).any(dim=1)
        queries = metadata.query_indices_Q
        legal = (
            metadata.valid_T[queries, None]
            & metadata.valid_T[None, :]
            & (metadata.documents_T[queries, None] == metadata.documents_T[None, :])
            & (metadata.positions_T[queries, None] >= metadata.positions_T[None, :])
        )
        expected = (indices[:, :, None] == k).any(dim=1) & legal
        torch.testing.assert_close(actual, expected)
        torch.manual_seed(42)
        q = torch.randn(4, 2, 8, requires_grad=True)
        k = torch.randn(8, 2, 8, requires_grad=True)
        v = torch.randn(8, 2, 4, requires_grad=True)
        reference = [tensor.detach().clone().requires_grad_() for tensor in (q, k, v)]
        output = Glm5SparseAttention.Config().build()(
            q, k, v, attention_metadata=metadata, topk_indices_QS=indices, scale=0.25
        )
        logits = torch.einsum("qhd,khd->hqk", reference[0], reference[1]) * 0.25
        probabilities = logits.masked_fill(~expected[None], -torch.inf).softmax(-1)
        probabilities = probabilities.nan_to_num()
        dense = torch.einsum("hqk,khv->qhv", probabilities, reference[2])
        torch.testing.assert_close(output, dense, atol=1e-6, rtol=1e-5)
        output.sum().backward()
        # Empty rows must not feed NaN derivatives through the oracle softmax.
        safe_logits = logits.masked_fill(~expected[None], -1e30)
        safe_probabilities = safe_logits.softmax(-1) * expected.any(-1)[None, :, None]
        safe_dense = torch.einsum("hqk,khv->qhv", safe_probabilities, reference[2])
        safe_dense.sum().backward()
        for actual_tensor, expected_tensor in zip((q, k, v), reference, strict=True):
            torch.testing.assert_close(
                actual_tensor.grad, expected_tensor.grad, atol=1e-6, rtol=1e-5
            )

    def test_padding_mask_excludes_queries_and_keys(self) -> None:
        mask = create_dsa_causal_mask(
            torch.arange(4),
            dtype=torch.float32,
            padding_mask_T=torch.tensor([False, False, True, True]),
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
