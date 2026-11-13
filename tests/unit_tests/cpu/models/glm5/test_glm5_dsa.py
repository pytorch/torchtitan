# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from torchtitan.models.glm5.dsa import (
    DSAIndexerTopK,
    DsaMetadata,
    Glm5CPMLAAttention,
    Glm5SparseAttention,
)
from torchtitan.models.glm5.sharding import glm5_indexer_topk_sharding_config
from torchtitan.models.glm5.indexer import select_local_topk


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

    def test_indexer_uses_bounded_reference_without_gradients(self) -> None:
        q = torch.ones(4, 2, 8, requires_grad=True)
        k = torch.ones(4, 8, requires_grad=True)
        weights = torch.ones(4, 2, requires_grad=True)
        indexer = DSAIndexerTopK.Config(index_topk=2).build()
        # Exercise the same local-region construction as distributed parallelize().
        indexer._sharding_config = glm5_indexer_topk_sharding_config()
        indexer.forward = indexer._maybe_wrap_with_local_region(indexer.forward)
        with patch(
            "torchtitan.models.glm5.dsa.select_local_topk", wraps=select_local_topk
        ) as kernel:
            indices = indexer(
                q, k, weights, metadata=DsaMetadata.from_positions(torch.arange(4))
            )
        kernel.assert_called_once()
        self.assertEqual(kernel.call_args.args[0].dtype, torch.float32)
        self.assertFalse(indices.requires_grad)

    def test_unsharded_reference_crosses_key_tiles_with_packed_padding(self) -> None:
        count = 1031
        positions = torch.cat((torch.arange(13), torch.arange(count - 13)))
        padding = torch.zeros(count, dtype=torch.bool)
        padding[[12, count - 1]] = True
        q = torch.ones(count, 1, 8)
        k = (torch.arange(count).float() + 1)[:, None].expand(-1, 8)
        metadata = DsaMetadata.from_positions(positions, padding)
        with patch(
            "torchtitan.models.glm5.dsa.lightning_indexer",
            side_effect=AssertionError("reference must not allocate stock full scores"),
        ):
            actual = DSAIndexerTopK.Config(index_topk=4).build()(
                q, k, torch.ones(count, 1), metadata=metadata
            )
        for query in (0, 11, 12, 13, 1024, 1029, 1030):
            start = 0 if query < 13 else 13
            candidates = [] if padding[query] else list(range(start, query + 1))[-4:]
            expected = torch.tensor(
                [-1] * (4 - len(candidates)) + candidates, dtype=torch.int32
            )
            torch.testing.assert_close(actual[query].sort().values, expected)

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
        q = torch.rand(16, 4, 8) + 0.5
        k = torch.arange(1, 17, dtype=torch.float32)[:, None].expand(16, 8)
        weights = torch.ones(16, 4)
        positions = torch.cat((torch.arange(5), torch.arange(11)))
        padding = torch.zeros(16, dtype=torch.bool)
        padding[[3, 14]] = True
        padding[12:] = True
        metadata = DsaMetadata.from_positions(positions, padding)
        indexer = DSAIndexerTopK.Config(index_topk=4).build()
        expected = indexer(q, k, weights, metadata=metadata)
        attention = Glm5SparseAttention.Config().build()
        full_inputs = (
            torch.randn(16, 2, 8, requires_grad=True),
            torch.randn(16, 8, requires_grad=True),
        )
        full_output = attention(
            *full_inputs, attention_metadata=metadata, topk_indices_QS=expected,
            scale=0.25,
        )
        for size, rank in [(size, rank) for size in (2, 4) for rank in range(size)]:
            width = 16 // size
            local = slice(rank * width, (rank + 1) * width)
            group = SimpleNamespace(rank=lambda: rank, size=lambda: size)
            with patch(
                "torchtitan.models.glm5.dsa.spmd_mesh_group", return_value=group
            ):
                local_metadata = Glm5CPMLAAttention.prepare_cp_metadata(
                    metadata, permutation=None
                )

            with (
                patch("torchtitan.models.glm5.dsa.spmd_mesh_group", return_value=group),
                patch(
                    "torchtitan.models.glm5.dsa.select_local_topk",
                    wraps=select_local_topk,
                ) as selector,
                patch(
                    "torchtitan.models.glm5.dsa.spmd.redistribute",
                    side_effect=AssertionError("top-k must not gather Q or shard full results"),
                ),
            ):
                actual = indexer(
                    q[local], k, weights[local], metadata=local_metadata
                )
            args, _ = selector.call_args
            self.assertEqual(args[0].shape[0], width)
            self.assertEqual(args[1].shape[0], 16)
            self.assertEqual(args[3].shape, (width, 2))
            # Compare token-level membership; top-k ties need not have equal order.
            torch.testing.assert_close(
                actual.to(torch.int32).sort(-1).values,
                expected[local].sort(-1).values,
            )
            local_inputs = (
                full_inputs[0][local].detach().clone().requires_grad_(),
                full_inputs[1].detach().clone().requires_grad_(),
            )
            output = attention(
                *local_inputs, attention_metadata=local_metadata,
                topk_indices_QS=actual, scale=0.25,
            )
            torch.testing.assert_close(output, full_output[local], rtol=1e-5, atol=1e-6)
            gradients = torch.autograd.grad(output.sum(), local_inputs)
            reference_gradients = torch.autograd.grad(
                full_output[local].sum(), full_inputs, retain_graph=True
            )
            torch.testing.assert_close(
                gradients[0], reference_gradients[0][local], rtol=1e-5, atol=1e-6
            )
            torch.testing.assert_close(
                gradients[1:], reference_gradients[1:], rtol=1e-5, atol=1e-6
            )

    def test_cp_rejects_permutation(self) -> None:
        metadata = DsaMetadata.from_positions(torch.arange(8))
        with self.assertRaisesRegex(ValueError, "without a permutation"):
            Glm5CPMLAAttention.prepare_cp_metadata(
                metadata, permutation=torch.arange(7, -1, -1)
            )

    def test_indexer_requires_complete_sequences(self) -> None:
        indexer = DSAIndexerTopK.Config(index_topk=2).build()
        with self.assertRaisesRegex(ValueError, "complete query and key sequences"):
            indexer(
                torch.ones(2, 1, 8),
                torch.ones(4, 8),
                torch.ones(2, 1),
                metadata=DsaMetadata.from_positions(torch.arange(4)),
            )

    def test_gather_matches_dense_output_and_gradients_with_packed_padding(
        self,
    ) -> None:
        positions = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        padding = torch.tensor([False, False, False, True, False, False, True, True])
        original = DsaMetadata.from_positions(positions, padding)
        metadata = DsaMetadata(
            positions,
            original.documents_T,
            ~padding,
            torch.tensor([2, 3, 4, 5]),
        )
        torch.manual_seed(42)
        qi, ki, weights = torch.randn(8, 2, 8), torch.randn(8, 8), torch.randn(8, 2)
        selected = DSAIndexerTopK.Config(index_topk=4).build()(
            qi, ki, weights, metadata=original
        )[2:6]
        k = torch.arange(8)[None, :]
        actual = (selected[:, :, None] == k).any(dim=1)
        queries = metadata.query_indices_Q
        legal = (
            metadata.valid_T[queries, None]
            & metadata.valid_T[None, :]
            & (metadata.documents_T[queries, None] == metadata.documents_T[None, :])
            & (metadata.positions_T[queries, None] >= metadata.positions_T[None, :])
        )
        scores = (
            torch.relu(torch.einsum("qhd,kd->qhk", qi[2:6], ki))
            * weights[2:6, :, None]
        ).sum(1)
        indices = scores.masked_fill(~legal, -torch.inf).topk(4, dim=-1).indices
        expected = (indices[:, :, None] == k).any(dim=1) & legal
        torch.testing.assert_close(actual, expected)
        torch.manual_seed(42)
        q = torch.randn(4, 2, 8, requires_grad=True)
        k = torch.randn(8, 8, requires_grad=True)
        reference = [tensor.detach().clone().requires_grad_() for tensor in (q, k)]
        output = Glm5SparseAttention.Config().build()(
            q, k, attention_metadata=metadata, topk_indices_QS=selected, scale=0.25
        )
        logits = torch.einsum("qhd,kd->hqk", reference[0], reference[1]) * 0.25
        probabilities = logits.masked_fill(~expected[None], -torch.inf).softmax(-1)
        probabilities = probabilities.nan_to_num()
        dense = torch.einsum("hqk,kd->qhd", probabilities, reference[1])
        torch.testing.assert_close(output, dense, atol=1e-6, rtol=1e-5)
        output.sum().backward()
        # Empty rows must not feed NaN derivatives through the oracle softmax.
        safe_logits = logits.masked_fill(~expected[None], -1e30)
        safe_probabilities = safe_logits.softmax(-1) * expected.any(-1)[None, :, None]
        safe_dense = torch.einsum("hqk,kd->qhd", safe_probabilities, reference[1])
        safe_dense.sum().backward()
        for actual_tensor, expected_tensor in zip((q, k), reference, strict=True):
            torch.testing.assert_close(
                actual_tensor.grad, expected_tensor.grad, atol=1e-6, rtol=1e-5
            )


if __name__ == "__main__":
    unittest.main()
