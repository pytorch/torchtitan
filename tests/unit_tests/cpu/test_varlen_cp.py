# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for K/V all-gather varlen context parallelism."""

from types import SimpleNamespace
from unittest import mock

import spmd_types as spmd
import torch
from torch.distributed.tensor.experimental._attention import _HeadTailLoadBalancer
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.distributed.cuda_graph import CUDAGraphInputSpec
from torchtitan.models.common.attention import VarlenInnerAttention, VarlenMetadata
from torchtitan.models.common.cp_attention import (
    HeadTailCPVarlenMetadata,
    KVAllGatherCPVarlenInnerAttention,
)


def _metadata(offsets: list[int], seq_len: int) -> VarlenMetadata:
    if offsets[0] != 0 or offsets[-1] != seq_len:
        raise ValueError("Offsets must start at 0 and end at seq_len.")
    cu_seq = torch.tensor(offsets, dtype=torch.int32)
    max_seq = int(torch.diff(cu_seq).max().item())
    return VarlenMetadata(cu_seq, cu_seq, max_seq, max_seq)


class TestHeadTailCPVarlenMetadata(TestCase):
    @staticmethod
    def _from_global(
        metadata: VarlenMetadata,
        *,
        permutation: torch.Tensor | None,
    ) -> HeadTailCPVarlenMetadata:
        return HeadTailCPVarlenMetadata.from_global(
            metadata,
            permutation=permutation,
        )

    def test_restores_head_tail_order(self) -> None:
        global_metadata = _metadata([0, 3, 7, 8], 8)
        permutation = _HeadTailLoadBalancer(
            8, 2, torch.device("cpu")
        )._generate_indices()

        metadata = self._from_global(
            global_metadata,
            permutation=permutation,
        )

        reordered = torch.arange(8).index_select(0, permutation[0])
        self.assertEqual(
            reordered.index_select(0, metadata.kv_restore_indices),
            torch.arange(8),
        )

    def test_builds_fixed_chunk_metadata(self) -> None:
        metadata = self._from_global(
            _metadata([0, 3, 7, 8], 8),
            permutation=_HeadTailLoadBalancer(
                8, 2, torch.device("cpu")
            )._generate_indices(),
        )

        chunk = metadata.chunk_metadata(start=2, end=6, kv_start=0)

        self.assertEqual(chunk.cu_seq_q, torch.tensor([0, 1, 4, 4], dtype=torch.int32))
        self.assertEqual(chunk.cu_seq_k, torch.tensor([0, 3, 6, 6], dtype=torch.int32))
        self.assertEqual(chunk.max_q, 4)
        self.assertEqual(chunk.max_k, 4)

    def test_builds_sliding_window_metadata(self) -> None:
        metadata = self._from_global(
            _metadata([0, 3, 7, 8], 8),
            permutation=_HeadTailLoadBalancer(
                8, 2, torch.device("cpu")
            )._generate_indices(),
        )

        chunk = metadata.chunk_metadata(start=4, end=6, kv_start=2)

        self.assertEqual(chunk.cu_seq_q, torch.tensor([0, 0, 2, 2], dtype=torch.int32))
        self.assertEqual(chunk.cu_seq_k, torch.tensor([0, 1, 4, 4], dtype=torch.int32))

    def test_rejects_missing_or_invalid_permutation(self) -> None:
        global_metadata = _metadata([0, 8], 8)
        with self.assertRaisesRegex(ValueError, "requires a permutation"):
            self._from_global(global_metadata, permutation=None)
        with self.assertRaisesRegex(ValueError, "requires a permutation"):
            self._from_global(
                global_metadata,
                permutation=torch.arange(8, dtype=torch.int32),
            )

    def test_rejects_cross_attention(self) -> None:
        metadata = VarlenMetadata(
            cu_seq_q=torch.tensor([0, 4, 8], dtype=torch.int32),
            cu_seq_k=torch.tensor([0, 3, 8], dtype=torch.int32),
            max_q=4,
            max_k=5,
        )
        permutation = _HeadTailLoadBalancer(
            8, 2, torch.device("cpu")
        )._generate_indices()
        with self.assertRaisesRegex(ValueError, "self-attention"):
            self._from_global(metadata, permutation=permutation)


class TestVarlenMetadataCudaGraphInputs(TestCase):
    @staticmethod
    def _cp_metadata() -> HeadTailCPVarlenMetadata:
        return HeadTailCPVarlenMetadata(
            varlen_metadata=_metadata([0, 4], 4),
            kv_restore_indices=torch.arange(4, dtype=torch.int32),
        )

    def test_metadata_tensors_are_graph_inputs(self) -> None:
        leaves = CUDAGraphInputSpec({"metadata": self._cp_metadata()}).flatten(
            {"metadata": self._cp_metadata()}
        )
        tensors = [leaf for leaf in leaves if isinstance(leaf, torch.Tensor)]
        self.assertEqual(3, len(tensors))

    def test_rebuilt_metadata_has_stable_non_tensor_leaves(self) -> None:
        spec = CUDAGraphInputSpec({"metadata": self._cp_metadata()})
        captured = spec.flatten({"metadata": self._cp_metadata()})
        rebuilt = spec.flatten({"metadata": self._cp_metadata()})
        for first, second in zip(captured, rebuilt, strict=True):
            if not isinstance(first, torch.Tensor):
                self.assertEqual(first, second)


class TestKVAllGatherCPVarlenInnerAttention(TestCase):
    @staticmethod
    def _cp_metadata() -> HeadTailCPVarlenMetadata:
        permutation = _HeadTailLoadBalancer(
            8, 2, torch.device("cpu")
        )._generate_indices()
        restore_indices = torch.empty_like(permutation[0])
        restore_indices[permutation[0]] = torch.arange(8, dtype=torch.int32)
        return HeadTailCPVarlenMetadata(
            varlen_metadata=_metadata([0, 3, 7, 8], 8),
            kv_restore_indices=restore_indices,
        )

    def _run_forward(self, window_size: tuple[int, int]):
        attention = KVAllGatherCPVarlenInnerAttention(
            KVAllGatherCPVarlenInnerAttention.Config(window_size=window_size)
        )
        q_THK = torch.randn(4, 1, 4)
        gathered_k_THK = torch.arange(32).view(8, 1, 4).float()
        gathered_v_THV = gathered_k_THK + 100
        permutation = _HeadTailLoadBalancer(
            8, 2, torch.device("cpu")
        )._generate_indices()[0]
        permuted_k_THK = gathered_k_THK.index_select(0, permutation)
        permuted_v_THV = gathered_v_THV.index_select(0, permutation)
        outputs = [torch.randn(2, 1, 4), torch.randn(2, 1, 4)]
        group = SimpleNamespace(size=lambda: 2)

        with mock.patch.object(
            attention,
            "_all_gather_kv",
            return_value=(permuted_k_THK, permuted_v_THV),
        ), mock.patch.object(
            VarlenInnerAttention,
            "forward",
            autospec=True,
            side_effect=outputs,
        ) as inner_forward, mock.patch(
            "torchtitan.models.common.cp_attention.spmd_mesh_group",
            return_value=group,
        ), mock.patch(
            "torchtitan.models.common.cp_attention.dist.get_rank",
            return_value=0,
        ), mock.patch.object(
            spmd, "is_type_checking", return_value=False
        ):
            actual = attention(
                q_THK,
                torch.empty(0),
                torch.empty(0),
                attention_masks=self._cp_metadata(),
            )

        self.assertEqual(actual, torch.cat(outputs))
        return inner_forward.call_args_list, gathered_k_THK, gathered_v_THV

    def test_full_causal_uses_two_views_of_shared_kv(self) -> None:
        calls, gathered_k_THK, gathered_v_THV = self._run_forward((-1, 0))

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].args[2], gathered_k_THK[:2])
        self.assertEqual(calls[0].args[3], gathered_v_THV[:2])
        self.assertEqual(calls[1].args[2], gathered_k_THK[:8])
        self.assertEqual(calls[1].args[3], gathered_v_THV[:8])

    def test_sliding_window_uses_bounded_kv_views(self) -> None:
        calls, gathered_k_THK, gathered_v_THV = self._run_forward((2, 0))

        self.assertEqual(calls[0].args[2], gathered_k_THK[:2])
        self.assertEqual(calls[0].args[3], gathered_v_THV[:2])
        self.assertEqual(calls[1].args[2], gathered_k_THK[4:8])
        self.assertEqual(calls[1].args[3], gathered_v_THV[4:8])


if __name__ == "__main__":
    run_tests()
