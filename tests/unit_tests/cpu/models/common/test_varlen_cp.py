# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for K/V all-gather varlen context parallelism."""

from types import SimpleNamespace
from typing import NamedTuple
from unittest import mock

import spmd_types as spmd
import torch
import torch.distributed as dist
from spmd_types.checker import typecheck
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.experimental._attention import _HeadTailLoadBalancer
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed.fake_pg import FakeStore

from torchtitan.distributed.accelerator_graph import AcceleratorGraphInputSpec
from torchtitan.distributed.spmd_types import set_current_spmd_mesh
from torchtitan.models.common.attention import (
    cp_attention,
    VarlenAttentionMetadata,
    VarlenInnerAttention,
)
from torchtitan.models.common.attention.cp_attention import (
    AllGatherKVSlidingWindowVarlenInnerAttention,
    AllGatherKVVarlen,
    AllGatherKVVarlenAttentionMetadata,
    AllGatherKVVarlenInnerAttention,
)


def _metadata(offsets: list[int]) -> VarlenAttentionMetadata:
    cu_seq = torch.tensor(offsets, dtype=torch.int32)
    max_seq = int(torch.diff(cu_seq).max().item())
    return VarlenAttentionMetadata(cu_seq, cu_seq, max_seq, max_seq)


class TestAllGatherKVVarlenPreparation(TestCase):
    def test_restores_head_tail_order(self) -> None:
        permutation = _HeadTailLoadBalancer(
            8, 2, torch.device("cpu")
        )._generate_indices()
        metadata = AllGatherKVVarlen.prepare_cp_metadata(
            _metadata([0, 3, 7, 8]), permutation=permutation
        )
        reordered = torch.arange(8).index_select(0, permutation[0])
        self.assertEqual(
            reordered.index_select(0, metadata.inverse_permutation), torch.arange(8)
        )

    def test_builds_chunk_metadata(self) -> None:
        metadata = AllGatherKVVarlen.prepare_cp_metadata(
            _metadata([0, 3, 7, 8]), permutation=None
        )
        self.assertIsNone(metadata.inverse_permutation)
        for kv_start, expected_k in ((0, [0, 3, 6, 6]), (2, [0, 1, 4, 4])):
            with self.subTest(kv_start=kv_start):
                chunk = AllGatherKVVarlen._chunk_metadata(
                    metadata.varlen_metadata, q_start=4, end=6, kv_start=kv_start
                )
                self.assertEqual(
                    chunk.cu_seq_q, torch.tensor([0, 0, 2, 2], dtype=torch.int32)
                )
                self.assertEqual(
                    chunk.cu_seq_k, torch.tensor(expected_k, dtype=torch.int32)
                )
                self.assertEqual(chunk.max_q, 2)
                self.assertEqual(chunk.max_k, 4)

    def test_chunk_metadata_spmd_types(self) -> None:
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=8)
        self.addCleanup(dist.destroy_process_group)
        mesh = init_device_mesh("cpu", (2, 2, 2), mesh_dim_names=("dp", "cp", "tp"))
        global_metadata = _metadata([0, 3, 7, 8])
        metadata = AllGatherKVVarlen.prepare_cp_metadata(
            global_metadata, permutation=None
        )
        with set_current_spmd_mesh(mesh), typecheck():
            chunk = AllGatherKVVarlen._chunk_metadata(
                metadata.varlen_metadata, q_start=4, end=6, kv_start=2
            )
            spmd.assert_type(
                global_metadata.cu_seq_q,
                {"dp": spmd.V, "cp": spmd.R, "tp": spmd.R},
                partition_spec=spmd.PartitionSpec("dp"),
            )
            spmd.assert_type(
                (chunk.cu_seq_q, chunk.cu_seq_k),
                {"dp": spmd.V, "cp": spmd.V, "tp": spmd.R},
                partition_spec=spmd.PartitionSpec("dp"),
            )

    def test_rejects_invalid_permutation(self) -> None:
        for permutation in (torch.arange(8), torch.arange(8).repeat(2, 1)):
            with self.subTest(shape=permutation.shape):
                with self.assertRaisesRegex(ValueError, "requires a permutation"):
                    AllGatherKVVarlen.prepare_cp_metadata(
                        _metadata([0, 8]), permutation=permutation
                    )

    def test_rejects_cross_attention(self) -> None:
        metadata = VarlenAttentionMetadata(
            cu_seq_q=torch.tensor([0, 4, 8], dtype=torch.int32),
            cu_seq_k=torch.tensor([0, 3, 8], dtype=torch.int32),
            max_q=4,
            max_k=5,
        )
        with self.assertRaisesRegex(ValueError, "self-attention"):
            AllGatherKVVarlen.prepare_cp_metadata(metadata, permutation=None)

    def test_rejects_noncausal_windows(self) -> None:
        for window_size in ((-1, -1), (2, 1), (-2, 0)):
            with self.subTest(window_size=window_size):
                metadata = _metadata([0, 8])._replace(window_size=window_size)
                with self.assertRaisesRegex(ValueError, "requires a causal window"):
                    AllGatherKVVarlen.prepare_cp_metadata(metadata, permutation=None)


class TestVarlenMetadataCudaGraphInputs(TestCase):
    def test_metadata_inputs_are_stable(self) -> None:
        for inverse_permutation in (None, torch.arange(4, dtype=torch.int32)):
            with self.subTest(head_tail=inverse_permutation is not None):
                captured_metadata = AllGatherKVVarlenAttentionMetadata(
                    varlen_metadata=_metadata([0, 4]),
                    inverse_permutation=inverse_permutation,
                )
                rebuilt_metadata = AllGatherKVVarlenAttentionMetadata(
                    varlen_metadata=_metadata([0, 4]),
                    inverse_permutation=inverse_permutation,
                )
                spec = AcceleratorGraphInputSpec({"metadata": captured_metadata})
                captured = spec.flatten({"metadata": captured_metadata})
                rebuilt = spec.flatten({"metadata": rebuilt_metadata})
                self.assertEqual(
                    spec.unflatten(rebuilt), {"metadata": rebuilt_metadata}
                )
                tensors = [leaf for leaf in captured if isinstance(leaf, torch.Tensor)]
                self.assertEqual(len(tensors), 2 if inverse_permutation is None else 3)
                for first, second in zip(captured, rebuilt, strict=True):
                    if not isinstance(first, torch.Tensor):
                        self.assertEqual(first, second)


class _ForwardCase(NamedTuple):
    head_tail: bool
    window: tuple[int, int]
    kv_ranges: tuple[tuple[int, int], ...]
    rank: int = 0


class TestKVAllGatherVarlenInnerAttention(TestCase):
    def test_forward_uses_bounded_shared_kv_views(self) -> None:
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=8)
        self.addCleanup(dist.destroy_process_group)
        mesh = init_device_mesh("cpu", (2, 2, 2), mesh_dim_names=("dp", "cp", "tp"))
        seq_len = 8
        cases = (
            _ForwardCase(head_tail=True, window=(-1, 0), kv_ranges=((0, 2), (2, 8))),
            _ForwardCase(head_tail=True, window=(10, 0), kv_ranges=((0, 2), (2, 8))),
            _ForwardCase(head_tail=True, window=(2, 0), kv_ranges=((0, 2), (4, 8))),
            _ForwardCase(head_tail=False, window=(-1, 0), kv_ranges=((0, 8),), rank=1),
            _ForwardCase(head_tail=False, window=(2, 0), kv_ranges=((2, 8),), rank=1),
        )
        with set_current_spmd_mesh(mesh):
            for case in cases:
                with self.subTest(case=case):
                    config = (
                        AllGatherKVVarlenInnerAttention.Config()
                        if case.window == (-1, 0)
                        else AllGatherKVSlidingWindowVarlenInnerAttention.Config(
                            sliding_window_size=case.window[0] + 1
                        )
                    )
                    attention = config.build()
                    permutation = (
                        _HeadTailLoadBalancer(
                            seq_len, 2, torch.device("cpu")
                        )._generate_indices()
                        if case.head_tail
                        else None
                    )
                    metadata = AllGatherKVVarlen.prepare_cp_metadata(
                        _metadata([0, 3, 7, 8])._replace(window_size=case.window),
                        permutation=permutation,
                    )
                    q_THK = torch.randn(seq_len // 2, 1, 4)
                    k_THK = torch.arange(seq_len * 4).view(seq_len, 1, 4).float()
                    v_THV = k_THK + 100
                    gathered_k_THK, gathered_v_THV = k_THK, v_THV
                    if permutation is not None:
                        gathered_k_THK = k_THK.index_select(0, permutation[0])
                        gathered_v_THV = v_THV.index_select(0, permutation[0])
                    chunk_len = q_THK.shape[0] // len(case.kv_ranges)
                    outputs = [torch.randn(chunk_len, 1, 4) for _ in case.kv_ranges]
                    spmd.assert_type(
                        (q_THK, *outputs),
                        spmd.PartitionSpec(("dp", "cp"), "tp", None),
                        spmd.V,
                    )
                    spmd.assert_type(
                        (gathered_k_THK, gathered_v_THV),
                        spmd.R,
                        {"dp": spmd.V, "tp": spmd.V},
                        partition_spec=spmd.PartitionSpec("dp", "tp", None),
                    )
                    with typecheck(), mock.patch.object(
                        cp_attention,
                        "cp_all_gather_tensors",
                        return_value=(gathered_k_THK, gathered_v_THV),
                    ), mock.patch.object(
                        VarlenInnerAttention,
                        "forward",
                        autospec=True,
                        side_effect=outputs,
                    ) as inner_forward, mock.patch.object(
                        cp_attention,
                        "spmd_mesh_group",
                        return_value=SimpleNamespace(size=lambda: 2),
                    ), mock.patch.object(
                        cp_attention.dist,
                        "get_rank",
                        return_value=case.rank,
                    ):
                        actual = attention(
                            q_THK,
                            torch.empty(0),
                            torch.empty(0),
                            attention_metadata=metadata,
                        )
                    self.assertEqual(actual, torch.cat(outputs))
                    calls = inner_forward.call_args_list
                    self.assertEqual(len(calls), len(case.kv_ranges))
                    for index, (call, (start, end)) in enumerate(
                        zip(calls, case.kv_ranges)
                    ):
                        self.assertEqual(
                            call.args[1],
                            q_THK[index * chunk_len : (index + 1) * chunk_len],
                        )
                        self.assertEqual(call.args[2], k_THK[start:end])
                        self.assertEqual(call.args[3], v_THV[start:end])
                        self.assertEqual(
                            call.kwargs["attention_metadata"].window_size, case.window
                        )
                        for arg_index in (2, 3):
                            self.assertEqual(
                                call.args[arg_index].untyped_storage().data_ptr(),
                                calls[0].args[arg_index].untyped_storage().data_ptr(),
                            )
                    if not case.head_tail:
                        self.assertEqual(
                            calls[0].args[2].untyped_storage().data_ptr(),
                            k_THK.untyped_storage().data_ptr(),
                        )


if __name__ == "__main__":
    run_tests()
