# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPU tests for K/V all-gather varlen context parallelism."""

from types import SimpleNamespace
from unittest import mock

import torch
from torch.distributed.tensor.experimental._attention import _HeadTailLoadBalancer
from torch.nn.attention.varlen import varlen_attn
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.models.common.attention import VarlenMetadata
from torchtitan.models.common.cp_attention import (
    HeadTailCPVarlenMetadata,
    KVAllGatherCPVarlenInnerAttention,
)


class TestKVAllGatherCPVarlenInnerAttention(TestCase):
    @staticmethod
    def _metadata(
        offsets: list[int], permutation: torch.Tensor
    ) -> HeadTailCPVarlenMetadata:
        cu_seq = torch.tensor(offsets, device="cuda", dtype=torch.int32)
        lengths = torch.diff(cu_seq)
        global_metadata = VarlenMetadata(
            cu_seq,
            cu_seq,
            int(lengths.max().item()),
            int(lengths.max().item()),
        )
        return HeadTailCPVarlenMetadata.from_global(
            global_metadata,
            permutation=permutation,
        )

    def test_matches_global_varlen_attention(self) -> None:
        torch.manual_seed(0)
        seq_len = 16
        permutation = _HeadTailLoadBalancer(
            seq_len, 2, torch.device("cuda")
        )._generate_indices()
        metadata = self._metadata([0, 3, 10, 16], permutation)
        group = SimpleNamespace(size=lambda: 2)

        for rank in range(2):
            for window_size in ((-1, 0), (2, 0)):
                q_THK = torch.randn(
                    seq_len, 4, 64, device="cuda", dtype=torch.bfloat16
                ).requires_grad_()
                k_THK = torch.randn(
                    seq_len, 2, 64, device="cuda", dtype=torch.bfloat16
                ).requires_grad_()
                v_THV = torch.randn(
                    seq_len, 2, 64, device="cuda", dtype=torch.bfloat16
                ).requires_grad_()
                local_positions = permutation[
                    0, rank * seq_len // 2 : (rank + 1) * seq_len // 2
                ].long()
                local_q_THK = q_THK.detach()[local_positions].clone().requires_grad_()
                gathered_k_THK = (
                    k_THK.detach()[permutation[0].long()].clone().requires_grad_()
                )
                gathered_v_THV = (
                    v_THV.detach()[permutation[0].long()].clone().requires_grad_()
                )
                attention = KVAllGatherCPVarlenInnerAttention(
                    KVAllGatherCPVarlenInnerAttention.Config(window_size=window_size)
                )

                with mock.patch.object(
                    attention,
                    "_all_gather_kv",
                    return_value=(gathered_k_THK, gathered_v_THV),
                ), mock.patch(
                    "torchtitan.models.common.cp_attention.spmd_mesh_group",
                    return_value=group,
                ), mock.patch(
                    "torchtitan.models.common.cp_attention.dist.get_rank",
                    return_value=rank,
                ):
                    output_THV = attention(
                        local_q_THK,
                        torch.empty(0, device="cuda"),
                        torch.empty(0, device="cuda"),
                        attention_masks=metadata,
                        enable_gqa=True,
                    )
                output_THV.float().sum().backward()

                reference_THV = varlen_attn(
                    q_THK,
                    k_THK,
                    v_THV,
                    metadata.varlen_metadata.cu_seq_q,
                    metadata.varlen_metadata.cu_seq_k,
                    metadata.varlen_metadata.max_q,
                    metadata.varlen_metadata.max_k,
                    window_size=window_size,
                    enable_gqa=True,
                )
                reference_THV[local_positions].float().sum().backward()

                assert local_q_THK.grad is not None
                assert gathered_k_THK.grad is not None
                assert gathered_v_THV.grad is not None
                assert q_THK.grad is not None
                assert k_THK.grad is not None
                assert v_THV.grad is not None
                self.assertEqual(output_THV, reference_THV[local_positions])
                torch.testing.assert_close(
                    local_q_THK.grad,
                    q_THK.grad[local_positions],
                    atol=0.02,
                    rtol=0.02,
                )
                torch.testing.assert_close(
                    gathered_k_THK.grad.index_select(0, metadata.kv_restore_indices),
                    k_THK.grad,
                    atol=0.02,
                    rtol=0.02,
                )
                torch.testing.assert_close(
                    gathered_v_THV.grad.index_select(0, metadata.kv_restore_indices),
                    v_THV.grad,
                    atol=0.02,
                    rtol=0.02,
                )

    @torch.no_grad()
    def test_cuda_graph_replays_with_new_document_boundaries(self) -> None:
        torch.manual_seed(1)
        seq_len = 16
        permutation = _HeadTailLoadBalancer(
            seq_len, 2, torch.device("cuda")
        )._generate_indices()
        metadata = self._metadata([0, 3, 10, 16], permutation)
        group = SimpleNamespace(size=lambda: 2)
        q_THK = torch.randn(seq_len // 2, 4, 64, device="cuda", dtype=torch.bfloat16)
        k_THK = torch.randn(seq_len, 2, 64, device="cuda", dtype=torch.bfloat16)
        v_THV = torch.randn(seq_len, 2, 64, device="cuda", dtype=torch.bfloat16)
        permuted_k_THK = k_THK.index_select(0, permutation[0])
        permuted_v_THV = v_THV.index_select(0, permutation[0])
        attention = KVAllGatherCPVarlenInnerAttention(
            KVAllGatherCPVarlenInnerAttention.Config(window_size=(-1, 0))
        )

        patches = (
            mock.patch.object(
                attention,
                "_all_gather_kv",
                return_value=(permuted_k_THK, permuted_v_THV),
            ),
            mock.patch(
                "torchtitan.models.common.cp_attention.spmd_mesh_group",
                return_value=group,
            ),
            mock.patch(
                "torchtitan.models.common.cp_attention.dist.get_rank",
                return_value=0,
            ),
        )
        with patches[0], patches[1], patches[2]:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    attention(
                        q_THK,
                        torch.empty(0, device="cuda"),
                        torch.empty(0, device="cuda"),
                        attention_masks=metadata,
                        enable_gqa=True,
                    )
            torch.cuda.current_stream().wait_stream(stream)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output_THV = attention(
                    q_THK,
                    torch.empty(0, device="cuda"),
                    torch.empty(0, device="cuda"),
                    attention_masks=metadata,
                    enable_gqa=True,
                )

            new_offsets = torch.tensor([0, 5, 11, 16], device="cuda", dtype=torch.int32)
            metadata.varlen_metadata.cu_seq_q.copy_(new_offsets)
            graph.replay()
            torch.cuda.synchronize()

        global_q_THK = torch.empty(seq_len, 4, 64, device="cuda", dtype=torch.bfloat16)
        global_q_THK[permutation[0, : seq_len // 2].long()] = q_THK
        reference_THV = varlen_attn(
            global_q_THK,
            k_THK,
            v_THV,
            new_offsets,
            new_offsets,
            metadata.varlen_metadata.max_q,
            metadata.varlen_metadata.max_k,
            window_size=(-1, 0),
            enable_gqa=True,
        )
        local_positions = permutation[0, : seq_len // 2].long()
        self.assertEqual(output_THV, reference_THV[local_positions])


if __name__ == "__main__":
    run_tests()
