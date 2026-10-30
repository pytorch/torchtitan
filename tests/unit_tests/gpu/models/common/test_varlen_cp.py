# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPU tests for K/V all-gather varlen context parallelism."""

from collections.abc import Iterator
from contextlib import contextmanager
from itertools import product
from types import SimpleNamespace
from unittest import mock

import torch
from torch.distributed.tensor.experimental._attention import _HeadTailLoadBalancer
from torch.nn.attention.varlen import varlen_attn
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.models.common.attention import cp_attention, VarlenAttentionMetadata
from torchtitan.models.common.attention.cp_attention import (
    AllGatherKVSlidingWindowVarlenInnerAttention,
    AllGatherKVVarlen,
    AllGatherKVVarlenAttentionMetadata,
    AllGatherKVVarlenInnerAttention,
)


@contextmanager
def _patch_cp(
    gathered_kv: tuple[torch.Tensor, torch.Tensor], *, rank: int
) -> Iterator[None]:
    with mock.patch.object(
        cp_attention, "cp_all_gather_tensors", return_value=gathered_kv
    ), mock.patch.object(
        cp_attention,
        "spmd_mesh_group",
        return_value=SimpleNamespace(size=lambda: 2),
    ), mock.patch.object(
        cp_attention.dist, "get_rank", return_value=rank
    ):
        yield


class TestKVAllGatherVarlenInnerAttention(TestCase):
    @staticmethod
    def _metadata(
        offsets: list[int],
        permutation: torch.Tensor | None,
        *,
        window_size: tuple[int, int] = (-1, 0),
    ) -> AllGatherKVVarlenAttentionMetadata:
        cu_seq = torch.tensor(offsets, device="cuda", dtype=torch.int32)
        lengths = torch.diff(cu_seq)
        global_metadata = VarlenAttentionMetadata(
            cu_seq,
            cu_seq,
            int(lengths.max().item()),
            int(lengths.max().item()),
            window_size=window_size,
        )
        return AllGatherKVVarlen.prepare_cp_metadata(
            global_metadata,
            permutation=permutation,
        )

    def test_matches_global_varlen_attention(self) -> None:
        torch.manual_seed(0)
        for head_tail, rank, window_size in product(
            (False, True), range(2), ((-1, 0), (2, 0))
        ):
            seq_len = 16 if head_tail else 14
            permutation = (
                _HeadTailLoadBalancer(
                    seq_len, 2, torch.device("cuda")
                )._generate_indices()
                if head_tail
                else torch.arange(seq_len, device="cuda").unsqueeze(0)
            )
            metadata = self._metadata(
                [0, 3, 10, seq_len],
                permutation if head_tail else None,
                window_size=window_size,
            )
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
            config = (
                AllGatherKVVarlenInnerAttention.Config()
                if window_size == (-1, 0)
                else AllGatherKVSlidingWindowVarlenInnerAttention.Config(
                    sliding_window_size=window_size[0] + 1
                )
            )
            attention = config.build()

            with _patch_cp((gathered_k_THK, gathered_v_THV), rank=rank):
                output_THV = attention(
                    local_q_THK,
                    torch.empty(0, device="cuda"),
                    torch.empty(0, device="cuda"),
                    attention_metadata=metadata,
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

            assert gathered_k_THK.grad is not None
            assert gathered_v_THV.grad is not None
            assert q_THK.grad is not None
            self.assertEqual(output_THV, reference_THV[local_positions])
            torch.testing.assert_close(
                local_q_THK.grad,
                q_THK.grad[local_positions],
                atol=0.02,
                rtol=0.02,
            )
            torch.testing.assert_close(
                gathered_k_THK.grad.index_select(0, permutation[0].argsort()),
                k_THK.grad,
                atol=0.02,
                rtol=0.02,
            )
            torch.testing.assert_close(
                gathered_v_THV.grad.index_select(0, permutation[0].argsort()),
                v_THV.grad,
                atol=0.02,
                rtol=0.02,
            )

    @torch.no_grad()
    def test_accelerator_graph_replays_with_new_document_boundaries(self) -> None:
        for head_tail in (False, True):
            torch.manual_seed(1)
            seq_len = 16
            permutation = (
                _HeadTailLoadBalancer(
                    seq_len, 2, torch.device("cuda")
                )._generate_indices()
                if head_tail
                else torch.arange(seq_len, device="cuda").unsqueeze(0)
            )
            metadata = self._metadata(
                [0, 3, 6, 9, 12, 16, 16], permutation if head_tail else None
            )
            q_THK = torch.randn(
                seq_len // 2, 4, 64, device="cuda", dtype=torch.bfloat16
            )
            k_THK = torch.randn(seq_len, 2, 64, device="cuda", dtype=torch.bfloat16)
            v_THV = torch.randn(seq_len, 2, 64, device="cuda", dtype=torch.bfloat16)
            permuted_k_THK = k_THK.index_select(0, permutation[0])
            permuted_v_THV = v_THV.index_select(0, permutation[0])
            attention = AllGatherKVVarlenInnerAttention(
                AllGatherKVVarlenInnerAttention.Config()
            )

            with _patch_cp((permuted_k_THK, permuted_v_THV), rank=0):
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        attention(
                            q_THK,
                            torch.empty(0, device="cuda"),
                            torch.empty(0, device="cuda"),
                            attention_metadata=metadata,
                            enable_gqa=True,
                        )
                torch.cuda.current_stream().wait_stream(stream)

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output_THV = attention(
                        q_THK,
                        torch.empty(0, device="cuda"),
                        torch.empty(0, device="cuda"),
                        attention_metadata=metadata,
                        enable_gqa=True,
                    )

                new_offsets = torch.tensor(
                    [0, 2, 6, 10, 14, 16, 16], device="cuda", dtype=torch.int32
                )
                metadata.varlen_metadata.cu_seq_q.copy_(new_offsets)
                graph.replay()
                torch.cuda.synchronize()

            global_q_THK = torch.empty(
                seq_len, 4, 64, device="cuda", dtype=torch.bfloat16
            )
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
