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
import torch_remat as remat
from torch.distributed.tensor.experimental._attention import _HeadTailLoadBalancer
from torch.nn.attention.varlen import varlen_attn
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import cp_attention, VarlenAttentionMetadata
from torchtitan.models.common.attention.cp_attention import (
    AllGatherKVSlidingWindowVarlenInnerAttention,
    AllGatherKVVarlen,
    AllGatherKVVarlenAttentionMetadata,
    AllGatherKVVarlenInnerAttention,
    UlyssesVarlenInnerAttention,
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
            cu_seq_q=cu_seq,
            cu_seq_k=cu_seq,
            max_q=int(lengths.max().item()),
            max_k=int(lengths.max().item()),
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
                metadata.cu_seq_q,
                metadata.cu_seq_k,
                metadata.max_q,
                metadata.max_k,
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
                metadata.cu_seq_q.copy_(new_offsets)
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
                metadata.max_q,
                metadata.max_k,
                window_size=(-1, 0),
                enable_gqa=True,
            )
            local_positions = permutation[0, : seq_len // 2].long()
            self.assertEqual(output_THV, reference_THV[local_positions])

    def test_kernel_region_retains_only_output_and_lse(self) -> None:
        """Saving only the kernel regions keeps each kernel's output and LSE: the
        all-gathered K/V and the Q/K/V the kernel reads are rebuilt by replay."""
        torch.manual_seed(2)
        seq_len, num_heads = 16, 4
        permutation = _HeadTailLoadBalancer(
            seq_len, 2, torch.device("cuda")
        )._generate_indices()
        metadata = self._metadata([0, 3, 10, seq_len], permutation)
        q_THK = torch.randn(
            seq_len // 2, num_heads, 64, device="cuda", dtype=torch.bfloat16
        )
        k_THK = torch.randn(seq_len, 2, 64, device="cuda", dtype=torch.bfloat16)
        v_THV = torch.randn(seq_len, 2, 64, device="cuda", dtype=torch.bfloat16)
        attention = AllGatherKVVarlenInnerAttention(
            AllGatherKVVarlenInnerAttention.Config()
        )

        def run(save_patterns):
            with _patch_cp((k_THK, v_THV), rank=0), mock.patch.object(
                cp_attention,
                "cp_all_gather_tensors",
                side_effect=lambda kv, **_: tuple(
                    t.index_select(0, permutation[0]) for t in kv
                ),
            ):
                return _run_remat(
                    attention, (q_THK, k_THK, v_THV), metadata, save_patterns
                )

        reference_THV, reference_grads, _ = run([])
        _, _, saved_all = run(["*"])
        output_THV, grads, retained = run(["inner_attention.kernel*"])

        # Saving the whole inner_attention region retains the gathered K/V.
        self.assertTrue(any(shape[1:] == (2, 64) for shape in saved_all))
        half = seq_len // 4
        # Per kernel: the output (T/2, H, V) and the LSE (H, T/2).
        self.assertEqual(
            sorted(retained),
            sorted([(half, num_heads, 64), (num_heads, half)] * 2),
        )
        self.assertEqual(output_THV, reference_THV)
        for grad, reference_grad in zip(grads, reference_grads):
            self.assertEqual(grad, reference_grad)

    def test_ulysses_kernel_region_retains_only_output_and_lse(self) -> None:
        """Under Ulysses, saving the kernel region keeps its output and LSE; the
        all-to-all outputs the kernel reads are rebuilt by replay."""
        torch.manual_seed(3)
        seq_len, num_heads = 16, 4
        cu_seq = torch.tensor([0, 3, 10, seq_len], device="cuda", dtype=torch.int32)
        metadata = VarlenAttentionMetadata(
            cu_seq_q=cu_seq, cu_seq_k=cu_seq, max_q=7, max_k=7
        )
        inputs = (
            torch.randn(seq_len, num_heads, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(seq_len, 2, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(seq_len, 2, 64, device="cuda", dtype=torch.bfloat16),
        )
        attention = UlyssesVarlenInnerAttention(UlyssesVarlenInnerAttention.Config())

        def run(save_patterns):
            # One rank: the all-to-alls become copies made inside the region.
            with mock.patch.object(
                cp_attention,
                "spmd_mesh_group",
                return_value=SimpleNamespace(size=lambda: 1),
            ), mock.patch.object(
                cp_attention.spmd, "redistribute", side_effect=lambda x, *_, **__: x * 1
            ):
                return _run_remat(attention, inputs, metadata, save_patterns)

        reference_THV, reference_grads, _ = run([])
        _, _, saved_all = run(["*"])
        output_THV, grads, retained = run(["inner_attention.kernel"])

        # Saving the whole inner_attention region retains the kernel's Q.
        self.assertIn((seq_len, num_heads, 64), saved_all)
        self.assertEqual(
            sorted(retained), sorted([(seq_len, num_heads, 64), (num_heads, seq_len)])
        )
        self.assertEqual(output_THV, reference_THV)
        for grad, reference_grad in zip(grads, reference_grads):
            self.assertEqual(grad, reference_grad)


class _Caller(Module):
    """Wraps a CP kernel module in a region the way attention modules do."""

    def __init__(self, inner_attention: Module, metadata) -> None:
        super().__init__()
        self.inner_attention = inner_attention
        self.metadata = metadata

    def forward(self, q_THK, k_THK, v_THV):
        return remat.region(
            self.inner_attention,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(
            q_THK * 1,
            k_THK,
            v_THV,
            attention_metadata=self.metadata,
            enable_gqa=True,
        )


def _run_remat(inner_attention, inputs, metadata, save_patterns):
    """Run one checkpointed block; return its output, input gradients and the
    shapes of the activation storages it retains for backward."""
    caller = _Caller(inner_attention, metadata)
    caller.configure_remat_regions(save_patterns)
    q, k, v = (t.clone().requires_grad_() for t in inputs)
    # A save region's output and its kernel's saved output share one storage.
    retained = {}

    def pack(tensor):
        info = remat.current_saved_tensor_info()
        # Integer document offsets are small; count activations only.
        if (
            info.kind is not remat.SavedTensorKind.CHECKPOINT_INPUT
            and tensor.is_floating_point()
        ):
            retained[tensor.untyped_storage().data_ptr()] = tuple(tensor.shape)
        return tensor

    with remat.saved_tensors_hooks(pack, lambda tensor: tensor):
        output_THV = remat.checkpoint(region_name="block", preserve_rng_state=False)(
            caller
        )(q, k, v)
        output_THV.float().sum().backward()
    return output_THV, (q.grad, k.grad, v.grad), list(retained.values())


if __name__ == "__main__":
    run_tests()
