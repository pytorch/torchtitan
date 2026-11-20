# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
from typing import cast
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.utils._pytree as pytree
import torch_remat as remat
from torch.distributed.tensor.experimental._attention import _HeadTailLoadBalancer
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.distributed.context_parallel import cp_query_ranges
from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import (
    cp_attention as attention_module,
    VarlenInnerAttention,
)
from torchtitan.models.common.attention.cp_attention import (
    _run_varlen_cp_attention,
    SelectiveGatherKVSlidingWindowVarlenInnerAttention,
)

pytest.importorskip("attn_gym.distributed.selective_gather")


def _poison_exchange_buffers(exchange, group):
    for buffer in (exchange.forward_zone, exchange.backward_zone):
        buffer.k.fill_(torch.nan)
        buffer.v.fill_(torch.nan)
    # Finish every rank's fill before peers can overwrite the buffers.
    torch.cuda.synchronize()
    dist.barrier(group=group)


def _poisoned_exchange_kv(module, exchange, k, v):
    (
        compact_k,
        compact_v,
    ) = SelectiveGatherKVSlidingWindowVarlenInnerAttention._exchange_kv(
        module, exchange, k, v
    )
    assert module._layout is not None
    valid_tokens = len(module._layout[exchange.plan.rank]) * module.gather_block_size
    padding = (
        torch.arange(compact_k.shape[0], device=k.device)[:, None, None] >= valid_tokens
    )
    return torch.where(padding, torch.nan, compact_k), torch.where(
        padding, torch.nan, compact_v
    )


pytestmark = pytest.mark.multi_gpu


@pytest.mark.skipif(
    torch.cuda.device_count() < 4
    or not all(
        torch.cuda.can_device_access_peer(device, peer)
        for device in range(4)
        for peer in range(4)
        if device != peer
    ),
    reason="Requires four CUDA devices with peer access.",
)
class TestSelectiveGatherVarlenAttention(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    def _prepare(
        self,
        *,
        device: torch.device,
        head_tail: bool,
        sliding_window_size: int,
        document_lengths: tuple[int, ...] = (3, 7, 6, 9, 7),
    ):
        config = SelectiveGatherKVSlidingWindowVarlenInnerAttention.Config(
            sliding_window_size=sliding_window_size,
            gather_block_size=2,
        )
        positions = torch.cat(
            [torch.arange(length, device=device) for length in document_lengths]
        )
        metadata = config.build_attention_metadata(
            positions,
            max_num_documents=7,
            max_context_length=9,
        )
        permutation = (
            _HeadTailLoadBalancer(32, self.world_size, device)._generate_indices()
            if head_tail
            else None
        )
        cp_metadata = (
            SelectiveGatherKVSlidingWindowVarlenInnerAttention.prepare_cp_metadata(
                metadata,
                permutation=permutation,
            )
        )
        indices = (
            permutation[0]
            if permutation is not None
            else torch.arange(32, device=device)
        )
        return config.build(), cp_metadata, indices

    @staticmethod
    def _copy_metadata_tensors(
        destination,
        source,
    ) -> None:
        destination_leaves, destination_spec = pytree.tree_flatten(destination)
        source_leaves, source_spec = pytree.tree_flatten(source)
        assert destination_spec == source_spec
        for destination_leaf, source_leaf in zip(
            destination_leaves, source_leaves, strict=True
        ):
            if isinstance(destination_leaf, torch.Tensor):
                destination_leaf.copy_(source_leaf)
            else:
                assert destination_leaf == source_leaf

    def _all_gather_reference_from_local(
        self,
        attention: VarlenInnerAttention,
        q_local_THK: torch.Tensor,
        k_local_THK: torch.Tensor,
        v_local_THV: torch.Tensor,
        metadata,
        permutation: torch.Tensor,
        group: dist.ProcessGroup,
        output_weight_THV: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,]:
        gathered_k_THK = torch.empty(
            (32, *k_local_THK.shape[1:]),
            dtype=k_local_THK.dtype,
            device=k_local_THK.device,
        )
        gathered_v_THV = torch.empty(
            (32, *v_local_THV.shape[1:]),
            dtype=v_local_THV.dtype,
            device=v_local_THV.device,
        )
        dist.all_gather_into_tensor(gathered_k_THK, k_local_THK.detach(), group=group)
        dist.all_gather_into_tensor(gathered_v_THV, v_local_THV.detach(), group=group)
        restore_indices = torch.empty_like(permutation)
        restore_indices[permutation] = torch.arange(
            32,
            device=permutation.device,
            dtype=permutation.dtype,
        )
        global_k_THK = (
            gathered_k_THK.index_select(0, restore_indices).detach().requires_grad_()
        )
        global_v_THV = (
            gathered_v_THV.index_select(0, restore_indices).detach().requires_grad_()
        )
        local_q_THK = q_local_THK.detach().clone().requires_grad_()

        def select_kv(start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
            return global_k_THK[start:end], global_v_THV[start:end]

        output_THV = _run_varlen_cp_attention(
            attention,
            local_q_THK,
            attention_metadata=metadata,
            query_runs=cp_query_ranges(
                32, group.size(), group.rank(), head_tail=metadata.head_tail
            ),
            select_kv=select_kv,
        )
        (output_THV.float() * output_weight_THV).sum().backward()
        assert local_q_THK.grad is not None
        assert global_k_THK.grad is not None
        assert global_v_THV.grad is not None
        dist.all_reduce(global_k_THK.grad, group=group)
        dist.all_reduce(global_v_THV.grad, group=group)
        rank_start = dist.get_rank(group) * 8
        local_indices = permutation[rank_start : rank_start + 8]
        return (
            output_THV,
            local_q_THK.grad,
            global_k_THK.grad.index_select(0, local_indices),
            global_v_THV.grad.index_select(0, local_indices),
        )

    def _sliding_attention_cuda_graph_case(
        self, use_checkpoint, *, head_tail, sliding_window_size
    ) -> None:
        device = torch.device(self.device_type)
        group = cast(dist.ProcessGroup, dist.new_group(tuple(range(self.world_size))))
        attention, metadata, permutation = self._prepare(
            device=device,
            head_tail=head_tail,
            sliding_window_size=sliding_window_size,
        )
        other_attention, _, _ = self._prepare(
            device=device,
            head_tail=head_tail,
            sliding_window_size=sliding_window_size,
        )
        _, replay_metadata, _ = self._prepare(
            device=device,
            document_lengths=(8, 5, 3, 7, 9),
            head_tail=head_tail,
            sliding_window_size=sliding_window_size,
        )
        assert not torch.equal(metadata.cu_seq_q, replay_metadata.cu_seq_q)
        if use_checkpoint:
            block = Module()
            block.add_module("inner_attention", attention)
            block.add_module("other_inner_attention", other_attention)
            block.configure_remat_regions([])
        rank_start = dist.get_rank(group) * 8
        local_indices = permutation[rank_start : rank_start + 8]
        graph = None
        torch.manual_seed(42)
        output_weight_THV = torch.randn(8, 2, 24, dtype=torch.float32, device=device)

        def inputs(
            offset: float,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            torch.manual_seed(123)
            q_THK = torch.randn(32, 2, 16, dtype=torch.bfloat16, device=device)
            k_THK = torch.randn(32, 2, 16, dtype=torch.bfloat16, device=device)
            v_THV = torch.randn(32, 2, 24, dtype=torch.bfloat16, device=device)
            return (
                q_THK.index_select(0, local_indices).add_(offset).requires_grad_(),
                k_THK.index_select(0, local_indices).add_(offset).requires_grad_(),
                v_THV.index_select(0, local_indices).add_(offset).requires_grad_(),
            )

        def run(q_THK, k_THK, v_THV):
            with mock.patch.object(
                attention_module, "spmd_mesh_group", return_value=group
            ):

                def forward(q_THK, k_THK, v_THV):
                    return attention(
                        q_THK, k_THK, v_THV, attention_metadata=metadata
                    ) + other_attention(
                        q_THK + 1, k_THK + 1, v_THV + 1, attention_metadata=metadata
                    )

                train = (
                    remat.checkpoint(region_name="layer")(forward)
                    if use_checkpoint
                    else forward
                )
                out_THV = train(q_THK, k_THK, v_THV)
                (out_THV.float() * output_weight_THV).sum().backward()
            return out_THV

        def check_reference(
            local_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            output_THV: torch.Tensor,
            reference_metadata,
        ) -> None:
            q_THK, k_THK, v_THV = local_inputs
            references = [
                self._all_gather_reference_from_local(
                    cast(VarlenInnerAttention, attention),
                    q_THK + offset,
                    k_THK + offset,
                    v_THV + offset,
                    reference_metadata,
                    permutation,
                    group,
                    output_weight_THV,
                )
                for offset in (0, 1)
            ]
            reference = tuple(
                first + second for first, second in zip(*references, strict=True)
            )
            # assert_close accepts matching infinities.
            assert all(torch.isfinite(tensor).all() for tensor in reference)
            actual = (output_THV, *(tensor.grad for tensor in local_inputs))
            torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)

        try:
            eager_inputs = inputs(0)
            _, k, v = eager_inputs
            exchange = attention._get_exchange(
                k, v, cp_group=group, head_tail=head_tail
            )
            assert (
                other_attention._get_exchange(k, v, cp_group=group, head_tail=head_tail)
                is exchange
            )
            _poison_exchange_buffers(exchange, group)
            plan = exchange.plan
            has_padding = (
                len(plan.local_destination_blocks) + len(plan.remote_destination_blocks)
                < plan.num_assembled_blocks
            )
            padded_ranks = torch.tensor(
                int(has_padding), dtype=torch.int32, device=device
            )
            dist.all_reduce(padded_ranks, group=group)
            if sliding_window_size == 1:
                assert int(padded_ranks) == 0
            else:
                assert int(padded_ranks) > 0
            with mock.patch.object(
                attention,
                "_exchange_kv",
                side_effect=lambda *args: _poisoned_exchange_kv(attention, *args),
            ):
                eager_output_THV = run(*eager_inputs)
            check_reference(eager_inputs, eager_output_THV, metadata)
            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                run(*inputs(1))
            torch.cuda.current_stream().wait_stream(capture_stream)

            static_inputs = inputs(2)
            # Release unused eager graphs before capture.
            gc.collect()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                graph, stream=capture_stream, capture_error_mode="thread_local"
            ):
                output_THV = run(*static_inputs)
            first_output_THV = output_THV.clone()

            with torch.no_grad():
                for tensor in static_inputs:
                    tensor.add_(1)
                    assert tensor.grad is not None
                    tensor.grad.zero_()
                self._copy_metadata_tensors(metadata, replay_metadata)
            graph.replay()
            torch.cuda.synchronize()

            self.assertFalse(torch.equal(output_THV, first_output_THV))
            check_reference(static_inputs, output_THV, replay_metadata)
        finally:
            del graph
            gc.collect()
            torch.cuda.synchronize()
            dist.destroy_process_group(group)

    @with_comms
    def test_sliding_attention_cuda_graph_replay(self) -> None:
        for sliding_window_size in (1, 4):
            for head_tail in (False, True):
                for use_checkpoint in (False, True):
                    with self.subTest(
                        sliding_window_size=sliding_window_size,
                        head_tail=head_tail,
                        use_checkpoint=use_checkpoint,
                    ):
                        self._sliding_attention_cuda_graph_case(
                            use_checkpoint,
                            head_tail=head_tail,
                            sliding_window_size=sliding_window_size,
                        )
