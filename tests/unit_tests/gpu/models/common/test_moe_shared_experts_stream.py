# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import spmd_types as spmd
import torch
import torch.nn as nn
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.common.activation import Sigmoid
from torchtitan.models.common.config_utils import (
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
    make_shared_expert_ffn_config,
)
from torchtitan.models.common.moe.moe import _shared_experts_side_stream


def _build_moe(*, dim: int, enable_ep: bool, enable_sp: bool) -> nn.Module:
    hidden_dim, num_experts, top_k = 512, 8, 2
    return make_moe_config(
        num_experts=num_experts,
        router=make_router_config(
            dim=dim,
            num_experts=num_experts,
            gate_param_init={},
            score_func=Sigmoid.Config(),
            top_k=top_k,
        ),
        routed_experts=make_routed_experts_config(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            param_init={},
            enable_ep=enable_ep,
        ),
        shared_experts=make_shared_expert_ffn_config(
            dim=dim,
            hidden_dim=hidden_dim,
            w13_param_init={},
            w2_param_init={},
            enable_sp=enable_sp,
        ),
        enable_sp=enable_sp,
    ).build()


def _fill_parameters(moe: nn.Module) -> None:
    torch.manual_seed(0)
    with torch.no_grad():
        for p in moe.parameters():
            p.normal_(std=0.02)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestSharedExpertsSideStream(unittest.TestCase):
    def test_cuda_graph_matches_eager(self):
        # Under capture the shared experts run on a side stream; same kernels, same output.
        dim = 256
        moe = _build_moe(dim=dim, enable_ep=False, enable_sp=False)
        _fill_parameters(moe)
        moe = moe.to(device="cuda", dtype=torch.bfloat16).eval()
        x_TD = torch.randn(34, dim, device="cuda", dtype=torch.bfloat16)

        with torch.no_grad():
            expected_TD = moe(x_TD)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                # Without SP the forward below forks; with SP it stays inline.
                self.assertIsNotNone(_shared_experts_side_stream())
                with patch(
                    "torchtitan.models.common.moe.moe.spmd_dense_sp_enabled",
                    return_value=True,
                ):
                    self.assertIsNone(_shared_experts_side_stream())
                out_TD = moe(x_TD)
            graph.replay()

        torch.testing.assert_close(out_TD, expected_TD, rtol=0, atol=0)


class _LocalRoutedExperts(nn.Module):
    """Routed experts without dispatch, so the MoE captures without an EP backend."""

    def forward(self, x_TD, topk_scores_TK, topk_expert_ids_TK, counts_E, **kwargs):
        return x_TD * topk_scores_TK.sum(dim=-1, keepdim=True).to(x_TD.dtype)


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestSharedExpertsSideStreamTP(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_side_stream_with_tp_but_not_sp(self):
        # Without SP the shared output stays TP-partial and the MoE's final all-reduce
        # runs on the main stream; with SP the shared experts gather their own input.
        dim, num_tokens = 256, 32
        for enable_sp in (False, True):
            with self.subTest(enable_sp=enable_sp):
                moe = _build_moe(dim=dim, enable_ep=True, enable_sp=enable_sp)
                _fill_parameters(moe)
                moe = moe.to(device=self.device_type, dtype=torch.bfloat16)
                parallelism_context = ParallelismContext(
                    dp_replicate=1,
                    dp_shard=1,
                    cp=1,
                    tp=self.world_size,
                    pp=1,
                    ep=self.world_size,
                    world_size=self.world_size,
                    enable_sequence_parallel=enable_sp,
                )
                with patch(
                    "torchtitan.distributed.parallelism_context.device_type",
                    self.device_type,
                ):
                    parallelism_context.build_mesh()
                moe._parallelize(parallelism_context)
                moe.routed_experts = _LocalRoutedExperts()
                moe.eval()

                torch.manual_seed(1)
                x_TD = torch.randn(
                    num_tokens, dim, device=self.device_type, dtype=torch.bfloat16
                )
                if enable_sp:
                    x_TD = x_TD.chunk(self.world_size)[self.rank].contiguous()
                # Every redistribute on the side stream except the no-op I -> R.
                side_stream_collectives = []
                redistribute = spmd.redistribute

                def record(x, group, *, src, dst, **kwargs):
                    on_side = torch.cuda.current_stream() == side_stream
                    if on_side and (src, dst) != (spmd.I, spmd.R):
                        side_stream_collectives.append((src, dst))
                    return redistribute(x, group, src=src, dst=dst, **kwargs)

                with parallelism_context.activate_spmd(), torch.no_grad():
                    expected_TD = moe(x_TD)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph), patch.object(
                        spmd, "redistribute", record
                    ):
                        side_stream = _shared_experts_side_stream()
                        out_TD = moe(x_TD)
                    graph.replay()
                    matches_eager = torch.equal(out_TD, expected_TD)
                # A failing assert that keeps the graph alive hangs process-group teardown.
                del graph

                self.assertEqual(side_stream is not None, not enable_sp)
                self.assertEqual(side_stream_collectives, [])
                self.assertTrue(matches_eager)


if __name__ == "__main__":
    unittest.main()
