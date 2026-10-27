# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TritonRoutedExperts at EP > 1 on DeepEP: tokens go once per rank, the kernels read the routes."""

import unittest
from unittest.mock import patch

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.spmd_types import set_current_spmd_mesh, set_spmd_meshes
from torchtitan.models.base.module import Module
from torchtitan.models.common.activation import SwiGLU
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.moe.token_dispatcher import DeepEPTokenDispatcher
from torchtitan_recipes.overrides.fused_swiglu import FusedSwiGLU
from torchtitan_recipes.overrides.triton_routed_experts import TritonRoutedExperts

deep_ep = pytest.importorskip("deep_ep")

pytestmark = pytest.mark.multi_gpu

# The Triton kernels need dims divisible by 256.
_DIM, _HIDDEN, _TOP_K, _MAX_TOKENS = 512, 256, 4, 256


def _build(cls: type, num_experts: int, activation=None) -> RoutedExperts:
    """Routed experts holding this rank's expert shard, on DeepEP's inference layout.

    Built field by field (no parallelize step), as test_deepep_activation_checkpoint does:
    the weights are the local shard, the dispatcher keeps the global expert count.
    """
    num_local_experts = num_experts // torch.distributed.get_world_size()
    experts = cls.__new__(cls)
    Module.__init__(experts)
    experts.w13 = GroupedLinear.Config(
        group_size=num_local_experts,
        in_features=_DIM,
        out_features=_HIDDEN,
        num_linears=2,
    ).build()
    experts.w2 = GroupedLinear.Config(
        group_size=num_local_experts, in_features=_HIDDEN, out_features=_DIM
    ).build()
    experts.activation_fn = (activation or SwiGLU.Config()).build()
    experts.output_postprocess = None
    experts.token_dispatcher = DeepEPTokenDispatcher.Config(
        num_experts=num_experts,
        top_k=_TOP_K,
        hidden_dim=_DIM,
        num_max_tokens_per_rank=_MAX_TOKENS,
        cuda_graph_compatible=True,
    ).build()
    return experts


def _routing(num_tokens: int, num_experts: int, seed: int, rank0_only: bool = False):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x_TD = torch.randn(
        num_tokens, _DIM, device="cuda", dtype=torch.bfloat16, generator=gen
    )
    logits_TE = torch.rand(num_tokens, num_experts, device="cuda", generator=gen)
    if rank0_only:  # every route goes to rank 0's experts: rank 1 receives nothing
        logits_TE[:, num_experts // torch.distributed.get_world_size() :] = -1
    scores_TK, ids_TK = logits_TE.topk(_TOP_K, dim=-1)
    return x_TD, scores_TK / scores_TK.sum(-1, keepdim=True), ids_TK


def _fp64_reference(w13_E2FD, w2_EDF, x_TD, scores_TK, ids_TK) -> torch.Tensor:
    w13_TK2FD = w13_E2FD.double()[ids_TK]
    gate_TKF = torch.einsum("td,tkfd->tkf", x_TD.double(), w13_TK2FD[:, :, 0])
    up_TKF = torch.einsum("td,tkfd->tkf", x_TD.double(), w13_TK2FD[:, :, 1])
    hidden_TKF = torch.nn.functional.silu(gate_TKF) * up_TKF
    out_TKD = torch.einsum("tkf,tkdf->tkd", hidden_TKF, w2_EDF.double()[ids_TK])
    return (out_TKD * scores_TK.double().unsqueeze(-1)).sum(1)


def _all_shards(weight: torch.Tensor) -> torch.Tensor:
    shards = [
        torch.empty_like(weight) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(shards, weight.contiguous())
    return torch.cat(shards)


def _rel(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.double() - expected).norm() / expected.norm()).item()


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestDeepEPRoutedExperts(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_reads_routes_closer_to_fp64_than_stock_and_graph_safe(self):
        mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        set_spmd_meshes(dense_mesh=mesh, sparse_mesh=mesh, dense_sp_enabled=False)
        num_experts = 4 * self.world_size
        with set_current_spmd_mesh(mesh), torch.no_grad():
            torch.manual_seed(self.rank)
            triton_experts = _build(TritonRoutedExperts, num_experts).cuda()
            for param in triton_experts.parameters():
                param.normal_(std=0.05)
            # The fused_swiglu override's activation takes the same path.
            fused = _build(TritonRoutedExperts, num_experts, FusedSwiGLU.Config())
            stock = _build(RoutedExperts, num_experts)
            for module in (triton_experts, fused, stock):
                module.load_state_dict(triton_experts.state_dict())
                module.cuda().bfloat16()
            triton_experts.token_dispatcher.init_buffer()  # one buffer serves all three
            # Every rank's shard, for the fp64 reference of this rank's tokens.
            w13_E2FD = _all_shards(triton_experts.w13.weight)
            w2_EDF = _all_shards(triton_experts.w2.weight)

            cases = [(1, False), (37, False), (_MAX_TOKENS, False), (37, True)]
            for num_tokens, rank0_only in cases:
                with self.subTest(num_tokens=num_tokens, rank0_only=rank0_only):
                    routing = _routing(num_tokens, num_experts, self.rank, rank0_only)
                    expected = _fp64_reference(w13_E2FD, w2_EDF, *routing)
                    outputs = []
                    for experts in (triton_experts, fused):
                        with patch.object(
                            deep_ep.ElasticBuffer,
                            "dispatch",
                            autospec=True,
                            side_effect=deep_ep.ElasticBuffer.dispatch,
                        ) as dispatch:
                            actual = experts(*routing, None)
                        # One compact dispatch without the host sync: no expand copies.
                        layouts = [
                            (call.kwargs["do_expand"], call.kwargs["do_cpu_sync"])
                            for call in dispatch.call_args_list
                        ]
                        self.assertEqual(layouts, [(False, False)])
                        self.assertLess(_rel(actual, expected), 5e-3)
                        outputs.append(actual)
                    ids_TK = routing[2]
                    counts_E = torch.bincount(ids_TK.flatten(), minlength=num_experts)
                    stock_out = stock(*routing, counts_E)
                    self.assertLessEqual(
                        _rel(outputs[0], expected), _rel(stock_out, expected)
                    )
            routing_map_TE = torch.ones(1, num_experts, dtype=torch.bool, device="cuda")
            self.assertIsNone(triton_experts.count_tokens(routing_map_TE))

            # Captured once, replayed with new tokens and routes: same as eager.
            static = _routing(37, num_experts, seed=100 + self.rank)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                triton_experts(*static, None)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graphed = triton_experts(*static, None)
            fresh = _routing(37, num_experts, seed=200 + self.rank)
            for buf, value in zip(static, fresh):
                buf.copy_(value)
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(graphed, triton_experts(*fresh, None)))
