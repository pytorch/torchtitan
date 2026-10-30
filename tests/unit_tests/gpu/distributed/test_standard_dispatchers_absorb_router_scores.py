# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""fp64 parity of absorbed router scores across the standard dispatchers."""

import unittest
from dataclasses import dataclass

import pytest
import torch
import torch_remat as remat
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.spmd_types import set_current_spmd_mesh, set_spmd_meshes
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.moe.token_dispatcher import (
    AllToAllTokenDispatcher,
    LocalTokenDispatcher,
    TorchAOTokenDispatcher,
)

pytestmark = pytest.mark.multi_gpu

_MODEL_DIM = 8
_NUM_TOKENS = 13
_TOP_K = 2
# TorchAO's fused permute pads rows to this multiple.
_PAD_MULTIPLE = 16


class _ExactGroupedLinear(GroupedLinear):
    """GroupedLinear whose GEMM runs in the weight dtype."""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedLinear.Config):
        pass

    def _grouped_mm(self, *, input_RI, weight_EOI, offsets_E):
        input_RI = input_RI.to(weight_EOI.dtype)
        starts = [0, *offsets_E[:-1].tolist()]
        return torch.cat(
            [
                input_RI[start:end] @ weight_EOI[expert].transpose(-2, -1)
                for expert, (start, end) in enumerate(zip(starts, offsets_E.tolist()))
            ],
            dim=0,
        )


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestStandardDispatchersAbsorbRouterScoresParity(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @staticmethod
    def _cases():
        # name, dispatcher Config, extra kwargs, ep size (1 = no sparse mesh)
        return (
            ("local", LocalTokenDispatcher.Config, {}, 1),
            ("alltoall_ep1", AllToAllTokenDispatcher.Config, {}, 1),
            ("alltoall_ep2", AllToAllTokenDispatcher.Config, {}, 2),
            (
                "torchao_ep1",
                TorchAOTokenDispatcher.Config,
                {"pad_multiple": _PAD_MULTIPLE},
                1,
            ),
            (
                "torchao_ep2",
                TorchAOTokenDispatcher.Config,
                {"pad_multiple": _PAD_MULTIPLE},
                2,
            ),
        )

    @with_comms
    def test_absorbed_equals_post_combine(self):
        mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        for name, config_cls, kwargs, ep_size in self._cases():
            set_spmd_meshes(
                dense_mesh=mesh,
                sparse_mesh=mesh if ep_size > 1 else None,
                dense_sp_enabled=False,
            )
            with (
                self.subTest(dispatcher=name),
                set_current_spmd_mesh(mesh),
                torch.autograd.set_multithreading_enabled(False),
            ):
                torch.manual_seed(42)
                model = (
                    RoutedExperts.Config(
                        w13=_ExactGroupedLinear.Config(
                            group_size=8,
                            in_features=_MODEL_DIM,
                            out_features=_MODEL_DIM,
                            num_linears=2,
                        ),
                        w2=_ExactGroupedLinear.Config(
                            group_size=8,
                            in_features=_MODEL_DIM,
                            out_features=_MODEL_DIM,
                        ),
                        token_dispatcher=config_cls(
                            num_experts=8, top_k=_TOP_K, **kwargs
                        ),
                    )
                    .build()
                    .to(self.device_type, torch.float64)
                )
                # Emulate EP's local weight shard; the dispatcher retains global expert IDs.
                with torch.no_grad():
                    for projection in (model.w13, model.w2):
                        projection.weight = torch.nn.Parameter(
                            projection.weight.chunk(ep_size)[
                                self.rank % ep_size
                            ].clone()
                        )
                        projection.weight.normal_(std=0.5)
                torch.manual_seed(self.rank)
                x_TD = torch.randn(
                    _NUM_TOKENS,
                    _MODEL_DIM,
                    device=self.device_type,
                    dtype=torch.float64,
                    requires_grad=True,
                )
                scores_TK = torch.rand(
                    _NUM_TOKENS,
                    _TOP_K,
                    device=self.device_type,
                    dtype=torch.float64,
                    requires_grad=True,
                )
                expert_ids_TK = (
                    torch.arange(_NUM_TOKENS, device=self.device_type)[:, None]
                    + torch.arange(_TOP_K, device=self.device_type)
                ) % 8
                routing_map_TE = torch.nn.functional.one_hot(expert_ids_TK, 8).any(
                    dim=1
                )
                results = []
                for absorb in (False, True):
                    model.token_dispatcher.absorb_router_scores = absorb
                    out_TD = model(x_TD, scores_TK, expert_ids_TK, routing_map_TE)
                    remat.recompute_needs_tensor(out_TD)
                    loss = out_TD.square().sum()
                    grads = torch.autograd.grad(
                        loss, (x_TD, scores_TK, *model.parameters())
                    )
                    results.append((loss.detach(), *grads))
                # W2 input scaling uses float32 even with the exact GEMM seam.
                for actual, expected in zip(results[1], results[0]):
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
