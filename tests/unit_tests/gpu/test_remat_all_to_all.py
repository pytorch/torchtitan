# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from unittest.mock import patch

import pytest
import spmd_types as spmd
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.activation_checkpoint import RegionAC
from torchtitan.distributed.spmd_types import set_current_spmd_mesh, set_spmd_meshes
from torchtitan.models.common.activation import SwiGLU
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.protocols.module import Module, ModuleDict


pytestmark = pytest.mark.multi_gpu

_MODEL_DIM = 8  # BF16 grouped-MM rows require a 16-byte stride.


class _AllToAllBlock(Module):
    def __init__(self, num_experts: int):
        super().__init__()
        # RoutedExperts.forward runs after EP has selected this rank's local
        # expert-weight shard. This test bypasses parallelization, so construct
        # that local view directly while the dispatcher retains the global E.
        num_local_experts = num_experts // torch.distributed.get_world_size()
        routed_experts = RoutedExperts.__new__(RoutedExperts)
        Module.__init__(routed_experts)
        routed_experts.w13 = GroupedLinear.Config(
            group_size=num_local_experts,
            in_features=_MODEL_DIM,
            out_features=_MODEL_DIM,
            num_linears=2,
        ).build()
        routed_experts.w2 = GroupedLinear.Config(
            group_size=num_local_experts,
            in_features=_MODEL_DIM,
            out_features=_MODEL_DIM,
        ).build()
        routed_experts.activation_fn = SwiGLU.Config().build()
        routed_experts.token_dispatcher = AllToAllTokenDispatcher.Config(
            num_experts=num_experts,
            top_k=1,
        ).build()
        with torch.no_grad():
            for parameter in routed_experts.parameters():
                parameter.normal_()
        self.routed_experts = routed_experts
        self.num_experts = num_experts

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        num_tokens = x_TD.shape[0]
        expert_ids_TK = (
            torch.arange(num_tokens, device=x_TD.device) % self.num_experts
        ).unsqueeze(-1)
        routing_scores_TK = torch.ones(
            num_tokens,
            1,
            device=x_TD.device,
            dtype=x_TD.dtype,
        )
        num_tokens_per_expert_E = torch.bincount(
            expert_ids_TK.flatten(), minlength=self.num_experts
        )
        return self.routed_experts(
            x_TD,
            routing_scores_TK,
            expert_ids_TK,
            num_tokens_per_expert_E,
        ).sum()


class _Model(Module):
    def __init__(self, block: Module):
        super().__init__()
        self.layers = ModuleDict({"0": block})

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.layers["0"](x_TD)


def _run_forward_backward(
    model: Module,
    x_TD: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    input_TD = x_TD.detach().clone().requires_grad_(True)
    output = model(input_TD)
    output.backward()
    assert input_TD.grad is not None
    parameter_grads = [
        parameter.grad.detach().clone()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    return output.detach(), input_TD.grad.detach().clone(), parameter_grads


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestAllToAllRematRegions(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @staticmethod
    def _assert_results_equal(expected, actual) -> None:
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        for actual_grad, expected_grad in zip(actual[2], expected[2]):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    @with_comms
    def test_collective_replay_follows_region_policy(self):
        mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("ep",),
        )
        set_spmd_meshes(
            dense_mesh=mesh,
            sparse_mesh=mesh,
            dense_sp_enabled=False,
        )

        for save_regions, expected_replay_collectives in (
            ([], 3),
            (["routed_experts.token_dispatcher.ep_communication"], 0),
        ):
            with (
                self.subTest(save_regions=save_regions),
                torch.autograd.set_multithreading_enabled(False),
                set_current_spmd_mesh(mesh),
            ):
                torch.manual_seed(42)
                baseline = _Model(_AllToAllBlock(self.world_size)).to(self.device_type)
                remat_model = _Model(_AllToAllBlock(self.world_size)).to(
                    self.device_type
                )
                remat_model.load_state_dict(baseline.state_dict())
                RegionAC.Config(save_regions=save_regions).build().apply(remat_model)

                num_collectives = 0
                original_all_to_all = spmd.all_to_all

                def counted_all_to_all(*args, **kwargs):
                    nonlocal num_collectives
                    num_collectives += 1
                    return original_all_to_all(*args, **kwargs)

                x_TD = torch.randn(4, _MODEL_DIM, device=self.device_type)
                with patch.object(spmd, "all_to_all", side_effect=counted_all_to_all):
                    expected = _run_forward_backward(baseline, x_TD)
                    baseline_collectives = num_collectives
                    num_collectives = 0
                    actual = _run_forward_backward(remat_model, x_TD)

                self._assert_results_equal(expected, actual)
                self.assertEqual(baseline_collectives, 3)
                self.assertEqual(
                    num_collectives,
                    baseline_collectives + expected_replay_collectives,
                )


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
