# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Replaying DeepEP dispatch/combine with a deterministic buffer is exact.

The DeepEP autograd nodes keep the dispatch layout from the original forward.
With ``deterministic=True`` a replayed dispatch returns rows in the same order,
so every activation checkpointing policy that replays it must produce gradients
bitwise equal to running without checkpointing.
"""

import unittest

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

pytest.importorskip("deep_ep")

from torchtitan.distributed.activation_checkpoint import (  # noqa: E402
    FullAC,
    RegionAC,
    SelectiveAC,
)
from torchtitan.distributed.spmd_types import (  # noqa: E402
    set_current_spmd_mesh,
    set_spmd_meshes,
)
from torchtitan.models.common.activation import SwiGLU  # noqa: E402
from torchtitan.models.common.linear import GroupedLinear, Linear  # noqa: E402
from torchtitan.models.common.moe import RoutedExperts  # noqa: E402
from torchtitan.models.common.token_dispatcher import (  # noqa: E402
    DeepEPTokenDispatcher,
)
from torchtitan.protocols.module import Module, ModuleDict  # noqa: E402


pytestmark = pytest.mark.multi_gpu

# Shape legend: T local tokens, D model dim, F expert hidden dim, K top-k.
_T = 128
_D = 256
_F = 128
_K = 4
_NUM_LOCAL_EXPERTS = 8


class _DeepEPBlock(Module):
    """``x + moe(x)`` with an fp32 router, so router, expert and input grads all
    flow through DeepEP dispatch and combine."""

    def __init__(self, num_experts: int):
        super().__init__()
        self.gate = Linear.Config(
            in_features=_D, out_features=num_experts, bias=False
        ).build()
        # RoutedExperts.forward runs after EP has selected this rank's local
        # expert-weight shard. Build that local view directly, while the
        # dispatcher keeps the global expert count.
        routed_experts = RoutedExperts.__new__(RoutedExperts)
        Module.__init__(routed_experts)
        routed_experts.w13 = GroupedLinear.Config(
            group_size=_NUM_LOCAL_EXPERTS,
            in_features=_D,
            out_features=_F,
            num_linears=2,
        ).build()
        routed_experts.w2 = GroupedLinear.Config(
            group_size=_NUM_LOCAL_EXPERTS,
            in_features=_F,
            out_features=_D,
        ).build()
        routed_experts.activation_fn = SwiGLU.Config().build()
        routed_experts.output_postprocess = None
        routed_experts.token_dispatcher = DeepEPTokenDispatcher.Config(
            num_experts=num_experts,
            top_k=_K,
            num_max_tokens_per_rank=_T,
            hidden_dim=_D,
            deterministic=True,
        ).build()
        self.routed_experts = routed_experts
        self.num_experts = num_experts

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        scores_TE = self.gate(x_TD.float()).softmax(dim=-1)
        topk_scores_TK, topk_expert_ids_TK = scores_TE.topk(_K, dim=-1)
        num_tokens_per_expert_E = torch.bincount(
            topk_expert_ids_TK.flatten(), minlength=self.num_experts
        )
        out_TD = self.routed_experts(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_tokens_per_expert_E,
        )
        return x_TD + out_TD


class _Model(Module):
    def __init__(self, num_experts: int):
        super().__init__()
        self.layers = ModuleDict({"0": _DeepEPBlock(num_experts)})

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.layers["0"](x_TD)


class _SelectiveACSavingTopkOnly(SelectiveAC):
    """A custom SAC save set that leaves the DeepEP ops to be replayed."""

    def get_save_ops(self) -> set:
        return {torch.ops.aten.topk.default}


def _run_forward_backward(
    model: Module, x_TD: torch.Tensor, grad_out_TD: torch.Tensor
) -> list[torch.Tensor]:
    model.zero_grad(set_to_none=True)
    input_TD = x_TD.detach().clone().requires_grad_(True)
    output_TD = model(input_TD)
    output_TD.backward(grad_out_TD)
    assert input_TD.grad is not None
    return [output_TD.detach(), input_TD.grad.detach()] + [
        parameter.grad.detach().clone()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestDeepEPActivationCheckpointing(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 4)

    @with_comms
    def test_replay_with_deterministic_buffer_matches_no_checkpointing(self):
        mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        set_spmd_meshes(dense_mesh=mesh, sparse_mesh=mesh, dense_sp_enabled=False)
        num_experts = _NUM_LOCAL_EXPERTS * self.world_size

        torch.manual_seed(42)
        baseline = _Model(num_experts).to(self.device_type, torch.bfloat16)
        baseline.layers["0"].gate.float()
        baseline.layers["0"].routed_experts.token_dispatcher.init_buffer()

        torch.manual_seed(1000 + self.rank)
        x_TD = torch.randn(_T, _D, device=self.device_type, dtype=torch.bfloat16)
        grad_out_TD = torch.randn_like(x_TD)

        policies = {
            "full_ac": FullAC.Config().build(),
            "sac_saves_topk_only": _SelectiveACSavingTopkOnly(SelectiveAC.Config()),
            "region_ac_saves_nothing": RegionAC.Config(save_regions=[]).build(),
        }
        with (
            torch.autograd.set_multithreading_enabled(False),
            set_current_spmd_mesh(mesh),
        ):
            expected = _run_forward_backward(baseline, x_TD, grad_out_TD)
            for name, policy in policies.items():
                with self.subTest(policy=name):
                    model = _Model(num_experts).to(self.device_type, torch.bfloat16)
                    model.layers["0"].gate.float()
                    model.load_state_dict(baseline.state_dict())
                    policy.apply(model)
                    actual = _run_forward_backward(model, x_TD, grad_out_TD)
                    self.assertEqual(len(actual), len(expected))
                    for actual_tensor, expected_tensor in zip(actual, expected):
                        torch.testing.assert_close(
                            actual_tensor, expected_tensor, rtol=0, atol=0
                        )


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
