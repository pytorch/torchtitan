# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from unittest.mock import patch

import pytest
import torch
import torch_remat as remat
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.activation_checkpoint import FullAC, RegionAC, SelectiveAC
from torchtitan.distributed.spmd_types import set_current_spmd_mesh, set_spmd_meshes
from torchtitan.models.common.activation import SwiGLU
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.token_dispatcher import DeepEPTokenDispatcher
from torchtitan.protocols.module import Module, ModuleDict

deep_ep = pytest.importorskip("deep_ep")

pytestmark = pytest.mark.multi_gpu

# DeepEP's combine needs the hidden size to be a multiple of 256.
_MODEL_DIM = 256
_NUM_TOKENS = 64
_TOP_K = 2


class _DeepEPBlock(Module):
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
        routed_experts.output_postprocess = None
        routed_experts.token_dispatcher = DeepEPTokenDispatcher.Config(
            num_experts=num_experts,
            top_k=_TOP_K,
            hidden_dim=_MODEL_DIM,
            num_max_tokens_per_rank=_NUM_TOKENS,
        ).build()
        self.gate = torch.nn.Linear(_MODEL_DIM, num_experts, bias=False)
        with torch.no_grad():
            for parameter in routed_experts.parameters():
                parameter.normal_(std=0.05)
        self.routed_experts = routed_experts
        self.num_experts = num_experts

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        num_tokens = x_TD.shape[0]
        token_ids_T1 = torch.arange(num_tokens, device=x_TD.device).unsqueeze(-1)
        offsets_1K = torch.arange(_TOP_K, device=x_TD.device).unsqueeze(0)
        expert_ids_TK = (token_ids_T1 + offsets_1K) % self.num_experts
        # Scores depend on the gate, so the router gets a gradient too.
        scores_TK = torch.sigmoid(self.gate(x_TD)).gather(1, expert_ids_TK)
        num_tokens_per_expert_E = torch.bincount(
            expert_ids_TK.flatten(), minlength=self.num_experts
        )
        out_TD = self.routed_experts(
            x_TD,
            scores_TK,
            expert_ids_TK,
            num_tokens_per_expert_E,
        )
        # The loss is a bare consumer of the routed-expert output.
        remat.recompute_needs_tensor(out_TD)
        return out_TD.float().square().sum()


class _Model(Module):
    def __init__(self, block: Module):
        super().__init__()
        self.layers = ModuleDict({"0": block})

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.layers["0"](x_TD)


def _run_forward_backward(
    model: Module,
    x_TD: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    input_TD = x_TD.detach().clone().requires_grad_(True)
    model(input_TD).backward()
    assert input_TD.grad is not None
    grads = [input_TD.grad.detach().clone()]
    grads += [parameter.grad.detach().clone() for parameter in model.parameters()]
    return grads


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestDeepEPActivationCheckpointing(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_dispatch_and_combine_are_saved_not_replayed(self):
        # DeepEP fills receive slots with atomics, so a replayed dispatch can
        # receive rows in another order than the forward's handle records.
        # Backward would then pair each token's gradient with another token's
        # activations. Every AC policy must save dispatch and combine instead.
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
        num_experts = 4 * self.world_size
        for ac_config in (
            FullAC.Config(),
            SelectiveAC.Config(),
            RegionAC.Config(save_regions=[]),
        ):
            with (
                self.subTest(ac=type(ac_config).__qualname__),
                torch.autograd.set_multithreading_enabled(False),
                set_current_spmd_mesh(mesh),
            ):
                torch.manual_seed(42)
                baseline = _Model(_DeepEPBlock(num_experts)).to(
                    self.device_type, torch.bfloat16
                )
                ac_model = _Model(_DeepEPBlock(num_experts)).to(
                    self.device_type, torch.bfloat16
                )
                ac_model.load_state_dict(baseline.state_dict())
                baseline.layers["0"].routed_experts.token_dispatcher.init_buffer()
                ac_config.build().apply(ac_model)

                num_calls = {"dispatch": 0, "combine": 0}
                original = {
                    name: getattr(deep_ep.ElasticBuffer, name) for name in num_calls
                }

                def counted(name):
                    def call(*args, **kwargs):
                        num_calls[name] += 1
                        return original[name](*args, **kwargs)

                    return call

                torch.manual_seed(self.rank)
                x_TD = torch.randn(
                    _NUM_TOKENS,
                    _MODEL_DIM,
                    device=self.device_type,
                    dtype=torch.bfloat16,
                )
                with (
                    patch.object(
                        deep_ep.ElasticBuffer,
                        "dispatch",
                        autospec=True,
                        side_effect=counted("dispatch"),
                    ),
                    patch.object(
                        deep_ep.ElasticBuffer,
                        "combine",
                        autospec=True,
                        side_effect=counted("combine"),
                    ),
                ):
                    expected = _run_forward_backward(baseline, x_TD)
                    baseline_calls = dict(num_calls)
                    num_calls.update(dispatch=0, combine=0)
                    actual = _run_forward_backward(ac_model, x_TD)

                # Forward dispatch and combine, plus their backward passes.
                self.assertEqual(baseline_calls, {"dispatch": 2, "combine": 2})
                self.assertEqual(
                    num_calls, baseline_calls, msg=f"AC replayed DeepEP: {num_calls}"
                )
                # Not bitwise: the receive order can differ between the two runs,
                # which changes the summation order of the expert weight grads.
                for actual_grad, expected_grad in zip(actual, expected):
                    torch.testing.assert_close(actual_grad, expected_grad)
