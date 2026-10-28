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
from torchtitan.models.base.module import Module, ModuleDict
from torchtitan.models.common.activation import SwiGLU
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.moe.token_dispatcher import (
    AllToAllTokenDispatcher,
    LocalTokenDispatcher,
    TorchAOTokenDispatcher,
)

pytest.importorskip("torchao")

pytestmark = pytest.mark.multi_gpu

_MODEL_DIM = 8
_NUM_TOKENS = 64
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


class _RoutedExpertsBlock(Module):
    """One routed-experts block behind a configurable token dispatcher."""

    def __init__(
        self,
        num_experts: int,
        num_local_experts: int,
        dispatcher_config: LocalTokenDispatcher.Config,
    ):
        super().__init__()
        # Build the EP-local expert view directly; the dispatcher keeps the global E.
        routed_experts = RoutedExperts.__new__(RoutedExperts)
        Module.__init__(routed_experts)
        routed_experts.w13 = _ExactGroupedLinear.Config(
            group_size=num_local_experts,
            in_features=_MODEL_DIM,
            out_features=_MODEL_DIM,
            num_linears=2,
        ).build()
        routed_experts.w2 = _ExactGroupedLinear.Config(
            group_size=num_local_experts,
            in_features=_MODEL_DIM,
            out_features=_MODEL_DIM,
        ).build()
        routed_experts.activation_fn = SwiGLU.Config().build()
        routed_experts.output_postprocess = None
        routed_experts.token_dispatcher = dispatcher_config.build()
        self.gate = torch.nn.Linear(_MODEL_DIM, num_experts, bias=False)
        with torch.no_grad():
            for parameter in routed_experts.parameters():
                parameter.normal_(std=0.5)
        self.routed_experts = routed_experts
        self.num_experts = num_experts

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        num_tokens = x_TD.shape[0]
        token_ids_T1 = torch.arange(num_tokens, device=x_TD.device).unsqueeze(-1)
        offsets_1K = torch.arange(_TOP_K, device=x_TD.device).unsqueeze(0)
        expert_ids_TK = (token_ids_T1 + offsets_1K) % self.num_experts
        # Scores depend on the gate, so a misplaced score moves the gradients.
        scores_TK = torch.sigmoid(self.gate(x_TD)).gather(1, expert_ids_TK)
        routing_map_TE = torch.nn.functional.one_hot(
            expert_ids_TK, num_classes=self.num_experts
        ).any(dim=1)
        out_TD = self.routed_experts(
            x_TD,
            scores_TK,
            expert_ids_TK,
            routing_map_TE,
        )
        remat.recompute_needs_tensor(out_TD)
        return out_TD.square().sum()


class _Model(Module):
    def __init__(self, block: Module):
        super().__init__()
        self.layers = ModuleDict({"0": block})

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.layers["0"](x_TD)


def _forward_backward(
    model: Module,
    x_TD: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run one forward/backward; return the loss and input/parameter grads."""
    model.zero_grad(set_to_none=True)
    input_TD = x_TD.detach().clone().requires_grad_(True)
    loss = model(input_TD)
    loss.backward()
    assert input_TD.grad is not None
    grads = [input_TD.grad.detach().clone()]
    grads += [parameter.grad.detach().clone() for parameter in model.parameters()]
    return loss.detach().clone(), grads


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
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("ep",),
        )
        num_experts = 4 * self.world_size
        for name, config_cls, kwargs, ep_size in self._cases():
            set_spmd_meshes(
                dense_mesh=mesh,
                sparse_mesh=mesh if ep_size > 1 else None,
                dense_sp_enabled=False,
            )
            with (
                self.subTest(dispatcher=name),
                torch.autograd.set_multithreading_enabled(False),
                set_current_spmd_mesh(mesh),
            ):
                self._check_absorb_parity(
                    config_cls,
                    kwargs,
                    num_experts,
                    num_experts // ep_size,
                )

    def _check_absorb_parity(
        self,
        config_cls,
        kwargs,
        num_experts: int,
        num_local_experts: int,
    ) -> None:
        torch.manual_seed(42)
        baseline = _Model(
            _RoutedExpertsBlock(
                num_experts,
                num_local_experts,
                config_cls(
                    num_experts=num_experts,
                    top_k=_TOP_K,
                    absorb_router_scores=False,
                    **kwargs,
                ),
            )
        ).to(self.device_type, torch.float64)
        absorbed = _Model(
            _RoutedExpertsBlock(
                num_experts,
                num_local_experts,
                config_cls(
                    num_experts=num_experts,
                    top_k=_TOP_K,
                    absorb_router_scores=True,
                    **kwargs,
                ),
            )
        ).to(self.device_type, torch.float64)
        absorbed.load_state_dict(baseline.state_dict())

        torch.manual_seed(self.rank)
        x_TD = torch.randn(
            _NUM_TOKENS, _MODEL_DIM, device=self.device_type, dtype=torch.float64
        )
        expected_loss, expected = _forward_backward(baseline, x_TD)
        actual_loss, actual = _forward_backward(absorbed, x_TD)

        # Scores are applied in float32, so compare at float32 scale.
        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-5, atol=1e-6)
        for actual_grad, expected_grad in zip(actual, expected):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)
