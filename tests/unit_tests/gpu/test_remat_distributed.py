# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import unittest

from unittest.mock import patch

import pytest
import spmd_types as spmd
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.nn.attention.flex_attention import create_block_mask
from torch.profiler import profile, ProfilerActivity
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.activation_checkpoint import RegionAC
from torchtitan.distributed.spmd_types import set_current_spmd_mesh
from torchtitan.models.common.attention import GQAttention
from torchtitan.models.common.cp_attention import (
    KVAllGatherCPFlexInnerAttention,
    UlyssesCPFlexInnerAttention,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.protocols.module import Module, ModuleDict


pytestmark = pytest.mark.multi_gpu


@contextlib.contextmanager
def _use_spmd_types(mesh):
    with set_current_spmd_mesh(mesh):
        yield


class _Model(Module):
    def __init__(self, block: Module):
        super().__init__()
        self.layers = ModuleDict({"0": block})

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.layers["0"](x_TD)


class _QKVProjection(Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear.Config(in_features=32, out_features=32).build()

    def forward(
        self, x_TD: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_THK = self.linear(x_TD).unflatten(-1, (2, 16))
        return q_THK, q_THK, q_THK


class _CPAttentionBlock(Module):
    def __init__(self, inner_attention: Module, *, device: torch.device):
        super().__init__()
        attention = GQAttention.__new__(GQAttention)
        Module.__init__(attention)
        attention.n_heads = 2
        attention.n_kv_heads = 2
        attention.head_dim = 16
        attention.enable_gqa = False
        attention.rope = None
        attention.qkv_linear = _QKVProjection()
        attention.wo = Linear.Config(in_features=32, out_features=32).build()
        attention.inner_attention = inner_attention
        attention.q_norm = None
        attention.k_norm = None
        attention.scaling = None
        self.attention = attention
        num_query_tokens = (
            8 if isinstance(inner_attention, UlyssesCPFlexInnerAttention) else 4
        )
        self.attention_mask = create_block_mask(
            lambda b, h, q_idx, kv_idx: q_idx >= 0,
            B=None,
            H=None,
            Q_LEN=num_query_tokens,
            KV_LEN=8,
            device=device,
            _compile=False,
        )

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.attention(x_TD, attention_masks=self.attention_mask).sum()


class _LocalExpert(Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 4))

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_local_expert_e: torch.Tensor,
    ) -> torch.Tensor:
        del num_tokens_per_local_expert_e
        return x_RD @ self.weight


class _AllToAllBlock(Module):
    def __init__(self, ep_mesh):
        super().__init__()
        routed_experts = RoutedExperts.__new__(RoutedExperts)
        Module.__init__(routed_experts)
        routed_experts.inner_experts = _LocalExpert()
        dispatcher = AllToAllTokenDispatcher.Config(
            num_experts=ep_mesh.size(), top_k=1
        ).build()
        dispatcher.wire_meshes(ep_mesh=ep_mesh)
        routed_experts.token_dispatcher = dispatcher
        self.routed_experts = routed_experts
        self.num_experts = ep_mesh.size()

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        num_tokens = x_TD.shape[0]
        expert_ids_TK = (
            torch.arange(num_tokens, device=x_TD.device) % self.num_experts
        ).unsqueeze(-1)
        routing_scores_TK = torch.ones(
            num_tokens, 1, device=x_TD.device, dtype=x_TD.dtype
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


def _run_forward_backward(model: Module, x_TD: torch.Tensor):
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
class TestDistributedRematRegions(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def _assert_results_equal(self, expected, actual):
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        for actual_grad, expected_grad in zip(actual[2], expected[2]):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    @with_comms
    def test_cp_collective_replay(self):
        mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("cp",)
        )
        for inner_attention_type, profiler_key in (
            (
                KVAllGatherCPFlexInnerAttention,
                "_c10d_functional::all_gather_into_tensor",
            ),
            (
                UlyssesCPFlexInnerAttention,
                "nccl:all_to_all",
            ),
        ):
            with self.subTest(
                inner_attention=inner_attention_type.__name__
            ), _use_spmd_types(mesh):
                torch.manual_seed(42)

                def build_model():
                    return _Model(
                        _CPAttentionBlock(
                            inner_attention_type.Config().build(),
                            device=torch.device(self.device_type),
                        )
                    ).to(self.device_type)

                baseline = build_model()
                saved = build_model()
                recomputed = build_model()
                saved.load_state_dict(baseline.state_dict())
                recomputed.load_state_dict(baseline.state_dict())
                RegionAC.Config(
                    save_regions=["attention.inner_attention"]
                ).build().apply(saved)
                RegionAC.Config(save_regions=[]).build().apply(recomputed)

                x_TD = torch.randn(4, 32, device=self.device_type)

                # Warm up FlexAttention compilation so profiler output only
                # contains collectives from the measured iteration.
                for model in (baseline, saved, recomputed):
                    _run_forward_backward(model, x_TD)

                def run(model):
                    with profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
                    ) as profiler:
                        result = _run_forward_backward(model, x_TD)
                    collective_count = sum(
                        event.count
                        for event in profiler.key_averages()
                        if event.key.startswith(profiler_key)
                    )
                    return result, collective_count

                expected, baseline_collectives = run(baseline)
                saved_result, saved_collectives = run(saved)
                recomputed_result, recomputed_collectives = run(recomputed)

                self._assert_results_equal(expected, saved_result)
                self._assert_results_equal(expected, recomputed_result)
                self.assertEqual(saved_collectives, baseline_collectives)
                self.assertGreater(recomputed_collectives, baseline_collectives)

    @with_comms
    def test_standard_all_to_all_collective_replay(self):
        mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        for save_regions, expected_extra_collectives in (
            ([], 3),
            (["routed_experts.ep_communication"], 0),
        ):
            with self.subTest(save_regions=save_regions), _use_spmd_types(mesh):
                torch.manual_seed(42)
                baseline = _Model(_AllToAllBlock(mesh)).to(self.device_type)
                remat_model = _Model(_AllToAllBlock(mesh)).to(self.device_type)
                remat_model.load_state_dict(baseline.state_dict())
                RegionAC.Config(save_regions=save_regions).build().apply(remat_model)

                num_collectives = 0
                original_all_to_all = spmd.all_to_all

                def counted_all_to_all(*args, **kwargs):
                    nonlocal num_collectives
                    num_collectives += 1
                    return original_all_to_all(*args, **kwargs)

                x_TD = torch.randn(4, 4, device=self.device_type)
                with patch.object(spmd, "all_to_all", side_effect=counted_all_to_all):
                    expected = _run_forward_backward(baseline, x_TD)
                    baseline_collectives = num_collectives
                    num_collectives = 0
                    actual = _run_forward_backward(remat_model, x_TD)

                self._assert_results_equal(expected, actual)

                self.assertEqual(baseline_collectives, 3)
                self.assertEqual(
                    num_collectives,
                    baseline_collectives + expected_extra_collectives,
                )


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
