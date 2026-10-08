# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import unittest

from unittest.mock import patch

import pytest
import spmd_types as spmd
import torch
import torch_remat as remat
from torch.distributed.device_mesh import init_device_mesh
from torch.multiprocessing.reductions import StorageWeakRef
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.activation_checkpoint import RegionAC, SelectiveAC
from torchtitan.distributed.spmd_types import set_current_spmd_mesh, set_spmd_meshes
from torchtitan.models.common.activation import SwiGLU
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    TorchAOTokenDispatcher,
)
from torchtitan.protocols.module import Module, ModuleDict


pytestmark = pytest.mark.multi_gpu

_MODEL_DIM = 8  # BF16 grouped-MM rows require a 16-byte stride.


class _AllToAllBlock(Module):
    def __init__(
        self,
        num_experts: int,
        dispatcher_config: AllToAllTokenDispatcher.Config | None = None,
    ):
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
        if dispatcher_config is None:
            dispatcher_config = AllToAllTokenDispatcher.Config(
                num_experts=num_experts,
                top_k=1,
            )
        routed_experts.token_dispatcher = dispatcher_config.build()
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
        out_TD = self.routed_experts(
            x_TD,
            routing_scores_TK,
            expert_ids_TK,
            num_tokens_per_expert_E,
        )
        # The sum is a bare consumer of the routed-expert output.
        remat.recompute_needs_tensor(out_TD)
        return out_TD.sum()


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

        # dispatch holds the count exchange, its device-to-host sync, and the
        # dispatch all-to-all; combine holds the combine all-to-all. SelectiveAC
        # recomputes the routed-expert projections but must retain both
        # dispatcher regions under the same routed_experts prefix.
        dispatch = "routed_experts.token_dispatcher.dispatch"
        combine = "routed_experts.token_dispatcher.combine"
        for policy_config, expected_replay_collectives, expected_replay_syncs in (
            (RegionAC.Config(save_regions=[]), 3, 1),
            (RegionAC.Config(save_regions=[dispatch]), 1, 0),
            (RegionAC.Config(save_regions=[combine]), 2, 1),
            (RegionAC.Config(save_regions=[dispatch, combine]), 0, 0),
            (SelectiveAC.Config(), 0, 0),
        ):
            with (
                self.subTest(policy_config=policy_config),
                torch.autograd.set_multithreading_enabled(False),
                set_current_spmd_mesh(mesh),
            ):
                torch.manual_seed(42)
                baseline = _Model(_AllToAllBlock(self.world_size)).to(self.device_type)
                remat_model = _Model(_AllToAllBlock(self.world_size)).to(
                    self.device_type
                )
                remat_model.load_state_dict(baseline.state_dict())
                policy_config.build().apply(remat_model)

                num_collectives = 0
                num_syncs = 0
                original_all_to_all = spmd.all_to_all
                original_sync = AllToAllTokenDispatcher._sync_token_count_exchange

                def counted_all_to_all(*args, **kwargs):
                    nonlocal num_collectives
                    num_collectives += 1
                    return original_all_to_all(*args, **kwargs)

                def counted_sync(*args, **kwargs):
                    nonlocal num_syncs
                    num_syncs += 1
                    return original_sync(*args, **kwargs)

                x_TD = torch.randn(4, _MODEL_DIM, device=self.device_type)
                with (
                    patch.object(spmd, "all_to_all", side_effect=counted_all_to_all),
                    patch.object(
                        AllToAllTokenDispatcher,
                        "_sync_token_count_exchange",
                        autospec=True,
                        side_effect=counted_sync,
                    ),
                ):
                    expected = _run_forward_backward(baseline, x_TD)
                    baseline_collectives = num_collectives
                    num_collectives = 0
                    num_syncs = 0
                    actual = _run_forward_backward(remat_model, x_TD)

                self._assert_results_equal(expected, actual)
                self.assertEqual(baseline_collectives, 3)
                self.assertEqual(
                    num_collectives,
                    baseline_collectives + expected_replay_collectives,
                )
                self.assertEqual(num_syncs, 1 + expected_replay_syncs)

    @with_comms
    def test_saved_w2_output_is_not_retained_under_ep(self):
        # Unpermute and the combine all-to-all do not need the w2 output in
        # backward, so saving w2 and the combine must not keep the w2 output
        # alive. bf16 activations, as in mixed-precision training, need no cast
        # after w2.
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
        with (
            torch.autograd.set_multithreading_enabled(False),
            set_current_spmd_mesh(mesh),
        ):
            torch.manual_seed(42)
            baseline = _Model(_AllToAllBlock(self.world_size)).to(
                self.device_type, torch.bfloat16
            )
            remat_model = _Model(_AllToAllBlock(self.world_size)).to(
                self.device_type, torch.bfloat16
            )
            remat_model.load_state_dict(baseline.state_dict())
            RegionAC.Config(
                save_regions=[
                    "routed_experts.w2.grouped_mm",
                    "routed_experts.token_dispatcher.combine",
                ]
            ).build().apply(remat_model)
            w2 = remat_model.layers["0"].routed_experts.w2
            w2_output_refs = []
            original_forward = GroupedLinear.forward

            def recorded_forward(module, *args, **kwargs):
                output = original_forward(module, *args, **kwargs)
                # The forward also runs (with its region skipped) during replay.
                if module is w2 and not remat.is_recomputing():
                    # remat retains a detached alias, so track the storage.
                    w2_output_refs.append(StorageWeakRef(output.untyped_storage()))
                return output

            x_TD = torch.randn(
                4, _MODEL_DIM, device=self.device_type, dtype=torch.bfloat16
            )
            expected = _run_forward_backward(baseline, x_TD)
            with patch.object(
                GroupedLinear, "forward", autospec=True, side_effect=recorded_forward
            ):
                input_TD = x_TD.clone().requires_grad_(True)
                loss = remat_model(input_TD)
                gc.collect()
                retained = not w2_output_refs[0].expired()
                loss.backward()

            self.assertFalse(retained)
            self.assertEqual(len(w2_output_refs), 1)
            torch.testing.assert_close(input_TD.grad, expected[1], rtol=0, atol=0)
            for parameter, expected_grad in zip(remat_model.parameters(), expected[2]):
                torch.testing.assert_close(
                    parameter.grad, expected_grad, rtol=0, atol=0
                )

    @with_comms
    def test_torchao_padded_dispatch_matches_unpadded(self):
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

        def build(dispatcher_config=None, save_regions=None):
            torch.manual_seed(42)
            model = _Model(_AllToAllBlock(self.world_size, dispatcher_config))
            model = model.to(self.device_type)
            if save_regions is not None:
                RegionAC.Config(save_regions=save_regions).build().apply(model)
            return model

        padded_config = TorchAOTokenDispatcher.Config(
            num_experts=self.world_size,
            top_k=1,
            pad_multiple=16,
        )
        with (
            torch.autograd.set_multithreading_enabled(False),
            set_current_spmd_mesh(mesh),
        ):
            x_TD = torch.randn(6, _MODEL_DIM, device=self.device_type)
            unpadded = _run_forward_backward(build(), x_TD)
            padded = _run_forward_backward(build(padded_config), x_TD)
            torch.testing.assert_close(padded[0], unpadded[0])
            torch.testing.assert_close(padded[1], unpadded[1])
            for padded_grad, unpadded_grad in zip(padded[2], unpadded[2]):
                torch.testing.assert_close(padded_grad, unpadded_grad)

            # Replaying the padded permute/unpermute must match the eager run.
            for save_regions in ([], ["routed_experts.*"]):
                with self.subTest(save_regions=save_regions):
                    actual = _run_forward_backward(
                        build(padded_config, save_regions), x_TD
                    )
                    self._assert_results_equal(padded, actual)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
