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

from torchtitan.distributed.activation_remat import (
    ActivationRematConfig,
    DefaultActivationRematConfig,
    FullActivationRematConfig,
)
from torchtitan.distributed.spmd_types import set_current_spmd_mesh, set_spmd_meshes
from torchtitan.models.base.module import Module, ModuleDict
from torchtitan.models.common.activation import SwiGLU
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.moe.token_dispatcher import (
    AllToAllTokenDispatcher,
    DeepEPTokenDispatcher,
)

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
        routing_map_TE = torch.zeros(
            num_tokens, self.num_experts, dtype=torch.bool, device=x_TD.device
        ).scatter_(1, expert_ids_TK, True)
        out_TD = self.routed_experts(
            x_TD,
            scores_TK,
            expert_ids_TK,
            routing_map_TE,
        )
        # The loss is a bare consumer of the routed-expert output.
        remat.recompute_needs_tensor(out_TD)
        # Weight tokens differently, so a combine that returns rows to the wrong tokens
        # changes the loss and the grads.
        return (out_TD.float().square() * (token_ids_T1 + 1)).sum()


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
        # activations. These policies save dispatch and combine instead.
        self._check_deepep_activation_checkpointing(
            (
                DefaultActivationRematConfig(),
                ActivationRematConfig(save_regions=["*ep_communication"]),
            ),
            num_replays=0,
        )

    @with_comms
    def test_matches_all_to_all_dispatcher(self):
        # The stock all-to-all dispatcher is the reference. With the same weights and
        # routing (top-2 of 4 * world experts: some tokens send both picks to one rank,
        # some split them), DeepEP's expand path matches it up to bf16 rounding: the output,
        # the input grad, the gate grad (through the routing scores) and the expert grads.
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
        with (
            torch.autograd.set_multithreading_enabled(False),
            set_current_spmd_mesh(mesh),
        ):
            torch.manual_seed(42)
            deepep = _Model(_DeepEPBlock(num_experts)).to(
                self.device_type, torch.bfloat16
            )
            all_to_all = _Model(_DeepEPBlock(num_experts)).to(
                self.device_type, torch.bfloat16
            )
            all_to_all.load_state_dict(deepep.state_dict())
            dispatcher = AllToAllTokenDispatcher.Config(
                num_experts=num_experts, top_k=_TOP_K
            ).build()
            all_to_all.layers["0"].routed_experts.token_dispatcher = dispatcher
            deepep.layers["0"].routed_experts.token_dispatcher.init_buffer()
            torch.manual_seed(self.rank)
            x_TD = torch.randn(
                _NUM_TOKENS,
                _MODEL_DIM,
                device=self.device_type,
                dtype=torch.bfloat16,
            )
            outputs = []
            for model in (deepep, all_to_all):
                # Also compare the [T, D] output: a row returned to the wrong token
                # changes it even where the summed loss barely moves.
                model.layers["0"].routed_experts.register_forward_hook(
                    lambda module, args, out_TD: outputs.append(out_TD.detach())
                )
            # No-grad forward first: the host-synced path validation runs.
            with torch.no_grad():
                deepep(x_TD)
                all_to_all(x_TD)
            actual = _run_forward_backward(deepep, x_TD)
            expected = _run_forward_backward(all_to_all, x_TD)
            # Assert only after both ranks finish every collective.
            no_grad_actual, no_grad_expected, actual_output, expected_output = outputs
            for actual_tensor, expected_tensor in zip(
                [no_grad_actual, actual_output, *actual],
                [no_grad_expected, expected_output, *expected],
                strict=True,
            ):
                error = (actual_tensor - expected_tensor).float().norm()
                self.assertLess((error / expected_tensor.float().norm()).item(), 1e-2)

    @with_comms
    def test_region_ac_replay_is_exact(self):
        # The full policy and policies without ep_communication saved replay dispatch and
        # combine. DeepEP then uses a deterministic buffer, which receives rows in
        # the forward's order, so the gradients match the run without AC bitwise.
        self._check_deepep_activation_checkpointing(
            (
                FullActivationRematConfig(),
                ActivationRematConfig(save_regions=[]),
            ),
            num_replays=1,
        )

    def _check_deepep_activation_checkpointing(
        self,
        ac_configs: tuple,
        *,
        num_replays: int,
    ) -> None:
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
        for activation_remat in ac_configs:
            with (
                self.subTest(ac=type(activation_remat).__qualname__),
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
                activation_remat.apply(ac_model)
                # As in training, the buffer is created after AC is applied. Both
                # models share it, so the baseline also runs deterministic when
                # the AC model replays.
                baseline.layers["0"].routed_experts.token_dispatcher.init_buffer()
                ac_model.layers["0"].routed_experts.token_dispatcher.init_buffer()

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
                    num_calls,
                    {"dispatch": 2 + num_replays, "combine": 2 + num_replays},
                    msg=f"unexpected DeepEP replays: {num_calls}",
                )
                if num_replays:
                    for actual_grad, expected_grad in zip(actual, expected):
                        torch.testing.assert_close(
                            actual_grad, expected_grad, rtol=0, atol=0
                        )
                else:
                    # Not bitwise: the receive order can differ between the two
                    # runs, which changes the summation order of the expert
                    # weight grads.
                    for actual_grad, expected_grad in zip(actual, expected):
                        torch.testing.assert_close(actual_grad, expected_grad)

    @with_comms
    def test_experts_and_combine_apply_scores_alike(self):
        # Exact SwiGLU takes the routing scores from DeepEP and applies them before w2;
        # a SwiGLU subclass leaves them to the combine. Each applies them once, so the two
        # agree up to bf16 rounding; a double or a missing multiply would not.
        class _SwiGLUSubclass(SwiGLU):
            pass

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
        with (
            torch.autograd.set_multithreading_enabled(False),
            set_current_spmd_mesh(mesh),
        ):
            torch.manual_seed(42)
            experts_apply = _Model(_DeepEPBlock(num_experts)).to(
                self.device_type, torch.bfloat16
            )
            combine_applies = _Model(_DeepEPBlock(num_experts)).to(
                self.device_type, torch.bfloat16
            )
            combine_applies.load_state_dict(experts_apply.state_dict())
            activation_fn = _SwiGLUSubclass(SwiGLU.Config())
            combine_applies.layers["0"].routed_experts.activation_fn = activation_fn
            experts_apply.layers["0"].routed_experts.token_dispatcher.init_buffer()
            torch.manual_seed(self.rank)
            x_TD = torch.randn(
                _NUM_TOKENS,
                _MODEL_DIM,
                device=self.device_type,
                dtype=torch.bfloat16,
            )
            with torch.no_grad():
                torch.testing.assert_close(
                    experts_apply(x_TD), combine_applies(x_TD), rtol=1e-2, atol=0
                )
            actual = _run_forward_backward(experts_apply, x_TD)
            expected = _run_forward_backward(combine_applies, x_TD)
            for actual_grad, expected_grad in zip(actual, expected, strict=True):
                error = (actual_grad - expected_grad).float().norm()
                self.assertLess((error / expected_grad.float().norm()).item(), 1e-2)
